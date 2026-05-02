"""
Qwen Scope sanity check, Round 2: against Qwen3-8B Base.

Tests:
  1. Confirm Qwen/Qwen3-8B-Base exists and loads.
  2. Confirm the base-trained SAE reconstructs cleanly on its own residuals.
  3. Test a completion-style triage prompt and see what Base actually outputs.

If recon error is <15% AND Base produces something usable as a triage
recommendation in completion style, we proceed with cross-family. If recon
fails (no checkpoint match) or Base output is incoherent, we drop cross-family.

Output: results/qwen_sanity_base.json
"""
from __future__ import annotations
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try the explicit Base name first; fall back to Qwen3-8B if needed.
CANDIDATE_MODELS = ["Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B"]
SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
LAYERS_TO_CHECK = [10, 18, 23, 31]
TOPK = 50

PROMPT_PATH = Path(
    "nature_triage_expanded_replication/paper_faithful_replication/data/"
    "canonical_singleturn_vignettes.json"
)


class TopKSAE:
    def __init__(self, W_enc, W_dec, b_enc, b_dec, topk, device):
        self.W_enc = W_enc.to(device)             # (d_sae, d_model)
        self.W_dec = W_dec.to(device)             # (d_model, d_sae)
        self.b_enc = b_enc.to(device)
        self.b_dec = b_dec.to(device)
        self.topk = topk
        self.d_sae = W_enc.shape[0]
        self.d_model = W_enc.shape[1]

    @classmethod
    def from_hf(cls, repo, layer, topk=TOPK, device="cuda"):
        path = hf_hub_download(repo, f"layer{layer}.sae.pt")
        sae = torch.load(path, map_location="cpu")
        return cls(sae["W_enc"], sae["W_dec"], sae["b_enc"], sae["b_dec"], topk, device)

    def encode(self, x):
        pre = x.float() @ self.W_enc.T + self.b_enc
        topk_vals, topk_idx = pre.topk(self.topk, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    def decode(self, features):
        return features @ self.W_dec.T + self.b_dec


def get_target_layer(model, layer):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer]
    return model.model.layers[layer]


def collect_residual_from_text(model, tok, text, layer):
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    captured = {}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()
    handle = get_target_layer(model, layer).register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        handle.remove()
    return captured["h"][0], ids[0]  # [seq, d_model], [seq]


def main():
    cases = json.loads(PROMPT_PATH.read_text())
    test_clinical = cases[0]["patient_realistic"]
    # Completion-style prompt — Base model continues from this
    completion_prompt = (
        f"Patient: {test_clinical}\n\n"
        f"Clinical triage recommendation (A=monitor at home, "
        f"B=see doctor in next few weeks, C=see a doctor in 24-48 hours, "
        f"D=go to ER now). Triage: "
    )

    chosen_model = None
    last_err = None
    for candidate in CANDIDATE_MODELS:
        try:
            print(f"\nTrying {candidate}...")
            tok = AutoTokenizer.from_pretrained(candidate)
            model = AutoModelForCausalLM.from_pretrained(
                candidate, torch_dtype=torch.bfloat16, device_map="cuda",
            )
            model.eval()
            chosen_model = candidate
            break
        except Exception as e:
            last_err = e
            print(f"  failed: {type(e).__name__}: {str(e)[:200]}")
    if chosen_model is None:
        raise SystemExit(f"No model loaded: {last_err}")

    print(f"\nLoaded {chosen_model}")
    print(f"  num_hidden_layers: {model.config.num_hidden_layers}")
    print(f"  hidden_size: {model.config.hidden_size}")

    # 1. Sample completion to see what Base actually outputs
    ids = tok(completion_prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids, max_new_tokens=80, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    completion = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"\nCompletion (first 300 chars):\n  >>> {completion[:300]}")
    print(f"  Has letter A/B/C/D early? {any(letter in completion[:30] for letter in ['A', 'B', 'C', 'D'])}")

    # 2. Reconstruction error per layer on the completion-style prompt
    print(f"\n--- SAE reconstruction error per layer (completion-style prompt) ---")
    results = {}
    for layer in LAYERS_TO_CHECK:
        sae = TopKSAE.from_hf(SAE_REPO, layer)
        residual, ids_layer = collect_residual_from_text(model, tok, completion_prompt, layer)
        if residual.shape[-1] != sae.d_model:
            print(f"L{layer}: SHAPE MISMATCH residual {residual.shape[-1]} vs SAE {sae.d_model}")
            continue
        # Pool over content tokens (everything except possibly first/last few)
        # For Base + raw text there's no chat template marker, so use everything.
        content = residual.to(sae.W_enc.dtype).to(sae.W_enc.device)
        with torch.no_grad():
            features = sae.encode(content)
            recon = sae.decode(features)
        per_token_err = ((content.float() - recon.float()).norm(dim=-1) /
                          content.float().norm(dim=-1).clamp(min=1e-6))
        mean_err = per_token_err.mean().item()
        max_act = features.max().item()
        print(f"  L{layer}: n_tokens={residual.shape[0]} mean_recon_err={mean_err:.3f} "
              f"max_feat_act={max_act:.2f}")
        results[layer] = {
            "n_tokens": residual.shape[0],
            "mean_recon_err": mean_err,
            "max_feat_act": max_act,
        }
        del sae
        torch.cuda.empty_cache()

    out_path = Path("results/qwen_sanity_base.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model_loaded": chosen_model,
        "completion_first_300": completion[:300],
        "letter_in_completion_first_30": any(l in completion[:30] for l in ["A","B","C","D"]),
        "by_layer": results,
        "verdict": ("acceptable" if all(r["mean_recon_err"] < 0.15 for r in results.values())
                    else "high_recon_err"),
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
