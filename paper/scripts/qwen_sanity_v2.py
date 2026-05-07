"""
Quick test: try the b_dec-subtraction encode convention.

Many SAEs (especially Anthropic-style and Llama-Scope) use:
  pre = (x - b_dec) @ W_enc.T + b_enc
  recon = features @ W_dec.T + b_dec

This centers the input at the decoder bias before encoding.

If recon error drops substantially with this convention, my Phase 2 sanity
script had a bug. If it's the same, the SAE genuinely doesn't reconstruct
well even on its own training distribution.
"""
import json
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"
SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
LAYERS = [10, 18, 23, 31]
TOPK = 50

PROMPT_PATH = Path("nature_triage_expanded_replication/paper_faithful_replication/data/canonical_singleturn_vignettes.json")


def load_sae(layer):
    p = hf_hub_download(SAE_REPO, f"layer{layer}.sae.pt")
    return torch.load(p, map_location="cuda")


def encode_v1(x, sae):
    """No b_dec subtraction (what we did before)."""
    pre = x.float() @ sae["W_enc"].T + sae["b_enc"]
    vals, idx = pre.topk(TOPK, dim=-1)
    out = torch.zeros_like(pre)
    out.scatter_(-1, idx, vals)
    return out


def encode_v2(x, sae):
    """With b_dec subtraction (Anthropic / Llama-Scope convention)."""
    pre = (x.float() - sae["b_dec"]) @ sae["W_enc"].T + sae["b_enc"]
    vals, idx = pre.topk(TOPK, dim=-1)
    out = torch.zeros_like(pre)
    out.scatter_(-1, idx, vals)
    return out


def decode(features, sae):
    return features @ sae["W_dec"].T + sae["b_dec"]


def get_target(model, layer):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer]
    return model.model.layers[layer]


def main():
    cases = json.loads(PROMPT_PATH.read_text())
    text = cases[0]["patient_realistic"]

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"Loaded {MODEL_ID}")

    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    captured = {}
    def hook_for(L):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[L] = h.detach()[0]  # [seq, d_model]
        return hook

    handles = []
    for L in LAYERS:
        handles.append(get_target(model, L).register_forward_hook(hook_for(L)))
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for h in handles: h.remove()

    print(f"\n{'L':<4} {'mean_no_subtract':>20} {'mean_with_subtract':>20}")
    results = {}
    for L in LAYERS:
        sae = load_sae(L)
        residual = captured[L].to(sae["W_enc"].dtype).to(sae["W_enc"].device)

        f1 = encode_v1(residual, sae)
        r1 = decode(f1, sae)
        e1 = ((residual.float() - r1.float()).norm(dim=-1) /
              residual.float().norm(dim=-1).clamp(min=1e-6)).mean().item()

        f2 = encode_v2(residual, sae)
        r2 = decode(f2, sae)
        e2 = ((residual.float() - r2.float()).norm(dim=-1) /
              residual.float().norm(dim=-1).clamp(min=1e-6)).mean().item()

        print(f"L{L:<3} {e1:>20.4f} {e2:>20.4f}")
        results[L] = {"v1_no_subtract": e1, "v2_with_subtract": e2}
        del sae
        torch.cuda.empty_cache()

    Path("results/qwen_sanity_v2.json").write_text(json.dumps({
        "model": MODEL_ID, "sae_repo": SAE_REPO,
        "by_layer": results,
        "verdict": (
            "v2_better" if all(results[L]["v2_with_subtract"] < results[L]["v1_no_subtract"] for L in LAYERS)
            else "v1_better_or_same"),
    }, indent=2))
    print(f"\nWrote results/qwen_sanity_v2.json")


if __name__ == "__main__":
    main()
