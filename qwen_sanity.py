"""
Qwen Scope sanity check.

Verify before committing more compute:
  1. Qwen3-8B Instruct loads and runs the chat template correctly.
  2. The Qwen Scope SAE (trained on Qwen3-8B Base) is loadable.
  3. SAE reconstruction error on Qwen3-8B Instruct residuals is acceptable
     (target: <20% relative L2 at the chosen layer).

If reconstruction error is too high we'd have to switch to Qwen3-8B Base
and adapt prompts to non-instruct format, which is more work.

Output: results/qwen_sanity.json
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"  # instruction-tuned (Qwen 3 final variant is IT)
SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
LAYERS_TO_CHECK = [10, 18, 23, 31]  # matched-depth to Gemma 4B's 9/17/22/29 (~27/50/65/85% of 36 layers)
TOPK = 50

# Test prompt: clinical scenario from the paper-faithful corpus
PROMPT_PATH = Path(
    "nature_triage_expanded_replication/paper_faithful_replication/data/"
    "canonical_singleturn_vignettes.json"
)


class TopKSAE:
    """Qwen Scope TopK SAE. Encode keeps only top-k pre-activations.

    Note Qwen's W_enc is shape (d_sae, d_model), opposite to Gemma's (d_model, d_sae).
    We transpose on load so encode/decode are uniform.
    """
    def __init__(self, W_enc, W_dec, b_enc, b_dec, topk, device):
        # Qwen storage: W_enc (d_sae, d_model), W_dec (d_model, d_sae)
        self.W_enc = W_enc.to(device)             # (d_sae, d_model)
        self.W_dec = W_dec.to(device)             # (d_model, d_sae)
        self.b_enc = b_enc.to(device)             # (d_sae,)
        self.b_dec = b_dec.to(device)             # (d_model,)
        self.topk = topk
        self.d_sae = W_enc.shape[0]
        self.d_model = W_enc.shape[1]

    @classmethod
    def from_hf(cls, repo, layer, topk=TOPK, device="cuda"):
        path = hf_hub_download(repo, f"layer{layer}.sae.pt")
        sae = torch.load(path, map_location="cpu")
        return cls(sae["W_enc"], sae["W_dec"], sae["b_enc"], sae["b_dec"], topk, device)

    def encode(self, x):
        # x: (..., d_model). Pre-acts: (..., d_sae)
        pre = x.float() @ self.W_enc.T + self.b_enc
        topk_vals, topk_idx = pre.topk(self.topk, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    def decode(self, features):
        # features: (..., d_sae) -> (..., d_model)
        return features @ self.W_dec.T + self.b_dec


def get_target_layer(model, layer):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer]
    return model.model.layers[layer]


def collect_residual(model, tok, prompt, layer):
    """Forward; capture residual at end of model.layers[layer]; return [seq, d_model]."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=False,
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    captured = {}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()
    handle = get_target_layer(model, layer).register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids.to(model.device))
    finally:
        handle.remove()
    return captured["h"][0]  # [seq_len, d_model]


def find_user_content_range(input_ids: list, tok):
    """Find start and end of user content tokens for Qwen chat template.
    Qwen uses <|im_start|>user\n...<|im_end|> blocks.
    """
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    user_token = tok.convert_tokens_to_ids("user")
    # Find first <|im_start|>user\n
    n = len(input_ids)
    for i in range(n - 2):
        if input_ids[i] == im_start and input_ids[i+1] == user_token:
            start = i + 3  # skip <|im_start|>, user, \n
            for j in range(start, n):
                if input_ids[j] == im_end:
                    return start, j
            break
    # Fallback: skip first 3 tokens, end at last
    return 3, n


def main():
    print(f"=== Qwen Scope sanity check ===\nModel: {MODEL_ID}\nSAE: {SAE_REPO}")
    cases = json.loads(PROMPT_PATH.read_text())
    test_prompt = cases[0]["patient_realistic"]
    print(f"\nTest prompt (first 100 chars): {test_prompt[:100]}...")

    print(f"\nLoading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    print(f"Model loaded. Type: {type(model).__name__}")
    print(f"  num_hidden_layers: {model.config.num_hidden_layers}")
    print(f"  hidden_size: {model.config.hidden_size}")

    # Gen a small test to verify chat template works
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=False,
    )
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    print(f"\nChat-templated 'What is 2+2?' tokens (first 20): {ids[0][:20].tolist()}")
    with torch.no_grad():
        out = model.generate(
            input_ids=ids.to(model.device),
            max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id,
        )
    print(f"  Generation: {tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)}")

    print(f"\n--- SAE reconstruction error per layer ---")
    results = {}
    for layer in LAYERS_TO_CHECK:
        sae = TopKSAE.from_hf(SAE_REPO, layer)
        residual = collect_residual(model, tok, test_prompt, layer)  # [seq, d_model]
        # Verify shape compatibility
        if residual.shape[-1] != sae.d_model:
            print(f"L{layer}: SHAPE MISMATCH residual {residual.shape[-1]} vs SAE d_model {sae.d_model}")
            continue
        # Find user content tokens
        ids = tok.apply_chat_template(
            [{"role": "user", "content": test_prompt}], add_generation_prompt=True,
            return_tensors="pt", return_dict=False,
        )
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
        ids_list = ids[0].tolist()
        start, end = find_user_content_range(ids_list, tok)
        n_content = end - start
        content_residuals = residual[start:end].to(sae.W_enc.dtype).to(sae.W_enc.device)
        with torch.no_grad():
            features = sae.encode(content_residuals)
            recon = sae.decode(features)
        per_token_err = ((content_residuals.float() - recon.float()).norm(dim=-1) /
                          content_residuals.float().norm(dim=-1).clamp(min=1e-6))
        mean_err = per_token_err.mean().item()
        max_act = features.max().item()
        nz = (features > 0).float().sum(dim=-1).mean().item()
        print(f"  L{layer}: d_model={sae.d_model} d_sae={sae.d_sae} "
              f"n_content_tokens={n_content} mean_recon_err={mean_err:.3f} "
              f"max_feat_act={max_act:.2f} avg_active_per_tok={nz:.0f}/{TOPK}")
        results[layer] = {
            "d_model": sae.d_model, "d_sae": sae.d_sae,
            "n_content_tokens": n_content,
            "mean_recon_err_relative_l2": mean_err,
            "max_feature_activation": max_act,
            "avg_features_active_per_token": nz,
            "topk_setting": TOPK,
        }
        del sae
        torch.cuda.empty_cache()

    out_path = Path("results/qwen_sanity.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": MODEL_ID, "sae_repo": SAE_REPO,
        "user_content_range": [start, end],
        "by_layer": results,
        "verdict": ("acceptable" if all(r["mean_recon_err_relative_l2"] < 0.20 for r in results.values())
                    else "high_recon_err — consider Base model"),
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
