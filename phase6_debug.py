"""
Quick debug: verify the ablation hook in phase6_causal_intervention.py is
actually modifying the residual stream during generation.
"""
from __future__ import annotations
import json, re
from pathlib import Path

import torch
import safetensors.torch as sft
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-4b-it"
SAE_REPO = "google/gemma-scope-2-4b-it"
LAYER = 29
FORMAT_FEATURES = [3833, 10012, 980]


class JumpReLUSAE:
    def __init__(self, w_enc, w_dec, b_enc, b_dec, threshold, device):
        self.w_enc = w_enc.to(device); self.w_dec = w_dec.to(device)
        self.b_enc = b_enc.to(device); self.b_dec = b_dec.to(device)
        self.threshold = threshold.to(device)
    def encode(self, x):
        pre = x.float() @ self.w_enc + self.b_enc
        return pre * (pre > self.threshold).float()


def load_sae():
    sub = f"resid_post/layer_{LAYER}_width_16k_l0_medium/params.safetensors"
    p = sft.load_file(hf_hub_download(SAE_REPO, sub))
    return JumpReLUSAE(p["w_enc"], p["w_dec"], p["b_enc"], p["b_dec"], p["threshold"], "cuda")


def get_target(model, layer):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer]
    return model.model.layers[layer]


def main():
    print("Loading model + SAE...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    sae = load_sae()

    # Build a minimal NL prompt with the forced-letter scaffold
    fl = json.loads(Path("nature_triage_expanded_replication/paper_faithful_forced_letter/data/canonical_forced_letter_vignettes.json").read_text())
    prompt = fl[0]["natural_forced_letter"]  # E1 case
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=False)
    if not isinstance(ids, torch.Tensor): ids = ids["input_ids"]
    ids = ids.to(model.device)
    print(f"Prompt tokens: {ids.shape[1]}")

    # PASS 1: capture vanilla L29 output
    captured_vanilla = {}
    def cap_hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured_vanilla["h"] = h.detach().clone()
    h1 = get_target(model, LAYER).register_forward_hook(cap_hook)
    with torch.no_grad():
        _ = model(input_ids=ids)
    h1.remove()
    print(f"Vanilla L29 output: shape={captured_vanilla['h'].shape}, "
          f"norm/token mean={captured_vanilla['h'].float().norm(dim=-1).mean().item():.2f}")

    # PASS 2: with ablation hook (counts how many times it fires)
    fire_count = [0]
    feats_t = torch.tensor(FORMAT_FEATURES, dtype=torch.long, device="cuda")
    captured_ablated = {}
    def ablate_hook(_m, _i, out):
        if isinstance(out, tuple):
            h, *rest = out
        else:
            h, rest = out, None
        h_flat = h.reshape(-1, h.shape[-1])
        feats = sae.encode(h_flat)
        sub = feats[:, feats_t]
        contribution = sub.to(sae.w_dec.dtype) @ sae.w_dec[feats_t]
        h_new = h_flat - contribution.to(h.dtype)
        h_new = h_new.reshape(h.shape)
        fire_count[0] += 1
        captured_ablated["h"] = h_new.detach().clone()
        captured_ablated["sub"] = sub.detach().clone()
        captured_ablated["contribution"] = contribution.detach().clone()
        if rest is not None:
            return (h_new, *rest)
        return h_new
    h2 = get_target(model, LAYER).register_forward_hook(ablate_hook)
    with torch.no_grad():
        _ = model(input_ids=ids)
    h2.remove()

    print(f"\nAblation hook fired {fire_count[0]} time(s) on the forward pass")
    print(f"Sub (feature activations) shape: {captured_ablated['sub'].shape}")
    print(f"Sub stats: max={captured_ablated['sub'].max().item():.2f}, "
          f"mean={captured_ablated['sub'].mean().item():.2f}, "
          f"nonzero count: {(captured_ablated['sub'] > 0).sum().item()}/{captured_ablated['sub'].numel()}")
    print(f"Contribution (subtracted from residual) norm/token mean: "
          f"{captured_ablated['contribution'].float().norm(dim=-1).mean().item():.4f}")
    diff = (captured_vanilla["h"] - captured_ablated["h"]).float()
    print(f"Difference (vanilla - ablated) norm/token mean: {diff.norm(dim=-1).mean().item():.4f}")
    print(f"Difference max norm: {diff.norm(dim=-1).max().item():.4f}")

    # PASS 3: end-to-end test with model.generate, see if logits actually differ
    print("\n--- Generation test ---")
    with torch.no_grad():
        out_v = model.generate(input_ids=ids, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
    print(f"Vanilla output: {tok.decode(out_v[0, ids.shape[1]:], skip_special_tokens=True)!r}")

    h3 = get_target(model, LAYER).register_forward_hook(ablate_hook)
    fire_count[0] = 0
    try:
        with torch.no_grad():
            out_a = model.generate(input_ids=ids, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        h3.remove()
    print(f"Ablated output: {tok.decode(out_a[0, ids.shape[1]:], skip_special_tokens=True)!r}")
    print(f"Hook fired {fire_count[0]} time(s) during 5-token generation")
    print(f"Generation token IDs equal? {(out_v == out_a).all().item()}")


if __name__ == "__main__":
    main()
