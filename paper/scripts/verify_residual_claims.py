"""verify_residual_claims.py -- single-script GPU verification of three
numerical claims in the paper that weren't directly stored in committed
results JSONs:

  (A) Gemma Scope 2 4B IT l0_medium L29 typical per-token L0 (the
      "60-100 active features" claim).
  (B) Phase 6 hook diagnostics: mean/peak norm of contribution
      subtracted by ablating format-direction features 3833/10012/980
      at Gemma 4B L29 on the canonical NL prompt for case E1.
  (C) Qwen Scope L31 reconstruction error on Qwen3-8B (the "~38%"
      claim; script docstring says "~40%").

Saves to results/verify_residual_claims.json. Compute: 1× small GPU
~10-15 min, ~$0.50.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import safetensors.torch as sft
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/verify_residual_claims.json"


class JumpReLUSAE:
    def __init__(self, w_enc, w_dec, b_enc, b_dec, threshold, device):
        self.w_enc = w_enc.to(device); self.w_dec = w_dec.to(device)
        self.b_enc = b_enc.to(device); self.b_dec = b_dec.to(device)
        self.threshold = threshold.to(device)

    def encode(self, x):
        pre = x.float() @ self.w_enc + self.b_enc
        return pre * (pre > self.threshold).float()

    def decode(self, f):
        return f.to(self.w_dec.dtype) @ self.w_dec + self.b_dec


def load_gemma_scope_sae(layer, repo, device="cuda"):
    sub = f"resid_post/layer_{layer}_width_16k_l0_medium/params.safetensors"
    p = sft.load_file(hf_hub_download(repo, sub))
    return JumpReLUSAE(p["w_enc"], p["w_dec"], p["b_enc"], p["b_dec"], p["threshold"], device)


def get_layer(model, layer):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer]
    return model.model.layers[layer]


def get_residuals(model, tok, prompt, layer):
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=False)
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    ids = ids.to(model.device)
    cap = {}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h.detach()
    h_handle = get_layer(model, layer).register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        h_handle.remove()
    return cap["h"][0], ids[0]  # [seq, d], [seq]


def main():
    out = {}

    # ─── A + B: Gemma 4B L29 ──────────────────────────────────────────────
    print("\n=== Loading Gemma 3 4B IT + Gemma Scope 2 SAE L29 ===")
    MODEL_ID = "google/gemma-3-4b-it"
    SAE_REPO = "google/gemma-scope-2-4b-it"
    LAYER = 29
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    sae = load_gemma_scope_sae(LAYER, SAE_REPO)

    # Load canonical NL prompts (60 cases)
    fl = json.loads((ROOT / "paper/data/canonical_forced_letter_vignettes.json").read_text())
    nf = json.loads((ROOT / "paper/data/canonical_singleturn_vignettes.json").read_text())

    # ─── A: per-token L0 distribution ────────────────────────────────────
    print("\n--- (A) Per-token L0 distribution at L29 (Gemma 3 4B IT, l0_medium) ---")
    print("    sampling 10 NL prompts to get a representative distribution")
    all_l0 = []
    all_norms = []
    for v in fl[:10]:
        h, ids = get_residuals(model, tok, v["natural_forced_letter"], LAYER)
        # Skip the first 4 chat-template tokens
        h_content = h[4:].to(sae.w_enc.dtype).to(sae.w_enc.device)
        with torch.no_grad():
            feats = sae.encode(h_content)
        l0 = (feats > 0).sum(dim=-1).float().cpu().numpy()
        norms = h_content.float().norm(dim=-1).cpu().numpy()
        all_l0.extend(l0.tolist())
        all_norms.extend(norms.tolist())
    import numpy as np
    l0_arr = np.array(all_l0); norm_arr = np.array(all_norms)
    print(f"    n_tokens={len(l0_arr)}, L0: median={np.median(l0_arr):.0f}, mean={l0_arr.mean():.1f}, "
          f"5–95th pctile=[{np.percentile(l0_arr,5):.0f}, {np.percentile(l0_arr,95):.0f}], "
          f"range=[{l0_arr.min():.0f}, {l0_arr.max():.0f}]")
    print(f"    per-token residual norm: median={np.median(norm_arr):.0f}, mean={norm_arr.mean():.0f}")
    out["A_gemma_4b_L29_l0_distribution"] = {
        "n_tokens": int(len(l0_arr)),
        "l0_median": float(np.median(l0_arr)),
        "l0_mean": float(l0_arr.mean()),
        "l0_p5": float(np.percentile(l0_arr, 5)),
        "l0_p95": float(np.percentile(l0_arr, 95)),
        "l0_min": float(l0_arr.min()),
        "l0_max": float(l0_arr.max()),
        "per_token_norm_median": float(np.median(norm_arr)),
        "per_token_norm_mean": float(norm_arr.mean()),
        "n_prompts_sampled": 10,
    }

    # ─── B: Phase 6 hook diagnostics ────────────────────────────────────
    print("\n--- (B) Phase 6 ablation contribution diagnostics ---")
    FORMAT_FEATURES = [3833, 10012, 980]
    feats_t = torch.tensor(FORMAT_FEATURES, dtype=torch.long, device="cuda")
    # Run on the canonical case E1 NL prompt (deterministic, matches the
    # original phase6_debug.py)
    e1 = next(v for v in fl if v["id"] == "E1")
    h, ids = get_residuals(model, tok, e1["natural_forced_letter"], LAYER)
    h_flat = h.reshape(-1, h.shape[-1]).to(sae.w_enc.dtype).to(sae.w_enc.device)
    with torch.no_grad():
        feats = sae.encode(h_flat)
        sub = feats[:, feats_t]  # [seq, 3]
        contribution = sub.to(sae.w_dec.dtype) @ sae.w_dec[feats_t]  # [seq, d]
    per_tok_contrib_norm = contribution.float().norm(dim=-1).cpu().numpy()
    per_tok_resid_norm   = h_flat.float().norm(dim=-1).cpu().numpy()
    pct = per_tok_contrib_norm / per_tok_resid_norm * 100  # %
    print(f"    n_tokens={len(per_tok_contrib_norm)}")
    print(f"    contribution norm: mean={per_tok_contrib_norm.mean():.1f}, "
          f"peak={per_tok_contrib_norm.max():.1f}, median={np.median(per_tok_contrib_norm):.2f}")
    print(f"    relative (contribution/residual): mean={pct.mean():.2f}%, "
          f"peak={pct.max():.2f}%, median={np.median(pct):.4f}%")
    out["B_phase6_diagnostic"] = {
        "case_id": "E1",
        "n_tokens": int(len(per_tok_contrib_norm)),
        "contribution_norm_mean": float(per_tok_contrib_norm.mean()),
        "contribution_norm_peak": float(per_tok_contrib_norm.max()),
        "contribution_norm_median": float(np.median(per_tok_contrib_norm)),
        "pct_of_residual_mean": float(pct.mean()),
        "pct_of_residual_peak": float(pct.max()),
        "pct_of_residual_median": float(np.median(pct)),
        "per_token_residual_norm_mean": float(per_tok_resid_norm.mean()),
        "per_token_residual_norm_peak": float(per_tok_resid_norm.max()),
        "format_features": FORMAT_FEATURES,
    }
    del model, sae, h, h_flat, feats, sub, contribution
    torch.cuda.empty_cache()

    # ─── C: Qwen Scope L31 reconstruction error ──────────────────────────
    print("\n=== Loading Qwen3-8B + Qwen Scope SAE L31 ===")
    QWEN_ID = "Qwen/Qwen3-8B"
    QWEN_SAE = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
    QWEN_LAYER = 31
    qtok = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
    qmodel = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    qmodel.eval()
    # Qwen Scope: layer{L}.sae.pt with keys W_enc [d_sae, d_model],
    # W_dec [d_sae, d_model], b_enc [d_sae], b_dec [d_model]. TopK=50.
    qpath = hf_hub_download(QWEN_SAE, f"layer{QWEN_LAYER}.sae.pt")
    qp = torch.load(qpath, map_location="cpu")
    print(f"Qwen SAE keys: {list(qp.keys())[:8]}")
    WE = qp["W_enc"].to("cuda")  # [d_sae, d_model]
    WD = qp["W_dec"].to("cuda")  # [d_sae, d_model]
    BE = qp["b_enc"].to("cuda")
    BD = qp["b_dec"].to("cuda")
    print(f"   W_enc shape: {WE.shape}, W_dec shape: {WD.shape}")
    TOPK = 50

    def qwen_encode(x):
        # Matches phase4_qwen_minimal.py:155-160 (TopKSAE.encode)
        pre = x.float() @ WE.T + BE
        topk_vals, topk_idx = pre.topk(TOPK, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    def qwen_decode(f):
        # phase4_qwen_minimal.py:162-163: features @ W_dec.T + b_dec
        # W_dec on disk is [d_model, d_sae]; transpose to [d_sae, d_model] for matmul
        return f @ WD.T + BD

    print("\n--- (C) Qwen Scope L31 reconstruction error on 5 NF prompts ---")
    sample_prompts = [v["patient_realistic"] for v in nf[:5]]

    rel_errs = []
    for p in sample_prompts:
        h, _ = get_residuals(qmodel, qtok, p, QWEN_LAYER)
        h_content = h[4:].float().to("cuda")
        with torch.no_grad():
            f = qwen_encode(h_content)
            r = qwen_decode(f).to(h_content.dtype)
        err = ((h_content - r).norm(dim=-1) / h_content.norm(dim=-1).clamp(min=1e-6)).cpu().numpy()
        rel_errs.extend(err.tolist())
    import numpy as np
    rel_errs = np.array(rel_errs)
    print(f"   n_tokens={len(rel_errs)}, relative L2 recon error: "
          f"mean={rel_errs.mean():.4f}, median={np.median(rel_errs):.4f}, "
          f"5–95 = [{np.percentile(rel_errs,5):.4f}, {np.percentile(rel_errs,95):.4f}]")
    out["C_qwen_L31_recon_err"] = {
        "n_tokens": int(len(rel_errs)),
        "rel_l2_mean":   float(rel_errs.mean()),
        "rel_l2_median": float(np.median(rel_errs)),
        "rel_l2_p5":     float(np.percentile(rel_errs, 5)),
        "rel_l2_p95":    float(np.percentile(rel_errs, 95)),
        "n_prompts_sampled": 5,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
