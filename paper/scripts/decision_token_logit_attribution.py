"""decision_token_logit_attribution.py -- decompose the NL pre-generation
logits for A/B/C/D into SAE-feature contributions, to test the reviewer's
"scaffold-primary, medical-partial" decision-token hypothesis.

Setup: at the NL last-prompt-position, the model emits a forced letter
(A/B/C/D). The reviewer's recommended analysis is to ask: which SAE
features at that position drive the letter-prior structure? Are they
scaffold-y (firing on letter-answer scaffolds) or medical (firing on
clinical content)?

Method (linear logit-lens approximation):
  1. Forward pass NL prompt; capture residual at the SAE layer at the
     last prompt position: h_dec [d_model].
  2. SAE-encode h_dec to get feature activations features [d_sae].
  3. For each active feature i, its linear contribution to the residual is
     features[i] * W_dec[i] [d_model] (= "feature_decoded").
  4. To project feature_decoded onto the A/B/C/D letter logits, apply the
     final layer norm (logit-lens style) and then dot with the unembedding
     row for each letter token: contribution[i, letter] =
     LN(feature_decoded[i]) @ W_unembed[letter_token_id].
  5. Identify which features push most strongly toward each letter; check
     whether they overlap with the v3-validated medical features or with
     "scaffold" features (defined as features whose peak activation in the
     B prompt is on a scaffold token, not a vignette token).

Note: this is an APPROXIMATE linear attribution. The full path from
layer-L to the unembedding involves several more transformer layers with
non-linearities, attention re-routing, etc. The logit-lens decomposition
captures the *direct linear effect* of each feature; non-linear effects
are not attributed. This is standard interp-toolkit usage (e.g. nostalgebraist
2020) and is informative directionally even though not perfectly causal.

Outputs:
  results/decision_token_logit_attribution_<MODEL>.json

Usage:
  python3 paper/scripts/decision_token_logit_attribution.py --model 4b
  python3 paper/scripts/decision_token_logit_attribution.py --model 12b
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import safetensors.torch as sft
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
VIGNETTES_FL = ROOT / "paper/data/canonical_forced_letter_vignettes.json"
VIGNETTES_NF = ROOT / "paper/data/canonical_singleturn_vignettes.json"

MODEL_CONFIGS = {
    "4b": {
        "model_id":  "google/gemma-3-4b-it",
        "sae_repo":  "google/gemma-scope-2-4b-it",
        "layer":     29,
        "medical_features": [12570, 893, 12845],
    },
    "12b": {
        "model_id":  "google/gemma-3-12b-it",
        "sae_repo":  "google/gemma-scope-2-12b-it",
        "layer":     31,
        "medical_features": [130, 85, 4773],
    },
}

# Token-ID lookup for the A/B/C/D letter tokens — varies by tokenizer
# Will be computed at runtime via tokenizer.encode("A", add_special_tokens=False)


class JumpReLUSAE:
    def __init__(self, w_enc, w_dec, b_enc, b_dec, threshold, device):
        self.w_enc = w_enc.to(device); self.w_dec = w_dec.to(device)
        self.b_enc = b_enc.to(device); self.b_dec = b_dec.to(device)
        self.threshold = threshold.to(device)
        self.d_sae = w_enc.shape[1]; self.d_model = w_enc.shape[0]

    @classmethod
    def from_hf(cls, repo, layer, device="cuda"):
        sub = f"resid_post/layer_{layer}_width_16k_l0_medium/params.safetensors"
        p = sft.load_file(hf_hub_download(repo, sub))
        return cls(p["w_enc"], p["w_dec"], p["b_enc"], p["b_dec"], p["threshold"], device)

    def encode(self, x):
        pre = x.float() @ self.w_enc + self.b_enc
        return pre * (pre > self.threshold).float()


def get_layer(model, layer):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer]
    return model.model.layers[layer]


def get_final_norm(model):
    """Return the model's final RMSNorm/LayerNorm before unembedding."""
    if hasattr(model.model, "language_model"):
        return model.model.language_model.norm
    return model.model.norm


def get_unembedding(model):
    """Return the model's unembedding matrix W_U [d_model, vocab]."""
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.data.T  # [d_model, vocab]
    return model.lm_head.weight.data.T


def get_residual_at_last_position(model, tok, prompt, layer):
    """Forward pass, return residual at `layer` at the LAST prompt position."""
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=False,
    )
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    ids = ids.to(model.device)
    cap = {}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h.detach()
    handle = get_layer(model, layer).register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(input_ids=ids).logits
    finally:
        handle.remove()
    return cap["h"][0, -1].float().cpu(), logits[0, -1].float().cpu(), ids[0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_CONFIGS), required=True)
    ap.add_argument("--top-k", type=int, default=20,
                    help="Top-K features (by activation) to report per case")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = MODEL_CONFIGS[args.model]

    # HF token
    token_path = Path.home() / ".cache/huggingface/token"
    if token_path.exists():
        os.environ["HF_TOKEN"] = token_path.read_text().strip()

    print(f"Loading {cfg['model_id']} ...")
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], torch_dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading SAE for L{cfg['layer']} ...")
    sae = JumpReLUSAE.from_hf(cfg["sae_repo"], cfg["layer"], device=args.device)
    final_norm = get_final_norm(model)
    W_U = get_unembedding(model).to(args.device).float()  # [d_model, vocab]
    print(f"  d_model={sae.d_model}, d_sae={sae.d_sae}, vocab={W_U.shape[1]}")

    # Letter token IDs
    letter_tokens = {}
    for L in "ABCD":
        # Try the standalone letter; some tokenizers add a leading space variant
        enc = tok.encode(L, add_special_tokens=False)
        letter_tokens[L] = enc[0] if enc else None
        # Also record the most-likely actual generation token (with leading space sometimes)
    print(f"  Letter token IDs: {letter_tokens}")

    # Load forced-letter prompts
    fl = json.loads(VIGNETTES_FL.read_text())
    nf = json.loads(VIGNETTES_NF.read_text())
    fl_by = {v["id"]: v for v in fl}

    # Identify "scaffold features" empirically from yesterday's masked-invariance
    # data: features whose peak in the B (NL) prompt is OUTSIDE the shared
    # vignette (i.e., peaks on a scaffold token).
    npz_path = ROOT / f"results/phase1b_masked_full_activations_{args.model}.npz"
    if npz_path.exists():
        z = np.load(npz_path, allow_pickle=True)
        B_max_content = np.asarray(z["B_max_content"])
        D_max_content = np.asarray(z["D_max_content"])
        B_max_vignette = np.asarray(z["B_max_vignette"])
        # A feature is "scaffold-peaking in B" if its B_max_content > B_max_vignette
        # by a meaningful margin (i.e., the B-prompt max activation is outside
        # the shared vignette mask)
        scaffold_score = (B_max_content - B_max_vignette).mean(0)  # [d_sae]
        # Top 100 features by scaffold_score
        scaffold_top100 = np.argsort(-scaffold_score)[:100].tolist()
        print(f"  Top 5 scaffold-peaking features in B: {scaffold_top100[:5]}")
    else:
        print(f"  WARN: {npz_path} not found, skipping scaffold-feature identification")
        scaffold_top100 = []

    medical_features = cfg["medical_features"]
    cases = sorted(fl_by.keys(), key=lambda s: (re.match(r"^(\D+)", s).group(1),
                                                 int(re.search(r"\d+", s).group())))

    # ── Main loop ─────────────────────────────────────────────────────
    per_case = []
    # Aggregate: contribution of each feature category to each letter logit
    # Categories: "medical" (v3 set), "scaffold" (top 100 by scaffold_score), "other"
    medical_set = set(medical_features)
    scaffold_set = set(scaffold_top100[:30])  # use top 30 as the scaffold proxy

    for i, cid in enumerate(cases):
        prompt = fl_by[cid]["natural_forced_letter"]
        h_dec, logits, ids = get_residual_at_last_position(model, tok, prompt, cfg["layer"])

        # Predicted letter
        # Note: in Gemma's vocab the leading-space variant " A" might be the
        # natural generation token; we report both
        pred_token_id = int(logits.argmax().item())
        pred_token_text = tok.decode([pred_token_id])

        # SAE encode
        h_dec_dev = h_dec.to(args.device)
        with torch.no_grad():
            feats = sae.encode(h_dec_dev).cpu().numpy()  # [d_sae]

        # Active features
        active_idx = np.flatnonzero(feats > 0)
        if len(active_idx) == 0:
            per_case.append({"case_id": cid, "n_active": 0, "skipped": True})
            continue

        # Compute per-feature linear contribution to A/B/C/D logits using
        # logit-lens: LN(feature_decoded) @ W_U[:, letter_token_id]
        # We do this in batched fashion for efficiency.
        # feature_decoded[i] = feats[i] * sae.w_dec[i]  [d_model]
        # We want LN(feature_decoded[i]) for each i, but LN is applied to the
        # full residual normally. For attribution, the standard logit-lens is:
        #   logit_contribution[i, L] = (feats[i] * W_dec[i]) @ W_U[:, L_id]
        # We use this DIRECT linear projection (without applying LN), which is
        # the simplest tractable attribution.
        W_dec = sae.w_dec.cpu().float().numpy()  # [d_sae, d_model]
        W_U_cpu = W_U.cpu().numpy()  # [d_model, vocab]
        letter_ids = [letter_tokens[L] for L in "ABCD"]
        letter_dirs = W_U_cpu[:, letter_ids]  # [d_model, 4]

        # For all active features at once:
        active_acts = feats[active_idx]              # [n_active]
        active_decs = W_dec[active_idx]              # [n_active, d_model]
        # Direct linear contribution: act_i * (W_dec_i @ letter_dir)
        contribs = active_acts[:, None] * (active_decs @ letter_dirs)  # [n_active, 4]

        # Categorize features
        categories = []
        for f in active_idx:
            if int(f) in medical_set:    categories.append("medical")
            elif int(f) in scaffold_set: categories.append("scaffold")
            else:                        categories.append("other")
        cat_arr = np.array(categories)

        # Aggregate per category: sum of contributions to each letter
        cat_letter_contrib = {}
        for cat in ("medical", "scaffold", "other"):
            mask = cat_arr == cat
            cat_letter_contrib[cat] = {
                L: float(contribs[mask, j].sum()) for j, L in enumerate("ABCD")
            }
            cat_letter_contrib[cat]["n_features"] = int(mask.sum())

        # Top-K features by absolute contribution to the PREDICTED letter
        # Predicted letter index (most likely is whatever pred_token decodes to)
        pred_letter = None
        ptt = pred_token_text.strip()
        if ptt and ptt[0] in "ABCD":
            pred_letter = ptt[0]
        if pred_letter is None:
            # use argmax of letter logits
            pred_letter = "ABCD"[int(logits[letter_ids].argmax().item())]
        pred_j = "ABCD".index(pred_letter)

        # Top contributors to predicted letter (positive direction)
        top_pos = np.argsort(-contribs[:, pred_j])[:args.top_k]
        top_features_pred = []
        for ti in top_pos:
            fi = int(active_idx[ti])
            top_features_pred.append({
                "feature": fi,
                "category": categories[ti],
                "activation": float(active_acts[ti]),
                "contrib_pred_letter": float(contribs[ti, pred_j]),
                "contrib_A": float(contribs[ti, 0]),
                "contrib_B": float(contribs[ti, 1]),
                "contrib_C": float(contribs[ti, 2]),
                "contrib_D": float(contribs[ti, 3]),
            })

        # Total contribution to each letter from all features (linear)
        total_per_letter = {L: float(contribs[:, j].sum()) for j, L in enumerate("ABCD")}
        # Actual letter logits
        actual_logits = {L: float(logits[letter_ids[j]]) for j, L in enumerate("ABCD")}

        per_case.append({
            "case_id": cid,
            "gold": fl_by[cid]["gold_standard_triage"],
            "n_active_features": int(len(active_idx)),
            "pred_letter": pred_letter,
            "pred_token_id": pred_token_id,
            "pred_token_text": pred_token_text,
            "letter_logits": actual_logits,
            "total_linear_contribution_per_letter": total_per_letter,
            "category_breakdown": cat_letter_contrib,
            "top_features_pushing_pred_letter": top_features_pred,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(cases)}] {cid}  pred={pred_letter}  n_active={len(active_idx)}")

    # ── Aggregate ─────────────────────────────────────────────────────
    # For each category × predicted-letter, average the contribution
    # across cases. Compute the fraction of cases where medical features
    # push toward the PREDICTED letter vs toward a non-predicted letter.

    aggregated_cat_to_letter = {cat: {L: [] for L in "ABCD"}
                                  for cat in ("medical", "scaffold", "other")}
    aggregated_cat_to_pred = {cat: [] for cat in ("medical", "scaffold", "other")}
    pred_letter_distribution = {"A": 0, "B": 0, "C": 0, "D": 0}

    for r in per_case:
        if "skipped" in r: continue
        pred_letter_distribution[r["pred_letter"]] += 1
        for cat, vals in r["category_breakdown"].items():
            for L in "ABCD":
                aggregated_cat_to_letter[cat][L].append(vals[L])
            aggregated_cat_to_pred[cat].append(vals[r["pred_letter"]])

    aggregate_summary = {}
    for cat in ("medical", "scaffold", "other"):
        aggregate_summary[cat] = {
            "mean_contribution_per_letter": {
                L: float(np.mean(aggregated_cat_to_letter[cat][L])) for L in "ABCD"
            },
            "mean_contribution_to_pred_letter": float(np.mean(aggregated_cat_to_pred[cat])),
            "median_contribution_to_pred_letter": float(np.median(aggregated_cat_to_pred[cat])),
        }

    summary = {
        "model": cfg["model_id"],
        "layer": cfg["layer"],
        "sae_repo": cfg["sae_repo"],
        "medical_features": cfg["medical_features"],
        "scaffold_feature_pool_top30": scaffold_top100[:30],
        "n_cases": len(per_case),
        "letter_token_ids": letter_tokens,
        "pred_letter_distribution": pred_letter_distribution,
        "category_aggregate": aggregate_summary,
        "per_case": per_case,
    }

    out_path = ROOT / f"results/decision_token_logit_attribution_{args.model}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_path}")
    print()
    print("=== Aggregate logit contribution per feature category ===")
    print(f"{'category':<12}{'n_avg':>8}{'mean→A':>10}{'mean→B':>10}{'mean→C':>10}{'mean→D':>10}  mean→pred")
    for cat in ("medical", "scaffold", "other"):
        s = aggregate_summary[cat]
        ml = s["mean_contribution_per_letter"]
        print(f"{cat:<12}{'?':>8}"
              f"{ml['A']:>10.3f}{ml['B']:>10.3f}{ml['C']:>10.3f}{ml['D']:>10.3f}"
              f"  {s['mean_contribution_to_pred_letter']:>8.3f}")
    print()
    print(f"Predicted letter distribution: {pred_letter_distribution}")


if __name__ == "__main__":
    main()
