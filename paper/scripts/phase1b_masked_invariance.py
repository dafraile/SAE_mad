"""phase1b_masked_invariance.py -- re-aggregate Phase 1b invariance
analysis with explicit token masks (vignette / scaffold / decision token).

Triggered by reviewer concern: pooling over the shared clinical-prefix
tokens gives an aggregate that is partially trivial under causal masking
(hidden states at vignette positions are identical in B and D prompts).
The non-trivial mechanistic claim has to live at non-shared positions —
the scaffold tokens (B-only), the decision token, or the generation
tokens.

This script:
  1. Forward-passes B and D prompts.
  2. Grabs residuals at the chosen layer for every position.
  3. SAE-encodes each token to get per-token feature activations.
  4. Identifies three token masks per case:
       - vignette_mask  : shared content tokens (same positions, same IDs
                          in B and D under chat-template alignment)
       - scaffold_mask  : B-only content tokens (answer-key + instruction)
       - decision_idx   : single position = last prompt content token
                          (last scaffold token in B, last vignette token
                           in D)
  5. Aggregates and reports:
       - vignette-mask: medical and random sMAPE/cosine (sanity check, ≈0)
       - decision-token: medical vs random sMAPE/cosine (the real claim)
       - full-content max-pool: medical vs random sMAPE/cosine (replicates
                                 the current headline number)
       - per-medical-feature peak position in B vs D
         (where does the feature's max activation occur?)

Output: results/phase1b_masked_invariance_<MODEL_TAG>.json

Usage:
  python paper/scripts/phase1b_masked_invariance.py --model 4b
  python paper/scripts/phase1b_masked_invariance.py --model 12b
  python paper/scripts/phase1b_masked_invariance.py --model qwen
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import safetensors.torch as sft
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
VIGNETTES_FL = ROOT / "paper/data/canonical_forced_letter_vignettes.json"
VIGNETTES_NF = ROOT / "paper/data/canonical_singleturn_vignettes.json"

# ─── per-model SAE config ───────────────────────────────────────────────────
MODEL_CONFIGS = {
    "4b": {
        "model_id":  "google/gemma-3-4b-it",
        "sae_repo":  "google/gemma-scope-2-4b-it",
        "layer":     29,
        "sae_kind":  "gemma_jumprelu_resid_post_16k_l0medium",
        "medical_features": [12570, 893, 12845],
        "n_random": 30,
        "random_seed": 42,
        "end_of_turn_id": 106,  # Gemma 3 chat template
        "chat_prefix_len": 4,   # <bos><start_of_turn>user\n
    },
    "12b": {
        "model_id":  "google/gemma-3-12b-it",
        "sae_repo":  "google/gemma-scope-2-12b-it",
        "layer":     31,
        "sae_kind":  "gemma_jumprelu_resid_post_16k_l0medium",
        # 12B L31 v3-validated medical features (from phase3b_12b_phase1b.json)
        "medical_features": [130, 85, 4773],
        "n_random": 30,
        "random_seed": 42,
        "end_of_turn_id": 106,
        "chat_prefix_len": 4,
    },
    "qwen": {
        "model_id":  "Qwen/Qwen3-8B",
        "sae_repo":  "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
        "layer":     31,
        "sae_kind":  "qwen_topk_64k_k50",
        # Qwen3-8B L31 medical features (from phase4_qwen_L31.json)
        "medical_features": [29074, 48973, 60699],
        "n_random": 30,
        "random_seed": 42,
        # Qwen chat template uses different markers; chat_prefix_len computed
        # at runtime from the actual tokenization
        "end_of_turn_id": None,
        "chat_prefix_len": None,
    },
}


# ─── SAE loading ────────────────────────────────────────────────────────────
class GemmaJumpReLUSAE:
    def __init__(self, w_enc, w_dec, b_enc, b_dec, threshold, device):
        self.w_enc = w_enc.to(device); self.w_dec = w_dec.to(device)
        self.b_enc = b_enc.to(device); self.b_dec = b_dec.to(device)
        self.threshold = threshold.to(device)
        self.d_sae   = w_enc.shape[1]
        self.d_model = w_enc.shape[0]
        self.kind = "jumprelu"

    @classmethod
    def from_hf(cls, repo, layer, device="cuda"):
        sub = f"resid_post/layer_{layer}_width_16k_l0_medium/params.safetensors"
        p = sft.load_file(hf_hub_download(repo, sub))
        return cls(p["w_enc"], p["w_dec"], p["b_enc"], p["b_dec"], p["threshold"], device)

    def encode(self, x):
        pre = x.float() @ self.w_enc + self.b_enc
        return pre * (pre > self.threshold).float()


class QwenTopKSAE:
    def __init__(self, w_enc, w_dec, b_enc, b_dec, topk, device):
        # Qwen Scope layout: W_enc [d_sae, d_model], W_dec [d_sae, d_model]
        # (decode is f @ W_dec.T + b_dec — see verify_residual_claims.py:198)
        self.w_enc = w_enc.to(device)
        self.w_dec = w_dec.to(device)
        self.b_enc = b_enc.to(device)
        self.b_dec = b_dec.to(device)
        self.topk = topk
        self.d_sae   = w_enc.shape[0]
        self.d_model = w_enc.shape[1]
        self.kind = "topk"

    @classmethod
    def from_hf(cls, repo, layer, topk=50, device="cuda"):
        path = hf_hub_download(repo, f"layer{layer}.sae.pt")
        p = torch.load(path, map_location="cpu")
        return cls(p["W_enc"], p["W_dec"], p["b_enc"], p["b_dec"], topk, device)

    def encode(self, x):
        pre = x.float() @ self.w_enc.T + self.b_enc
        vals, idx = pre.topk(self.topk, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, idx, vals)
        return out


# ─── helpers ────────────────────────────────────────────────────────────────
def get_layer(model, layer):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer]
    return model.model.layers[layer]


def get_per_token_residuals(model, tok, prompt, layer):
    """Forward pass; return (residuals [seq, d_model] on CPU, input_ids list)."""
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=False,
    )
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    ids_dev = ids.to(model.device)

    cap = {}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h.detach()

    handle = get_layer(model, layer).register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=ids_dev)
    finally:
        handle.remove()
    return cap["h"][0].float().cpu(), ids[0].tolist()


def find_shared_prefix_length(ids_B: list[int], ids_D: list[int]) -> int:
    """How many leading token IDs are identical between the two sequences?
    This is the cleanest way to identify the shared vignette range: it
    doesn't require knowing the chat template details."""
    n = min(len(ids_B), len(ids_D))
    for i in range(n):
        if ids_B[i] != ids_D[i]:
            return i
    return n


def smape_per_feature(b_vec: np.ndarray, d_vec: np.ndarray) -> np.ndarray:
    """Per-feature sMAPE values."""
    num = np.abs(b_vec - d_vec)
    den = (np.abs(b_vec) + np.abs(d_vec)) / 2
    return num / np.maximum(den, 1e-8)


def cosine(b: np.ndarray, d: np.ndarray) -> float:
    nb = np.linalg.norm(b); nd = np.linalg.norm(d)
    if nb < 1e-8 or nd < 1e-8: return 0.0
    return float(np.dot(b, d) / (nb * nd))


# ─── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_CONFIGS), required=True)
    ap.add_argument("--medical-features", type=int, nargs="+", default=None,
                    help="Override medical feature IDs (else use defaults)")
    ap.add_argument("--n-cases", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = MODEL_CONFIGS[args.model]
    if args.medical_features:
        cfg["medical_features"] = args.medical_features
    if not cfg["medical_features"]:
        # Try to load from the existing phase4_qwen_L31 / phase1_activation_invariance
        # canonical results files
        candidates = [
            ROOT / "results/phase4_qwen_L31.json",
            ROOT / "results/phase1_activation_invariance.json",
            ROOT / "results/phase3b_12b_phase1_activation.json",
        ]
        for cp in candidates:
            if not cp.exists(): continue
            try:
                d = json.loads(cp.read_text())
                if "medical_features" in d and args.model == "qwen":
                    cfg["medical_features"] = d["medical_features"]; break
                if "by_layer" in d and args.model == "4b":
                    cfg["medical_features"] = d["by_layer"][str(cfg["layer"])]["medical_features"]; break
                if "medical_features" in d and args.model == "12b":
                    cfg["medical_features"] = d["medical_features"]; break
            except Exception:
                continue
    assert cfg["medical_features"], (
        f"No medical_features available for {args.model}. Pass --medical-features.")

    # ─── Load model + SAE ─────────────────────────────────────────────────
    print(f"Loading {cfg['model_id']} ...")
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], torch_dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading SAE for L{cfg['layer']} ...")
    if cfg["sae_kind"].startswith("gemma"):
        sae = GemmaJumpReLUSAE.from_hf(cfg["sae_repo"], cfg["layer"], device=args.device)
    elif cfg["sae_kind"].startswith("qwen"):
        sae = QwenTopKSAE.from_hf(cfg["sae_repo"], cfg["layer"], topk=50, device=args.device)
    else:
        raise SystemExit(f"unknown sae_kind {cfg['sae_kind']}")

    # ─── Load vignettes ───────────────────────────────────────────────────
    fl = {v["id"]: v for v in json.loads(VIGNETTES_FL.read_text())}
    nf = {v["id"]: v for v in json.loads(VIGNETTES_NF.read_text())}
    case_ids = sorted(fl.keys(), key=lambda s: (re.match(r"^(\D+)", s).group(1),
                                                int(re.search(r"\d+", s).group())))[: args.n_cases]

    # ─── Random feature selection: features firing on any B/D forward pass ─
    print(f"Picking random features (seed {cfg['random_seed']}) ...")
    # Probe just a few cases to identify which features fire (so the random
    # pool is non-trivially nonzero), then freeze.
    probe = []
    for cid in case_ids[: min(8, len(case_ids))]:
        for which in ("B", "D"):
            prompt = fl[cid]["natural_forced_letter"] if which == "B" else nf[cid]["patient_realistic"]
            h, _ = get_per_token_residuals(model, tok, prompt, cfg["layer"])
            with torch.no_grad():
                f = sae.encode(h.to(args.device)).float().mean(0).cpu().numpy()
            probe.append(f)
    mean_act = np.stack(probe).mean(0)
    firing_idx = np.flatnonzero(mean_act > 0)
    pool = [int(i) for i in firing_idx if int(i) not in cfg["medical_features"]]
    rng = np.random.default_rng(cfg["random_seed"])
    if len(pool) >= cfg["n_random"]:
        random_features = sorted(rng.choice(pool, size=cfg["n_random"], replace=False).tolist())
    else:
        random_features = pool
    print(f"  medical: {cfg['medical_features']}")
    print(f"  random ({len(random_features)}): {random_features[:6]}...")

    med = cfg["medical_features"]
    rnd = random_features

    # ─── Main loop ────────────────────────────────────────────────────────
    # Buffers for full per-case max-pool activations across ALL features
    # (used downstream for task #14: resample N random pools without a
    # second GPU run).
    all_B_max_vignette = []  # max over vignette tokens
    all_D_max_vignette = []
    all_B_max_content  = []  # max over all post-prefix tokens
    all_D_max_content  = []
    all_B_decision     = []  # last-position activations
    all_D_decision     = []

    per_case = []
    for i, cid in enumerate(case_ids):
        B_prompt = fl[cid]["natural_forced_letter"]
        D_prompt = nf[cid]["patient_realistic"]

        h_B, ids_B = get_per_token_residuals(model, tok, B_prompt, cfg["layer"])
        h_D, ids_D = get_per_token_residuals(model, tok, D_prompt, cfg["layer"])

        # Token masks
        prefix_len = find_shared_prefix_length(ids_B, ids_D)
        # vignette_mask = [4 .. prefix_len) — drop chat-template header
        vignette_start = cfg.get("chat_prefix_len") or 4
        vignette_end_B = prefix_len  # exclusive
        vignette_end_D = prefix_len
        # Decision token: last position in each prompt (forward-pass length)
        dec_idx_B = h_B.shape[0] - 1
        dec_idx_D = h_D.shape[0] - 1
        # B-only scaffold = positions [prefix_len .. h_B.shape[0]) (includes
        # tail chat-template markers + answer key)

        # Encode all tokens
        with torch.no_grad():
            f_B = sae.encode(h_B.to(args.device)).float().cpu().numpy()  # [seq_B, d_sae]
            f_D = sae.encode(h_D.to(args.device)).float().cpu().numpy()  # [seq_D, d_sae]

        # Save full max-pool feature activations for downstream resampling
        # (task #14). Cast to float32 to save space.
        all_B_max_vignette.append(f_B[vignette_start:vignette_end_B].max(0).astype(np.float32) if vignette_end_B > vignette_start else np.zeros(f_B.shape[1], dtype=np.float32))
        all_D_max_vignette.append(f_D[vignette_start:vignette_end_D].max(0).astype(np.float32) if vignette_end_D > vignette_start else np.zeros(f_D.shape[1], dtype=np.float32))
        all_B_max_content.append(f_B[vignette_start:].max(0).astype(np.float32))
        all_D_max_content.append(f_D[vignette_start:].max(0).astype(np.float32))
        all_B_decision.append(f_B[dec_idx_B].astype(np.float32))
        all_D_decision.append(f_D[dec_idx_D].astype(np.float32))

        # ── (a) Vignette mask: shared positions ────────────────────────
        vig_B = f_B[vignette_start:vignette_end_B][:, med + rnd]
        vig_D = f_D[vignette_start:vignette_end_D][:, med + rnd]
        # Sanity check: should be ~identical (up to bf16 noise)
        vig_diff_med  = float(np.abs(vig_B[:, :len(med)] - vig_D[:, :len(med)]).max())
        vig_diff_rnd  = float(np.abs(vig_B[:, len(med):] - vig_D[:, len(med):]).max())

        # Vignette max-pool sMAPE (should be ~0 since vectors are identical)
        v_B_max_med = vig_B[:, :len(med)].max(0) if vig_B.size else np.zeros(len(med))
        v_D_max_med = vig_D[:, :len(med)].max(0) if vig_D.size else np.zeros(len(med))
        v_B_max_rnd = vig_B[:, len(med):].max(0) if vig_B.size else np.zeros(len(rnd))
        v_D_max_rnd = vig_D[:, len(med):].max(0) if vig_D.size else np.zeros(len(rnd))
        vignette_smape_med = float(smape_per_feature(v_B_max_med, v_D_max_med).mean()) if len(med) else 0.0
        vignette_smape_rnd = float(smape_per_feature(v_B_max_rnd, v_D_max_rnd).mean()) if len(rnd) else 0.0
        vignette_cos_med = cosine(v_B_max_med, v_D_max_med)
        vignette_cos_rnd = cosine(v_B_max_rnd, v_D_max_rnd)

        # ── (b) Decision-token comparison ──────────────────────────────
        dec_B_med = f_B[dec_idx_B, med]
        dec_D_med = f_D[dec_idx_D, med]
        dec_B_rnd = f_B[dec_idx_B, rnd]
        dec_D_rnd = f_D[dec_idx_D, rnd]
        dec_smape_med = float(smape_per_feature(dec_B_med, dec_D_med).mean())
        dec_smape_rnd = float(smape_per_feature(dec_B_rnd, dec_D_rnd).mean())
        dec_cos_med = cosine(dec_B_med, dec_D_med)
        dec_cos_rnd = cosine(dec_B_rnd, dec_D_rnd)

        # ── (c) Full-content max-pool (replicates current headline) ─────
        full_B = f_B[vignette_start:, :][:, med + rnd]
        full_D = f_D[vignette_start:, :][:, med + rnd]
        full_B_max_med = full_B[:, :len(med)].max(0)
        full_D_max_med = full_D[:, :len(med)].max(0)
        full_B_max_rnd = full_B[:, len(med):].max(0)
        full_D_max_rnd = full_D[:, len(med):].max(0)
        full_smape_med = float(smape_per_feature(full_B_max_med, full_D_max_med).mean())
        full_smape_rnd = float(smape_per_feature(full_B_max_rnd, full_D_max_rnd).mean())
        full_cos_med = cosine(full_B_max_med, full_D_max_med)
        full_cos_rnd = cosine(full_B_max_rnd, full_D_max_rnd)

        # ── (d) Per-medical-feature peak position (B vs D) ──────────────
        # Where does each medical feature peak? Is it in the vignette mask?
        peak_diag = []
        for f_id in med:
            argmax_B = int(f_B[vignette_start:, f_id].argmax()) + vignette_start
            argmax_D = int(f_D[vignette_start:, f_id].argmax()) + vignette_start
            peak_diag.append({
                "feature": f_id,
                "B_argmax": argmax_B,
                "D_argmax": argmax_D,
                "B_in_vignette": argmax_B < vignette_end_B,
                "D_in_vignette": argmax_D < vignette_end_D,
                "B_max": float(f_B[argmax_B, f_id]),
                "D_max": float(f_D[argmax_D, f_id]),
                "B_argmax_token": tok.decode([ids_B[argmax_B]]) if argmax_B < len(ids_B) else "?",
                "D_argmax_token": tok.decode([ids_D[argmax_D]]) if argmax_D < len(ids_D) else "?",
            })

        per_case.append({
            "case_id": cid,
            "vignette_token_count": vignette_end_B - vignette_start,
            "B_seq_len": h_B.shape[0],
            "D_seq_len": h_D.shape[0],
            "vignette_max_abs_diff_medical": vig_diff_med,
            "vignette_max_abs_diff_random": vig_diff_rnd,
            "vignette_smape_medical": vignette_smape_med,
            "vignette_smape_random":  vignette_smape_rnd,
            "vignette_cosine_medical": vignette_cos_med,
            "vignette_cosine_random":  vignette_cos_rnd,
            "decision_smape_medical": dec_smape_med,
            "decision_smape_random":  dec_smape_rnd,
            "decision_cosine_medical": dec_cos_med,
            "decision_cosine_random":  dec_cos_rnd,
            "full_smape_medical": full_smape_med,
            "full_smape_random":  full_smape_rnd,
            "full_cosine_medical": full_cos_med,
            "full_cosine_random":  full_cos_rnd,
            "medical_peak_diagnostic": peak_diag,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(case_ids)}] {cid}")

    # ─── Aggregates ───────────────────────────────────────────────────────
    arr = lambda k: np.array([c[k] for c in per_case], dtype=float)
    agg = {
        "n_cases": len(per_case),
        "model_id": cfg["model_id"],
        "sae_repo": cfg["sae_repo"],
        "layer":    cfg["layer"],
        "medical_features": med,
        "random_features":  rnd,
        "vignette_smape_medical_median":  float(np.median(arr("vignette_smape_medical"))),
        "vignette_smape_random_median":   float(np.median(arr("vignette_smape_random"))),
        "vignette_cosine_medical_median": float(np.median(arr("vignette_cosine_medical"))),
        "vignette_cosine_random_median":  float(np.median(arr("vignette_cosine_random"))),
        "decision_smape_medical_median":  float(np.median(arr("decision_smape_medical"))),
        "decision_smape_random_median":   float(np.median(arr("decision_smape_random"))),
        "decision_cosine_medical_median": float(np.median(arr("decision_cosine_medical"))),
        "decision_cosine_random_median":  float(np.median(arr("decision_cosine_random"))),
        "full_smape_medical_median":  float(np.median(arr("full_smape_medical"))),
        "full_smape_random_median":   float(np.median(arr("full_smape_random"))),
        "full_cosine_medical_median": float(np.median(arr("full_cosine_medical"))),
        "full_cosine_random_median":  float(np.median(arr("full_cosine_random"))),
        "vignette_max_abs_diff_medical_max": float(arr("vignette_max_abs_diff_medical").max()),
        "vignette_max_abs_diff_random_max":  float(arr("vignette_max_abs_diff_random").max()),
        "medical_peak_in_vignette_frac_B": float(np.mean(
            [pd["B_in_vignette"] for c in per_case for pd in c["medical_peak_diagnostic"]])),
        "medical_peak_in_vignette_frac_D": float(np.mean(
            [pd["D_in_vignette"] for c in per_case for pd in c["medical_peak_diagnostic"]])),
        "per_case": per_case,
    }
    out_path = ROOT / f"results/phase1b_masked_invariance_{args.model}.json"
    out_path.write_text(json.dumps(agg, indent=2, default=str))
    print(f"\nWrote {out_path}")

    # Dump full per-case max-pool feature vectors for downstream resampling
    npz_path = ROOT / f"results/phase1b_masked_full_activations_{args.model}.npz"
    np.savez_compressed(
        npz_path,
        case_ids=np.array(case_ids, dtype=object),
        medical_features=np.array(med, dtype=np.int64),
        random_features=np.array(rnd, dtype=np.int64),
        B_max_vignette=np.stack(all_B_max_vignette),  # [n_cases, d_sae]
        D_max_vignette=np.stack(all_D_max_vignette),
        B_max_content=np.stack(all_B_max_content),
        D_max_content=np.stack(all_D_max_content),
        B_decision=np.stack(all_B_decision),
        D_decision=np.stack(all_D_decision),
    )
    print(f"Wrote {npz_path} ({npz_path.stat().st_size/1e6:.1f} MB)")

    # ─── Print summary ────────────────────────────────────────────────────
    print(f"\n=== {args.model.upper()} L{cfg['layer']} masked invariance summary ===")
    print(f"Vignette mask (shared content, expected ~0 for both):")
    print(f"  medical sMAPE median: {agg['vignette_smape_medical_median']:.4f}")
    print(f"  random  sMAPE median: {agg['vignette_smape_random_median']:.4f}")
    print(f"  max |B-D| (medical features): {agg['vignette_max_abs_diff_medical_max']:.2e}")
    print(f"  max |B-D| (random  features): {agg['vignette_max_abs_diff_random_max']:.2e}")
    print(f"Decision token (last prompt position):")
    print(f"  medical sMAPE median: {agg['decision_smape_medical_median']:.4f}")
    print(f"  random  sMAPE median: {agg['decision_smape_random_median']:.4f}")
    print(f"  Δ = medical − random: {agg['decision_smape_medical_median'] - agg['decision_smape_random_median']:+.4f}")
    print(f"Full content max-pool (current headline):")
    print(f"  medical sMAPE median: {agg['full_smape_medical_median']:.4f}")
    print(f"  random  sMAPE median: {agg['full_smape_random_median']:.4f}")
    print(f"Peak position of medical features:")
    print(f"  fraction of (case × feature) peaks inside vignette mask: "
          f"B={agg['medical_peak_in_vignette_frac_B']:.1%}, D={agg['medical_peak_in_vignette_frac_D']:.1%}")


if __name__ == "__main__":
    main()
