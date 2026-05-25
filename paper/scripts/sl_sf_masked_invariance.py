"""sl_sf_masked_invariance.py -- mechanistic invariance for the SL−SF
pair (structured input × {forced-letter, free-text} output).

Parallel to `phase1b_masked_invariance.py` (which does NL−NF), but
pointed at the structured-input pair to demonstrate the format-
effect localization is robust across input style.

For each (model, case):
  - SL prompt = canonical `structured_forced_letter` text
  - SF prompt = SL with the "Reply with exactly one letter only.\nA=\n..."
                scaffold stripped (the construction `build_sf_prompt`
                from sf_behavioral.py)

Forward-pass each, capture residuals at the analyzed layer, SAE-encode,
compute per-token feature activations, max-pool over user content
tokens, then compute medical-vs-random sMAPE and cosine.

Output: `results/sl_sf_masked_invariance_<MODEL_TAG>.json`

Usage:
  python paper/scripts/sl_sf_masked_invariance.py --model 4b
  python paper/scripts/sl_sf_masked_invariance.py --model 12b
  python paper/scripts/sl_sf_masked_invariance.py --model qwen
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import safetensors.torch as sft
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
VIGNETTES_FL = ROOT / "paper/data/canonical_forced_letter_vignettes.json"

MODEL_CONFIGS = {
    "4b": {
        "model_id":         "google/gemma-3-4b-it",
        "sae_repo":         "google/gemma-scope-2-4b-it",
        "layer":            29,
        "sae_kind":         "gemma_jumprelu_resid_post_16k_l0medium",
        "medical_features": [12570, 893, 12845],
        "n_random":         30,
        "random_seed":      42,
    },
    "12b": {
        "model_id":         "google/gemma-3-12b-it",
        "sae_repo":         "google/gemma-scope-2-12b-it",
        "layer":            31,
        "sae_kind":         "gemma_jumprelu_resid_post_16k_l0medium",
        "medical_features": [130, 85, 4773],
        "n_random":         30,
        "random_seed":      42,
    },
    "qwen": {
        "model_id":         "Qwen/Qwen3-8B",
        "sae_repo":         "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
        "layer":            31,
        "sae_kind":         "qwen_topk_64k_k50",
        "medical_features": [29074, 48973, 60699],
        "n_random":         30,
        "random_seed":      42,
    },
}


# ─── SAE loaders (copied verbatim from phase1b_masked_invariance.py) ────────
class GemmaJumpReLUSAE:
    def __init__(self, w_enc, w_dec, b_enc, b_dec, threshold, device):
        self.w_enc = w_enc.to(device); self.w_dec = w_dec.to(device)
        self.b_enc = b_enc.to(device); self.b_dec = b_dec.to(device)
        self.threshold = threshold.to(device)
        self.d_sae   = w_enc.shape[1]
        self.d_model = w_enc.shape[0]

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
        self.w_enc = w_enc.to(device); self.w_dec = w_dec.to(device)
        self.b_enc = b_enc.to(device); self.b_dec = b_dec.to(device)
        self.topk = topk
        self.d_sae   = w_enc.shape[0]
        self.d_model = w_enc.shape[1]

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


def get_layer(model, layer):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer]
    return model.model.layers[layer]


def get_per_token_residuals(model, tok, prompt, layer):
    msgs = [{"role": "user", "content": prompt}]
    try:
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
            return_dict=False, enable_thinking=False,
        )
    except TypeError:
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


def find_shared_prefix_length(ids_a: list[int], ids_b: list[int]) -> int:
    n = min(len(ids_a), len(ids_b))
    for i in range(n):
        if ids_a[i] != ids_b[i]:
            return i
    return n


def smape_per_feature(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.abs(a - b)
    den = (np.abs(a) + np.abs(b)) / 2
    return num / np.maximum(den, 1e-8)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8: return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_sf_prompt(structured_forced_letter: str) -> str:
    """Strip the forced-letter scaffold to get the SF (free-text) variant.
    Matches the construction in sf_behavioral.py."""
    marker = "Reply with exactly one letter only."
    idx = structured_forced_letter.find(marker)
    if idx == -1:
        return structured_forced_letter
    return structured_forced_letter[:idx].rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_CONFIGS), required=True)
    ap.add_argument("--n-cases", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = MODEL_CONFIGS[args.model]

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

    fl = json.loads(VIGNETTES_FL.read_text())
    case_ids = sorted(fl, key=lambda v: (re.match(r"^(\D+)", v["id"]).group(1),
                                          int(re.search(r"\d+", v["id"]).group())))[: args.n_cases]

    # ─── Random feature pool (magnitude-matched on probe cases) ──────────
    print(f"Picking random features (seed {cfg['random_seed']}) ...")
    probe = []
    for v in case_ids[: min(8, len(case_ids))]:
        sl_prompt = v["structured_forced_letter"]
        sf_prompt = build_sf_prompt(sl_prompt)
        for prompt in (sl_prompt, sf_prompt):
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
    print(f"  random  ({len(random_features)}): {random_features[:6]}...")

    med = cfg["medical_features"]
    rnd = random_features

    # ─── Main loop ────────────────────────────────────────────────────────
    per_case = []
    all_SL_max_vignette = []
    all_SF_max_vignette = []
    all_SL_max_content  = []
    all_SF_max_content  = []
    all_SL_decision     = []
    all_SF_decision     = []

    for i, v in enumerate(case_ids):
        cid = v["id"]
        sl_prompt = v["structured_forced_letter"]
        sf_prompt = build_sf_prompt(sl_prompt)

        h_SL, ids_SL = get_per_token_residuals(model, tok, sl_prompt, cfg["layer"])
        h_SF, ids_SF = get_per_token_residuals(model, tok, sf_prompt, cfg["layer"])

        prefix_len = find_shared_prefix_length(ids_SL, ids_SF)
        vignette_start = 4  # after <bos><start_of_turn>user\n for Gemma; close enough for Qwen
        vignette_end = prefix_len

        dec_idx_SL = h_SL.shape[0] - 1
        dec_idx_SF = h_SF.shape[0] - 1

        with torch.no_grad():
            f_SL = sae.encode(h_SL.to(args.device)).float().cpu().numpy()
            f_SF = sae.encode(h_SF.to(args.device)).float().cpu().numpy()

        # Save full per-case max-pool feature vectors (for resampling later)
        if vignette_end > vignette_start:
            all_SL_max_vignette.append(f_SL[vignette_start:vignette_end].max(0).astype(np.float32))
            all_SF_max_vignette.append(f_SF[vignette_start:vignette_end].max(0).astype(np.float32))
        else:
            all_SL_max_vignette.append(np.zeros(f_SL.shape[1], dtype=np.float32))
            all_SF_max_vignette.append(np.zeros(f_SF.shape[1], dtype=np.float32))
        all_SL_max_content.append(f_SL[vignette_start:].max(0).astype(np.float32))
        all_SF_max_content.append(f_SF[vignette_start:].max(0).astype(np.float32))
        all_SL_decision.append(f_SL[dec_idx_SL].astype(np.float32))
        all_SF_decision.append(f_SF[dec_idx_SF].astype(np.float32))

        # Per-case sMAPE / cosine on the three masks
        # (a) Vignette mask: should be ~0 for both medical and random (sanity check)
        vig_SL = f_SL[vignette_start:vignette_end][:, med + rnd] if vignette_end > vignette_start else None
        vig_SF = f_SF[vignette_start:vignette_end][:, med + rnd] if vignette_end > vignette_start else None
        if vig_SL is not None and vig_SL.size:
            v_SL_max_med = vig_SL[:, :len(med)].max(0)
            v_SF_max_med = vig_SF[:, :len(med)].max(0)
            v_SL_max_rnd = vig_SL[:, len(med):].max(0)
            v_SF_max_rnd = vig_SF[:, len(med):].max(0)
            vignette_smape_med = float(smape_per_feature(v_SL_max_med, v_SF_max_med).mean())
            vignette_smape_rnd = float(smape_per_feature(v_SL_max_rnd, v_SF_max_rnd).mean())
            vignette_cos_med = cosine(v_SL_max_med, v_SF_max_med)
            vignette_cos_rnd = cosine(v_SL_max_rnd, v_SF_max_rnd)
        else:
            vignette_smape_med = vignette_smape_rnd = 0.0
            vignette_cos_med = vignette_cos_rnd = 0.0

        # (b) Full content max-pool
        full_SL = f_SL[vignette_start:, :][:, med + rnd]
        full_SF = f_SF[vignette_start:, :][:, med + rnd]
        f_SL_max_med = full_SL[:, :len(med)].max(0)
        f_SF_max_med = full_SF[:, :len(med)].max(0)
        f_SL_max_rnd = full_SL[:, len(med):].max(0)
        f_SF_max_rnd = full_SF[:, len(med):].max(0)
        full_smape_med = float(smape_per_feature(f_SL_max_med, f_SF_max_med).mean())
        full_smape_rnd = float(smape_per_feature(f_SL_max_rnd, f_SF_max_rnd).mean())
        full_cos_med = cosine(f_SL_max_med, f_SF_max_med)
        full_cos_rnd = cosine(f_SL_max_rnd, f_SF_max_rnd)

        # Per-medical-feature peak position in SL vs SF
        peak_diag = []
        for f_id in med:
            argmax_SL = int(f_SL[vignette_start:, f_id].argmax()) + vignette_start
            argmax_SF = int(f_SF[vignette_start:, f_id].argmax()) + vignette_start
            peak_diag.append({
                "feature": f_id,
                "SL_argmax": argmax_SL,
                "SF_argmax": argmax_SF,
                "SL_in_vignette": argmax_SL < vignette_end,
                "SF_in_vignette": argmax_SF < vignette_end,
                "SL_max": float(f_SL[argmax_SL, f_id]),
                "SF_max": float(f_SF[argmax_SF, f_id]),
            })

        per_case.append({
            "case_id": cid,
            "SL_seq_len": h_SL.shape[0],
            "SF_seq_len": h_SF.shape[0],
            "vignette_token_count": vignette_end - vignette_start,
            "vignette_smape_medical": vignette_smape_med,
            "vignette_smape_random":  vignette_smape_rnd,
            "vignette_cosine_medical": vignette_cos_med,
            "vignette_cosine_random":  vignette_cos_rnd,
            "full_smape_medical": full_smape_med,
            "full_smape_random":  full_smape_rnd,
            "full_cosine_medical": full_cos_med,
            "full_cosine_random":  full_cos_rnd,
            "medical_peak_diagnostic": peak_diag,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1:2}/{len(case_ids)}] {cid}  full med={full_smape_med:.4f}  rnd={full_smape_rnd:.4f}")

    # ─── Aggregates ──────────────────────────────────────────────────────
    arr = lambda k: np.array([c[k] for c in per_case], dtype=float)
    agg = {
        "n_cases": len(per_case),
        "model_id": cfg["model_id"],
        "model_tag": args.model,
        "pair": "SL_vs_SF",
        "layer": cfg["layer"],
        "medical_features": med,
        "random_features":  rnd,
        "vignette_smape_medical_median":  float(np.median(arr("vignette_smape_medical"))),
        "vignette_smape_random_median":   float(np.median(arr("vignette_smape_random"))),
        "vignette_cosine_medical_median": float(np.median(arr("vignette_cosine_medical"))),
        "vignette_cosine_random_median":  float(np.median(arr("vignette_cosine_random"))),
        "full_smape_medical_median":      float(np.median(arr("full_smape_medical"))),
        "full_smape_random_median":       float(np.median(arr("full_smape_random"))),
        "full_smape_medical_mean":        float(arr("full_smape_medical").mean()),
        "full_smape_random_mean":         float(arr("full_smape_random").mean()),
        "full_cosine_medical_median":     float(np.median(arr("full_cosine_medical"))),
        "full_cosine_random_median":      float(np.median(arr("full_cosine_random"))),
        "medical_peak_in_vignette_frac_SL": float(np.mean(
            [pd["SL_in_vignette"] for c in per_case for pd in c["medical_peak_diagnostic"]])),
        "medical_peak_in_vignette_frac_SF": float(np.mean(
            [pd["SF_in_vignette"] for c in per_case for pd in c["medical_peak_diagnostic"]])),
        "per_case": per_case,
    }

    # Paired bootstrap CI on the medical - random sMAPE gap (case-level)
    med_arr = arr("full_smape_medical")
    rnd_arr = arr("full_smape_random")
    diff = med_arr - rnd_arr
    rng = np.random.default_rng(0)
    n = len(diff)
    idx = rng.integers(0, n, size=(2000, n))
    bs_means = diff[idx].mean(axis=1)
    agg["paired_smape_diff_mean"] = float(diff.mean())
    agg["paired_smape_diff_95ci"] = [float(np.percentile(bs_means, 2.5)),
                                      float(np.percentile(bs_means, 97.5))]

    out_path = ROOT / f"results/sl_sf_masked_invariance_{args.model}.json"
    out_path.write_text(json.dumps(agg, indent=2, default=str))
    print(f"\nWrote {out_path}")

    npz_path = ROOT / f"results/sl_sf_masked_full_activations_{args.model}.npz"
    np.savez_compressed(
        npz_path,
        case_ids=np.array([c["case_id"] for c in per_case], dtype=object),
        medical_features=np.array(med, dtype=np.int64),
        random_features=np.array(rnd, dtype=np.int64),
        SL_max_vignette=np.stack(all_SL_max_vignette),
        SF_max_vignette=np.stack(all_SF_max_vignette),
        SL_max_content=np.stack(all_SL_max_content),
        SF_max_content=np.stack(all_SF_max_content),
        SL_decision=np.stack(all_SL_decision),
        SF_decision=np.stack(all_SF_decision),
    )
    print(f"Wrote {npz_path} ({npz_path.stat().st_size/1e6:.1f} MB)")

    print(f"\n=== {args.model.upper()} L{cfg['layer']} SL−SF masked invariance summary ===")
    print(f"Vignette mask (shared structured content, expected ~0 for both):")
    print(f"  medical sMAPE median: {agg['vignette_smape_medical_median']:.4f}")
    print(f"  random  sMAPE median: {agg['vignette_smape_random_median']:.4f}")
    print(f"Full content max-pool (the SL-SF headline):")
    print(f"  medical sMAPE median: {agg['full_smape_medical_median']:.4f}  (mean {agg['full_smape_medical_mean']:.4f})")
    print(f"  random  sMAPE median: {agg['full_smape_random_median']:.4f}  (mean {agg['full_smape_random_mean']:.4f})")
    print(f"  medical cosine median: {agg['full_cosine_medical_median']:.4f}")
    print(f"  random  cosine median: {agg['full_cosine_random_median']:.4f}")
    print(f"  paired sMAPE diff (med - rnd): {agg['paired_smape_diff_mean']:+.4f}  95% CI {agg['paired_smape_diff_95ci']}")
    print(f"Medical-feature peak in vignette: SL={agg['medical_peak_in_vignette_frac_SL']:.1%}, "
          f"SF={agg['medical_peak_in_vignette_frac_SF']:.1%}")


if __name__ == "__main__":
    main()
