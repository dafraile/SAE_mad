"""phase1b_random_pool_resample.py -- resample N random feature pools
from the full activation dumps saved by phase1b_masked_invariance.py and
report the medical-vs-random gap distribution. No GPU needed.

Triggered by reviewer concern: the random baseline uses a single
fixed seed for the 30-feature pool. Case-level bootstrap CIs do not
capture uncertainty over the random-feature sampling. Resample many
times to get the proper null distribution.

For each model, for each of N_RESAMPLES (default 1000) random seeds:
  - sample a 30-feature random pool excluding the medical features
  - compute the median per-case max-pool sMAPE for that pool, across the
    three masks (vignette, decision, full content)
  - record one (medical, random) pair per seed

Report:
  - medical median (single number)
  - random median across the N_RESAMPLES distribution: mean, p5, p95
  - z-score equivalent: how many SDs of the random distribution is
    medical below?
  - one-sided permutation p-value: fraction of random seeds whose
    median sMAPE is at-or-below the medical median (proxy for
    "could the gap be a chance artifact of random feature choice?")

Inputs: results/phase1b_masked_full_activations_<MODEL>.npz
Outputs: results/phase1b_random_pool_resample_<MODEL>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

N_RESAMPLES = 1000
POOL_SIZE = 30
RNG_SEED = 0


def smape_pair(b: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Element-wise sMAPE between two arrays of equal shape."""
    num = np.abs(b - d)
    den = (np.abs(b) + np.abs(d)) / 2
    return num / np.maximum(den, 1e-8)


def per_case_median_smape(B_mat: np.ndarray, D_mat: np.ndarray, feat_idx: np.ndarray) -> float:
    """Median, across cases, of per-case mean-over-features sMAPE.
    B_mat, D_mat: [n_cases, d_sae]
    feat_idx: [n_features] selected feature indices
    Returns: median over cases of mean(sMAPE(B[case, feat_idx], D[case, feat_idx]))
    """
    per_case = smape_pair(B_mat[:, feat_idx], D_mat[:, feat_idx]).mean(axis=-1)  # [n_cases]
    return float(np.median(per_case))


def resample_one_mask(B_mat, D_mat, medical_idx, n_resamples, pool_size, rng,
                      firing_idx=None):
    """Returns dict with medical median, random distribution stats.

    firing_idx: if provided, restrict random sampling to this magnitude-
                matched pool. This is critical — sampling uniformly over
                the full d_sae includes mostly zero-firing features whose
                sMAPE is near zero by denominator floor, artificially
                deflating the random null. The fair test is "is medical
                lower than random, given a random pool that also fires
                on the corpus?"
    """
    d_sae = B_mat.shape[1]
    if firing_idx is None:
        candidate_pool = np.array([i for i in range(d_sae) if i not in medical_idx], dtype=np.int64)
    else:
        candidate_pool = np.array([int(i) for i in firing_idx if int(i) not in medical_idx], dtype=np.int64)

    medical_median = per_case_median_smape(B_mat, D_mat, medical_idx)

    random_medians = np.zeros(n_resamples, dtype=np.float64)
    # If the magnitude-matched pool is smaller than the requested sample size,
    # sample with replacement (bootstrap-style) — this preserves the magnitude-
    # matched constraint at the cost of feature repetition.
    replace = len(candidate_pool) < pool_size
    for k in range(n_resamples):
        sample = rng.choice(candidate_pool, size=pool_size, replace=replace)
        random_medians[k] = per_case_median_smape(B_mat, D_mat, sample)

    p_value = float((random_medians <= medical_median).mean())
    return {
        "medical_median": medical_median,
        "random_mean":   float(random_medians.mean()),
        "random_p5":     float(np.percentile(random_medians, 5)),
        "random_p50":    float(np.median(random_medians)),
        "random_p95":    float(np.percentile(random_medians, 95)),
        "random_std":    float(random_medians.std()),
        "z_score":       float((medical_median - random_medians.mean()) / max(random_medians.std(), 1e-8)),
        "perm_p_one_sided_le": p_value,
        "n_resamples":   n_resamples,
        "candidate_pool_size": int(len(candidate_pool)),
    }


def main():
    out_all = {}
    for model_tag in ["4b", "12b", "qwen"]:
        npz_path = ROOT / f"results/phase1b_masked_full_activations_{model_tag}.npz"
        if not npz_path.exists():
            print(f"skip {model_tag}: {npz_path} not found")
            continue
        z = np.load(npz_path, allow_pickle=True)
        medical = z["medical_features"].astype(np.int64)

        # Identify "firing" pool: features whose corpus-level mean activation
        # on full content is > 0 in either B or D. This matches the original
        # magnitude-matched random-pool selection in phase1b_masked_invariance.py
        # (which sampled random features from firing_idx = features with mean
        # activation > 0 across probe cases).
        B_max_content = np.asarray(z["B_max_content"])
        D_max_content = np.asarray(z["D_max_content"])
        feat_mean = (B_max_content.mean(0) + D_max_content.mean(0)) / 2  # [d_sae]
        firing_idx = np.flatnonzero(feat_mean > 0)
        # Magnitude-matched pool: features that fire above the MIN medical
        # feature activation level. This is a stronger control — same
        # firing regime as the medical features.
        med_floor = float(feat_mean[medical].min())
        mag_matched_idx = np.flatnonzero(feat_mean >= med_floor)
        # Top decile of firing intensity (another magnitude-matched control)
        if firing_idx.size:
            top_decile_threshold = np.percentile(feat_mean[firing_idx], 90)
            top_decile_idx = np.flatnonzero(feat_mean >= top_decile_threshold)
        else:
            top_decile_idx = mag_matched_idx

        print(f"\n=== {model_tag.upper()} | medical features: {medical.tolist()} ===")
        print(f"  d_sae = {z['B_max_content'].shape[1]}, n_cases = {z['B_max_content'].shape[0]}")
        print(f"  firing pool (mean > 0): {len(firing_idx)} features")
        print(f"  magnitude-matched pool (mean ≥ medical-min={med_floor:.2f}): {len(mag_matched_idx)} features")
        print(f"  top-decile firing pool: {len(top_decile_idx)} features")
        rng = np.random.default_rng(RNG_SEED)
        out = {
            "model_tag": model_tag,
            "n_cases": int(z["B_max_content"].shape[0]),
            "d_sae":   int(z["B_max_content"].shape[1]),
            "medical_features": medical.tolist(),
            "n_resamples": N_RESAMPLES,
            "pool_size":   POOL_SIZE,
            "firing_pool_size": int(len(firing_idx)),
            "magnitude_matched_pool_size": int(len(mag_matched_idx)),
            "top_decile_pool_size": int(len(top_decile_idx)),
            "medical_min_mean_activation": med_floor,
        }
        for mask_name, B_key, D_key in [
            ("vignette",      "B_max_vignette", "D_max_vignette"),
            ("full_content",  "B_max_content",  "D_max_content"),
            ("decision",      "B_decision",     "D_decision"),
        ]:
            B = np.asarray(z[B_key])
            D = np.asarray(z[D_key])
            # Run two controls per mask: firing-only, magnitude-matched
            res_firing = resample_one_mask(B, D, medical, N_RESAMPLES, POOL_SIZE, rng,
                                            firing_idx=firing_idx)
            res_mag = resample_one_mask(B, D, medical, N_RESAMPLES, POOL_SIZE, rng,
                                         firing_idx=mag_matched_idx)
            res_topdec = resample_one_mask(B, D, medical, N_RESAMPLES, POOL_SIZE, rng,
                                            firing_idx=top_decile_idx)
            out[mask_name] = {
                "medical_median": res_firing["medical_median"],
                "firing_only":         res_firing,
                "magnitude_matched":   res_mag,
                "top_decile":          res_topdec,
            }
            print(f"  {mask_name:<14s} medical={res_firing['medical_median']:.4f}")
            print(f"      firing-only       random mean={res_firing['random_mean']:.4f}  "
                  f"p5/p95=[{res_firing['random_p5']:.4f},{res_firing['random_p95']:.4f}]  "
                  f"z={res_firing['z_score']:+.2f}  perm-p={res_firing['perm_p_one_sided_le']:.4f}")
            print(f"      magnitude-matched random mean={res_mag['random_mean']:.4f}  "
                  f"p5/p95=[{res_mag['random_p5']:.4f},{res_mag['random_p95']:.4f}]  "
                  f"z={res_mag['z_score']:+.2f}  perm-p={res_mag['perm_p_one_sided_le']:.4f}")
            print(f"      top-decile        random mean={res_topdec['random_mean']:.4f}  "
                  f"p5/p95=[{res_topdec['random_p5']:.4f},{res_topdec['random_p95']:.4f}]  "
                  f"z={res_topdec['z_score']:+.2f}  perm-p={res_topdec['perm_p_one_sided_le']:.4f}")
        out_path = ROOT / f"results/phase1b_random_pool_resample_{model_tag}.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  wrote {out_path}")
        out_all[model_tag] = out

    # Combined summary
    summary_path = ROOT / "results/phase1b_random_pool_resample_summary.json"
    summary_path.write_text(json.dumps(out_all, indent=2))
    print(f"\nWrote combined summary {summary_path}")


if __name__ == "__main__":
    main()
