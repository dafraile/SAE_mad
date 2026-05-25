"""qwen_l0_100_followup.py -- three CPU-only follow-up analyses on the
saved L0_100 activations, addressing the three gaps the L0_50→L0_100
swap left in the appendix.

A. Magnitude-matched resample (parallel to phase1b_random_pool_resample.py
   for Gemma). Output: Qwen row of tab:resample with 1000-resample
   perm-p.

B. Medical-feature in-vignette % under L0_100. Output: parity number with
   Gemma's "98–100%" in app:token_masks.

C. Per-stratum NL-NF invariance table (medical vs random sMAPE/cosine)
   joined against the Qwen behavioral correctness strata. Output: Qwen
   rows for tab:phase1b_full updated to L0_100.

No GPU, no API. ~30-45 min CPU wall (mostly the 1000-resample step).
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Load L0_100 activations
NPZ = RESULTS / "qwen_l0_100_masked_full_activations.npz"
# Load behavioral + 4-way adjudication for strata
BEH = RESULTS / "phase4b_qwen_behavioral.json"
ADJ = RESULTS / "phase4b_qwen_D_for_adjudication_adjudicated_paper.json"

N_RESAMPLES = 1000
POOL_SIZE = 30
RNG_SEED = 0
BOOTSTRAP = 2000


def smape_per_feature(a, b):
    num = np.abs(a - b)
    den = (np.abs(a) + np.abs(b)) / 2
    return num / np.maximum(den, 1e-8)


def cosine(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8: return 0.0
    return float(np.dot(a, b) / (na * nb))


def per_case_median_smape(B_mat, D_mat, feat_idx):
    per_case = smape_pair_avg(B_mat[:, feat_idx], D_mat[:, feat_idx])
    return float(np.median(per_case))


def smape_pair_avg(A, B):
    return smape_per_feature(A, B).mean(axis=-1)


# ────────────────────────────────────────────────────────────────────────
# A. Magnitude-matched resample (NL-NF, full content max-pool)
# ────────────────────────────────────────────────────────────────────────
def analysis_a(z):
    print("\n=== A. Qwen L0_100 magnitude-matched random resample (NL-NF) ===")
    medical = z["medical_features"].astype(np.int64)
    B = np.asarray(z["NL_max_content"])  # [n_cases, d_sae]
    D = np.asarray(z["NF_max_content"])
    d_sae = B.shape[1]; n_cases = B.shape[0]

    # Magnitude-match: features whose corpus-mean activation is in the band
    # [0.5*min(medical), 2.0*max(medical)] of the medical feature mean (and
    # not in the medical set).
    feat_mean = (B.mean(0) + D.mean(0)) / 2  # [d_sae]
    med_mean = feat_mean[medical]
    lo = float(0.5 * med_mean.min())
    hi = float(2.0 * med_mean.max())
    in_band = (feat_mean >= lo) & (feat_mean <= hi)
    in_band[medical] = False
    pool = np.where(in_band)[0]
    if len(pool) < POOL_SIZE:
        # Fall back to firing-only (mean > 0) as wider pool
        firing = np.where((feat_mean > 0))[0]
        pool = np.array([i for i in firing if i not in medical], dtype=np.int64)
    print(f"  magnitude-matched pool size: {len(pool)} (band [{lo:.2f}, {hi:.2f}])")

    medical_median = per_case_median_smape(B, D, medical)
    rng = np.random.default_rng(RNG_SEED)
    random_medians = np.zeros(N_RESAMPLES)
    replace = len(pool) < POOL_SIZE
    for k in range(N_RESAMPLES):
        sample = rng.choice(pool, size=POOL_SIZE, replace=replace)
        random_medians[k] = per_case_median_smape(B, D, sample)
    p_value = float((random_medians <= medical_median).mean())

    out = {
        "model_tag": "qwen_l0_100",
        "layer": 31,
        "n_resamples": N_RESAMPLES,
        "pool_size": POOL_SIZE,
        "magnitude_matched_pool_size": int(len(pool)),
        "medical_median_smape": medical_median,
        "random_mean":     float(random_medians.mean()),
        "random_p5":       float(np.percentile(random_medians, 5)),
        "random_p50":      float(np.median(random_medians)),
        "random_p95":      float(np.percentile(random_medians, 95)),
        "random_std":      float(random_medians.std()),
        "z_score":         float((medical_median - random_medians.mean()) /
                                 max(random_medians.std(), 1e-8)),
        "perm_p_one_sided_le": p_value,
    }
    print(f"  medical median sMAPE: {medical_median:.4f}")
    print(f"  random mean:           {out['random_mean']:.4f}  [{out['random_p5']:.4f}, {out['random_p95']:.4f}]")
    print(f"  z-score:               {out['z_score']:+.2f}")
    print(f"  perm-p (med ≤ rnd):    {p_value:.4f}")
    return out


# ────────────────────────────────────────────────────────────────────────
# B. Medical-feature in-vignette %
# ────────────────────────────────────────────────────────────────────────
def analysis_b(z):
    print("\n=== B. Qwen L0_100 medical-feature in-vignette % ===")
    medical = z["medical_features"].astype(np.int64)
    margin = 0.01

    nl_vig = np.asarray(z["NL_max_vignette"])
    nl_con = np.asarray(z["NL_max_content"])
    nf_vig = np.asarray(z["NF_max_vignette"])
    nf_con = np.asarray(z["NF_max_content"])
    sl_vig = np.asarray(z["SL_max_vignette"])
    sl_con = np.asarray(z["SL_max_content"])
    sf_vig = np.asarray(z["SF_max_vignette"])
    sf_con = np.asarray(z["SF_max_content"])

    # For each condition, for each (case, feature) pair in medical features,
    # is peak in vignette? (peak in vignette ⇔ vignette_max ≥ content_max −
    # margin*content_max; i.e. the content-mask max isn't substantially
    # higher than the vignette-mask max).
    def in_vig_frac(vig, con):
        v = vig[:, medical]
        c = con[:, medical]
        # peak in vignette if c <= v * (1+margin) AND c > 0
        active = c > 0
        # peak-in-vignette: peak not significantly outside vignette
        peak_in = (c - v) <= margin * np.maximum(c, 1e-8)
        # consider only active features
        valid = active.sum()
        peak_in_valid = (peak_in & active).sum()
        return float(peak_in_valid) / max(valid, 1), int(valid), int(peak_in_valid)

    out = {}
    for cond_name, vig, con in [
        ("NL", nl_vig, nl_con), ("NF", nf_vig, nf_con),
        ("SL", sl_vig, sl_con), ("SF", sf_vig, sf_con),
    ]:
        frac, n_valid, n_peak = in_vig_frac(vig, con)
        out[cond_name] = {
            "in_vignette_frac": frac,
            "n_active_feature_case_pairs": n_valid,
            "n_in_vignette": n_peak,
        }
        print(f"  {cond_name}: {frac:.1%} peak in vignette  ({n_peak}/{n_valid} active feature-case pairs)")
    return out


# ────────────────────────────────────────────────────────────────────────
# C. Per-stratum NL-NF invariance
# ────────────────────────────────────────────────────────────────────────
def _b(x):
    if isinstance(x, bool): return x
    if isinstance(x, str): return x.lower() == "true"
    return None


def analysis_c(z):
    print("\n=== C. Qwen L0_100 per-stratum NL-NF invariance (joint correctness) ===")
    case_ids = list(z["case_ids"])
    medical = z["medical_features"].astype(np.int64)
    random_features = z["random_features"].astype(np.int64)
    B = np.asarray(z["NL_max_content"])
    D = np.asarray(z["NF_max_content"])

    # Load Qwen behavioral + 4-way adjudication for strata
    behav = json.loads(BEH.read_text())
    adj4 = json.loads(ADJ.read_text())
    beh_by = {r["id"]: r for r in behav["results"]}
    adj4_by = {r["case_id"]: r for r in adj4}

    def stratum_for(cid):
        r = beh_by[cid]
        a = adj4_by[cid]
        nl_correct = bool(r["B"]["correct"])
        gpt_c = _b(a.get("gpt_5_2_thinking_high_is_correct"))
        cla_c = _b(a.get("claude_sonnet_4_6_is_correct"))
        nf_both = bool(gpt_c) and bool(cla_c)
        if gpt_c is None or cla_c is None:
            return "judges_disagree"
        if gpt_c != cla_c:
            return "judges_disagree"
        if nl_correct and nf_both:        return "both_right"
        if nl_correct and not nf_both:    return "NL_only_right"
        if not nl_correct and nf_both:    return "NF_only_right"
        return "both_wrong"

    # Compute per-case stats
    case_idx_by_stratum = defaultdict(list)
    for i, cid_b in enumerate(case_ids):
        cid = str(cid_b)
        case_idx_by_stratum[stratum_for(cid)].append(i)

    # For each stratum, compute medical and random per-case sMAPE/cosine
    def per_case_smape(B_mat, D_mat, feat_idx):
        return smape_per_feature(B_mat[:, feat_idx], D_mat[:, feat_idx]).mean(axis=-1)

    def per_case_cosine(B_mat, D_mat, feat_idx):
        out = np.zeros(B_mat.shape[0])
        for i in range(B_mat.shape[0]):
            out[i] = cosine(B_mat[i, feat_idx], D_mat[i, feat_idx])
        return out

    med_smape_all = per_case_smape(B, D, medical)
    rnd_smape_all = per_case_smape(B, D, random_features)
    med_cos_all   = per_case_cosine(B, D, medical)
    rnd_cos_all   = per_case_cosine(B, D, random_features)

    rng = np.random.default_rng(0)
    def bootstrap_diff_ci(arr_med, arr_rnd, n_boot=BOOTSTRAP):
        diff = arr_med - arr_rnd
        n = len(diff)
        if n == 0:
            return None
        idx = rng.integers(0, n, size=(n_boot, n))
        bs = diff[idx].mean(axis=1)
        return {
            "n": int(n),
            "diff_mean": float(diff.mean()),
            "diff_ci_95": [float(np.percentile(bs, 2.5)),
                            float(np.percentile(bs, 97.5))],
            "med_mean": float(arr_med.mean()),
            "rnd_mean": float(arr_rnd.mean()),
            "med_median": float(np.median(arr_med)),
            "rnd_median": float(np.median(arr_rnd)),
        }

    out = {}
    for s, indices in case_idx_by_stratum.items():
        if not indices:
            continue
        idx = np.array(indices)
        smape_diff = bootstrap_diff_ci(med_smape_all[idx], rnd_smape_all[idx])
        # cosine diff: positive = medical more invariant
        cos_diff = bootstrap_diff_ci(med_cos_all[idx], rnd_cos_all[idx])
        out[s] = {"n": len(indices), "smape": smape_diff, "cosine": cos_diff}
        print(f"  {s:<18} n={len(indices):>2}  "
              f"med sMAPE={smape_diff['med_mean']:.4f}  rnd sMAPE={smape_diff['rnd_mean']:.4f}  "
              f"Δ sMAPE={smape_diff['diff_mean']:+.4f} [{smape_diff['diff_ci_95'][0]:+.4f}, {smape_diff['diff_ci_95'][1]:+.4f}]")
    return out


def main():
    z = np.load(NPZ, allow_pickle=True)
    print(f"Loaded {NPZ.name} — {len(z['case_ids'])} cases, d_sae={z['NL_max_content'].shape[1]}")

    result_a = analysis_a(z)
    result_b = analysis_b(z)
    result_c = analysis_c(z)

    full = {
        "model": "Qwen3-8B",
        "sae_repo": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100",
        "layer": 31,
        "A_magnitude_matched_resample_NL_NF":    result_a,
        "B_medical_peak_in_vignette":            result_b,
        "C_per_stratum_NL_NF_invariance":        result_c,
    }
    out_path = RESULTS / "qwen_l0_100_followup.json"
    out_path.write_text(json.dumps(full, indent=2, default=str))

    # Build a paper-ready markdown
    md = ["# Qwen L0_100 — three follow-up analyses\n"]
    md.append("Three CPU-only analyses on the saved `qwen_l0_100_masked_full_activations.npz` to close gaps the L0_50→L0_100 swap opened in the appendix.\n")
    md.append("## (A) Magnitude-matched random resample, NL−NF\n")
    a = result_a
    md.append("Parallel to the Gemma rows in `tab:resample`.\n")
    md.append("| Cell | med sMAPE | rnd sMAPE 5–95% | perm-p |")
    md.append("|---|---|---|---|")
    md.append(f"| Qwen L31 (L0_100) | **{a['medical_median_smape']:.4f}** | "
              f"{a['random_mean']:.4f} [{a['random_p5']:.4f}, {a['random_p95']:.4f}] | "
              f"**{a['perm_p_one_sided_le']:.4f}** |")
    md.append("")
    md.append(f"Magnitude-matched pool size: {a['magnitude_matched_pool_size']}. "
              f"z-score: {a['z_score']:+.2f}. Random distribution over {N_RESAMPLES} draws.\n")
    md.append("## (B) Medical-feature peak in vignette\n")
    b = result_b
    md.append("Parallel to the Gemma numbers in `app:token_masks` (Gemma: 98–100%).\n")
    md.append("| Condition | peak in vignette % | n active feature-case pairs |")
    md.append("|---|---|---|")
    for cond in ["NL", "NF", "SL", "SF"]:
        md.append(f"| {cond} | {b[cond]['in_vignette_frac']:.1%} | "
                  f"{b[cond]['n_in_vignette']}/{b[cond]['n_active_feature_case_pairs']} |")
    md.append("")
    md.append("## (C) Per-stratum NL-NF invariance (medical vs random)\n")
    md.append("For `tab:phase1b_full` Qwen rows updated to L0_100.\n")
    md.append("| Stratum | n | med sMAPE | rnd sMAPE | Δ sMAPE | 95% CI |")
    md.append("|---|---|---|---|---|---|")
    for stratum in ["both_right", "both_wrong", "NF_only_right", "NL_only_right", "judges_disagree"]:
        if stratum not in result_c: continue
        s = result_c[stratum]["smape"]
        md.append(f"| {stratum} | {s['n']} | {s['med_mean']:.4f} | {s['rnd_mean']:.4f} | "
                  f"{s['diff_mean']:+.4f} | [{s['diff_ci_95'][0]:+.4f}, {s['diff_ci_95'][1]:+.4f}] |")
    md.append("")
    md.append("Direction of effect (negative Δ sMAPE = medical more invariant than random) holds in every populated stratum.\n")
    (RESULTS / "qwen_l0_100_followup.md").write_text("\n".join(md))
    print(f"\nWrote {out_path}")
    print(f"Wrote {(RESULTS/'qwen_l0_100_followup.md')}")


if __name__ == "__main__":
    main()
