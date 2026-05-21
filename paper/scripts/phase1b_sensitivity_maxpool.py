"""phase1b_sensitivity_maxpool.py -- K-sweep sensitivity for the
medical-vs-random invariance comparison under MAX-POOL aggregation.

Reviewer concern 4c: the existing Appendix A1B table reports mean-pool
sMAPE values, while the main §4.2 results table reports max-pool values.
This inconsistency lets a reviewer point at K-sweep numbers that don't
match the headline. This script closes that gap.

How: we reuse the saved per-case max-pool activations (B_max_content,
D_max_content) from results/phase1b_masked_full_activations_<model>.npz
(saved during the masked-invariance run on 2026-05-21). For each model,
we also reuse the top-20 contrastively-identified medical features from
the original mean-pool K-sweep results (results/phase1b_sensitivity_<m>_L<L>.json)
and the 30 magnitude-matched random features picked there. We then
compute max-pool sMAPE per case and bootstrap the mean (and the paired
medical-vs-random delta) for K ∈ {3, 5, 10, 20}.

No GPU is needed — all activations are cached in the npz files.

Output: results/phase1b_sensitivity_maxpool.json + .md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

N_BOOTSTRAP = 1000
BOOT_SEED = 42
KS = [3, 5, 10, 20]


def smape_per_feat(b: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Per-feature sMAPE between two arrays of equal shape (along last axis)."""
    num = np.abs(b - d)
    den = (np.abs(b) + np.abs(d)) / 2.0
    return num / np.maximum(den, 1e-8)


def boot_mean_ci(x, n_boot=N_BOOTSTRAP, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = x[idx].mean(axis=1)
    return {"mean": float(x.mean()), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "n": n}


def compute_K_sweep(B_mat, D_mat, medical_ids, random_ids, model_tag, layer):
    """Compute K-sweep sMAPE for medical and random feature pools.

    Per-case sMAPE for a feature subset = mean over features of per-feature sMAPE
    between B and D activation vectors at the case. We then bootstrap over the
    60 cases.
    """
    res = {"model": model_tag, "layer": layer,
           "medical_features": list(medical_ids),
           "random_features":  list(random_ids),
           "K_results": {}}

    # Trim if more medical features available than maximum K
    max_K_med = min(max(KS), len(medical_ids))
    max_K_rnd = min(max(KS), len(random_ids))

    for K in KS:
        K_med = min(K, max_K_med)
        K_rnd = min(K, max_K_rnd)
        if K_med == 0 or K_rnd == 0:
            res["K_results"][str(K)] = {"K_medical": K_med, "K_random": K_rnd,
                                          "note": "insufficient features"}
            continue

        med_ids = medical_ids[:K_med]
        rnd_ids = random_ids[:K_rnd]
        # Per-case per-feature sMAPE -> mean over features -> per-case scalar
        per_case_med = smape_per_feat(B_mat[:, med_ids], D_mat[:, med_ids]).mean(axis=1)
        per_case_rnd = smape_per_feat(B_mat[:, rnd_ids], D_mat[:, rnd_ids]).mean(axis=1)
        med_stats = boot_mean_ci(per_case_med)
        rnd_stats = boot_mean_ci(per_case_rnd)
        delta_stats = boot_mean_ci(per_case_med - per_case_rnd)

        # Sign verdict for paired delta CI
        if delta_stats["hi"] < 0:    verdict = "medical < random (sig)"
        elif delta_stats["lo"] > 0:  verdict = "medical > random (sig)"
        else:                        verdict = "ns"

        res["K_results"][str(K)] = {
            "K_medical": K_med, "K_random": K_rnd,
            "medical": med_stats,
            "random":  rnd_stats,
            "delta_paired_per_case": delta_stats,
            "verdict": verdict,
        }
    return res


def main():
    models = [
        # (tag, npz file, sensitivity-cache file, layer)
        ("4b",  RESULTS / "phase1b_masked_full_activations_4b.npz",
                RESULTS / "phase1b_sensitivity_4b_L29.json", 29),
        ("12b", RESULTS / "phase1b_masked_full_activations_12b.npz",
                RESULTS / "phase1b_sensitivity_12b_L31.json", 31),
    ]
    # Qwen has only 3 medical features and no top-20 contrastive ID available
    # without a fresh GPU run; we report K=3 only.
    qwen_paths = (RESULTS / "phase1b_masked_full_activations_qwen.npz",
                  None, 31)

    out = {"models": {}}
    for tag, npz_path, sens_path, layer in models:
        if not npz_path.exists():
            print(f"skip {tag}: {npz_path} missing"); continue
        if not sens_path.exists():
            print(f"skip {tag}: {sens_path} missing"); continue
        z = np.load(npz_path, allow_pickle=True)
        sens = json.loads(sens_path.read_text())
        medical_ids = sens["medical_features"][:20]
        random_ids  = sens["random_features"][:30]
        B_max = np.asarray(z["B_max_content"], dtype=np.float64)
        D_max = np.asarray(z["D_max_content"], dtype=np.float64)
        print(f"\n=== {tag.upper()} L{layer} ({B_max.shape[0]} cases, d_sae={B_max.shape[1]}) ===")
        res = compute_K_sweep(B_max, D_max, medical_ids, random_ids, tag, layer)
        out["models"][tag] = res
        for K_str, r in res["K_results"].items():
            if "note" in r:
                print(f"  K={K_str}: {r['note']}"); continue
            m = r["medical"]; rn = r["random"]; dl = r["delta_paired_per_case"]
            print(f"  K={r['K_medical']:>2}: med={m['mean']:.4f} [{m['lo']:.4f},{m['hi']:.4f}]  "
                  f"rnd={rn['mean']:.4f} [{rn['lo']:.4f},{rn['hi']:.4f}]  "
                  f"Δ={dl['mean']:+.4f} [{dl['lo']:+.4f},{dl['hi']:+.4f}]  {r['verdict']}")

    # Qwen K=3 (using the medical features from phase4_qwen_L31.json and a
    # 30-feature random pool from today's masked-invariance run)
    npz_q, _, q_layer = qwen_paths
    if npz_q.exists():
        z = np.load(npz_q, allow_pickle=True)
        medical_ids_q = list(z["medical_features"].astype(int))
        random_ids_q  = list(z["random_features"].astype(int))
        B_max = np.asarray(z["B_max_content"], dtype=np.float64)
        D_max = np.asarray(z["D_max_content"], dtype=np.float64)
        print(f"\n=== QWEN L{q_layer} (K=3 only; full K-sweep pending top-20 contrastive ID) ===")
        # Override the KS to just [3] for Qwen since we only have 3 medical features
        global KS
        KS_save = KS
        KS = [3]
        res_q = compute_K_sweep(B_max, D_max, medical_ids_q, random_ids_q, "qwen", q_layer)
        KS = KS_save
        out["models"]["qwen"] = res_q
        for K_str, r in res_q["K_results"].items():
            if "note" in r:
                print(f"  K={K_str}: {r['note']}"); continue
            m = r["medical"]; rn = r["random"]; dl = r["delta_paired_per_case"]
            print(f"  K={r['K_medical']:>2}: med={m['mean']:.4f} [{m['lo']:.4f},{m['hi']:.4f}]  "
                  f"rnd={rn['mean']:.4f} [{rn['lo']:.4f},{rn['hi']:.4f}]  "
                  f"Δ={dl['mean']:+.4f} [{dl['lo']:+.4f},{dl['hi']:+.4f}]  {r['verdict']}")

    out_json = RESULTS / "phase1b_sensitivity_maxpool.json"
    out_json.write_text(json.dumps(out, indent=2, default=lambda x: int(x) if hasattr(x, '__int__') else str(x)))
    print(f"\nWrote {out_json}")

    # ─── Markdown ─────────────────────────────────────────────────────
    md = ["# Phase 1b sensitivity to K — MAX-POOL sMAPE\n"]
    md.append("Companion to Appendix A1B. Closes reviewer Concern 4c (the "
              "appendix table currently reports mean-pool while the main "
              "text reports max-pool — this script gives the max-pool K-sweep "
              "for consistency).\n")
    md.append("Reused activations: saved per-case max-pool features from the "
              "2026-05-21 masked-invariance GPU run (`results/"
              "phase1b_masked_full_activations_*.npz`). Reused medical and "
              "random feature IDs: top-20 contrastively-identified features "
              "from the original mean-pool K-sweep "
              "(`results/phase1b_sensitivity_*_L*.json`).\n")
    md.append("Per-case sMAPE = mean over the K features of per-feature "
              "|B_max − D_max| / ((|B_max| + |D_max|)/2). "
              "Bootstrap CIs: 1000 resamples of the 60 cases. "
              "Δ_paired_per_case is the paired difference (medical − random) "
              "per case, then bootstrap mean and 95% CI.\n")
    for tag, r in out["models"].items():
        md.append(f"## {tag.upper()} L{r['layer']}\n")
        md.append("| K | medical sMAPE (mean, 95% CI) | random sMAPE (mean, 95% CI) | Δ_paired_per_case (mean, 95% CI) | verdict |")
        md.append("|---|---|---|---|---|")
        for K_str, kres in r["K_results"].items():
            if "note" in kres:
                md.append(f"| {K_str} | – | – | – | {kres['note']} |"); continue
            m = kres["medical"]; rn = kres["random"]; dl = kres["delta_paired_per_case"]
            md.append(f"| {kres['K_medical']} | "
                      f"{m['mean']:.4f} [{m['lo']:.4f}, {m['hi']:.4f}] | "
                      f"{rn['mean']:.4f} [{rn['lo']:.4f}, {rn['hi']:.4f}] | "
                      f"{dl['mean']:+.4f} [{dl['lo']:+.4f}, {dl['hi']:+.4f}] | "
                      f"{kres['verdict']} |")
        md.append("")
    md.append("**Note on Qwen:** only K=3 reported, because the top-20 "
              "contrastive medical-feature ID for Qwen3-8B at L31 has not "
              "been run (the v3-validated 3-feature set is what's used in "
              "the main text). A full top-20 contrastive ID for Qwen is "
              "future work and would require ~30 min of A100 time.\n")

    (RESULTS / "phase1b_sensitivity_maxpool.md").write_text("\n".join(md))
    print(f"Wrote {RESULTS/'phase1b_sensitivity_maxpool.md'}")


if __name__ == "__main__":
    main()
