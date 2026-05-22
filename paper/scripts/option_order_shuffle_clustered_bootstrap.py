"""option_order_shuffle_clustered_bootstrap.py -- clustered bootstrap CIs
for the option-order shuffle stability and accuracy metrics.

Reviewer concern: the K permutations within a case are not independent
(they share the same vignette content, the same model prior, etc.). A
naive bootstrap that resamples (case, permutation) pairs IID will
under-state the variance. The fair test is a case-clustered bootstrap:
resample 60 cases with replacement, aggregate across all K permutations
within each resampled case, then compute the metric.

This script:
  1. Loads results/option_order_shuffle_{model}{,_exhaustive}.json for
     each model
  2. For each model and each metric (same_letter_frac, same_content_frac,
     shuffled_accuracy), draws B=2000 bootstrap samples by resampling
     cases with replacement, then aggregating across permutations within
     each resampled case
  3. Reports the case-clustered 95% CI alongside the point estimate

Outputs:
  results/option_order_shuffle_clustered_bootstrap.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
B = 2000
SEED = 7


def load_run(tag: str, exhaustive: bool):
    suffix = "_exhaustive" if exhaustive else ""
    p = RESULTS / f"option_order_shuffle_{tag}{suffix}.json"
    return json.loads(p.read_text()) if p.exists() else None


def per_case_metric_arrays(data):
    """For each case, compute:
      - same_letter rate (out of K shuffles): how often the shuffled
        prediction matches the original-NL letter
      - same_content rate: how often shuffled prediction's content matches
        original-NL's content
      - shuffled accuracy: how often the shuffled prediction is gold-
        compatible under the case's shuffled letter mapping
    Returns three numpy arrays of shape [n_cases] (mean over the case's K
    shuffles).
    """
    cases = data["per_case"]
    rows_letter, rows_content, rows_acc = [], [], []
    for c in cases:
        orig_letter = c.get("original_pred_letter")
        orig_content = c.get("original_pred_content_id")
        K = len(c["shuffles"])
        if K == 0: continue
        same_letter = sum(1 for s in c["shuffles"] if s.get("pred_letter") == orig_letter) / K
        same_content = sum(1 for s in c["shuffles"] if s.get("pred_content_id") == orig_content) / K
        # shuffled accuracy = fraction of shuffles where pred_letter ∈ gold_letters_under_shuffle
        acc = sum(1 for s in c["shuffles"] if s.get("correct_under_shuffle")) / K
        rows_letter.append(same_letter); rows_content.append(same_content); rows_acc.append(acc)
    return (np.array(rows_letter, dtype=float),
            np.array(rows_content, dtype=float),
            np.array(rows_acc, dtype=float))


def clustered_bootstrap_ci(case_means: np.ndarray, n_boot=B, seed=SEED, alpha=0.05):
    """Case-clustered bootstrap CI on the mean of per-case metrics.
    Resample cases with replacement (each case's K permutations are
    aggregated into the case-mean), compute the grand mean across the
    resampled cases, repeat n_boot times, return percentile CI.
    """
    rng = np.random.default_rng(seed)
    n = len(case_means)
    if n == 0:
        return None
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = case_means[idx].mean(axis=1)
    lo = float(np.percentile(bs, 100 * alpha / 2))
    hi = float(np.percentile(bs, 100 * (1 - alpha / 2)))
    return {
        "point_estimate": float(case_means.mean()),
        "ci_lo_95": lo, "ci_hi_95": hi,
        "se_clustered": float(bs.std(ddof=1)),
        "n_boot": n_boot, "n_cases": n,
    }


def analyze(tag: str, data: dict):
    K = data.get("K_shuffles_per_case")
    n_cases = data["n_cases"]
    letter_arr, content_arr, acc_arr = per_case_metric_arrays(data)

    out = {
        "model_tag": tag,
        "n_cases": int(n_cases),
        "K_shuffles_per_case": int(K) if K is not None else None,
        "n_shuffle_total": int(data.get("n_shuffle_total", 0)),
        "same_letter_frac": clustered_bootstrap_ci(letter_arr),
        "same_content_frac": clustered_bootstrap_ci(content_arr),
        "shuffled_accuracy": clustered_bootstrap_ci(acc_arr),
        "canonical_NL_accuracy": data["accuracy"]["original_accuracy_pct"] / 100,
    }
    return out


def main():
    out_all = {"runs": {}}
    print("\n=== Case-clustered bootstrap CIs ===\n")
    for tag in ("4b", "12b", "qwen"):
        for exh in (False, True):
            data = load_run(tag, exh)
            if data is None: continue
            key = f"{tag}{'_exhaustive' if exh else ''}"
            res = analyze(tag, data)
            out_all["runs"][key] = res
            print(f"--- {key} (n_cases={res['n_cases']}, K={res['K_shuffles_per_case']}) ---")
            for m in ("same_letter_frac", "same_content_frac", "shuffled_accuracy"):
                r = res[m]
                if r is None: continue
                print(f"  {m:<20} = {r['point_estimate']*100:.1f}%  "
                      f"95% CI [{r['ci_lo_95']*100:.1f}%, {r['ci_hi_95']*100:.1f}%]  "
                      f"(SE_clustered = {r['se_clustered']*100:.2f} pp)")

    out_json = RESULTS / "option_order_shuffle_clustered_bootstrap.json"
    out_json.write_text(json.dumps(out_all, indent=2, default=str))

    md = [
        "# Option-order shuffle — case-clustered bootstrap CIs\n",
        f"Reviewer concern: the K permutations within a single case are not "
        f"independent (they share the same vignette content). A naive IID "
        f"bootstrap over (case, permutation) pairs under-states variance. "
        f"This script uses a **case-clustered bootstrap** ({B} resamples): "
        f"draw 60 cases with replacement, aggregate the K permutations within "
        f"each resampled case, compute the grand mean. Percentile CI at α=0.05.\n",
        "## Results\n",
        "| run | K | metric | point | 95% CI (case-clustered) | SE_clustered (pp) |",
        "|---|---|---|---|---|---|",
    ]
    for key, r in out_all["runs"].items():
        for m_name, m_label in (
            ("same_letter_frac",  "same-letter %"),
            ("same_content_frac", "same-content %"),
            ("shuffled_accuracy", "shuffled NL accuracy"),
        ):
            rr = r[m_name]
            if rr is None: continue
            md.append(f"| {key} | {r['K_shuffles_per_case']} | {m_label} | "
                      f"{rr['point_estimate']*100:.1f}% | "
                      f"[{rr['ci_lo_95']*100:.1f}%, {rr['ci_hi_95']*100:.1f}%] | "
                      f"{rr['se_clustered']*100:.2f} |")
    md.append("")
    md.append("Interpretation: a tight CI means the metric is stable across "
              "case resamples (the result is unlikely to be driven by a few "
              "extreme cases). Wider CIs (especially for shuffled accuracy on "
              "n=60 cases) reflect genuine case-level variability.")
    (RESULTS / "option_order_shuffle_clustered_bootstrap.md").write_text("\n".join(md))
    print(f"\nWrote {out_json}")
    print(f"Wrote {RESULTS/'option_order_shuffle_clustered_bootstrap.md'}")


if __name__ == "__main__":
    main()
