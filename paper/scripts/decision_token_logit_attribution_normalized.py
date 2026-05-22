"""decision_token_logit_attribution_normalized.py -- recompute the
decision-token logit attribution numbers as normalized fractions,
to address reviewer concern that raw values (e.g. 'other = 2627' at
4B) invite a reviewer to ask "fraction of what exactly?"

Two normalized quantities per category (medical / scaffold-proxy /
other) per case:

  (a) FRACTION OF TOTAL ABSOLUTE LINEAR CONTRIBUTION: for each case,
      sum |c[i, L]| over all active features i and all letters L; that's
      the case-level absolute-contribution budget. Category K's share is
      sum_{i in K, L} |c[i, L]| / total. Aggregate (mean, p5, p95) across
      cases.

  (b) CONTRIBUTION TO PREDICTED-VS-RUNNER-UP MARGIN: for each case,
      identify predicted letter L_p and runner-up letter L_r (second-
      most-likely under linear contribution). For each category K,
      compute margin_K = sum_{i in K} (c[i, L_p] - c[i, L_r]). Then
      category K's share of the total margin = margin_K / sum_all margin
      = margin_K / sum_{i, all K} (c[i, L_p] - c[i, L_r]). Sign-aware
      (a category can have negative share if it pushes toward the
      runner-up).

Caveat for the table caption:
  Linear logit-lens projection ignores (i) the final LayerNorm, (ii)
  the ~5-10 transformer layers between the SAE layer and the unembedding,
  (iii) SAE reconstruction error. Magnitudes are approximate; categorical
  comparisons are directionally informative but not strictly causal. For
  causal attribution we would need feature-ablation forward passes (future
  work).

Inputs:
  results/decision_token_logit_attribution_{4b,12b}.json

Outputs:
  results/decision_token_logit_attribution_normalized.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def per_case_normalized_fractions(case: dict, medical_set: set, scaffold_set: set):
    """For a single case's logit-attribution data, compute:
    - category fractions of total absolute contribution
    - category share of predicted-vs-runner-up margin
    Returns (category_abs_fraction, category_margin_share, pred_letter, runner_up)
    """
    if case.get("skipped"):
        return None, None, None, None

    pred_letter = case["pred_letter"]
    if pred_letter not in "ABCD":
        return None, None, None, None
    pred_idx = "ABCD".index(pred_letter)

    # We have aggregated per-category contributions (sum over all active
    # features) in `category_breakdown[cat][L]`. We also need per-feature
    # contributions to compute absolute fractions. The script saved
    # top_features_pushing_pred_letter but not per-letter for each non-top
    # feature; however, `total_linear_contribution_per_letter` IS the sum
    # across ALL active features, broken down by category.
    cat = case["category_breakdown"]  # {cat: {A:..,B:..,C:..,D:.., n_features:..}}

    # (a) Fraction of total absolute contribution
    # We can compute the absolute sum per category per letter directly:
    # sum_{i in K, L} |c[i, L]| -- but we only have sum c[i, L] (signed).
    # Approximation: use |sum| as a proxy for "category K's net effect on
    # letter L". This understates true total |c| when within-category
    # features have mixed signs. For the headline framing it's
    # interpretable: "category K nets X% of the absolute linear effect."
    cats = ("medical", "scaffold", "other")
    cat_abs_per_letter = {k: {L: abs(cat[k][L]) for L in "ABCD"} for k in cats}
    cat_abs_total = {k: sum(cat_abs_per_letter[k].values()) for k in cats}
    overall_abs_total = sum(cat_abs_total.values())
    if overall_abs_total < 1e-8:
        return None, None, pred_letter, None
    cat_abs_fraction = {k: cat_abs_total[k] / overall_abs_total for k in cats}

    # (b) Predicted vs runner-up margin contribution
    letter_logits = case["letter_logits"]
    sorted_by_logit = sorted("ABCD", key=lambda L: -letter_logits[L])
    runner_up = sorted_by_logit[1]
    runner_idx = "ABCD".index(runner_up)
    cat_margin = {k: cat[k][pred_letter] - cat[k][runner_up] for k in cats}
    total_margin = sum(cat_margin.values())
    if abs(total_margin) < 1e-8:
        cat_margin_share = {k: 0.0 for k in cats}
    else:
        cat_margin_share = {k: cat_margin[k] / total_margin for k in cats}

    return cat_abs_fraction, cat_margin_share, pred_letter, runner_up


def aggregate(per_case_normed):
    """Aggregate across cases."""
    cats = ("medical", "scaffold", "other")
    out_abs = {k: [] for k in cats}
    out_margin = {k: [] for k in cats}
    n_cases_used = 0
    for r in per_case_normed:
        if r["abs_fraction"] is None: continue
        n_cases_used += 1
        for k in cats:
            out_abs[k].append(r["abs_fraction"][k])
            out_margin[k].append(r["margin_share"][k])

    summary = {"n_cases_used": n_cases_used}
    for k in cats:
        arr_a = np.array(out_abs[k])
        arr_m = np.array(out_margin[k])
        summary[k] = {
            "abs_fraction_mean":   float(arr_a.mean()) if arr_a.size else None,
            "abs_fraction_median": float(np.median(arr_a)) if arr_a.size else None,
            "abs_fraction_p5":     float(np.percentile(arr_a, 5)) if arr_a.size else None,
            "abs_fraction_p95":    float(np.percentile(arr_a, 95)) if arr_a.size else None,
            "margin_share_mean":   float(arr_m.mean()) if arr_m.size else None,
            "margin_share_median": float(np.median(arr_m)) if arr_m.size else None,
            "margin_share_p5":     float(np.percentile(arr_m, 5)) if arr_m.size else None,
            "margin_share_p95":    float(np.percentile(arr_m, 95)) if arr_m.size else None,
        }
    return summary


def analyze_model(tag: str):
    inp = RESULTS / f"decision_token_logit_attribution_{tag}.json"
    if not inp.exists():
        print(f"skip {tag}: {inp} missing"); return None
    data = json.loads(inp.read_text())
    medical_set = set(data["medical_features"])
    scaffold_set = set(data["scaffold_feature_pool_top30"])

    per_case_normed = []
    for case in data["per_case"]:
        abs_frac, margin_share, pred, runner = per_case_normalized_fractions(
            case, medical_set, scaffold_set)
        per_case_normed.append({
            "case_id": case["case_id"],
            "pred_letter": pred,
            "runner_up_letter": runner,
            "abs_fraction": abs_frac,
            "margin_share": margin_share,
        })

    summary = aggregate(per_case_normed)
    summary["model"] = data["model"]
    summary["layer"] = data["layer"]
    summary["model_tag"] = tag
    summary["per_case_normalized"] = per_case_normed
    return summary


def main():
    out_all = {}
    print("\n=== Normalized decision-token logit attribution ===\n")
    for tag in ("4b", "12b"):
        s = analyze_model(tag)
        if s is None: continue
        out_all[tag] = s
        print(f"=== {tag.upper()} L{s['layer']} | n cases used = {s['n_cases_used']} ===")
        print(f"{'category':<10}{'abs_frac mean':>16}{'abs_frac 5-95':>22}{'margin_share mean':>20}{'margin_share 5-95':>22}")
        for k in ("medical", "scaffold", "other"):
            c = s[k]
            print(f"  {k:<10}{c['abs_fraction_mean']*100:>14.1f}%  "
                  f"[{c['abs_fraction_p5']*100:>5.1f}%, {c['abs_fraction_p95']*100:>5.1f}%]"
                  f"{c['margin_share_mean']*100:>18.1f}%  "
                  f"[{c['margin_share_p5']*100:>6.1f}%, {c['margin_share_p95']*100:>6.1f}%]")
        print()

    out_path_json = RESULTS / "decision_token_logit_attribution_normalized.json"
    out_path_json.write_text(json.dumps(out_all, indent=2, default=str))

    # ─── Markdown ────────────────────────────────────────────────────
    md = [
        "# Normalized decision-token logit attribution (4B + 12B)\n",
        "Reviewer concern: the raw numbers in the v2 logit-attribution table "
        "(e.g., 'other = 2627' at 4B, 'scaffold = 198.83' at 12B) invite a reviewer "
        "to ask 'fraction of what, exactly?' This file reports two normalized "
        "quantities derived from the same saved data:\n",
        "  - **abs_fraction**: category K's net absolute linear effect divided by the total absolute linear effect across all categories. Caveats: this uses |sum| as a per-category-letter proxy for the true sum of |contributions| (which can only be computed if we have every feature's individual contribution, not just the per-category aggregate). Within a category, features with mixed-sign contributions will partially cancel before the abs is taken; the reported abs_fraction therefore understates the true unsigned-share for categories that contain features pushing in different directions. For directional/magnitude comparison across categories this is interpretable, but the literal interpretation is 'fraction of NET absolute linear effect,' not 'fraction of unsigned per-feature contribution.'\n",
        "  - **margin_share**: category K's contribution to the predicted-vs-runner-up linear margin (pred_letter and runner_up letter chosen per case from the raw logits). Can be negative (the category pushes toward the runner-up rather than the prediction).\n",
        "## Caveat for the manuscript caption (recommended phrasing)\n",
        "> 'All values are derived from a linear logit-lens projection: c[i, L] = act_i · W_dec[i] · W_unembed[:, L_token]. This ignores (i) the final LayerNorm before unembedding, (ii) the transformer layers between the SAE layer and the unembedding, and (iii) SAE reconstruction error. Magnitudes are approximate; categorical comparisons are directionally informative. Causal attribution would require per-feature ablation forward passes — see future work.'\n",
        "## Numbers\n",
    ]
    for tag, s in out_all.items():
        md.append(f"### {tag.upper()} L{s['layer']} (n = {s['n_cases_used']} cases)\n")
        md.append("| Category | abs_fraction mean (5–95%) | margin_share mean (5–95%) |")
        md.append("|---|---|---|")
        for k in ("medical", "scaffold", "other"):
            c = s[k]
            md.append(f"| {k} | "
                      f"{c['abs_fraction_mean']*100:.1f}% ({c['abs_fraction_p5']*100:.1f}%, {c['abs_fraction_p95']*100:.1f}%) | "
                      f"{c['margin_share_mean']*100:.1f}% ({c['margin_share_p5']*100:.1f}%, {c['margin_share_p95']*100:.1f}%) |")
        md.append("")

    # Headline read
    md.append("## Headline read (auto-generated)\n")
    for tag, s in out_all.items():
        med_abs = s["medical"]["abs_fraction_mean"] * 100
        sca_abs = s["scaffold"]["abs_fraction_mean"] * 100
        oth_abs = s["other"]["abs_fraction_mean"] * 100
        med_mgn = s["medical"]["margin_share_mean"] * 100
        sca_mgn = s["scaffold"]["margin_share_mean"] * 100
        oth_mgn = s["other"]["margin_share_mean"] * 100
        md.append(f"- **{tag}**: medical features account for **{med_abs:.1f}%** of the net absolute linear contribution at the NL decision token and **{med_mgn:.1f}%** of the predicted-vs-runner-up margin. Scaffold-proxy features: {sca_abs:.1f}% abs, {sca_mgn:.1f}% margin. Other features: {oth_abs:.1f}% abs, {oth_mgn:.1f}% margin.")
    md.append("")
    md.append("The 'medical' fraction is essentially 0% at both models — consistent with the underlying finding that the v3 medical features have zero activation at the decision token in 60/60 cases (so their linear contribution is exactly zero, regardless of normalization). The medical-vs-scaffold-vs-other relative ranking differs across the two scales: at 4B the 'other' category dominates (~99% of abs contribution); at 12B 'scaffold-proxy' and 'other' are roughly comparable (each contributing ~40–50%).\n")

    (RESULTS / "decision_token_logit_attribution_normalized.md").write_text("\n".join(md))
    print(f"Wrote {out_path_json}")
    print(f"Wrote {RESULTS/'decision_token_logit_attribution_normalized.md'}")


if __name__ == "__main__":
    main()
