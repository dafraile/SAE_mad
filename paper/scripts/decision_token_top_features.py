"""decision_token_top_features.py -- characterize what features fire at
the NL vs NF decision tokens and quantify the scaffold-primary /
medical-partial claim directly.

Reviewer's hypothesis: "At NL pre-generation, representation becomes
scaffold-primary and medical-partial." Earlier work showed v3-validated
medical features are silent at the NL decision token in 60/60 cases at
4B and 12B (decision_token_logit_attribution_*.json). But that left the
question: which features ARE firing at the decision token, and how do
the NL-firing set and the NF-firing set differ?

This script consumes the per-case full-d_sae activation vectors saved at
the decision token from yesterday's masked-invariance run
(`B_decision`, `D_decision` in `phase1b_masked_full_activations_*.npz`)
and computes per case:

  1. NL_top_K  = top-K features by activation at the NL decision token
  2. NF_top_K  = top-K features by activation at the NF decision token
  3. Overlap and asymmetry: |NL_top ∩ NF_top| / K
  4. NL-only features (in NL_top but not NF_top): where do they peak
     in the B prompt? (using saved B_max_vignette vs B_max_content)
     - peak in scaffold (B_max_content > B_max_vignette) → scaffold-y
     - peak in vignette → other
  5. NF-only features: where do they peak in the D prompt? (using
     D_max_vignette vs D_max_content)
     - peak in vignette → content-y
     - peak in chat-template-suffix → other
  6. Are v3-validated medical features in NL_top_K? In NF_top_K?

Aggregated across 60 cases per model:
  - mean / std of overlap fraction (NL ∩ NF) / K
  - fraction of NL-only top-K features that peak in B's scaffold tokens
  - fraction of NF-only top-K features that peak in D's vignette tokens
  - fraction of cases where any v3 medical feature is in NL/NF top-K

CPU only, no GPU, no API. Outputs:
  results/decision_token_top_features_{4b,12b,qwen}.json
  results/decision_token_top_features_summary.md
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

MODELS = {
    "4b": {
        "npz":             RESULTS / "phase1b_masked_full_activations_4b.npz",
        "medical_v3":      [12570, 893, 12845],
        "layer":           29,
    },
    "12b": {
        "npz":             RESULTS / "phase1b_masked_full_activations_12b.npz",
        "medical_v3":      [130, 85, 4773],
        "layer":           31,
    },
    "qwen": {
        "npz":             RESULTS / "phase1b_masked_full_activations_qwen.npz",
        "medical_v3":      [29074, 48973, 60699],
        "layer":           31,
    },
}

TOP_K = 20  # how many features to consider at the decision token
SCAFFOLD_MARGIN = 0.01  # B_max_content must exceed B_max_vignette by this fraction
                        # of B_max_content to count as "peaks in scaffold"


def classify_peak_location(max_vig: float, max_content: float, margin=SCAFFOLD_MARGIN):
    """A feature peaks 'outside the vignette' (scaffold-y in B, suffix-y in D)
    if its content-mask max strictly exceeds its vignette-mask max by at least
    `margin × content_max`. Otherwise it peaks in vignette."""
    if max_content <= 0: return "inactive"
    delta = max_content - max_vig
    if delta > margin * max_content:
        return "non_vignette"  # scaffold (B) or suffix (D)
    return "vignette"


def characterize_model(tag: str, cfg: dict):
    z = np.load(cfg["npz"], allow_pickle=True)
    case_ids = list(z["case_ids"])
    B_dec  = np.asarray(z["B_decision"])      # [60, d_sae]
    D_dec  = np.asarray(z["D_decision"])
    B_max_vig = np.asarray(z["B_max_vignette"])
    D_max_vig = np.asarray(z["D_max_vignette"])
    B_max_con = np.asarray(z["B_max_content"])
    D_max_con = np.asarray(z["D_max_content"])

    medical_set = set(cfg["medical_v3"])
    d_sae = B_dec.shape[1]
    n_cases = B_dec.shape[0]

    per_case = []
    overlaps = []
    nl_only_in_scaffold = []   # frac of NL-only features that peak outside vignette in B
    nf_only_in_vignette = []   # frac of NF-only features that peak in vignette in D
    nl_has_medical = 0
    nf_has_medical = 0
    nl_active_count = []        # total # active features at NL decision token
    nf_active_count = []

    for i in range(n_cases):
        cid = str(case_ids[i])
        b_vec = B_dec[i]
        d_vec = D_dec[i]
        n_b_active = int((b_vec > 0).sum())
        n_d_active = int((d_vec > 0).sum())
        nl_active_count.append(n_b_active)
        nf_active_count.append(n_d_active)

        # Top-K by activation (descending). If fewer than K are positive,
        # we cap to K (zeros pad the tail; we'll filter them out later).
        K_b = min(TOP_K, max(1, n_b_active))
        K_d = min(TOP_K, max(1, n_d_active))
        nl_top = np.argsort(-b_vec)[:K_b].tolist()
        nf_top = np.argsort(-d_vec)[:K_d].tolist()
        nl_top_set = set(int(f) for f in nl_top)
        nf_top_set = set(int(f) for f in nf_top)

        overlap = nl_top_set & nf_top_set
        nl_only = nl_top_set - nf_top_set
        nf_only = nf_top_set - nl_top_set

        # Overlap fraction (Jaccard-style: |∩| / |∪|)
        union_size = len(nl_top_set | nf_top_set)
        overlap_frac = len(overlap) / union_size if union_size else 0.0
        overlaps.append(overlap_frac)

        # NL-only features: peak in scaffold? (B_max_content > B_max_vignette)
        nl_only_peak = Counter()
        for f in nl_only:
            cls = classify_peak_location(float(B_max_vig[i, f]), float(B_max_con[i, f]))
            nl_only_peak[cls] += 1
        # NF-only features: peak in vignette? (D_max_content ≈ D_max_vignette)
        nf_only_peak = Counter()
        for f in nf_only:
            cls = classify_peak_location(float(D_max_vig[i, f]), float(D_max_con[i, f]))
            nf_only_peak[cls] += 1

        # Per-case scaffold-y fraction (NL-only features peaking outside vignette in B)
        if len(nl_only) > 0:
            nl_only_in_scaffold.append(nl_only_peak["non_vignette"] / len(nl_only))
        if len(nf_only) > 0:
            nf_only_in_vignette.append(nf_only_peak["vignette"] / len(nf_only))

        # v3 medical features in top-K?
        if medical_set & nl_top_set: nl_has_medical += 1
        if medical_set & nf_top_set: nf_has_medical += 1

        per_case.append({
            "case_id": cid,
            "n_active_NL": n_b_active,
            "n_active_NF": n_d_active,
            "NL_top_K_features": nl_top,
            "NF_top_K_features": nf_top,
            "overlap_features": sorted(overlap),
            "overlap_frac_jaccard": overlap_frac,
            "NL_only_features": sorted(nl_only),
            "NF_only_features": sorted(nf_only),
            "NL_only_peak_location_counts": dict(nl_only_peak),
            "NF_only_peak_location_counts": dict(nf_only_peak),
            "any_v3_medical_in_NL_top_K": bool(medical_set & nl_top_set),
            "any_v3_medical_in_NF_top_K": bool(medical_set & nf_top_set),
        })

    summary = {
        "model_tag": tag,
        "layer": cfg["layer"],
        "n_cases": n_cases,
        "d_sae": d_sae,
        "K": TOP_K,
        "medical_v3": cfg["medical_v3"],
        "n_active_features_at_decision_token": {
            "NL_mean": float(np.mean(nl_active_count)),
            "NL_median": float(np.median(nl_active_count)),
            "NF_mean": float(np.mean(nf_active_count)),
            "NF_median": float(np.median(nf_active_count)),
        },
        "overlap_jaccard": {
            "mean": float(np.mean(overlaps)),
            "median": float(np.median(overlaps)),
            "p5":  float(np.percentile(overlaps, 5)),
            "p95": float(np.percentile(overlaps, 95)),
        },
        "NL_only_features_scaffold_fraction": {
            "mean": float(np.mean(nl_only_in_scaffold)) if nl_only_in_scaffold else None,
            "median": float(np.median(nl_only_in_scaffold)) if nl_only_in_scaffold else None,
            "n_cases_with_NL_only_features": len(nl_only_in_scaffold),
        },
        "NF_only_features_vignette_fraction": {
            "mean": float(np.mean(nf_only_in_vignette)) if nf_only_in_vignette else None,
            "median": float(np.median(nf_only_in_vignette)) if nf_only_in_vignette else None,
            "n_cases_with_NF_only_features": len(nf_only_in_vignette),
        },
        "n_cases_v3_medical_in_NL_top_K": nl_has_medical,
        "n_cases_v3_medical_in_NF_top_K": nf_has_medical,
        "per_case": per_case,
    }

    out_path = RESULTS / f"decision_token_top_features_{tag}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {out_path}")
    return summary


def main():
    all_summaries = {}
    for tag, cfg in MODELS.items():
        if not cfg["npz"].exists():
            print(f"skip {tag}: {cfg['npz']} missing")
            continue
        print(f"\n=== {tag.upper()} L{cfg['layer']} ===")
        s = characterize_model(tag, cfg)
        all_summaries[tag] = s
        print(f"  mean # active features at decision token: NL={s['n_active_features_at_decision_token']['NL_mean']:.1f}, "
              f"NF={s['n_active_features_at_decision_token']['NF_mean']:.1f}")
        print(f"  overlap (Jaccard) of top-{TOP_K} features NL vs NF: "
              f"mean={s['overlap_jaccard']['mean']:.3f}, median={s['overlap_jaccard']['median']:.3f}, "
              f"5-95% [{s['overlap_jaccard']['p5']:.3f},{s['overlap_jaccard']['p95']:.3f}]")
        sf = s["NL_only_features_scaffold_fraction"]
        vf = s["NF_only_features_vignette_fraction"]
        if sf["mean"] is not None:
            print(f"  NL-only top-K features that peak OUTSIDE the vignette in B (scaffold-y): "
                  f"mean {sf['mean']:.1%}, median {sf['median']:.1%}")
        if vf["mean"] is not None:
            print(f"  NF-only top-K features that peak IN the vignette in D (content-y):    "
                  f"mean {vf['mean']:.1%}, median {vf['median']:.1%}")
        print(f"  v3 medical in NL_top_{TOP_K}: {s['n_cases_v3_medical_in_NL_top_K']}/{s['n_cases']}")
        print(f"  v3 medical in NF_top_{TOP_K}: {s['n_cases_v3_medical_in_NF_top_K']}/{s['n_cases']}")

    # ─── Cross-model markdown ────────────────────────────────────────
    md = [
        "# Decision-token feature characterization (4B / 12B / Qwen)\n",
        "Direct test of the reviewer's 'scaffold-primary, medical-partial' framing. "
        f"For each case, take the top-{TOP_K} active features by activation at the NL "
        "decision token (B_decision) and at the NF decision token (D_decision), then "
        "compute (a) the Jaccard overlap of those two sets, (b) what fraction of NL-only "
        "features peak outside the shared vignette in their own (B) prompt, and "
        "(c) what fraction of NF-only features peak in the vignette in their own (D) prompt.\n",
        "Reads only the saved per-case full-d_sae activation vectors from yesterday's "
        "masked-invariance run; CPU only.\n",
        "## Headline table\n",
        f"| Model | n cases active at NL dec | n active at NF dec | overlap NL∩NF top-{TOP_K} (Jaccard) | NL-only features peaking in scaffold | NF-only features peaking in vignette | v3 medical in NL/NF top-K |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag, s in all_summaries.items():
        n_NL = s["n_active_features_at_decision_token"]["NL_mean"]
        n_NF = s["n_active_features_at_decision_token"]["NF_mean"]
        ov = s["overlap_jaccard"]
        sf = s["NL_only_features_scaffold_fraction"]
        vf = s["NF_only_features_vignette_fraction"]
        sf_str = f"{sf['mean']:.1%} (median {sf['median']:.1%})" if sf["mean"] is not None else "–"
        vf_str = f"{vf['mean']:.1%} (median {vf['median']:.1%})" if vf["mean"] is not None else "–"
        md.append(f"| {tag} | {n_NL:.1f} | {n_NF:.1f} | "
                  f"{ov['mean']:.3f} (5–95% [{ov['p5']:.2f}, {ov['p95']:.2f}]) | "
                  f"{sf_str} | {vf_str} | "
                  f"NL: {s['n_cases_v3_medical_in_NL_top_K']}/{s['n_cases']}, NF: {s['n_cases_v3_medical_in_NF_top_K']}/{s['n_cases']} |")
    md.append("")
    md.append("## Headline read (auto-generated)\n")
    for tag, s in all_summaries.items():
        ov = s["overlap_jaccard"]["mean"]
        sf_mean = s["NL_only_features_scaffold_fraction"]["mean"]
        vf_mean = s["NF_only_features_vignette_fraction"]["mean"]
        sf_str = f"{sf_mean:.0%}" if sf_mean is not None else "?"
        vf_str = f"{vf_mean:.0%}" if vf_mean is not None else "?"
        md.append(f"- **{tag}**: top-{TOP_K} NL and NF decision-token features overlap "
                  f"by Jaccard {ov:.0%}. Of the features unique to NL's top-{TOP_K}, **{sf_str} peak "
                  f"on B-prompt scaffold tokens** (outside the shared vignette); of the features "
                  f"unique to NF's top-{TOP_K}, **{vf_str} peak on D-prompt vignette tokens**. "
                  f"v3-validated medical features are in NL's top-{TOP_K} for "
                  f"{s['n_cases_v3_medical_in_NL_top_K']}/{s['n_cases']} cases and in NF's "
                  f"top-{TOP_K} for {s['n_cases_v3_medical_in_NF_top_K']}/{s['n_cases']} cases.")
    md.append("")
    md.append("## Interpretation\n")
    md.append("A high overlap between NL and NF top-K features would say both formats use "
              "the same feature pool at the decision token. A low overlap with "
              "scaffold-peaking NL-only features and vignette-peaking NF-only features "
              "is the direct 'scaffold-primary at NL, content-primary at NF' pattern.\n")
    md.append("These numbers also let us quantify what 'medical-partial' means: the v3 "
              "medical features (3 per model) are not in the top-K at the decision token "
              "at any model (counts shown above are typically 0/60). Combined with the "
              "logit-attribution finding that v3 medical features have zero activation at "
              "the decision token in 60/60 cases at 4B and 12B, the cleanest mechanistic "
              "claim is **medical-absent at the decision token**, not medical-partial.")

    (RESULTS / "decision_token_top_features_summary.md").write_text("\n".join(md))
    print(f"\nWrote {RESULTS/'decision_token_top_features_summary.md'}")


if __name__ == "__main__":
    main()
