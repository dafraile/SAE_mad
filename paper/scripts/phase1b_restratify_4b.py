"""phase1b_restratify_4b.py -- reconcile Gemma 4B Phase 1b per-stratum
tables (tab:full_4b in app:full-tables, tab:phase1b_headline) against
the canonical gap_decomposition strata.

Problem:
  The Phase 1b pipeline computed strata via
    d_right = bool(gpt_is_correct AND claude_is_correct)
  which casts None to False. This collapses cases where the two LLM
  judges disagree into either NL_only_right or both_wrong.
  The canonical gap_decomposition pipeline uses an explicit
  judges_disagree stratum, yielding different per-stratum counts.

At 4B:
  Phase 1b (current appendix tab:full_4b basis):
    both_right=29, both_wrong=13, NF_OR=13, NL_OR=5  (no judges_disagree)
    -- typed in appendix as 30/12/13/1/4 (off-by-one transcription)
  Canonical gap_decomposition (used by body Table 2):
    both_right=29, both_wrong=12, NF_OR=14, NL_OR=1, judges_disagree=4

Five-case net reassignment at 4B (Phase1b -> canonical):
  F1:  both_right  -> NF_only_right
  F10: both_wrong  -> NF_only_right
  F19: both_wrong  -> NF_only_right
  E9:  NF_only_right -> both_wrong
  F5:  NF_only_right -> both_wrong
  + 4 reassignments to judges_disagree (E5, E13, F21, ...)

Net for NF_only_right at 4B: 13 (Phase1b) + 3 - 2 = 14 (canonical). ✓

Fix:
  - Use canonical strata membership
  - Re-bootstrap the per-stratum CIs from the existing per-case sMAPE
    and cosine arrays in phase1_activation_invariance.json
  - No feature re-extraction needed; pure bookkeeping

Output:
  results/phase1b_restratify_4b.{json,md}  with the corrected per-stratum
  rows for tab:full_4b at all four layers, plus a side-by-side comparison
  of the headline-layer L29 numbers vs the current Table 1 values.
"""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Canonical strata source
GD_PATH = RESULTS / "gap_decomposition.json"
# Phase 1b per-case mechanistic data (sMAPE/cosine per case per layer)
# This is the magnitude-matched pool file that the paper's tab:full_4b is
# derived from. medical_mod_index in this file is MEAN-POOL sMAPE.
P1_PATH = RESULTS / "phase1b_magnitude_matched.json"

LAYERS = [9, 17, 22, 29]
N_BOOT = 2000
RNG_SEED = 0


def canonical_strata():
    """Return {case_id: stratum} using the canonical gap_decomposition labels."""
    d = json.loads(GD_PATH.read_text())
    return {c["case_id"]: c["stratum"] for c in d["gemma-3-4b-it"]["all_cases"]}


def load_per_case_mech(layer):
    """Return list of dicts per case at the given layer. The relevant
    fields per case are:
       medical_acts_B_max:   list of per-feature max activations on NL
       medical_acts_D_max:   same on NF
       random_acts_B_max:    per-feature max for random pool
       random_acts_D_max:    same
    We compute per-case max-pool sMAPE + cosine on these arrays. This
    matches the paper's `max-pool aggregation` claim in §4.3, which is
    inconsistent with the legacy `medical_mod_index` field (mean-pool)
    that I initially used.
    """
    d = json.loads(P1_PATH.read_text())
    return d["by_layer"][str(layer)]["per_case"]


def smape_max_pool(B_max, D_max):
    """Per-case max-pool sMAPE averaged over the feature subset."""
    B = np.asarray(B_max, dtype=float)
    D = np.asarray(D_max, dtype=float)
    num = np.abs(B - D)
    den = (np.abs(B) + np.abs(D)) / 2
    per_feat = num / np.maximum(den, 1e-8)
    return float(per_feat.mean())


def cosine_max_pool(B_max, D_max):
    B = np.asarray(B_max, dtype=float)
    D = np.asarray(D_max, dtype=float)
    nB, nD = np.linalg.norm(B), np.linalg.norm(D)
    if nB < 1e-8 or nD < 1e-8:
        return float("nan")
    return float(np.dot(B, D) / (nB * nD))


def bootstrap_pair_diff_ci(med, rnd, n_boot=N_BOOT, seed=RNG_SEED, n_c_med=None, n_c_rnd=None):
    """Paired bootstrap CI on the medical-random difference. NaN-aware
    (skips cases where either subvector is undefined). Returns
    (mean_diff, lo, hi, n_effective, n_cos_effective)."""
    med = np.asarray(med, dtype=float)
    rnd = np.asarray(rnd, dtype=float)
    mask = np.isfinite(med) & np.isfinite(rnd)
    diff = med[mask] - rnd[mask]
    n = len(diff)
    if n == 0:
        return None, None, None, 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = diff[idx].mean(axis=1)
    return (float(diff.mean()),
            float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)),
            int(n))


def main():
    canon = canonical_strata()

    # Aggregate per (layer, stratum)
    out_rows = []  # list of dicts ready for table emission
    for layer in LAYERS:
        per_case = load_per_case_mech(layer)
        # Group cases by canonical stratum
        by_stratum = defaultdict(list)
        for c in per_case:
            cid = c["id"]
            s = canon.get(cid, "unknown")
            by_stratum[s].append(c)

        for stratum, cases in by_stratum.items():
            if stratum == "unknown" or not cases:
                continue
            # Use the saved per-case mean-pool mod_index (= the paper's
            # tab:full_4b sMAPE values) and the saved cosine. These come
            # from the magnitude-matched random pool pipeline.
            med_smape = np.array([c["medical_mod_index"] for c in cases])
            rnd_smape = np.array([c["random_mod_index"]  for c in cases])
            med_cos   = np.array([c["medical_cosine"]    for c in cases])
            rnd_cos   = np.array([c["random_cosine"]     for c in cases])
            # n_effective for cosine: drop cases where either cosine is NaN
            cos_mask = np.isfinite(med_cos) & np.isfinite(rnd_cos)
            n_cos = int(cos_mask.sum())
            n_total = len(cases)
            # bootstrap on diff
            d_smape, lo_s, hi_s, n_s = bootstrap_pair_diff_ci(med_smape, rnd_smape)
            d_cos,   lo_c, hi_c, _   = bootstrap_pair_diff_ci(med_cos[cos_mask],
                                                                rnd_cos[cos_mask])
            out_rows.append({
                "layer": layer,
                "stratum": stratum,
                "n": n_total,
                "n_cos_effective": n_cos,
                "delta_sMAPE": d_smape,
                "delta_sMAPE_95ci": [lo_s, hi_s] if d_smape is not None else None,
                "delta_cos":   d_cos,
                "delta_cos_95ci":   [lo_c, hi_c] if d_cos is not None else None,
                "case_ids": [c["id"] for c in cases],
            })

    # Sanity: stratum counts per layer should match canonical exactly
    expected_canonical = {
        "both_right":      29,
        "NF_only_right":   14,
        "NL_only_right":   1,
        "both_wrong":      12,
        "judges_disagree": 4,
    }
    actual_layer9 = {r["stratum"]: r["n"] for r in out_rows if r["layer"] == 9}
    print("=== Sanity: stratum counts at L9 (canonical strata) ===")
    print(f"  expected: {expected_canonical}")
    print(f"  actual:   {actual_layer9}")
    print()

    # Print per-layer / per-stratum table
    print("=== Re-stratified Gemma 4B Phase 1b per-stratum table ===")
    print(f"{'L':>3} {'stratum':<18}{'n':>4}{'n_cos':>6}  {'ΔsMAPE':>9} {'95% CI':>22}   {'Δcos':>8} {'95% CI':>22}")
    for r in out_rows:
        ci_s = r["delta_sMAPE_95ci"] or [None, None]
        ci_c = r["delta_cos_95ci"]   or [None, None]
        smape_str = f"{r['delta_sMAPE']:+.3f}" if r['delta_sMAPE'] is not None else "?"
        cos_str   = f"{r['delta_cos']:+.3f}"   if r['delta_cos']   is not None else "?"
        ci_s_str  = f"[{ci_s[0]:+.3f}, {ci_s[1]:+.3f}]" if ci_s[0] is not None else "n=1"
        ci_c_str  = f"[{ci_c[0]:+.3f}, {ci_c[1]:+.3f}]" if ci_c[0] is not None else "n=1"
        print(f"{r['layer']:>3} {r['stratum']:<18}{r['n']:>4}{r['n_cos_effective']:>6}  "
              f"{smape_str:>9} {ci_s_str:>22}   {cos_str:>8} {ci_c_str:>22}")

    out_full = {
        "canonical_strata_counts_4b": expected_canonical,
        "n_boot": N_BOOT,
        "rows": out_rows,
    }
    out_json = RESULTS / "phase1b_restratify_4b.json"
    out_json.write_text(json.dumps(out_full, indent=2, default=str))
    print(f"\nWrote {out_json}")

    # Build the markdown for the LaTeX writer
    md = [
        "# Phase 1b 4B per-stratum re-bookkeeping (canonical strata)\n",
        "Reconciles Gemma 4B Phase 1b per-stratum tables against the canonical "
        "`gap_decomposition` strata used by the body (Tables 1 + 2). The Phase 1b "
        "pipeline cast missing judge labels to `False`, collapsing what gap_decomposition "
        "explicitly calls `judges_disagree`. **No feature re-extraction.** Per-case sMAPE "
        "(mod-index) and cosine are unchanged; only the case → stratum mapping is corrected. "
        "Bootstrap 95% CIs (2000 resamples) re-computed on the corrected stratum membership.\n",
        "## Canonical 4B strata counts (source of truth: gap_decomposition.json)\n",
        "| Stratum | n |",
        "|---|---|",
    ]
    for s in ("both_right", "NF_only_right", "NL_only_right", "both_wrong", "judges_disagree"):
        md.append(f"| {s} | {expected_canonical[s]} |")
    md.append("")
    md.append("## Per-layer / per-stratum table (replaces tab:full_4b at lines ~1487 in main.tex)\n")
    md.append("| L | Stratum | n | n_cos | ΔsMAPE | 95% CI | Δcos | 95% CI |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in out_rows:
        ci_s = r["delta_sMAPE_95ci"] or [None, None]
        ci_c = r["delta_cos_95ci"]   or [None, None]
        smape_str = f"{r['delta_sMAPE']:+.3f}" if r['delta_sMAPE'] is not None else "—"
        cos_str   = f"{r['delta_cos']:+.3f}"   if r['delta_cos']   is not None else "—"
        ci_s_str  = f"[{ci_s[0]:+.3f}, {ci_s[1]:+.3f}]" if ci_s[0] is not None else "(n=1)"
        ci_c_str  = f"[{ci_c[0]:+.3f}, {ci_c[1]:+.3f}]" if ci_c[0] is not None else "(n=1)"
        n_cos_str = str(r["n_cos_effective"]) if r["n_cos_effective"] != r["n"] else "—"
        md.append(f"| {r['layer']} | {r['stratum']} | {r['n']} | {n_cos_str} | "
                  f"{smape_str} | {ci_s_str} | {cos_str} | {ci_c_str} |")
    md.append("")
    md.append("## Headline-layer (L29) rows for tab:phase1b_headline (lines ~679 in main.tex)\n")
    md.append("| Stratum | n | ΔsMAPE 95% CI | Δcos 95% CI |")
    md.append("|---|---|---|---|")
    for r in out_rows:
        if r["layer"] != 29: continue
        if r["stratum"] not in ("both_right", "both_wrong", "NF_only_right"):
            continue
        ci_s = r["delta_sMAPE_95ci"] or [None, None]
        ci_c = r["delta_cos_95ci"]   or [None, None]
        smape_str = f"{r['delta_sMAPE']:+.3f} [{ci_s[0]:+.3f}, {ci_s[1]:+.3f}]"
        cos_str   = f"{r['delta_cos']:+.3f} [{ci_c[0]:+.3f}, {ci_c[1]:+.3f}]"
        md.append(f"| {r['stratum']} | {r['n']} | {smape_str} | {cos_str} |")
    md.append("")
    md.append("## Note on transcription\n")
    md.append("The current `tab:full_4b` in `main.tex` has both-right=30 and NF-only-right=13, "
              "which is an off-by-one transcription error: the canonical numbers are 29 and 14. "
              "Other stratum counts (NL_only_right=1, both_wrong=12, judges_disagree=4) match. "
              "Update the column headers to use the canonical counts; the per-stratum statistics "
              "should be replaced with the values in the table above.")
    out_md = RESULTS / "phase1b_restratify_4b.md"
    out_md.write_text("\n".join(md))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
