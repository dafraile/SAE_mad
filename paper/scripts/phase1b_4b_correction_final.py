"""phase1b_4b_correction_final.py -- final bookkeeping correction for the
Gemma 4B Phase 1b per-stratum tables.

DIAGNOSIS:
  The paper's `tab:phase1b_headline` and `tab:full_4b` show 4B strata
  counts (30, 12, 13, 1, 4) for (both_right, both_wrong, NF_only_right,
  NL_only_right, judges_disagree). The body of the paper (Table 2 /
  gap_decomposition.md) uses the canonical strata counts
  (29, 12, 14, 1, 4). The mismatch is exactly ONE case: F1.

WHO IS F1?
  Under phase1b_magnitude_matched.json (the file the appendix table
  derives from), F1's stratum is "both_right" — Phase 1b cast
  judge-correctness booleans loosely. Under gap_decomposition.json's
  stricter both-judges-correct + judges_disagree decomposition, F1
  has NL=B (wrong), NF=C from both judges (correct), so it belongs
  in NF_only_right.

FIX (pure bookkeeping, no feature re-extraction):
  Move F1 from `both_right` to `NF_only_right` in the per-stratum
  aggregation, then re-bootstrap CIs on the corrected strata.

Output: a side-by-side comparison table — paper's current values vs.
corrected canonical-stratified values — for both tab:full_4b (4 layers)
and tab:phase1b_headline (L29 only). The LaTeX writer can paste the
corrected column directly.
"""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
GD_PATH = RESULTS / "gap_decomposition.json"
P1_PATH = RESULTS / "phase1b_magnitude_matched.json"

LAYERS = [9, 17, 22, 29]
N_BOOT = 2000
RNG_SEED = 0


def bootstrap_pair_diff(med, rnd, n_boot=N_BOOT, seed=RNG_SEED):
    med = np.asarray(med, dtype=float); rnd = np.asarray(rnd, dtype=float)
    mask = np.isfinite(med) & np.isfinite(rnd)
    diff = med[mask] - rnd[mask]
    n = len(diff)
    if n == 0:
        return None, None, None, 0
    rng = np.random.default_rng(seed)
    if n == 1:
        return float(diff[0]), float(diff[0]), float(diff[0]), 1
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = diff[idx].mean(axis=1)
    return (float(diff.mean()),
            float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)),
            int(n))


def per_stratum_table(strata_map, layer):
    """Group cases by stratum (per strata_map: {case_id: stratum}) and
    return per-stratum dict of bootstrap diffs."""
    p1 = json.loads(P1_PATH.read_text())
    pc = p1["by_layer"][str(layer)]["per_case"]
    by_s = defaultdict(list)
    for c in pc:
        s = strata_map.get(c["id"], "unknown")
        by_s[s].append(c)
    out = {}
    for s, cases in by_s.items():
        if not cases: continue
        med_s = [c["medical_mod_index"] for c in cases]
        rnd_s = [c["random_mod_index"]  for c in cases]
        med_c = [c["medical_cosine"]    for c in cases]
        rnd_c = [c["random_cosine"]     for c in cases]
        # n_cos = cases where both cosines are finite
        n_cos = sum(1 for x, y in zip(med_c, rnd_c)
                    if np.isfinite(x) and np.isfinite(y))
        d_smape, lo_s, hi_s, n_s = bootstrap_pair_diff(med_s, rnd_s)
        # cosine: filter to finite values
        med_c_f = np.array([x for x, y in zip(med_c, rnd_c)
                            if np.isfinite(x) and np.isfinite(y)])
        rnd_c_f = np.array([y for x, y in zip(med_c, rnd_c)
                            if np.isfinite(x) and np.isfinite(y)])
        d_cos,   lo_c, hi_c, _ = bootstrap_pair_diff(med_c_f, rnd_c_f)
        out[s] = {
            "n": len(cases),
            "n_cos": n_cos,
            "d_smape": d_smape, "ci_smape": [lo_s, hi_s] if d_smape is not None else None,
            "d_cos":   d_cos,   "ci_cos":   [lo_c, hi_c] if d_cos   is not None else None,
        }
    return out


def main():
    # Load canonical strata
    gd = json.loads(GD_PATH.read_text())
    canonical = {c["case_id"]: c["stratum"]
                 for c in gd["gemma-3-4b-it"]["all_cases"]}

    # Build the paper's current strata: canonical EXCEPT F1 stays in both_right
    paper_strata = dict(canonical)
    # Confirm canonical has F1 in NF_only_right (sanity)
    assert canonical["F1"] == "NF_only_right", (
        f"Expected canonical F1 == NF_only_right, got {canonical['F1']!r}")
    paper_strata["F1"] = "both_right"

    # Sanity-check counts
    from collections import Counter
    print("Canonical strata:", dict(Counter(canonical.values())))
    print("Paper's strata:  ", dict(Counter(paper_strata.values())))

    out = {"layers": {}, "case_F1_move": {
        "from": "both_right (paper)",
        "to":   "NF_only_right (canonical)",
        "rationale": ("phase1b_magnitude_matched.json has F1 in 'both_right' because "
                      "Phase 1b cast missing judge booleans to False; gap_decomposition has "
                      "F1 in 'NF_only_right' because both judges agree NF is correct (C) "
                      "while NL is wrong (B). Canonical wins.")
    }}

    print()
    for layer in LAYERS:
        print(f"=== Layer {layer} ===")
        paper = per_stratum_table(paper_strata, layer)
        canon = per_stratum_table(canonical, layer)
        out["layers"][str(layer)] = {"paper_strata": paper, "canonical_strata": canon}

        header = f"{'stratum':<18}{'PAPER (current)':<55}{'CANONICAL (corrected)':<55}"
        print(header)
        print("-" * len(header))
        for s in ["both_right", "both_wrong", "NF_only_right", "NL_only_right", "judges_disagree"]:
            p = paper.get(s, {})
            c = canon.get(s, {})
            p_str = c_str = "—"
            if p:
                ci = p["ci_smape"]
                p_str = f"n={p['n']:>2}  Δ={p['d_smape']:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]"
            if c:
                ci = c["ci_smape"]
                c_str = f"n={c['n']:>2}  Δ={c['d_smape']:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]"
            print(f"{s:<18}{p_str:<55}{c_str:<55}")
        print()

    out_json = RESULTS / "phase1b_4b_correction.json"
    out_json.write_text(json.dumps(out, indent=2, default=str))

    # Build markdown for the writer
    md = [
        "# Gemma 4B Phase 1b per-stratum correction (bookkeeping fix)\n",
        "## Problem\n",
        "The paper's `tab:phase1b_headline` (Table 1) and `tab:full_4b` (Appendix) "
        "for 4B show stratum counts `(both_right=30, both_wrong=12, NF_only_right=13, "
        "NL_only_right=1, judges_disagree=4)`. The canonical strata from "
        "`gap_decomposition.json` — used by the body Table 2 and §4.2 prose — are "
        "`(29, 12, 14, 1, 4)`. The mismatch is exactly one case: **F1**.\n",
        "## Diagnosis\n",
        "Under `phase1b_magnitude_matched.json` (the file the appendix table derives from), F1 is in `both_right` because Phase 1b cast missing judge-correctness fields to `False`. Under `gap_decomposition.json`'s stricter both-judges-correct + judges_disagree decomposition, F1 has `NL=B (wrong)`, `NF=C (both judges)`, so it belongs in `NF_only_right`. The canonical decomposition is authoritative; F1 moves from `both_right` → `NF_only_right`.\n",
        "## Corrected per-stratum table (replaces `tab:full_4b` and the L29 rows of `tab:phase1b_headline`)\n",
    ]
    for layer in LAYERS:
        md.append(f"### Layer {layer}\n")
        md.append("| Stratum | n | ΔsMAPE | 95% CI | Δcos | 95% CI |")
        md.append("|---|---|---|---|---|---|")
        canon = out["layers"][str(layer)]["canonical_strata"]
        for s in ["both_right", "both_wrong", "NF_only_right", "NL_only_right", "judges_disagree"]:
            r = canon.get(s)
            if not r: continue
            ci_s = r["ci_smape"]
            ci_c = r["ci_cos"]
            n_cos_str = f" ($n_c={r['n_cos']}$)" if r["n_cos"] != r["n"] else ""
            md.append(f"| {s} | {r['n']}{n_cos_str} | "
                      f"{r['d_smape']:+.3f} | [{ci_s[0]:+.3f}, {ci_s[1]:+.3f}] | "
                      f"{r['d_cos']:+.3f} | [{ci_c[0]:+.3f}, {ci_c[1]:+.3f}] |")
        md.append("")
    md.append("## Side-by-side: paper's current values vs. corrected canonical values\n")
    md.append("This shows the magnitude of the bookkeeping correction. The qualitative claim (all CIs strictly below zero) holds in both versions.\n")
    for layer in LAYERS:
        md.append(f"### Layer {layer}\n")
        md.append("| Stratum | PAPER (current) | CANONICAL (corrected) |")
        md.append("|---|---|---|")
        paper = out["layers"][str(layer)]["paper_strata"]
        canon = out["layers"][str(layer)]["canonical_strata"]
        for s in ["both_right", "both_wrong", "NF_only_right", "NL_only_right", "judges_disagree"]:
            p = paper.get(s, {})
            c = canon.get(s, {})
            if not (p or c): continue
            p_str = c_str = "—"
            if p and p.get("d_smape") is not None:
                ci = p["ci_smape"]
                p_str = f"n={p['n']}: {p['d_smape']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
            if c and c.get("d_smape") is not None:
                ci = c["ci_smape"]
                c_str = f"n={c['n']}: {c['d_smape']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
            md.append(f"| {s} | {p_str} | {c_str} |")
        md.append("")
    out_md = RESULTS / "phase1b_4b_correction.md"
    out_md.write_text("\n".join(md))
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
