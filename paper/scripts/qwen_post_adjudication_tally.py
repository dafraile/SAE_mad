"""qwen_post_adjudication_tally.py -- consolidate Qwen3-8B Phase 4b
behavioral + 4-way paper-faithful + 5-way DEFERRED adjudication
outputs into a single bundle for the LaTeX-writer.

Produces:
  results/phase4b_qwen_post_adjudication_tally.json    (machine-readable)
  results/phase4b_qwen_post_adjudication_summary.md    (human-readable)

Fields per case:
  case_id, title, gold_raw, gold_letters,
  sl_correct (heuristic forced-letter A),
  nl_correct (heuristic forced-letter B),
  nf_heuristic_correct (regex-based D),
  nf_gpt_triage, nf_gpt_correct,
  nf_claude_triage, nf_claude_correct,
  nf_both_judges_correct (paper-faithful 4-way), nf_either_correct,
  nf_judges_agree, nf_judges_disagree,
  nf_gpt_deferred, nf_claude_deferred, nf_both_deferred,
  stratum (one of: both_right / NF_only_right / NL_only_right /
                   both_wrong / judges_disagree)

Headline aggregates:
  SL/NL accuracy from heuristic (forced-letter parsing is reliable),
  NF accuracy under: (a) heuristic, (b) GPT only, (c) Claude only,
                     (d) both-judges-correct (paper-faithful headline),
                     (e) either-judges-correct,
                     (f) DEFERRED rate (5-way, unanimous),
  Stratum counts (5 buckets summing to 60),
  Per-acuity (gold A/B/C/D bucket) accuracy.

Mechanistic re-stratification:
  For each stratum and each feature pool {medical, random}, compute
  median max-pool sMAPE and median cosine using phase4_qwen_L31.json's
  per_case dump. Lets us extend the §4.3 Qwen panel from a single
  all-60 row to the same 4-row breakdown 4B/12B have.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BEHAV       = ROOT / "results/phase4b_qwen_behavioral.json"
ADJ_PAPER   = ROOT / "results/phase4b_qwen_D_for_adjudication_adjudicated_paper.json"
ADJ_DEF     = ROOT / "results/phase4b_qwen_adjudicated_deferred.json"
MECH_L31    = ROOT / "results/phase4_qwen_L31.json"

OUT_JSON    = ROOT / "results/phase4b_qwen_post_adjudication_tally.json"
OUT_MD      = ROOT / "results/phase4b_qwen_post_adjudication_summary.md"


def gold_letters(g: str) -> set[str]:
    return set(re.findall(r"[ABCD]", g.upper()))


def smape(a: float, b: float) -> float:
    """Symmetric mean absolute percentage error (single-element)."""
    num = abs(a - b)
    den = (abs(a) + abs(b)) / 2
    return num / max(den, 1e-8)


def max_pool_smape_cosine(b_max: list[float], d_max: list[float]) -> tuple[float, float]:
    """Return (sMAPE_max-pool aggregated, cosine) for a feature subset.
    b_max, d_max are lists of per-feature max activations across the
    response (one number per feature in the pool). Aggregation:
      sMAPE = mean over features of per-feature sMAPE(b, d)
      cosine = cos(b_max_vec, d_max_vec)
    """
    b = np.array(b_max, dtype=float)
    d = np.array(d_max, dtype=float)
    per_feat = np.abs(b - d) / np.maximum((np.abs(b) + np.abs(d)) / 2, 1e-8)
    s = float(per_feat.mean())
    if np.linalg.norm(b) < 1e-8 or np.linalg.norm(d) < 1e-8:
        c = 0.0
    else:
        c = float(np.dot(b, d) / (np.linalg.norm(b) * np.linalg.norm(d)))
    return s, c


def main():
    behav = json.loads(BEHAV.read_text())
    if not ADJ_PAPER.exists():
        sys.exit(f"4-way adjudicator output not found yet: {ADJ_PAPER}")
    if not ADJ_DEF.exists():
        sys.exit(f"5-way DEFERRED adjudicator output not found yet: {ADJ_DEF}")
    adj_paper_rows = json.loads(ADJ_PAPER.read_text())
    adj_def = json.loads(ADJ_DEF.read_text())
    mech = json.loads(MECH_L31.read_text())

    # ── Index 4-way adjudicator rows by case_id ────────────────────────────
    paper_by_id = {r["case_id"]: r for r in adj_paper_rows}

    # ── Index 5-way DEFERRED judgments by (case_id, judge) ────────────────
    def_by_key: dict[tuple[str, str], dict] = {}
    for j in adj_def.get("judgments", []):
        def_by_key[(j["case_id"], j["judge"])] = j

    GPT = "gpt-5.2-thinking-high"
    CLA = "claude-sonnet-4.6"

    # ── Per-case rollup ────────────────────────────────────────────────────
    cases = []
    for r in behav["results"]:
        cid = r["id"]
        gold = r["gold_raw"]
        gold_set = gold_letters(gold)

        # NL heuristic = forced-letter parse, reliable
        sl_correct = bool(r["A"]["correct"])
        nl_correct = bool(r["B"]["correct"])
        nf_heur_correct = bool(r["D"]["correct"])

        # 4-way paper-faithful (A/B/C/D only)
        pr = paper_by_id.get(cid, {})
        gpt_t = pr.get("gpt_5_2_thinking_high_triage")
        gpt_corr = pr.get("gpt_5_2_thinking_high_is_correct")
        cla_t = pr.get("claude_sonnet_4_6_triage")
        cla_corr = pr.get("claude_sonnet_4_6_is_correct")
        # Coerce string-bools just in case
        def _b(x):
            if isinstance(x, bool): return x
            if isinstance(x, str): return x.lower() == "true"
            return None
        gpt_corr_b = _b(gpt_corr)
        cla_corr_b = _b(cla_corr)
        nf_both_correct = bool(gpt_corr_b) and bool(cla_corr_b)
        nf_either_correct = bool(gpt_corr_b) or bool(cla_corr_b)
        nf_judges_agree_on_correctness = (gpt_corr_b is not None and cla_corr_b is not None
                                          and gpt_corr_b == cla_corr_b)
        nf_judges_disagree = not nf_judges_agree_on_correctness

        # 5-way DEFERRED
        gpt_def = def_by_key.get((cid, GPT), {})
        cla_def = def_by_key.get((cid, CLA), {})
        gpt_def_t = gpt_def.get("triage")
        cla_def_t = cla_def.get("triage")
        gpt_is_def = gpt_def_t == "DEFERRED"
        cla_is_def = cla_def_t == "DEFERRED"
        both_deferred = gpt_is_def and cla_is_def
        either_deferred = gpt_is_def or cla_is_def

        # Stratum (paper's 5-bucket schema, using 4-way both-judges-correct)
        if not nf_judges_agree_on_correctness:
            stratum = "judges_disagree"
        elif nl_correct and nf_both_correct:
            stratum = "both_right"
        elif (not nl_correct) and nf_both_correct:
            stratum = "NF_only_right"
        elif nl_correct and (not nf_both_correct):
            stratum = "NL_only_right"
        else:
            stratum = "both_wrong"

        cases.append({
            "case_id": cid, "title": r["title"], "gold_raw": gold,
            "gold_letters": "/".join(sorted(gold_set)),
            "sl_correct": sl_correct,
            "nl_correct": nl_correct,
            "nl_letter": r["B"]["predicted"],
            "sl_letter": r["A"]["predicted"],
            "nf_heuristic_letter": r["D"]["predicted"],
            "nf_heuristic_correct": nf_heur_correct,
            "nf_gpt_triage": gpt_t, "nf_gpt_correct": gpt_corr_b,
            "nf_claude_triage": cla_t, "nf_claude_correct": cla_corr_b,
            "nf_both_judges_correct": nf_both_correct,
            "nf_either_judge_correct": nf_either_correct,
            "nf_judges_disagree": nf_judges_disagree,
            "nf_gpt_5way_triage": gpt_def_t,
            "nf_claude_5way_triage": cla_def_t,
            "nf_gpt_deferred": gpt_is_def,
            "nf_claude_deferred": cla_is_def,
            "nf_both_deferred": both_deferred,
            "nf_either_deferred": either_deferred,
            "stratum": stratum,
        })

    # ── Headline aggregates ───────────────────────────────────────────────
    n = len(cases)
    def acc(field): return sum(1 for c in cases if c[field]) / n

    accuracies = {
        "n": n,
        "SL_forced_letter_heuristic": acc("sl_correct"),
        "NL_forced_letter_heuristic": acc("nl_correct"),
        "NF_heuristic":               acc("nf_heuristic_correct"),
        "NF_gpt":                     sum(1 for c in cases if c["nf_gpt_correct"]) / n,
        "NF_claude":                  sum(1 for c in cases if c["nf_claude_correct"]) / n,
        "NF_both_judges_correct":     acc("nf_both_judges_correct"),
        "NF_either_judge_correct":    acc("nf_either_judge_correct"),
        "deferred_rate_both":         acc("nf_both_deferred"),
        "deferred_rate_either":       acc("nf_either_deferred"),
        "deferred_rate_gpt":          sum(1 for c in cases if c["nf_gpt_deferred"]) / n,
        "deferred_rate_claude":       sum(1 for c in cases if c["nf_claude_deferred"]) / n,
    }
    accuracies["NL_minus_NF_both_judges_pp"] = (
        accuracies["NL_forced_letter_heuristic"] - accuracies["NF_both_judges_correct"]) * 100

    # Stratum counts
    strat_counts = defaultdict(int)
    for c in cases:
        strat_counts[c["stratum"]] += 1

    # Per-acuity (gold-bucket) breakdown
    def acuity_bucket(gold_letters_str):
        # Take the *most-urgent* letter as the acuity bucket
        letters = re.findall(r"[ABCD]", gold_letters_str)
        if not letters: return "?"
        return max(letters, key=lambda L: "ABCD".index(L))
    acuity = defaultdict(lambda: {"n": 0, "sl_correct": 0, "nl_correct": 0,
                                   "nf_both_correct": 0, "nf_either_correct": 0,
                                   "nf_both_deferred": 0})
    for c in cases:
        b = acuity_bucket(c["gold_letters"])
        a = acuity[b]
        a["n"] += 1
        a["sl_correct"]       += int(c["sl_correct"])
        a["nl_correct"]       += int(c["nl_correct"])
        a["nf_both_correct"]  += int(c["nf_both_judges_correct"])
        a["nf_either_correct"]+= int(c["nf_either_judge_correct"])
        a["nf_both_deferred"] += int(c["nf_both_deferred"])

    # ── Mechanistic re-stratification ─────────────────────────────────────
    # For each stratum, compute median max-pool sMAPE and cosine for
    # medical features and for random features. Uses the per_case dump
    # already saved in phase4_qwen_L31.json.
    mech_by_id = {pc["id"]: pc for pc in mech["phase1b"]["per_case"]}
    stratum_pools = defaultdict(list)
    for c in cases:
        stratum_pools[c["stratum"]].append(c["case_id"])

    mech_by_stratum = {}
    for strat, cids in stratum_pools.items():
        med_smapes, med_coses = [], []
        rnd_smapes, rnd_coses = [], []
        for cid in cids:
            pc = mech_by_id.get(cid)
            if not pc: continue
            ms, mc = max_pool_smape_cosine(pc["medical_acts_B_max"], pc["medical_acts_D_max"])
            rs, rc = max_pool_smape_cosine(pc["random_acts_B_max"], pc["random_acts_D_max"])
            med_smapes.append(ms); med_coses.append(mc)
            rnd_smapes.append(rs); rnd_coses.append(rc)
        if not med_smapes:
            continue
        mech_by_stratum[strat] = {
            "n": len(cids),
            "medical_smape_median":  float(median(med_smapes)),
            "medical_cosine_median": float(median(med_coses)),
            "random_smape_median":   float(median(rnd_smapes)),
            "random_cosine_median":  float(median(rnd_coses)),
            "medical_smape_5_95":  [float(np.percentile(med_smapes, 5)),
                                     float(np.percentile(med_smapes, 95))],
            "medical_cosine_5_95": [float(np.percentile(med_coses, 5)),
                                     float(np.percentile(med_coses, 95))],
        }

    bundle = {
        "model": "Qwen/Qwen3-8B",
        "n_cases": n,
        "accuracies": accuracies,
        "stratum_counts": dict(strat_counts),
        "per_acuity": {k: dict(v) for k, v in acuity.items()},
        "mechanistic_by_stratum_L31_max_pool": mech_by_stratum,
        "cases": cases,
    }
    OUT_JSON.write_text(json.dumps(bundle, indent=2))

    # ── Markdown summary ──────────────────────────────────────────────────
    md = []
    md.append(f"# Qwen3-8B post-adjudication tally\n")
    md.append(f"**Model:** Qwen/Qwen3-8B (post-trained) · n={n} cases\n")
    md.append("## Headline accuracies (for §4.1 table row)\n")
    md.append(f"| Cell | Accuracy |\n|---|---|")
    md.append(f"| SL (forced-letter, structured)  | {accuracies['SL_forced_letter_heuristic']*100:5.1f}% (heuristic) |")
    md.append(f"| NL (forced-letter, natural)     | {accuracies['NL_forced_letter_heuristic']*100:5.1f}% (heuristic) |")
    md.append(f"| NF heuristic                    | {accuracies['NF_heuristic']*100:5.1f}% |")
    md.append(f"| NF GPT-5.2-thinking-high (4-way)| {accuracies['NF_gpt']*100:5.1f}% |")
    md.append(f"| NF Claude-Sonnet-4.6 (4-way)    | {accuracies['NF_claude']*100:5.1f}% |")
    md.append(f"| **NF both judges correct (paper-faithful)** | **{accuracies['NF_both_judges_correct']*100:5.1f}%** |")
    md.append(f"| NF either judge correct (envelope) | {accuracies['NF_either_judge_correct']*100:5.1f}% |")
    md.append(f"| **NL−NF (both-correct) gap**    | **{accuracies['NL_minus_NF_both_judges_pp']:+5.1f} pp** |\n")
    md.append("## 5-way DEFERRED rates (§4.2)\n")
    md.append(f"- Both judges DEFERRED (unanimous): **{accuracies['deferred_rate_both']*100:5.1f}%** "
              f"({int(accuracies['deferred_rate_both']*n)}/{n})")
    md.append(f"- Either judge DEFERRED: {accuracies['deferred_rate_either']*100:5.1f}% "
              f"({int(accuracies['deferred_rate_either']*n)}/{n})")
    md.append(f"- GPT-5.2: {accuracies['deferred_rate_gpt']*100:5.1f}% · "
              f"Claude-4.6: {accuracies['deferred_rate_claude']*100:5.1f}%\n")
    md.append("## Stratum counts (5-bucket schema)\n")
    for s in ["both_right", "NF_only_right", "NL_only_right", "both_wrong", "judges_disagree"]:
        c = strat_counts.get(s, 0)
        md.append(f"- `{s:<16s}`: {c}/{n} ({100*c/n:4.1f}%)")
    md.append("")
    md.append("## Per-acuity breakdown (most-urgent gold letter)\n")
    md.append("| Gold acuity | n | SL | NL | NF both | NF either | DEFERRED |\n|---|---|---|---|---|---|---|")
    for b in "ABCD":
        a = acuity.get(b)
        if not a: continue
        md.append(f"| {b} | {a['n']} | {100*a['sl_correct']/a['n']:.0f}% | "
                  f"{100*a['nl_correct']/a['n']:.0f}% | "
                  f"{100*a['nf_both_correct']/a['n']:.0f}% | "
                  f"{100*a['nf_either_correct']/a['n']:.0f}% | "
                  f"{100*a['nf_both_deferred']/a['n']:.0f}% |")
    md.append("")
    md.append("## §4.3 Qwen mechanistic re-stratification (L31, max-pool)\n")
    md.append("| Stratum | n | medical sMAPE | medical cosine | random sMAPE | random cosine |\n"
              "|---|---|---|---|---|---|")
    for s in ["both_right", "NF_only_right", "NL_only_right", "both_wrong", "judges_disagree"]:
        m = mech_by_stratum.get(s)
        if not m: continue
        md.append(f"| {s} | {m['n']} | {m['medical_smape_median']:.3f} | "
                  f"{m['medical_cosine_median']:.3f} | "
                  f"{m['random_smape_median']:.3f} | "
                  f"{m['random_cosine_median']:.3f} |")
    md.append("")
    md.append("## Per-case table (case_id, gold, NL letter, NL correct, NF both-judges correct, stratum)\n")
    md.append("| case_id | gold | NL letter | NL✓ | GPT | Cla | NF both✓ | 5way GPT | 5way Cla | both DEF | stratum |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cases:
        md.append(f"| {c['case_id']} | {c['gold_letters']} | {c['nl_letter'] or '?'} | "
                  f"{'✓' if c['nl_correct'] else '✗'} | "
                  f"{c['nf_gpt_triage'] or '?'} | {c['nf_claude_triage'] or '?'} | "
                  f"{'✓' if c['nf_both_judges_correct'] else '✗'} | "
                  f"{c['nf_gpt_5way_triage'] or '?'} | {c['nf_claude_5way_triage'] or '?'} | "
                  f"{'D' if c['nf_both_deferred'] else ' '} | "
                  f"{c['stratum']} |")
    md.append("")
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print()
    print(f"=== Headline ===")
    for k, v in accuracies.items():
        if isinstance(v, float):
            # values stored as fractions except *_pp which is already in pp
            if k.endswith("_pp"):
                print(f"  {k:38s} {v:+6.2f} pp")
            else:
                print(f"  {k:38s} {v*100:6.2f}%")
        else:
            print(f"  {k:38s} {v}")
    print()
    print("=== Stratum counts ===")
    for s in ["both_right", "NF_only_right", "NL_only_right", "both_wrong", "judges_disagree"]:
        print(f"  {s:<18s}  {strat_counts.get(s,0):>2d}/{n}")


if __name__ == "__main__":
    main()
