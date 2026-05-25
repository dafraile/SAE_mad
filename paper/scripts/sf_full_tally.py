"""sf_full_tally.py -- build the full SL/NL/NF/SF 2x2 comparison table
once the SF cell adjudication completes.

Reads:
  - SL/NL accuracy from existing behavioral JSONs (heuristic letter parse
    is reliable for forced-letter)
  - NF accuracy from existing 4-way + 5-way adjudicator outputs
  - SF accuracy from the new sf_*_D_for_adjudication_adjudicated_paper.json
    + sf_*_adjudicated_deferred.json

Writes:
  results/sf_2x2_comparison.{json,md}
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
V2 = RESULTS / "_v2"

GPT = "gpt-5.2-thinking-high"
CLA = "claude-sonnet-4.6"
GPT_FIELD = "gpt_5_2_thinking_high"
CLA_FIELD = "claude_sonnet_4_6"

MODELS = {
    "4b": {
        "behavioral":   V2 / "phase0_5_three_cells.json",
        "nf_paper":     V2 / "phase0_5_D_for_adjudication_adjudicated_paper.json",
        "nf_deferred":  V2 / "phase0_5_adjudicated_deferred.json",
        "sf_behavioral":RESULTS / "sf_behavioral_4b.json",
        "sf_paper":     RESULTS / "sf_4b_D_for_adjudication_adjudicated_paper.json",
        "sf_deferred":  RESULTS / "sf_4b_adjudicated_deferred.json",
    },
    "12b": {
        "behavioral":   V2 / "phase3b_12b_phase0_5.json",
        "nf_paper":     V2 / "phase3b_12b_D_for_adjudication_adjudicated_paper.json",
        "nf_deferred":  V2 / "phase3b_12b_adjudicated_deferred.json",
        "sf_behavioral":RESULTS / "sf_behavioral_12b.json",
        "sf_paper":     RESULTS / "sf_12b_D_for_adjudication_adjudicated_paper.json",
        "sf_deferred":  RESULTS / "sf_12b_adjudicated_deferred.json",
    },
    "qwen": {
        "behavioral":   RESULTS / "phase4b_qwen_behavioral.json",
        "nf_paper":     RESULTS / "phase4b_qwen_D_for_adjudication_adjudicated_paper.json",
        "nf_deferred":  RESULTS / "phase4b_qwen_adjudicated_deferred.json",
        "sf_behavioral":RESULTS / "sf_behavioral_qwen.json",
        "sf_paper":     RESULTS / "sf_qwen_D_for_adjudication_adjudicated_paper.json",
        "sf_deferred":  RESULTS / "sf_qwen_adjudicated_deferred.json",
    },
}


def _b(x):
    if isinstance(x, bool): return x
    if isinstance(x, str):  return x.lower() == "true"
    return None


def both_judges_correct_acc(adj_paper_path: Path) -> dict:
    rows = json.loads(adj_paper_path.read_text())
    n = len(rows)
    n_both = 0
    n_either = 0
    n_gpt_correct = 0
    n_cla_correct = 0
    for r in rows:
        g = _b(r.get(f"{GPT_FIELD}_is_correct"))
        c = _b(r.get(f"{CLA_FIELD}_is_correct"))
        if g: n_gpt_correct += 1
        if c: n_cla_correct += 1
        if g and c: n_both += 1
        if g or c:  n_either += 1
    return {
        "n": n,
        "n_both_correct": n_both,
        "n_either_correct": n_either,
        "n_gpt_correct": n_gpt_correct,
        "n_cla_correct": n_cla_correct,
        "both_acc_pct":   100 * n_both / n,
        "either_acc_pct": 100 * n_either / n,
    }


def deferred_rates(adj_def_path: Path) -> dict:
    d = json.loads(adj_def_path.read_text())
    by_case = defaultdict(dict)
    for j in d["judgments"]:
        by_case[j["case_id"]][j["judge"]] = j["triage"]
    n = len(by_case)
    unan = sum(1 for ts in by_case.values()
               if len(ts) == 2 and all(v == "DEFERRED" for v in ts.values()))
    either = sum(1 for ts in by_case.values()
                  if any(v == "DEFERRED" for v in ts.values()))
    gpt_def = sum(1 for ts in by_case.values() if ts.get(GPT) == "DEFERRED")
    cla_def = sum(1 for ts in by_case.values() if ts.get(CLA) == "DEFERRED")
    return {"n": n, "unanimous_deferred": unan, "either_deferred": either,
            "gpt_deferred": gpt_def, "claude_deferred": cla_def,
            "unanimous_pct": 100 * unan / n if n else 0,
            "either_pct": 100 * either / n if n else 0}


def sl_nl_accuracies(behav_path: Path) -> dict:
    """SL = A cell, NL = B cell (heuristic forced-letter parse, reliable)."""
    d = json.loads(behav_path.read_text())
    results = d["results"]
    n = len(results)
    sl_correct = sum(1 for r in results if r["A"]["correct"])
    nl_correct = sum(1 for r in results if r["B"]["correct"])
    return {
        "n": n,
        "SL_correct": sl_correct,
        "NL_correct": nl_correct,
        "SL_acc_pct": 100 * sl_correct / n,
        "NL_acc_pct": 100 * nl_correct / n,
    }


def main():
    out = {"models": {}}
    for tag, paths in MODELS.items():
        sl_nl = sl_nl_accuracies(paths["behavioral"])
        nf = both_judges_correct_acc(paths["nf_paper"])
        nf_def = deferred_rates(paths["nf_deferred"])
        sf = both_judges_correct_acc(paths["sf_paper"])
        sf_def = deferred_rates(paths["sf_deferred"])
        out["models"][tag] = {
            "n": sl_nl["n"],
            "SL_acc": sl_nl["SL_acc_pct"],
            "NL_acc": sl_nl["NL_acc_pct"],
            "NF_both_acc": nf["both_acc_pct"],
            "NF_either_acc": nf["either_acc_pct"],
            "NF_unan_deferred_pct": nf_def["unanimous_pct"],
            "NF_unan_deferred_n":   nf_def["unanimous_deferred"],
            "SF_both_acc": sf["both_acc_pct"],
            "SF_either_acc": sf["either_acc_pct"],
            "SF_unan_deferred_pct": sf_def["unanimous_pct"],
            "SF_unan_deferred_n":   sf_def["unanimous_deferred"],
        }

    (RESULTS / "sf_2x2_comparison.json").write_text(json.dumps(out, indent=2, default=str))

    md = ["# 2×2 design — full SL / NL / NF / SF comparison (4-way both-judges-correct headline)\n"]
    md.append("Completes the 2×2 input × output factorial. SL (structured + forced-letter) "
              "and NL (natural + forced-letter) accuracies are from heuristic letter parsing "
              "(reliable for forced-letter). NF and SF accuracies are from the paper-faithful "
              "4-way LLM-judge adjudicator (both judges agreeing on a gold-compatible letter); "
              "DEFERRED rates are from the 5-way adjudicator.\n")

    md.append("## Headline 2×2 (4-way both-judges-correct)\n")
    md.append("| | **Forced-Letter output** | **Free-Text output** | NL−NF gap | SL−SF gap |")
    md.append("|---|---|---|---|---|")
    for tag in ("4b", "12b", "qwen"):
        m = out["models"][tag]
        nl_nf = m["NL_acc"] - m["NF_both_acc"]
        sl_sf = m["SL_acc"] - m["SF_both_acc"]
        md.append(f"| **{tag} structured** | SL: {m['SL_acc']:.1f}% | "
                  f"SF: {m['SF_both_acc']:.1f}% | – | {sl_sf:+.1f} pp |")
        md.append(f"| **{tag} natural**    | NL: {m['NL_acc']:.1f}% | "
                  f"NF: {m['NF_both_acc']:.1f}% | {nl_nf:+.1f} pp | – |")
    md.append("")

    md.append("## Side-by-side comparison\n")
    md.append("| Model | SL | NL | NF (4-way both) | SF (4-way both) | NF unanim DEFER | SF unanim DEFER |")
    md.append("|---|---|---|---|---|---|---|")
    for tag in ("4b", "12b", "qwen"):
        m = out["models"][tag]
        md.append(f"| {tag} | {m['SL_acc']:.1f}% | {m['NL_acc']:.1f}% | "
                  f"{m['NF_both_acc']:.1f}% | {m['SF_both_acc']:.1f}% | "
                  f"{m['NF_unan_deferred_n']}/{m['n']} ({m['NF_unan_deferred_pct']:.1f}%) | "
                  f"{m['SF_unan_deferred_n']}/{m['n']} ({m['SF_unan_deferred_pct']:.1f}%) |")
    md.append("")

    md.append("## Headline read (auto-generated)\n")
    for tag in ("4b", "12b", "qwen"):
        m = out["models"][tag]
        # Cell ordering by accuracy
        cells = sorted([("SL", m['SL_acc']), ("NL", m['NL_acc']),
                        ("NF", m['NF_both_acc']), ("SF", m['SF_both_acc'])],
                       key=lambda x: -x[1])
        ranking = " > ".join(f"{n}({a:.0f}%)" for n, a in cells)
        nl_nf = m["NL_acc"] - m["NF_both_acc"]
        sl_sf = m["SL_acc"] - m["SF_both_acc"]
        md.append(f"- **{tag}**: cell ranking = {ranking}. "
                  f"NL−NF = {nl_nf:+.1f} pp, SL−SF = {sl_sf:+.1f} pp. "
                  f"DEFERRED rates: NF {m['NF_unan_deferred_n']}/{m['n']}, "
                  f"SF {m['SF_unan_deferred_n']}/{m['n']}.")
    md.append("")

    md.append("## Reading guide for §4.1 / §5 (the 2×2 interpretation)\n")
    md.append("Two key cross-cuts:")
    md.append("")
    md.append("**(a) Forced-letter vs Free-text within the same input type.** "
              "Does removing the forced-letter constraint help, hurt, or wash?")
    md.append("- NL → NF (natural input): tells us whether the canonical NL→NF gap "
              "(documented in §4.1) is driven by the output-format constraint.")
    md.append("- SL → SF (structured input): same question, structured input. If the "
              "gap goes the same direction in both rows, the format effect is robust "
              "across input style.")
    md.append("")
    md.append("**(b) Structured vs Natural within the same output type.** Does "
              "patient-voice vs clinician-notes-style affect accuracy?")
    md.append("- SL → NL: forced-letter only, isolated input effect.")
    md.append("- SF → NF: free-text only, isolated input effect.")
    md.append("")
    md.append("Together these decompose the variance: format effect (rows), input "
              "effect (columns), and their interaction (= the 4-corner residual).")


    (RESULTS / "sf_2x2_comparison.md").write_text("\n".join(md))
    print(f"Wrote {RESULTS/'sf_2x2_comparison.json'}")
    print(f"Wrote {RESULTS/'sf_2x2_comparison.md'}")
    print()
    print("=== Cross-model 2×2 ===")
    for tag in ("4b", "12b", "qwen"):
        m = out["models"][tag]
        print(f"  {tag:>4}  SL={m['SL_acc']:.1f}%  NL={m['NL_acc']:.1f}%  "
              f"NF={m['NF_both_acc']:.1f}%  SF={m['SF_both_acc']:.1f}%   "
              f"(NF DEF {m['NF_unan_deferred_n']}/{m['n']}, SF DEF {m['SF_unan_deferred_n']}/{m['n']})")


if __name__ == "__main__":
    main()
