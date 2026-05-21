"""paired_tests_and_confusion.py -- reviewer-requested supplementary
statistics for the §4.1 behavioral table.

Produces three things per model (4B, 12B, Qwen):

  (1) Paired NL-vs-NF test (McNemar exact + paired bootstrap 95% CI on
      the accuracy difference).
  (2) Per-acuity (gold A/B/C/D bucket) breakdown of SL / NL / NF accuracy
      and deferral rate.
  (3) Confusion matrices for SL / NL / NF (predicted vs gold letter), plus
      under-triage / over-triage rates with severity-weighted error rates.

Acuity ordering: A < B < C < D (lower index = lower urgency).

NF predicted letter under 4-way: the both-judges-agree letter, or "?" if
they disagree.

Outputs:
  results/paired_tests_and_confusion.json    (machine-readable)
  results/paired_tests_and_confusion.md      (paper-ready tables)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

MODELS = {
    "gemma-3-4b-it": {
        "behavioral":  RESULTS / "_v2/phase0_5_three_cells.json",
        "adj_4way":    RESULTS / "_v2/phase0_5_D_for_adjudication_adjudicated_paper.json",
        "adj_5way":    RESULTS / "_v2/phase0_5_adjudicated_deferred.json",
    },
    "gemma-3-12b-it": {
        "behavioral":  RESULTS / "_v2/phase3b_12b_phase0_5.json",
        "adj_4way":    RESULTS / "_v2/phase3b_12b_D_for_adjudication_adjudicated_paper.json",
        "adj_5way":    RESULTS / "_v2/phase3b_12b_adjudicated_deferred.json",
    },
    "qwen3-8b": {
        "behavioral":  RESULTS / "phase4b_qwen_behavioral.json",
        "adj_4way":    RESULTS / "phase4b_qwen_D_for_adjudication_adjudicated_paper.json",
        "adj_5way":    RESULTS / "phase4b_qwen_adjudicated_deferred.json",
    },
}

GPT = "gpt-5.2-thinking-high"
CLA = "claude-sonnet-4.6"
N_BOOT = 2000
BOOT_SEED = 7

def _b(x):
    if isinstance(x, bool): return x
    if isinstance(x, str):  return x.lower() == "true"
    return None

def gold_letters(g):
    return set(re.findall(r"[ABCD]", g.upper()))

def acuity_bucket(g):
    """Most-urgent gold letter (deterministic acuity bucket)."""
    letters = sorted(gold_letters(g), key=lambda L: "ABCD".index(L))
    return letters[-1] if letters else "?"

def acuity_index(letter):
    return "ABCD".index(letter) if letter in "ABCD" else None


# ─── McNemar exact + bootstrap ─────────────────────────────────────────────
def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar binomial p-value on discordant cells (b, c).

    Under H0, each discordant pair is equally likely to be in b or c.
    P-value = P(B(b+c, 0.5) ≤ min(b,c)) * 2, clipped at 1.
    """
    from math import comb
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    cdf = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return float(min(1.0, 2 * cdf))


def paired_bootstrap_ci(nl_correct, nf_correct, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired bootstrap 95% CI on (NL_acc - NF_acc)."""
    rng = np.random.default_rng(seed)
    arr_nl = np.asarray(nl_correct, dtype=float)
    arr_nf = np.asarray(nf_correct, dtype=float)
    n = len(arr_nl)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[k] = arr_nl[idx].mean() - arr_nf[idx].mean()
    return (float(np.percentile(diffs, 2.5)) * 100,
            float(np.percentile(diffs, 97.5)) * 100,
            float(diffs.mean()) * 100,
            float(diffs.std()) * 100)


# ─── Build per-case rollup ─────────────────────────────────────────────────
def build_cases(name, paths):
    beh = json.loads(paths["behavioral"].read_text())
    adj4 = json.loads(paths["adj_4way"].read_text())
    adj5 = json.loads(paths["adj_5way"].read_text())

    beh_by_id = {r["id"]: r for r in beh["results"]}
    adj4_by_id = {r["case_id"]: r for r in adj4}
    adj5_by_case = defaultdict(dict)
    for j in adj5["judgments"]:
        adj5_by_case[j["case_id"]][j["judge"]] = j

    cases = []
    for cid, r in beh_by_id.items():
        gold = r["gold_raw"]
        gset = gold_letters(gold)
        bucket = acuity_bucket(gold)

        # SL / NL: heuristic forced-letter parse (reliable — direct letter)
        sl_letter = r.get("A", {}).get("predicted")
        nl_letter = r.get("B", {}).get("predicted")
        sl_correct = bool(r.get("A", {}).get("correct", False))
        nl_correct = bool(r.get("B", {}).get("correct", False))

        # NF 4-way (judge labels)
        r4 = adj4_by_id.get(cid, {})
        gpt_t = r4.get("gpt_5_2_thinking_high_triage")
        cla_t = r4.get("claude_sonnet_4_6_triage")
        gpt_c = _b(r4.get("gpt_5_2_thinking_high_is_correct"))
        cla_c = _b(r4.get("claude_sonnet_4_6_is_correct"))
        nf_both = bool(gpt_c) and bool(cla_c)
        nf_either = bool(gpt_c) or bool(cla_c)
        # "Best single letter" for confusion matrix: the both-agree letter,
        # else None
        nf_letter = gpt_t if gpt_t == cla_t else None

        # NF 5-way DEFERRED
        g5 = adj5_by_case.get(cid, {}).get(GPT, {}).get("triage")
        c5 = adj5_by_case.get(cid, {}).get(CLA, {}).get("triage")
        both_def = (g5 == "DEFERRED" and c5 == "DEFERRED")

        cases.append({
            "case_id": cid,
            "gold_raw": gold,
            "gold_letters": "/".join(sorted(gset)),
            "acuity_bucket": bucket,
            "sl_letter": sl_letter, "sl_correct": sl_correct,
            "nl_letter": nl_letter, "nl_correct": nl_correct,
            "nf_letter_agree": nf_letter,
            "nf_gpt": gpt_t, "nf_claude": cla_t,
            "nf_both_correct": nf_both, "nf_either_correct": nf_either,
            "nf_unanim_deferred": both_def,
        })
    return cases


# ─── Confusion-matrix helpers ──────────────────────────────────────────────
def best_gold_letter_for_prediction(gset: set[str], pred: str | None) -> str:
    """When gold has multiple letters and the model picks one of them,
    that's the gold "intended" letter for the confusion-matrix row.
    If the model picks something outside the gold set, the "intended"
    gold letter is the most-urgent one (acuity-conservative).
    """
    if not gset: return "?"
    if pred in gset: return pred
    return sorted(gset, key=lambda L: "ABCD".index(L))[-1]


def triage_direction(gset: set[str], pred: str | None) -> str:
    """under_triage / over_triage / correct / unknown."""
    if not pred or pred == "?" or pred == "DEFERRED": return "no_commit"
    if pred in gset: return "correct"
    pred_i = acuity_index(pred)
    if pred_i is None: return "unknown"
    # Compare against the gold range
    gold_idxs = [acuity_index(g) for g in gset if acuity_index(g) is not None]
    if not gold_idxs: return "unknown"
    min_gi, max_gi = min(gold_idxs), max(gold_idxs)
    if pred_i < min_gi:    return "under_triage"
    if pred_i > max_gi:    return "over_triage"
    return "correct"  # should be caught above


def severity_weighted_error(cases, format_pred_key):
    """Number of acuity steps off from the closest gold letter, summed."""
    steps = []
    for c in cases:
        gset = gold_letters(c["gold_raw"])
        pred = c.get(format_pred_key)
        if not pred or pred in {"?", "DEFERRED"}:
            steps.append(None)
            continue
        if pred in gset:
            steps.append(0)
            continue
        pred_i = acuity_index(pred)
        if pred_i is None:
            steps.append(None); continue
        gold_idxs = [acuity_index(g) for g in gset if acuity_index(g) is not None]
        # closest gold letter in acuity space
        d = min(abs(pred_i - gi) for gi in gold_idxs)
        steps.append(d)
    arr = np.array([s for s in steps if s is not None], dtype=float)
    return {
        "n_committed": int((~np.isnan(arr)).sum()) if arr.size else 0,
        "n_no_commit": sum(1 for s in steps if s is None),
        "mean_steps_off": float(arr.mean()) if arr.size else None,
        "median_steps_off": float(np.median(arr)) if arr.size else None,
        "max_steps_off": float(arr.max()) if arr.size else None,
        "frac_correct": float((arr == 0).mean()) if arr.size else None,
        "frac_one_step_off": float((arr == 1).mean()) if arr.size else None,
        "frac_two_plus_off": float((arr >= 2).mean()) if arr.size else None,
    }


# ─── Main ──────────────────────────────────────────────────────────────────
def analyze_model(name, paths):
    cases = build_cases(name, paths)
    n = len(cases)

    # (1) Paired NL-vs-NF (4-way both-judges-correct)
    nl_correct = [c["nl_correct"] for c in cases]
    nf_correct = [c["nf_both_correct"] for c in cases]
    a = sum(1 for c in cases if c["nl_correct"] and c["nf_both_correct"])
    b = sum(1 for c in cases if c["nl_correct"] and not c["nf_both_correct"])
    cc = sum(1 for c in cases if not c["nl_correct"] and c["nf_both_correct"])
    d_ = sum(1 for c in cases if not c["nl_correct"] and not c["nf_both_correct"])
    p_mcn = mcnemar_exact_p(b, cc)
    ci_lo, ci_hi, boot_mean, boot_sd = paired_bootstrap_ci(nl_correct, nf_correct)

    paired_test = {
        "n": n,
        "NL_acc_pct": 100 * sum(nl_correct) / n,
        "NF_both_judges_acc_pct": 100 * sum(nf_correct) / n,
        "diff_NL_minus_NF_pp": 100 * (sum(nl_correct) - sum(nf_correct)) / n,
        "contingency": {"NL_right_NF_right": a, "NL_right_NF_wrong": b,
                        "NL_wrong_NF_right": cc, "NL_wrong_NF_wrong": d_},
        "mcnemar_exact_p_two_sided": p_mcn,
        "paired_bootstrap_95ci_NL_minus_NF_pp": [ci_lo, ci_hi],
        "paired_bootstrap_mean_pp": boot_mean,
        "paired_bootstrap_sd_pp":   boot_sd,
        "n_boot": N_BOOT,
    }

    # (2) Per-acuity (gold-bucket) breakdown
    acuity = defaultdict(lambda: {"n": 0, "sl_correct": 0, "nl_correct": 0,
                                   "nf_both_correct": 0, "nf_either_correct": 0,
                                   "nf_unanim_deferred": 0})
    for c in cases:
        b_ = c["acuity_bucket"]
        a_ = acuity[b_]
        a_["n"] += 1
        a_["sl_correct"]         += int(c["sl_correct"])
        a_["nl_correct"]         += int(c["nl_correct"])
        a_["nf_both_correct"]    += int(c["nf_both_correct"])
        a_["nf_either_correct"]  += int(c["nf_either_correct"])
        a_["nf_unanim_deferred"] += int(c["nf_unanim_deferred"])
    per_acuity = {k: dict(v) for k, v in acuity.items()}

    # (3) Confusion matrices + under/over-triage rates
    fmt_keys = [("SL", "sl_letter"), ("NL", "nl_letter"), ("NF", "nf_letter_agree")]
    confusion = {}
    triage_dir_summary = {}
    severity = {}
    for fmt_name, k in fmt_keys:
        cm = defaultdict(lambda: Counter())  # row=gold-bucket, col=pred
        dirs = Counter()
        for c in cases:
            gset = gold_letters(c["gold_raw"])
            pred = c.get(k) or "?"
            row = best_gold_letter_for_prediction(gset, pred)
            cm[row][pred] += 1
            dirs[triage_direction(gset, pred)] += 1
        confusion[fmt_name] = {gold: dict(cnts) for gold, cnts in cm.items()}
        triage_dir_summary[fmt_name] = dict(dirs)
        severity[fmt_name] = severity_weighted_error(cases, k)

    return {
        "model": name, "n_cases": n,
        "paired_NL_vs_NF": paired_test,
        "per_acuity": per_acuity,
        "confusion_matrix": confusion,
        "triage_direction_counts": triage_dir_summary,
        "severity_weighted_error": severity,
    }


def main():
    out = {}
    for name, paths in MODELS.items():
        if not all(p.exists() for p in paths.values()):
            print(f"  skip {name}: missing files")
            continue
        out[name] = analyze_model(name, paths)

    (RESULTS / "paired_tests_and_confusion.json").write_text(
        json.dumps(out, indent=2, default=str))

    # ─── Markdown ─────────────────────────────────────────────────────
    md = [
        "# Paired tests + per-acuity + confusion matrices\n",
        "Reviewer-asked supplementary statistics for §4.1 / §6.\n",
    ]

    md.append("## (1) Paired NL vs NF test (4-way both-judges-correct)\n")
    md.append("| Model | n | NL acc | NF acc | NL−NF (pp) | 95% CI | McNemar p | Discordant b:c |")
    md.append("|---|---|---|---|---|---|---|---|")
    for name, d in out.items():
        p = d["paired_NL_vs_NF"]
        ci = p["paired_bootstrap_95ci_NL_minus_NF_pp"]
        cont = p["contingency"]
        md.append(f"| {name} | {p['n']} | {p['NL_acc_pct']:.1f}% | "
                  f"{p['NF_both_judges_acc_pct']:.1f}% | "
                  f"{p['diff_NL_minus_NF_pp']:+.1f} | "
                  f"[{ci[0]:+.1f}, {ci[1]:+.1f}] | "
                  f"{p['mcnemar_exact_p_two_sided']:.4f} | "
                  f"{cont['NL_right_NF_wrong']}:{cont['NL_wrong_NF_right']} |")
    md.append("")
    md.append("Interpretation:")
    md.append("- McNemar exact two-sided p-value on the discordant cells (NL right & NF wrong vs NL wrong & NF right). 60-case sample → discordant pairs are small but tests are well-defined.")
    md.append("- 95% CI is the paired bootstrap (2000 resamples) on the per-case accuracy difference.")
    md.append("")

    md.append("## (2) Per-acuity (most-urgent gold letter) breakdown\n")
    for name, d in out.items():
        md.append(f"### {name}")
        md.append("| Gold acuity | n | SL | NL | NF (both) | NF (either) | DEFERRED |")
        md.append("|---|---|---|---|---|---|---|")
        for letter in "ABCD":
            a_ = d["per_acuity"].get(letter)
            if not a_: continue
            nn = a_["n"]
            md.append(f"| {letter} | {nn} | "
                      f"{100*a_['sl_correct']/nn:.0f}% | "
                      f"{100*a_['nl_correct']/nn:.0f}% | "
                      f"{100*a_['nf_both_correct']/nn:.0f}% | "
                      f"{100*a_['nf_either_correct']/nn:.0f}% | "
                      f"{100*a_['nf_unanim_deferred']/nn:.0f}% |")
        md.append("")

    md.append("## (3) Confusion matrices (rows = gold acuity bucket, columns = predicted letter)\n")
    for name, d in out.items():
        md.append(f"### {name}")
        for fmt in ("SL", "NL", "NF"):
            md.append(f"#### {fmt}")
            md.append("| gold ↓ \\ pred → | A | B | C | D | ? | DEFERRED |")
            md.append("|---|---|---|---|---|---|---|")
            cm = d["confusion_matrix"].get(fmt, {})
            for letter in "ABCD":
                row = cm.get(letter, {})
                md.append(f"| {letter} | {row.get('A',0)} | {row.get('B',0)} | "
                          f"{row.get('C',0)} | {row.get('D',0)} | "
                          f"{row.get('?',0) + sum(v for k,v in row.items() if k not in 'ABCD?' and k != 'DEFERRED')} | "
                          f"{row.get('DEFERRED',0)} |")
            md.append("")

    md.append("## (4) Triage direction + severity-weighted error\n")
    md.append("| Model | Format | n correct | n under-triage | n over-triage | n no-commit | mean steps off (committed) | frac ≥2 steps off |")
    md.append("|---|---|---|---|---|---|---|---|")
    for name, d in out.items():
        for fmt in ("SL", "NL", "NF"):
            tc = d["triage_direction_counts"].get(fmt, {})
            sw = d["severity_weighted_error"].get(fmt, {})
            mso = sw.get("mean_steps_off")
            fpo = sw.get("frac_two_plus_off")
            mso_str = f"{mso:.2f}" if mso is not None else "–"
            fpo_str = f"{fpo:.2f}" if fpo is not None else "–"
            md.append(f"| {name} | {fmt} | "
                      f"{tc.get('correct',0)} | {tc.get('under_triage',0)} | "
                      f"{tc.get('over_triage',0)} | {tc.get('no_commit',0)} | "
                      f"{mso_str} | {fpo_str} |")
    md.append("")
    md.append("Under-triage = predicted letter has lower acuity than the lowest gold letter; over-triage = predicted is higher than the highest gold letter; no-commit = judges disagreed (NF) or letter unparseable.")

    (RESULTS / "paired_tests_and_confusion.md").write_text("\n".join(md))
    print(f"Wrote {RESULTS/'paired_tests_and_confusion.json'}")
    print(f"Wrote {RESULTS/'paired_tests_and_confusion.md'}")
    print()
    for name, d in out.items():
        p = d["paired_NL_vs_NF"]
        ci = p["paired_bootstrap_95ci_NL_minus_NF_pp"]
        print(f"--- {name} ---")
        print(f"  NL = {p['NL_acc_pct']:.1f}%, NF (both) = {p['NF_both_judges_acc_pct']:.1f}%, "
              f"gap = {p['diff_NL_minus_NF_pp']:+.1f} pp")
        print(f"  95% paired-boot CI on (NL - NF) = [{ci[0]:+.1f}, {ci[1]:+.1f}] pp")
        print(f"  McNemar exact two-sided p = {p['mcnemar_exact_p_two_sided']:.4f}")
        print(f"  Triage dirs (NL):  {d['triage_direction_counts'].get('NL', {})}")
        print(f"  Triage dirs (NF):  {d['triage_direction_counts'].get('NF', {})}")
        print()


if __name__ == "__main__":
    main()
