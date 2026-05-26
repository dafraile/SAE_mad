"""mcnemar_sf_sl.py -- paired McNemar exact tests on the SF-SL accuracy gap.

Mirrors the NF-NL McNemar test reported in §4.1. For each model:
  - SL accuracy from heuristic forced-letter parse (reliable)
  - SF accuracy from 4-way LLM-judge both-judges-correct adjudication
  - 2x2 contingency on per-case (SL_correct, SF_correct)
  - McNemar's exact two-sided p-value from Binom(b+c, 0.5)
  - Paired bootstrap 95% CI on SF-SL (2000 resamples)
"""
import json
from pathlib import Path
from math import comb

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def _b(x):
    if isinstance(x, bool): return x
    if isinstance(x, str):  return x.lower() == "true"
    return None


def both_judges_correct(row):
    g = _b(row.get("gpt_5_2_thinking_high_is_correct"))
    c = _b(row.get("claude_sonnet_4_6_is_correct"))
    return bool(g) and bool(c)


def mcnemar_exact_two_sided(b, c):
    """Two-sided exact p-value for McNemar from Binom(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    p_one = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * p_one)


def paired_bootstrap_ci(sl, sf, n_boot=2000, seed=0):
    """95% CI on mean(SF - SL) across cases."""
    diff = (np.asarray(sf, dtype=int) - np.asarray(sl, dtype=int))
    n = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = diff[idx].mean(axis=1) * 100  # pp
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


MODELS = {
    "4b": {
        "name": "Gemma 3 4B IT",
        "behavioral": RESULTS / "_v2/phase0_5_three_cells.json",
        "sf_paper":   RESULTS / "sf_4b_D_for_adjudication_adjudicated_paper.json",
    },
    "12b": {
        "name": "Gemma 3 12B IT",
        "behavioral": RESULTS / "_v2/phase3b_12b_phase0_5.json",
        "sf_paper":   RESULTS / "sf_12b_D_for_adjudication_adjudicated_paper.json",
    },
    "qwen": {
        "name": "Qwen3-8B",
        "behavioral": RESULTS / "phase4b_qwen_behavioral.json",
        "sf_paper":   RESULTS / "sf_qwen_D_for_adjudication_adjudicated_paper.json",
    },
}


def main():
    rows = []
    print(f"{'Model':<18}{'n':>4}{'SL%':>8}{'SF%':>8}{'SF-SL':>9}  contingency        McNemar p  95% CI (paired)")
    print("-" * 95)
    for tag, cfg in MODELS.items():
        beh = json.loads(cfg["behavioral"].read_text())
        sf_adj = {r["case_id"]: r for r in json.loads(cfg["sf_paper"].read_text())}

        sl_correct = []
        sf_correct = []
        for r in beh["results"]:
            cid = r["id"]
            sl = bool(r["A"]["correct"])
            sf_row = sf_adj.get(cid)
            if sf_row is None:
                continue
            sf = both_judges_correct(sf_row)
            sl_correct.append(sl)
            sf_correct.append(sf)

        n = len(sl_correct)
        sl_acc = 100 * sum(sl_correct) / n
        sf_acc = 100 * sum(sf_correct) / n
        diff_pp = sf_acc - sl_acc

        a = sum(1 for x, y in zip(sl_correct, sf_correct) if x and y)
        b = sum(1 for x, y in zip(sl_correct, sf_correct) if x and not y)  # SL right, SF wrong
        c = sum(1 for x, y in zip(sl_correct, sf_correct) if not x and y)  # SL wrong, SF right
        d = sum(1 for x, y in zip(sl_correct, sf_correct) if not x and not y)
        p = mcnemar_exact_two_sided(b, c)
        ci_lo, ci_hi = paired_bootstrap_ci(sl_correct, sf_correct)

        rows.append({
            "model": cfg["name"], "tag": tag, "n": n,
            "SL_acc_pct": sl_acc, "SF_acc_pct": sf_acc,
            "diff_SF_minus_SL_pp": diff_pp,
            "contingency": {"SL_right_SF_right": a, "SL_right_SF_wrong": b,
                            "SL_wrong_SF_right": c, "SL_wrong_SF_wrong": d},
            "n_discordant": b + c, "min_bc": min(b, c),
            "mcnemar_exact_p_two_sided": p,
            "paired_bootstrap_95ci_pp": [ci_lo, ci_hi],
        })
        cont_str = f"a={a},b={b},c={c},d={d}"
        print(f"{cfg['name']:<18}{n:>4}{sl_acc:>7.1f}%{sf_acc:>7.1f}%{diff_pp:>+8.1f}pp  {cont_str:<18} {p:>9.4f}  [{ci_lo:+6.1f}, {ci_hi:+6.1f}]pp")

    out_json = RESULTS / "mcnemar_sf_sl.json"
    out_json.write_text(json.dumps(rows, indent=2, default=str))

    md = [
        "# McNemar paired test on SF − SL gap (per model)\n",
        "Parallel to the NF−NL McNemar reported in §4.1. SL correctness is the heuristic forced-letter parse on the structured prompt. SF correctness is the both-judges-correct verdict under 4-way LLM-judge adjudication on the free-text response to the same structured input.\n",
        "## Headline\n",
        "| Model | n | SL | SF | SF−SL | b (SL✓ SF✗) | c (SL✗ SF✓) | n_discordant | McNemar exact p (two-sided) | 95% CI (paired, pp) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ct = r["contingency"]
        md.append(f"| {r['model']} | {r['n']} | {r['SL_acc_pct']:.1f}% | {r['SF_acc_pct']:.1f}% | "
                  f"{r['diff_SF_minus_SL_pp']:+.1f}pp | {ct['SL_right_SF_wrong']} | "
                  f"{ct['SL_wrong_SF_right']} | {r['n_discordant']} | "
                  f"**{r['mcnemar_exact_p_two_sided']:.4f}** | "
                  f"[{r['paired_bootstrap_95ci_pp'][0]:+.1f}, {r['paired_bootstrap_95ci_pp'][1]:+.1f}] |")
    md.append("")
    md.append("## Drop-in §4.1 sentence (extending the existing NF−NL sentence)\n")
    md.append("> \"...$p{=}0.031$ at both Gemma scales and $p{=}0.45$ at Qwen ($n.s.$ at $n{=}60$). The same test on the SF$-$SL gap gives "
              + ", ".join([
                  f"$p{{=}}{r['mcnemar_exact_p_two_sided']:.3f}$ at {r['model'].replace('Gemma 3 ','').replace(' IT','')}"
                  for r in rows
              ]) + ".\"")
    (RESULTS / "mcnemar_sf_sl.md").write_text("\n".join(md))
    print(f"\nWrote {out_json}")
    print(f"Wrote {RESULTS/'mcnemar_sf_sl.md'}")


if __name__ == "__main__":
    main()
