"""option_order_shuffle_summary.py -- cross-model summary of the
option-order shuffle experiment for 4B / 12B / Qwen.

Consumes results/option_order_shuffle_{4b,12b,qwen}.json and writes:
  results/option_order_shuffle_all_models.md  (paper-ready table)
  results/option_order_shuffle_all_models.json (compact bundle)

Comparison columns (matching the agent's request):
  - same-letter %        (chance ≈25%; high = position bias)
  - same-content %       (chance ≈25%; high = content prior)
  - canonical NL accuracy (from existing behavioral data)
  - shuffled NL accuracy
  - canonical letter distribution
  - shuffled content distribution
  - convergence to NF: shuffled accuracy vs NF (4-way both-judges) accuracy

The NF accuracy reference numbers come from
results/paired_tests_and_confusion.json (which already has the
4-way both-judges-correct NF accuracy for each model).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def load(model_tag):
    p = RESULTS / f"option_order_shuffle_{model_tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_nf_accuracy():
    """Pull NF (4-way both-judges) accuracy per model from the paired tests file."""
    p = RESULTS / "paired_tests_and_confusion.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    nf_acc = {}
    name_map = {"gemma-3-4b-it": "4b", "gemma-3-12b-it": "12b", "qwen3-8b": "qwen"}
    for full_name, tag in name_map.items():
        if full_name in d:
            nf_acc[tag] = d[full_name]["paired_NL_vs_NF"]["NF_both_judges_acc_pct"]
    return nf_acc


def main():
    nf_acc = load_nf_accuracy()
    rows = []
    for tag in ("4b", "12b", "qwen"):
        d = load(tag)
        if d is None:
            print(f"skip {tag}: results missing")
            continue
        stab = d["stability_signals"]
        acc = d["accuracy"]
        rows.append({
            "tag": tag,
            "model": d["model"],
            "n_cases": d["n_cases"],
            "K_shuffles": d["K_shuffles_per_case"],
            "n_shuffle_total": d["n_shuffle_total"],
            "same_letter_pct":  100 * stab["same_letter_frac"],
            "same_content_pct": 100 * stab["same_content_frac"],
            "canonical_NL_acc_pct": acc["original_accuracy_pct"],
            "shuffled_NL_acc_pct": acc["shuffled_accuracy_pct"],
            "NF_4way_both_judges_acc_pct": nf_acc.get(tag),
            "letter_dist_canonical": d["letter_distribution_original_NL"],
            "letter_dist_shuffles":  d["letter_distribution_shuffles"],
            "content_dist_canonical": d["content_distribution_original_NL"],
            "content_dist_shuffles":  d["content_distribution_shuffles"],
        })

    out_json = RESULTS / "option_order_shuffle_all_models.json"
    out_json.write_text(json.dumps(rows, indent=2))

    # ─── Markdown summary ────────────────────────────────────────────
    md = ["# Option-order shuffle — cross-model summary (4B / 12B / Qwen)\n"]
    md.append("Falsifiable test of position-bias vs content-prior at the forced-letter scaffold. For each of the 60 canonical cases, K=3 random non-identity permutations of the letter→content mapping; greedy forced-letter generation; score same-letter, same-content, accuracy under shuffle.\n")
    md.append("## Stability + accuracy\n")
    md.append("| Model | n | K | same-letter % (chance ≈25%) | same-content % (chance ≈25%) | canonical NL acc | shuffled NL acc | NF (4-way both) | shuffled→NF gap |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        nf = r['NF_4way_both_judges_acc_pct']
        gap = (r['shuffled_NL_acc_pct'] - nf) if nf is not None else None
        nf_str = f"{nf:.1f}%" if nf is not None else "?"
        gap_str = f"{gap:+.1f} pp" if gap is not None else "?"
        md.append(f"| {r['tag']} | {r['n_cases']} | {r['K_shuffles']} | "
                  f"{r['same_letter_pct']:.1f}% | "
                  f"{r['same_content_pct']:.1f}% | "
                  f"{r['canonical_NL_acc_pct']:.1f}% | "
                  f"{r['shuffled_NL_acc_pct']:.1f}% | "
                  f"{nf_str} | {gap_str} |")
    md.append("")
    md.append("Interpretation: high same-letter % → position bias; high same-content % → content prior. The shuffled-NL-vs-NF gap tells us whether option-order randomization 'erases' the forced-letter mode's letter-binding artifact (gap ≈ 0 pp) or only partially (gap > 0).")
    md.append("")
    md.append("## Letter distribution (canonical NL vs shuffled NL)\n")
    md.append("| Model | NL canonical | NL shuffles |")
    md.append("|---|---|---|")
    for r in rows:
        cl = r["letter_dist_canonical"]
        sl = r["letter_dist_shuffles"]
        md.append(f"| {r['tag']} | A:{cl.get('A',0)} B:{cl.get('B',0)} C:{cl.get('C',0)} D:{cl.get('D',0)} | "
                  f"A:{sl.get('A',0)} B:{sl.get('B',0)} C:{sl.get('C',0)} D:{sl.get('D',0)} |")
    md.append("")
    md.append("## Content distribution (canonical NL vs shuffled NL)\n")
    md.append("Shows which acuity content the model picks (regardless of which letter that content is assigned to). Under shuffles, a content prior shows up here as concentration on one row.\n")
    md.append("| Model | Canonical: Fine / Weeks / 24-48h / ER | Shuffles: Fine / Weeks / 24-48h / ER |")
    md.append("|---|---|---|")
    for r in rows:
        cd = r["content_dist_canonical"]
        sd = r["content_dist_shuffles"]
        # keys are content texts; canonical preferred letter dist
        labels = ["Fine to monitor at home",
                  "See my doctor in the next few weeks",
                  "See a doctor within 24-48 hours",
                  "Go to the ER now"]
        cd_str = " / ".join(str(cd.get(L, 0)) for L in labels)
        sd_str = " / ".join(str(sd.get(L, 0)) for L in labels)
        md.append(f"| {r['tag']} | {cd_str} | {sd_str} |")
    md.append("")
    md.append("## Headline read (auto-generated)\n")
    for r in rows:
        sl = r['same_letter_pct']
        sc = r['same_content_pct']
        canon = r['canonical_NL_acc_pct']
        shuf = r['shuffled_NL_acc_pct']
        nf = r['NF_4way_both_judges_acc_pct']
        verdict_bias = ("content prior" if sc - sl > 20 else
                       ("position bias" if sl - sc > 20 else "mixed / weak signal"))
        delta = shuf - canon
        delta_dir = "↑" if delta > 0 else ("↓" if delta < 0 else "≈")
        conv_str = (f"shuffled NL {shuf:.1f}% vs NF {nf:.1f}% (Δ {shuf-nf:+.1f} pp)" if nf else "")
        md.append(f"- **{r['tag']}**: same-letter {sl:.1f}% vs same-content {sc:.1f}% → **{verdict_bias}**. "
                  f"Shuffled NL acc {delta_dir} {abs(delta):.1f} pp vs canonical NL "
                  f"({canon:.1f}% → {shuf:.1f}%); {conv_str}.")
    md.append("")

    out_md = RESULTS / "option_order_shuffle_all_models.md"
    out_md.write_text("\n".join(md))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print()
    for r in rows:
        print(f"--- {r['tag']} ---")
        print(f"  same-letter:  {r['same_letter_pct']:.1f}%")
        print(f"  same-content: {r['same_content_pct']:.1f}%")
        print(f"  canonical NL acc: {r['canonical_NL_acc_pct']:.1f}%")
        print(f"  shuffled NL acc:  {r['shuffled_NL_acc_pct']:.1f}%")
        print(f"  NF 4-way both:    {r['NF_4way_both_judges_acc_pct']}")
        print()


if __name__ == "__main__":
    main()
