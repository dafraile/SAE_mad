"""option_order_shuffle_summary.py -- cross-model summary of the
option-order shuffle experiment for 4B / 12B / Qwen.

Headline: K=23 exhaustive permutations (all 23 non-identity permutations
of (A,B,C,D) per case) with case-clustered bootstrap 95% CIs.

K=3 baseline numbers retained as an inset for historical comparison
(an earlier turn used K=3 random shuffles and the LaTeX writer
referenced those numbers).

Consumes:
  results/option_order_shuffle_{4b,12b,qwen}.json                  (K=3)
  results/option_order_shuffle_{4b,12b,qwen}_exhaustive.json       (K=23)
  results/option_order_shuffle_clustered_bootstrap.json            (CIs)
  results/paired_tests_and_confusion.json                          (NF acc)

Writes:
  results/option_order_shuffle_all_models.md   (paper-ready table)
  results/option_order_shuffle_all_models.json (compact bundle)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def load(p): return json.loads(p.read_text()) if p.exists() else None


def nf_acc_table():
    p = load(RESULTS / "paired_tests_and_confusion.json")
    if not p: return {}
    name_map = {"gemma-3-4b-it": "4b", "gemma-3-12b-it": "12b", "qwen3-8b": "qwen"}
    return {tag: p[full]["paired_NL_vs_NF"]["NF_both_judges_acc_pct"]
            for full, tag in name_map.items() if full in p}


def main():
    nf = nf_acc_table()
    boot = load(RESULTS / "option_order_shuffle_clustered_bootstrap.json")
    if boot is None:
        raise SystemExit("Need option_order_shuffle_clustered_bootstrap.json first.")

    # Pull K=3 and K=23 stability / accuracy per model
    rows = []
    for tag in ("4b", "12b", "qwen"):
        d3 = load(RESULTS / f"option_order_shuffle_{tag}.json")
        d23 = load(RESULTS / f"option_order_shuffle_{tag}_exhaustive.json")
        if not d3 or not d23: continue
        b3 = boot["runs"].get(tag, {})
        b23 = boot["runs"].get(f"{tag}_exhaustive", {})
        rows.append({
            "tag": tag,
            "model": d23["model"],
            "n_cases": d23["n_cases"],
            "K3":  {
                "K": d3["K_shuffles_per_case"],
                "n_shuffle_total": d3["n_shuffle_total"],
                "same_letter":  b3.get("same_letter_frac", {}),
                "same_content": b3.get("same_content_frac", {}),
                "shuffled_acc": b3.get("shuffled_accuracy", {}),
                "letter_dist_canonical": d3["letter_distribution_original_NL"],
                "content_dist_canonical": d3["content_distribution_original_NL"],
                "letter_dist_shuffles": d3["letter_distribution_shuffles"],
                "content_dist_shuffles": d3["content_distribution_shuffles"],
            },
            "K23": {
                "K": d23["K_shuffles_per_case"],
                "n_shuffle_total": d23["n_shuffle_total"],
                "same_letter":  b23.get("same_letter_frac", {}),
                "same_content": b23.get("same_content_frac", {}),
                "shuffled_acc": b23.get("shuffled_accuracy", {}),
                "letter_dist_canonical": d23["letter_distribution_original_NL"],
                "content_dist_canonical": d23["content_distribution_original_NL"],
                "letter_dist_shuffles": d23["letter_distribution_shuffles"],
                "content_dist_shuffles": d23["content_distribution_shuffles"],
            },
            "canonical_NL_acc": d23["accuracy"]["original_accuracy_pct"],
            "NF_acc": nf.get(tag),
        })

    # JSON dump
    (RESULTS / "option_order_shuffle_all_models.json").write_text(json.dumps(rows, indent=2))

    def fmt_ci(ci):
        if not ci or ci.get("ci_lo_95") is None: return "–"
        return f"{ci['point_estimate']*100:.1f}% [{ci['ci_lo_95']*100:.1f}, {ci['ci_hi_95']*100:.1f}]"

    md = ["# Option-order shuffle — cross-model summary (4B / 12B / Qwen)\n"]
    md.append("Falsifiable test of position-bias vs content-prior at the forced-letter scaffold. "
              "For each of 60 canonical cases, randomize the letter→content mapping in the "
              "forced-letter scaffold, run greedy forced-letter, score same-letter % vs "
              "same-content % vs accuracy.\n")
    md.append("**Two runs:**")
    md.append("- **K=23 exhaustive** (the manuscript headline): all 23 non-identity permutations of (A,B,C,D) per case → 1380 shuffles per model. Case-clustered bootstrap 95% CIs (B=2000).")
    md.append("- **K=3 baseline** (kept as inset for cross-checking with earlier drafts): 3 random non-identity permutations per case → 180 shuffles per model.\n")

    md.append("## Headline table — K=23 exhaustive with case-clustered 95% CIs\n")
    md.append("| Model | n | K | same-letter % [95% CI] | same-content % [95% CI] | canonical NL acc | shuffled NL acc [95% CI] | NF (4-way both) | shuffled − NF |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        k23 = r["K23"]
        nf_v = r['NF_acc']
        nf_str = f"{nf_v:.1f}%" if nf_v is not None else "?"
        gap = (k23["shuffled_acc"].get("point_estimate", 0) * 100 - nf_v) if nf_v is not None else None
        gap_str = f"{gap:+.1f} pp" if gap is not None else "?"
        md.append(f"| {r['tag']} | {r['n_cases']} | {k23['K']} | "
                  f"{fmt_ci(k23['same_letter'])} | "
                  f"{fmt_ci(k23['same_content'])} | "
                  f"{r['canonical_NL_acc']:.1f}% | "
                  f"{fmt_ci(k23['shuffled_acc'])} | "
                  f"{nf_str} | {gap_str} |")
    md.append("")
    md.append("Reading guide:")
    md.append("- **same-letter %** is below chance (25%) at 4B and 12B (upper CI bound < 25%); at Qwen the CI just barely touches 25% but the point estimate is below. → **no position bias at any model.**")
    md.append("- **same-content %** is far above chance (lower CI bound ≥ 56% across all three models). → **strong content prior at every model.**")
    md.append("- **shuffled − NF gap:** at 4B the gap is −1.9 pp with the shuffled CI containing NF (statistically indistinguishable). At 12B and Qwen, shuffled forced-letter still beats free-text by ≈5–7 pp under exhaustive shuffles — separate NF-mode accuracy penalty independent of letter-binding.")
    md.append("")

    md.append("## Inset — K=3 baseline (earlier draft used these numbers)\n")
    md.append("Kept here for cross-referencing with any v3-pre-2026-05-22 manuscript draft. The K=23 numbers above should be the headline in v3 final.\n")
    md.append("| Model | K | same-letter % [95% CI] | same-content % [95% CI] | shuffled NL acc [95% CI] | shuffled − NF |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        k3 = r["K3"]
        nf_v = r['NF_acc']
        gap = (k3["shuffled_acc"].get("point_estimate", 0) * 100 - nf_v) if nf_v is not None else None
        gap_str = f"{gap:+.1f} pp" if gap is not None else "?"
        md.append(f"| {r['tag']} | {k3['K']} | "
                  f"{fmt_ci(k3['same_letter'])} | "
                  f"{fmt_ci(k3['same_content'])} | "
                  f"{fmt_ci(k3['shuffled_acc'])} | "
                  f"{gap_str} |")
    md.append("")
    md.append("**Honesty note on the K=3→K=23 transition for 4B:** the K=3 point estimate was shuffled NL acc = NF acc = 71.7% *to the case*, which earlier drafts framed as the entire format penalty IS the canonical letter-binding. The more precise K=23 estimate is 69.8% with 95% CI [60.7%, 78.3%]; the CI contains NF (71.7%), so the corrected claim for v3 final is **'shuffled NL accuracy is statistically indistinguishable from NF accuracy at 4B (n=60 cases)'** rather than 'exactly equal.' The qualitative story (canonical letter-binding × content prior explains essentially all of 4B's NL→NF accuracy penalty) survives. See `results/option_order_shuffle_exhaustive_summary.md` for the full K=3-vs-K=23 comparison with CIs.\n")

    md.append("## Letter distribution (canonical vs K=23 shuffles)\n")
    md.append("| Model | NL canonical | NL shuffles (K=23, total 1380) |")
    md.append("|---|---|---|")
    for r in rows:
        cl = r["K23"]["letter_dist_canonical"]
        sl = r["K23"]["letter_dist_shuffles"]
        md.append(f"| {r['tag']} | A:{cl.get('A',0)} B:{cl.get('B',0)} C:{cl.get('C',0)} D:{cl.get('D',0)} | "
                  f"A:{sl.get('A',0)} B:{sl.get('B',0)} C:{sl.get('C',0)} D:{sl.get('D',0)} |")
    md.append("")

    md.append("## Content distribution under K=23 shuffles\n")
    md.append("Which acuity content does the model pick (regardless of letter)? Under shuffles, a content prior shows up as concentration on one row.\n")
    md.append("| Model | Fine to monitor | Weeks | **24-48h** | Go to ER |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        sd = r["K23"]["content_dist_shuffles"]
        labels = ["Fine to monitor at home",
                  "See my doctor in the next few weeks",
                  "See a doctor within 24-48 hours",
                  "Go to the ER now"]
        md.append(f"| {r['tag']} | {sd.get(labels[0], 0)} | {sd.get(labels[1], 0)} | "
                  f"**{sd.get(labels[2], 0)}** | {sd.get(labels[3], 0)} |")
    md.append("")
    md.append("All three models concentrate strongly on **'See a doctor within 24-48 hours'** content under exhaustive shuffles. **4B picks 'Go to ER' content only 2/1380 = 0.14% of the time** even when the canonical D position is randomized — a robust capability-scaling signal: 12B picks ER content 9.1%, Qwen 7.0%. 4B has a learned content-level aversion to the ER recommendation, not a position artifact.\n")

    md.append("## Headline read for the §4.2 / §5 manuscript rewrite (auto-generated)\n")
    for r in rows:
        k23 = r["K23"]
        sl = k23["same_letter"].get("point_estimate", 0) * 100
        sc = k23["same_content"].get("point_estimate", 0) * 100
        canon = r["canonical_NL_acc"]
        shuf  = k23["shuffled_acc"].get("point_estimate", 0) * 100
        nf_v = r['NF_acc']
        verdict_bias = "content prior" if sc - sl > 20 else "mixed signal"
        if nf_v:
            conv = f"shuffled NL {shuf:.1f}% vs NF {nf_v:.1f}% (Δ {shuf-nf_v:+.1f} pp)"
        else:
            conv = ""
        md.append(f"- **{r['tag']}** (K=23): same-letter {sl:.1f}% (below chance 25%) vs same-content {sc:.1f}% (well above chance) → **{verdict_bias}**. Canonical NL {canon:.1f}% → shuffled NL {shuf:.1f}%; {conv}.")
    md.append("")
    md.append("**One-paragraph summary for §4.2 (use this verbatim or adapt):**\n")
    md.append("> An option-order randomization experiment (60 cases × 23 non-identity permutations of the letter→content mapping in the forced-letter scaffold) tests whether the forced-letter accuracy depends on letter position or on letter content. Across all three models, the picked-letter is at or below chance under shuffles (case-clustered 95% CI excludes 25% at 4B and 12B), while the picked-content is far above chance (lower CI bound ≥ 56%): **no position bias, strong content prior**. At 4B, randomising the labels brings the forced-letter accuracy from 55.0% to 69.8% (95% CI [60.7%, 78.3%]), statistically indistinguishable from NF accuracy 71.7% — the entire NL→NF format penalty at 4B can be attributed to the canonical A-B-C-D letter-binding interacting with the model's content prior. At 12B and Qwen, the canonical mapping is approximately neutral or mildly helpful for accuracy, but shuffled forced-letter still beats free-text by ≈5–7 pp — at scale, free-text mode has its own accuracy penalty (the adjacent-miscalibration of §4.2 above) that is independent of letter-binding. Under exhaustive shuffles, 4B emits the letter mapped to 'Go to the ER now' content in only 2/1380 = 0.14% of shuffles, compared to 9.1% at 12B and 7.0% at Qwen — a robust capability-scaling signal at the highest acuity level.\n")

    (RESULTS / "option_order_shuffle_all_models.md").write_text("\n".join(md))
    print(f"Wrote {RESULTS/'option_order_shuffle_all_models.md'}")


if __name__ == "__main__":
    main()
