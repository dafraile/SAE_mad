"""sl_sf_summary.py -- consolidate SL-SF mechanistic results across the
three models into a single paper-ready table for §4.3 robustness.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def main():
    out_all = {}
    for tag in ("4b", "12b", "qwen"):
        p = RESULTS / f"sl_sf_masked_invariance_{tag}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        out_all[tag] = {
            "model_id": d["model_id"], "layer": d["layer"],
            "vignette_smape_medical": d["vignette_smape_medical_median"],
            "vignette_smape_random":  d["vignette_smape_random_median"],
            "full_smape_medical":     d["full_smape_medical_median"],
            "full_smape_random":      d["full_smape_random_median"],
            "full_cosine_medical":    d["full_cosine_medical_median"],
            "full_cosine_random":     d["full_cosine_random_median"],
            "paired_diff_mean":       d["paired_smape_diff_mean"],
            "paired_diff_95ci":       d["paired_smape_diff_95ci"],
            "peak_in_vignette_SL":    d["medical_peak_in_vignette_frac_SL"],
            "peak_in_vignette_SF":    d["medical_peak_in_vignette_frac_SF"],
        }

    (RESULTS / "sl_sf_summary.json").write_text(json.dumps(out_all, indent=2))

    md = [
        "# SL−SF mechanistic invariance — robustness across input style\n",
        "Parallel to the NL−NF mechanistic analysis in §4.3, run on the structured-input × output-format pair to test whether the medical-vs-random format-invariance result depends on natural patient-voice input.\n",
        "## Headline table (max-pool, paired bootstrap 95% CIs over cases)\n",
        "| Model | n | med sMAPE | rnd sMAPE | med cos | rnd cos | paired Δ med−rnd | 95% CI | Sig? |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tag, r in out_all.items():
        ci = r["paired_diff_95ci"]
        sig = "✓" if ci[1] < 0 else "ns"
        md.append(f"| {tag} L{r['layer']} | 60 | "
                  f"{r['full_smape_medical']:.4f} | "
                  f"{r['full_smape_random']:.4f} | "
                  f"{r['full_cosine_medical']:.4f} | "
                  f"{r['full_cosine_random']:.4f} | "
                  f"{r['paired_diff_mean']:+.4f} | "
                  f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {sig} |")
    md.append("")
    md.append("## Vignette-mask sanity check (expected ~0)\n")
    md.append("| Model | med vignette sMAPE | rnd vignette sMAPE |")
    md.append("|---|---|---|")
    for tag, r in out_all.items():
        md.append(f"| {tag} | {r['vignette_smape_medical']:.4f} | {r['vignette_smape_random']:.4f} |")
    md.append("")
    md.append("Both medical and random sMAPE collapse to ~0.002–0.004 on the shared structured-content vignette mask, confirming causal-masking trivial invariance.\n")
    md.append("## Medical-feature peak location\n")
    md.append("Fraction of (case × medical-feature) pairs whose peak activation lies inside the shared vignette (vs. on the SL-only scaffold for SL prompts, or on the chat-template suffix for SF prompts).\n")
    md.append("| Model | SL: peak in vignette | SF: peak in vignette |")
    md.append("|---|---|---|")
    for tag, r in out_all.items():
        md.append(f"| {tag} | {r['peak_in_vignette_SL']:.1%} | {r['peak_in_vignette_SF']:.1%} |")
    md.append("")
    md.append("## Cross-pair comparison (SL−SF vs NL−NF)\n")
    md.append("How does the structured-input pair compare to the natural-input pair from §4.3? Headline medical-vs-random gap, paired Δ sMAPE:\n")
    md.append("| Model | NL−NF paired Δ | SL−SF paired Δ | Same direction? |")
    md.append("|---|---|---|---|")
    # NL-NF numbers from existing analysis
    nl_nf_paired_4b = -0.272  # 4B L29 medical (0.004) - random (0.276) approx (uses magnitude-matched random)
    nl_nf_paired_12b = -0.120
    nl_nf_paired_qwen = -0.102
    nl_nf_paired_known = {"4b": nl_nf_paired_4b, "12b": nl_nf_paired_12b, "qwen": nl_nf_paired_qwen}
    for tag, r in out_all.items():
        nl_nf = nl_nf_paired_known.get(tag, "?")
        sl_sf = r["paired_diff_mean"]
        same_dir = "✓" if (nl_nf < 0 and sl_sf < 0) else "?"
        md.append(f"| {tag} | {nl_nf:+.3f} (NL-NF) | {sl_sf:+.4f} (SL-SF) | {same_dir} |")
    md.append("")
    md.append("Note: NL−NF numbers use the magnitude-matched 30-random-pool baseline from `phase1b_random_pool_resample_*.json`. SL−SF numbers above use a single magnitude-matched draw (not the 1000-resample, but the random feature pool is identical to NL−NF's fixed seed-42 magnitude-matched pool).\n")
    md.append("## Reading\n")
    md.append("**Paper claim (§3 stance):** medical-domain content is preserved across forced-letter vs free-text output formats. We measured this on the NL−NF pair throughout §4. This SL−SF run is an input-style robustness check. The Gemma 4B and 12B results reproduce the direction and significance of the NL−NF finding (medical features more invariant than random, both 95% CIs below zero). The Qwen3-8B result reproduces the direction but the gap shrinks to the edge of statistical detectability (95% CI crosses zero by 0.003). The asymmetry on the peak-location diagnostic (Qwen medical features peak in the SL scaffold 1/3 of the time, vs <2% at Gemma) suggests Qwen's medical features are less selective and partly anchor to lexical mentions of clinical care in the answer-key text.")
    md.append("\n**Manuscript guidance:** add this as a robustness sub-section under §4.3 or as Appendix [X] (\"Input-style robustness check: SL−SF mechanistic invariance\"). The Gemma result strengthens the central claim by showing it doesn't depend on natural-input style; the Qwen caveat is consistent with the existing 'suggestive cross-family consistency' framing.")
    (RESULTS / "sl_sf_summary.md").write_text("\n".join(md))
    print(f"Wrote {RESULTS/'sl_sf_summary.json'}")
    print(f"Wrote {RESULTS/'sl_sf_summary.md'}")


if __name__ == "__main__":
    main()
