# SL−SF mechanistic invariance — robustness across input style

Parallel to the NL−NF mechanistic analysis in §4.3, run on the structured-input × output-format pair to test whether the medical-vs-random format-invariance result depends on natural patient-voice input.

## Headline table (max-pool, paired bootstrap 95% CIs over cases)

| Model | n | med sMAPE | rnd sMAPE | med cos | rnd cos | paired Δ med−rnd | 95% CI | Sig? |
|---|---|---|---|---|---|---|---|---|
| 4b L29 | 60 | 0.0042 | 0.0827 | 1.0000 | 0.9845 | -0.0810 | [-0.096, -0.066] | ✓ |
| 12b L31 | 60 | 0.0025 | 0.0692 | 1.0000 | 0.9863 | -0.0618 | [-0.083, -0.034] | ✓ |
| qwen L31 | 60 | 0.1181 | 0.1382 | 0.9907 | 0.9812 | -0.0075 | [-0.019, +0.003] | ns |

## Vignette-mask sanity check (expected ~0)

| Model | med vignette sMAPE | rnd vignette sMAPE |
|---|---|---|
| 4b | 0.0042 | 0.0024 |
| 12b | 0.0025 | 0.0026 |
| qwen | 0.0028 | 0.0038 |

Both medical and random sMAPE collapse to ~0.002–0.004 on the shared structured-content vignette mask, confirming causal-masking trivial invariance.

## Medical-feature peak location

Fraction of (case × medical-feature) pairs whose peak activation lies inside the shared vignette (vs. on the SL-only scaffold for SL prompts, or on the chat-template suffix for SF prompts).

| Model | SL: peak in vignette | SF: peak in vignette |
|---|---|---|
| 4b | 100.0% | 98.9% |
| 12b | 99.4% | 100.0% |
| qwen | 66.1% | 98.9% |

## Cross-pair comparison (SL−SF vs NL−NF)

How does the structured-input pair compare to the natural-input pair from §4.3? Headline medical-vs-random gap, paired Δ sMAPE:

| Model | NL−NF paired Δ | SL−SF paired Δ | Same direction? |
|---|---|---|---|
| 4b | -0.272 (NL-NF) | -0.0810 (SL-SF) | ✓ |
| 12b | -0.120 (NL-NF) | -0.0618 (SL-SF) | ✓ |
| qwen | -0.102 (NL-NF) | -0.0075 (SL-SF) | ✓ |

Note: NL−NF numbers use the magnitude-matched 30-random-pool baseline from `phase1b_random_pool_resample_*.json`. SL−SF numbers above use a single magnitude-matched draw (not the 1000-resample, but the random feature pool is identical to NL−NF's fixed seed-42 magnitude-matched pool).

## Reading

**Paper claim (§3 stance):** medical-domain content is preserved across forced-letter vs free-text output formats. We measured this on the NL−NF pair throughout §4. This SL−SF run is an input-style robustness check. The Gemma 4B and 12B results reproduce the direction and significance of the NL−NF finding (medical features more invariant than random, both 95% CIs below zero). The Qwen3-8B result reproduces the direction but the gap shrinks to the edge of statistical detectability (95% CI crosses zero by 0.003). The asymmetry on the peak-location diagnostic (Qwen medical features peak in the SL scaffold 1/3 of the time, vs <2% at Gemma) suggests Qwen's medical features are less selective and partly anchor to lexical mentions of clinical care in the answer-key text.

**Manuscript guidance:** add this as a robustness sub-section under §4.3 or as Appendix [X] ("Input-style robustness check: SL−SF mechanistic invariance"). The Gemma result strengthens the central claim by showing it doesn't depend on natural-input style; the Qwen caveat is consistent with the existing 'suggestive cross-family consistency' framing.