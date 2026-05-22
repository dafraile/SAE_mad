# Decision-token feature characterization (4B / 12B / Qwen)

Direct test of the reviewer's 'scaffold-primary, medical-partial' framing. For each case, take the top-20 active features by activation at the NL decision token (B_decision) and at the NF decision token (D_decision), then compute (a) the Jaccard overlap of those two sets, (b) what fraction of NL-only features peak outside the shared vignette in their own (B) prompt, and (c) what fraction of NF-only features peak in the vignette in their own (D) prompt.

Reads only the saved per-case full-d_sae activation vectors from yesterday's masked-invariance run; CPU only.

## Headline table

| Model | n cases active at NL dec | n active at NF dec | overlap NL∩NF top-20 (Jaccard) | NL-only features peaking in scaffold | NF-only features peaking in vignette | v3 medical in NL/NF top-K |
|---|---|---|---|---|---|---|
| 4b | 49.7 | 46.7 | 0.000 (5–95% [0.00, 0.00]) | 87.0% (median 90.0%) | 27.8% (median 30.0%) | NL: 0/60, NF: 0/60 |
| 12b | 50.0 | 63.0 | 0.001 (5–95% [0.00, 0.00]) | 88.3% (median 90.0%) | 8.9% (median 10.0%) | NL: 0/60, NF: 0/60 |
| qwen | 50.0 | 50.0 | 0.324 (5–95% [0.25, 0.38]) | 94.7% (median 100.0%) | 10.4% (median 10.0%) | NL: 0/60, NF: 0/60 |

## Headline read (auto-generated)

- **4b**: top-20 NL and NF decision-token features overlap by Jaccard 0%. Of the features unique to NL's top-20, **87% peak on B-prompt scaffold tokens** (outside the shared vignette); of the features unique to NF's top-20, **28% peak on D-prompt vignette tokens**. v3-validated medical features are in NL's top-20 for 0/60 cases and in NF's top-20 for 0/60 cases.
- **12b**: top-20 NL and NF decision-token features overlap by Jaccard 0%. Of the features unique to NL's top-20, **88% peak on B-prompt scaffold tokens** (outside the shared vignette); of the features unique to NF's top-20, **9% peak on D-prompt vignette tokens**. v3-validated medical features are in NL's top-20 for 0/60 cases and in NF's top-20 for 0/60 cases.
- **qwen**: top-20 NL and NF decision-token features overlap by Jaccard 32%. Of the features unique to NL's top-20, **95% peak on B-prompt scaffold tokens** (outside the shared vignette); of the features unique to NF's top-20, **10% peak on D-prompt vignette tokens**. v3-validated medical features are in NL's top-20 for 0/60 cases and in NF's top-20 for 0/60 cases.

## Interpretation

A high overlap between NL and NF top-K features would say both formats use the same feature pool at the decision token. A low overlap with scaffold-peaking NL-only features and vignette-peaking NF-only features is the direct 'scaffold-primary at NL, content-primary at NF' pattern.

These numbers also let us quantify what 'medical-partial' means: the v3 medical features (3 per model) are not in the top-K at the decision token at any model (counts shown above are typically 0/60). Combined with the logit-attribution finding that v3 medical features have zero activation at the decision token in 60/60 cases at 4B and 12B, the cleanest mechanistic claim is **medical-absent at the decision token**, not medical-partial.