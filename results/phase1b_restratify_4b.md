# Phase 1b 4B per-stratum re-bookkeeping (canonical strata)

Reconciles Gemma 4B Phase 1b per-stratum tables against the canonical `gap_decomposition` strata used by the body (Tables 1 + 2). The Phase 1b pipeline cast missing judge labels to `False`, collapsing what gap_decomposition explicitly calls `judges_disagree`. **No feature re-extraction.** Per-case sMAPE (mod-index) and cosine are unchanged; only the case → stratum mapping is corrected. Bootstrap 95% CIs (2000 resamples) re-computed on the corrected stratum membership.

## Canonical 4B strata counts (source of truth: gap_decomposition.json)

| Stratum | n |
|---|---|
| both_right | 29 |
| NF_only_right | 14 |
| NL_only_right | 1 |
| both_wrong | 12 |
| judges_disagree | 4 |

## Per-layer / per-stratum table (replaces tab:full_4b at lines ~1487 in main.tex)

| L | Stratum | n | n_cos | ΔsMAPE | 95% CI | Δcos | 95% CI |
|---|---|---|---|---|---|---|---|
| 9 | both_wrong | 12 | — | -0.202 | [-0.238, -0.167] | +0.024 | [+0.019, +0.029] |
| 9 | both_right | 29 | — | -0.232 | [-0.265, -0.201] | +0.023 | [+0.019, +0.026] |
| 9 | NF_only_right | 14 | — | -0.213 | [-0.241, -0.186] | +0.024 | [+0.016, +0.034] |
| 9 | judges_disagree | 4 | — | -0.207 | [-0.229, -0.190] | +0.026 | [+0.020, +0.033] |
| 9 | NL_only_right | 1 | — | -0.197 | [-0.197, -0.197] | +0.019 | [+0.019, +0.019] |
| 17 | both_wrong | 12 | — | -0.290 | [-0.363, -0.225] | +0.039 | [+0.025, +0.056] |
| 17 | both_right | 29 | 28 | -0.269 | [-0.305, -0.231] | +0.035 | [+0.023, +0.050] |
| 17 | NF_only_right | 14 | 13 | -0.258 | [-0.315, -0.205] | +0.027 | [+0.017, +0.039] |
| 17 | judges_disagree | 4 | — | -0.209 | [-0.269, -0.148] | +0.024 | [+0.012, +0.041] |
| 17 | NL_only_right | 1 | — | -0.206 | [-0.206, -0.206] | +0.047 | [+0.047, +0.047] |
| 22 | both_wrong | 12 | — | -0.093 | [-0.175, +0.010] | +0.022 | [-0.030, +0.060] |
| 22 | both_right | 29 | — | -0.090 | [-0.142, -0.023] | +0.037 | [+0.007, +0.061] |
| 22 | NF_only_right | 14 | — | -0.141 | [-0.186, -0.095] | +0.043 | [+0.026, +0.058] |
| 22 | judges_disagree | 4 | — | -0.141 | [-0.211, -0.052] | +0.041 | [-0.006, +0.070] |
| 22 | NL_only_right | 1 | — | -0.170 | [-0.170, -0.170] | +0.050 | [+0.050, +0.050] |
| 29 | both_wrong | 12 | 11 | -0.336 | [-0.409, -0.265] | +0.060 | [+0.025, +0.101] |
| 29 | both_right | 29 | 24 | -0.275 | [-0.329, -0.229] | +0.054 | [+0.039, +0.072] |
| 29 | NF_only_right | 14 | 12 | -0.338 | [-0.402, -0.280] | +0.049 | [+0.028, +0.071] |
| 29 | judges_disagree | 4 | — | -0.274 | [-0.331, -0.217] | +0.075 | [+0.031, +0.139] |
| 29 | NL_only_right | 1 | — | -0.334 | [-0.334, -0.334] | +0.058 | [+0.058, +0.058] |

## Headline-layer (L29) rows for tab:phase1b_headline (lines ~679 in main.tex)

| Stratum | n | ΔsMAPE 95% CI | Δcos 95% CI |
|---|---|---|---|
| both_wrong | 12 | -0.336 [-0.409, -0.265] | +0.060 [+0.025, +0.101] |
| both_right | 29 | -0.275 [-0.329, -0.229] | +0.054 [+0.039, +0.072] |
| NF_only_right | 14 | -0.338 [-0.402, -0.280] | +0.049 [+0.028, +0.071] |

## Note on transcription

The current `tab:full_4b` in `main.tex` has both-right=30 and NF-only-right=13, which is an off-by-one transcription error: the canonical numbers are 29 and 14. Other stratum counts (NL_only_right=1, both_wrong=12, judges_disagree=4) match. Update the column headers to use the canonical counts; the per-stratum statistics should be replaced with the values in the table above.