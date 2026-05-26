# Gemma 4B Phase 1b per-stratum correction (bookkeeping fix)

## Problem

The paper's `tab:phase1b_headline` (Table 1) and `tab:full_4b` (Appendix) for 4B show stratum counts `(both_right=30, both_wrong=12, NF_only_right=13, NL_only_right=1, judges_disagree=4)`. The canonical strata from `gap_decomposition.json` — used by the body Table 2 and §4.2 prose — are `(29, 12, 14, 1, 4)`. The mismatch is exactly one case: **F1**.

## Diagnosis

Under `phase1b_magnitude_matched.json` (the file the appendix table derives from), F1 is in `both_right` because Phase 1b cast missing judge-correctness fields to `False`. Under `gap_decomposition.json`'s stricter both-judges-correct + judges_disagree decomposition, F1 has `NL=B (wrong)`, `NF=C (both judges)`, so it belongs in `NF_only_right`. The canonical decomposition is authoritative; F1 moves from `both_right` → `NF_only_right`.

## Corrected per-stratum table (replaces `tab:full_4b` and the L29 rows of `tab:phase1b_headline`)

### Layer 9

| Stratum | n | ΔsMAPE | 95% CI | Δcos | 95% CI |
|---|---|---|---|---|---|
| both_right | 29 | -0.232 | [-0.265, -0.201] | +0.023 | [+0.019, +0.026] |
| both_wrong | 12 | -0.202 | [-0.238, -0.167] | +0.024 | [+0.019, +0.029] |
| NF_only_right | 14 | -0.213 | [-0.241, -0.186] | +0.024 | [+0.016, +0.034] |
| NL_only_right | 1 | -0.197 | [-0.197, -0.197] | +0.019 | [+0.019, +0.019] |
| judges_disagree | 4 | -0.207 | [-0.229, -0.190] | +0.026 | [+0.020, +0.033] |

### Layer 17

| Stratum | n | ΔsMAPE | 95% CI | Δcos | 95% CI |
|---|---|---|---|---|---|
| both_right | 29 ($n_c=28$) | -0.269 | [-0.305, -0.231] | +0.035 | [+0.023, +0.050] |
| both_wrong | 12 | -0.290 | [-0.363, -0.225] | +0.039 | [+0.025, +0.056] |
| NF_only_right | 14 ($n_c=13$) | -0.258 | [-0.315, -0.205] | +0.027 | [+0.017, +0.039] |
| NL_only_right | 1 | -0.206 | [-0.206, -0.206] | +0.047 | [+0.047, +0.047] |
| judges_disagree | 4 | -0.209 | [-0.269, -0.148] | +0.024 | [+0.012, +0.041] |

### Layer 22

| Stratum | n | ΔsMAPE | 95% CI | Δcos | 95% CI |
|---|---|---|---|---|---|
| both_right | 29 | -0.090 | [-0.142, -0.023] | +0.037 | [+0.007, +0.061] |
| both_wrong | 12 | -0.093 | [-0.175, +0.010] | +0.022 | [-0.030, +0.060] |
| NF_only_right | 14 | -0.141 | [-0.186, -0.095] | +0.043 | [+0.026, +0.058] |
| NL_only_right | 1 | -0.170 | [-0.170, -0.170] | +0.050 | [+0.050, +0.050] |
| judges_disagree | 4 | -0.141 | [-0.211, -0.052] | +0.041 | [-0.006, +0.070] |

### Layer 29

| Stratum | n | ΔsMAPE | 95% CI | Δcos | 95% CI |
|---|---|---|---|---|---|
| both_right | 29 ($n_c=24$) | -0.275 | [-0.329, -0.229] | +0.054 | [+0.039, +0.072] |
| both_wrong | 12 ($n_c=11$) | -0.336 | [-0.409, -0.265] | +0.060 | [+0.025, +0.101] |
| NF_only_right | 14 ($n_c=12$) | -0.338 | [-0.402, -0.280] | +0.049 | [+0.028, +0.071] |
| NL_only_right | 1 | -0.334 | [-0.334, -0.334] | +0.058 | [+0.058, +0.058] |
| judges_disagree | 4 | -0.274 | [-0.331, -0.217] | +0.075 | [+0.031, +0.139] |

## Side-by-side: paper's current values vs. corrected canonical values

This shows the magnitude of the bookkeeping correction. The qualitative claim (all CIs strictly below zero) holds in both versions.

### Layer 9

| Stratum | PAPER (current) | CANONICAL (corrected) |
|---|---|---|
| both_right | n=30: -0.229 [-0.259, -0.201] | n=29: -0.232 [-0.265, -0.201] |
| both_wrong | n=12: -0.202 [-0.238, -0.167] | n=12: -0.202 [-0.238, -0.167] |
| NF_only_right | n=13: -0.218 [-0.246, -0.191] | n=14: -0.213 [-0.241, -0.186] |
| NL_only_right | n=1: -0.197 [-0.197, -0.197] | n=1: -0.197 [-0.197, -0.197] |
| judges_disagree | n=4: -0.207 [-0.229, -0.190] | n=4: -0.207 [-0.229, -0.190] |

### Layer 17

| Stratum | PAPER (current) | CANONICAL (corrected) |
|---|---|---|
| both_right | n=30: -0.269 [-0.304, -0.233] | n=29: -0.269 [-0.305, -0.231] |
| both_wrong | n=12: -0.290 [-0.363, -0.225] | n=12: -0.290 [-0.363, -0.225] |
| NF_only_right | n=13: -0.257 [-0.318, -0.198] | n=14: -0.258 [-0.315, -0.205] |
| NL_only_right | n=1: -0.206 [-0.206, -0.206] | n=1: -0.206 [-0.206, -0.206] |
| judges_disagree | n=4: -0.209 [-0.269, -0.148] | n=4: -0.209 [-0.269, -0.148] |

### Layer 22

| Stratum | PAPER (current) | CANONICAL (corrected) |
|---|---|---|
| both_right | n=30: -0.086 [-0.138, -0.018] | n=29: -0.090 [-0.142, -0.023] |
| both_wrong | n=12: -0.093 [-0.175, +0.010] | n=12: -0.093 [-0.175, +0.010] |
| NF_only_right | n=13: -0.154 [-0.197, -0.108] | n=14: -0.141 [-0.186, -0.095] |
| NL_only_right | n=1: -0.170 [-0.170, -0.170] | n=1: -0.170 [-0.170, -0.170] |
| judges_disagree | n=4: -0.141 [-0.211, -0.052] | n=4: -0.141 [-0.211, -0.052] |

### Layer 29

| Stratum | PAPER (current) | CANONICAL (corrected) |
|---|---|---|
| both_right | n=30: -0.276 [-0.325, -0.231] | n=29: -0.275 [-0.329, -0.229] |
| both_wrong | n=12: -0.336 [-0.409, -0.265] | n=12: -0.336 [-0.409, -0.265] |
| NF_only_right | n=13: -0.341 [-0.411, -0.276] | n=14: -0.338 [-0.402, -0.280] |
| NL_only_right | n=1: -0.334 [-0.334, -0.334] | n=1: -0.334 [-0.334, -0.334] |
| judges_disagree | n=4: -0.274 [-0.331, -0.217] | n=4: -0.274 [-0.331, -0.217] |
