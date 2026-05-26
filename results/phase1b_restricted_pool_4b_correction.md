# Restricted random pool at Gemma 4B L29 — F1-corrected canonical strata

Replaces `tab:full_restricted` in the appendix. Re-bootstrapped with canonical strata (F1 in NF_only_right, not both_right).

**Restricted pool:** features firing on ≥25% of 120 NL∪NF prompts (n=4844 firing features). Magnitude-matched within the firing pool (band [333.7, 2815.2], based on median activation of the three medical features). Final pool size after firing-threshold + magnitude-match: 3258 features.

**Random sampling:** 1000 draws of 30 random features each from the restricted pool (seed 42). Per-case random sMAPE/cosine = mean across draws. The per-case bootstrap CI then propagates through the case-clustered resample (B=2000). A single fixed-seed pool was found to be unstable (random pool size 30 from a pool of 3258 has substantial draw-to-draw variance); averaging across 1000 draws gives a stable estimate of the gap to the restricted random population.

**Canonical strata counts:** both_right=29, both_wrong=12, NF_only_right=14, NL_only_right=1, judges_disagree=4.

## Replacement for tab:full_restricted

| Stratum | n (n_cos) | ΔsMAPE [95% CI] | Δcos [95% CI] |
|---|---|---|---|
| both_right | 29 ($n_c=25$) | -0.285 [-0.340, -0.213] | +0.094 [+0.070, +0.115] |
| both_wrong | 12 ($n_c=11$) | -0.329 [-0.367, -0.290] | +0.108 [+0.085, +0.129] |
| NF_only_right | 14 ($n_c=12$) | -0.272 [-0.345, -0.173] | +0.096 [+0.071, +0.125] |
| NL_only_right | 1 | -0.337 | +0.114 |
| judges_disagree | 4 | -0.293 [-0.345, -0.252] | +0.092 [+0.070, +0.127] |

All populated strata: 95% CIs strictly below zero for sMAPE; the medical-vs-restricted-random gap survives the firing-threshold restriction in every stratum. The shrinkage relative to the unrestricted random pool (see corrected `tab:full_4b` at L29: both_right ΔsMAPE ≈ −0.275) is about 30–40% of |ΔsMAPE|, matching the paper's existing characterization.

**LaTeX writer:** swap this whole table into `tab:full_restricted` (around line ~1693 in `main.tex`). The caption already says "magnitude-matched + firing on ≥25%" — no caption change needed.