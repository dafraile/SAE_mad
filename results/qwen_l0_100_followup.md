# Qwen L0_100 — three follow-up analyses

Three CPU-only analyses on the saved `qwen_l0_100_masked_full_activations.npz` to close gaps the L0_50→L0_100 swap opened in the appendix.

## (A) Magnitude-matched random resample, NL−NF

Parallel to the Gemma rows in `tab:resample`.

| Cell | med sMAPE | rnd sMAPE 5–95% | perm-p |
|---|---|---|---|
| Qwen L31 (L0_100) | **0.0021** | 0.2177 [0.0943, 0.3614] | **0.0000** |

Magnitude-matched pool size: 338. z-score: -2.65. Random distribution over 1000 draws.

## (B) Medical-feature peak in vignette

Parallel to the Gemma numbers in `app:token_masks` (Gemma: 98–100%).

| Condition | peak in vignette % | n active feature-case pairs |
|---|---|---|
| NL | 100.0% | 164/164 |
| NF | 100.0% | 164/164 |
| SL | 99.4% | 168/169 |
| SF | 98.8% | 168/170 |

## (C) Per-stratum NL-NF invariance (medical vs random)

For `tab:phase1b_full` Qwen rows updated to L0_100.

| Stratum | n | med sMAPE | rnd sMAPE | Δ sMAPE | 95% CI |
|---|---|---|---|---|---|
| both_right | 35 | 0.0028 | 0.0640 | -0.0612 | [-0.0811, -0.0436] |
| both_wrong | 6 | 0.0044 | 0.0423 | -0.0379 | [-0.0792, -0.0108] |
| NF_only_right | 6 | 0.0029 | 0.1238 | -0.1209 | [-0.1899, -0.0576] |
| NL_only_right | 8 | 0.0019 | 0.0541 | -0.0522 | [-0.0807, -0.0265] |
| judges_disagree | 5 | 0.0039 | 0.1362 | -0.1323 | [-0.1924, -0.0934] |

Direction of effect (negative Δ sMAPE = medical more invariant than random) holds in every populated stratum.
