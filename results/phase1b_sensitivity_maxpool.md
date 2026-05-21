# Phase 1b sensitivity to K — MAX-POOL sMAPE

Companion to Appendix A1B. Closes reviewer Concern 4c (the appendix table currently reports mean-pool while the main text reports max-pool — this script gives the max-pool K-sweep for consistency).

Reused activations: saved per-case max-pool features from the 2026-05-21 masked-invariance GPU run (`results/phase1b_masked_full_activations_*.npz`). Reused medical and random feature IDs: top-20 contrastively-identified features from the original mean-pool K-sweep (`results/phase1b_sensitivity_*_L*.json`).

Per-case sMAPE = mean over the K features of per-feature |B_max − D_max| / ((|B_max| + |D_max|)/2). Bootstrap CIs: 1000 resamples of the 60 cases. Δ_paired_per_case is the paired difference (medical − random) per case, then bootstrap mean and 95% CI.

## 4B L29

| K | medical sMAPE (mean, 95% CI) | random sMAPE (mean, 95% CI) | Δ_paired_per_case (mean, 95% CI) | verdict |
|---|---|---|---|---|
| 3 | 0.0063 [0.0031, 0.0115] | 0.0312 [0.0208, 0.0435] | -0.0249 [-0.0384, -0.0130] | medical < random (sig) |
| 5 | 0.0115 [0.0032, 0.0268] | 0.2285 [0.1815, 0.2778] | -0.2170 [-0.2680, -0.1683] | medical < random (sig) |
| 10 | 0.1329 [0.0973, 0.1687] | 0.2601 [0.2165, 0.3059] | -0.1272 [-0.1632, -0.0942] | medical < random (sig) |
| 20 | 0.1531 [0.1305, 0.1763] | 0.2139 [0.1727, 0.2559] | -0.0608 [-0.0891, -0.0341] | medical < random (sig) |

## 12B L31

| K | medical sMAPE (mean, 95% CI) | random sMAPE (mean, 95% CI) | Δ_paired_per_case (mean, 95% CI) | verdict |
|---|---|---|---|---|
| 3 | 0.0058 [0.0031, 0.0095] | 0.7414 [0.6480, 0.8330] | -0.7356 [-0.8253, -0.6431] | medical < random (sig) |
| 5 | 0.0053 [0.0036, 0.0077] | 0.4549 [0.3968, 0.5115] | -0.4496 [-0.5061, -0.3918] | medical < random (sig) |
| 10 | 0.0389 [0.0232, 0.0592] | 0.5920 [0.5533, 0.6385] | -0.5531 [-0.5997, -0.5069] | medical < random (sig) |
| 20 | 0.0871 [0.0705, 0.1049] | 0.4120 [0.3823, 0.4459] | -0.3249 [-0.3630, -0.2914] | medical < random (sig) |

## QWEN L31

| K | medical sMAPE (mean, 95% CI) | random sMAPE (mean, 95% CI) | Δ_paired_per_case (mean, 95% CI) | verdict |
|---|---|---|---|---|
| 3 | 0.0348 [0.0257, 0.0450] | 0.0579 [0.0134, 0.1025] | -0.0231 [-0.0721, +0.0239] | ns |

**Note on Qwen:** only K=3 reported, because the top-20 contrastive medical-feature ID for Qwen3-8B at L31 has not been run (the v3-validated 3-feature set is what's used in the main text). A full top-20 contrastive ID for Qwen is future work and would require ~30 min of A100 time.
