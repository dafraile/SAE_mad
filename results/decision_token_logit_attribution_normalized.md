# Normalized decision-token logit attribution (4B + 12B)

Reviewer concern: the raw numbers in the v2 logit-attribution table (e.g., 'other = 2627' at 4B, 'scaffold = 198.83' at 12B) invite a reviewer to ask 'fraction of what, exactly?' This file reports two normalized quantities derived from the same saved data:

  - **abs_fraction**: category K's net absolute linear effect divided by the total absolute linear effect across all categories. Caveats: this uses |sum| as a per-category-letter proxy for the true sum of |contributions| (which can only be computed if we have every feature's individual contribution, not just the per-category aggregate). Within a category, features with mixed-sign contributions will partially cancel before the abs is taken; the reported abs_fraction therefore understates the true unsigned-share for categories that contain features pushing in different directions. For directional/magnitude comparison across categories this is interpretable, but the literal interpretation is 'fraction of NET absolute linear effect,' not 'fraction of unsigned per-feature contribution.'

  - **margin_share**: category K's contribution to the predicted-vs-runner-up linear margin (pred_letter and runner_up letter chosen per case from the raw logits). Can be negative (the category pushes toward the runner-up rather than the prediction).

## Caveat for the manuscript caption (recommended phrasing)

> 'All values are derived from a linear logit-lens projection: c[i, L] = act_i · W_dec[i] · W_unembed[:, L_token]. This ignores (i) the final LayerNorm before unembedding, (ii) the transformer layers between the SAE layer and the unembedding, and (iii) SAE reconstruction error. Magnitudes are approximate; categorical comparisons are directionally informative. Causal attribution would require per-feature ablation forward passes — see future work.'

## Numbers

### 4B L29 (n = 60 cases)

| Category | abs_fraction mean (5–95%) | margin_share mean (5–95%) |
|---|---|---|
| medical | 0.0% (0.0%, 0.0%) | 0.0% (0.0%, 0.0%) |
| scaffold | 0.1% (0.0%, 0.4%) | 0.1% (-0.4%, 1.0%) |
| other | 99.9% (99.6%, 100.0%) | 99.9% (99.0%, 100.4%) |

### 12B L31 (n = 60 cases)

| Category | abs_fraction mean (5–95%) | margin_share mean (5–95%) |
|---|---|---|
| medical | 0.0% (0.0%, 0.0%) | 0.0% (0.0%, 0.0%) |
| scaffold | 50.3% (45.6%, 54.9%) | 26.5% (-27.4%, 63.6%) |
| other | 49.7% (45.1%, 54.4%) | 73.5% (36.4%, 127.4%) |

## Headline read (auto-generated)

- **4b**: medical features account for **0.0%** of the net absolute linear contribution at the NL decision token and **0.0%** of the predicted-vs-runner-up margin. Scaffold-proxy features: 0.1% abs, 0.1% margin. Other features: 99.9% abs, 99.9% margin.
- **12b**: medical features account for **0.0%** of the net absolute linear contribution at the NL decision token and **0.0%** of the predicted-vs-runner-up margin. Scaffold-proxy features: 50.3% abs, 26.5% margin. Other features: 49.7% abs, 73.5% margin.

The 'medical' fraction is essentially 0% at both models — consistent with the underlying finding that the v3 medical features have zero activation at the decision token in 60/60 cases (so their linear contribution is exactly zero, regardless of normalization). The medical-vs-scaffold-vs-other relative ranking differs across the two scales: at 4B the 'other' category dominates (~99% of abs contribution); at 12B 'scaffold-proxy' and 'other' are roughly comparable (each contributing ~40–50%).
