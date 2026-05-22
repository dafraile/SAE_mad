# Option-order shuffle — case-clustered bootstrap CIs

Reviewer concern: the K permutations within a single case are not independent (they share the same vignette content). A naive IID bootstrap over (case, permutation) pairs under-states variance. This script uses a **case-clustered bootstrap** (2000 resamples): draw 60 cases with replacement, aggregate the K permutations within each resampled case, compute the grand mean. Percentile CI at α=0.05.

## Results

| run | K | metric | point | 95% CI (case-clustered) | SE_clustered (pp) |
|---|---|---|---|---|---|
| 4b | 3 | same-letter % | 21.1% | [15.0%, 27.2%] | 3.16 |
| 4b | 3 | same-content % | 67.2% | [57.8%, 76.7%] | 4.91 |
| 4b | 3 | shuffled NL accuracy | 71.7% | [61.7%, 81.1%] | 5.10 |
| 12b | 3 | same-letter % | 25.0% | [18.9%, 32.2%] | 3.39 |
| 12b | 3 | same-content % | 80.6% | [73.3%, 87.8%] | 3.79 |
| 12b | 3 | shuffled NL accuracy | 78.9% | [68.9%, 88.3%] | 4.86 |
| qwen | 3 | same-letter % | 25.6% | [19.4%, 32.2%] | 3.29 |
| qwen | 3 | same-content % | 82.2% | [75.0%, 88.9%] | 3.58 |
| qwen | 3 | shuffled NL accuracy | 72.8% | [62.8%, 82.8%] | 5.18 |

Interpretation: a tight CI means the metric is stable across case resamples (the result is unlikely to be driven by a few extreme cases). Wider CIs (especially for shuffled accuracy on n=60 cases) reflect genuine case-level variability.