# Acuity-tier probes (rebuttal, C92e W1)

Emergency target: gold includes D (28 vs 32). LOO ROC-AUC, 1000-permutation p (full LOO refit per permutation).

| Model | Cond | Position | AUC (emergency) | perm p | 4-class macro-OVR AUC |
|---|---|---|---|---|---|
| 4b | NL | vignette | 0.954 | 0.0010 | 0.976 |
| 4b | NF | vignette | 0.962 | 0.0010 | 0.979 |
| 4b | NL | decision | 0.871 | 0.0010 | 0.824 |
| 4b | NF | decision | 0.879 | 0.0010 | 0.854 |
| 12b | NL | vignette | 0.993 | 0.0010 | 0.988 |
| 12b | NF | vignette | 0.994 | 0.0010 | 0.990 |
| 12b | NL | decision | 0.836 | 0.0020 | 0.871 |
| 12b | NF | decision | 0.796 | 0.0010 | 0.843 |
| qwen | NL | vignette | 0.999 | 0.0010 | 0.993 |
| qwen | NF | vignette | 1.000 | 0.0010 | 0.993 |
| qwen | NL | decision | 0.525 | 0.3806 | 0.678 |
| qwen | NF | decision | 0.485 | 0.5255 | 0.657 |

## Paired contrasts (case-bootstrap 95% CI on delta AUC)

| Contrast | delta AUC | 95% CI |
|---|---|---|
| 4b/vignette/NL-NF | -0.008 | [-0.030, +0.002] |
| 4b/decision/NL-NF | -0.009 | [-0.094, +0.072] |
| 4b/NL/vignette-decision | +0.084 | [-0.017, +0.201] |
| 4b/NF/vignette-decision | +0.083 | [-0.008, +0.188] |
| 12b/vignette/NL-NF | -0.001 | [-0.007, +0.000] |
| 12b/decision/NL-NF | +0.040 | [-0.070, +0.142] |
| 12b/NL/vignette-decision | +0.157 | [+0.053, +0.277] |
| 12b/NF/vignette-decision | +0.199 | [+0.088, +0.328] |
| qwen/vignette/NL-NF | -0.001 | [-0.007, +0.000] |
| qwen/decision/NL-NF | +0.039 | [-0.148, +0.236] |
| qwen/NL/vignette-decision | +0.474 | [+0.321, +0.615] |
| qwen/NF/vignette-decision | +0.515 | [+0.367, +0.667] |
