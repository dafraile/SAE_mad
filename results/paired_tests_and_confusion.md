# Paired tests + per-acuity + confusion matrices

Reviewer-asked supplementary statistics for §4.1 / §6.

## (1) Paired NL vs NF test (4-way both-judges-correct)

| Model | n | NL acc | NF acc | NL−NF (pp) | 95% CI | McNemar p | Discordant b:c |
|---|---|---|---|---|---|---|---|
| gemma-3-4b-it | 60 | 55.0% | 71.7% | -16.7 | [-30.0, -3.3] | 0.0309 | 4:14 |
| gemma-3-12b-it | 60 | 81.7% | 71.7% | +10.0 | [+3.3, +18.3] | 0.0312 | 6:0 |
| qwen3-8b | 60 | 75.0% | 68.3% | +6.7 | [-6.7, +20.0] | 0.4545 | 10:6 |

Interpretation:
- McNemar exact two-sided p-value on the discordant cells (NL right & NF wrong vs NL wrong & NF right). 60-case sample → discordant pairs are small but tests are well-defined.
- 95% CI is the paired bootstrap (2000 resamples) on the per-case accuracy difference.

## (2) Per-acuity (most-urgent gold letter) breakdown

### gemma-3-4b-it
| Gold acuity | n | SL | NL | NF (both) | NF (either) | DEFERRED |
|---|---|---|---|---|---|---|
| A | 8 | 12% | 12% | 0% | 12% | 0% |
| B | 10 | 80% | 90% | 80% | 100% | 0% |
| C | 14 | 50% | 36% | 86% | 86% | 0% |
| D | 28 | 68% | 64% | 82% | 86% | 0% |

### gemma-3-12b-it
| Gold acuity | n | SL | NL | NF (both) | NF (either) | DEFERRED |
|---|---|---|---|---|---|---|
| A | 8 | 38% | 25% | 25% | 25% | 0% |
| B | 10 | 90% | 100% | 90% | 90% | 30% |
| C | 14 | 79% | 93% | 79% | 79% | 0% |
| D | 28 | 93% | 86% | 75% | 75% | 4% |

### qwen3-8b
| Gold acuity | n | SL | NL | NF (both) | NF (either) | DEFERRED |
|---|---|---|---|---|---|---|
| A | 8 | 75% | 62% | 38% | 38% | 0% |
| B | 10 | 60% | 70% | 80% | 100% | 10% |
| C | 14 | 71% | 86% | 64% | 71% | 0% |
| D | 28 | 82% | 75% | 75% | 82% | 4% |

## (3) Confusion matrices (rows = gold acuity bucket, columns = predicted letter)

### gemma-3-4b-it
#### SL
| gold ↓ \ pred → | A | B | C | D | ? | DEFERRED |
|---|---|---|---|---|---|---|
| A | 2 | 7 | 0 | 0 | 0 | 0 |
| B | 0 | 8 | 2 | 0 | 0 | 0 |
| C | 0 | 7 | 25 | 0 | 0 | 0 |
| D | 0 | 5 | 4 | 0 | 0 | 0 |

#### NL
| gold ↓ \ pred → | A | B | C | D | ? | DEFERRED |
|---|---|---|---|---|---|---|
| A | 2 | 7 | 0 | 0 | 0 | 0 |
| B | 1 | 10 | 0 | 0 | 0 | 0 |
| C | 0 | 9 | 21 | 0 | 0 | 0 |
| D | 0 | 6 | 4 | 0 | 0 | 0 |

#### NF
| gold ↓ \ pred → | A | B | C | D | ? | DEFERRED |
|---|---|---|---|---|---|---|
| A | 1 | 5 | 2 | 0 | 1 | 0 |
| B | 0 | 7 | 0 | 0 | 2 | 0 |
| C | 0 | 1 | 34 | 1 | 0 | 0 |
| D | 0 | 1 | 3 | 1 | 1 | 0 |

### gemma-3-12b-it
#### SL
| gold ↓ \ pred → | A | B | C | D | ? | DEFERRED |
|---|---|---|---|---|---|---|
| A | 4 | 5 | 0 | 0 | 0 | 0 |
| B | 1 | 8 | 0 | 0 | 0 | 0 |
| C | 0 | 0 | 28 | 3 | 0 | 0 |
| D | 0 | 1 | 1 | 9 | 0 | 0 |

#### NL
| gold ↓ \ pred → | A | B | C | D | ? | DEFERRED |
|---|---|---|---|---|---|---|
| A | 3 | 6 | 0 | 0 | 0 | 0 |
| B | 0 | 9 | 0 | 0 | 0 | 0 |
| C | 0 | 0 | 28 | 1 | 0 | 0 |
| D | 0 | 2 | 2 | 9 | 0 | 0 |

#### NF
| gold ↓ \ pred → | A | B | C | D | ? | DEFERRED |
|---|---|---|---|---|---|---|
| A | 2 | 6 | 0 | 0 | 0 | 0 |
| B | 0 | 9 | 1 | 0 | 0 | 0 |
| C | 0 | 1 | 29 | 2 | 0 | 0 |
| D | 1 | 3 | 3 | 3 | 0 | 0 |

### qwen3-8b
#### SL
| gold ↓ \ pred → | A | B | C | D | ? | DEFERRED |
|---|---|---|---|---|---|---|
| A | 8 | 2 | 0 | 0 | 0 | 0 |
| B | 4 | 4 | 0 | 0 | 0 | 0 |
| C | 0 | 1 | 31 | 3 | 0 | 0 |
| D | 2 | 0 | 3 | 2 | 0 | 0 |

#### NL
| gold ↓ \ pred → | A | B | C | D | ? | DEFERRED |
|---|---|---|---|---|---|---|
| A | 7 | 3 | 0 | 0 | 0 | 0 |
| B | 3 | 5 | 0 | 0 | 0 | 0 |
| C | 2 | 0 | 30 | 0 | 0 | 0 |
| D | 4 | 0 | 3 | 3 | 0 | 0 |

#### NF
| gold ↓ \ pred → | A | B | C | D | ? | DEFERRED |
|---|---|---|---|---|---|---|
| A | 4 | 5 | 0 | 0 | 0 | 0 |
| B | 0 | 8 | 0 | 0 | 3 | 0 |
| C | 0 | 3 | 21 | 1 | 1 | 0 |
| D | 0 | 3 | 2 | 4 | 5 | 0 |

## (4) Triage direction + severity-weighted error

| Model | Format | n correct | n under-triage | n over-triage | n no-commit | mean steps off (committed) | frac ≥2 steps off |
|---|---|---|---|---|---|---|---|
| gemma-3-4b-it | SL | 35 | 16 | 9 | 0 | 0.42 | 0.00 |
| gemma-3-4b-it | NL | 33 | 20 | 7 | 0 | 0.45 | 0.00 |
| gemma-3-4b-it | NF | 43 | 5 | 8 | 4 | 0.27 | 0.04 |
| gemma-3-12b-it | SL | 49 | 3 | 8 | 0 | 0.18 | 0.00 |
| gemma-3-12b-it | NL | 49 | 4 | 7 | 0 | 0.18 | 0.00 |
| gemma-3-12b-it | NF | 43 | 8 | 9 | 0 | 0.30 | 0.02 |
| qwen3-8b | SL | 45 | 10 | 5 | 0 | 0.28 | 0.03 |
| qwen3-8b | NL | 45 | 12 | 3 | 0 | 0.32 | 0.07 |
| qwen3-8b | NF | 37 | 8 | 6 | 9 | 0.27 | 0.00 |

Under-triage = predicted letter has lower acuity than the lowest gold letter; over-triage = predicted is higher than the highest gold letter; no-commit = judges disagreed (NF) or letter unparseable.