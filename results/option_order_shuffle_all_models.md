# Option-order shuffle — cross-model summary (4B / 12B / Qwen)

Falsifiable test of position-bias vs content-prior at the forced-letter scaffold. For each of 60 canonical cases, randomize the letter→content mapping in the forced-letter scaffold, run greedy forced-letter, score same-letter % vs same-content % vs accuracy.

**Two runs:**
- **K=23 exhaustive** (the manuscript headline): all 23 non-identity permutations of (A,B,C,D) per case → 1380 shuffles per model. Case-clustered bootstrap 95% CIs (B=2000).
- **K=3 baseline** (kept as inset for cross-checking with earlier drafts): 3 random non-identity permutations per case → 180 shuffles per model.

## Headline table — K=23 exhaustive with case-clustered 95% CIs

| Model | n | K | same-letter % [95% CI] | same-content % [95% CI] | canonical NL acc | shuffled NL acc [95% CI] | NF (4-way both) | shuffled − NF |
|---|---|---|---|---|---|---|---|---|
| 4b | 60 | 23 | 22.4% [21.4, 23.4] | 64.5% [55.9, 73.0] | 55.0% | 69.8% [60.7, 78.3] | 71.7% | -1.9 pp |
| 12b | 60 | 23 | 20.8% [18.6, 23.0] | 80.3% [73.6, 86.6] | 81.7% | 76.3% [66.3, 85.3] | 71.7% | +4.6 pp |
| qwen | 60 | 23 | 23.3% [20.5, 26.5] | 82.6% [76.2, 88.4] | 75.0% | 75.4% [66.0, 84.5] | 68.3% | +7.1 pp |

Reading guide:
- **same-letter %** is below chance (25%) at 4B and 12B (upper CI bound < 25%); at Qwen the CI just barely touches 25% but the point estimate is below. → **no position bias at any model.**
- **same-content %** is far above chance (lower CI bound ≥ 56% across all three models). → **strong content prior at every model.**
- **shuffled − NF gap:** at 4B the gap is −1.9 pp with the shuffled CI containing NF (statistically indistinguishable). At 12B and Qwen, shuffled forced-letter still beats free-text by ≈5–7 pp under exhaustive shuffles — separate NF-mode accuracy penalty independent of letter-binding.

## Inset — K=3 baseline (earlier draft used these numbers)

Kept here for cross-referencing with any v3-pre-2026-05-22 manuscript draft. The K=23 numbers above should be the headline in v3 final.

| Model | K | same-letter % [95% CI] | same-content % [95% CI] | shuffled NL acc [95% CI] | shuffled − NF |
|---|---|---|---|---|---|
| 4b | 3 | 21.1% [15.0, 27.2] | 67.2% [57.8, 76.7] | 71.7% [61.7, 81.1] | -0.0 pp |
| 12b | 3 | 25.0% [18.9, 32.2] | 80.6% [73.3, 87.8] | 78.9% [68.9, 88.3] | +7.2 pp |
| qwen | 3 | 25.6% [19.4, 32.2] | 82.2% [75.0, 88.9] | 72.8% [62.8, 82.8] | +4.4 pp |

**Honesty note on the K=3→K=23 transition for 4B:** the K=3 point estimate was shuffled NL acc = NF acc = 71.7% *to the case*, which earlier drafts framed as the entire format penalty IS the canonical letter-binding. The more precise K=23 estimate is 69.8% with 95% CI [60.7%, 78.3%]; the CI contains NF (71.7%), so the corrected claim for v3 final is **'shuffled NL accuracy is statistically indistinguishable from NF accuracy at 4B (n=60 cases)'** rather than 'exactly equal.' The qualitative story (canonical letter-binding × content prior explains essentially all of 4B's NL→NF accuracy penalty) survives. See `results/option_order_shuffle_exhaustive_summary.md` for the full K=3-vs-K=23 comparison with CIs.

## Letter distribution (canonical vs K=23 shuffles)

| Model | NL canonical | NL shuffles (K=23, total 1380) |
|---|---|---|
| 4b | A:3 B:32 C:25 D:0 | A:466 B:339 C:324 D:251 |
| 12b | A:3 B:17 C:30 D:10 | A:401 B:259 C:348 D:372 |
| qwen | A:16 B:8 C:33 D:3 | A:477 B:280 C:292 D:331 |

## Content distribution under K=23 shuffles

Which acuity content does the model pick (regardless of letter)? Under shuffles, a content prior shows up as concentration on one row.

| Model | Fine to monitor | Weeks | **24-48h** | Go to ER |
|---|---|---|---|---|
| 4b | 27 | 346 | **1005** | 2 |
| 12b | 53 | 271 | **930** | 126 |
| qwen | 289 | 128 | **866** | 97 |

All three models concentrate strongly on **'See a doctor within 24-48 hours'** content under exhaustive shuffles. **4B picks 'Go to ER' content only 2/1380 = 0.14% of the time** even when the canonical D position is randomized — a robust capability-scaling signal: 12B picks ER content 9.1%, Qwen 7.0%. 4B has a learned content-level aversion to the ER recommendation, not a position artifact.

## Headline read for the §4.2 / §5 manuscript rewrite (auto-generated)

- **4b** (K=23): same-letter 22.4% (below chance 25%) vs same-content 64.5% (well above chance) → **content prior**. Canonical NL 55.0% → shuffled NL 69.8%; shuffled NL 69.8% vs NF 71.7% (Δ -1.9 pp).
- **12b** (K=23): same-letter 20.8% (below chance 25%) vs same-content 80.3% (well above chance) → **content prior**. Canonical NL 81.7% → shuffled NL 76.3%; shuffled NL 76.3% vs NF 71.7% (Δ +4.6 pp).
- **qwen** (K=23): same-letter 23.3% (below chance 25%) vs same-content 82.6% (well above chance) → **content prior**. Canonical NL 75.0% → shuffled NL 75.4%; shuffled NL 75.4% vs NF 68.3% (Δ +7.1 pp).

**One-paragraph summary for §4.2 (use this verbatim or adapt):**

> An option-order randomization experiment (60 cases × 23 non-identity permutations of the letter→content mapping in the forced-letter scaffold) tests whether the forced-letter accuracy depends on letter position or on letter content. Across all three models, the picked-letter is at or below chance under shuffles (case-clustered 95% CI excludes 25% at 4B and 12B), while the picked-content is far above chance (lower CI bound ≥ 56%): **no position bias, strong content prior**. At 4B, randomising the labels brings the forced-letter accuracy from 55.0% to 69.8% (95% CI [60.7%, 78.3%]), statistically indistinguishable from NF accuracy 71.7% — the entire NL→NF format penalty at 4B can be attributed to the canonical A-B-C-D letter-binding interacting with the model's content prior. At 12B and Qwen, the canonical mapping is approximately neutral or mildly helpful for accuracy, but shuffled forced-letter still beats free-text by ≈5–7 pp — at scale, free-text mode has its own accuracy penalty (the adjacent-miscalibration of §4.2 above) that is independent of letter-binding. Under exhaustive shuffles, 4B emits the letter mapped to 'Go to the ER now' content in only 2/1380 = 0.14% of shuffles, compared to 9.1% at 12B and 7.0% at Qwen — a robust capability-scaling signal at the highest acuity level.
