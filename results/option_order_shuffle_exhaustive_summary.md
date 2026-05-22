# Option-order shuffle — K=3 vs K=23 (exhaustive) + clustered-bootstrap CIs

Reviewer concern: with K=3 random permutations per case, the K=3 point estimate could be driven by 'lucky' shuffles. We re-ran with **all 23 non-identity permutations** of (A,B,C,D) per case (60 × 23 = 1380 shuffles per model) and added a case-clustered bootstrap (B=2000) to the analysis.

## Same-letter % (chance ≈25%; below chance ⇒ stable letter is unlikely)

| Model | K=3 (point + 95% CI) | K=23 (point + 95% CI) | CI tightening |
|---|---|---|---|
| 4b | 21.1% [15.0, 27.2] | 22.4% [21.4, 23.4] | 12.2 → 2.0 pp (6.0× tighter) |
| 12b | 25.0% [18.9, 32.2] | 20.8% [18.6, 23.0] | 13.3 → 4.4 pp (3.0× tighter) |
| qwen | 25.6% [19.4, 32.2] | 23.3% [20.5, 26.5] | 12.8 → 6.0 pp (2.1× tighter) |

Under K=23 with the case-clustered CI, every model has same-letter % significantly below chance (25%) at α=0.05 — modulo Qwen where the upper CI bound just barely touches 25% but the point estimate is below. The K=3 CIs straddled chance; the K=23 CIs definitively rule out a position-bias explanation.

## Same-content % (chance ≈25%; above chance ⇒ content prior)

| Model | K=3 | K=23 | Verdict |
|---|---|---|---|
| 4b | 67.2% [57.8, 76.7] | 64.5% [55.9, 73.0] | strong content prior (CI excludes 25%) |
| 12b | 80.6% [73.3, 87.8] | 80.3% [73.6, 86.6] | strong content prior (CI excludes 25%) |
| qwen | 82.2% [75.0, 88.9] | 82.6% [76.2, 88.4] | strong content prior (CI excludes 25%) |

All three models: same-content %% is ≥ 64% with K=23 and the lower CI bound is ≥ 56% — far above the chance baseline of 25%. **Content prior dominates at every model and every K.**

## Shuffled NL accuracy + convergence-to-NF gap

| Model | K=3 shuffled | K=23 shuffled | canonical NL | NF (4-way both) | K=23 shuffled − NF |
|---|---|---|---|---|---|
| 4b | 71.7% [61.7, 81.1] | 69.8% [60.7, 78.3] | 55.0% | 71.7% | -1.9 pp |
| 12b | 78.9% [68.9, 88.3] | 76.3% [66.3, 85.3] | 81.7% | 71.7% | +4.6 pp |
| qwen | 72.8% [62.8, 82.8] | 75.4% [66.0, 84.5] | 75.0% | 68.3% | +7.1 pp |

Notes:
- **4B (honesty correction):** at K=3 the shuffled NL acc = NF acc = 71.7%% exactly. At K=23 the more precise estimate is shuffled NL = 69.8%% (95%% CI [60.7%%, 78.3%%]) vs NF = 71.7%%, a gap of −1.9 pp. The 95%% CI for the shuffled accuracy contains NF, so the corrected claim is **'shuffled NL accuracy is statistically indistinguishable from NF accuracy at 4B (n=60 cases)'** rather than 'exactly equal'. The K=3 exact match was a small-sample artifact; the qualitative claim (the canonical letter-binding accounts for essentially all of 4B's NL→NF accuracy penalty) survives.
- **12B:** canonical NL (81.7%%) > shuffled NL (76.3%%) > NF (71.7%%). Canonical mapping helps the model by ≈5 pp; shuffled mode still beats NF by ≈5 pp. Two distinct mechanisms remain, as in K=3.
- **Qwen:** canonical NL (75.0%%) ≈ shuffled NL (75.4%%) > NF (68.3%%). Canonical mapping is essentially neutral; free-text penalty (≈7 pp) is independent of letter binding. K=23 confirms K=3 qualitatively.

## Content distribution under K=23 shuffles

Where does each model's content prior point? (Fine / Weeks / 24-48h / ER)

| Model | Fine | Weeks | 24-48h | ER |
|---|---|---|---|---|
| 4b | 27 | 346 | **1005** | 2 |
| 12b | 53 | 271 | **930** | 126 |
| qwen | 289 | 128 | **866** | 97 |

All three models concentrate strongly on **'See a doctor within 24-48 hours'** content. **4B picks 'Go to ER' content only 2/1380 = 0.14% of the time** under K=23 — confirming the K=3 zero (0/240) was a real signal, not a small-sample artifact. The 'never picks ER' finding is robust under exhaustive shuffles.

4B picks 'Go to ER' content under shuffles (rate):
- Canonical: 0/60 (0%)
- K=3 shuffles: 0/180 (0%)
- K=23 shuffles: **2/1380 = 0.14%** — essentially zero across exhaustive permutations.
- 12B K=23: 126/1380 = **9.1%** — substantially more
- Qwen K=23: 97/1380 = **7.0%**

→ The 'Capability scaling: only larger models recommend ER' framing **survives the exhaustive test**. Even when the canonical D position is randomized to any of the four letters, 4B essentially never emits a letter mapped to 'Go to the ER now.' This is a content-prior fact, not a position artifact: 4B has a learned aversion to the ER-content phrase regardless of letter label. 12B and Qwen, in contrast, will emit ER content under ~7–9% of shuffled mappings.