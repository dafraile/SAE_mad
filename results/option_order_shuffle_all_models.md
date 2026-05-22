# Option-order shuffle — cross-model summary (4B / 12B / Qwen)

Falsifiable test of position-bias vs content-prior at the forced-letter scaffold. For each of the 60 canonical cases, K=3 random non-identity permutations of the letter→content mapping; greedy forced-letter generation; score same-letter, same-content, accuracy under shuffle.

## Stability + accuracy

| Model | n | K | same-letter % (chance ≈25%) | same-content % (chance ≈25%) | canonical NL acc | shuffled NL acc | NF (4-way both) | shuffled→NF gap |
|---|---|---|---|---|---|---|---|---|
| 4b | 60 | 3 | 21.1% | 67.2% | 55.0% | 71.7% | 71.7% | +0.0 pp |
| 12b | 60 | 3 | 25.0% | 80.6% | 81.7% | 78.9% | 71.7% | +7.2 pp |
| qwen | 60 | 3 | 25.6% | 82.2% | 75.0% | 72.8% | 68.3% | +4.4 pp |

Interpretation: high same-letter % → position bias; high same-content % → content prior. The shuffled-NL-vs-NF gap tells us whether option-order randomization 'erases' the forced-letter mode's letter-binding artifact (gap ≈ 0 pp) or only partially (gap > 0).

## Letter distribution (canonical NL vs shuffled NL)

| Model | NL canonical | NL shuffles |
|---|---|---|
| 4b | A:3 B:32 C:25 D:0 | A:60 B:41 C:38 D:41 |
| 12b | A:3 B:17 C:30 D:10 | A:52 B:26 C:43 D:59 |
| qwen | A:16 B:8 C:33 D:3 | A:67 B:28 C:39 D:46 |

## Content distribution (canonical NL vs shuffled NL)

Shows which acuity content the model picks (regardless of which letter that content is assigned to). Under shuffles, a content prior shows up here as concentration on one row.

| Model | Canonical: Fine / Weeks / 24-48h / ER | Shuffles: Fine / Weeks / 24-48h / ER |
|---|---|---|
| 4b | 3 / 32 / 25 / 0 | 4 / 49 / 127 / 0 |
| 12b | 3 / 17 / 30 / 10 | 6 / 42 / 117 / 15 |
| qwen | 16 / 8 / 33 / 3 | 38 / 19 / 108 / 15 |

## Headline read (auto-generated)

- **4b**: same-letter 21.1% vs same-content 67.2% → **content prior**. Shuffled NL acc ↑ 16.7 pp vs canonical NL (55.0% → 71.7%); shuffled NL 71.7% vs NF 71.7% (Δ +0.0 pp).
- **12b**: same-letter 25.0% vs same-content 80.6% → **content prior**. Shuffled NL acc ↓ 2.8 pp vs canonical NL (81.7% → 78.9%); shuffled NL 78.9% vs NF 71.7% (Δ +7.2 pp).
- **qwen**: same-letter 25.6% vs same-content 82.2% → **content prior**. Shuffled NL acc ↓ 2.2 pp vs canonical NL (75.0% → 72.8%); shuffled NL 72.8% vs NF 68.3% (Δ +4.4 pp).
