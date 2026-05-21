# Qwen3-8B post-adjudication tally

**Model:** Qwen/Qwen3-8B (post-trained) · n=60 cases

## Headline accuracies (for §4.1 table row)

| Cell | Accuracy |
|---|---|
| SL (forced-letter, structured)  |  75.0% (heuristic) |
| NL (forced-letter, natural)     |  75.0% (heuristic) |
| NF heuristic                    |  50.0% |
| NF GPT-5.2-thinking-high (4-way)|  75.0% |
| NF Claude-Sonnet-4.6 (4-way)    |  70.0% |
| **NF both judges correct (paper-faithful)** | ** 68.3%** |
| NF either judge correct (envelope) |  76.7% |
| **NL−NF (both-correct) gap**    | ** +6.7 pp** |

## 5-way DEFERRED rates (§4.2)

- Both judges DEFERRED (unanimous): **  3.3%** (2/60)
- Either judge DEFERRED:   6.7% (4/60)
- GPT-5.2:   6.7% · Claude-4.6:   3.3%

## Stratum counts (5-bucket schema)

- `both_right      `: 35/60 (58.3%)
- `NF_only_right   `: 6/60 (10.0%)
- `NL_only_right   `: 8/60 (13.3%)
- `both_wrong      `: 6/60 (10.0%)
- `judges_disagree `: 5/60 ( 8.3%)

## Per-acuity breakdown (most-urgent gold letter)

| Gold acuity | n | SL | NL | NF both | NF either | DEFERRED |
|---|---|---|---|---|---|---|
| A | 8 | 75% | 62% | 38% | 38% | 0% |
| B | 10 | 60% | 70% | 80% | 100% | 10% |
| C | 14 | 71% | 86% | 64% | 71% | 0% |
| D | 28 | 82% | 75% | 75% | 82% | 4% |

## §4.3 Qwen mechanistic re-stratification (L31, max-pool)

| Stratum | n | medical sMAPE | medical cosine | random sMAPE | random cosine |
|---|---|---|---|---|---|
| both_right | 35 | 0.034 | 1.000 | 0.133 | 0.988 |
| NF_only_right | 6 | 0.020 | 1.000 | 0.172 | 0.984 |
| NL_only_right | 8 | 0.025 | 0.999 | 0.171 | 0.986 |
| both_wrong | 6 | 0.000 | 1.000 | 0.161 | 0.985 |
| judges_disagree | 5 | 0.046 | 1.000 | 0.267 | 0.982 |

## Per-case table (case_id, gold, NL letter, NL correct, NF both-judges correct, stratum)

| case_id | gold | NL letter | NL✓ | GPT | Cla | NF both✓ | 5way GPT | 5way Cla | both DEF | stratum |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | C/D | A | ✗ | B | B | ✗ | B | B |   | both_wrong |
| E2 | B/C | C | ✓ | B | B | ✓ | B | B |   | both_right |
| E3 | C | C | ✓ | B | B | ✗ | B | B |   | NL_only_right |
| E4 | C | C | ✓ | B | B | ✗ | B | B |   | NL_only_right |
| E5 | A | A | ✓ | A | A | ✓ | A | A |   | both_right |
| E6 | B/C | A | ✗ | C | C | ✓ | C | C |   | NF_only_right |
| E7 | C/D | A | ✗ | C | C | ✓ | C | C |   | NF_only_right |
| E8 | A | A | ✓ | B | B | ✗ | B | B |   | NL_only_right |
| E9 | D | C | ✗ | C | C | ✗ | C | C |   | both_wrong |
| E10 | C/D | C | ✓ | D | D | ✓ | D | D |   | both_right |
| E11 | C/D | C | ✓ | D | C | ✓ | D | D |   | both_right |
| E12 | C/D | C | ✓ | C | C | ✓ | C | C |   | both_right |
| E13 | D | D | ✓ | D | D | ✓ | D | D |   | both_right |
| E14 | C/D | C | ✓ | D | D | ✓ | D | D |   | both_right |
| E15 | C/D | D | ✓ | D | D | ✓ | D | D |   | both_right |
| E16 | C/D | C | ✓ | C | C | ✓ | C | C |   | both_right |
| E17 | A | A | ✓ | B | B | ✗ | B | B |   | NL_only_right |
| E18 | B | B | ✓ | B | B | ✓ | B | B |   | both_right |
| E19 | B | A | ✗ | B | A | ✗ | B | B |   | judges_disagree |
| E20 | A | B | ✗ | B | B | ✗ | B | B |   | both_wrong |
| E21 | B | B | ✓ | B | B | ✓ | B | B |   | both_right |
| E22 | C/D | C | ✓ | C | C | ✓ | C | C |   | both_right |
| E23 | A/B | A | ✓ | B | A | ✓ | B | A |   | both_right |
| E24 | B | B | ✓ | B | B | ✓ | B | B |   | both_right |
| E25 | C | C | ✓ | C | C | ✓ | C | C |   | both_right |
| E26 | C/D | C | ✓ | C | C | ✓ | C | C |   | both_right |
| E27 | C/D | C | ✓ | D | C | ✓ | C | C |   | both_right |
| F1 | C/D | C | ✓ | B | B | ✗ | B | B |   | NL_only_right |
| F2 | B/C | C | ✓ | B | B | ✓ | B | B |   | both_right |
| F3 | C | C | ✓ | C | C | ✓ | C | C |   | both_right |
| F4 | C | C | ✓ | B | B | ✗ | B | B |   | NL_only_right |
| F5 | A | A | ✓ | A | A | ✓ | A | A |   | both_right |
| F6 | B/C | A | ✗ | C | C | ✓ | C | C |   | NF_only_right |
| F7 | C/D | A | ✗ | C | C | ✓ | DEFERRED | C |   | NF_only_right |
| F8 | A | B | ✗ | B | B | ✗ | B | B |   | both_wrong |
| F9 | D | C | ✗ | C | C | ✗ | C | C |   | both_wrong |
| F10 | C/D | A | ✗ | C | C | ✓ | B | B |   | NF_only_right |
| F11 | C/D | C | ✓ | B | B | ✗ | C | B |   | NL_only_right |
| F12 | C/D | C | ✓ | C | C | ✓ | C | C |   | both_right |
| F13 | D | C | ✗ | D | C | ✗ | C | C |   | judges_disagree |
| F14 | C/D | D | ✓ | D | C | ✓ | D | D |   | both_right |
| F15 | C/D | C | ✓ | B | C | ✗ | DEFERRED | DEFERRED | D | judges_disagree |
| F16 | C/D | C | ✓ | C | C | ✓ | C | C |   | both_right |
| F17 | A | A | ✓ | A | A | ✓ | B | A |   | both_right |
| F18 | B | B | ✓ | B | B | ✓ | B | B |   | both_right |
| F19 | B | A | ✗ | B | A | ✗ | DEFERRED | DEFERRED | D | judges_disagree |
| F20 | A | B | ✗ | B | B | ✗ | B | B |   | both_wrong |
| F21 | B | B | ✓ | B | B | ✓ | B | B |   | both_right |
| F22 | C/D | C | ✓ | C | C | ✓ | C | C |   | both_right |
| F23 | A/B | A | ✓ | A | A | ✓ | A | A |   | both_right |
| F24 | B | A | ✗ | B | B | ✓ | B | B |   | NF_only_right |
| F25 | C | C | ✓ | D | D | ✗ | D | D |   | NL_only_right |
| F26 | C/D | C | ✓ | C | C | ✓ | C | C |   | both_right |
| F27 | C/D | C | ✓ | C | C | ✓ | C | C |   | both_right |
| MH1 | C | C | ✓ | C | C | ✓ | C | C |   | both_right |
| MH2 | C/D | C | ✓ | C | C | ✓ | C | C |   | both_right |
| MH3 | C | C | ✓ | C | C | ✓ | C | C |   | both_right |
| NH1 | C | C | ✓ | C | C | ✓ | C | C |   | both_right |
| NH2 | C/D | C | ✓ | C | C | ✓ | C | C |   | both_right |
| NH3 | C | C | ✓ | C | B | ✗ | DEFERRED | B |   | judges_disagree |
