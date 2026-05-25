# 2×2 design — full SL / NL / NF / SF comparison (4-way both-judges-correct headline)

Completes the 2×2 input × output factorial. SL (structured + forced-letter) and NL (natural + forced-letter) accuracies are from heuristic letter parsing (reliable for forced-letter). NF and SF accuracies are from the paper-faithful 4-way LLM-judge adjudicator (both judges agreeing on a gold-compatible letter); DEFERRED rates are from the 5-way adjudicator.

## Headline 2×2 (4-way both-judges-correct)

| | **Forced-Letter output** | **Free-Text output** | NL−NF gap | SL−SF gap |
|---|---|---|---|---|
| **4b structured** | SL: 58.3% | SF: 63.3% | – | -5.0 pp |
| **4b natural**    | NL: 55.0% | NF: 71.7% | -16.7 pp | – |
| **12b structured** | SL: 81.7% | SF: 73.3% | – | +8.3 pp |
| **12b natural**    | NL: 81.7% | NF: 71.7% | +10.0 pp | – |
| **qwen structured** | SL: 75.0% | SF: 70.0% | – | +5.0 pp |
| **qwen natural**    | NL: 75.0% | NF: 68.3% | +6.7 pp | – |

## Side-by-side comparison

| Model | SL | NL | NF (4-way both) | SF (4-way both) | NF unanim DEFER | SF unanim DEFER |
|---|---|---|---|---|---|---|
| 4b | 58.3% | 55.0% | 71.7% | 63.3% | 0/60 (0.0%) | 4/60 (6.7%) |
| 12b | 81.7% | 81.7% | 71.7% | 73.3% | 4/60 (6.7%) | 2/60 (3.3%) |
| qwen | 75.0% | 75.0% | 68.3% | 70.0% | 2/60 (3.3%) | 0/60 (0.0%) |

## Headline read (auto-generated)

- **4b**: cell ranking = NF(72%) > SF(63%) > SL(58%) > NL(55%). NL−NF = -16.7 pp, SL−SF = -5.0 pp. DEFERRED rates: NF 0/60, SF 4/60.
- **12b**: cell ranking = SL(82%) > NL(82%) > SF(73%) > NF(72%). NL−NF = +10.0 pp, SL−SF = +8.3 pp. DEFERRED rates: NF 4/60, SF 2/60.
- **qwen**: cell ranking = SL(75%) > NL(75%) > SF(70%) > NF(68%). NL−NF = +6.7 pp, SL−SF = +5.0 pp. DEFERRED rates: NF 2/60, SF 0/60.

## Reading guide for §4.1 / §5 (the 2×2 interpretation)

Two key cross-cuts:

**(a) Forced-letter vs Free-text within the same input type.** Does removing the forced-letter constraint help, hurt, or wash?
- NL → NF (natural input): tells us whether the canonical NL→NF gap (documented in §4.1) is driven by the output-format constraint.
- SL → SF (structured input): same question, structured input. If the gap goes the same direction in both rows, the format effect is robust across input style.

**(b) Structured vs Natural within the same output type.** Does patient-voice vs clinician-notes-style affect accuracy?
- SL → NL: forced-letter only, isolated input effect.
- SF → NF: free-text only, isolated input effect.

Together these decompose the variance: format effect (rows), input effect (columns), and their interaction (= the 4-corner residual).