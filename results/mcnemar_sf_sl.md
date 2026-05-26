# McNemar paired test on SF − SL gap (per model)

Parallel to the NF−NL McNemar reported in §4.1. SL correctness is the heuristic forced-letter parse on the structured prompt. SF correctness is the both-judges-correct verdict under 4-way LLM-judge adjudication on the free-text response to the same structured input.

## Headline

| Model | n | SL | SF | SF−SL | b (SL✓ SF✗) | c (SL✗ SF✓) | n_discordant | McNemar exact p (two-sided) | 95% CI (paired, pp) |
|---|---|---|---|---|---|---|---|---|---|
| Gemma 3 4B IT | 60 | 58.3% | 63.3% | +5.0pp | 7 | 10 | 17 | **0.6291** | [-8.3, +18.3] |
| Gemma 3 12B IT | 60 | 81.7% | 73.3% | -8.3pp | 7 | 2 | 9 | **0.1797** | [-18.3, +0.0] |
| Qwen3-8B | 60 | 75.0% | 70.0% | -5.0pp | 8 | 5 | 13 | **0.5811** | [-16.7, +6.7] |

## Drop-in §4.1 sentence (extending the existing NF−NL sentence)

> "...$p{=}0.031$ at both Gemma scales and $p{=}0.45$ at Qwen ($n.s.$ at $n{=}60$). The same test on the SF$-$SL gap gives $p{=}0.629$ at 4B, $p{=}0.180$ at 12B, $p{=}0.581$ at Qwen3-8B."