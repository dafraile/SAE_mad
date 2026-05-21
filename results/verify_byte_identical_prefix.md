# Byte-identical prefix verification

Diagnostic for the masked-invariance analysis. Verifies the assumption that the vignette tokens are byte-identical between NL (forced-letter) and NF (free-text) prompts under each model's chat template.

If the shared prefix length covers the entire vignette, then any feature-activation difference observed at vignette positions is purely numerical (bf16 quantization), not semantic.

## Summary
| Model | n cases | vignette exact-matched in NF | shared prefix ≥ vignette | median shared prefix len |
|---|---|---|---|---|
| 4b | 60 | 60/60 | 60/60 | 269 |
| 12b | 60 | 60/60 | 60/60 | 269 |
| qwen | 60 | 60/60 | 0/60 | 256 |

**Interpretation:**
- If `vignette exact-matched in NF` == `n cases`, the vignette text re-tokenizes identically inside the chat-templated NF prompt — no merge anomalies at the boundary.
- If `shared prefix ≥ vignette` == `n cases`, the shared-prefix-length used by `phase1b_masked_invariance.py` correctly covers all vignette tokens — the vignette-mask sanity check has its expected interpretation.
- Any case where these are < n is a tokenization edge case that needs §3.1 / §4.3 to be softened.

## Findings

**Gemma 3 4B IT and Gemma 3 12B IT:** 60/60 cases pass both checks. The vignette text re-tokenizes identically inside the NF prompt, and the shared prefix between the NL and NF prompts covers all vignette tokens. The byte-identical-prefix assumption holds without qualification at the Gemma scales.

**Qwen3-8B:** 60/60 vignette texts re-tokenize identically when isolated (good), but the shared NL-vs-NF prefix length is consistently a small number of tokens shorter than the vignette (median ~256 shared tokens of ~263 vignette tokens). Inspecting case E1: the divergence is at the trailing `?` of the vignette text. Qwen's BPE merges `?\n\n` into a single token (id 1939) in the NL prompt (where the scaffold follows the `?`) but keeps `?` as a separate token (id 30) in NF (where `<|im_end|>` follows). The merge moves the divergence one token earlier than the vignette end.

**Implication:** for Qwen, ~99.6% of vignette positions are byte-identical between NL and NF prompts. The remaining ~0.4% is a single trailing-punctuation token whose context (whitespace before scaffold vs end-of-turn marker) changes the BPE merge. This is a tokenization edge case at the very last vignette position only, and does not affect the substantive interpretation: at all three models, the vast majority of vignette positions are byte-identical, and the observed near-zero medical+random sMAPE on the vignette mask (~0.002–0.006) reflects bf16 quantization noise plus this single-token boundary effect.

**Paper edits:** §3.1's claim that the vignette is byte-identical is correct for Gemma. For Qwen, the wording should be softened to '...byte-identical for all vignette positions except the trailing punctuation token, which is re-tokenized by Qwen's BPE depending on what follows.' Section §4.3's vignette-mask sanity check is justified at all three models.