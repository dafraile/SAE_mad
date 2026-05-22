# LaTeX-writer handoff — v3 (post-reviewer-concerns, 2026-05-21)

This package collects everything the LaTeX-writer needs to do a single
consolidated revision pass on PAPER_DRAFT.md before ARR submission.

The revisions address an external reviewer's six major concerns. Three
of them (deferral-as-driver, prefix-token invariance, deployable-monitor
overclaim) were paper-killers as written; the others are credibility
tightening. The data work for the first two is complete (see “What we
ran today,” below); the third is a framing pass.

There is **no new GPU run pending** for v3 — the K-sweep re-run with
max-pool sMAPE (reviewer Concern 4c) is the only remaining GPU task and
can be done later. Sections that depended on K-sweep numbers can ship
with a "v3 update pending" footnote pointing to the current Appendix
A1B table, marked as mean-pool.

---

## 1. Headline narrative changes (do these before any other edit)

### 1a. Deferral does NOT drive the 12B inversion

**Old claim** (currently in §4.2, §1 contribution iii, abstract, §5):
> The 12B NL > NF inversion is driven by deferral.

**Why it's wrong:** all 4 unanimous DEFERRED cases at 12B (F15, F19,
F23, F24) happen to flatten to gold-compatible letters under 4-way
scoring (F15→C∈C/D, F19→B∈B, F23→B∈A/B, F24→B∈B). They are counted
**correct** by both judges and live in the `both_right` stratum, not
`NL_only_right`. They contribute **zero** to the measured accuracy gap.

**Replace with:** two distinct phenomena, not one.
> The accuracy gap (in either direction) is dominated by **single-acuity-step miscalibrations** where the two formats disagree on the gold letter by one step (B↔C or C↔D). Deferral is a separate, smaller benchmark-adequacy phenomenon: the A/B/C/D label space cannot represent clinical hedges, but the unanimous deferrals at 12B happen to flatten to gold-compatible letters under 4-way scoring and therefore do not contribute to the accuracy inversion.

This is **paper-positive**: the output-mapping-artifact hypothesis is
strictly stronger under "model commits to a clinically reasonable
adjacent letter in one format but not the other" than under "model
refuses to commit."

### 1b. Prefix-token invariance is now properly contextualised

**Reviewer Concern 1:** under causal masking, hidden states at shared
clinical-prefix tokens are byte-identical between B and D. Pooling
over those tokens gives a trivially-zero sMAPE component that
dominates the headline number.

**What we did:** re-ran Phase 1b with explicit token masks (vignette /
scaffold / decision-token) on all three models and resampled the random
pool 1000× under magnitude-matched controls. See `phase1b_masked_*.json`
and `phase1b_random_pool_resample_*.json`.

**Punch line:**
1. **Vignette mask sanity check passes** — both medical and random
   sMAPE ≈ 0.002–0.006 on the shared prefix in all three models.
   Confirms the reviewer's expected baseline. (Causal masking working
   as advertised.)
2. **Medical features peak in the shared vignette content in 81–100%
   of (case × feature) combinations across all three models** — the
   clean interpretation: medical-feature *peaks* are anchored in the
   clinical narrative, so the full max-pool sMAPE being ~0 for medical
   reflects content anchoring, not pooling-trivial identity.
3. **Full-content max-pool medical-vs-random gap survives
   magnitude-matched random resampling** with permutation p < 0.012
   across all three models (p < 0.001 at 4B and 12B). The original
   fixed-seed random baseline understated the gap because it included
   zero-firing features whose sMAPE is artificially zero by denominator
   floor.

### 1c. "Deployable monitor" walks back to "candidate readout"

**Reviewer Concern 6:** no calibration, no held-out clinical/nonclinical
benchmark, no robustness study. The "deployable monitor of clinical
groundedness" claim isn't supported.

**Replace throughout:**
- "deployable monitor" → "candidate readout, requires prospective
  clinical validation"
- "deployable" → delete the word, qualify the claim instead

### 1d. "Cross-family confirmation" softens to "suggestive cross-family consistency"

**Why:** Qwen has higher SAE reconstruction error, only one analyzed
layer, smaller behavioral gap, and only 2 unanimous deferrals. The
mechanistic *pattern* reproduces (medical features invariant,
adjacent-miscalibration driving the gap) but the strength of the
single Qwen data point is below what "confirmation" connotes.

**Replace:** "cross-family confirmation" → "suggestive cross-family
consistency"

---

## 2. Numbers to use in tables and prose

### 2a. §4.1 Behavioral phenomenon table — full updated row set

Source: `results/gap_decomposition.md` and `results/phase4b_qwen_post_adjudication_summary.md`.

| Model | SL | NL | NF (heuristic) | NF (4-way both-judges) | NF (either judge) | Gap NL−NF (both) |
|---|---|---|---|---|---|---|
| Gemma 3 4B IT  | 50.0% | 55.0% | 60.0% | **71.7%** | 76.7% | **−16.7 pp** |
| Gemma 3 12B IT | 80.0% | 81.7% | 70.0% | **71.7%** | 81.7% | **+10.0 pp** |
| Qwen3-8B       | 75.0% | 75.0% | 50.0% | **68.3%** | 76.7% | **+6.7 pp** |

**Paired NL vs NF inference (Bucket A — added 2026-05-22):**

| Model | 95% paired-boot CI on NL−NF | McNemar exact two-sided p |
|---|---|---|
| 4B  | [−30.0, −3.3] pp | **0.031** (sig) |
| 12B | [+3.3, +18.3] pp | **0.031** (sig) |
| Qwen| [−6.7, +20.0] pp | 0.45 (**not** sig) |

→ The Qwen gap is not statistically significant on the paired McNemar with n=60; **this is the empirical reason to soften to "suggestive cross-family consistency"** rather than "confirmation." 4B and 12B gaps are both significant at α=0.05 by McNemar exact.

Source files for replication:
- 4B: `results/_v2/phase0_5_three_cells.json`, `results/_v2/phase0_5_D_for_adjudication_adjudicated_paper.json`, `results/_v2/phase0_5_adjudicated_deferred.json`
- 12B: same pattern with `phase3b_12b_*`
- Qwen: `results/phase4b_qwen_*` (committed today in `2f58f9c`)

### 2b. §4.2 Deferral / stratum decomposition

| Model | Unanim DEFERRED (5-way) | NF_only_right | NL_only_right | both_wrong | judges_disagree | both_right |
|---|---|---|---|---|---|---|
| 4B   | 0/60 | 14 (all adjacent) | 1 (adjacent) | 12 | 4 | 29 |
| 12B  | **4/60 (all in `both_right`)** | 0 | 6 (5 adjacent, 1 non-adj) | 11 | 0 | 43 |
| Qwen | **2/60 (both in `judges_disagree`)** | 6 (1 adj, 5 non-adj) | 8 (all adjacent) | 6 | 5 | 35 |

**Where the deferrals land:**
- 12B: all 4 unanim DEFERRED flatten to gold-compatible letters → counted correct → live in `both_right` → contribute 0 to the gap
- Qwen: 2 unanim DEFERRED produce 4-way judge disagreement → live in `judges_disagree` → contribute partially to the gap
- 4B: 0 unanim DEFERRED

**Adjacency dominates (corrected 2026-05-22 after reviewer audit):** **14/15 NL_only_right cases and 15/20 NF_only_right cases across all three models are single-acuity-step miscalibrations.** Earlier drafts said "19/20 NL_only_right" — this was a stale tally from before the Qwen numbers landed. Source of truth: `results/gap_decomposition.json`. See V3_CHANGES.md for the single-table source-of-truth tally and the formal adjacency definition.

**Triage-direction breakdown (Bucket A — added 2026-05-22):**

| Model | Format | Under-triage | Correct | Over-triage |
|---|---|---|---|---|
| 4B   | NL | **20** | 33 | 7 |
| 4B   | NF | 5 | 43 | 8 |
| 12B  | NL | 4 | 49 | 7 |
| 12B  | NF | 8 | 43 | 9 |
| Qwen | NL | **12** | 45 | 3 |
| Qwen | NF | 8 | 37 | 6 |

→ **Forced-letter mode systematically under-triages at 4B and Qwen** (20 and 12 cases respectively, vs. 7 and 3 over-triage). Free-text mode is more balanced at all three models. Under-triage is the clinically dangerous direction.

**Per-acuity gradient (added 2026-05-22):**
NF accuracy on **A-bucket (monitor-at-home)** cases is the weakest at every model: 4B 0/8, 12B 2/8, Qwen 3/8. NF over-triages low-acuity cases. Worth flagging in §6 limitations as a benchmark-design observation (the A-bucket is the smallest gold sample, n=8).

**4B singleton-D failure (reviewer-flagged, confirmed 2026-05-22):**
4B predicts D = **0/9** on D-only-gold cases in NL forced-letter mode; NF only 1/9. The model essentially never picks "emergency" as a singleton answer in forced-letter mode — clinically concerning and worth a sentence in §6.

### 2c. §4.3 Mechanistic invariance — new table with magnitude-matched controls

Source: `results/phase1b_masked_invariance_*.json` and `results/phase1b_random_pool_resample_*.json`.

**Vignette-mask sanity check** (shared content tokens, expected ≈0):

| Model | Medical sMAPE | Random (mag-matched) sMAPE | perm-p |
|---|---|---|---|
| 4B L29   | 0.0038 | 0.0061 | <0.001 |
| 12B L31  | 0.0027 | 0.0054 | <0.001 |
| Qwen L31 | 0.0018 | 0.0031 | <0.001 |

**Full-content max-pool (headline)**:

| Model | Medical sMAPE | Random (mag-matched) sMAPE 5–95% | perm-p |
|---|---|---|---|
| 4B L29   | **0.0040** | 0.2757 (5–95% = [0.138, 0.429]) | <0.001 |
| 12B L31  | **0.0027** | 0.1232 (5–95% = [0.072, 0.171]) | <0.001 |
| Qwen L31 | **0.0261** | 0.1281 (5–95% = [0.048, 0.226]) | 0.012 |

**Medical-feature peaks inside the shared vignette mask**:

| Model | B (NL) prompt | D (NF) prompt |
|---|---|---|
| 4B   | 99.4% | 99.4% |
| 12B  | 98.3% | 100.0% |
| Qwen | 81.1% | 100.0% |

### 2d. Qwen per-stratum mechanistic re-stratification (§4.3 new Qwen panel)

Source: `results/phase4b_qwen_post_adjudication_summary.md`.

| Stratum | n | medical sMAPE | medical cosine | random sMAPE | random cosine |
|---|---|---|---|---|---|
| both_right       | 35 | 0.034 | 1.000 | 0.133 | 0.988 |
| NF_only_right    |  6 | 0.020 | 1.000 | 0.172 | 0.984 |
| NL_only_right    |  8 | 0.025 | 0.999 | 0.171 | 0.986 |
| both_wrong       |  6 | 0.000 | 1.000 | 0.161 | 0.985 |
| judges_disagree  |  5 | 0.046 | 1.000 | 0.267 | 0.982 |

### 2e. Appendix A1B — K-sensitivity under max-pool sMAPE (Bucket B, added 2026-05-22)

Reviewer Concern 4c: the current Appendix A1B reports **mean-pool** sMAPE while the main §4.2 table reports **max-pool**. Re-run with max-pool for consistency.

Source: `results/phase1b_sensitivity_maxpool.json` and `.md`. Max-pool sMAPE, top-K medical features from the original contrastive ID, K-matched random pool from the same source. Bootstrap 1000 resamples over the 60 cases; paired delta = medical − random per case.

**Gemma 3 4B IT L29:**

| K | medical sMAPE (mean, 95% CI) | random sMAPE (mean, 95% CI) | Δ_paired (95% CI) | sig? |
|---|---|---|---|---|
| 3  | 0.0063 [0.003, 0.012] | 0.0312 [0.021, 0.044] | −0.0249 [−0.038, −0.013] | ✓ |
| 5  | 0.0115 [0.003, 0.027] | 0.2285 [0.182, 0.278] | −0.2170 [−0.268, −0.168] | ✓ |
| 10 | 0.1329 [0.097, 0.169] | 0.2601 [0.217, 0.306] | −0.1272 [−0.163, −0.094] | ✓ |
| 20 | 0.1531 [0.131, 0.176] | 0.2139 [0.173, 0.256] | −0.0608 [−0.089, −0.034] | ✓ |

**Gemma 3 12B IT L31:**

| K | medical sMAPE (mean, 95% CI) | random sMAPE (mean, 95% CI) | Δ_paired (95% CI) | sig? |
|---|---|---|---|---|
| 3  | 0.0058 [0.003, 0.010] | 0.7414 [0.648, 0.833] | −0.7356 [−0.825, −0.643] | ✓ |
| 5  | 0.0053 [0.004, 0.008] | 0.4549 [0.397, 0.512] | −0.4496 [−0.506, −0.392] | ✓ |
| 10 | 0.0389 [0.023, 0.059] | 0.5920 [0.553, 0.639] | −0.5531 [−0.600, −0.507] | ✓ |
| 20 | 0.0871 [0.071, 0.105] | 0.4120 [0.382, 0.446] | −0.3249 [−0.363, −0.291] | ✓ |

**Qwen3-8B L31 (K=3 only; full top-20 contrastive ID is future work):**

| K | medical sMAPE | random sMAPE | Δ_paired | sig? |
|---|---|---|---|---|
| 3 | 0.0348 [0.026, 0.045] | 0.0579 [0.013, 0.103] | −0.0231 [−0.072, +0.024] | ns |

**Key findings:**
- At every K ∈ {3, 5, 10, 20}, medical features show significantly lower sMAPE than the K-matched random pool, at both Gemma scales (paired-bootstrap 95% CI on Δ excludes 0).
- 4B shows K-monotonicity: medical sMAPE grows from 0.006 (K=3) to 0.15 (K=20). The top 3 medical features are by far the most invariant; broadening to the top-20 set dilutes the effect (because lower-ranked contrastive features fire less reliably).
- 12B is rock-solid invariant across all K (medical sMAPE 0.006–0.09 vs random 0.41–0.74). The 12B medical-feature subspace is the cleanest.
- Qwen K=3 is the weakest result; the random pool there happens to be very invariant too (rnd = 0.058 vs med = 0.035). Consistent with the masked-invariance morning finding that the Qwen full-content gap is the smallest (med 0.026, rnd-mag-matched 0.128, perm-p 0.012). The K-sweep random pool definition here is tighter (mean-pool magnitude band) than the masked-invariance perm-test random pool (max-pool magnitude band) — the two analyses use different controls and report consistent direction but different magnitudes. This is a methods-section clarification we should add.

**Replacement for Appendix A1B caption / text:** the existing K-sweep numbers (medical 0.188, random 0.222 at K=3 for 4B) are MEAN-POOL and inconsistent with the main text's max-pool. Replace with the table above. Note in the caption: "All sMAPE values computed under max-pool aggregation, matching the main §4.2 table; the original mean-pool K-sweep is retained at `results/phase1b_sensitivity_4b_L29.json` / `_12b_L31.json` for reference."

### 2f. The 4B "NL=B when gold=C" pattern (callout for §4.2, added 2026-05-22)

Source: `results/gap_decomposition.md`. Buried inside §2b above is a much sharper claim than "single-notch miscalibration" for 4B specifically.

Of the **14 NF_only_right cases** at 4B (the cases driving the −16.7 pp NF > NL gap):
- **13 of 14 have NL letter = B AND gold letter = C** (the same exact pattern repeated)
- The 14th has NL = A and gold = B (still adjacent, still under-triage)

So at 4B, the forced-letter mode doesn't just *miscalibrate adjacently* — it has a **systematic B-preference when gold is C**. The model defaults to "see a doctor in the next few weeks" (B) when the gold answer is "see a doctor within 24–48 hours" (C).

This is a specific, testable artifact rather than a vague format penalty. The §4.2 rewrite should call it out by name:

> "At 4B, the NF > NL accuracy gap is dominated by a systematic 'NL = B when gold = C' miscalibration: 13 of 14 NF_only_right cases follow exactly this pattern. The model under-triages by one acuity step in forced-letter mode but produces the correct urgency in free-text on the same vignette."

This pattern motivates a follow-up experiment (option-order randomization) flagged in §5 as the cleanest way to distinguish a position artifact ("model always picks position-2") from a content artifact ("model has a learned 'when in doubt, say B' prior"). We have not run this experiment yet; both readings are consistent with the output-mapping-artifact hypothesis.

### 2g. Title and §3.1 wording (added 2026-05-22)

**Title walk-back.** If the current paper title contains "clinical reasoning preserved" or similar, soften per Concern 2. Working suggestions, ordered:
- "Format-Robust Medical-Content Representation in LLM Triage: A Cross-Family SAE Analysis"
- "Apparent Triage Failures as Output-Mapping Artefacts: Mechanistic Evidence from SAE Analysis at Three Model Scales"
- "What Gets Lost When LLMs Triage With Letters: Medical-Content Representation Survives Format Changes Across Three Models"

Avoid "clinical reasoning preserved" / "deployable monitor" in the title.

**§3.1 vignette wording.** "Byte-identical clinical content between NL and NF prompts" holds without qualification at Gemma 4B and 12B but not at Qwen — Qwen's BPE merges the trailing punctuation token of the vignette differently depending on what follows. See `results/verify_byte_identical_prefix.md` for the per-model diagnostic.

Suggested phrasing for §3.1:

> "The NL and NF prompts for a given case share the patient vignette text verbatim and differ only by an appended forced-letter scaffold in NL. Under causal masking, hidden states at vignette token positions cannot depend on later scaffold tokens, so SAE feature activations at vignette positions are byte-identical between NL and NF at the Gemma scales. At Qwen3-8B the BPE merges the trailing punctuation differently depending on what follows the vignette, which shifts the divergence one token earlier than the vignette end; the remaining ~99.6% of vignette tokens are byte-identical at Qwen too."

### 2h. Cross-model option-order shuffle (4B + 12B + Qwen) (added 2026-05-22)

Source: `results/option_order_shuffle_{4b,12b,qwen}.json` + `option_order_shuffle_all_models.{json,md}` (script `paper/scripts/option_order_shuffle_4b.py`, generalized to take `--model`).

Falsifiable test of the "letter-binding interacts with content prior" hypothesis. For each of the 60 canonical cases, we generated K=3 random non-identity permutations of the letter→content mapping in the forced-letter scaffold and ran each model greedy forced-letter. Total 180 shuffled predictions per model.

**Cross-model headline table:**

| Signal | 4B | 12B | Qwen |
|---|---|---|---|
| Same-letter % (chance ≈25%) | **21.1%** | 25.0% | 25.6% |
| Same-content % (chance ≈25%) | **67.2%** | **80.6%** | **82.2%** |
| Canonical NL acc | 55.0% | 81.7% | 75.0% |
| **Shuffled NL acc** | **71.7%** | **78.9%** | **72.8%** |
| NF (4-way both judges) acc | 71.7% | 71.7% | 68.3% |
| Shuffled − canonical NL | **+16.7 pp** | −2.8 pp | −2.2 pp |
| **Shuffled − NF (convergence gap)** | **+0.0 pp (EXACT)** | +7.2 pp | +4.4 pp |

**Three big findings:**

1. **No position bias at any model.** Same-letter % is at chance everywhere (21–26%). Strong content prior at all three (67–82%).

2. **At 4B, the entire NL→NF format penalty IS the canonical letter-binding.** Shuffled NL accuracy = NF accuracy = 71.7% *to the case*. Randomize the letter assignment and forced-letter accuracy collapses *exactly* to free-text. This is a remarkably clean, falsifiable mechanistic claim.

3. **At 12B and Qwen, canonical letter-binding mildly helps the model.** Canonical accuracy beats shuffled by +2.8 and +2.2 pp respectively. But shuffled forced-letter still beats free-text by +7.2 pp (12B) and +4.4 pp (Qwen) — meaning free-text mode at scale has its own accuracy penalty (the *adjacent-miscalibration in NF* of §2b) that is **separate from letter-binding**.

**Letter distribution (canonical vs shuffled) shows the content-prior:**

| Model | Canonical letter dist | Shuffled letter dist |
|---|---|---|
| 4B   | A:3 B:32 C:25 **D:0**  | A:60 B:41 C:38 D:41 |
| 12B  | A:3 B:17 C:30 D:10     | A:52 B:26 C:43 **D:59** |
| Qwen | A:16 B:8 C:33 D:3      | A:67 B:28 C:39 D:46 |

**Shuffled content distribution (which acuity content is picked, regardless of letter):**

| Model | Fine | Weeks | 24-48h | ER |
|---|---|---|---|---|
| 4B   |   4 |   49 | **127** |  **0** |
| 12B  |   6 |   42 | **117** | 15 |
| Qwen |  38 |   19 | **108** | 15 |

All three models concentrate strongly on "See a doctor within 24-48 hours" content. **4B never picks "Go to ER now"** in any of its 240 forced-letter predictions across canonical + shuffled (capability-scaling signal — 12B and Qwen DO pick ER, though rarely).

**Unified mechanistic claim for the §4.2 rewrite:**

> "Two distinct sources of format-dependent accuracy emerge from the cross-model option-order shuffle:
>   (a) **canonical letter-binding × content-prior interaction** — scale-dependent: at 4B the canonical A-B-C-D mapping costs 16.7 pp; at 12B it gains 2.8 pp; at Qwen 2.2 pp.
>   (b) **NF-mode adjacent-miscalibration** — present at 12B and Qwen but not at 4B (where shuffled NL exactly matches NF).
>
> Across all three models, NONE show a position bias (same-letter % ≈ chance); ALL show a strong content prior (same-content % 67–82%). Under random letter labels the three models' accuracies converge to a narrow 72–79% range, suggesting the canonical NL accuracy spread (55–82%) is largely an artifact of how well each model's content prior aligns with the canonical A-B-C-D acuity ordering."

**Paper move:** the §4.2 rewrite should call this out across all three models. The "format penalty" is no longer a vague concept — at every scale, position bias is ruled out (same-letter % ≈ chance) and content prior is confirmed (same-content % 67–82%). The canonical letter-binding's effect on accuracy is scale-dependent and not always negative. Suggested wording:

> "An option-order randomization experiment (60 cases × 3 random non-identity permutations of the letter→content mapping) tests whether the forced-letter accuracy depends on position or on content. Across all three models, the picked-letter is at chance under shuffles (21–26%, vs same-content 67–82%): no position bias, strong content prior. At 4B, randomising the labels recovers the free-text accuracy to the case (shuffled NL 71.7% = NF 71.7%) — the entire NL→NF format penalty at 4B is the canonical A-B-C-D letter-binding interacting with the model's 'see-a-doctor-within-24-48h' content prior. At 12B and Qwen the canonical mapping mildly *helps* the model (canonical accuracy beats shuffled by 2–3 pp), but shuffled forced-letter still beats free-text by 4–7 pp — at scale, free-text mode has its own accuracy penalty (the adjacent-miscalibration documented above) that is separate from letter-binding. The convergence of shuffled accuracies (72–79%) across scales is in itself informative: the canonical NL accuracy spread (55–82%) is partly an artifact of how well each model's content prior aligns with the canonical acuity ordering."

### 2i. Decision-token logit attribution: v3 medical features are silent at the decision token (added 2026-05-22)

Source: `results/decision_token_logit_attribution_4b.json` and `_12b.json` (script `paper/scripts/decision_token_logit_attribution.py`).

For each of the 60 NL prompts at 4B and 12B, at the last prompt position (where the model emits the forced letter), we (a) capture the residual at the SAE layer, (b) SAE-encode to get feature activations, (c) for each active feature compute the linear contribution `act_i * W_dec[i] @ W_unembed[L]` for each letter L ∈ {A,B,C,D}. This is the standard logit-lens decomposition for SAE features (nostalgebraist 2020). It ignores non-linearities between the SAE layer and the unembedding — magnitudes are approximate, directional breakdown is informative.

**Headline finding:**

> The v3-validated medical-content features have ZERO activation at the decision token in 60/60 cases at both 4B and 12B.

This is the strongest mechanistic answer to the reviewer's Concern 5. The combined picture across the two Phase-1b analyses:

| Stage | Where do medical features fire? |
|---|---|
| Clinical-narrative tokens (shared vignette) | **Yes**, in 81–100% of (case × feature) peak positions across all three models |
| B-prompt scaffold tokens | Minimal |
| Decision token (last prompt position) | **No, 0/60 cases active at 4B; 0/60 at 12B** |

Linear logit-lens decomposition of mean contribution to predicted letter (across 60 cases):

| Model | Medical (v3) | Scaffold-proxy (top 30) | Other (~47 features) |
|---|---|---|---|
| 4B  L29 | 0.0   | 2.0   | **2627** |
| 12B L31 | 0.0   | 199   | **266**  |

At both scales, the letter prediction is driven by features other than the v3-validated medical-content detectors. At 12B the scaffold-proxy and "other" categories both contribute substantially; at 4B "other" dominates strongly (the scaffold-proxy set defined from the masked-invariance run may not catch the format-related features that actually fire at the decision token — that's a known limitation of using the B_max_content–B_max_vignette proxy).

**Paper move:** elevate to a new sub-section (perhaps §4.4 or as a Phase-2 mechanistic-invariance complement). Suggested wording:

> "We test the reviewer's hypothesised 'scaffold-primary, medical-partial' decision-token state directly via linear logit-lens decomposition of the A/B/C/D letter logits. At both 4B (L29) and 12B (L31), the three v3-validated medical-content features have *zero activation* at the last prompt position in 60/60 cases. The letter prediction is decoded by approximately 50 other features that *are* active at that position, none of which are the contrastively-identified medical-content detectors. Combined with §4.3's finding that medical features peak in the shared clinical vignette content in 81–100% of (case × feature) combinations, this gives a clean mechanistic picture: medical content is internally represented during clinical-narrative processing but is not in the letter-decision pathway. The clinical content was 'received' but the output-mapping circuit doesn't draw on the same features that detected it."

**Caveat (be honest about it):** the "scaffold-proxy" features we used (top 30 by `B_max_content - B_max_vignette` from the masked-invariance run) are features that *peak* somewhere in B's scaffold positions — they may not be the same features that fire *at the specific decision token*. See §2j below for the proper decision-token-firing analysis using saved B_decision/D_decision dumps.

### 2j. Decision-token feature characterization (added 2026-05-22)

Source: `results/decision_token_top_features_{4b,12b,qwen}.json` + `_summary.md` (script `paper/scripts/decision_token_top_features.py`).

Direct test of the reviewer's "scaffold-primary, medical-partial at NL pre-generation" framing using saved decision-token activation vectors. For each case, identify top-20 features by activation at the NL decision token (`B_decision`) and at the NF decision token (`D_decision`). Quantify (a) overlap, (b) where NL-only features peak in B, (c) where NF-only features peak in D.

| Model | NL∩NF top-20 (Jaccard) | NL-only features peaking in **B scaffold** | NF-only features peaking in **D vignette** | v3 medical in NL top-20 | v3 medical in NF top-20 |
|---|---|---|---|---|---|
| 4B  | **0.000** | **87.0%** | 27.8% | 0/60 | 0/60 |
| 12B | **0.001** | **88.3%** | 8.9% | 0/60 | 0/60 |
| Qwen | 0.324 | **94.7%** | 10.4% | 0/60 | 0/60 |

**Three findings, each more direct than the logit-lens version above:**

1. **NL and NF use essentially disjoint top-20 feature sets at the decision token at both Gemma scales** (Jaccard overlap ≈ 0 across all 60 cases at each scale). Different format → completely different computational pathway at the decision token. At Qwen the overlap rises to ~33%, but the asymmetry pattern still holds.

2. **87–95% of NL-only top-20 decision-token features peak on B's scaffold tokens** across all three models. Direct mechanistic confirmation of "scaffold-primary at NL." The feature pool driving forced-letter prediction is dominated by features that fire on the answer-key scaffold text, not on the vignette.

3. **v3 medical features are 0/60 in NL's top-20 AND 0/60 in NF's top-20 at every model.** Combined with the §2i logit-attribution finding that medical features are 0-active at the NL decision token in 60/60 cases at 4B and 12B, the cleanest framing is **medical-absent at the decision token, not medical-partial.**

**Paper move:** §2j should replace the caveated §2i framing. Suggested wording:

> "At the NL pre-generation position, top-20 active features are essentially disjoint from those at the NF pre-generation position (Jaccard ≈ 0 at 4B and 12B, ≈ 0.32 at Qwen). Of the NL-unique top-20 features at every model, 87–95% have their peak activation on the forced-letter scaffold tokens (the 'Reply with exactly one letter' instruction and the A/B/C/D answer-key bullets). The v3-validated medical-content features are not in the top-20 at either NL or NF pre-generation positions at any model (0/60 across all three). Combined with the masked-invariance finding (§4.3) that medical features peak inside the shared clinical vignette in 81–100% of (case × feature) combinations, the mechanistic picture is: medical content IS represented during clinical-narrative processing, but by the time the model reaches the pre-generation position, the feature pool has shifted to scaffold-firing features (NL) or chat-template-suffix features (NF), with medical-content features absent in both. The forced-letter mechanism is reading the scaffold, not the patient narrative."

This finding deserves its own §4.4 or §4.5 sub-section (perhaps "Decision-token feature analysis") given how clean and direct it is.

Medical features stay invariant across all behavioral strata at Qwen;
random features differ noticeably. Same direction as 4B and 12B.

---

## 3. Section-by-section edit punch list

### Abstract

**Find and replace:**

OLD framing about deferral driving inversion:
> Behaviorally, at 4B free-text answers score +17pp higher than forced-letter answers; at 12B the gap inverts via a deferral mechanism that the A/B/C/D label space cannot represent.

NEW:
> Behaviorally, at 4B free-text answers score +17 pp higher than forced-letter answers (both judges); at 12B the gap inverts (+10 pp the other way); at Qwen3-8B it sits between (+7 pp). The accuracy gap in every model is dominated by single-acuity-step miscalibrations between formats — not by deferral. Separately, at 12B 4/60 free-text responses are unanimously labeled DEFERRED by both LLM judges; the A/B/C/D label space cannot represent these clinical hedges, but they happen to flatten to gold-compatible letters and contribute zero to the measured gap.

If there is "deployable monitor" language in the abstract: replace with
"candidate readout requiring prospective validation."

### §1 Introduction — Contribution (iii)

OLD:
> a scaling-dependent shift in the behavioral failure mode driven by deferral

NEW:
> a scaling-dependent shift in the behavioral failure mode dominated by single-acuity-step miscalibration between the two output formats, together with a complementary deferral phenomenon that the A/B/C/D benchmark cannot represent

Optionally add: "We show that this pattern reproduces in a second model family (Qwen3-8B) with smaller magnitude — suggestive cross-family consistency."

### §4.1 Behavioral phenomenon

- Add a Qwen column to Table 1 using §2a numbers.
- Delete any sentence saying "Qwen3-8B is used in this work for the mechanistic analyses only" — that disclaimer is now false.
- Add a row/paragraph noting the +6.7 pp Qwen gap and the cross-family pattern.

### §4.2 Deferral analysis — REWRITE

Replace the existing §4.2 body with the **adjacent-miscalibration + independent-deferral framing** from §1a above. Concrete structure:

1. **What drives the accuracy gap** (one paragraph + a table):
   - 4B's NF > NL gap: 14/14 NF_only_right cases are adjacent (NL one notch below gold, NF on gold; 13/14 are the exact "NL=B, gold=C" pattern)
   - 12B's NL > NF gap: 6 NL_only_right cases, 5/6 adjacent (NF picks an adjacent letter to NL's correct one)
   - Qwen sits between with both patterns at lower magnitude (6 NF_only_right, 8 NL_only_right, both predominantly adjacent except 5 Qwen NF_only_right where NL severely under-triages two notches — "NL=A while gold is B or C")
   - **The format that wins on a given case is the one that doesn't single-notch miscalibrate.**

2. **The deferral phenomenon, as a separate point** (one paragraph + the §2b table):
   - 0/60 at 4B, 2/60 at Qwen, 4/60 at 12B (scaling-up trend)
   - At 12B: all 4 happen to flatten to gold-compatible letters under 4-way scoring → live in `both_right` → 0 contribution to the gap
   - At Qwen: both 2 produce judge disagreement under 4-way scoring → live in `judges_disagree` → partial contribution
   - The deferral phenomenon is a benchmark-adequacy concern about the A/B/C/D label space, not the driver of the measured inversion

3. **The mechanistic invariance result framing** needs the new context too:
   - Shared clinical-prefix tokens give trivial invariance under causal masking (vignette mask sMAPE ≈ 0 for both medical and random; sanity check passes)
   - The non-trivial finding is: medical features peak inside the shared vignette content in 81–100% of (case × feature) combinations, while random firing features peak more diversely
   - Full-content max-pool sMAPE medical-vs-random gap survives magnitude-matched random resampling (perm-p ≤ 0.012 across all three models)

### §4.3 Mechanistic invariance — Add Qwen panel + magnitude-matched table

- Replace the existing single-row Qwen mechanistic table with the 5-row per-stratum table from §2d above.
- Add the magnitude-matched random resampling result (§2c) as a new column or as a methods paragraph: "Random baseline robustness: 1000 resamples from a magnitude-matched random pool give perm-p < 0.012 at all three models."
- Add a sentence: "Medical-feature *peaks* are anchored in the shared clinical vignette in 81–100% of (case × feature) combinations, providing a content-anchoring interpretation of the medical-vs-random gap."

### §4.5 Section header retains, but tighten language

Remove any "exclusive firing" language. If our top-tokens analysis only looked at top-K activations, say so explicitly.

### §5 Discussion

**(i) Mechanistic claim** — soften from "internal reasoning preserved" to:
> medical-domain representation is preserved on the shared clinical prefix (expected under causal masking) and medical features remain anchored to clinical-content tokens across format conditions. We do not claim the model represents the *correct triage disposition*; the SAE features we analyze are medical-vs-nonmedical detectors, not acuity probes.

**(iii) Scaling paragraph** — replace "deferral-driven" with "adjacent-miscalibration in opposite directions at the two Gemma scales, with Qwen between." See §1a.

**Anywhere that says** "deployable monitor" → "candidate readout"

### §6 Limitations

Add a paragraph acknowledging:

1. **Mechanistic claim scoped to shared-prefix invariance + decision-token redesign as future work.** The Phase 1b mechanistic analysis as currently designed measures invariance over content tokens (which include the shared clinical prefix, where causal masking guarantees identity). We have shown via per-mask decomposition that (a) the prefix-mask invariance is trivial as expected; (b) medical features peak in the shared prefix in 81–100% of (case × feature) combinations; (c) the full-content max-pool gap survives magnitude-matched random resampling at p < 0.012 across all three models. A decision-token analysis (comparing the last *content* token, not the last *prompt* token) would more directly test whether the answer-token representation retains medical-domain content; we leave this and logit-attribution analyses for future work.

2. **SAE features as medical-vs-nonmedical detectors, not acuity probes.** "Medical content represented" is not the same as "correct triage disposition represented." We do not claim the latter.

3. **Random baseline magnitude-matching.** The original fixed-seed random pool included zero-firing features whose sMAPE is artificially zero by denominator floor. We re-ran with 1000 magnitude-matched resamples and report permutation p-values; results survive.

4. **Cross-family evidence is one model, one layer, one SAE.** Qwen3-8B with the Qwen Scope L31 TopK=50 SAE shows the same direction as Gemma 4B/12B but with weaker numerical separation; cross-family evidence is suggestive consistency, not confirmation.

5. **Deferral phenomenon vs. measured gap.** The unanimous deferrals at 12B and Qwen do not contribute (12B) or contribute only partially (Qwen) to the measured accuracy gap under 4-way scoring; we report them separately as a benchmark-adequacy observation.

6. **Clinician sample is small (n=16, one rater) and enriched.** We do not claim representative clinical-practice rates.

If §6 currently says "single-family within-family comparison" — delete that, the Qwen behavioral data now exists.

### Appendices

- **Appendix A1B (K-sweep)**: footnote that this table reports mean-pool sMAPE values from a prior pipeline version. A max-pool re-run is pending and will be added in v4.
- **Appendix Ax (token-mask decomposition)**: new appendix with the full §2c tables and a brief methods description for the per-mask analysis. Source: `results/phase1b_masked_invariance_*.json`.
- **Appendix Ax (gap decomposition)**: include the per-case decomposition table from `results/gap_decomposition.md` showing all 14 4B NF_only_right cases, 6 12B NL_only_right cases, and 14 Qwen cases in the two driving strata.

---

## 4. Now done (Buckets A + B, 2026-05-22)

1. ✅ **Paired McNemar / bootstrap CIs for NL vs NF gaps** —
   `results/paired_tests_and_confusion.{json,md}`. See §2a for table
   and the verdict (4B/12B significant at α=0.05, Qwen not).

2. ✅ **Per-acuity breakdown for 4B and 12B (and Qwen, already had it)** —
   `results/paired_tests_and_confusion.md`, §(2). All three models
   share the same gradient: A-bucket is the weakest NF accuracy
   bucket. Worth a sentence in §4.1 or a column in Table 1.

3. ✅ **Confusion matrices + triage-direction breakdown** — same file,
   §(3) and §(4). The under-triage finding is clinically meaningful and
   reviewer-requested. Strong candidate for a new §4.1 sub-paragraph
   ("Triage direction"). The 4B singleton-D = 0% failure is in the
   confusion matrices and should be flagged in §6.

4. ✅ **K-sweep with max-pool sMAPE** (Bucket B / Concern 4c) —
   `results/phase1b_sensitivity_maxpool.{json,md}`. Table in §2e
   above. Replaces the inconsistent mean-pool Appendix A1B table.

## 4b. Wording fixes for the LaTeX-writer (no compute, just text)

- **Encoder→detector relabeling** (Concern 4d): in §4.4 and Appendix
  where we currently say "loading onto the SAE feature direction" or
  "projection onto feature direction," replace with "detector
  alignment" or "encoder-direction alignment." Encoder columns are
  detector directions, not residual contribution directions.

- **"Residual-dimension max-pool" interpretive caveat** (Concern 4e):
  if any analysis uses max-over-tokens per residual *dimension* (as
  opposed to max-over-tokens of *feature activations*), flag explicitly
  that this creates a synthetic vector that may not occur at any token.
  For feature activations max-pooling is fine and is what we do.

---

## 5. What we are explicitly NOT doing in v3

- **Decision-token redesign** (better last-content-token comparison): we have the per-token activations saved in `results/phase1b_masked_full_activations_*.npz`, so a future script could re-aggregate without a new GPU run, but the redesign needs careful token-boundary work. **Limitations §6 mention; defer to future work.**
- **Logit attribution to A/B/C/D letters** (reviewer's recommended additional analysis #2): substantial new work. Mention as future work in §6.
- **Option-order randomization** (reviewer's #3): would require regenerating all 60 vignettes with shuffled letters and re-running forced-letter eval — significant compute and adjudication. Defer.
- **External-corpus feature identification** (reviewer's #4 / 4.1 circularity concern): the 4B medical features are already identified on an external medical/non-medical corpus (Phase 5 contrastive). For 12B and Qwen we re-used the 60 cases — we should add a sentence in §6 Limitations noting the circularity at 12B/Qwen.
- **Deferral-aware clinical scoring** (reviewer's #5): we have the data — the "either judge correct" envelope partially addresses this. Could add a third column to Table 1 if desired. Cheap.
- **K-sweep re-run with max-pool sMAPE**: needs ~1 hr GPU. Defer with footnote.
- **Clinician expansion beyond n=16**: not feasible for v3; §6 honest about this.

---

## 6. Files the LaTeX-writer should have alongside this doc

All committed to `main` as of `554e8bc` (today's last commit):

| File | Purpose |
|---|---|
| `results/gap_decomposition.md` | Per-case decomposition for all three models, with adjacency analysis |
| `results/gap_decomposition.json` | Same, machine-readable |
| `results/phase4b_qwen_post_adjudication_summary.md` | Qwen headline numbers + per-stratum mechanistic table |
| `results/phase1b_masked_invariance_{4b,12b,qwen}.json` | Per-mask sMAPE/cosine/peak-position data |
| `results/phase1b_random_pool_resample_{4b,12b,qwen}.json` | Magnitude-matched resampling results |
| `paper/scripts/gap_decomposition.py` | Reproducible script for §4.2 numbers |
| `paper/scripts/phase1b_masked_invariance.py` | Reproducible script for §4.3 numbers |
| `paper/scripts/phase1b_random_pool_resample.py` | Reproducible script for the magnitude-matched controls |
| `paper/scripts/qwen_post_adjudication_tally.py` | Reproducible script for the Qwen column |

---

## 7. Dependency graph for the edits

```
Abstract rewrite ── depends on §4.2 framing decision
§1 contribution (iii) ── depends on §4.2 framing decision
§4.1 table       ── independent (just add Qwen column from §2a)
§4.2 (REWRITE)   ── load-bearing; everything depends on this
§4.3 panel       ── depends on §4.2's content-anchoring framing
§5 discussion    ── depends on §4.2 + §4.3
§6 limitations   ── independent additions to existing limits
Appendices       ── independent
```

**Recommended order:** §4.2 first → §4.3 second → §4.1 + §1 + abstract together → §5 → §6 → appendices.

---

## 8. Cost / time tally for the data work backing v3

- Qwen behavioral A100 (40 min): ~$0.80
- Qwen adjudication (240 API calls): ~$3
- Masked invariance A100 (~30 min): ~$0.55
- Three failed instance launches (image-tag typo, broken-GPU host): $0 (none billed)

**Total v3 data work: ~$4.35 + LLM-writer tokens.**
