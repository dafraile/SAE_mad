# v3 changes — internal change-log

Internal document tracking what changed from v2 → v3 of the SAE-routing
triage paper. Triggered by an external GPT-5.5-Pro review acting as a
reviewer simulator. Each numbered concern below is mapped to (a) what we
did about it, (b) where the new evidence lives in the repo, (c) what
framing implications fall out for the manuscript.

This is NOT a reviewer-response letter — the "reviewer" was an LLM
acting as a peer reviewer for internal calibration. The document is for
us (and the LaTeX writer) to track what survives, what changed, and
what's been added.

---

## TL;DR — the v3 story upgrade

Two empirical findings from this revision cycle materially change the
paper's headline:

1. **Adjacent miscalibration drives the NL↔NF accuracy gap; deferral does not.** Of the 4 unanimous DEFERRED cases at 12B (F15, F19, F23, F24), *all four* flatten to gold-compatible letters under 4-way scoring and live in `both_right`. They contribute zero to the measured accuracy gap. The replacement story is much cleaner: at every model, 75–100% of the gap-driving cases are single-acuity-step adjacent miscalibrations (B↔C or C↔D), not deferrals. Deferral remains a real benchmark-adequacy phenomenon (independent of the accuracy gap).

2. **The forced-letter accuracy penalty at 4B IS the canonical letter-binding × content-prior interaction.** Option-order shuffle (60 cases × 3 random label permutations) at 4B gives shuffled NL accuracy = 71.7% = NF accuracy *to the case*. Randomize the letter assignment, the format penalty disappears entirely. At 12B and Qwen the canonical mapping mildly *helps* (+2-3 pp); at all three models, there is no position bias and a strong content prior (67–82% same-content under shuffles).

The mechanistic chain across all three layers of analysis:
- Vignette processing: medical features fire on clinical content (81–100% of (case×feature) peak positions)
- Decision token: medical features go silent (0/60 active at 4B and 12B; 0/60 in top-20 at any model)
- Forced-letter output: dominated by content prior over the four option-texts; canonical letter mapping happens to align well at 12B (+2.8 pp) but poorly at 4B (-16.7 pp)

Title walk-back: "clinical reasoning preserved" → "medical-domain representation preserved on shared prefix; absent from the letter-decision pathway."

---

## Concern-by-concern change log

Numbering follows the GPT-5.5-Pro review (Concerns 1–6).

### Concern 1 — Prefix-token invariance is partially tautological

**Status:** Empirically addressed.

**Reviewer's point:** under causal masking, hidden states at shared clinical-prefix tokens are byte-identical between NL and NF. Pooling over those positions gives a sMAPE component that is trivially zero. We claimed "medical features stay invariant" without specifying where the invariance lives.

**What we did:**
- Re-ran Phase 1b with explicit token masks: vignette / scaffold / decision token. Three masks computed per case, per model. (`paper/scripts/phase1b_masked_invariance.py`)
- Verified the byte-identical-prefix assumption holds: 60/60 cases at Gemma 4B and 12B have the vignette tokens byte-identical between NL and NF; at Qwen, 60/60 vignette texts re-tokenize identically when isolated but the BPE merges the trailing punctuation differently in NL vs NF, shifting the divergence one token earlier than the vignette end. (`paper/scripts/verify_byte_identical_prefix.py`)
- Re-aggregated under a per-feature peak-position diagnostic: medical features peak inside the shared vignette in 81–100% of (case × feature) combinations across all three models. This gives the original "medical sMAPE ≈ 0" claim a clean non-trivial interpretation: medical-feature *peaks* are anchored in the clinical narrative.
- Replaced the fixed-seed random baseline with 1000 magnitude-matched random pool resamples. Medical-vs-random sMAPE gap survives at all three models: 4B perm-p < 0.001, 12B perm-p < 0.001, Qwen perm-p = 0.012. (`paper/scripts/phase1b_random_pool_resample.py`)

**Where it lives:**
- `results/phase1b_masked_invariance_{4b,12b,qwen}.json`
- `results/phase1b_random_pool_resample_{4b,12b,qwen}.json`
- `results/verify_byte_identical_prefix.{json,md}`

**Manuscript implications:**
- §3.1: clarify the byte-identical-prefix claim — holds at Gemma; Qwen has a single-token BPE boundary effect at the trailing punctuation only.
- §4.3: replace the headline number with the per-mask decomposition. Vignette-mask sanity check passes (sMAPE ≈ 0 for both medical and random); full-content max-pool sMAPE gap is non-trivial and survives magnitude-matched resampling.

### Concern 2 — Medical content ≠ triage reasoning

**Status:** Partially addressed via decision-token findings; the rest is framing-only.

**Reviewer's point:** SAE medical features are medical-vs-nonmedical detectors. Showing they fire on clinical content does not show the model represents the correct triage disposition. The claim "clinical reasoning preserved" overclaims.

**What we did:**
- Generated the decision-token findings (see Concern 5 below) which show the v3 medical features are silent at the letter-prediction position. This means medical features *detect* clinical content but are not in the letter-decision pathway. The honest claim is "medical-domain representation preserved on shared prefix" rather than "clinical reasoning preserved."
- Flagged title walk-back to the LaTeX writer with three candidate titles avoiding "clinical reasoning preserved."
- Replaced "deployable monitor" → "candidate readout requiring prospective validation" throughout the framing notes.

**Manuscript implications:**
- §5 Discussion: replace "clinical reasoning preserved" with "medical-domain representation is preserved on the shared clinical prefix; the letter-decision pathway does not draw on the medical-content features."
- §6 Limitations: add an explicit acknowledgement: "Our SAE features are medical-vs-nonmedical detectors, not acuity probes. We do not claim the model represents the correct triage disposition."

### Concern 3 — Deferral accounting may be internally inconsistent

**Status:** Empirically addressed. The reviewer was *more* right than they realized.

**Reviewer's point:** the §4.2 v2 framing said the 12B inversion is "driven by deferral," but the appendix table shows the four unanimous deferrals flatten to gold-compatible letters under 4-way scoring. They cannot drive a 10 pp gap.

**What we found:**
- Reviewer estimated 3/4 of unanimous deferrals at 12B are counted correct. Actually **4/4**: F15 → C (gold C/D), F19 → B (gold B), F23 → B (gold A/B), F24 → B (gold B). All in `both_right`. Zero contribution to the accuracy gap.
- We ran a full case-by-case decomposition of the gap drivers across all three models. (`paper/scripts/gap_decomposition.py`)
- At 4B, all 14 NF_only_right cases are adjacent miscalibrations; 13/14 follow the exact pattern "NL=B, gold=C."
- At 12B, 5/6 NL_only_right cases are adjacent miscalibrations; 0/6 are unanimous deferrals.
- At Qwen, 8/8 NL_only_right are adjacent; 6/6 NF_only_right are adjacent miscalibrations in the opposite direction (mostly "NL=A, gold=C" — a 2-step under-triage pattern).
- Overall, 19/20 NL_only_right and 15/20 NF_only_right cases across all three models are single-acuity-step miscalibrations.

**Where it lives:**
- `results/gap_decomposition.{json,md}` — the full per-case decomposition table

**Manuscript implications:**
- §4.2 rewrite (load-bearing): replace "deferral drives the inversion" with "adjacent miscalibration drives the gap in both directions; deferral is a separate benchmark-adequacy phenomenon."
- §1 Contribution (iii): same replacement.
- Abstract: same replacement.
- §5 Discussion: scaling-related text needs reframing.

### Concern 4 — Random baseline, K-sweep, encoder/decoder, residual max-pool

**Status:** Empirically addressed (a, b, c); wording-only fixes (d, e) for LaTeX writer.

#### 4a. Feature-selection circularity at 12B and Qwen

The 12B and Qwen medical features were identified using the same 60 cases later used in the invariance analysis. At 4B the features come from an external contrastive corpus (Phase 5).

**What we did:** flagged in §6 Limitations as an honest acknowledgement. External-corpus feature ID for 12B and Qwen is future work (would need fresh GPU runs and careful corpus design).

#### 4b. Fixed random-feature seed

**What we did:** 1000 magnitude-matched random pool resamples per model. Permutation p-values: 4B < 0.001, 12B < 0.001, Qwen = 0.012. The original fixed-seed random baseline systematically understated the gap because the firing-threshold was too lenient (included near-zero features whose sMAPE is artificially zero by denominator floor). (`paper/scripts/phase1b_random_pool_resample.py`)

#### 4c. K-sweep mean-pool vs main-text max-pool inconsistency

**What we did:** re-ran the K-sweep with max-pool sMAPE for K ∈ {3, 5, 10, 20} at 4B L29 and 12B L31, plus K=3 at Qwen. All K significant under paired-bootstrap 95% CI. CPU only (reused saved per-case max-pool activations). (`paper/scripts/phase1b_sensitivity_maxpool.py`, output `results/phase1b_sensitivity_maxpool.{json,md}`)

#### 4d. Encoder vs decoder column projection

**What we did:** flagged for LaTeX writer in §4.4 — rename "loading onto the SAE feature direction" → "detector alignment" or "encoder-direction alignment."

#### 4e. Residual-dimension max-pool interpretation

**What we did:** flagged for LaTeX writer — feature-activation max-pool is fine and is what we do; residual-dimension max-pool would be a different and harder-to-interpret operation. Wording-only clarification in the methods section.

### Concern 5 — Decision-token analysis should become central

**Status:** Empirically addressed with three layers of evidence (the most-improved section of v3).

**Reviewer's point:** the strongest mechanistic claim should live at the decision token, not on the shared vignette. Recommended logit attribution as the analysis "that would most directly support the output-mapping claim."

**What we did — three complementary analyses:**

#### (i) Logit-lens attribution (`paper/scripts/decision_token_logit_attribution.py`)

Linear logit-lens decomposition of A/B/C/D logits at the NL last-prompt-position into SAE-feature contributions. For each active feature, compute `act_i × W_dec[i] @ W_unembed[L]` for each letter L ∈ {A, B, C, D}, then categorize features as medical (v3 set), scaffold-proxy (top 30 features by `B_max_content − B_max_vignette` from the masked-invariance run), or other.

**Precise category means (mean linear contribution to letter logits across 60 cases):**

**4B L29:**

| Category | →A | →B | →C | →D | →pred |
|---|---|---|---|---|---|
| medical | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| scaffold (top 30) | 2.079 | −0.348 | 2.610 | 0.448 | 2.047 |
| other (~47 features) | 778.749 | 2017.909 | 2202.574 | 962.299 | **2627.302** |

**12B L31:**

| Category | →A | →B | →C | →D | →pred |
|---|---|---|---|---|---|
| medical | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| scaffold (top 30) | 261.926 | 132.021 | 239.692 | 156.226 | **198.834** |
| other (~47 features) | 277.293 | 28.022 | 307.884 | 140.519 | **266.462** |

**These are the precise numbers the LaTeX writer asked for to fill in Table 6's 12B row.** The 12B scaffold-to-predicted-letter mean is 198.834 (not "analogous" — the scaffold-proxy features have a substantial mean contribution at 12B, much more than at 4B; the "other" category dominates at 4B but is closer to the scaffold-proxy at 12B).

**Headline finding:** v3 medical features have *zero activation* at the decision token in 60/60 cases at both 4B and 12B. Letter prediction is decoded by ~50 other features that ARE active at the decision token.

#### (ii) Top-feature characterization (`paper/scripts/decision_token_top_features.py`)

For each case, compute top-20 features by activation at the NL decision token and at the NF decision token. Quantify (a) Jaccard overlap, (b) where NL-only features peak in B (vignette vs scaffold), (c) where NF-only features peak in D (vignette vs suffix). CPU only.

| Model | Jaccard NL ∩ NF top-20 | NL-only features peaking in **B scaffold** | NF-only features peaking in **D vignette** | v3 medical in NL/NF top-20 |
|---|---|---|---|---|
| 4B | **0.000** | **87.0%** | 27.8% | 0/60, 0/60 |
| 12B | **0.001** | **88.3%** | 8.9% | 0/60, 0/60 |
| Qwen | 0.324 | **94.7%** | 10.4% | 0/60, 0/60 |

**Three findings:**
- NL and NF use essentially disjoint top-20 feature sets at the decision token at both Gemma scales. Different format → completely different computational pathway. Qwen has ~33% overlap, but the asymmetry pattern still holds.
- 87–95% of NL-only top-20 decision-token features peak on B's scaffold tokens (outside the shared vignette) across all three models. Direct confirmation of "scaffold-primary at NL."
- v3 medical features are 0/60 in top-20 at both NL and NF decision tokens at every model.

#### (iii) Option-order shuffle — see Concern 5 + 6 below

**Where it lives:**
- `results/decision_token_logit_attribution_{4b,12b}.json`
- `results/decision_token_top_features_{4b,12b,qwen}.json`
- `results/decision_token_top_features_summary.md`

**Manuscript implications:**
- Reorder §4 to put the decision-token analysis earlier, as the reviewer recommended. The masked-invariance result becomes a sanity-check baseline for the more substantive decision-token finding.
- Add a new sub-section (§4.4 or §4.5) titled something like "Decision-token feature pool" with the cross-model headline table and the disjoint-top-K finding.

### Concern 6 — "Deployable monitor" overclaim

**Status:** Framing-only walkback.

**What we did:** flagged for LaTeX writer to replace "deployable monitor" with "candidate readout requiring prospective validation" everywhere. Three candidate softer titles supplied.

**Manuscript implications:**
- Abstract: drop "deployable" / "monitor" language.
- §7 Conclusion: walk back to "candidate readout."
- §6 Limitations: add an explicit list of what would be needed for the deployable-monitor claim (calibration, robustness, prospective validation, model-family generalization). We do none of these.

---

## New experiments added in v3 that the reviewer did NOT ask for

These came out of the reviewer-prompted investigation but are paper-strengthening contributions in their own right.

### Cross-family Qwen behavioral 4-cell evaluation

The v2 draft used Qwen3-8B for mechanistic analysis only. We added a full SL/NL/NF behavioral evaluation on the 60 canonical vignettes, plus 4-way and 5-way LLM-judge adjudication.

**Headlines:** Qwen SL = NL = 75.0%, NF (4-way both judges) = 68.3%. NL−NF gap = +6.7 pp, McNemar exact p = 0.45 (underpowered at n=60; reason to soften to "suggestive cross-family consistency" rather than "confirmation").

**Where:** `results/phase4b_qwen_behavioral.json`, `results/phase4b_qwen_post_adjudication_summary.md`.

### Paired tests + per-acuity + confusion matrices + under-triage rates

Added per-acuity (gold A/B/C/D bucket) breakdown for 4B and 12B (Qwen already had it), confusion matrices for SL/NL/NF, and under-triage / over-triage / no-commit rates. The under-triage finding is clinically meaningful: 4B NL under-triages on 20 of 60 cases (vs 7 over-triage); Qwen NL under-triages 12 cases (vs 3 over-triage). Free-text mode is more balanced at every model.

The 4B singleton-D failure the reviewer flagged is confirmed: 4B picks D = 0 of 9 D-only-gold cases in NL forced-letter mode.

**Where:** `results/paired_tests_and_confusion.{json,md}`.

### Cross-model option-order shuffle (4B + 12B + Qwen)

The single strongest new finding. For each case, 3 random non-identity permutations of the letter→content mapping; greedy forced-letter; score same-letter % vs same-content % vs accuracy.

| | 4B | 12B | Qwen |
|---|---|---|---|
| Same-letter % | 21.1% | 25.0% | 25.6% |
| Same-content % | 67.2% | 80.6% | 82.2% |
| Canonical NL acc | 55.0% | 81.7% | 75.0% |
| Shuffled NL acc | 71.7% | 78.9% | 72.8% |
| NF (4-way both judges) acc | 71.7% | 71.7% | 68.3% |
| Shuffled vs canonical | +16.7 pp | −2.8 pp | −2.2 pp |
| **Shuffled vs NF** | **+0.0 pp (EXACT)** | +7.2 pp | +4.4 pp |

At 4B the format penalty *is* the canonical letter-binding interacting with content prior. At 12B and Qwen the canonical mapping helps slightly, but there is a separate NF-mode penalty (the adjacent-miscalibration of §4.2) that is independent of letter-binding.

**Where:** `results/option_order_shuffle_{4b,12b,qwen}.json` + `results/option_order_shuffle_all_models.{json,md}`.

---

## Files added in v3

### New scripts in `paper/scripts/`

- `gap_decomposition.py` — case-level NL ↔ NF gap decomposition + adjacency classification
- `phase1b_masked_invariance.py` — per-mask SAE invariance (vignette / scaffold / decision)
- `phase1b_random_pool_resample.py` — 1000-sample magnitude-matched random pool resampling
- `phase1b_sensitivity_maxpool.py` — K-sweep with max-pool sMAPE (closes Concern 4c)
- `phase4b_qwen_behavioral.py` — Qwen3-8B SL/NL/NF behavioral evaluation
- `wire_adjudicator_qwen.py` — wires Qwen NF outputs for the paper-faithful adjudicator
- `qwen_post_adjudication_tally.py` — bundles Qwen behavioral + adjudication results
- `verify_byte_identical_prefix.py` — tokenization sanity check across the three models
- `paired_tests_and_confusion.py` — paired McNemar / bootstrap CIs / per-acuity / confusion matrices
- `option_order_shuffle_4b.py` — option-order randomization experiment (generalized to all three models via `--model`)
- `option_order_shuffle_summary.py` — cross-model summary builder
- `decision_token_logit_attribution.py` — linear logit-lens attribution at the NL decision token
- `decision_token_top_features.py` — top-K feature characterization at NL vs NF decision tokens

### New result files in `results/`

- `phase4b_qwen_behavioral.json` + `phase4b_qwen_D_for_adjudication{,_adjudicated_paper}.{json,csv}` + `phase4b_qwen_adjudicated_deferred.json` + `phase4b_qwen_post_adjudication_summary.md` + `phase4b_qwen_post_adjudication_tally.json` — Qwen behavioral closure
- `gap_decomposition.{json,md}` — corrected adjacent-miscalibration story
- `phase1b_masked_invariance_{4b,12b,qwen}.json` + `phase1b_masked_full_activations_{4b,12b,qwen}.npz` — masked Phase 1b
- `phase1b_random_pool_resample_{4b,12b,qwen,summary}.json` — 1000-sample resampling
- `phase1b_sensitivity_maxpool.{json,md}` — max-pool K-sweep
- `verify_byte_identical_prefix.{json,md}` — tokenization sanity check
- `paired_tests_and_confusion.{json,md}` — paired McNemar / per-acuity / confusion matrices
- `option_order_shuffle_{4b,12b,qwen}.json` + `option_order_shuffle_all_models.{json,md}` — option-order shuffle
- `decision_token_logit_attribution_{4b,12b}.json` — linear logit-lens attribution
- `decision_token_top_features_{4b,12b,qwen}.json` + `decision_token_top_features_summary.md` — top-K characterization

### New documents in `paper/`

- `LATEX_WRITER_HANDOFF_v3.md` — consolidated edit punch-list for the LaTeX writer, with ready-to-paste prose for §4.2, abstract, §1, §5, §6 plus three new sub-sections (§2g–§2j) covering the option-order shuffle, the per-mask invariance, the logit attribution, and the top-feature characterization.
- `V3_CHANGES.md` — this document.

---

## Compute spend across the v3 cycle

| GPU session | Wall time | A100 cost |
|---|---|---|
| Qwen behavioral (Phase 4b)            | 40 min | ~$0.80 |
| Qwen adjudication (240 API calls)     | 12 min | ~$3.00 (API, not GPU) |
| Masked invariance (4B + 12B + Qwen)   | 30 min | ~$0.55 |
| Task 4 + Task 5 (4B options + logit attribution) | 35 min | ~$0.65 |
| Task 18 (12B + Qwen options shuffle)  | 10 min | ~$0.20 |

**Total: roughly $5–6 of GPU + ~$3 of LLM-judge API.** Plus three killed instances (failed image pulls, broken-GPU host) that didn't bill — lessons learned: always verify the Docker tag exists and do a basic CUDA sanity-allocation before bootstrapping a full run.

---

## What's NOT in v3 (deferred to future work, §6-acknowledged)

- **Decision-token last-content-token analysis** (vs last-prompt-position). The current decision-token analyses use the chat-template suffix token. A cleaner analysis would compare at the last content token in each prompt (last scaffold token in NL, last vignette token in NF). Future work.
- **Logit attribution with full transformer forward-pass** (vs linear logit-lens). The current attribution ignores non-linearities between the SAE layer and the unembedding. Future work.
- **External-corpus feature identification for 12B and Qwen.** 4B uses external Phase-5 contrastive features; 12B and Qwen re-use the 60 cases. Future work.
- **Acuity-aware feature probe.** Concern 2 is partly addressed by decision-token findings; a triage-acuity-specific feature probe would be cleaner. Future work.
- **Option-order shuffle at K > 3 permutations per case.** Power is fine at K=3, but K=10 would give tighter CIs on shuffled accuracy. Cheap to extend.
- **Top-feature characterization with token-text annotation.** We classify NL-only features as "peaks in scaffold vs vignette" by position; annotating the actual token text at the peak position would give a richer qualitative picture of what each feature detects. Doable from saved activations + tokenizer; ~hour of work if you want it later.
- **Clinician expansion beyond n=16.** Out of scope for v3.
- **Option-order shuffle interpretive depth.** The interaction at 12B ("canonical mapping helps + 2.8 pp") is not fully explained mechanistically. We hypothesize but don't prove that 12B's content prior is more diffuse than 4B's. Future work.

---

## For the LaTeX writer

Everything needed for the v3 revision is in `paper/LATEX_WRITER_HANDOFF_v3.md`. Specific numbers for Table 6's logit-attribution row at 12B are in this document under Concern 5, with the precise category means and per-letter splits. The 4B and 12B numbers in `decision_token_logit_attribution_{4b,12b}.json` are the source of truth.

Dependency order for the edits is in the handoff doc; §4.2 rewrite is load-bearing and should be done first because the abstract, §1, and §5 all reference it.
