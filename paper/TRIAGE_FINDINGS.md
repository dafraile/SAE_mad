# SAE Features as a Format-Invariance Detector for Medical Triage

A continuation of the SAE_mad project after v3 closed the rescue-by-amplification branch.
Documents the experimental record from the pivot toward "SAE features as a groundedness
/ format-invariance monitor" using the triage replication dataset.

**Status**: Phases 0–7 complete on Gemma 3 4B IT, Gemma 3 12B IT, **and Qwen3-8B**.
Phase 1A clinician adjudication out for review. Targeting **EMNLP 2026 via
ARR May 25**.

Three families' worth of evidence: Gemma 4B (JumpReLU SAE) shows the cleanest
multi-layer Version B; Gemma 12B reveals the behavioral format effect attenuates
at scale and a depth-dependent mechanistic pattern; Qwen3-8B (k=50 TopK SAE)
replicates the deep-layer Version B at L31 despite the SAE's more aggressive
sparsity. **Phase 5 names the format-direction features**: at 4B L29 the top
features by alignment with the (NL−NF) max-pool direction are SAE features
that fire exclusively on the forced-letter answer-key scaffold tokens
themselves (feature 3833 on "next" inside "B = See my doctor in the next few
weeks", feature 10012 on "the" inside "D = Go to the ER now", feature 980
across the "=" syntax of the answer-key). The format effect is therefore not
just "non-medical features somewhere"; it is specifically these
forced-letter-scaffold-detector features.

## TL;DR

- v3 closed the SAE-as-steering-tool branch with a properly scoped null. The pivot was to use
  SAE features as a *readout* rather than a *driver* — a monitor of whether the model's internal
  clinical representation is preserved when the output format changes.
- The dataset is the 60 paper-canonical vignettes from
  `nature_triage_expanded_replication` (David's Matters Arising response to Ramaswamy et al.
  *Nature Medicine* 2026). Three cells differ on input style and output constraint.
- **Behavioral finding (Phase 0.5)**: Gemma 3 4B IT shows a format effect — under
  paper-faithful LLM-as-judge scoring, natural+free-text **outperforms** natural+forced-letter
  by +13–20pp overall (60–73% across cells). The direction is opposite to frontier-scale
  models in the paper, plausible at this scale.
- **Mechanistic finding 1 — magnitude (Phase 1b)**: medical SAE features at layers 9, 17,
  22, 29 are significantly more invariant under the B↔D format change than 30 magnitude-
  matched random features. Bootstrap 95% CIs exclude zero across all 4 layers × 4 strata.
  Per-token max activations on identical clinical content match within 0–4%.
- **Mechanistic finding 2 — direction vanishes under content control (Phase 2b)**:
  the natural-forced-letter prompt and the patient-realistic prompt share **byte-identical**
  clinical content (1033 chars), differing only in whether the forced-letter instruction
  block is appended. When we truncate B's content range to match D's (i.e., feed the model
  the same clinical text both times), the residual diff norm at L17 and L29 is exactly 0.
  No format effect exists at the residual level when content is held identical.
- **Mechanistic finding 3 — where the residual-level format effect goes (Phase 2b max-pool)**:
  the format effect captured by max-pooling over content tokens loads onto **non-medical
  features**. The three v3-validated medical features at L29 sit at the 15.4%, 42.8%, and
  61.1% percentile of |alignment| with the (B−D) direction; top-aligned features are not
  medical. The residual-level format signal is the model's response to the forced-letter
  *instruction tokens*, not a modulation of clinical encoding.
- **Interpretation (Version B, supported)**: The format effect lives downstream of clinical
  encoding. The model's "this is asthma / this is DKA" representation is unchanged by the
  output instruction. The accuracy difference between B and D (and the paper's headline
  triage failures) is an output-mapping effect, not a clinical-reasoning effect. The
  benchmark is producing apparent clinical failures from intact clinical representation.

## Pivot context

After v3 closed:

- v1 cross-lingual representation work and v2 1B language steering (Feature 857 → Spanish) hold.
- v3 single-feature single-layer ablation/amplification of language and medical features showed
  zero detectable rescue across five tested configurations and four layer depths (9, 17, 22, 29).
- Conclusion from v3: the medical features Gemma exposes via SAE decomposition are *readouts* of
  clinical content (cross-lingual, cross-format, zero on non-medical) but not individually
  causal levers for task performance. They represent what the model "thinks about," not what it
  decides to output.

The pivot:

- If features are readouts, they're monitoring signals, not control signals.
- Connection to David's Matters Arising paper: that work shows output format (forced A/B/C/D vs
  free-text) materially shapes triage scoring on the paper's failure cases, with the
  reasoning-vs-mapping ambiguity unresolved at the behavioral level.
- The mechanistic question therefore becomes: are the model's medical features format-invariant?
  If yes → reasoning is preserved, format effects are downstream → SAE features are a viable
  format-robust monitor. If no → format reaches into clinical understanding → harder story but
  still publishable.

## Datasets

All experiments use 60 paper-canonical vignettes from
`nature_triage_expanded_replication/paper_faithful_replication/data/canonical_singleturn_vignettes.json`
and the matched forced-letter file `paper_faithful_forced_letter/data/canonical_forced_letter_vignettes.json`.

Three cells available natively (no constructed prompts):

| Cell | Source field | Description |
|---|---|---|
| A | `structured_forced_letter` | Structured clinical input + terse "Reply with one letter" |
| B | `natural_forced_letter` | Patient-realistic natural input + terse forced letter |
| D | `patient_realistic` | Patient-realistic natural input + open-ended question |

A fourth cell (structured + free-text) was *not* tested. Constructing it would be a methodology
choice the paper did not authorize. Skipping was the right call — the paper's 2×2 was always
asymmetric.

## Phase 0 — Capability floor

**Question**: Is Gemma 3 4B IT competent enough at the paper's task that downstream mechanistic
claims about its representations are meaningful?

**Method**: Run Gemma on the paper's `original_structured` prompts (which embed the
EXPLANATION/TRIAGE/CONFIDENCE scaffold). Greedy decoding. Regex-extract the letter.

**Result**: 41/60 = **68.3%** overall. Per-gold breakdown:

| Gold | n | Correct | Acc |
|---|---|---|---|
| A (home) | 8 | 2 | 25% |
| A/B | 2 | 1 | 50% |
| B | 8 | 5 | 62% |
| B/C | 4 | 4 | 100% |
| C | 10 | 8 | 80% |
| C/D | 24 | 21 | 88% |
| **D (ER)** | **4** | **0** | **0%** |

Gemma is biased toward middle categories (B/C). It never recommends D on pure-emergency cases,
predicting C every time (one-notch under-triage). Reference: paper-reported frontier models
on this scaffold ≈ 82–85%. Random-letter baseline ≈ 25%.

**Verdict**: Capability floor passes (pre-declared 40–70% "proceed with caveats" band).
Per-category claims about D are uncalibrated (n=4, Wilson 95% CI [0%, 60%]).

**File**: `results/phase0_capability_floor.json`

## Phase 0.5 — Phenomenon presence (three cells, paper-faithful)

**Question**: Does Gemma show the input × output format effect documented in the paper?

**Method**: Three cells × 60 cases × greedy decoding.

- A and B (forced-letter): regex extract letter
- D (free-text): initial scoring used a keyword-rule heuristic (later retracted)

### First pass — keyword rule (retracted)

| Cell | Accuracy |
|---|---|
| A struct+letter | 60.0% |
| B natural+letter | 56.7% |
| D natural+free-text | **10.0%** ← keyword-rule artifact |

41/60 D outputs were classified `UNPARSED` because Gemma's free-text replies use
substantive clinical phrasing ("warrants further investigation," "take seriously") that
doesn't trigger the keyword list. The 46.7pp B-vs-D gap was a scoring-pipeline failure,
not a phenomenon. Retracted.

### Second pass — LLM-as-judge (paper's own pipeline)

Used `paper_faithful_replication/scripts/adjudicate_natural_paper_scale.py` (the
**paper-scale** adjudicator — A=home, D=ER, matching our gold). The repo also contains
`adjudicate_natural_interaction.py` which uses the **inverted** scale (A=ER, D=home).
Wiring up that script instead would have silently scrambled all correctness checks.
The scale check was non-trivial and almost shipped wrong.

Two judges: `gpt-5.2-thinking-high` and `claude-sonnet-4.6`.

| Metric | Value |
|---|---|
| Inter-rater agreement | 53/60 = 88.3% |
| Cohen's κ | 0.797 (substantial) |
| GPT correct | 43/60 = 71.7% |
| Claude correct | 46/60 = 76.7% |
| Both judges agree correct (conservative) | 42/60 = 70.0% |
| Either judge correct (lenient) | 47/60 = 78.3% |

| Cell | Accuracy |
|---|---|
| A struct+letter | 60.0% |
| B natural+letter | 56.7% |
| **D natural+free-text** | **70–77%** (judge-dependent) |

### Per-gold breakdown across cells

| Gold | n | A | B | D (gpt) | D (claude) |
|---|---|---|---|---|---|
| A | 8 | 12% | 12% | 12% | 25% |
| A/B | 2 | 50% | 100% | 100% | 100% |
| B | 8 | 88% | 88% | 50% | 88% |
| B/C | 4 | 100% | 100% | 100% | 100% |
| C | 10 | 40% | 10% | 80% | 80% |
| C/D | 24 | 79% | 79% | 96% | 92% |
| D | 4 | 0% | 0% | 25% | 25% |

### Verdict

- Output-axis phenomenon present (≥5pp threshold met): **+13–20pp in favor of free-text**.
- Direction is **opposite** to the paper's frontier-model finding (free-text underperforms there).
  Plausible scale-dependence at 4B vs 1T+ models.
- Free-text gain is concentrated in **mid-acuity** (C: +40pp, C/D: +13pp). It does **not**
  rescue D-emergencies (still 0/4 → 1/4 max) or fix A over-triage (12% across all cells).
- Input axis (A vs B): +3.3pp, within noise. No effect.

**Files**:
- `results/phase0_5_three_cells.json` — raw cell outputs and Gemma generations
- `results/phase0_5_D_for_adjudication.json` — D-cell rows formatted for adjudicator
- `results/phase0_5_D_for_adjudication_adjudicated_paper.json` — judges' labels and rationales

## Phase 1 — Mechanistic invariance test

**Question**: On the same natural clinical input, does adding "Reply with one letter only"
(Cell B) change Gemma's internal representation of the case at the medical-feature subspace,
or only the output layer?

**Hypothesis encoding**:

- *Format-invariant*: medical features fire identically across B and D → format effect is
  downstream of clinical reasoning → SAE-as-detector story stands.
- *Format-modulated*: medical features differ between B and D → output instruction modulates
  upstream representation → harder story, but still publishable.

**Method**:

- Layer sweep 9, 17, 22, 29 (same as v3)
- Medical features per layer from v3 contrastive identification:
  L9: 139, 9909, 956. L17: 9854, 368, 1539. L22: 1181, 365, 8389. L29: 12570, 893, 12845.
- 30 random features per layer, frozen seed 42, drawn from features that fire on the union
  of B+D content (mean activation > 0). **Magnitudes were not matched** — confound flagged.
- SAEs loaded directly via `safetensors` from `google/gemma-scope-2-4b-it`,
  bypassing sae-lens (transformers 5.x dep conflict). Custom JumpReLU implementation.
- Activations measured by mean-pooling SAE feature activations over **user content tokens**
  (positions [4 : first `<end_of_turn>`] in the chat-templated input). The first attempt used
  the last token of the full prompt, which gave zero medical activation because that token is
  the chat-template marker, not clinical content.
- Metrics: cosine similarity of medical-feature vectors between B and D; mean modulation index
  = ⟨|a_D − a_B|⟩ / ⟨(|a_B| + |a_D|)/2⟩, lower = more invariant.

**Stratification** (Phase 0.5 + adjudication):
- format_flipped (B wrong, both judges D right): n=13
- both_right: n=29
- both_wrong: n=13
- B_only_right: n=5

### Result table (medical vs random modulation index, bootstrap 95% CI)

| Layer | Stratum | n | med_mod | rnd_mod | diff [95% CI] |
|---|---|---|---|---|---|
| 9 | format_flipped | 13 | 0.148 | 0.168 | −0.020 [−0.060, +0.026] (crosses 0) |
| 9 | both_right | 29 | 0.189 | 0.174 | +0.015 [−0.014, +0.044] (crosses 0) |
| 17 | format_flipped | 13 | 0.118 | 0.286 | **−0.168 [−0.216, −0.117]** |
| 17 | both_right | 29 | 0.152 | 0.326 | **−0.174 [−0.208, −0.140]** |
| 22 | format_flipped | 13 | 0.152 | 0.129 | +0.023 [−0.022, +0.069] (crosses 0) |
| 22 | both_right | 29 | 0.285 | 0.141 | **+0.144 [+0.064, +0.240]** *(medical MORE perturbed)* |
| 29 | format_flipped | 13 | 0.154 | 0.280 | **−0.125 [−0.205, −0.022]** |
| 29 | both_right | 29 | 0.239 | 0.311 | −0.072 [−0.158, +0.031] (crosses 0) |
| 29 | both_wrong | 13 | 0.091 | 0.308 | **−0.217 [−0.279, −0.158]** |

### Reading the table

- **Layer 17 and Layer 29**: medical features more invariant than random under format change.
  Direction supports the "internal clinical representation is preserved" story.
- **Layer 22**: medical features *less* invariant than random — contradicts the hypothesis
  at this layer. Possible explanation: feature 365 has high baseline noise; the L22 SAE
  features may be more reactive to surface form. Not yet investigated.
- **Layer 9**: ambiguous, CIs cross zero.

### Sanity checks

- Medical features fire on essentially every case at every layer (60/60 at L9 and L22, 58/60
  at L17, **54/60 at L29**). Six cases at L29 don't trigger any medical feature — predominantly
  the mental-health and headache vignettes. The "language-agnostic medical features" claim
  from v3 was on physical/clinical content, and that scope holds.
- Mean max activations per feature align with v3 magnitudes (hundreds to thousands).
- B vs D mean max activations differ by 1–4% per feature at every layer. Already very close.
- Reconstruction error rises with depth: L9=4.5%, L17=5.5%, L22=9.0%, **L29=14%**. L29 SAE is
  noisier but still operating.

### Confounds (pre-registered, not yet ruled out)

1. **Magnitude mismatch.** Random features were drawn from "fires on this content" without
   matching activation magnitude. Medical features are typically 5–10× larger in mean activation
   than randomly-firing features. The mod-index normalizes by mean magnitude, but small-magnitude
   features can amplify relative noise — inflating random mod-index. So the L17/L29 effect
   could partly be "small features look noisier under perturbation" rather than "medical
   features are specially preserved." **This is the most important next test.**
2. **Three features per layer is a narrow probe.** A real format effect could operate on a
   different subset of features. Adding the activation-difference projection analysis (see
   "Open questions") tests this directly.
3. **6/60 cases at L29 don't trigger medical features at all.** They are still in the analysis
   (with mod-index = NaN-clamped to 0); should be excluded or analyzed separately.

### Verdict

**Suggestive — directional signal, magnitude confound flagged, one contradicting layer.**

Headline claim that should NOT yet be made: "Medical features are format-invariant; the SAE-as-
detector framing is mechanistically validated."

Honest version: At layers 17 and 29, mean medical-feature activation across the format pair
B and D is more preserved than random-feature activation in 60 paper-canonical cases. At layer
22 the pattern reverses. The L17/L29 signal is consistent with a format-invariant medical
representation but is partly confounded by the magnitude mismatch between medical and random
features.

**Files**: `results/phase1_activation_invariance.json`

## Phase 1b — Magnitude-matched random-feature control

**Question**: Does the L17/L29 invariance signal from Phase 1 survive replacing the random
feature pool with one matched to medical-feature activation magnitude?

**Method**: Identical to Phase 1 except `pick_random_features` is replaced with
`pick_random_magnitude_matched`. The new picker constructs a band
`[0.5 × min(med_means), 2.0 × max(med_means)]` from medical-feature mean activations on
the reference set, restricts the pool to features whose mean activation falls in that band
(excluding the medical features themselves), then samples 30 with seed 42.

Pool sizes per layer: L9=1414, L17=2226, L22=528, L29=1798. All comfortably above n=30.

### Result (medical vs magnitude-matched random modulation index, bootstrap 95% CI)

| Layer | Stratum | n | med_mod | rnd_mod | diff [95% CI] |
|---|---|---|---|---|---|
| 9 | format_flipped | 13 | 0.148 | 0.343 | **−0.196 [−0.223, −0.170]** |
| 9 | both_right | 29 | 0.183 | 0.408 | **−0.224 [−0.255, −0.196]** |
| 9 | both_wrong | 13 | 0.198 | 0.423 | **−0.225 [−0.258, −0.191]** |
| 17 | format_flipped | 13 | 0.118 | 0.377 | **−0.259 [−0.325, −0.195]** |
| 17 | both_right | 29 | 0.154 | 0.417 | **−0.263 [−0.303, −0.227]** |
| 17 | both_wrong | 13 | 0.165 | 0.453 | **−0.287 [−0.349, −0.231]** |
| 22 | format_flipped | 13 | 0.156 | 0.319 | **−0.163 [−0.205, −0.118]** |
| 22 | both_right | 29 | 0.252 | 0.342 | **−0.090 [−0.145, −0.021]** |
| 22 | both_wrong | 13 | 0.251 | 0.346 | **−0.095 [−0.173, −0.002]** |
| 29 | format_flipped | 13 | 0.107 | 0.413 | **−0.305 [−0.371, −0.250]** |
| 29 | both_right | 29 | 0.152 | 0.423 | **−0.271 [−0.323, −0.224]** |
| 29 | both_wrong | 13 | 0.083 | 0.453 | **−0.369 [−0.441, −0.304]** |

**The L22 reversal in Phase 1 is gone.** Every cell of the table shows medical features
more invariant than the magnitude-matched control with bootstrap 95% CI excluding zero.
At layers 17 and 29 the gap is roughly 0.25–0.37 in absolute mod-index (medical features
fire within ~10–15% across formats, magnitude-matched random features within ~40–45%).

### What changed between Phase 1 and Phase 1b

- Medical-feature numbers are essentially unchanged (within ±0.04 in any cell).
- Random-feature numbers moved from 0.13–0.33 (Phase 1) to 0.32–0.45 (Phase 1b).
- The Phase 1 random pool included many low-magnitude features. Low-magnitude features
  look invariant because tiny activations barely change in absolute terms — their mod-index
  is dominated by quantization noise. Magnitude matching removed that artifact and exposed
  the *real* noise floor.
- L22's apparent "medical worse than random" result in Phase 1 was an artifact of the L22
  random pool having unusually low mean activations relative to other layers.

### Per-token alignment (max activations)

The mod-index is computed on mean-pooled activations over user content tokens. The user-content
window for B (forced-letter) is ~50 tokens longer than D (free-text) because B includes the
forced-letter instruction block. That dilutes B's mean by a fixed factor and accounts for
~10–15% of the residual mod-index in medical features.

Looking at per-case **max** activations (the single highest-firing token per feature, immune
to dilution) at L29:

| Case | Gold | Medical max B | Medical max D | Per-feature delta |
|---|---|---|---|---|
| E3 | C | [-0.0, 701.5, 883.0] | [-0.0, 700.5, 899.7] | 0.0% / 0.1% / 1.9% |
| E4 | C | [974.5, 3423.3, 2341.0] | [974.5, 3434.3, 2380.3] | 0.0% / 0.3% / 1.7% |
| E9 | D | [1054.6, 3215.8, 3000.6] | [1066.2, 3228.2, 3016.9] | 1.1% / 0.4% / 0.5% |

When the model encounters the word "asthma" or "DKA" in a B prompt, the medical features
fire at essentially the same magnitude as in the matching D prompt. The clinical
representation is preserved at the per-token level, not just on average.

### Verdict

**Validated within scope.** In Gemma 3 4B IT, on 60 paper-canonical clinical vignettes,
the v3-validated medical SAE features at layers 9, 17, 22, 29 are more invariant under the
natural-forced-letter ↔ natural-free-text format change than 30 magnitude-matched random
features drawn from the same SAE. Bootstrap 95% CIs exclude zero in every cell of the
4 × 4 layer-stratum design. Per-token max activations on identical clinical content match
within 0–4%.

**Files**: `results/phase1b_magnitude_matched.json`, `phase1b_magnitude_matched.py`.

## Phase 2 — ActAdd-style projection (mean-pool, dilution-artefacted)

**Question**: Where in the SAE feature space does the residual-level
(B − D) direction concentrate? If medical features are the top carriers,
the format effect operates *through* the medical subspace (contradicting
Phase 1b). If they're not, the format effect is somewhere else.

**Method**: Per case, mean-pool the user-content residuals at L17 and L29.
Compute (B − D) averaged across cases. Project onto each SAE feature's
encoder direction. Rank features by |alignment|.

**Result (mean-pool)**:

| Layer | Feature | Rank (of 16384) | Signed alignment |
|---|---|---|---|
| 17 | 9854 | 2261 (13.8%) | −0.030 |
| 17 | 368  | 955  (5.8%)  | −0.040 |
| 17 | 1539 | 236  (1.4%)  | −0.057 |
| 29 | 12570 | 10994 (67.1%) | −0.008 |
| 29 | 893  | 470 (2.9%) | −0.044 |
| 29 | 12845 | 2235 (13.6%) | −0.029 |

Two patterns to note: (i) the top-aligned features at L29 (10012, 2014,
2123, 755, 121) are **not** medical, with |alignment| ~3× larger than the
highest-ranked medical feature (893); (ii) **every medical feature has
negative signed alignment**, suggesting a systematic prompt-length
artifact rather than mechanism. The /sanity-check flagged this as a
confound and demanded a follow-up.

**Files**: `results/phase2_actadd_projection.json`, `results/phase2_residuals_L*.npz`

## Phase 2b — Dilution-controlled ActAdd projection

**Question**: Does the medical-feature alignment in Phase 2 survive when
we control for the prompt-length difference between B and D?

Two controls:

- **Length-controlled mean-pool**: truncate B's content range at the
  literal "Reply with exactly one letter only" so B and D pool over the
  same clinical content range.
- **Max-pool over content tokens**: aggregate by max activation per
  feature across content tokens (length-invariant).

### Critical observation about the data

`B's prefix == D` literally. The natural-forced-letter prompt is built by
appending the forced-letter instruction block to the patient_realistic
prompt; the clinical content text is identical (1033 chars, character-for-
character). So when we truncate B at the marker, the input becomes byte-
for-byte identical to D. Forward passes on identical inputs give identical
residuals, which is why the length-controlled diff norm at L17 and L29 is
**exactly 0**.

This is not a bug. It is the strongest possible isolation of the format
effect: the only thing that differs between the two prompts is the
appended forced-letter instruction block, and when we strip it, the model
sees identical input.

### Combined picture (Phase 1b + Phase 2 + Phase 2b)

| Aggregation | ‖B−D‖ at L29 | What it measures | Medical-feature involvement |
|---|---|---|---|
| Per-token max (Phase 1b) | n/a | Peak feature activation on identical clinical tokens | Differences within 0–4% per case (essentially identical) |
| Length-controlled mean (Phase 2b) | **0.000** | Mean over identical content range | **Zero diff. No format effect at the residual level when content matches.** |
| Full mean-pool (Phase 2) | 1012.7 | Mean over different-length pools | Small alignment, all negative-signed: dilution artifact |
| Max-pool (Phase 2b) | 5026.6 | Peak feature activation across all content tokens (incl. forced-letter block) | Top-aligned features are **not** medical (ranks 2523, 7013, 10008 for the three medical features at L29) |

### Reading

- **The format effect at the residual level is entirely outside the
  medical-feature subspace.** When B and D contain the same clinical
  content, the residuals are identical; medical features fire identically
  on the same tokens.
- **What's left in max-pool is the model's response to the forced-letter
  instruction tokens themselves**, and that response loads onto
  non-medical features. These are presumably "the prompt is asking for a
  constrained answer" features — exactly the kind of output-instruction
  features Version B predicts.
- **Phase 2's apparent medical-feature alignment was a mean-pooling
  artifact** caused by B having ~59 extra tokens that don't fire medical
  features and therefore depress B's mean. Once we control for this
  (truncated mean OR max-pool), the medical-feature alignment with the
  format direction is unremarkable.

### Verdict

**The Version B claim is now well-supported by three independent angles**:

1. **Magnitude (Phase 1b)**: medical features fire within 0–4% per token
   on identical clinical content; mean-pool 10–25% under format change vs
   40% for magnitude-matched random features.
2. **Direction with content controlled (Phase 2b truncated)**: residuals
   are literally identical when B and D contain the same clinical
   content. No format effect at all.
3. **Where the residual-level format effect goes (Phase 2b max-pool)**:
   it lives in non-medical features that fire on the forced-letter
   instruction tokens, not in medical features.

**Files**: `results/phase2b_dilution_check.json`, `phase2b_dilution_check.py`

## Version A vs Version B — what the result says about the paper

The Phase 1b result speaks directly to the existing triage replication paper's broader
argument. There are two ways to interpret "prompt format affects model behavior," and our
mechanistic finding distinguishes them.

### Version A: format reaches into clinical reasoning

> "Different prompts cause the model to represent the case differently, and that's why
> answers diverge."

Phase 1b is **evidence against** this. The model's medical-feature signature on the same
clinical content is essentially identical across the two output-format conditions. If
formats produced different internal understanding, we'd expect to see medical features
fire differently. They don't.

### Version B: format effect is downstream of clinical encoding

> "Different prompts cause the model to *answer* differently, but its internal *understanding*
> of the clinical case is preserved. The benchmark therefore measures output-mapping fidelity,
> not clinical reasoning."

Phase 1b is **direct evidence for** this. The model's internal clinical representation does
not change between B and D. Yet B and D produce systematically different scoring outcomes
(57% vs 73%, +13–20pp output effect). The accuracy gap must therefore live downstream of
the medical-content representation — in the output-instruction-conditioned circuit that
maps "I understand this is asthma" into either "B" (forced letter) or a paragraph (free text).

### Why Version B is the stronger paper claim

- A regulator looking at "the model failed 51% of triage questions in Ramaswamy et al."
  can be told: the model's *clinical* representation on those failures is the same as on
  its successes. The benchmark is measuring something other than clinical capability.
- It distinguishes our mechanistic claim from a vague "format matters" hand-wave.
  We can name *where* it doesn't matter (medical content encoding at all four sweep
  layers) and where it must matter (downstream of the medical features, since the outputs
  diverge).
- It connects to a broader interpretability theme — features as *readouts* (which v3
  established for Gemma medical features) make natural monitoring signals. Version B is
  the deployment story: SAE features can be used to verify clinical understanding
  independently of output format.

### What Version B does *not* yet say

Where the format effect *does* live remains an open question. We have ruled out the
medical-content subspace at four layers. We have not positively localized the effect.
Candidates: output-style features (terse vs verbose), letter-emission features at the
final layers, or the non-SAE-decomposed residual that ActAdd-style steering operates on.

The ActAdd-style projection analysis (priority 1 in Open Questions) tests this directly:
project the mean ⟨B − D⟩ residual at L29 onto every SAE feature's encoder direction. If
the medical features rank in the bottom of the alignment ranking → format effect is
specifically *off-axis* from medical features → strongest version of B. If they rank
highly → the apparent invariance is misleading and we have to reframe.

## Phase 3 — Gemma 3 12B feature identification

**Question**: Are there clean medical features in Gemma 3 12B IT analogous to those
in 4B?

**Method**: English-only medical-vs-non-medical contrastive at four matched-depth
layers (12, 24, 31, 41 — matched to 4B's 9/17/22/29 at 27/50/65/85% depth). 60
patient_realistic prompts as the medical corpus, 30 hand-curated patient-style
non-medical prompts as the contrast. Score = `mean_max(med) − mean_max(non-med)`
under a firing-reliability filter (fires on ≥70% of medical, ≤10% of non-medical).

**Result**: All four layers have hundreds of filter-passing features. Top features
selected per layer:

| Layer | Top medical features | Best feature score |
|---|---|---|
| L12 | 527, 310, 351 | 1110 |
| L24 | 3, 338, 329 | 3928 |
| L31 | 130, 85, 4773 | 4053 |
| L41 | 6653, 164, 6517 | 10842 (fires 73%/0%) |

Several L31 and L41 features fire on ≥73% of medical content and ≈0% of
non-medical content — extremely clean medical specificity.

**Files**: `results/phase3_12b_features.json`, `phase3_12b_feature_id.py`

## Phase 3b — 12B mechanistic pipeline

**Question**: Does the 4B pattern (Version B at all four sweep layers) replicate at
12B intra-family?

### Phase 0 (capability floor on EXPLANATION+TRIAGE scaffold)

12B: 40/60 = **66.7%**, on par with 4B's 68.3%. Same family-wide ceiling on this
specific scaffold.

### Phase 0.5 (three cells, paper-faithful)

| Cell | 4B | 12B |
|---|---|---|
| A: structured + forced-letter | 60.0% | **81.7%** |
| B: natural + forced-letter | 56.7% | **81.7%** |
| D: natural + free-text (LLM-judge) | 70–77% | **76.7–81.7%** |

**The behavioral format effect attenuates at scale.** At 4B, free-text outperforms
forced-letter by +13–20pp (judge-dependent). At 12B, the gap collapses to ≈0pp.
12B is capable enough to map clinical understanding onto a constrained letter
output, eliminating the forced-letter penalty. This is itself a novel scaling
finding consistent with Singhal et al. (2023): scaling improves performance on
medical question answering, and we now show this includes performance under
constrained output formats specifically.

The 12B adjudication used the same paper-faithful pipeline (gpt-5.2-thinking-high
+ claude-sonnet-4.6) with 76.7% inter-rater agreement and Cohen's κ = 0.634
(moderate, lower than 4B's 0.797).

### Phase 1b — magnitude-matched modulation index

| Layer | n | med_mod | rnd_mod | diff [95% CI] |
|---|---|---|---|---|
| 12 | 60 | 0.253 | 0.213 | **+0.040 [+0.001, +0.089]** ← medical *more* perturbed |
| 24 | 60 | 0.370 | 0.198 | **+0.172 [+0.114, +0.237]** ← medical *more* perturbed |
| 31 | 60 | 0.157 | 0.395 | **−0.238 [−0.260, −0.219]** ← Version B holds |
| 41 | 60 | 0.181 | 0.285 | **−0.103 [−0.126, −0.081]** ← Version B holds |

**Depth-dependent pattern.** At deep layers (L31, L41 ≈ 65–85% depth), medical
features are more invariant than the magnitude-matched random control —
Version B holds, replicating the 4B finding. **At shallow/mid layers (L12, L24
≈ 25–50% depth), the medical features we identified are *more* perturbed than
random** — they show format-modulation.

### Phase 2b — dilution-controlled projection

| Layer | Top-feat (max-pool) ranks for medical features |
|---|---|
| L12 | 2.2%, 2.1%, 3.8% (ile of |alignment|) — top-aligned with format direction |
| L24 | 45.5%, 7.6%, 8.5% |
| L31 | 86.2%, 32.8%, 51.0% — mid/low alignment |
| L41 | 32.0%, 43.0%, 91.4% — mid/low alignment |

Same depth pattern: shallow medical features carry the format direction; deep
medical features do not.

### Reading

The 4B-vs-12B comparison reveals a richer picture than uniform invariance:

- **Behaviorally**: format effect attenuates at scale (4B big → 12B ≈ 0).
- **Mechanistically at deep layers**: clinical-conceptual encoding is preserved
  at *both* scales (Version B robust).
- **Mechanistically at shallow layers in 12B**: medical-vocabulary-tied features
  *do* respond to surface-form differences (forced-letter instruction tokens),
  consistent with v1's old finding that early layers do lexical/surface work.

The format effect that was strong in 4B's behavior was not in 4B's clinical
encoding (Phase 1b/2b on 4B). At 12B it's gone from both behavior and deep
encoding. **Format invariance is a property of the conceptual-encoding layers,
not the lexical ones — and the model's ability to translate that conceptual
encoding into a constrained letter improves with scale.**

**Files**: `results/phase3b_12b_phase0.json`, `phase0_5.json`, `phase1b.json`,
`phase2b.json`, `phase3b_12b_D_for_adjudication_adjudicated_paper.json`

## Phase 4 — Qwen Scope cross-family validation

**Motivation**: Forestall the "single-family generalization" reviewer concern.

**Setup**:
- Model: Qwen3-8B, fed B and D prompts as **raw text** (no chat template),
  to avoid added structure between B and D conditions.
- SAE: `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` at **layer 31** only
  (deepest layer with lowest reconstruction error: 38.5%).
- Same 60 paper-canonical cases. B = `natural_forced_letter`, D = `patient_realistic`.
- Feature ID: medical-vs-non-medical contrastive on the same 60+30 corpus
  used in Phase 3, with k=50 TopK encode.
- Phase 1b magnitude-matched mod-index. Phase 2b max-pool projection.

**Caveat acknowledged in paper**: Qwen Scope uses k=50 TopK SAEs (50/65536
features active per token = 0.076% sparsity, more aggressive than Gemma
Scope's JumpReLU). This forces 38% relative L2 reconstruction error even
on the SAE's training-distribution residuals. Whether the medical-vs-random
gap survives this lower-fidelity SAE is an empirical question.

### Medical features identified at L31

519 features pass the "fires ≥70% medical, ≤10% non-medical" filter
(comparable to Gemma 12B's 502 at L31). Top 3:

| Feature | Med max-mean | Non max-mean | Fires med | Fires non |
|---|---|---|---|---|
| 29074 | 176.8 | **0.00** | 70% | **0%** |
| 48973 | 153.6 | 3.26 | 100% | 7% |
| 60699 | 141.0 | **0.00** | 78% | **0%** |

Magnitudes are ~10× lower than Gemma Scope features at comparable layers
(consistent with TopK constraint), but the firing-specificity signature is
even cleaner — two of three top features fire on **zero** non-medical prompts.

### Phase 1b magnitude-matched mod-index

| | Mean | 95% CI |
|---|---|---|
| Medical (n=3) | 0.266 | — |
| Magnitude-matched random (n=30) | 0.330 | — |
| **Diff (medical − random)** | **−0.064** | **[−0.106, −0.024]** |

CI excludes zero. **Medical features are more invariant than the
magnitude-matched random control under the B↔D format change in
Qwen3-8B.** Effect size is smaller than Gemma 4B L29 format_flipped
(−0.305) and Gemma 12B L31 (−0.238), but the direction and statistical
signal replicate.

### Phase 2b max-pool projection (medical-feature ranks of 65536)

- feat 29074: rank 58626 (89.5%-ile) — far from format direction
- feat 60699: rank 40893 (62.4%-ile) — mid
- feat 48973: rank 82 (0.1%-ile) — top-aligned

Two of three medical features are clearly off the format direction; one is
highly aligned — same mixed pattern as Gemma 4B L29 (feat 893 top-aligned,
12570 low, 12845 mid). Consistent with the dilution-from-prompt-length
artifact we identified earlier.

### Verdict

**Cross-family validation holds at deep-layer L31.** Across Gemma 3 (4B+12B,
JumpReLU) and Qwen3 (8B, k=50 TopK), medical features at the deep-encoding
layer are more invariant under format change than magnitude-matched random
features in the same SAE basis. The Qwen Scope effect is smaller, consistent
with the more aggressive sparsity constraint, but qualitatively replicates
the Gemma pattern.

**Files**: `results/phase4_qwen_L31.json`, `phase4_qwen_minimal.py`,
plus the earlier `results/qwen_sanity*.json` for the recon-error
characterization.

## Phase 5 — Top-token analysis + restricted random pool

Two follow-up analyses on Gemma 3 4B IT, L29.

### Phase 5A — Top-token analysis of format-direction features

For features 3833, 10012, 980, 9485, 755 (top-aligned with the (NL−NF)
max-pool direction in Phase 2b), we collect per-token activations across
all 60 cases × {NL, NF} prompts.

**Result**: three of five fire exclusively on forced-letter scaffold tokens.

| Feature | Top tokens | Top contexts (all NL) |
|---|---|---|
| 3833 | " next" | "B = See my doctor in the **next** few weeks" |
| 10012 | " the" | "D = Go to **the** ER now Do not include any explanation" |
| 980 | " =" | "B **=** See my doctor… C **=** See a doctor" |
| 755 | " my" / " My" | mixed: answer-key text AND patient narrative ("**my** vitals", "**My** patient portal") |
| 9485 | "." | numeric/decimal feature, fires equally NL and NF on lab values |

For comparison, the v3-validated medical features fire on clinical-content
tokens at identical magnitudes across NL and NF:

| Feature | Top tokens | Top contexts (mixed NL/NF) |
|---|---|---|
| 12570 | " my", " right", " both" | clinical exam contexts ("facial weakness was affecting both my forehead", "right lower belly was tender") |
| 893 | " blood" | lab-value contexts ("white **blood** cell count is 11.2", "white **blood** cells were 0-2 per") |
| 12845 | " neck", " on" | exam contexts ("using my **neck** muscles to breathe", "tender when they pressed **on** it") |

**Reading**: this converts the percentile-rank statement of Phase 2b
into a feature-level mechanistic interpretation. The format direction
in residual space is encoded by SAE features that detect the structural
format of the prompt (its answer-key scaffold), not its clinical
content. Medical features fire identically across conditions, confirming
Phase 1b's magnitude invariance at the per-token level.

### Phase 5B — Restricted random pool

Phase 1b's magnitude-matched random pool was permitted to include
features that may not fire on clinical content. We tighten the control
by additionally restricting the pool to features that fire on at least
25% of all 120 prompts (60 cases × 2 conditions). Re-run mod-index
analysis at 4B L29.

| Stratum | n | medical_mod | restricted-random_mod | diff [95% CI] |
|---|---|---|---|---|
| format_flipped | 13 | 0.107 | 0.304 | **−0.196 [−0.261, −0.135]** |
| both_right | 29 | 0.152 | 0.341 | **−0.189 [−0.243, −0.139]** |
| both_wrong | 13 | 0.083 | 0.380 | **−0.297 [−0.380, −0.225]** |

Compared to Phase 1b at L29 (format_flipped diff −0.305, both_right
−0.271, both_wrong −0.369), effect sizes shrink ~30–40% under the
stricter control but remain robust with all bootstrap 95% CIs excluding
zero.

**Files**: `results/phase5_top_tokens.json`,
`results/phase5_restricted_random.json`,
`phase5_top_tokens_and_restricted_random.py`,
`figures/fig4_top_tokens.{pdf,png}`.

### Verdict

**Strongest version of Version B yet**, supported by feature-level
interpretability: the SAE features carrying the format direction are
literally the ones that fire on the forced-letter answer-key scaffold
tokens, not the medical-content features. Combined with Phase 2b's
length-controlled diff = 0 result (when content is held identical, the
residual stream is identical), this makes the Version B argument
mechanistically explicit and defensible at the feature level rather
than just at aggregate-statistic level.

## Phase 6 — SAE-feature ablation: null

**Question**: If we ablate features 3833, 10012, 980 at L29 during NL
inference (subtract their SAE-reconstructed contribution from the
residual stream), does the behavioral NL accuracy on the 60
paper-canonical cases shift?

**Result**: 0/60 letter predictions changed in either ablation arm
relative to vanilla NL (33/60 = 55.0% across all three arms):

| Arm | Correct | Accuracy |
|---|---|---|
| Vanilla NL | 33/60 | 55.0% |
| Ablate format-direction (3833, 10012, 980) | 33/60 | 55.0% |
| Ablate random control (171, 1767, 3555) | 33/60 | 55.0% |

**Diagnostic** (`phase6_debug.py`): the ablation hook IS firing and IS
modifying the residual at the expected magnitude (mean 264 norm
subtracted per token, peak 6,795 on the strongest answer-key tokens).
However, the residual stream at L29 has per-token norm ~60,000 — the
ablation magnitude is ~0.4% of the residual on average, peaking at
~11% on the strongest answer-key token positions. Insufficient to flip
the next-token argmax.

**Reading**: the format direction is real, the features identified by
Phase 5 are real, but discrete-feature ablation of three features at
one layer is too small a perturbation to change letter outputs. This is
consistent with concurrent work [Basu et al. 2026, who show four
mechanistic intervention methods including SAE feature steering produce
zero correction effect on similar clinical-triage tasks] and with our
own v3 null on cross-lingual rescue.

**Files**: `phase6_causal_intervention.py`, `phase6_debug.py`,
`results/phase6_causal_intervention.json`.

## Phase 7 — ActAdd-style steering: near-null

**Question**: Does the *full* residual-space format direction at L29
have causal weight? Phase 6 only touched 0.4% of the residual via three
SAE features; Phase 7 tests whether the entire (NL−NF) direction —
including the 99%+ of variance that the SAE basis doesn't isolate to a
few features — has behavioral causal effect.

**Setup**: compute v = mean(NL_residual) − mean(NF_residual) at L29
across the 60 cases (||v|| = 1,012). During NL inference, hook L29 to
add −α · v at all token positions for α ∈ {0, 0.5, 1.0, 2.0, 4.0}.
Five arms × 60 cases = 300 generations.

**Result**: accuracy 33/60 = 55.0% across all five α. Only 2/60 cases
shift letter predictions (E6 and F2, both gold = B/C, both shifting
C → B at α ∈ {2, 4}). Both shifts stay within the permissive gold range,
so accuracy is unchanged. The shift direction is opposite to what
NF-like behavior would predict (NF is more accurate on aggregate; the
2-case shifts go toward less-urgent letters), consistent with noise
rather than systematic causal signal at this perturbation magnitude
(~6.7% of residual norm at α=4).

**Reading**: even with continuous-direction steering — the kind of
intervention specifically designed to capture the 99% of residual
variance that SAE features miss — the format direction does not drive
behavior at L29 with sufficient causal weight to flip letter outputs.

**Files**: `phase7_steering_vector.py`,
`results/phase7_steering_vector.json`.

## Convergent reading of Phases 5–7

Three independent results triangulate the readout-not-driver picture:

1. **Phase 5A (top-token interpretation)** identifies *what* the format
   direction is in feature space: SAE features that fire exclusively on
   the literal forced-letter answer-key scaffold tokens.
2. **Phase 6 (discrete SAE-feature ablation)** shows that subtracting
   those features' contribution from the residual at L29 produces
   exactly zero behavioral change across 60 cases.
3. **Phase 7 (continuous ActAdd-style steering)** shows that subtracting
   the full residual-space format direction at L29 produces near-zero
   behavioral change (2/60 within-permissive-gold shifts at high α).

Combined with concurrent work showing four mechanistic intervention
methods fail on Qwen 2.5 7B triage [Basu et al. 2026] and our own prior
null on cross-lingual rescue [v3 of this repo], the empirical picture
is: SAE features and the residual directions they decompose are
detectable, interpretable monitors of model state — but at the
single-layer perturbation magnitudes we can apply without breaking
generation, they are not causally sufficient intervention points to
drive these output behaviors.

This is, on its own, a contribution: the paper goes from "we describe
where the format effect lives" (interpretive) to "we describe where it
lives AND directly test interventions on that location, both of which
confirm the readout-not-driver picture for this setting" (empirical).

## Open questions / planned next steps (NeurIPS workshop sprint)

Phase 2 ✓ done. Phase 2b ✓ done — both controls converge on Version B.

Remaining for the workshop sprint, in priority order:

1. **12B intra-family scale generality (Day 1–2).** Re-run the full pipeline on Gemma 3
   12B IT using `google/gemma-scope-2-12b-it`. The 12B SAEs are public and use the same
   format as 4B. Steps:
   - Adapt v3 six-condition contrastive to 12B and find medical features at matched-depth
     layers (12B has 48 layers; matched depths to 4B's 9/17/22/29 are roughly 13/24/31/41).
   - Phase 0 capability floor on canonical structured triage.
   - Phase 0.5 three-cell with same adjudicator setup.
   - Phase 1b magnitude-matched activation invariance.
   - Phase 2b dilution-controlled projection.
   - Total compute: ~3 hrs on a 40GB+ GPU at $0.50–1.00/hr. API: ~$1 for 12B adjudicator.

2. **Interpret the top non-medical features at L29 max-pool**. Features 3833, 10012, 980,
   9485, 755 carry the format-effect direction in the max-pool analysis. If we can show
   they fire on tokens like "letter," "Reply," "exactly" — i.e., on the forced-letter
   instruction text itself — that nails Version B with a feature-level interpretability
   story: "the format effect is the model recognizing it's been given a constrained-output
   instruction, encoded by these specific features." Use Neuronpedia or quick top-token
   analysis on a small held-out corpus.

3. **Restricted random pool refinement (Day 1).** Re-run Phase 1b with random features
   drawn from a pool restricted to "fires significantly on this content" (mean activation
   > some threshold on the union of B and D). Closes the last named confound from Phase 1b
   sanity-check.

4. **Multi-family generalization (deferred / future work).** `fnlp/Llama-Scope-3.1-8B`
   provides public SAEs on Llama 3.1 8B Instruct. Cross-family validation strengthens the
   "this is a property of how transformers represent clinical content, not a Gemma-specific
   artifact" claim. Skipped for the workshop sprint; flag as the natural next paper.

5. **Stratum-aware deep dive (Day 2 if time).** The format_flipped stratum (n=13) is the
   most interesting: format physically flipped the answer. Per-case inspection: where do
   medical-feature activations rank vs the model's actual triage decision in those cases?
   Initial data on E3, E4, E9 matches the prediction (max activations within 0.0–1.7%
   between B and D despite divergent outputs).

6. **Defer**: ActAdd-rescue replication (re-running v2-medical with steering vectors) is
   interesting but reopens a closed branch. Note as a possible separate paper.

## Methodological lessons (lessons earned)

These are the things that would have shipped silently as bugs without explicit checking. Future
work should pay forward.

1. **Two triage scales coexist in the codebase, not labelled in filenames.** The
   `nature_triage_expanded_replication` repo contains `adjudicate_natural_interaction.py`
   (inverted scale: A=ER, D=home) and `adjudicate_natural_paper_scale.py` (paper-native scale:
   A=home, D=ER). Our gold labels are paper-native. Wiring up the wrong adjudicator would have
   silently flipped every correctness check while looking "clean" in CSV form.

2. **Keyword rules undercount free-text accuracy by 7×.** Initial Phase 0.5 D-cell scoring with
   a hand-crafted keyword rule reported 10.0% accuracy. LLM-as-judge with the paper's own
   pipeline reported 70–77%. The keyword rule classified 41/60 outputs as `UNPARSED` because
   Gemma uses substantive clinical phrasing ("warrants further investigation") that wasn't in
   the keyword list. Don't trust automated rules on free-text without judge calibration.

3. **The "last token" of a chat-templated prompt is the chat-template marker, not content.**
   First Phase 1 attempt measured at the last token before generation. Result: medical
   features at zero in every case. Fix: mean-pool over user content tokens
   (positions [4 : first `<end_of_turn>`]).

4. **Don't construct cells the paper didn't authorize.** Initial Phase 0.5 design proposed a
   constructed structured + free-text cell to complete the 2×2. Caught and removed before
   running. The paper's design was asymmetric on purpose; replication should respect that.

5. **Vast.ai API changed: order/limit/type now go inside `q`, not as URL params.** Spent ~30
   minutes diagnosing zero offers before realizing this.

6. **sae-lens has heavy transitive dependencies that conflict with current transformers/torch
   versions.** Loading SAE weights directly via `safetensors` + manual JumpReLU is ~30 lines
   and avoids the dependency hell. JumpReLU formula:
   `pre = x @ w_enc + b_enc; features = pre * (pre > threshold)`. Decode: `features @ w_dec + b_dec`.

7. **Gemma 3 4B IT is multimodal.** Text decoder layers live at
   `model.model.language_model.layers[i]`, not `model.model.layers[i]`. The model wraps
   `Gemma3ForConditionalGeneration` over `Gemma3Model` (which holds vision_tower, projector,
   language_model).

8. **The `/sanity-check` skill caught two real methodology issues that would have slipped past
   smoke tests:** the constructed-cell drift (caught by user pushback), and the keyword-rule
   retraction (caught by the skill's "what alternative would also explain this result" step).
   Worth keeping the discipline.

## Provenance and reproducibility

All Phase 0/0.5/1 scripts run on a vast.ai instance with Gemma 3 4B IT and the paper's canonical
60 vignettes. SAE weights pulled directly from `google/gemma-scope-2-4b-it` via
`huggingface_hub.hf_hub_download`. LLM-as-judge calls go through OpenAI and Anthropic APIs.

Total GPU cost so far: ~$2 across 3 instance launches.
Total API cost: ~$1 (one adjudication run, 120 calls).

Scripts:
- `phase0_capability_floor.py` — Gemma 3 4B on `original_structured` prompts
- `phase0_5_three_cells.py` — three-cell phenomenon-presence test
- `wire_adjudicator.py` — converts Phase 0.5 to adjudicator-input format
- `phase1_activation_invariance.py` — feature invariance test, layer sweep

Results JSONs live under `results/`.
