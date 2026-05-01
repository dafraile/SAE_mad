# SAE Features as a Format-Invariance Detector for Medical Triage

A continuation of the SAE_mad project after v3 closed the rescue-by-amplification branch.
Documents the experimental record from the pivot toward "SAE features as a groundedness
/ format-invariance monitor" using the triage replication dataset.

**Status**: Phase 0, 0.5, 1, and 1b complete on Gemma 3 4B IT. The magnitude-matched
follow-up (Phase 1b) closes the main confound from Phase 1 and the result is robust at
all four sweep layers. **Targeting NeurIPS workshop submission, 4-day sprint.** Next steps
(in priority order): ActAdd-style projection analysis on the full feature space; restricted
random pool refinement; 12B confirmation for intra-family scale generality.

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
- **Mechanistic finding (Phase 1b)**: medical SAE features at layers 9, 17, 22, 29 are
  significantly more invariant under the B↔D format change than 30 magnitude-matched random
  features drawn from the same SAE. Bootstrap 95% CIs exclude zero across all 4 layers × 4 strata.
  At the per-token level, max activations of medical features on the same clinical content
  match within 0–4% across formats.
- **Interpretation (Version B)**: The format effect lives **downstream of clinical encoding**.
  The model's internal "this is asthma / this is DKA" representation is unchanged by the
  output instruction. The accuracy difference between B (forced letter) and D (free text)
  is therefore an output-mapping effect, not a clinical-reasoning effect. This is a
  *stronger* version of the paper's behavioral claim: the benchmark is producing apparent
  clinical failures from intact clinical representation.

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

## Open questions / planned next steps (NeurIPS workshop sprint)

In priority order for the 4-day workshop sprint:

1. **ActAdd-style activation-difference projection (Day 0–1).** Compute the mean ⟨B − D⟩
   residual vector at L29 (and L17) across all 60 cases. Project that direction onto each
   SAE feature's encoder direction. Rank features by alignment magnitude. Check where the
   medical features (12570, 893, 12845 at L29) land:
   - **Bottom of ranking → strongest Version B**: format effect is specifically off-axis
     from medical features in the full 16k-feature space, not just within the 3-feature
     medical subspace.
   - **Top of ranking → contradicts Phase 1b**: medical features actually carry the
     format-difference direction; the apparent invariance is misleading, reframe needed.
   - Pure analysis on saved residuals; ~30 min wall on a small GPU to extract residuals if
     they aren't already cached, then local CPU compute.

2. **Restricted random pool (Day 1).** Re-run Phase 1b with random features drawn from a
   pool restricted to "fires significantly on these vignettes" (e.g. mean activation > 5
   on the union of B and D). This closes the last named confound from the Phase 1b
   sanity-check: that magnitude-matched random features may fire on tokens that are
   non-clinical and therefore see different perturbations under format change. ~1 hr GPU.

3. **12B intra-family scale generality (Day 1–2).** Re-run the full pipeline on Gemma 3
   12B IT using `google/gemma-scope-2-12b-it`. The 12B SAEs are public and use the same
   format. Steps:
   - Adapt v3 six-condition contrastive to 12B and find medical features at matched-depth
     layers (12B has 48 layers; matched depths to 4B's 9/17/22/29 are roughly 13/24/31/41).
   - Phase 0 capability floor on canonical structured triage.
   - Phase 0.5 three-cell with same adjudicator setup.
   - Phase 1b magnitude-matched activation invariance.
   - ActAdd projection on 12B L41 (or whichever layer gives the cleanest medical-feature
     identification).
   - Total compute: ~3 hrs on a 40GB+ GPU at $0.50–1.00/hr. API: ~$1 for 12B adjudicator.

4. **Multi-family generalization (deferred / future work).** `fnlp/Llama-Scope-3.1-8B`
   provides public SAEs on Llama 3.1 8B Instruct. Cross-family validation strengthens the
   "this is a property of how transformers represent clinical content, not a Gemma-specific
   artifact" claim. Skipped for the workshop sprint; flag as the natural next paper.

5. **Stratum-aware deep dive (Day 2 if time).** The format_flipped stratum (n=13) is the
   most interesting: format physically flipped the answer. Per-case inspection: where do
   medical-feature activations rank vs the model's actual triage decision in those cases?
   For Version B to be the right framing, format-flipped cases should show *identical*
   medical activations across B and D despite divergent outputs. Initial data on E3, E4, E9
   matches this prediction (max activations within 0.0–1.7%).

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
