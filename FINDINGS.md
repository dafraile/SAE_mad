# SAE-Guided Capability Routing: Experimental Findings

**Status**: Project concluded. Rescue hypothesis not supported. Some real mechanistic findings remain.

## Research Question

Can Sparse Autoencoder (SAE) features serve as routing signals for selectively activating shared-parameter computation in LLMs, analogous to how biological brains route information through cooperating subsystems?

The applied question we ended up testing: **can amplifying SAE features at inference time rescue cross-lingual performance gaps on knowledge-intensive tasks?**

Short answer: **No, not via single-layer feature amplification.** Features that represent domain content exist and are identifiable, but ablating them doesn't hurt task performance and amplifying them doesn't improve it. The representation exists; the causal pathway from representation to answer selection does not go through these features in a way we can exploit.

## Setup

- **Models**: `google/gemma-3-1b-it` (26 layers, d_model=1152) for initial steering; `google/gemma-3-4b-it` (34 layers, d_model=2560) for the knowledge-transfer experiments
- **SAEs**: Gemma Scope 2 residual-stream SAEs, layer 22 (1B) and layer 29 (4B), 16k width, medium L0
- **Benchmark**: MMLU/MMMLU for cross-lingual MCQ evaluation
- **Real English**: `cais/mmlu` (see critical note below)

## Critical Methodological Note (added in v3)

Our initial v2-medical experiments used MMMLU's `default` config as "English". **This config is not English** — it contains all 14 non-English language translations concatenated, with Arabic first. Our "English" filter returned the first N items per subject, which are Arabic. The initial finding of a "reverse gap" (non-English > English) was therefore Arabic-vs-Spanish, not English-vs-Spanish.

Additional methodological issues in v2 surfaced by external review:
- "Single feature" claim was actually top-3 or top-10 features
- Net rescue was measured on a small subset, not the full benchmark
- No random-feature control
- Subject+position pairing between MMMLU configs was assumed but not verified

v3 addressed all of these. Quantitative rescue claims from v2-medical are retracted.

---

## What survives: validated claims

### 1. Cross-lingual representation analysis (v1)
Characterization of how Gemma 3 1B represents semantically equivalent content across EN/ES/FR using our own parallel corpus. Key findings:
- Middle layers are strikingly language-agnostic (cosine similarity 0.999+)
- SAE features expose structure that raw residual similarity hides
- Language/culture entanglement is partial and has a depth signature: loanword-driven at shallow layers (lexical routing), concept-driven at late layers

Intact. Used our own corpus, not contaminated by the MMMLU issue.

### 2. Language-output steering on Gemma 3 1B (v2)
Feature 857 at layer 22 causally steers output language: clamping it at 2x-5x of its natural activation on neutral English prompts produces smoothly graded Spanish output. Features 1207 and 3201 similarly steer toward French with different dynamic ranges.

Intact. This is a clean, small result: SAE features can be used to control the model's output language. It does NOT extend to controlling knowledge retrieval.

### 3. Real multilingual baselines on Gemma 3 4B (v3)
Measured against proper English from `cais/mmlu`:

| Language | Medical accuracy | 95% CI | Gap vs EN |
|----------|-----------------|--------|-----------|
| English  | **58.4%** | [55.1, 61.5] | — |
| Spanish  | 54.3% | [51.2, 57.6] | +4.1% |
| French   | 52.8% | [49.5, 56.1] | +5.6% |
| Hindi    | 48.5% | [45.5, 51.6] | +9.9% |
| Arabic   | 43.2% | [40.1, 46.0] | **+15.2%** |
| Swahili  | 39.5% | [36.5, 42.4] | **+18.9%** |
| Yoruba   | 31.2% | [28.5, 34.2] | **+27.2%** |

English dominates as the literature predicts. The gap grows substantially for lower-resource languages.

### 4. Language-agnostic domain-selective features exist (v3 domain + validation)
Using 6-condition contrastive analysis (EN/ES/FR × medical/non-medical), we identified features that fire on medical content across all three trusted languages and zero on non-medical content:

| Feature | EN med | ES med | FR med | EN nonmed | ES nonmed | FR nonmed |
|---------|--------|--------|--------|-----------|-----------|-----------|
| 893 | 833.6 | 541.8 | 536.2 | 0.0 | 0.0 | 0.1 |
| 12570 | 606.0 | 365.7 | 330.0 | 0.0 | 0.0 | 0.0 |
| 12845 | 234.7 | 149.0 | 128.6 | 0.0 | 0.0 | 0.0 |

These features also fire on Arabic medical content (weakly) and on free-form non-MCQ medical text (e.g., peaks on "myocardial", "renin", "ventral", "coccus"), and zero on free-form non-medical text (philosophy, history, economics, literature, geography). They are genuine cross-lingual cross-format medical content representations.

### 5. These features are not individually necessary under layer-29 ablation (v3 feature validation)
Ablating features 893, 12570, and 12845 during MCQ inference — zeroing out their contribution to the residual stream at layer 29 — changed accuracy by **exactly 0.00%** across EN, ES, FR, on both medical and non-medical conditions.

Precise claim: these features are **not individually necessary for medical MCQ accuracy under single-feature layer-29 ablation**, and they are **not sufficient for rescue under single-layer amplification** (established in v3 domain rescue above). "Readouts, not drivers" is a fair shorthand for this intervention regime.

### 6. The readout pattern generalizes across depth (v3 layer sweep)
To address whether the layer-29 null was an artifact of that specific layer choice (middle layers might host the actual causal pathway for knowledge retrieval), we repeated the identification + ablation procedure at layers 9, 17, 22, and 29:

| Layer | Depth | Avg med Δ | Avg non-med Δ | Med-specific |
|-------|-------|-----------|---------------|--------------|
| 9 | ~27% | 0.00% | -0.17% | -0.17% |
| 17 | ~50% | -0.11% | 0.00% | +0.11% |
| 22 | ~65% | -0.25% | 0.00% | **+0.25%** |
| 29 | ~85% | 0.00% | 0.00% | 0.00% |

Baseline 95% CIs are ~±3 percentage points wide. Every delta observed is at least 10x smaller than the CI width. The directional pattern at layer 22 (small medical drop, zero non-medical drop) is suggestive but not significant at this intervention strength.

Each layer's contrastive analysis finds its own distinct set of clean medical-selective features (different feature indices at every layer, all with the same "zero on non-medical" property). Medical content is redundantly represented across depth, but single-feature ablation at any of these layers does not measurably affect MCQ accuracy.

The v3 readout/null conclusion therefore holds **across all four tested layers**, not just layer 29.

---

## What is retracted

- **v2-medical quantitative rescue claims** (+3 to +9 net rescues, "3-7% per cell", "single feature rescues cross-lingual gaps"): artifacts of Arabic mislabeled as English, subset evaluation, and no random-feature control
- **Reversed multilingual gap** (ES > EN): was Arabic < Spanish, not English < Spanish
- **"Cross-lingual knowledge transfer via SAE feature steering"** as the paper's headline claim: not supported by the controlled replication

## Null results (established under controlled conditions)

| Experiment | Feature type | Victim language | Net effect vs random control |
|-----------|--------------|----------------|------------------------------|
| v3 replication | ES language features → EN medical | English | Within noise floor (±0.2%) |
| v3 low-resource | EN language features → YO medical | Yoruba | Within noise floor |
| v3 low-resource | ES language features → YO medical | Yoruba | Within noise floor |
| v3 domain rescue | Language-agnostic med features → AR medical | Arabic | Within noise floor |
| v3 domain rescue | Language-agnostic med features → YO medical | Yoruba | Within noise floor |

Full-benchmark evaluation (all paired items), bootstrap 95% CIs, matched random-feature control in every case. **No experimental condition produced rescue above the random-feature baseline.**

---

## Interpretation

The routing hypothesis at its simplest — amplify the right feature, retrieve the knowledge — is **not supported at the single-layer-amplification scale in Gemma 3 4B**. The findings constrain what the mechanism would have to look like if it exists:

1. Features that *represent* a domain can exist (893, 12570 are genuine medical content features)
2. Features that *control output language* exist (Feature 857 on 1B)
3. These two capacities do not obviously compose into "features that control knowledge retrieval" through the interventions we tested
4. Layer-29 single-feature ablation of these medical-readout features is a zero-op on MCQ accuracy, which means the model does not individually require them at this layer for answer selection. That is a narrower statement than "the model is not using these features" — the features could still contribute under multi-feature, multi-layer, or task-conditional interventions we did not run.

Possible reasons amplification failed that we did not test:
- Knowledge access might require coordinated changes across multiple layers
- SAE features might be lossy: the real knowledge-driving pathway might be in the residual components the SAE doesn't capture
- The intervention point (layer 29, ~85% depth) might be too late — answer computation may have already occurred
- Continuous steering vectors (activation differences) rather than discrete features might preserve more of the signal
- Redundancy: other features may carry the same information, so single-feature ablation is masked

These are hypotheses; we did not test them.

### Concluding statement

We identify highly selective, language-agnostic medical SAE features in Gemma 3 4B that activate across English, Spanish, and French, and more weakly in lower-resource languages such as Arabic and Yoruba. However, under controlled evaluation, neither amplifying nor ablating these features changes medical MCQ accuracy above the random-feature noise floor. These features appear to function as interpretable cross-lingual readouts of medical content rather than direct control points for capability transfer under simple single-layer intervention. Whether more elaborate interventions (multi-layer, coordinated, or task-conditional) could use them for rescue is an open question this project did not answer.

---

## Experimental Timeline

- [x] **v1**: Cross-lingual representation exploration (corpus-based)
- [x] **v2 steering**: Causal language-output control on 1B (Feature 857 → Spanish output)
- [x] **v2 medical**: Original rescue story — **results retracted**
- [x] **v3 replication**: Controlled re-run with real English → null
- [x] **v3 low-resource rescue**: EN/ES features → AR/YO → null
- [x] **v3 domain rescue**: Language-agnostic medical features → AR/YO → null
- [x] **v3 feature validation**: Target features are readouts, not drivers (ablation = 0)
- [x] **v3 layer sweep**: Readout pattern replicates at layers 9, 17, 22, 29 — confirms the null is not layer-29-specific
- [x] **Project closure**: This document

---

## Key Files

### Documentation
| File | Purpose |
|------|---------|
| `README.md` | Repo overview + reproduction instructions |
| `FINDINGS.md` | This document |
| `sae_routing_experiment_handoff.md` | Original research plan |

### Data
| File | Purpose |
|------|---------|
| `corpus.json` | v1 parallel trilingual corpus (EN/ES/FR) |

### Scripts
| File | Purpose |
|------|---------|
| `hw{1-5}_*.py` | Hello-world validation (model/SAE loading) |
| `v1_exploration.py` | Cross-lingual similarity and feature characterization |
| `v2_steering.py` | Language-output steering on 1B (VALID RESULT) |
| `v2_medical_pilot.py` | 4B baseline — used broken MMMLU "default" |
| `v2_medical_rescue.py` | Original rescue — **retracted** |
| `v2_medical_rescue_v2.py` | Targeted features — **retracted** |
| `v2_generalization.py` | Cross-domain/lang rescue — **retracted** |
| `v2_flip_distant_combined.py` | Flip/combined — **retracted** |
| `v3_replication.py` | Controlled replication with real English, full benchmark, random control |
| `v3_lowresource_rescue.py` | EN→weak-language rescue attempts |
| `v3_domain_rescue.py` | Language-agnostic medical feature identification + rescue |
| `v3_feature_validation.py` | Three-test validation: top tokens, ablation, free-form medical |
| `v3_layer_sweep.py` | Ablation across layers 9, 17, 22, 29 — confirms null is not specific to layer 29 |

### Infrastructure
| File | Purpose |
|------|---------|
| `vast_gpu.sh` | Vast.ai instance management |
| `bootstrap_remote.sh` | Remote setup |

### Results
| File | Experiment |
|------|-----------|
| `results/analysis1_similarity.png` | v1 cross-lingual similarity |
| `results/v1_full_output.txt` | v1 full stdout |
| `results/v2_steering_output.txt` | v2 steering generation examples (1B) |
| `results/v2_medical_*.json` | v2 medical experiments (**retracted**) |
| `results/v2_generalization.json` | v2 generalization (**retracted**) |
| `results/v2_flip_distant_combined.json` | v2 flip/combined (**retracted**) |
| `results/v3_replication.json` | v3 controlled replication (null) |
| `results/v3_lowresource_rescue.json` | v3 weak-language rescue (null) |
| `results/v3_domain_rescue.json` | v3 domain feature rescue (null) |
| `results/v3_feature_validation_output.txt` | v3 validation (features are readouts) |
| `results/v3_layer_sweep.json` | v3 layer sweep — ablation across 9/17/22/29 |

---

## Honest Project Summary

We started with a routing hypothesis, got excited by apparent cross-lingual rescue results that turned out to be artifacts of a data-loading bug (MMMLU `default` is not English) compounded by subset evaluation and no random-feature control. We ran controlled replications and landed on a null. Along the way we validated that:

1. SAE features can steer output language at small scale (Feature 857 on 1B, real)
2. Language-agnostic domain-selective SAE features exist in Gemma 3 4B at layer 29 (features 893, 12570, 12845 — real)
3. These features generalize beyond the original MCQ setup to free-form medical text, across multiple languages (tested, confirmed)
4. Simple single-layer clamping of these features does not improve weak-language medical MCQ performance above the random-feature noise floor (tested, confirmed)
5. Single-feature ablation of these features at layer 29 does not measurably reduce medical MCQ accuracy (tested, confirmed)

The routing hypothesis is not refuted in its strongest form — we only ruled out the simplest interventions (single-feature amplification, single-feature ablation, single layer). Multi-layer, multi-feature, or coordinated interventions remain untested. But the project did not produce a publishable positive signal, and the honest state of things is that the easy version of the idea does not work at this scale with these tools.
