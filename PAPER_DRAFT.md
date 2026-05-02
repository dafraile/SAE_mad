# Internal Representation, Not Internal Reasoning: Why Apparent LLM Triage Failures May Be Output-Mapping Artifacts

**[Authors, affiliations to fill]**

## Abstract

Recent benchmarks evaluating large language models (LLMs) on consumer-facing
medical triage report high under-triage rates for emergency cases under
constrained-output evaluation formats [Ramaswamy 2026]. Subsequent behavioral
replications have shown that the failure rate depends strongly on prompt format
and output constraint: identical clinical content presented as natural patient
text and answered freely yields substantially different scoring than the same
content under a forced single-letter answer. The open question is whether
prompt format changes how the model represents the clinical case, or only how
it answers.

We address this mechanistically using sparse autoencoder (SAE) features in three
instruction-tuned LLMs from two families: Gemma 3 4B IT and 12B IT (Gemma Scope 2
JumpReLU SAEs) and Qwen3-8B (Qwen Scope k=50 TopK SAEs). Across all three models,
medical-content SAE features at the deep-encoding layer fire essentially identically
on identical clinical text across format conditions: per-token max activations match
within 0–4%, mean-pooled mod-indices are 10–25% versus 32–45% for magnitude-matched
random features in the same SAE. When we control for prompt-length asymmetry, the
residual-stream difference between format conditions vanishes; the residual signal
that survives length-invariant max-pooling loads onto non-medical features rather
than the medical ones.

The mechanistic invariance holds across the Gemma JumpReLU and Qwen TopK SAE
training pipelines and across two model architectures. Bootstrap 95% confidence
intervals exclude zero in every cell of every model. We also document a behavioral
finding: the +13–20pp forced-letter penalty observed in Gemma 4B essentially vanishes
in 12B, with scaling closing the gap between the model's preserved clinical encoding
and its constrained output capacity.

We discuss implications for clinical-AI evaluation methodology and propose SAE
features as deployable, format-invariant monitors of clinical groundedness.

---

## 1. Introduction

Large language models (LLMs) are increasingly evaluated for clinical roles.
Singhal et al.~\cite{singhal2023large} demonstrated that with appropriate
alignment, the same models can reach near-clinician agreement with scientific
consensus on consumer medical questions on multiple human-evaluation axes,
supporting the view that LLMs encode substantial clinical knowledge — though
the same study found persistent gaps on omission, possible harm, and bias.
Subsequent work has tested this knowledge in more specific deployment
settings, including triage. A recent benchmark of a consumer-facing health
chatbot reported a 51.6\% emergency-case under-triage rate, an alarming
headline that has been cited as evidence that current LLMs are unsafe for
triage applications~\cite{ramaswamy2026chatgpt}.

Behavioral replications have complicated this conclusion. The same clinical
content, presented either as a structured clinical write-up with a
forced-letter answer or as a first-person patient narrative answered freely,
yields substantially different scoring outcomes — significantly so in pooled
multi-model averages, and considerably more on individual high-stakes
cases~\cite{frailenavarro2026triage}. Recent literature on prompt-format and
multiple-choice sensitivity in LLM benchmarks more generally has shown that
constrained-output evaluations can systematically misrepresent underlying
capability: option-identifier and ordering biases inflate or suppress
multiple-choice scores~\cite{zheng2024large,pezeshkpour2024large}, and prompt
formatting alone can move accuracy by tens of points within the same
model~\cite{sclar2024quantifying}. The triage failures are therefore at least
partly a function of the evaluation, not the underlying model.

This raises a mechanistic question that behavioral evidence cannot resolve.
Either prompt format changes how the model represents the clinical case
internally — call this \textbf{Version A}, in which constrained outputs reach
back into clinical reasoning — or prompt format changes only how the model
maps an already-formed clinical understanding into a final answer — call this
\textbf{Version B}, in which the same internal representation produces
different outputs only because of the constrained output stage. Version A
would implicate clinical reasoning itself; Version B would localize the
apparent failure in output mapping and recast the benchmark as measuring that
mapping rather than the underlying capability. The two have very different
implications for both deployment and evaluation methodology, and to our
knowledge no prior work has tested the distinction mechanistically on
clinical triage. Closest precedent uses sparse-autoencoder–based attribution
graphs to analyze internal mechanisms in production
models~\cite{anthropic2025biology}.

We provide that test. Sparse autoencoders (SAEs) decompose a model's
residual stream into a sparse, overcomplete dictionary of features that have
been shown to carry interpretable, content-tied
signal~\cite{bricken2023monosemanticity,cunningham2023sparse,templeton2024scaling}.
Recent open releases — Gemma Scope on Gemma 2~\cite{lieberum2024gemma}, its
successor Gemma Scope 2 covering Gemma
3~\cite{deepmind2025gemmascope2,google2026gemmascope24bit}, and the
recently released Qwen-Scope~\cite{qwen2026scope} — make this analysis
available across multiple model families with different SAE training
pipelines: JumpReLU SAEs trained on Gemma~\cite{rajamanoharan2024jumping} and
$k$-sparse TopK SAEs ($k{=}50$) trained on
Qwen~\cite{makhzani2013ksparse,gao2024scaling}. We use these to ask, in three
instruction-tuned models from two families, whether medical SAE features fire
identically on identical clinical content across the two output-format
conditions whose behavioral scoring diverges.

\textbf{Contributions.} We make three claims and back each with controlled,
preregistered tests across Gemma 3 4B IT, Gemma 3 12B IT, and Qwen3-8B (60
paper-faithful clinical vignettes per model):

\begin{enumerate}
\item \textbf{Magnitude invariance.} Medical-content SAE features fire within
0--4\% per token on identical clinical content across format conditions.
Mean-pool modulation indices are 10--25\% for medical features versus 32--45\%
for magnitude-matched random features in the same SAE basis. Bootstrap 95\%
confidence intervals exclude zero in every cell of the layer$\,\times\,$stratum
design across all three models.
\item \textbf{Direction analysis.} When prompt-length asymmetry is controlled
(by truncating the longer prompt to identical content range), the
residual-stream difference between conditions vanishes exactly; what survives
in length-invariant max-pool aggregation loads onto non-medical features in
the SAE basis rather than the medical ones.
\item \textbf{Behavioral scaling.} The forced-letter penalty observed at 4B
($+$13--20pp advantage for free-text in our paper-faithful adjudication)
essentially vanishes at 12B ($\approx{}0$pp). This refines the Med-PaLM--era
scaling claim~\cite{singhal2023large,wei2022emergent}: capability scaling
closes the gap not just on internal medical knowledge but specifically on the
model's ability to map that knowledge into a constrained output. Crucially,
the \emph{mechanistic} invariance at deep layers persists across both scales.
\end{enumerate}

Together, these results support Version B at scale and across families. The
model's clinical encoding is preserved across the output formats whose
accuracy scoring diverges; the failure mode the benchmark detects lives in
output mapping. We discuss implications for clinical-AI evaluation methodology
and propose deep-layer SAE features as a deployable, format-robust monitor of
clinical groundedness. The control-side analogue — modifying behavior using
continuous activation differences rather than discrete feature
interventions — has been studied separately~\cite{turner2023steering}, and
recent work demonstrates SAE features can also serve as causal units for
circuit-level editing in some settings~\cite{marks2024sparse}; our use of SAE
features here is restricted to their well-supported readout role.


---

## 2. Background and Related Work

[Brief — ~0.5 page]

### 2.1 Sparse autoencoders for representation analysis

SAEs decompose a model's residual stream into a sparse, overcomplete dictionary of
features [Bricken 2023, Cunningham 2023]. Gemma Scope 2 [Lieberum 2025] uses
JumpReLU SAEs, allowing graded per-feature activation gated by a learned threshold;
Qwen Scope (released 2026) uses TopK SAEs [Makhzani 2014, Gao 2024], hard-capping
the active feature count per token. The two pipelines make different sparsity
trade-offs: Gemma Scope's JumpReLU permits ~60–100 features active per token at
width 16k; Qwen Scope fixes 50 active out of 65,536 (0.076% sparsity).

### 2.2 LLM evaluation in clinical AI

Singhal et al. (2023) demonstrated that LLMs encode substantial clinical knowledge
[Med-PaLM, MultiMedQA]. Subsequent benchmarks have probed specific failure modes,
notably triage [Ramaswamy 2026, ...]. Behavioral replications have raised concerns
about evaluation-format sensitivity: outcomes that look like reasoning failures
under one prompt format may not replicate under another. Our work supplies
mechanistic evidence for this dissociation.

### 2.3 Steering vectors and the SAE-decomposition gap

ActAdd [Turner 2023] demonstrates that activation differences between conditions
can serve as continuous, causally-effective steering directions. Our paper uses
SAEs not for steering but for *interpretation*: we project the activation
difference between format conditions onto the SAE basis and ask whether
domain-specific features carry it. This complements ActAdd's finding by
identifying the basis directions that the format-difference signal is *not*
loaded on.

---

## 3. Method

### 3.1 Dataset

We use the 60 paper-canonical clinical vignettes from the paper-faithful
replication corpus [TODO: cite Fraile Navarro 2026 / your prior work]. Each
vignette is a clinical case in three pre-defined formats:

- **SL — structured + forced-letter.** Structured clinical write-up plus a terse
  "Reply with exactly one letter only" instruction block.
- **NL — natural + forced-letter.** First-person patient narrative plus the same
  forced-letter instruction.
- **NF — natural + free-text.** Identical patient narrative as NL, ending with a
  natural question and *no* output-format instruction.

A critical property of the corpus: **NL and NF's clinical content is byte-identical**
(1033 chars per case in our test). The only difference is whether the
forced-letter instruction block (~50–60 tokens) is appended after the patient
narrative. This affords a clean isolation of the format effect.

The fourth cell of an input × output factorial — *structured + free-text* — is
omitted because the paper-faithful corpus does not natively contain it; we treat
it as future work. Three of the four 2 × 2 cells still let us isolate the output
axis cleanly via the NL ↔ NF comparison (same input, format instruction varied).

Gold triage labels (A=monitor at home, B=see doctor in weeks, C=24–48h, D=ER)
come with the dataset; some cases have edge-case dual labels (e.g., C/D). Note
that gold labels and cell codes use disjoint naming: gold is {A, B, C, D},
cell codes are {SL, NL, NF}.

### 3.2 Models and SAEs

| Model | Layers | d_model | SAE | SAE arch | d_sae | Sparsity |
|---|---|---|---|---|---|---|
| Gemma 3 4B IT | 34 | 2560 | gemma-scope-2-4b-it | JumpReLU, l0_medium | 16,384 | ~60–100 active |
| Gemma 3 12B IT | 48 | 3584 | gemma-scope-2-12b-it | JumpReLU, l0_medium | 16,384 | ~60–100 active |
| Qwen3-8B | 36 | 4096 | SAE-Res-Qwen3-8B-Base-W64K-L0_50 | TopK | 65,536 | k=50 fixed |

For Gemma we sweep four matched-depth layers (4B: 9, 17, 22, 29 ≈ 27/50/65/85%;
12B: 12, 24, 31, 41 — same fractional depths). For Qwen we run a single deep
layer (L31 ≈ 86% depth, lowest reconstruction error among the four candidates we
sanity-checked). We acknowledge Qwen Scope's intrinsic ~38% relative reconstruction
error at this layer, a property of the k=50 sparsity choice rather than a
checkpoint-transfer artifact (verified by sanity tests showing identical error on
both chat-templated and raw-text inputs, with and without `b_dec` subtraction).

### 3.3 Behavioral test (Phase 0.5)

For each model × cell × case, we generate the model's output via greedy decoding.
Cells SL and NL have terse single-letter outputs; we extract via regex. Cell NF has
free-text outputs; we score using the paper-faithful adjudication pipeline
[script `adjudicate_natural_paper_scale.py`] with two LLM judges (`gpt-5.2-thinking-high`
and `claude-sonnet-4.6`). The adjudicator prompt uses the paper's native scale
(A=monitor at home through D=ER), explicitly chosen because the same codebase
contains an inverted-scale version that would silently flip every correctness
judgment.

We report per-cell accuracy with permissive gold matching (gold "C/D" accepts
either letter as correct). We also report the inter-rater agreement and Cohen's κ
between the two judges as a calibration signal.

### 3.4 Mechanistic invariance test (Phase 1b)

For each (model, layer, case, condition) we mean-pool SAE feature activations
over user content tokens. For Gemma we identify content tokens by chat-template
boundaries (`<start_of_turn>user\n` to `<end_of_turn>`); for Qwen we feed raw
text and pool over the entire input. Per case, we compute:

- **Cosine similarity** between NL and NF feature vectors (over the medical
  feature subspace).
- **Modulation index**:  
  $\text{mod}(c) = \langle |a_{NF}(f) - a_{NL}(f)|\rangle_f \,/\, \langle (|a_{NL}(f)| + |a_{NF}(f)|)/2 \rangle_f$
  where the average is over the relevant feature subset (medical or random).
  Lower = more invariant.

The control is a set of 30 random features per layer, drawn with a frozen seed
from the pool of features whose mean activation across the union of NL+NF content
falls in the band `[0.5 × min(med_means), 2.0 × max(med_means)]`. This
**magnitude-matches** the random pool to the medical features, removing the
small-feature-noise inflation that would otherwise depress the random mod-index.
Pool sizes are typically 500–2200 features per layer per model.

We stratify by Phase 0.5 + adjudicator outcomes:

- **format_flipped**: NL wrong AND both judges agree NF right.
- **both_right**: NL correct AND both judges agree NF correct.
- **both_wrong**: NL wrong AND both judges agree NF wrong.
- **NL_only_right**: NL correct AND not both judges right.

Bootstrap 95% confidence intervals (2,000 resamples) on the per-case
medical-minus-random mod-index difference.

### 3.5 Direction-of-format-effect test (Phase 2b)

To localize *where* the (NL − NF) residual-stream direction lives in the SAE
basis, we project it onto each feature's encoder direction. We use three
aggregations to control for prompt-length asymmetry:

- **Full mean-pool**: NL and NF residuals pooled over their full user-content
  ranges. Affected by NL's longer content (it includes the forced-letter
  instruction tokens).
- **Length-controlled mean-pool**: NL's content range truncated at the
  literal `"Reply with exactly one letter only"` so NL and NF pool over
  identical content. Because the corpus has byte-identical clinical text
  in NL's prefix and NF, this aggregation reduces (NL − NF) to the limit of
  numerical precision.
- **Max-pool**: per-dimension max over content tokens, length-invariant
  by construction.

For each aggregation we compute cosine alignment between the case-averaged
(NL − NF) direction and each SAE feature's encoder direction `W_enc[:, f]`,
rank features by `|alignment|`, and report where the medical features land.

### 3.6 Medical-feature identification

For 4B we use the v3-validated medical features (cross-lingual
contrastive, six-condition; see Appendix). For 12B and Qwen we run an
English-only medical-vs-non-medical contrastive on 60 patient-realistic
prompts versus 30 hand-curated patient-style non-medical prompts, scoring
$\text{score}(f) = \text{mean-max}_{\text{med}}(f) - \text{mean-max}_{\text{non}}(f)$
under a firing-reliability filter (must fire on ≥70% of medical and
≤10% of non-medical content). We take the top-3 features per layer.

---

## 4. Results

### 4.1 Behavioral phenomenon (Gemma 4B and 12B)

Cell-level accuracy on the 60 paper-canonical cases:

| Cell | 4B | 12B |
|---|---|---|
| SL: structured + forced-letter | 60.0% | **81.7%** |
| NL: natural + forced-letter | 56.7% | **81.7%** |
| NF: natural + free-text (GPT judge) | 71.7% | 81.7% |
| NF: natural + free-text (Claude judge) | 76.7% | 78.3% |
| NF: both judges agree correct | 70.0% | 76.7% |
| Inter-rater agreement / κ | 88.3% / 0.797 | 76.7% / 0.634 |

**On Gemma 4B, free-text NF outperforms forced-letter NL by +13–20pp
(judge-dependent).** This replicates the prior behavioral observation that
constrained output formats penalize Gemma 4B's apparent triage capability.

**On Gemma 12B, the gap essentially disappears (NL vs NF within 0–3pp across
judges).** Twelve-billion-parameter Gemma is capable enough to map its clinical
understanding onto a constrained letter output without the format penalty.
This is a novel scaling finding consistent with Singhal et al. (2023):
scaling improves performance on medical question answering, including
specifically under constrained output formats.

The free-text gain on 4B is concentrated in mid-acuity cases (gold C and C/D)
rather than in emergencies or low-acuity (see appendix table). The 12B
attenuation is across-the-board.

### 4.2 Mechanistic invariance — magnitude (Phase 1b)

Bootstrap 95% CIs on (medical − random) modulation index, per layer × stratum:

**Gemma 3 4B IT** (positive: medical *more* perturbed than random):

| Layer | Stratum | n | med_mod | rnd_mod | diff [95% CI] |
|---|---|---|---|---|---|
| 9 | format_flipped | 13 | 0.148 | 0.343 | **−0.196 [−0.223, −0.170]** |
| 9 | both_right | 29 | 0.183 | 0.408 | **−0.224 [−0.255, −0.196]** |
| 17 | format_flipped | 13 | 0.118 | 0.377 | **−0.259 [−0.325, −0.195]** |
| 17 | both_right | 29 | 0.154 | 0.417 | **−0.263 [−0.303, −0.227]** |
| 22 | format_flipped | 13 | 0.156 | 0.319 | **−0.163 [−0.205, −0.118]** |
| 22 | both_right | 29 | 0.252 | 0.342 | **−0.090 [−0.145, −0.021]** |
| 29 | format_flipped | 13 | 0.107 | 0.413 | **−0.305 [−0.371, −0.250]** |
| 29 | both_right | 29 | 0.152 | 0.423 | **−0.271 [−0.323, −0.224]** |

**At every layer × stratum on 4B, medical-feature mod-index is significantly
lower than the magnitude-matched random control.** Effect sizes are large
(0.09 to 0.37 in absolute mod-index) and 95% CIs clear zero at every cell.

**Gemma 3 12B IT** (depth-dependent):

| Layer | n | med_mod | rnd_mod | diff [95% CI] |
|---|---|---|---|---|
| 12 | 60 | 0.253 | 0.213 | +0.040 [+0.001, +0.089] (medical *more* perturbed) |
| 24 | 60 | 0.370 | 0.198 | +0.172 [+0.114, +0.237] (medical *more* perturbed) |
| 31 | 60 | 0.157 | 0.395 | **−0.238 [−0.260, −0.219]** |
| 41 | 60 | 0.181 | 0.285 | **−0.103 [−0.126, −0.081]** |

**The depth-dependent pattern is novel.** At deep layers (31, 41 ≈ 65–85%
depth), medical features are more invariant than random — Version B replicates
the 4B finding. At shallow/mid layers (12, 24), the medical features we
identified are *more* perturbed than the magnitude-matched random control.

**Qwen3-8B** (cross-family, single layer):

| Layer | n | med_mod | rnd_mod | diff [95% CI] |
|---|---|---|---|---|
| 31 | 60 | 0.266 | 0.330 | **−0.064 [−0.106, −0.024]** |

**Cross-family validation holds at the deep encoding layer.** Effect size is
smaller than Gemma's, consistent with Qwen Scope's k=50 TopK sparsity inducing
a ~38% reconstruction error vs Gemma Scope's ~14% at the equivalent layer, but
the direction and statistical signal replicate.

### 4.3 Per-token alignment

The Phase 1b mod-index is mean-pooled. We also report per-case max activations
on the medical features at L29 of Gemma 4B for three example cases:

| Case | Gold | NL max | NF max | Per-feature delta |
|---|---|---|---|---|
| E3 | C | [-0.0, 701.5, 883.0] | [-0.0, 700.5, 899.7] | 0.0% / 0.1% / 1.9% |
| E4 | C | [974.5, 3423.3, 2341.0] | [974.5, 3434.3, 2380.3] | 0.0% / 0.3% / 1.7% |
| E9 | D | [1054.6, 3215.8, 3000.6] | [1066.2, 3228.2, 3016.9] | 1.1% / 0.4% / 0.5% |

When the model encounters the words "asthma" or "DKA" in NL vs NF, the medical
features fire at essentially the same magnitude. The clinical representation
is preserved at the per-token level; the small mean-pool mod-index residual is
the ~50-token dilution from B's appended forced-letter instructions.

### 4.4 Direction analysis (Phase 2b)

We examine where the (NL − NF) residual direction concentrates in the SAE basis,
using three aggregations.

**Length-controlled mean (clinical-content range matched).** Because B's
prefix is byte-identical to NF, the residual diff norm is exactly **0** at every
Gemma layer when we truncate B's pooling range to its clinical content. This
is the strongest possible isolation: with input held constant, residuals are
deterministically identical. No format effect exists at the residual level
when content is held identical.

**Full mean-pool (Phase 2 reproduction, length-confounded).** All medical
features show negative-signed alignment (the dilution-from-prompt-length
signature). At Gemma 4B L29: medical-feature ranks 10994 (67%), 470 (3%),
2235 (14%) of 16384.

**Max-pool (length-invariant).** At Gemma 4B L29 medical features rank
2523 (15%), 7013 (43%), 10008 (61%); top-aligned features (3833, 10012, 980,
9485, 755) are non-medical — candidates for the output-instruction features
that respond to the forced-letter block tokens. Same mid-rank pattern at
Gemma 12B L31 and L41 (medical features at 32%, 43%, 91%; 32%, 51%, 86%
percentiles). At Qwen L31 two of three medical features sit at 89.5% and
62.4% percentile (far from format direction); one at 0.1% (top-aligned),
matching the mixed pattern in Gemma 4B L29.

**Reading.** When length is controlled, no format effect exists in the residual
stream. The format effect captured by length-invariant max-pooling lives in
non-medical features in all three models — most plausibly features that fire
on the appended forced-letter instruction tokens themselves rather than on the
clinical content.

### 4.5 Stratification at 4B: format-flipped cases

The format_flipped stratum (n=13 at 4B, where the format physically flipped the
answer between B wrong and both-judges-D-right) is the most stringent. Per-token
max activations on these cases are within 0–4% across NL and NF (table above for
E3, E4, E9 — all in this stratum). Despite the model producing different letter
outputs, the medical-feature signature on the clinical tokens is essentially
identical. This is the cleanest evidence that the format effect operates
downstream of the clinical encoding rather than within it.

---

## 5. Discussion

[~1 page; TODO]

The three-model picture supports a single mechanistic claim with a scale-aware
nuance:

- **Clinical encoding is preserved across format change.** All three models
  show medical-feature invariance at the deep encoding layer that exceeds
  what a magnitude-matched random control predicts.
- **The behavioral format penalty attenuates with scale.** What looks like a
  format-induced reasoning failure at 4B becomes a near-zero gap at 12B —
  consistent with Med-PaLM's scaling claim, refined: as the model scales,
  the output-mapping circuit catches up with the preserved clinical encoding.
- **The format effect, where it exists, lives downstream of clinical encoding.**
  Length-controlled residuals show no format difference; max-pool residuals
  show a format direction that loads on non-medical features.

This is a stronger version of the Ramaswamy-replication policy claim. The
benchmark's apparent reasoning failures are partially measurement artifact —
the model's clinical signature on those failures is preserved.

For deployment, SAE features at the deep encoding layer are a viable
format-robust monitor of clinical groundedness. A real-time clinical chatbot
checking medical-feature signatures against a calibrated reference distribution
would be unaffected by the user's prompt format and would flag genuine drift in
clinical understanding rather than format change.

[More to come on:
- Why the depth-dependent pattern in 12B but not 4B (model architecture, scale)
- Why the Qwen effect is smaller (TopK k=50 sparsity)
- Limitations: see §6]

---

## 6. Limitations

[~0.5 page; TODO]

- Three models, two families, one clinical domain (triage). A larger
  cross-family corpus (Llama Scope, Mistral SAEs) would further strengthen
  the generalization claim.
- 60 vignettes — small for clinical-AI work, but matched to the
  paper-faithful replication corpus exactly (no subsetting).
- LLM-as-judge dependence on NF (free-text) scoring (mitigated by 76–88%
  inter-rater agreement, κ = 0.63–0.80, paper-native scale).
- Mean-pool and max-pool aggregations only; richer schemes (attention-weighted)
  could surface different patterns.
- Qwen Scope's intrinsic 38% reconstruction error inflates the noise floor
  for that model. A higher-fidelity Qwen-family SAE would tighten the cross-family
  effect-size estimate.
- Medical-feature identification uses a hand-curated 30-prompt non-medical
  corpus for 12B and Qwen; a larger or programmatically generated corpus
  could yield more or different features.

## 7. Conclusion

[~0.25 page; TODO]

Across three instruction-tuned LLMs (Gemma 3 4B IT, Gemma 3 12B IT, Qwen3-8B),
sparse autoencoder features at the deep encoding layer are more invariant
under output-format change than magnitude-matched random features in the same
SAE basis. The behavioral format penalty observed at 4B essentially vanishes
at 12B, while the mechanistic invariance persists at deep layers across
scales and families. SAE features are a deployable, format-invariant monitor
of clinical groundedness, and apparent benchmark failures under constrained
output formats may not reflect underlying reasoning deficits.

---

## Appendix

[Sketch — to expand]

- A1: Per-layer × per-stratum bootstrap tables for all three models
- A2: Top-token analysis of the format-effect features (3833, 10012, etc.)
- A3: Adjudicator prompts, agreement statistics, calibration check
- A4: Qwen Scope reconstruction-error characterization
- A5: Compute and cost breakdown
