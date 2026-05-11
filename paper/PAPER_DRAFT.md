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
mapping rather than the underlying capability.

The distinction has well-established conceptual precedent. Probing-based
methods recover knowledge beyond what model outputs reveal~\cite{burns2023discovering};
model self-knowledge depends strongly on elicitation format~\cite{kadavath2022language};
and verbalized reasoning can be unfaithful to the actual determinants of
model predictions~\cite{turpin2023language}. In the clinical-triage setting
specifically, concurrent work demonstrates a 53-percentage-point
knowledge-action gap on Qwen 2.5 7B Instruct: linear probes discriminate
hazardous from benign triage cases at 98.2\% AUROC while the same model's
output sensitivity is only 45.1\%, and four mechanistic interventions —
including SAE feature steering — fail to reliably correct the resulting
errors~\cite{basu2026interpretability}. That work establishes that the
gap is real and that current interpretability methods do not close it
via direct intervention; complementary methodological work uses
attribution-graph analysis to dissect internal mechanisms in production
models more generally~\cite{anthropic2025biology}. We ask a different,
more specific question: \emph{where} does the gap live mechanistically?
By varying output format while holding clinical content byte-identical, we
test whether the apparent failure arises within the clinical encoding itself
or downstream of it. To our knowledge no prior work has tested
format-induced representation invariance on clinically identical triage
content using SAE features across multiple model families.

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
scaling claim~\cite{singhal2023large,singhal2025expert,wei2022emergent}:
capability scaling closes the gap not just on internal medical knowledge but
specifically on the model's ability to map that knowledge into a constrained
output. Crucially, the \emph{mechanistic} invariance at deep layers persists
across both scales.
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

\textbf{Sparse autoencoders for representation analysis.} SAEs decompose a
model's residual stream into a sparse, overcomplete dictionary of features
that have been shown to be interpretable and content-tied~\cite{bricken2023monosemanticity,
cunningham2023sparse,templeton2024scaling}. Two open release lines cover the
families we use: \textit{Gemma Scope}~\cite{lieberum2024gemma} and
its successor \textit{Gemma Scope 2}~\cite{deepmind2025gemmascope2,
google2026gemmascope24bit} train JumpReLU SAEs~\cite{rajamanoharan2024jumping}
on Gemma 2 and Gemma 3 respectively, using a per-feature learned threshold
that allows graded activation; \textit{Qwen-Scope}~\cite{qwen2026scope}
trains $k$-sparse TopK SAEs~\cite{makhzani2013ksparse,gao2024scaling} on
Qwen3 base checkpoints with $k{=}50$ active features per token out of
65{,}536 (0.076\% sparsity, more aggressive than Gemma Scope's
$\sim{}60$--$100$ active at width 16k). The two pipelines therefore make
different reconstruction--sparsity trade-offs; we treat this as a feature
of our cross-family validation rather than a confound. Concurrent
work~\cite{frasertaliente2026nla} introduces Natural Language Autoencoders
(NLAs), an alternative unsupervised method that produces
natural-language descriptions of LLM activations on Claude models.
NLAs complement rather than replace SAE-based decomposition: they
explain activations expressively but require two trained LLM modules
per target model, making them heavyweight relative to the
inference-time monitoring application we develop here. NLAs are not
currently available for the open-weight model families our work uses,
and we treat NLA-based extension of our analyses as future work.

\textbf{Internal--external dissociation in LLMs.} The conceptual precedent
for the distinction we test is well established. Probing methods recover
knowledge beyond what model outputs reveal~\cite{burns2023discovering};
self-knowledge depends on elicitation
format~\cite{kadavath2022language}; and verbalized reasoning can be
unfaithful to the actual determinants of model
predictions~\cite{turpin2023language}. The Anthropic interpretability
program has used attribution graphs over SAE features to dissect
internal mechanisms in production models~\cite{anthropic2025biology}.

\textbf{LLM evaluation in clinical AI.} Med-PaLM established that LLMs
encode substantial clinical knowledge~\cite{singhal2023large}, with
follow-on work pushing further on alignment and
factuality~\cite{singhal2025expert}. Specific benchmarks have probed
failure modes, notably consumer-facing
triage~\cite{ramaswamy2026chatgpt}, where a recent headline reported a
51.6\% emergency-case under-triage rate. Behavioral replications by our
group~\cite{frailenavarro2026triage} and concurrent mechanistic work on
the same task~\cite{basu2026interpretability} both indicate that
constrained-output evaluation is a major contributor to the apparent
failure rates. The broader literature on prompt format and
multiple-choice sensitivity in LLM benchmarks supports the same
direction~\cite{zheng2024large,pezeshkpour2024large,sclar2024quantifying}.

\textbf{Steering vectors and SAE features as causal levers.} ActAdd-style
steering~\cite{turner2023steering} uses continuous activation differences
between conditions as steering directions; sparse feature
circuits~\cite{marks2024sparse} demonstrate that small sets of SAE
features can serve as interpretable causal units for circuit-level
editing in some settings. Our use of SAE features here is restricted to
their well-supported readout role: we project activation differences onto
the SAE basis and ask which features carry them, rather than using the
SAE to intervene on model behavior. The complementary intervention path
is studied separately, with prior null
results~\cite{frailenavarro2026saemad,basu2026interpretability} indicating
that single-feature SAE interventions have not, to date, produced
reliable behavioral change in clinical question-answering.

---

## 3. Method

### 3.1 Dataset

We use the 60 paper-canonical clinical vignettes from the paper-faithful
replication corpus~\cite{frailenavarro2026triage}. Each
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

### 3.3 Behavioral test

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

### 3.4 Mechanistic invariance test

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

We stratify by behavioral-test correctness × adjudicator-agreed correctness:

- **format_flipped**: NL wrong AND both judges agree NF right.
- **both_right**: NL correct AND both judges agree NF correct.
- **both_wrong**: NL wrong AND both judges agree NF wrong.
- **NL_only_right**: NL correct AND not both judges right.

Bootstrap 95% confidence intervals (2,000 resamples) on the per-case
medical-minus-random mod-index difference.

### 3.5 Direction-of-format-effect test

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

### 4.2 Mechanistic invariance — magnitude

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

The mod-index above is mean-pooled. We also report per-case max activations
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

### 4.4 Direction analysis

We examine where the (NL − NF) residual direction concentrates in the SAE basis,
using three aggregations.

**Length-controlled mean (clinical-content range matched).** Because B's
prefix is byte-identical to NF, the residual diff norm is exactly **0** at every
Gemma layer when we truncate B's pooling range to its clinical content. This
is the strongest possible isolation: with input held constant, residuals are
deterministically identical. No format effect exists at the residual level
when content is held identical.

**Full mean-pool (length-confounded).** All medical
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
non-medical features in all three models. Section~4.5 makes this concrete by
naming the features and showing what they fire on.

### 4.5 What the format-direction features actually fire on

We identify the top features by absolute alignment with the (NL${-}$NF)
max-pool direction at Gemma 3 4B IT, L29 — features 3833, 10012, and 980 —
and inspect their top-activating (token, context) pairs across all 60
cases ${\times}$ \{NL, NF\} prompts (Figure~4, left panel).
\textbf{All three features fire exclusively in the NL condition, on the
literal forced-letter answer-key scaffold tokens themselves}: feature 3833
on the word ``next'' inside ``\emph{B = See my doctor in the [next] few
weeks}'', feature 10012 on ``the'' inside ``\emph{D = Go to [the] ER now
Do not include any explanation}'', feature 980 on ``='' across the
answer-key syntax. The top-3 (NL, NF) firing counts per feature are
$(3, 0)$ for each. By contrast, the v3-validated medical features 12570,
893, and 12845 fire on clinical-content tokens (e.g., ``blood'' in
lab-value contexts, ``neck''/``on'' in clinical-exam contexts, anatomical
context for facial weakness) at identical magnitudes across NL and NF, with
top-3 (NL, NF) firing counts that mix between conditions on the same
clinical-content tokens (Figure~4, right panel).

This converts the percentile-rank statement of Section~4.4 into a
feature-level mechanistic interpretation: the format-direction is encoded
by SAE features that detect the structural format of the prompt rather
than its clinical content.

### 4.6 Restricted random pool

The magnitude-matched random control pool above was permitted to
include features that may not fire on clinical content. We tighten the
control by additionally restricting the pool to features that fire on at
least 25\% of the 120 prompts in the union NL${\,\cup\,}$NF. With this
stricter control at Gemma 3 4B IT, L29, the medical${-}$random
modulation-index gap shrinks but remains significant in every stratum:
format-flipped diff $-0.196$ [95\% CI: $-0.261, -0.135$]; both-right
$-0.189$ [$-0.243, -0.139$]; both-wrong $-0.297$ [$-0.380, -0.225$]. All
CIs continue to exclude zero. The roughly $30$--$40\%$ shrinkage from the
unrestricted bootstrap is the size we would expect from a more
content-relevant random pool, and the qualitative direction is unchanged.

### 4.7 Stratification at 4B: format-flipped cases

The format-flipped stratum (n=13 at 4B, where the format physically flipped the
answer between NL wrong and both-judges-NF-right) is the most stringent. Per-token
max activations on these cases are within 0–4% across NL and NF (table above for
E3, E4, E9 — all in this stratum). Despite the model producing different letter
outputs, the medical-feature signature on the clinical tokens is essentially
identical. This is the cleanest evidence that the format effect operates
downstream of the clinical encoding rather than within it.

### 4.8 Causal interventions on the format direction

Sections 4.4--4.5 localize the format direction in residual space and name
the SAE features that carry it. Whether intervening on that direction
*causally* changes letter outputs is a separate, stronger claim. We test
two intervention modalities at Gemma 3 4B IT, L29, on the same 60 NL
prompts.

\textbf{Discrete SAE-feature ablation.} For each NL forward pass we
register a hook at L29 that subtracts the SAE-reconstructed contribution
of the three top format-direction features identified in
Section~4.5 (3833, 10012, 980). A control arm ablates three
magnitude-matched random features in the same SAE basis. A separate
diagnostic confirms the hook fires on every forward and modifies the
residual at the expected magnitude (mean 264 norm subtracted per token,
peak 6{,}795 norm on the strongest answer-key tokens). Result:
\textbf{0/60 letter predictions change} in either ablation arm
relative to vanilla NL (33/60 = 55.0\% in all three arms).
The intervention is genuine but small relative to the L29 residual
norm (${\sim}60{,}000$ per token); the ablation magnitude is ${\sim}0.4\%$
of the residual on average and peaks at ${\sim}11\%$ on the strongest
answer-key token positions, insufficient to flip the next-token argmax.

\textbf{Continuous ActAdd-style steering.} A stronger intervention test
that subtracts the full residual-space format direction rather than only
its projection onto three discrete features. We compute
$v = \langle r_{NL}\rangle - \langle r_{NF}\rangle$ at L29 (case-averaged
mean residuals at content tokens; $\|v\| = 1{,}012$), then for each NL
forward pass at $\alpha \in \{0, 0.5, 1.0, 2.0, 4.0\}$ register a hook
that adds $-\alpha \cdot v$ to the L29 output at every token. Result:
\textbf{accuracy is 33/60 = 55.0\% across all five $\alpha$}; only 2/60
letter predictions change at $\alpha \in \{2, 4\}$ (cases E6 and F2,
both gold = B/C, both shifting from C to B and remaining within the
permissive gold range). The shift direction is opposite to what
NF-like behavior would predict, consistent with noise rather than
meaningful causal signal at this perturbation magnitude
(${\sim}6.7\%$ of the L29 residual norm at $\alpha = 4$).

\textbf{Reading.} Both intervention modalities produce near-null
behavioral effects: the discrete-feature ablation cleanly null, the
continuous direction subtraction with at most 2/60 prediction shifts
that do not change accuracy. Combined with concurrent intervention
results on a different model in the same task family
\cite{basu2026interpretability} and our own prior null on a
different clinical
question~\cite{frailenavarro2026saemad}, three intervention modalities
across three studies converge on a single conclusion: the format effect
is detectable and interpretable in representation space but not, at
the magnitude of perturbation we can apply without breaking generation,
isolatable to a single layer/direction with sufficient causal control to
drive the behavior. The deployment claim of this paper --- SAE features
as a format-invariant monitor of clinical groundedness --- plays to
their well-supported role as readouts; the complementary application
of using these same features as causal steering levers is, on present
evidence, not viable for this task at single-layer scale.

---

## 5. Discussion

\textbf{Convergent evidence for Version B.} Four pieces of independent
evidence converge on the same mechanistic conclusion. (i) Magnitude
invariance: in all three models the deep-layer medical features fire within
0--4\% per token on identical clinical content across format conditions, with
mean-pool modulation indices 0.10--0.27 versus 0.30--0.43 for
magnitude-matched random features in the same SAE basis (Section 4.2). The
result survives a stricter control that further restricts the random pool to
features that fire on clinical content (Section 4.6). (ii) Length-controlled
direction analysis: when prompt-length asymmetry is removed by truncating the
forced-letter block from NL so its content range matches NF's exactly, the
residual-stream difference between conditions vanishes to 0 at numerical
precision. The corpus property that NL's prefix is byte-identical to NF
makes this isolation as clean as the data allows. (iii) Length-invariant
direction analysis: the residual difference that survives max-pool
aggregation loads on \emph{non-medical} features in the SAE basis, with
medical features sitting at the 13--90\% percentile of $|$alignment$|$
across the three models. (iv) Causal-intervention nulls: discrete
ablation of the top format-direction features (Section~4.8) and continuous
ActAdd-style steering along the case-averaged format direction at L29
both produce near-null behavioral effects (0/60 and 2/60 prediction
shifts respectively), establishing that the format direction is detectable
in representation space but not, at single-layer perturbation magnitudes,
isolatable to a few features or a single residual direction with sufficient
causal control to drive letter outputs.

\textbf{Top-token analysis names the format-direction features.} In Gemma 3
4B IT at L29, the top features by alignment with the (NL${-}$NF) max-pool
direction are 3833, 10012, and 980. Their top-activating tokens across
$60\,{\times}\,2 = 120$ prompts are exclusively in NL, on the literal
forced-letter scaffold tokens themselves: feature 3833 fires on
``next'' inside ``\emph{B = See my doctor in the next few weeks}'',
feature 10012 on ``the'' inside ``\emph{D = Go to the ER now Do not include
any explanation}'', feature 980 on ``='' across the answer-key syntax.
We can therefore localize the format effect at the feature level: it lives
in features that detect the structural format of the prompt rather than
its clinical content. The v3-validated medical features 12570, 893, and
12845, by contrast, fire on clinical-content tokens (e.g., ``blood'' in
lab-value contexts, ``facial weakness'', ``tender belly'') at identical
magnitudes across NL and NF conditions.

\textbf{Concurrent work.} Basu et al.~\cite{basu2026interpretability}
demonstrate a 53-percentage-point knowledge--action gap on Qwen 2.5 7B
Instruct using clinical triage vignettes: linear probes discriminate
hazardous from benign cases at 98.2\% AUROC, whereas output sensitivity is
only 45.1\%. They additionally show that four mechanistic interventions to
\emph{correct} this gap, including SAE feature steering, fail to reliably
do so. Our work provides the complementary localization their behavioral
result invites: by varying output format while holding clinical content
byte-identical, we show that the gap is preserved \emph{in the very
direction that varies behaviorally}, and that the format-difference
direction in residual space lives outside the medical-content subspace at
the feature level. Where Basu et al. show that interpretability cannot
\emph{fix} the gap by intervention, we show that interpretability can
\emph{describe} where the gap lives.

\textbf{Behavioral scaling and the depth-dependent mechanistic pattern.}
The forced-letter penalty at Gemma 3 4B (NL accuracy 56.7\% vs NF 70--77\%)
essentially vanishes at Gemma 3 12B (NL 81.7\% vs NF 76.7--81.7\%).
Capability scaling closes the behavioral gap not by changing the clinical
representation (which is preserved at both scales at deep layers) but by
improving the output-mapping circuit's translation of that representation
into a constrained letter answer. The 12B mechanistic data show a depth-
dependent nuance: at deep layers (31, 41) Version B holds and medical
features are more invariant than the magnitude-matched random control,
but at shallow/mid layers (12, 24) the medical features we identify show
\emph{larger} format-induced perturbation than random. The most plausible
account, consistent with prior depth-of-processing observations, is that
shallow features are bound to local lexical/surface forms while deep
features encode the conceptualized clinical state. Format invariance is
therefore a property of conceptual-encoding layers, not lexical ones,
and the 4B finding generalizes to 12B specifically at depths $\geq{}65\%$.

\textbf{Cross-family generality and the SAE fidelity trade-off.} The
Qwen Scope SAE used for the cross-family validation has $\sim{}38\%$
reconstruction error at the layer we use, an intrinsic consequence of
its $k{=}50$ TopK sparsity choice rather than a checkpoint-transfer
artifact. Despite this lower fidelity than Gemma Scope's JumpReLU SAEs,
the medical-vs-random gap survives at L31 with a 95\% bootstrap
confidence interval clearing zero. This is reassuring evidence that the
result is not specific to Gemma Scope's training pipeline; the smaller
Qwen effect size relative to Gemma is, conversely, the kind of
shrinkage one would expect from a noisier dictionary, and we do not
draw fine-grained comparisons across the two pipelines.

\textbf{Implications for evaluation methodology.} Constrained-output
clinical benchmarks~\cite{ramaswamy2026chatgpt} produce headline failure
rates that depend on a property of the evaluation rather than the model's
clinical reasoning. Behavioral
replication~\cite{frailenavarro2026triage} has shown this in
aggregate; our work shows that the model's internal clinical signature on
those failures is preserved across format conditions whose accuracy
scoring diverges. SAE features at the deep encoding layer are a candidate
deployable monitor: a real-time clinical chatbot could check
medical-feature signatures against a calibrated reference distribution
and flag genuine drift in clinical understanding independently of any
format change introduced by the calling application.

The application plays to what SAE features have been shown to be
(interpretable readouts of model state) rather than what
intervention-based approaches have struggled to make them into
(causally-sufficient correctors of behavior). The two intervention
experiments in Section~4.8 are direct evidence for this in our setting:
discrete SAE-feature ablation and continuous ActAdd-style steering on
the same residual direction both produce near-null behavioral effects,
even when the features and direction we target are precisely those
identified by the top-token analysis of Section~4.5. Combined with
concurrent work showing four mechanistic intervention methods fail to
reliably correct triage errors on a different
model~\cite{basu2026interpretability} and our prior null on
cross-lingual rescue~\cite{frailenavarro2026saemad}, the present
evidence indicates that single-layer interventions on small numbers of
features or single residual directions do not provide reliable causal
control of these output behaviors. The monitoring application succeeds
on what SAE features deliver well; an intervention application would
require either richer multi-layer/multi-feature interventions, or
modifying the output stage directly, neither of which we attempt here.

---

## 6. Limitations

\textbf{Scope of model coverage.} Three instruction-tuned models from two
families, one clinical domain (consumer triage). Llama-Scope SAEs and
Mistral-family SAEs were not used here; cross-family generalization
beyond Gemma and Qwen is left to future work. The behavioral
attenuation-with-scale finding rests on a single $4{\rightarrow}12$\,B
within-family comparison.

\textbf{Sample size.} 60 paper-canonical vignettes per model, matched
exactly to the paper-faithful replication corpus~\cite{frailenavarro2026triage}
without subsetting. Stratum-level sample sizes are correspondingly small
($n{=}13$ in the format-flipped stratum at 4B), which is reflected in the
bootstrap confidence intervals reported throughout. We do not run
multi-seed temperature-sampled behavioral runs; greedy decoding was used
across the board.

\textbf{LLM-as-judge dependence.} NF (free-text) scoring relies on two
LLM judges (gpt-5.2-thinking-high and claude-sonnet-4.6) running the
paper-faithful adjudication pipeline (paper-native A=home, D=ER scale).
Inter-rater agreement is $76$--$88\%$ with Cohen's $\kappa$ in
$[0.63, 0.80]$, calibrating the LLM judges against each other but not
against a clinician. We additionally include a clinician-adjudicated
subset of $\,n{=}16$ stratified cases (Appendix~A4) as ground-truth
calibration of the LLM judges.

\textbf{Mechanistic-method scope.} We use mean-pool and max-pool
aggregations of per-token feature activations, magnitude-matched random
baselines (in two variants: unrestricted and content-restricted), and
encoder-direction projection. Richer aggregations (attention-weighted,
context-conditional) and more sophisticated probing methods could surface
patterns we miss. Our causal intervention experiment (Section~4.6 / 4.7)
ablates the format-direction features named by Phase~5; a fuller
intervention sweep over many feature subsets is out of scope for v1.

\textbf{SAE fidelity.} Qwen Scope's intrinsic $\sim{}38\%$ relative L2
reconstruction error at L31 is a property of its $k{=}50$ TopK sparsity
rather than a checkpoint-transfer artifact, but it nonetheless raises
the noise floor of every Qwen-side analysis we report. The cross-family
effect-size shrinkage we observe could be partly attributable to this
noise; a higher-fidelity Qwen-family SAE would let us decompose how much
of the shrinkage is real cross-family attenuation versus measurement
noise.

\textbf{Feature-identification corpus.} Medical-feature identification at
12B and Qwen uses a hand-curated 30-prompt non-medical contrastive
corpus. A larger, programmatically-generated corpus could surface
different features and tighten our magnitude-matched random pools.

\textbf{Activation-decomposition method.} We interpret the
format-direction features identified in Phase~5 via manual top-token
analysis on a held-out corpus. Concurrent work on Natural Language
Autoencoders~\cite{frasertaliente2026nla} suggests an automated
alternative: producing natural-language descriptions of activations
directly, in place of token-level inspection. Applying NLAs to the
format-direction features named here is a natural extension of this
work but requires training NLAs on open-weight models like Gemma~3 and
Qwen3, for which NLAs are not yet publicly available. We leave this
cross-method comparison as future work.

## 7. Conclusion

We test, mechanistically, whether the apparent failures of consumer-facing
clinical LLMs under constrained-output triage benchmarks reflect degraded
clinical reasoning or output-mapping artifacts. Across three
instruction-tuned LLMs from two families (Gemma 3 4B IT, Gemma 3 12B IT,
Qwen3-8B) and across two SAE training pipelines (JumpReLU on Gemma,
$k{=}50$ TopK on Qwen), medical SAE features at the deep encoding layer
are significantly more invariant under output-format change than
magnitude-matched random features in the same SAE basis. When the
prompt-length asymmetry between formats is controlled, the residual-stream
difference vanishes; the residual-direction difference that survives
length-invariant aggregation loads on non-medical features that we
identify as firing on the forced-letter scaffold tokens themselves. The
behavioral format penalty observed at 4B essentially vanishes at 12B,
while the deep-layer mechanistic invariance persists across scales,
families, and the depth at which clinical content is conceptually
encoded.

The model's clinical encoding is preserved across the output formats
whose accuracy scoring diverges: the failure mode the benchmark detects
lives in output mapping, not in clinical reasoning. SAE features are a
deployable, format-invariant monitor of clinical groundedness — an
application that plays to what SAE features have been shown to be
(interpretable readouts) and complements concurrent
work~\cite{basu2026interpretability} showing that mechanistic
interventions cannot directly close the resulting knowledge--action gap.

---

## Appendix

[Sketch — to expand]

- A1: Per-layer × per-stratum bootstrap tables for all three models
- A2: Top-token analysis of the format-effect features (3833, 10012, etc.)
- A3: Adjudicator prompts, agreement statistics, calibration check
- A4: Qwen Scope reconstruction-error characterization
- A5: Compute and cost breakdown
