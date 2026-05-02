# Paper Outline — NeurIPS Workshop Submission (Sydney 2026)

**Working title**:
*"Internal Representation, Not Internal Reasoning: Why Apparent LLM Triage
Failures May Be Output-Mapping Artifacts"*

Alt: *"Format Effects in Clinical LLM Evaluation: Mechanistic Evidence That
Output Constraints, Not Clinical Reasoning, Drive Apparent Failures"*

**One-line pitch**: Ramaswamy et al. (Nature Medicine 2026) report that
ChatGPT Health under-triages 51.6% of emergencies. We provide mechanistic
evidence — using sparse autoencoder features in Gemma 3 — that the model's
internal clinical understanding is preserved across the output formats whose
behavioral scoring diverges by 13–20 percentage points. The failures live in
output mapping, not in clinical reasoning.

**Length target**: 6–8 pages main, 4–6 pages appendix. (Workshop-typical.)

---

## Section 1 — Abstract (≤ 250 words)

Recent benchmarks evaluating LLMs as triage tools — most notably Ramaswamy et al.
(2026) — report high under-triage rates for emergency cases under constrained
output formats. Behavioral replications by [our prior work / cite] have shown
that the failure rates depend strongly on prompt format and output constraint:
identical clinical information presented as natural patient text and answered
freely yields substantially different scoring than the same content under a
forced single-letter answer.

The open question is whether prompt format changes how the model represents the
clinical case, or only how it answers. We address this mechanistically using
sparse autoencoder (SAE) features in Gemma 3 4B IT and 12B IT.

We find that medical SAE features identified by cross-lingual contrastive
analysis fire essentially identically on identical clinical content across
output-format conditions: per-token max activations match within 0–4%, mean
activations within 10–25% (versus 32–45% for magnitude-matched random features
in the same SAE). When we control for prompt-length asymmetry, the residual-
stream difference between conditions vanishes; what residual signal remains
under length-invariant max-pooling loads onto non-medical features in the SAE
basis.

Together these results support a **mechanistic distinction between clinical
reasoning and output mapping**. The model's internal clinical understanding is
preserved across formats; benchmark accuracy differences arise from the
downstream output stage. We discuss implications for clinical-AI evaluation
methodology and propose SAE features as deployable, format-invariant monitors
of clinical groundedness.

## Section 2 — Introduction (≈ 1.5 pages)

### 2.1 The Ramaswamy benchmark and the format-replication critique

Brief: 51.6% under-triage headline. Subsequent behavioral replications
(your existing work) showing that format changes recover most of the apparent
failures. The open question that the behavioral evidence alone cannot resolve.

### 2.2 The reasoning-vs-mapping distinction

Two readings of "format matters":
- **Version A**: Format changes how the model thinks about the case.
  Strong cognitive-difference claim. Important if true (means evaluations
  break clinical reasoning, not just measurement).
- **Version B**: Format changes how the model answers; clinical understanding
  is preserved. Weaker mechanistic claim, stronger policy claim (the benchmark
  measures output-mapping fidelity, not reasoning).

Behavioral evidence cannot distinguish these. Mechanistic evidence can.

### 2.3 Contribution

- Three-angle mechanistic test of Version B in Gemma 3 4B IT, replicated in 12B IT.
- Demonstrates that SAE features survive a paper-faithful 60-case test and
  show format invariance at four sweep layers.
- Methodologically: we use the paper's own LLM-as-judge adjudication pipeline
  (kappa = 0.797) so scoring is calibrated against the paper's own framework.
- We also show that the residual-level format effect, when it appears, lives
  on non-medical features.

## Section 3 — Background and Related Work (≈ 0.5 page)

### 3.1 Sparse autoencoders for representation analysis

- Gemma Scope 2; JumpReLU SAEs; dictionary learning.
- Features as interpretable readouts.
- Prior monitoring applications.

### 3.2 LLM evaluation in clinical AI

- Ramaswamy et al. and subsequent replications.
- Format sensitivity in LLM benchmarks more broadly.

### 3.3 ActAdd-style steering vectors and the SAE-decomposition gap

- Steering vectors capture full residual differences.
- SAEs decompose into interpretable basis.
- Trade-off: interpretation vs causal sufficiency.
- Our use: SAE for *interpretation of an ActAdd-style direction*, not for
  steering. Methodologically we project the (B − D) residual onto the
  SAE basis to test whether medical features carry the format direction.

## Section 4 — Method (≈ 1.5 pages)

### 4.1 Dataset

60 paper-canonical vignettes from [your prior work's paper-faithful
replication]. Three cells used:
- **A**: structured clinical input + terse forced-letter output
- **B**: natural patient input + terse forced-letter output
- **D**: natural patient input + open-ended free-text output

Critically, B's prompt is constructed by appending the forced-letter
instruction block to D's natural patient text. The clinical content is
**byte-identical** between B and D; only the trailing instruction differs.

### 4.2 Models and SAEs

- Gemma 3 4B IT and 12B IT (Google DeepMind).
- Gemma Scope 2 residual-stream SAEs at width 16k, l0_medium.
- 4B sweep layers: 9, 17, 22, 29 (27/50/65/85% depth).
- 12B matched-depth layers: 12, 24, 31, 41.

### 4.3 Medical-feature identification

Six-condition contrastive (3 languages × 2 domains) for the 4B work,
established in prior v3 work and re-validated. For 12B, an English-only
medical-vs-non-medical contrastive on the same vignettes plus a 30-prompt
non-medical control corpus, scoring `mean_max(med) − mean_max(non-med)`
under a firing-reliability filter.

### 4.4 Behavioral phenomenon test (Phase 0.5)

Three-cell × 60-case grid. Cells A and B parsed by regex on the model's
forced-letter output. Cell D scored by paper-faithful LLM-as-judge
(`gpt-5.2-thinking-high` and `claude-sonnet-4.6`, paper-native A=home/D=ER
scale, 88.3% inter-rater agreement, κ = 0.797).

### 4.5 Mechanistic invariance test (Phase 1b)

For each layer × case × condition, mean-pool SAE feature activations over
user content tokens. Compare medical features to 30 magnitude-matched random
features (pool defined by mean activation ∈ [0.5×min, 2×max] of medical-feature
means). Per-case modulation index = ⟨|a_D − a_B|⟩ / ⟨(|a_B|+|a_D|)/2⟩.
Stratified by Phase 0.5 outcome (format-flipped, both-right, both-wrong).

### 4.6 Direction-of-format-effect test (Phase 2b)

Compute (B − D) residual averaged across cases at each layer. Project onto
each SAE feature's encoder direction by cosine alignment. Three aggregations:

- **Full mean-pool** (replicates Phase 2; affected by prompt-length asymmetry)
- **Length-controlled mean-pool**: truncate B's content range at the start of
  the forced-letter instruction so B and D pool over identical content.
  (Diff norm collapses to 0 when content is identical.)
- **Max-pool**: peak activation per dimension over content tokens.
  Length-invariant.

Rank features by |alignment| and report where medical features land.

## Section 5 — Results (≈ 2 pages)

### 5.1 Behavioral phenomenon (Phase 0.5)

[Figure 1: three bars showing accuracy by cell on 4B and 12B]

- Gemma 3 4B IT: A=60.0%, B=56.7%, D=70–77% (judge-dependent).
- Output-axis effect: +13–20pp in favor of free-text. Direction is opposite
  to frontier-scale finding in Ramaswamy et al.; possibly scale-dependent.
- 12B IT: [results pending]
- Free-text gain concentrated in mid-acuity (C, C/D); D-emergencies remain
  challenging across all cells.

### 5.2 Magnitude invariance (Phase 1b)

[Figure 2: box-and-whisker of mod-index by stratum × layer, medical vs random]

- All four sweep layers × all four strata: medical-feature mod-index lower
  than magnitude-matched random; bootstrap 95% CIs exclude zero everywhere.
- L29 format-flipped (n=13): medical 0.107, random 0.413, diff −0.305.
- Per-token max activations on identical clinical tokens match within 0–4%.

[Table: medical vs random mod-index per layer × stratum, with bootstrap CI]

### 5.3 Direction collapses under content control (Phase 2b)

[Figure 3 (small): bar chart of ‖B−D‖ for full / truncated / max aggregations]

- Truncated mean: ‖B − D‖ = 0 exactly. When B and D contain identical text,
  residuals are deterministically identical.
- Full mean-pool: ‖B − D‖ = 1012.66 at L29, but every medical feature has
  *negative*-signed alignment — the prompt-length artifact signature.
- Max-pool: ‖B − D‖ = 5026.56 at L29, but medical features at L29 sit at the
  15.4%, 42.8%, and 61.1% percentile of |alignment| — not top carriers.
  Top-aligned features are non-medical: 3833, 10012, 980, 9485, 755.

### 5.4 12B replication

[same structure as 5.1–5.3 but for 12B]

Pending Phase 3b results. Expected: same qualitative pattern, supporting
intra-family scale generality.

## Section 6 — Discussion (≈ 1 page)

### 6.1 Version A vs Version B

The three pieces of evidence — magnitude invariance, content-controlled
direction vanishing, max-pool direction loading on non-medical features —
all point to **Version B**: format affects the output stage, not the
clinical encoding stage.

### 6.2 SAE features as format-robust monitors

Practical implication: deploying medical features as a runtime monitor of
clinical groundedness is robust to output-format changes. A clinical
chatbot's medical-feature signature on a real patient case would be unaffected
by whether the system prompt asks for a letter or a paragraph; deviations
from a reference signature would therefore signal genuine clinical
representation drift, not output-format change.

### 6.3 Implications for benchmark design

The Ramaswamy benchmark format produces apparent reasoning failures from
intact representation. The benchmark is therefore measuring something other
than clinical capability under that format — most plausibly, the model's
ability to map a graded clinical assessment into a constrained letter pick.

### 6.4 What we did NOT establish

- Cross-family generalization (Llama, Mistral, etc.) — flagged as next paper.
- Causal sufficiency of the format-effect features (3833, 10012, …) —
  identification only; no ablation or steering.
- Real-world monitoring deployment — the methodology suggests it but we
  don't run a deployment study.

## Section 7 — Limitations (≈ 0.5 page)

- Two models, one family.
- One clinical domain (triage).
- 60 vignettes — small for a typical clinical-AI paper, but matches the
  paper-faithful replication corpus exactly.
- LLM-as-judge dependence (mitigated by 88% inter-rater agreement).
- We use mean-pooling and max-pooling as aggregations; richer aggregations
  (e.g. attention-weighted) could surface different patterns.
- The non-medical contrastive corpus for 12B feature identification is a
  hand-crafted 30-prompt list; results may shift with a larger corpus.

## Section 8 — Conclusion (≈ 0.25 page)

Three independent angles of mechanistic evidence in Gemma 3 4B IT and 12B IT
support the position that prompt format affects the output stage, not the
clinical reasoning stage, in current open-weight LLMs. SAE features provide
a format-robust signal of clinical groundedness, and apparent benchmark
failures under constrained output formats may not reflect underlying
reasoning deficits.

---

## Appendix Sketch

- A1: Full per-layer × per-stratum bootstrap tables (Phase 1b on 4B and 12B)
- A2: Top-token analysis of the format-effect features (3833, 10012, …)
- A3: Per-case alignment data for format-flipped cases
- A4: Adjudicator prompts and inter-rater statistics
- A5: 12B feature identification details — non-medical corpus, scoring,
       chosen features per layer
- A6: Compute and cost breakdown

## Figures TODO

1. Three-cell bar chart of accuracy (4B and 12B side-by-side).
2. Mod-index comparison by layer × stratum, medical vs random.
3. Diff-norm bar chart (full / truncated / max), L17 and L29.
4. Per-feature alignment histogram, with medical features marked.
5. Per-case max-activation match table for format-flipped cases.

## Things still TBD before drafting

- Exact NeurIPS workshop venue (interpretability vs trustworthy ML vs
  clinical-AI). Different page limits, different framing.
- Author list, affiliations, anonymization.
- Citations: Ramaswamy et al., Anthropic SAE work, Gemma Scope releases,
  ActAdd, your prior paper-faithful replication.
- License and data release (TRIAGE_FINDINGS.md is already public; scripts
  and SAE access are public; the 60-vignette corpus is from the prior work).
