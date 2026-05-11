# SAE_mad — Mechanistic interpretability for clinical-AI evaluation

A research repository spanning two related projects on sparse-autoencoder
(SAE) features in instruction-tuned LLMs:

1. **Phase 2 — SAE features as a format-invariance monitor (active work).**
   Cross-family mechanistic test of whether the apparent triage failures of
   consumer-facing LLMs under constrained-output evaluation are
   *output-mapping* artifacts rather than failures of clinical reasoning.
   Three models, two SAE training pipelines, full pipeline preregistered with
   bootstrap controls. Targeting **EMNLP 2026** (submission via ARR May 25).

2. **Phase 1 — SAE features for cross-lingual rescue (null result, complete).**
   An earlier project that hypothesized SAE features could serve as routing
   signals for cross-lingual capability transfer. Initial 3–7% "rescue"
   results retracted after external review surfaced a dataset bug; controlled
   replications across five configurations confirmed no effect above a
   random-feature noise floor. The project closed cleanly with the null.

The two phases share the core observation that emerged from Phase 1 and
motivates Phase 2: **SAE features are reliable readouts of clinical
content but are not, in our setting, individually causal levers** — a finding
since corroborated by Basu et al. (2026), who report zero effect from SAE
feature steering when trying to correct triage errors on a different model
family.

---

## Phase 2 (active, SAE-as-detector) — TL;DR

**Question.** When a clinical LLM appears to fail under constrained
forced-letter triage evaluation, has the model's *clinical representation*
itself degraded, or is the failure introduced later when the representation
is mapped into a constrained answer format? The behavioral side of this
debate is already in the literature [Ramaswamy et al. 2026,
*Nature Medicine*; Fraile Navarro et al. 2026, arXiv:2603.11413]; we test
the mechanistic side.

**Setup.** 60 clinically-canonical vignettes from a paper-faithful
replication corpus, three cells (structured + forced-letter, natural +
forced-letter, natural + free-text), where the forced-letter and free-text
prompts share **byte-identical clinical content** and differ only in whether
a forced-letter instruction block is appended. Three instruction-tuned
models (Gemma 3 4B IT, Gemma 3 12B IT, Qwen3-8B) with their open SAE
releases (Gemma Scope 2 JumpReLU; Qwen-Scope k=50 TopK).

**Headline finding.** **The model's clinical encoding of these cases is
preserved across the format conditions whose accuracy scoring diverges.**
The residual-stream direction along which the forced-letter and free-text
conditions differ lives in SAE features that fire on the literal answer-key
scaffold tokens themselves — not in the features that encode clinical
content. The benchmark's apparent reasoning failures are therefore largely
output-mapping artifacts at the small-and-mid scale, with the behavioral
gap closing entirely by 12B.

This is supported by four independent pieces of evidence:

1. **Magnitude invariance.** Medical-content SAE features fire within 0–4%
   per token on identical clinical content across format conditions.
   Mean-pool modulation indices are 10–25% for medical features versus
   32–45% for magnitude-matched random features in the same SAE basis;
   bootstrap 95% CIs exclude zero in every cell of the layer × stratum
   design across all three models.
2. **Direction analysis.** When prompt-length asymmetry is controlled
   (truncating the longer prompt to identical content range), the
   residual-stream difference between conditions vanishes exactly; what
   survives in length-invariant max-pool aggregation loads onto
   **non-medical features**.
3. **Feature-level interpretation.** The top features by alignment with
   the (NL−NF) max-pool direction at Gemma 4B L29 fire *exclusively* in
   NL on the literal forced-letter answer-key scaffold tokens — feature
   3833 on "next" inside "B = See my doctor in the next few weeks",
   feature 10012 on "the" inside "D = Go to the ER now", feature 980 on
   the "=" symbol of the answer-key syntax. The format direction is
   localized to features detecting prompt structure, not clinical content.
4. **Causal-intervention nulls.** Neither discrete SAE-feature ablation
   of the format-direction features (0/60 letter changes) nor continuous
   ActAdd-style steering along the same direction (2/60 within-permissive-
   gold changes at α corresponding to 6.7% of the residual norm) produces
   meaningful behavioral shifts. The format direction is detectable and
   interpretable but, at single-layer perturbation magnitudes, not
   isolatable to a few features or a single residual direction with
   sufficient causal control to drive outputs. This corroborates concurrent
   work showing four mechanistic intervention methods fail on the same
   task family on a different model [Basu et al. 2026, arXiv:2603.18353].

**Cross-family + scaling extensions.** The deep-layer invariance replicates
on Qwen3-8B with Qwen-Scope SAEs at L31 despite that SAE's intrinsic ~38%
reconstruction error from its k=50 TopK sparsity. The forced-letter
behavioral penalty observed at Gemma 4B (+13–20pp NF advantage in
paper-faithful adjudication) essentially vanishes at Gemma 12B (≈0pp);
the deep-layer mechanistic invariance persists across both scales.

**Reading.** SAE features at the deep encoding layer are a deployable,
format-invariant monitor of clinical groundedness — an application that
plays to what SAE features are well-supported as (interpretable readouts
of model state) rather than what intervention-based methods, including
ours, struggle to make them into (causally-sufficient correctors of
behavior).

**Full record.** [`paper/TRIAGE_FINDINGS.md`](paper/TRIAGE_FINDINGS.md) —
phase-by-phase empirical record with bootstrap tables, sanity-check
verdicts, and methodology lessons. [`paper/PAPER_DRAFT.md`](paper/PAPER_DRAFT.md)
— workshop paper draft with all sections in prose form (currently
AI-cadenced; the prose will be re-authored by the lead author before
EMNLP submission). [`paper/ABSTRACT_STRUCTURE.md`](paper/ABSTRACT_STRUCTURE.md)
— sentence-slot scaffold for the abstract.

---

## Phase 1 (closed null) — TL;DR

**Hypothesis.** Amplifying language- or domain-specific SAE features at
inference time could rescue cross-lingual performance gaps (e.g., improve
Arabic medical QA by steering toward English features).

**Initial result.** 3–7% rescue effects on a multilingual medical QA benchmark.

**External review.** Surfaced four methodological issues, the most serious
being that our "English" baseline was actually Arabic — MMMLU's `default`
config is non-English. Controlled replications using `cais/mmlu` for real
English, full-benchmark evaluation, random-feature controls, and explicit
single- vs multi-feature distinctions confirmed:

> **Under proper evaluation, no rescue effect exists above the random-feature
> noise floor in any of five tested configurations.**

**What survived as positive findings:**

- Cross-lingual representations in Gemma 3 1B show a depth signature
  (lexical/surface at shallow layers, conceptual at late layers).
- SAE features can causally control output language at the small scale
  (Feature 857 at layer 22 of Gemma 3 1B → graded continuous dial for
  Spanish output).
- Language-agnostic medical-content features in Gemma 3 4B at layer 29
  (features 893, 12570, 12845) fire on EN/ES/FR medical content and zero
  on non-medical — these are the features Phase 2 builds on.

**Lessons captured for future work.** A `/sanity-check` skill (in
`~/.claude/commands/`) was distilled from the failure mode that produced
the original 3–7% rescue claim. It enforces an adversarial checklist on any
result that supports the working hypothesis: did we verify the dataset that
was actually loaded, do we have a real noise floor, what alternative story
would also produce this result. The skill has subsequently caught two
methodology-level issues in Phase 2 that smoke-testing alone would not have
caught.

**Full record.** [`exploratory/FINDINGS.md`](exploratory/FINDINGS.md) —
complete empirical record including the retraction.

---

## Repo orientation for collaborators

If you've just been given access, here's what to read for what.

**Just want the gist** → this README + the headline finding above. ~5 min.

**Reviewer-level read** → [`paper/TRIAGE_FINDINGS.md`](paper/TRIAGE_FINDINGS.md),
the empirical record. Phase-by-phase, with bootstrap tables, sanity-check
verdicts, methodology lessons, and the convergent reading of Phases 5–7.
~30 min.

**Paper read** → [`paper/PAPER_DRAFT.md`](paper/PAPER_DRAFT.md). All sections
in prose form, with figure references and citation keys. Disclaimer: prose
voice is recognizably AI-cadenced and will be re-authored by the lead author
before EMNLP submission per ACL paper-integrity policy. The structure,
arguments, and findings are stable; the words are not. The
[`paper/ABSTRACT_STRUCTURE.md`](paper/ABSTRACT_STRUCTURE.md) is a
sentence-slot scaffold for the abstract written so the author can fill in
voice.

**Want to verify a specific claim in the paper** → look up the relevant
phase in [`paper/TRIAGE_FINDINGS.md`](paper/TRIAGE_FINDINGS.md), then go to
the matching `paper/scripts/phase*.py` script and `results/phase*_*.json`
output. The figure-generation pipeline is `paper/make_figures.py` and
`paper/make_fig4.py`, which re-run deterministically from the JSONs at the
project root.

**Clinician adjudication package** (out for review at time of writing)
→ [`clinician_package/`](clinician_package/). Sixteen blinded cases
stratified by behavioral outcome; the unblinding key is in the same
folder for our internal post-adjudication analysis.

### Layout

```
.
├── README.md                       # This file
├── paper/                          # Active workshop paper
│   ├── PAPER_DRAFT.md              # Full draft (awaiting author rewrite)
│   ├── ABSTRACT_STRUCTURE.md       # Sentence-slot scaffold for the abstract
│   ├── TRIAGE_FINDINGS.md          # Lab notebook — canonical empirical record
│   ├── references.bib              # Verified BibTeX
│   ├── make_figures.py, make_fig4.py
│   ├── figures/                    # fig1–fig4 PDF (vector) + PNG (preview)
│   └── scripts/                    # one script per phase
├── exploratory/                    # Closed prior work (Phase 1 multilingual rescue null)
│   ├── README.md, FINDINGS.md
│   ├── corpus.json, corpus_template.json
│   ├── hw{1..5}_*.py, v1_*.py, v2_*.py, v3_*.py
├── results/                        # All experiment outputs (JSON + CSV)
├── clinician_package/              # Blinded adjudication package + unblinding key
├── infra/                          # Vast.ai bootstrap helpers
└── nature_triage_expanded_replication/  # Cloned reference corpus (gitignored)
```

### Phase index for the active project

All scripts below live under `paper/scripts/`. Run from project root.

| Phase | What it tests | Result | Script | Output |
|---|---|---|---|---|
| 0 | Capability floor on EXPLANATION+TRIAGE scaffold | 4B 68%, 12B 67% | `phase0_capability_floor.py` | `results/phase0_capability_floor.json` |
| 0.5 | Three-cell behavioral phenomenon (4B+12B) | NF outperforms NL by +13–20pp at 4B; gap closes at 12B | `phase0_5_three_cells.py`, `phase3b_12b_pipeline.py` | `results/phase0_5_*.json`, `results/phase3b_12b_*.json` |
| 1b | Magnitude-matched mod-index invariance test | Medical features more invariant than random; CIs exclude zero | `phase1b_magnitude_matched.py` | `results/phase1b_magnitude_matched.json` |
| 2b | Length-controlled + max-pool direction analysis | Length-controlled diff = 0 exactly; max-pool format direction loads on non-medical features | `phase2b_dilution_check.py` | `results/phase2b_dilution_check.json` |
| 3 | 12B medical-feature identification | 4 layers, hundreds of filter-passing features each | `phase3_12b_feature_id.py` | `results/phase3_12b_features.json` |
| 4 | Cross-family Qwen3-8B replication at L31 | Effect replicates with smaller magnitude (k=50 TopK noise) | `phase4_qwen_minimal.py` | `results/phase4_qwen_L31.json` |
| 5 | Top-token analysis + restricted random pool | Format-direction features fire literally on forced-letter scaffold tokens | `phase5_top_tokens_and_restricted_random.py` | `results/phase5_top_tokens.json`, `results/phase5_restricted_random.json` |
| 6 | SAE feature ablation (causal intervention I) | 0/60 letter changes; hook verified, perturbation 0.4% of residual | `phase6_causal_intervention.py`, `phase6_debug.py` | `results/phase6_causal_intervention.json` |
| 7 | ActAdd-style steering at L29 (causal intervention II) | 2/60 within-gold shifts at α=4 (~6.7% perturbation); near-null | `phase7_steering_vector.py` | `results/phase7_steering_vector.json` |

---

## Reproducibility

Phase 2 experiments run on `vast.ai` GPUs (medium tier, 22–48 GB VRAM
depending on model). Total Phase 2 GPU cost ≈ $5 across all three models
including the cross-family Qwen run; LLM-as-judge adjudicator API cost ≈
$2 across two adjudication runs. The figures in `paper/figures/` regenerate
deterministically from the committed `results/*.json`.

The Phase 1 pipeline is documented in `exploratory/FINDINGS.md` and is fully
reproducible end-to-end on a 22GB GPU in ~3 hours, ~$0.80.

Models used (all gated on HuggingFace, requires HF auth):
`google/gemma-3-4b-it`, `google/gemma-3-12b-it`, `Qwen/Qwen3-8B`.

SAE releases used:
- [`google/gemma-scope-2-4b-it`](https://huggingface.co/google/gemma-scope-2-4b-it) (JumpReLU, l0_medium, residual stream)
- [`google/gemma-scope-2-12b-it`](https://huggingface.co/google/gemma-scope-2-12b-it) (JumpReLU, l0_medium, residual stream)
- [`Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`](https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50) (TopK k=50, residual stream)

---

## Related work

The Phase 2 work sits at the intersection of three literatures:

- **Consumer-AI clinical evaluation**: Ramaswamy et al. 2026, *Nature
  Medicine*; Fraile Navarro et al. 2026, arXiv:2603.11413 (the prior
  behavioral replication, our group); Basu et al. 2026, arXiv:2603.18353
  (concurrent clinical-mechanistic work showing a 53pp knowledge-action gap
  and SAE-feature-steering null on Qwen 2.5 7B).
- **Prompt-format and constrained-output sensitivity**: Sclar et al. 2024;
  Zheng et al. 2024; Pezeshkpour & Hruschka 2024.
- **Mechanistic interpretability of LLMs**: Bricken et al. 2023;
  Cunningham et al. 2023; Templeton et al. 2024; Lieberum et al. 2024
  (Gemma Scope); Anthropic 2025 (attribution-graph biology); the
  conceptual-precedent line through Burns 2023 (latent knowledge),
  Kadavath 2022 (model self-knowledge), Turpin 2023 (unfaithful
  verbalization).

Full BibTeX in [`paper/references.bib`](paper/references.bib).

---

## Citation

```bibtex
@misc{frailenavarro2026saemad,
  title  = {SAE features as a format-invariance monitor for clinical-AI
            evaluation: cross-family mechanistic evidence},
  author = {Fraile Navarro, David},
  year   = {2026},
  url    = {https://github.com/dafraile/SAE_mad},
  note   = {Workshop paper in preparation (target: EMNLP 2026 via ARR).
            See paper/TRIAGE_FINDINGS.md for the full empirical record.}
}
```

## License

MIT. The `nature_triage_expanded_replication/` corpus is reproduced from
the Phase 2 lead author's prior work (arXiv:2603.11413) under the same
license; see that paper's repository for canonical access.
