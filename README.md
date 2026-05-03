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
itself degraded, or is the failure introduced later when the representation is
mapped into a constrained answer format? The behavioral side of this debate is
already in the literature [Ramaswamy et al. 2026, *Nature Medicine*; Fraile
Navarro et al. 2026, arXiv:2603.11413]; we test the mechanistic side.

**Setup.** 60 clinically-canonical vignettes from a paper-faithful
replication corpus, three cells (structured + forced-letter, natural +
forced-letter, natural + free-text), where the forced-letter and free-text
prompts share **byte-identical clinical content** and differ only in whether
a forced-letter instruction block is appended. Three instruction-tuned
models (Gemma 3 4B IT, Gemma 3 12B IT, Qwen3-8B) with their open SAE
releases (Gemma Scope 2 JumpReLU; Qwen-Scope k=50 TopK).

**Findings.**

- **Magnitude invariance.** Medical-content SAE features fire within 0–4%
  per token on identical clinical content across format conditions.
  Mean-pool modulation indices are 10–25% for medical features versus 32–45%
  for magnitude-matched random features in the same SAE basis. Bootstrap
  95% CIs exclude zero in every cell of the layer × stratum design across
  all three models.
- **Direction analysis.** When prompt-length asymmetry is controlled
  (truncating the longer prompt to identical content range), the
  residual-stream difference between conditions vanishes exactly; what
  survives in length-invariant max-pool aggregation loads onto **non-medical
  features** in the SAE basis rather than the medical ones.
- **Behavioral scaling.** The forced-letter penalty observed at 4B
  (+13–20pp advantage for free-text in paper-faithful adjudication)
  essentially vanishes at 12B (≈0pp). The mechanistic invariance at deep
  layers persists across both scales.
- **Cross-family replication.** The deep-layer invariance pattern replicates
  on Qwen3-8B with Qwen-Scope SAEs at L31 despite the SAE's intrinsic ~38%
  reconstruction error (a property of its k=50 TopK sparsity).

Together these support the hypothesis that the model's clinical encoding
is preserved across the output formats whose accuracy scoring diverges; the
failure mode the benchmark detects lives in output mapping. SAE features
emerge as a deployable, format-robust monitor of clinical groundedness.

**Full record.** [`TRIAGE_FINDINGS.md`](TRIAGE_FINDINGS.md) — phase-by-phase
empirical record with bootstrap tables, sanity-check verdicts, and
methodology lessons. [`PAPER_DRAFT.md`](PAPER_DRAFT.md) — workshop paper
draft (in progress) with methods + results sections complete.

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

**Full record.** [`FINDINGS.md`](FINDINGS.md) — complete empirical
record including the retraction.

---

## Repo orientation

| File | Purpose |
|---|---|
| `TRIAGE_FINDINGS.md` | Phase 2 empirical record (519 lines) |
| `PAPER_DRAFT.md` | Workshop paper draft (in progress) |
| `references.bib` | Verified citations for the workshop paper |
| `FINDINGS.md` | Phase 1 empirical record (the closed null) |
| `figures/` | Camera-ready PDFs + PNG previews |
| `phase0_*.py … phase4_*.py` | Phase 2 experimental scripts |
| `v1_*.py, v2_*.py, v3_*.py` | Phase 1 experimental scripts |
| `make_figures.py` | Regenerates all paper figures from `results/` |
| `results/` | All experiment outputs (JSON + CSV) |
| `nature_triage_expanded_replication/` | Cloned reference corpus (gitignored, see prior paper) |

---

## Reproducibility

Phase 2 experiments run on `vast.ai` GPUs (medium tier, 22–48 GB VRAM
depending on model). Total Phase 2 GPU cost ≈ $5 across all three models
including the cross-family Qwen run; LLM-as-judge adjudicator API cost ≈
$2 across two adjudication runs. The figures in `figures/` regenerate
deterministically from the committed `results/*.json`.

The Phase 1 pipeline is documented in the older sections of `FINDINGS.md`
and is fully reproducible end-to-end on a 22GB GPU in ~3 hours, ~$0.80.

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

Full BibTeX in [`references.bib`](references.bib).

---

## Citation

```bibtex
@misc{frailenavarro2026saemad,
  title  = {SAE features as a format-invariance monitor for clinical-AI
            evaluation: cross-family mechanistic evidence},
  author = {Fraile Navarro, David and Magrabi, Farah and Coiera, Enrico},
  year   = {2026},
  url    = {https://github.com/dafraile/SAE_mad},
  note   = {Workshop paper in preparation (target: EMNLP 2026 via ARR).
            See TRIAGE_FINDINGS.md for the full empirical record.}
}
```

## License

MIT. The `nature_triage_expanded_replication/` corpus is reproduced from
the Phase 2 lead author's prior work (arXiv:2603.11413) under the same
license; see that paper's repository for canonical access.
