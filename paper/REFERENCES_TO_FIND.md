# References to find for the workshop paper

This file consolidates every `[CITE: ...]` placeholder in `PAPER_DRAFT.md` so an
external research agent can chase them in a single pass. Each entry includes:
- **Where used**: the section of the paper draft.
- **What we need**: the specific claim or context the citation supports.
- **Search hints**: what's likely to surface, where to look, common pitfalls.

When the agent finds a reference, please return: full BibTeX entry + a 1-line
note on whether the paper says what we claim it says. We've over-claimed citations
in past rounds — verify the actual content matches our use.

---

## Already provided by the user (no search needed)

These two are already in hand; just need to format BibTeX correctly.

1. **Singhal et al. 2023, Med-PaLM, Nature** — `10.1038/s41586-023-06291-2`.
   PDF in Downloads at `s41586-023-06291-2.pdf`. Use this as the foundational
   "LLMs encode clinical knowledge" reference.

2. **Ramaswamy et al. 2026, Nature Medicine** — the triage benchmark whose 51.6%
   under-triage figure is our paper's motivating finding. Likely DOI prefix
   `10.1038/s41591-...`. The user has the full citation in their existing
   replication paper.

3. **Fraile Navarro et al. 2026 — Matters Arising / paper-faithful triage
   replication.** Author/user's own prior work. The user has this. Used in
   the intro for the natural-vs-structured Wilcoxon and the DKA/asthma
   case-level breakdown.

---

## SAE foundations (interpretability lineage)

4. **[CITE: Bricken et al. 2023 — Anthropic monosemanticity / "Towards
   Monosemanticity: Decomposing Language Models with Dictionary Learning"]**
   Used in §1: "SAEs decompose a model's residual stream into a sparse,
   overcomplete dictionary of features."
   - URL: transformer-circuits.pub/2023/monosemantic-features
   - Search: "Bricken monosemanticity Anthropic 2023" or transformer-circuits.pub.
   - Verify: confirms SAE-style sparse decomposition into interpretable features.

5. **[CITE: Cunningham et al. 2023 — "Sparse Autoencoders Find Highly
   Interpretable Features in Language Models"]**
   Used in §1: same context as Bricken — companion foundational reference.
   - arXiv: 2309.08600
   - Verify: SAE feature interpretability claim on autoregressive LMs.

6. **[CITE: Templeton et al. 2024 — Anthropic Claude 3 Sonnet feature analysis,
   "Scaling Monosemanticity"]**
   Used in §1: feature interpretability scales to production-class models.
   - URL: transformer-circuits.pub/2024/scaling-monosemanticity
   - Verify: explicitly demonstrates SAE features carry content-tied signal at scale.

7. **[CITE: Rajamanoharan et al. 2024 — JumpReLU SAEs]**
   Used in §1: "JumpReLU... vs k=50 TopK..." comparison.
   - DeepMind tech report or arXiv ~2024.
   - Search: "JumpReLU SAE Rajamanoharan" or "Improving SAE training"
   - Verify: introduces JumpReLU activation for SAEs; reports reconstruction
     vs sparsity trade-off relevant to our 60–100-active-features claim.

8. **[CITE: Makhzani & Frey 2014 — k-Sparse Autoencoders]**
   Used in §1: TopK SAE foundation.
   - arXiv: 1312.5663
   - Verify: introduces top-k sparse activation for autoencoders. Old paper but
     this is the canonical TopK reference; modern LM-SAE TopK often cites it.

9. **[CITE: Gao et al. 2024 — "Scaling and evaluating sparse autoencoders"]**
   Used in §1: modern TopK SAE for LMs. OpenAI paper.
   - arXiv: 2406.04093
   - Verify: this is OpenAI's TopK SAE paper; reports the sparsity vs reconstruction
     curve we cite.

---

## Open SAE releases for our specific models

10. **[CITE: Lieberum et al. 2024 — Gemma Scope]**
    Used in §1: SAE release for Gemma 2.
    - arXiv: 2408.05147
    - Verify: this is the original Gemma Scope tech report covering Gemma 2 SAEs
      across attn, mlp, residual.

11. **[CITE: Gemma Scope 2, Google DeepMind 2025]**
    Used in §1: the SAE release we actually use for Gemma 3.
    - This may not have an arXiv tech report yet; possibly only a HuggingFace
      model card and DeepMind blog post.
    - Search: "Gemma Scope 2 Google DeepMind 2025" or check
      huggingface.co/google/gemma-scope-2-4b-it for citation guidance.
    - Verify: confirms it's a JumpReLU-style release on Gemma 3; report the
      exact citation format the model card recommends.

12. **[CITE: Qwen Scope blog post / tech report, qwen.ai/blog, 2026]**
    Used in §1: cross-family SAE release.
    - URL: https://qwen.ai/blog?id=qwen-scope (user provided)
    - This is brand new (released ~April 2026). Likely no arXiv yet.
    - Verify: confirms TopK k=50 and the model coverage we describe (Qwen3 base
      models including 8B at residual stream).

---

## Format-sensitivity / MCQ artifact literature

13. **[CITE: Zheng et al. 2024 — "Large Language Models Are Not Robust Multiple
    Choice Selectors"]**
    Used in §1: prior work on constrained-output evaluation underrepresenting
    capability.
    - arXiv: 2309.03882
    - Verify: documents systematic biases in MCQ evaluation that motivate our
      "format affects scoring" framing.

14. **[CITE: Pezeshkpour & Hruschka 2024 — MCQ option-order sensitivity]**
    Used in §1: same context.
    - arXiv: 2308.11483
    - Verify: option-order changes scoring substantially — supports our broader
      framing.

15. **[CITE: Sclar et al. 2024 — Prompt format sensitivity]**
    Used in §1: format effects beyond MCQ.
    - arXiv: 2310.11324 (likely)
    - Verify: shows broad prompt-format sensitivity in LLM benchmarks.

---

## Steering vectors and SAE-as-driver vs SAE-as-readout

16. **[CITE: Turner et al. 2023 — ActAdd / activation addition steering]**
    Used in §1 conclusion paragraph: SAE as monitor (readout) rather than
    steering (driver).
    - arXiv: 2308.10248 (user already linked)
    - Verify: introduces continuous activation-difference steering and shows it
      preserves off-target behavior.

17. **[CITE: Marks et al. 2024 — SAE feature ablation results / sparse feature
    circuits]** *(optional but strengthens the argument)*
    Used in §1 conclusion: documenting cases where SAE features are NOT causal
    levers, motivating our readout framing.
    - arXiv: 2403.19647 (likely — "Sparse Feature Circuits")
    - Verify: confirms ablation of individual SAE features doesn't always
      change downstream behavior.

18. **[CITE: Anthropic emotion feature paper / "On the Biology of a Large
    Language Model" or similar]** *(optional — strengthens the
    "feature-as-readout vs feature-as-driver distinction" claim)*
    Used in §1: prior precedent for the distinction we operationalize.
    - This may be the Anthropic Claude 3 Opus emotion paper or the
      "biology of a large language model" Anthropic publication.
    - Search: "Anthropic emotion feature steering Claude" or
      "biology large language model Anthropic 2024 emotional features".
    - Verify: confirms that some SAE features ARE causal drivers (emotional
      response, sycophancy etc.) — this is the contrast point we want.

---

## Optional / nice-to-have

19. **[CITE: Wei et al. 2022 — Emergent abilities of LLMs]** *(optional)*
    Used in §1 contributions list: the scaling-improves-clinical-Q&A claim
    builds on this; could be cited if discussing capability emergence.
    - arXiv: 2206.07682
    - Verify: documents the canonical scaling-induced ability emergence pattern.

20. **[CITE: Anthropic / OpenAI / Google Gemini system cards]** *(optional)*
    Could be cited where we discuss "current open-weight LLMs" or specific
    model documentation.
    - Skip unless space allows.

---

## In our own prior work (link, don't search)

21. The v3 SAE-rescue null result. This is committed in `FINDINGS.md` of this
    same repository (github.com/dafraile/SAE_mad). When framing the
    "feature-as-readout vs feature-as-driver" contribution we can self-cite:
    > "Prior work in our own group [CITE: Fraile Navarro 2026 — SAE-Guided
    > Cross-Lingual Knowledge Transfer in LLMs: A Null Result with Preserved
    > Findings, github.com/dafraile/SAE_mad] found that single-feature
    > amplification of medical SAE features does not rescue cross-lingual
    > medical QA performance, consistent with the readout interpretation
    > exploited here."

---

## Notes for the agent

- **NeurIPS workshop format**: typically a 4–9 page paper with light bib.
  Aim for ~20-25 references total (we currently have ~20 placeholders;
  some optional ones can be dropped if space is tight).
- **Verification matters**: when we found Med-PaLM in our reading, the actual
  paper said something more nuanced than the casual "LLMs know clinical
  knowledge" headline (humans evaluated answers along multiple axes including
  factuality, comprehension, reasoning, harm, bias; the model is good at
  factuality, less good at omission/harm). Apply that level of care to every
  reference: read the abstract + key results before adopting the citation.
- **Return format**: BibTeX entry, the URL, and a one-line note saying whether
  the reference says what we claim. If a claim doesn't quite match, flag that
  too — we'd rather drop a citation than misrepresent it.
