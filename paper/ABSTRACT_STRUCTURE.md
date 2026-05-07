# Abstract structure — for the author to write in their own voice

This file lays out an abstract structure (~250 words) as numbered
sentence-slots with talking points. Each slot is one sentence or compound
clause. The substantive content of each slot is set; the actual prose is
the author's to write.

The structure leads with the **actionable framing** rather than the
academic incremental framing: the model's clinical understanding is
preserved; the failure is in output mapping; this reframes a benchmark-
headline safety concern as a tractable evaluation/output-stage problem.

## Slot-by-slot

### Slot 1 — Hook (1 sentence)

State the empirical concern in the literature concretely:
- A consumer-facing health LLM was reported to under-triage 51.6% of
  emergencies in a recent benchmark
- Cite Ramaswamy et al. 2026 (Nature Medicine)
- Tone: this is a high-stakes claim that's been read as evidence of
  unsafe clinical reasoning

### Slot 2 — Behavioral context (1 sentence)

Set up the open question:
- Behavioral replications including our own group's prior work show that
  the failure rate is highly format-sensitive
- Cite Fraile Navarro et al. 2026 (the user's prior arXiv paper)
- The open question: is the failure in clinical reasoning, or in the
  output stage?

### Slot 3 — Our claim (the headline) (1 sentence)

The core actionable finding:
- We provide mechanistic evidence that the model's internal clinical
  representation of these cases is preserved across format conditions
- Whose accuracy diverges substantially under standard adjudication
- Therefore the failure mode lives in output mapping, not in clinical
  reasoning

### Slot 4 — Method snapshot (1–2 sentences)

What we actually did, briefly:
- Sparse autoencoder feature analysis
- Three instruction-tuned LLMs from two families (Gemma 3 4B IT, Gemma 3
  12B IT, Qwen3-8B)
- Two SAE training pipelines (Gemma Scope JumpReLU, Qwen Scope k=50 TopK)
- 60 paper-canonical clinical vignettes from your prior work, three cells

### Slot 5 — Magnitude finding (1 sentence)

First piece of mechanistic evidence:
- Medical SAE features fire identically (within 0–4% per token) on
  identical clinical content across format conditions
- Modulation indices: 10–25% for medical features versus 32–45% for
  magnitude-matched random features in the same SAE basis
- Bootstrap 95% CIs exclude zero in every cell

### Slot 6 — Direction finding (1 sentence)

Second piece, with the byte-identical content trick:
- When the prompt-length asymmetry between conditions is controlled
  (truncating to byte-identical content range), the residual-stream
  difference vanishes exactly
- The format direction that survives length-invariant aggregation loads
  on non-medical features rather than the clinical ones

### Slot 7 — Feature interpretation (1 sentence) (NEW POSITION OF EMPHASIS)

The third piece, which is the strongest mechanistic claim and the
paper's interpretive contribution:
- We name the features that carry the format direction: SAE features
  whose top activations are exclusively on the literal forced-letter
  answer-key scaffold tokens themselves
- E.g., features that fire on "next" within "B = See my doctor in the
  next few weeks", on "the" within "D = Go to the ER now", on "="
  across the answer-key syntax
- The format effect is mechanistically localized to features detecting
  the structural format of the prompt, not its content

### Slot 8 — Behavioral scaling (1 sentence, optional but worth keeping)

The 4B → 12B attenuation:
- The behavioral format penalty observed at 4B (+13–20pp gap between
  forced-letter and free-text) essentially vanishes at 12B
- The deep-layer mechanistic invariance persists across both scales

### Slot 9 — Implication (1 sentence) — THE actionable line

The policy / deployment claim:
- The benchmark headline is partly an evaluation artifact: the model's
  clinical signature on its apparent failures is preserved
- This recasts the safety concern from "the model doesn't reason
  clinically" (intractable, opaque) to "the output mapping under
  constrained format degrades" (tractable, addressable at the
  evaluation and deployment layers)
- SAE features at the deep encoding layer offer a deployable
  format-invariant monitor of clinical groundedness

## Notes on tone

- Lead with the empirical claim, not the method.
- The reader should know after sentence 3 what we found, not be waiting
  through methods.
- Keep "we provide mechanistic evidence" — the word "mechanistic" is
  load-bearing for the reviewer audience, signals what kind of paper
  this is.
- Don't bury the policy claim. The last sentence is the one a clinical
  reviewer reads twice.

## Length target

~230–270 words. The version currently in PAPER_DRAFT.md is around 270
words, which fits the EMNLP-style abstract length cap. If you find
yourself going over 280, cut Slot 8 (the scaling claim) — it's the most
expendable in the abstract specifically.

## What this is NOT

- Not the actual abstract prose. The author writes the words.
- Not a prescription of phrasing. The talking points constrain
  *content*; word choice is open.
- Not a replacement for the body of the paper. The body needs its own
  rewrite pass; this file is just the abstract scaffold.
