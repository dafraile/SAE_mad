# SAE-Guided Capability Routing: Experimental Plan

## Context for the implementing assistant

This document is a handoff from a long design conversation between a user and Claude. The user is technically literate, has good research taste, is trilingual (English/Spanish/French), but is not primarily an ML engineer. They want to start running real experiments toward a research direction we developed together. Your job is to help them turn this plan into working code on their local machine.

Be willing to push back if something here is wrong or impractical in their environment, but understand that the *research question* and *experimental scoping* were arrived at carefully and shouldn't be casually rewritten. The engineering choices (libraries, exact model sizes, notebook structure) are open to revision based on what's actually available and what runs on their hardware.

## The big-picture research question

Modern LLMs activate essentially all of their parameters for every token. Mixture-of-Experts is the mainstream attempt at sparse, conditional computation, but mainstream MoE keeps experts parameter-disjoint, which is wasteful and doesn't match how brains seem to organize specialized-but-cooperating subsystems.

The hypothesis we're poking at is: **interpretability features (specifically SAE features) and architectural routing signals (MoE gates) are doing structurally similar things, and the distinction between them is softer than the field treats it.** If SAE features causally control downstream behavior—as recent Anthropic interpretability work on functional emotions suggests they do—then they could in principle serve as routing signals for selectively activating shared-parameter computation, rather than just being passive readouts.

The end-state of the project (the "north star," not the first experiment) would be: a small architectural module that uses discovered SAE features to route computation through a shared parameter pool, trained to strengthen connections between feature clusters that need to cooperate for specific tasks. This is closer to how brains do parallel-subsystem cooperation than standard transformer architectures, and it's underexplored partly because the field treats interpretability and architecture as separate disciplines.

We are nowhere near building that yet. The plan below is the staircase to it.

## The staircase

- **v1 (this document's main focus):** Exploration. Characterize how Gemma 3 1B represents content across languages and cultures. Find out what's actually separable in feature space before designing any intervention. Build the tooling.
- **v2:** Same-modality composition test. Pick a task that requires combining two distinct capabilities the v1 exploration confirmed are separable. Find the relevant features. Intervene on them. Measure whether intervention predictably changes performance on the composed task.
- **v3:** Cross-modal version of the same test, using Gemma 3 4B (multimodal). The architectural separation between vision and text features is guaranteed, which makes this the cleanest test of the routing hypothesis. Will likely require training our own SAEs for the vision pathway since Gemma Scope 2 is text-only — out of scope for now but worth designing v1 tooling to be modality-agnostic so we don't have to refactor later.
- **v4:** Train a small additional module that does explicit routing based on discovered features, rather than just steering existing ones. This is the "real" version of the hypothesis and is many months away.

**Critical principle:** Each step exists to teach us how to use the tools and de-risk the next step. Don't skip ahead. Most toy experiments die because the researcher got interested in something tangential and wandered off; having the staircase mapped keeps us honest about what each piece is for.

## v1: detailed plan

### What we are NOT doing in v1

We are not doing interventions yet. We are not testing the routing hypothesis yet. We are not training anything. We are exploring what's already in the model so that v2 has a defensible foundation.

### What v1 IS

A characterization of how Gemma 3 1B (instruction-tuned) represents semantically equivalent content across English, Spanish, and French, and how it represents culturally-weighted content presented in different languages. The output of v1 is *understanding*, plus a working notebook that loads the model and the relevant SAEs and runs activation analyses on a corpus.

### The corpus

This is the part that depends on the user, specifically. Their trilingual fluency is a real experimental asset because:
- They can write idiomatic parallel texts rather than relying on machine translation, which introduces translationese artifacts
- They can spot-check whether features that fire on Spanish content actually correspond to "Spanish-ness" in a culturally meaningful way
- They can pick idioms that don't cleanly back-translate, which are often the most linguistically revealing probes

**Target corpus shape:**
- ~30-40 parallel triples (same idea expressed in English, Spanish, French)
- Texts should be 50-150 tokens each — short enough that activations are clean, long enough that there's something to analyze
- Idiomatic in each language, NOT word-for-word translations
- Mix of statement texts and question texts (features may differ between declarative and interrogative framings)
- Topics arranged on a gradient of cultural weight:
  - **Neutral:** basic physics, geography, generic human activities
  - **Moderate:** food, literature, common traditions
  - **Strong:** bullfighting, regional customs, language-specific humor
- Plus ~10-15 "cross-cultural" items: topics culturally tied to one language but text written in *another* language (e.g. an English text about French cheese aging, a French text about flamenco). These are critical for the most interesting question — see below.

The user will draft this. The implementing assistant should not generate the corpus.

### The diagnostic questions v1 should answer

1. **Layer-wise language separability.** At each layer of Gemma 3 1B, how similar are the residual stream representations across languages for the same semantic content? Expected pattern from the multilingual-models literature: early layers language-specific, middle layers converging toward language-agnostic concepts, late layers diverging again for output. We need to verify this pattern actually holds for a 1B model — if it doesn't, that itself changes what experiments make sense.

2. **Which SAE features fire differently across languages for the same content?** Use Gemma Scope 2 SAEs at multiple layers (suggest one early, one middle, one late). For each parallel triple, identify features that activate on one language but not the others. These are candidate "language identity features."

3. **Which features fire on the same content regardless of language?** These are candidate "language-agnostic concept features." This is what we'd want to use as the substrate for cross-lingual operations later.

4. **THE KEY QUESTION: are language and culture separable in feature space, or entangled?** When the user runs a culturally-Spanish topic written in English through the model, do features that normally only fire on Spanish-language text activate? Or are there separate features for "Spanish language" and "Spanish culture" with no crosstalk? The answer determines what v2 looks like:
   - If language and culture are **separate**, v2 is a cross-lingual factual recall task: ask in language A about an entity associated with language B, and test whether strengthening the connection between content features and target-language-output features improves performance.
   - If language and culture are **entangled**, the cross-lingual approach won't work cleanly and we need a different substrate for v2. Options to consider in that case include: entity-property composition tasks (named entity recognition + associative recall), or — if we want to escalate — moving to v3 multimodal earlier than planned because the architectural separation there is guaranteed.

**Both outcomes are interesting research findings.** There are no null results in v1, only unexpected ones. Frame the writeup that way.

## Tooling and environment

### Models

- **Primary:** `google/gemma-3-1b-it` (instruction-tuned, ~1B parameters, runs on free Colab T4 or any consumer GPU with 8GB+ VRAM)
- **Backup if 1B is too small for the effects we want to see:** `google/gemma-3-4b-it` (needs ~16GB VRAM)
- Use the instruction-tuned (`-it`) variant, NOT the base (`-pt`) variant. Base models wander instead of trying to answer prompts, which makes any task-based evaluation harder than it needs to be.

### SAEs

- **Gemma Scope 2** collection on Hugging Face: https://huggingface.co/collections/google/gemma-scope-2
- Specifically `google/gemma-scope-2-1b-it` for our primary model
- These are pretrained SAEs on the residual stream at multiple layers — exactly what we need
- Start with residual-stream SAEs (not MLP-output SAEs); residual stream captures the model's "working representation" at that depth and is the right starting point

### Libraries

- **SAELens** (`pip install sae-lens`) — standard library for loading pretrained SAEs including Gemma Scope; has tutorials that do almost exactly what we want for v1 and can be bent toward our specific questions
  - GitHub: https://github.com/jbloomAus/SAELens
  - The tutorials are a much better starting point than writing from scratch
- **TransformerLens** — underlying library for hooking into model internals; SAELens uses it
- **transformers** and **torch** — obviously
- **numpy**, **pandas**, **matplotlib** or **plotly** for the activation analysis and visualizations

### Hardware

A single consumer GPU (8GB+ for 1B, 16GB+ for 4B) is sufficient for v1 and v2. Free Colab T4 works for the 1B model. No frontier compute needed at any point in the staircase until possibly v4.

## v1 notebook skeleton

This is a structural sketch, not runnable code. The implementing assistant should fill this in using actual SAELens APIs (which the assistant should verify by reading current SAELens documentation rather than guessing — APIs change).

```python
# 1. Setup
# - Load Gemma 3 1B IT via transformers / TransformerLens
# - Load Gemma Scope 2 SAEs for several layers (early/middle/late) via SAELens
# - Verify everything loads and a simple forward pass works

# 2. Load the corpus
# - Read the user's parallel triples from a structured file (JSON or YAML)
# - Each item should have: id, topic, cultural_weight (neutral/moderate/strong),
#   text_en, text_es, text_fr, and optionally a "cultural_tag" field indicating
#   which culture (if any) the topic is associated with
# - Cross-cultural items should be a separate list with: id, topic, cultural_tag,
#   language_of_text, text

# 3. Run the corpus through the model and capture activations
# - For each text, run a forward pass and capture:
#   - Residual stream activations at each chosen layer
#   - SAE feature activations at each chosen layer
# - Store results in a structure that lets you index by (item_id, language, layer)

# 4. Analysis 1: cross-lingual representation similarity per layer
# - For each parallel triple and each layer, compute cosine similarity between
#   the (mean-pooled or last-token) residual stream representations of the three
#   language versions
# - Plot similarity vs layer depth — expecting a curve that rises in middle layers
#   and falls in late layers if the standard multilingual story holds for Gemma 3 1B

# 5. Analysis 2: language-specific vs language-agnostic features
# - For each layer's SAE, identify features that consistently activate on one
#   language but not others (language-specific) vs features that activate on
#   all three (language-agnostic)
# - Rank by consistency, not just magnitude — a feature that fires on every
#   Spanish text and almost no English/French text is more interesting than one
#   that fires strongly on a few Spanish texts

# 6. Analysis 3 (THE KEY ONE): culture-language entanglement
# - Take features identified as "Spanish-specific" in step 5
# - Run the cross-cultural items through the model (e.g., English text about
#   bullfighting)
# - Check whether the Spanish-specific features activate on the English-text-
#   about-Spanish-culture items
# - If yes → language and culture are entangled in feature space
# - If no → they're separable, and we can use language features as routing signals
#   in v2 independent of content

# 7. Write up findings
# - Even if results are messy, document what was found
# - Note any features that surprised the user when interpreted via their
#   trilingual knowledge — qualitative interpretation matters here
# - Decide which v2 path to pursue based on the entanglement finding
```

## Practical warnings

Things that will probably go wrong, based on general experience with this kind of work:

1. **SAE features are messier than papers make them look.** You will find features that seem to correspond to your target but also fire on surprising unrelated things, and features that should be there but are polysemantic or fire weakly. This is normal. Don't tune the experiment until the messiness disappears — interpret it.

2. **Don't trust mean activations alone.** Look at which specific examples drive a feature's activation. A feature that "fires on Spanish text" might actually be firing on a single common Spanish word that happens to appear in many examples. The user's language fluency is exactly the right tool to catch this.

3. **Tokenization differences across languages will create artifacts.** Spanish and French use accented characters that may tokenize differently than English equivalents. Some apparent "language features" might be tokenization features in disguise. Worth noting when you find suspicious features.

4. **Last-token vs mean-pooled representations give different answers.** For short texts, last-token is usually fine. For longer texts, consider both. The choice affects analysis 1 in particular.

5. **Choose layers thoughtfully.** Gemma 3 1B has ~26 layers. Picking "early/middle/late" should mean something like layers 4, 13, 22 — not 1, 13, 25, because the very first and last layers are doing token-level work that's not what we care about. Verify against what Gemma Scope 2 actually provides (some layers may not have SAEs released).

6. **Instruction-tuned model behavior on bare text.** The user's corpus will probably be plain texts, not chat-formatted prompts. The instruction-tuned model may behave slightly differently on bare text than on `<start_of_turn>user...<end_of_turn>` formatted input. Try both formats and note differences. If this is a problem, we may need to wrap each text in a minimal chat template.

## What the user should bring back when they return

- A draft corpus file (JSON or YAML) with at least 10-15 parallel triples and a few cross-cultural items, even if rough — the corpus can grow but we need something to start
- Their target hardware (which GPU, how much VRAM) so we can finalize whether 1B is the right starting model
- Any specific topics or themes they want represented that I (Claude) didn't think of — their cultural intuition is the input I can't substitute for

## What the implementing assistant should do first

1. Verify the user has Python 3.10+, a working PyTorch install with GPU support, and enough disk space (~5GB for the model and SAEs combined)
2. Get them through a "hello world" of loading Gemma 3 1B IT and running a single forward pass before touching SAEs
3. Then a "hello world" of loading one Gemma Scope 2 SAE and capturing its feature activations on a single text
4. Only then start building the v1 notebook above
5. **Read current SAELens documentation directly** rather than relying on training-data knowledge of the API — it changes, and getting the API wrong here will waste hours

## Reference materials

- Gemma Scope 2 collection: https://huggingface.co/collections/google/gemma-scope-2
- SAELens: https://github.com/jbloomAus/SAELens
- The Anthropic functional emotions paper that inspired the SAE-routing connection: https://transformer-circuits.pub/2026/emotions/index.html
- Tao's wiki of AI contributions to math research, for context on what current models can do in collaborative research mode: https://github.com/teorth/erdosproblems/wiki/AI-contributions-to-Erd%C5%91s-problems

## A note on the spirit of this project

The user came to this from first principles, reasoning their way from "is hex quantization a thing?" through MoE design, SAE structure, brain analogies, and the interpretability/architecture distinction over the course of one conversation. They have good research taste and they want to learn by doing, not by being handed a finished pipeline. The implementing assistant should treat this as a collaborative research project where the user is the principal investigator and the assistant is a capable but non-autonomous research engineer. Explain what you're doing and why. Surface unexpected findings. When something doesn't work, debug it together rather than silently fixing it. The point of the project is partly the experiment and partly the user learning how this kind of work feels from the inside.

Good luck. This is a real research direction, scoped to be tractable for a small setup, and the v1 result will be informative no matter which way it goes.
