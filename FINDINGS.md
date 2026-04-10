# SAE-Guided Capability Routing: Experimental Findings

## Research Question

Can Sparse Autoencoder (SAE) features serve as routing signals for selectively activating shared-parameter computation in LLMs, analogous to how biological brains route information through parallel, cooperating subsystems?

This project tests the preconditions for that hypothesis through a staircase of progressively more ambitious experiments on Gemma 3 1B IT using Gemma Scope 2 SAEs.

## Setup

- **Model**: `google/gemma-3-1b-it` (26 layers, hidden size 1152)
- **SAEs**: Gemma Scope 2 (`google/gemma-scope-2-1b-it`), residual stream post, 16k width
- **Target layers**: 7 (early, ~27%), 13 (middle, ~50%), 22 (late, ~85%)
- **SAE releases**: `gemma-scope-2-1b-it-res` (layers 13/17/22, medium L0), `gemma-scope-2-1b-it-res-all` (all layers, small/big L0)
- **Pipeline**: HuggingFace transformers for inference + manual hooks for activation extraction, SAELens for SAE encode/decode. HookedSAETransformer works for small runs but OOMs on large corpora.

---

## v1: Cross-lingual Representation Exploration

**Goal**: Characterize how Gemma 3 1B represents semantically equivalent content across English, Spanish, and French. Determine whether language and culture are separable in feature space.

**Corpus**: 32 parallel triples (same idea in EN/ES/FR) across neutral, moderate, and strongly culturally-weighted topics, plus 17 cross-cultural items (e.g., English text about bullfighting). Cross-cultural items split into loanword (with Spanish/French vocabulary) and no-loanword (native vocabulary only) variants.

### Finding 1: The model's middle layers are strikingly language-agnostic

Cross-lingual cosine similarity of residual stream representations:

| Layer | Mean-pooled | Last-token |
|-------|-------------|------------|
| 7     | 0.9991      | 0.9973     |
| 13    | 0.9993      | 0.9975     |
| 22    | 0.9866      | 0.9672     |

The expected "early divergent, middle convergent, late divergent" pattern is barely visible because the model is already highly convergent by layer 7. Cultural weight (neutral vs strong) makes almost no difference. Last-token representations show slightly more divergence than mean-pooled, especially at layer 22 (0.967 vs 0.987), confirming that mean-pooling partially masks language-specific structure.

### Finding 2: SAE features are far more discriminating than cosine similarity

While raw residual cosine similarity is 0.999+ at layers 7-13, SAE feature decomposition reveals substantial language-specific structure:

| Layer | Language-specific features | Language-agnostic features |
|-------|--------------------------|---------------------------|
| 7     | 148                      | 262                       |
| 13    | 209                      | 1020                      |
| 22    | **936**                  | 914                       |

The jump from 209 to 936 language-specific features between layers 13 and 22 marks where the model's output-preparation machinery builds language-specific representations. This empirically demonstrates why SAE-based analysis is more informative than raw similarity metrics for this research question.

**Key features discovered**:
- Feature 857 (layer 22): fires on 100% of Spanish texts, 6% of others, mean activation 383. The cleanest language-specific feature found.
- Feature 1207 (layer 22): fires on 100% of French texts, 14% of others, mean activation 1782. Very strong but high baseline activation.
- Feature 3201 (layer 22): fires on 100% of French texts, 6% of others, mean activation 192.

### Finding 3: Language and culture are partially entangled, and loanwords amplify it

Overall cross-cultural/baseline ratio: **2.46x** (entangled).

The loanword experiment (design suggestion from the planning conversation) produced a clean result:

| Condition | Avg cultural feature fire rate | N |
|-----------|-------------------------------|---|
| With loanwords | 38.5% | 36 |
| Without loanwords | 21.0% | 15 |

**Loanwords amplify entanglement by +17.5 percentage points**, but the 21% fire rate on no-loanword items (vs ~6-8% baseline) shows genuine conceptual-cultural entanglement exists beyond lexical routing.

### Finding 4: Layer depth distinguishes lexical from conceptual entanglement

The most informative result from v1. For the bullfighting-in-English cross-cultural item:

| Layer | With loanwords (corrida, matador) | Without loanwords (fighter, ring) |
|-------|-----------------------------------|-----------------------------------|
| 7     | 6.92x                             | 1.54x                             |
| 13    | 3.64x                             | 1.82x                             |
| 22    | 10.00x                            | **5.00x**                         |

The loanword version is entangled at every layer (lexical routing -- the tokens themselves trigger Spanish features). The no-loanword version shows minimal entanglement at early layers but strong entanglement at layer 22 -- the signature of conceptual rather than lexical processing. Early layers see English tokens and shrug; late layers reconstruct the cultural concept and the language binding activates.

---

## v2: Feature Steering (Causal Control Test)

**Goal**: Test the most basic precondition for the routing hypothesis -- do SAE features behave as causal control signals, or are they just correlates?

**Method**: Clamp Feature 857 (Spanish detector) to various multiples of its natural activation during autoregressive generation on neutral English prompts. Observe whether output shifts toward Spanish.

### Finding 5: Features ARE causal control signals

Feature 857 produces a clean, graded language shift:

| Steering strength | Effect on "Water is made of" |
|-------------------|-------------------------------|
| 0x (baseline)     | "hydrogen and oxygen. The water cycle is a continuous process..." |
| 0.5x              | No visible change |
| 1.0x              | "hydrogen and oxygen atoms. The hydrogen atoms are very light..." (subtle content shift) |
| 2.0x              | "hydrogen and oxygen atoms. The hydrogen atoms are very light and can easily escape..." |
| **5.0x**          | **"H2O. H2O es un compuesto de dos atomos de hidrogeno y un atomo de oxigeno..."** |

The transition is smooth -- not a binary switch but a continuous dial from English to code-mixed to full Spanish. This is exactly the behavior needed for a routing signal.

**Feature 1207 (French)** also steers causally but with a caveat: at 2x, it produces coherent French ("de leur structure osseuse, leur large surface de peau..."). At 5x, it degenerates to repetitive output ("plus plus plus..."). This indicates routing must operate within a feature's natural dynamic range.

**Feature 3201 (French)** barely steers at all, even at 5x. Not all language-specific features are equally useful as controls -- some are readouts rather than drivers.

### Verdict

**The routing hypothesis precondition is met.** At least some SAE features in Gemma 3 1B behave as genuine causal control signals that can smoothly redirect model behavior. Feature 857 specifically acts as a clean, graded language routing switch. This validates pursuing the full routing architecture in later experiments.

---

## Neuronpedia Cross-Validation

We looked up our key features on [Neuronpedia](https://neuronpedia.org) (which hosts Gemma Scope 2 feature dashboards with automated interpretability labels) to validate our feature identification. The API endpoint is `https://www.neuronpedia.org/api/feature/{model}/{source}/{index}`.

| Feature | Our label | Neuronpedia label | Match? |
|---------|-----------|-------------------|--------|
| [857 @ L22](https://www.neuronpedia.org/gemma-3-1b-it/22-gemmascope-2-res-16k/857) | Spanish detector | "the phrase aquí te" | Partial -- narrower than our label but confirms Spanish activation |
| [1207 @ L22](https://www.neuronpedia.org/gemma-3-1b-it/22-gemmascope-2-res-16k/1207) | French detector | **"French common words"** | Confirmed |
| [3201 @ L22](https://www.neuronpedia.org/gemma-3-1b-it/22-gemmascope-2-res-16k/3201) | French detector | **"French articles followed by nouns"** | Confirmed (more specific -- French syntax) |
| 576 @ L22 | French-specific | "german words followed by common german words" | **Wrong** -- actually German, not French |
| 9293 @ L7 | French early layer | "software development and validation" | **Wrong** -- topical, not linguistic |
| 10036 @ L7 | French early layer | "lemon juice, zest, wedges" | **Wrong** -- food/cooking feature |

**Key takeaway**: Late-layer features (L22) are genuinely linguistic and validate well. Early-layer features (L7) that we identified as "language-specific" were actually topical features that correlated with French cooking/tech texts in our corpus. This is the polysemanticity warning from the handoff doc in action -- and it means v2's steering experiment was right to focus on L22 features, which are the clean ones.

---

## Experimental Staircase (Updated)

- [x] **v1**: Exploration -- characterize cross-lingual representations, find separable features
- [x] **v2**: Steering -- verify features are causal controls, not just correlates
- [ ] **v3**: Composition -- test whether steering multiple features simultaneously produces predictable combined behavior (e.g., "Spanish language + formal register")
- [ ] **v4**: Cross-modal routing -- extend to Gemma 3 4B multimodal, where vision/text separation is architecturally guaranteed
- [ ] **v5**: Train routing module -- build a small architectural module that uses discovered features to route computation through shared parameters

---

## Methodology Notes

- **Mean-pooling vs last-token**: Mean-pooling slightly inflates cross-lingual similarity. Use last-token for sharper signal, especially at late layers.
- **SAE reconstruction quality**: cosine similarity 0.998+ at layers 7 and 13, 0.987 at layer 22. Sufficient for analysis.
- **HookedSAETransformer vs manual hooks**: HookedSAETransformer works for small batches but OOMs on 100+ item corpora due to weight conversion overhead. Use HuggingFace transformers + manual hooks for activation collection.
- **Steering approach**: Add the steering delta (steered reconstruction minus original reconstruction) to the original residual, rather than replacing it. This preserves information not captured by the SAE.
- **Feature activation ranges matter**: Feature 1207 (mean_act=1782) degenerates at 5x because the magnitude overwhelms the model. Feature 857 (mean_act=383) steers cleanly at 5x. Routing signals need to operate within the feature's natural dynamic range.

## Key Files

| File | Purpose |
|------|---------|
| `sae_routing_experiment_handoff.md` | Original research plan from design conversation |
| `corpus.json` | 32 parallel triples + 17 cross-cultural items (EN/ES/FR, with proper accents) |
| `v1_exploration.py` | Full v1 analysis: similarity, feature identification, entanglement, loanword comparison, per-token attribution |
| `v2_steering.py` | v2 steering experiment: clamp features during generation |
| `vast_gpu.sh` | Vast.ai instance management wrapper |
| `bootstrap_remote.sh` | One-command remote instance setup |
| `hw*.py` | Hello-world validation scripts (model loading, SAE loading, pipeline tests) |
| `results/` | Plots and full output logs |
