# Hand-over to LaTeX writer — 2026-05-22 (post-reviewer-audit cycle)

Single self-contained brief covering the changes since your last
pull. Two sections: **immediate patches** (must apply before next
draft) and **table/prose updates** (use the corrected numbers).

Everything is on `main` as of commit `de29701`. The two source-of-
truth docs in `paper/` are:
- `V3_CHANGES.md` — narrative changelog
- `LATEX_WRITER_HANDOFF_v3.md` — operational punch-list (with the
  K=23 numbers added)

Cross-reference key data files in `results/`:
- `gap_decomposition.{json,md}` — single source of truth for the
  adjacent-miscalibration tally
- `option_order_shuffle_all_models.md` — cross-model option-order
  summary (K=23 headline + K=3 inset + paragraph for §4.2)
- `option_order_shuffle_exhaustive_summary.md` — full K=3-vs-K=23
  comparison with case-clustered 95% CIs
- `decision_token_logit_attribution_normalized.{json,md}` —
  normalized logit-attribution table for Table 6

---

## 1. Immediate patches to `latex/v2_short/main.tex`

### 1a. Adjacent-miscalibration arithmetic — TWO lines

**Line 711–712 currently:**
```
Across all three models, $19/20$ \texttt{NL\_only\_right} cases
and $15/20$ \texttt{NF\_only\_right} cases are adjacent
miscalibrations
```
**Replace with:**
```
Across all three models, $14/15$ \texttt{NL\_only\_right} cases
and $15/20$ \texttt{NF\_only\_right} cases are adjacent
miscalibrations ($29/35$ combined)
```

**Line 1349–1350 currently:**
```
appear at lower magnitude with $19/20$ \texttt{NL\_only\_right} +
$\texttt{NF\_only\_right}$ cases being adjacent miscalibrations
```
**Replace with:**
```
appear at lower magnitude with $29/35$ \texttt{NL\_only\_right} $+$
\texttt{NF\_only\_right} cases being adjacent miscalibrations
```

Source of truth: `results/gap_decomposition.json`. Per-model breakdown:
- 4B   NL_OR: 1 (1 adj)        NF_OR: 14 (14 adj)
- 12B  NL_OR: 6 (5 adj, 1 non) NF_OR: 0
- Qwen NL_OR: 8 (8 adj)        NF_OR: 6 (1 adj, 5 non)
- Total: NL_OR 15 (14 adj), NF_OR 20 (15 adj), combined 35 (29 adj)

Formal adjacency definition the paper should use: given predicted
letter `p` and gold letter set `G ⊆ {A,B,C,D}`, define
`d(p, G) = min_{g ∈ G} |i_p − i_g|` where `i_X` is the acuity
index (A=0, B=1, C=2, D=3). A case is **adjacent** iff
`d(p, G) = 1` for the wrong prediction in the NL_only_right
or NF_only_right stratum.

### 1b. Table 6 (logit attribution) — REPLACE with normalized table

The v2 manuscript's Table 6 used raw values (2627 at 4B, 198.83 at
12B). Replace with normalized fractions. **Recommended Table 6
content:**

| Model | Category | abs-fraction mean (5–95% CI) | margin-share mean (5–95% CI) |
|---|---|---|---|
| 4B  | medical (v3, 3 features) | 0.0% (0.0%, 0.0%)  | 0.0% (0.0%, 0.0%)  |
| 4B  | scaffold-proxy (top 30)  | 0.1% (0.0%, 0.4%)  | 0.1% (−0.4%, 1.0%) |
| 4B  | other (~47 features)     | **99.9%** (99.6%, 100.0%) | **99.9%** (99.0%, 100.4%) |
| 12B | medical (v3, 3 features) | 0.0% (0.0%, 0.0%)  | 0.0% (0.0%, 0.0%)  |
| 12B | scaffold-proxy (top 30)  | **50.3%** (45.6%, 54.9%) | 26.5% (−27.4%, 63.6%) |
| 12B | other (~47 features)     | 49.7% (45.1%, 54.4%) | **73.5%** (36.4%, 127.4%) |

**Recommended caption:**

> Table 6: Decision-token logit attribution at the NL last-prompt
> position for 4B (L29) and 12B (L31), normalized to make
> categorical comparisons interpretable. *abs-fraction* = category
> K's net absolute linear effect on the A/B/C/D logits divided by
> the total absolute linear effect across all categories.
> *margin-share* = category K's signed contribution to the
> predicted-vs-runner-up logit margin. Linear logit-lens projection
> `c[i, L] = act_i · W_dec[i] · W_unembed[:, L_token]`; ignores
> the final LayerNorm, the transformer layers between the SAE
> layer and the unembedding, and SAE reconstruction error.
> Magnitudes approximate; categorical comparisons directionally
> informative. Causal attribution would require per-feature
> ablation forward passes (future work). At both scales, medical
> features contribute essentially zero to the linear logit effect
> at the decision token, consistent with the per-case finding
> that they have zero activation at that position in 60/60 cases.

Source: `results/decision_token_logit_attribution_normalized.{json,md}`.
The raw values (2627 / 198.83) stay in the Appendix table for
researchers who want them.

### 1c. §4.2 option-order shuffle prose — REPLACE with K=23 numbers + 4B honesty correction

The §4.2 paragraph currently references the K=3 result and may
include "shuffled NL accuracy equals NF accuracy *to the case*" at
4B. **Replace with the following self-contained paragraph (use
verbatim or adapt):**

> An option-order randomization experiment (60 cases × 23
> non-identity permutations of the letter→content mapping in the
> forced-letter scaffold) tests whether the forced-letter accuracy
> depends on letter position or on letter content. Across all
> three models, the picked-letter is at or below chance under
> shuffles (case-clustered 95% CI excludes 25% at 4B and 12B),
> while the picked-content is far above chance (lower CI bound
> ≥56%): no position bias, strong content prior. At 4B,
> randomising the labels brings the forced-letter accuracy from
> 55.0% to 69.8% (95% CI [60.7%, 78.3%]), statistically
> indistinguishable from NF accuracy 71.7% — the entire NL→NF
> format penalty at 4B can be attributed to the canonical A-B-C-D
> letter-binding interacting with the model's content prior. At
> 12B and Qwen, the canonical mapping is approximately neutral or
> mildly helpful for accuracy, but shuffled forced-letter still
> beats free-text by ≈5–7 pp — at scale, free-text mode has its
> own accuracy penalty (the adjacent-miscalibration discussed
> above) that is independent of letter-binding. Under exhaustive
> shuffles, 4B emits the letter mapped to "Go to the ER now"
> content in only 2/1380 = 0.14% of shuffles, compared to 9.1%
> at 12B and 7.0% at Qwen — a robust capability-scaling signal
> at the highest acuity level.

**Things to change in any earlier draft of this paragraph:**

- Any claim that "shuffled NL accuracy equals NF accuracy *to
  the case*" at 4B — REMOVE. The K=3 estimate happened to be
  71.7%=71.7% exactly; the K=23 estimate is 69.8% vs 71.7% with
  CI containing NF. Corrected claim is "statistically
  indistinguishable."
- Any claim that "4B never picks ER and that's a small-sample
  artifact" — REMOVE if present. The K=23 confirms 4B picks ER
  content only 2/1380 of the time, robust.
- Same-letter % is now significantly below chance (not at chance)
  thanks to K=23 + clustered CIs.

### 1d. Table 1 (§4.1) — confirm paired-test numbers + add Qwen column

If Table 1 doesn't already have these, please patch in:

| Model | n | NL acc | NF acc | NL−NF (pp) | 95% CI (paired bootstrap) | McNemar exact p |
|---|---|---|---|---|---|---|
| 4B   | 60 | 55.0% | 71.7% | **−16.7** | [−30.0, −3.3] | **0.031** |
| 12B  | 60 | 81.7% | 71.7% | **+10.0** | [+3.3, +18.3] | **0.031** |
| Qwen | 60 | 75.0% | 68.3% | +6.7 | [−6.7, +20.0] | 0.45 (ns; underpowered at n=60) |

Note Qwen p=0.45 — the gap is real-directional but not
statistically significant at n=60. The honest cross-family claim
remains **"suggestive cross-family consistency"** rather than
"cross-family confirmation."

Source: `results/paired_tests_and_confusion.json`.

---

## 2. New experiments since your last pull — please integrate where they fit

### 2a. Decision-token feature characterization (Concern 5 redux)

We did THREE complementary analyses of the NL decision-token state.
They should be presented together in §4.4 or §4.5 (the reviewer
recommended elevating this).

**(i) Logit attribution (normalized).** See Table 6 patch above.
Headline: medical features contribute 0% at both scales; at 4B
99.9% of linear effect is in "other" features; at 12B scaffold-
proxy and "other" each contribute ~50% abs, "other" dominates the
margin.

**(ii) Top-K active feature characterization.** For each case,
top-20 features by activation at the NL decision token vs at the
NF decision token. Jaccard overlap and peak-position
characterization:

| Model | NL∩NF top-20 (Jaccard) | NL-only features peaking in **B scaffold** | NF-only features peaking in **D vignette** | v3 medical in NL or NF top-20 |
|---|---|---|---|---|
| 4B  | **0.000** | **87.0%** | 27.8% | 0/60 (both) |
| 12B | **0.001** | **88.3%** | 8.9% | 0/60 (both) |
| Qwen | 0.324 | **94.7%** | 10.4% | 0/60 (both) |

Source: `results/decision_token_top_features_{4b,12b,qwen}.json`
+ `_summary.md`.

Three findings:
1. At both Gemma scales, NL and NF use essentially disjoint top-20
   feature sets at the decision token (Jaccard ≈ 0).
2. 87–95% of NL-only top-20 features peak on B's scaffold tokens
   across all three models — direct confirmation of "scaffold-
   primary at NL."
3. v3 medical features are 0/60 in NL top-20 AND 0/60 in NF
   top-20 at every model.

**(iii) Option-order shuffle.** See §1c above.

### 2b. Tokenization sanity check (§3.1 wording)

We verified the byte-identical-prefix assumption explicitly:

- **Gemma 4B and 12B: 60/60 cases byte-identical** at vignette
  positions between NL and NF.
- **Qwen: 60/60 vignette texts re-tokenize identically when
  isolated**, but Qwen's BPE merges the trailing punctuation
  differently when followed by the scaffold (NL) vs the
  end-of-turn marker (NF), shifting the divergence one token
  earlier than the vignette end. ~99.6% of vignette positions
  remain byte-identical.

Suggested §3.1 phrasing:

> "The NL and NF prompts for a given case share the patient
> vignette text verbatim and differ only by an appended forced-
> letter scaffold in NL. Under causal masking, hidden states at
> vignette token positions cannot depend on later scaffold tokens,
> so SAE feature activations at vignette positions are byte-
> identical between NL and NF at the Gemma scales. At Qwen3-8B the
> BPE merges the trailing punctuation differently depending on
> what follows the vignette, which shifts the divergence one token
> earlier than the vignette end; the remaining ~99.6% of vignette
> tokens are byte-identical at Qwen too."

Source: `results/verify_byte_identical_prefix.{json,md}`.

### 2c. K-sweep Appendix A1B — REPLACE mean-pool with max-pool

The current Appendix A1B reports mean-pool sMAPE (medical 0.188,
random 0.222 at K=3 for 4B). Replace with max-pool (matching the
main text):

**4B L29:**

| K | medical sMAPE (mean, 95% CI) | random sMAPE (mean, 95% CI) | Δ_paired (95% CI) | sig? |
|---|---|---|---|---|
| 3  | 0.0063 [0.003, 0.012] | 0.0312 [0.021, 0.044] | −0.0249 [−0.038, −0.013] | ✓ |
| 5  | 0.0115 [0.003, 0.027] | 0.2285 [0.182, 0.278] | −0.2170 [−0.268, −0.168] | ✓ |
| 10 | 0.1329 [0.097, 0.169] | 0.2601 [0.217, 0.306] | −0.1272 [−0.163, −0.094] | ✓ |
| 20 | 0.1531 [0.131, 0.176] | 0.2139 [0.173, 0.256] | −0.0608 [−0.089, −0.034] | ✓ |

**12B L31:**

| K | medical sMAPE | random sMAPE | Δ_paired | sig? |
|---|---|---|---|---|
| 3  | 0.0058 [0.003, 0.010] | 0.7414 [0.648, 0.833] | −0.7356 [−0.825, −0.643] | ✓ |
| 5  | 0.0053 [0.004, 0.008] | 0.4549 [0.397, 0.512] | −0.4496 [−0.506, −0.392] | ✓ |
| 10 | 0.0389 [0.023, 0.059] | 0.5920 [0.553, 0.639] | −0.5531 [−0.600, −0.507] | ✓ |
| 20 | 0.0871 [0.071, 0.105] | 0.4120 [0.382, 0.446] | −0.3249 [−0.363, −0.291] | ✓ |

Add a caption note: "All sMAPE values under max-pool aggregation,
matching the main §4.2 table; the original mean-pool K-sweep is
retained at `results/phase1b_sensitivity_4b_L29.json` /
`_12b_L31.json` for reference."

Source: `results/phase1b_sensitivity_maxpool.{json,md}`.

### 2d. Triage direction + per-acuity (§4.1 or §6 addition)

Reviewer-requested supplementary stats:

**Triage direction (forced-letter mode under-triages clinically):**

| Model | NL: under | NL: correct | NL: over | NF: under | NF: correct | NF: over |
|---|---|---|---|---|---|---|
| 4B   | **20** | 33 | 7 | 5  | 43 | 8 |
| 12B  | 4  | 49 | 7 | 8  | 43 | 9 |
| Qwen | **12** | 45 | 3 | 8  | 37 | 6 (+9 no-commit) |

→ NL mode at 4B and Qwen systematically under-triages (the
clinically dangerous direction). NF mode is more balanced. Worth
a sentence in §4.1 ("clinical safety implication") and §6
("free-text recovers some safety properties that forced-letter
scoring obscures").

**Per-acuity (gold A/B/C/D bucket):** A-bucket is the weakest at
every model × format. At 4B: NF correctly handles only 0/8 A-cases.
Worth flagging in §6 limitations.

**4B singleton-D failure (reviewer-flagged):** 4B predicts D = 0/9
on D-only-gold cases in NL forced-letter mode (and only 1/9 in NF).
Clinically concerning; flag in §6.

Source: `results/paired_tests_and_confusion.{json,md}`.

### 2e. Random-baseline magnitude-matching (Concern 4b)

The original fixed-seed random baseline understated the medical-vs-
random gap by including zero-firing features whose sMAPE is
artificially zero by denominator floor. 1000 magnitude-matched
random pool resamples:

| Model | medical sMAPE | random (mag-matched) sMAPE | perm-p one-sided |
|---|---|---|---|
| 4B   | 0.0040 | 0.2757 [0.138, 0.429] | **< 0.001** |
| 12B  | 0.0027 | 0.1232 [0.072, 0.171] | **< 0.001** |
| Qwen | 0.0261 | 0.1281 [0.048, 0.226] | **0.012** |

This is the proper random baseline for the main §4.2 table; the
medical-vs-random gap survives at all three models.

Source: `results/phase1b_random_pool_resample_{4b,12b,qwen}.json`
+ `_summary.json`.

---

## 3. Title / framing walk-backs

(Concern 2 and 6 from the reviewer audit.)

- Replace "deployable monitor" → "candidate readout, requires
  prospective clinical validation" throughout.
- Replace "clinical reasoning preserved" → "medical-domain
  representation preserved on the shared clinical prefix and
  silent in the letter-decision pathway."
- If the title contains "clinical reasoning preserved" or
  "deployable monitor", consider one of:
  - "Format-Robust Medical-Content Representation in LLM Triage:
    A Cross-Family SAE Analysis"
  - "Apparent Triage Failures as Output-Mapping Artefacts:
    Mechanistic Evidence from SAE Analysis at Three Model Scales"
  - "What Gets Lost When LLMs Triage With Letters: Medical-Content
    Representation Survives Format Changes Across Three Models"
- "Cross-family confirmation" → "suggestive cross-family
  consistency" (Qwen NL−NF gap is not statistically significant
  at n=60).

---

## 4. Bottom line

After this batch, the v3 reviewer-correction empirical loop is
**closed in all directions a reviewer can push**. The remaining
work is yours: §4.2 rewrite, walk-back framing pass, Table 6
patch, the two main.tex arithmetic patches, and the title
softening. Everything else (Bucket A statistics, K=23 exhaustive
shuffle, normalized logit attribution, clustered bootstrap, gap
decomposition, masked invariance, byte-identical-prefix
verification) is now in `results/` with reproducible scripts in
`paper/scripts/`.

Compute spend this cycle: ~$8–9 GPU + ~$3 LLM-judge API.

If you find a number that doesn't reconcile to the source-of-truth
JSON, ping back. The arithmetic-bug fix was prompted by exactly
that kind of careful audit, and we'd rather catch one more now
than have a reviewer catch it later.
