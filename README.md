# SAE-Guided Cross-Lingual Knowledge Transfer in LLMs

Research project investigating whether Sparse Autoencoder (SAE) features can serve as routing signals for transferring capability across languages in large language models.

**Status**: **Project concluded, null result.** Quantitative rescue claims from v2-medical are retracted. See [FINDINGS.md](FINDINGS.md) for full retraction, controlled replications, and final validated findings.

## TL;DR

We started with the hypothesis that amplifying language-specific SAE features at inference time could rescue cross-lingual performance gaps (e.g., improve Arabic medical QA by steering toward English features). Initial experiments (v2-medical) showed apparent rescue effects of 3-7%.

External review surfaced methodological issues:
1. Our "English" baseline was actually Arabic (MMMLU `default` config is non-English)
2. Rescue was measured on a small subset, not the full benchmark
3. No random-feature control
4. "Single feature" claim was actually multi-feature

v3 addressed all of these with controlled replications. **Under proper evaluation, no rescue effect exists above the random-feature noise floor, in any of five tested configurations.**

What remains real:

- **v1** — Cross-lingual representation characterization using our own trilingual corpus (unaffected by the MMMLU issue)
- **v2 steering on 1B** — SAE features causally control output language (Feature 857 at layer 22 → Spanish text on neutral English prompts, graded with strength)
- **v3 domain feature identification** — Language-agnostic medical content features exist in Gemma 3 4B at layer 29 (features 893, 12570, 12845). They fire on medical content across EN/ES/FR (MCQ and free-form), zero on non-medical content, and more weakly on Arabic and Yoruba medical
- **v3 null under controlled evaluation** — Neither amplifying nor ablating these features at layer 29 changes medical MCQ accuracy above the random-feature noise floor, across 5 tested rescue configurations

Precise scope of the null: we ruled out simple single-layer, single-feature amplification and single-layer, single-feature ablation at layer 29. Multi-layer, multi-feature, coordinated, or task-conditional interventions remain untested. The routing hypothesis is not refuted at its strongest — just the simplest version of it.

## Repo Structure

```
.
├── README.md                              ← You are here
├── FINDINGS.md                            ← Full empirical record (main artifact)
├── sae_routing_experiment_handoff.md      ← Original research plan
│
├── corpus.json                            ← Parallel EN/ES/FR corpus for v1
├── corpus_template.json                   ← Template for expanding the corpus
│
├── hw1_load_model.py                      ← Hello-world: model loading
├── hw2_sae_bridge.py                      ← Hello-world: SAELens bridge test
├── hw2b_fallback.py                       ← Hello-world: manual hooks fallback
├── hw3_load_sae.py                        ← Hello-world: SAE loading
├── hw4_end_to_end.py                      ← Hello-world: end-to-end test
├── hw5_multilingual.py                    ← Hello-world: multilingual smoke test
│
├── v1_exploration.py                      ← Cross-lingual representation analysis
├── v2_steering.py                         ← Causal control test (1B model)
├── v2_medical_pilot.py                    ← Medical QA baseline (1B + 4B)
├── v2_medical_rescue.py                   ← First rescue: generic ES features
├── v2_medical_rescue_v2.py                ← Targeted rescue: ES-medical features
├── v2_generalization.py                   ← 5 domains × 2 languages
├── v2_flip_distant_combined.py            ← Coding + Chinese + combined features
│
├── vast_gpu.sh                            ← Vast.ai instance management
├── bootstrap_remote.sh                    ← One-command remote setup
│
├── results/                               ← All experiment outputs
│   ├── analysis1_similarity.png           ← v1 similarity plot
│   ├── v1_full_output.txt                 ← v1 full stdout
│   ├── v2_steering_output.txt             ← v2 generation examples (1B, validated)
│   ├── v2_medical_*.json                  ← v2 medical runs — RETRACTED (see FINDINGS.md)
│   ├── v2_generalization.json             ← v2 generalization — RETRACTED
│   ├── v2_flip_distant_combined.json      ← v2 extended — RETRACTED
│   ├── v3_replication.json                ← Controlled replication (null)
│   ├── v3_lowresource_rescue.json         ← Weak-language rescue (null)
│   ├── v3_domain_rescue.json              ← Domain feature identification + rescue (null)
│   └── v3_feature_validation_output.txt   ← Feature validation (readouts, not drivers)
│
└── remote_cache/                          ← Large cached files (gitignored)
    └── cached_activations.pt              ← v1 activations (~1.3GB)
```

## Reproducing the Results

### Environment

All experiments ran on vast.ai GPUs with the base image `nvidia/cuda:12.4.0-runtime-ubuntu22.04`. Dependencies installed via:

```bash
pip install torch transformers datasets accelerate sae-lens
```

Model: `google/gemma-3-4b-it` (gated -- requires HuggingFace auth)
SAE: `google/gemma-scope-2-4b-it-res/layer_29_width_16k_l0_medium`

### Running an experiment

Most experiments follow the same pattern:

1. Launch a vast.ai instance (22GB+ VRAM for 4B model):
   ```bash
   # Via /gpu slash command (see github.com/dafraile/claude-gpu-skill)
   /gpu launch medium
   ```

2. Bootstrap the instance:
   ```bash
   bash bootstrap_remote.sh <port> <ip>
   ```

3. Run the desired script:
   ```bash
   ssh -i ~/.ssh/vastai -p <port> root@<ip>
   cd /root
   python3 v2_medical_pilot.py  # or any other v2_*.py script
   ```

4. Pull results back:
   ```bash
   scp -i ~/.ssh/vastai -P <port> root@<ip>:/root/results/*.json ./results/
   ```

5. Destroy the instance when done:
   ```bash
   /gpu destroy <instance_id>
   ```

### Recommended experiment order (if reproducing from scratch)

The v2-medical scripts are retained for historical completeness but their quantitative claims are retracted — see FINDINGS.md. For the validated results, run:

1. `hw1_load_model.py` → `hw5_multilingual.py` — verify environment
2. `v1_exploration.py` — cross-lingual representation analysis on our trilingual corpus (validated)
3. `v2_steering.py` — causal language-output steering on 1B, Feature 857 → Spanish (validated)
4. `v3_replication.py` — controlled replication with real English from `cais/mmlu`, full-benchmark eval, random-feature control (null result, correctly scoped)
5. `v3_lowresource_rescue.py` — EN features → Arabic/Swahili/Yoruba medical (null)
6. `v3_domain_rescue.py` — language-agnostic medical feature identification + rescue attempts (features identified, rescue null)
7. `v3_feature_validation.py` — three-test validation: top tokens on free-form text, ablation, cross-format (features are cross-lingual cross-format medical readouts)

## Timing and Cost

Rough GPU cost estimates on vast.ai ($0.15-0.30/hr for 24GB GPUs):

| Experiment | Duration | Cost |
|------------|----------|------|
| Hello-worlds (all) | ~5 min | $0.01 |
| v1 exploration | ~15 min | $0.05 |
| v2 steering (1B) | ~10 min | $0.03 |
| v3 replication | ~30 min | $0.15 |
| v3 low-resource rescue | ~40 min | $0.20 |
| v3 domain rescue | ~45 min | $0.22 |
| v3 feature validation | ~30 min | $0.15 |
| **Total (validated pipeline)** | **~3 hours** | **~$0.80** |

## Key Findings (see [FINDINGS.md](FINDINGS.md) for details)

**Validated positive findings:**

1. **v1**: Cross-lingual representations in Gemma 3 1B show a depth signature — lexical/surface entanglement at shallow layers, conceptual entanglement at late layers
2. **v2 steering (1B)**: Feature 857 at layer 22 is a graded, continuous dial for output language. Clamping at 2x–5x produces smoothly controllable Spanish output from neutral English prompts
3. **v3 feature identification**: Language-agnostic, domain-selective medical features exist in Gemma 3 4B at layer 29 (features 893, 12570, 12845). They fire on medical content across EN/ES/FR in both MCQ and free-form text, fire weakly on Arabic medical, and fire zero on non-medical content. The six-condition contrastive discovery pipeline is reusable for other domains.

**Null results, precisely scoped:**

4. Single-feature layer-29 amplification of language or domain features does not rescue cross-lingual medical MCQ performance above the random-feature noise floor (five tested configurations, full benchmark, bootstrap 95% CIs)
5. Single-feature layer-29 ablation of the identified medical features produces exactly 0.00% change in accuracy across EN/ES/FR medical and non-medical conditions

**Retracted (see FINDINGS.md):**

- All v2-medical quantitative rescue claims (e.g., "3-7% rescue", "reversed multilingual gap", "universal across languages/domains") — invalidated by a dataset bug (MMMLU `default` is not English) and other methodology gaps. The controlled replication in v3 found no effect.

## Citation

If this work informs your research, please cite the repository:

```
@misc{fraile-navarro-2026-sae-cross-lingual,
  author = {Fraile Navarro, David},
  title  = {SAE-Guided Cross-Lingual Knowledge Transfer in LLMs: A Null Result with Preserved Findings},
  year   = {2026},
  url    = {https://github.com/dafraile/SAE_mad}
}
```

## License

MIT. See handoff document for context on collaborative attribution.
