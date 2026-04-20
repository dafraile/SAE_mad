# SAE-Guided Cross-Lingual Knowledge Transfer in LLMs

Research project investigating whether Sparse Autoencoder (SAE) features can serve as routing signals for transferring capability across languages in large language models.

**Status**: Complete findings for a short workshop paper. See [FINDINGS.md](FINDINGS.md) for results.

## TL;DR

On Gemma 3 4B, we show that amplifying a single "Spanish language" SAE feature during English inference causally rescues 3-7% of the English-vs-Spanish performance gap on medical, STEM, and humanities MCQ benchmarks. The mechanism:

- **Works across languages** (ES, FR, ZH -- including linguistically distant Chinese)
- **Works across domains** (medical, philosophy, STEM, humanities, global facts)
- **Is causal** (monotonic with steering strength, clean 0x sanity check)
- **Requires language-general features** -- domain-specific features don't transfer
- **Has a ceiling** -- combining ES + FR features gives no additional rescue beyond ES alone

See [FINDINGS.md](FINDINGS.md) for the full experimental record.

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
│   ├── v2_steering_output.txt             ← v2 generation examples
│   ├── v2_medical_baseline.json           ← Pilot (1B)
│   ├── v2_medical_baseline_4b.json        ← Pilot (4B)
│   ├── v2_medical_rescue.json             ← First rescue result (+9)
│   ├── v2_medical_rescue_v2.json          ← Targeted rescue (+1, negative)
│   ├── v2_generalization.json             ← Cross-domain/language matrix
│   └── v2_flip_distant_combined.json      ← Extended experiments
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

1. `hw1_load_model.py` → `hw5_multilingual.py` -- Verify environment
2. `v1_exploration.py` -- Build feature understanding (uses `corpus.json`)
3. `v2_steering.py` -- Confirm features are causal (1B model)
4. `v2_medical_pilot.py` -- Establish the reverse gap (both 1B and 4B)
5. `v2_medical_rescue.py` -- Main result: generic feature rescue
6. `v2_generalization.py` -- Generalization across domains/languages
7. `v2_flip_distant_combined.py` -- Boundaries of the mechanism

## Timing and Cost

Rough GPU cost estimates on vast.ai ($0.15-0.30/hr for 24GB GPUs):

| Experiment | Duration | Cost |
|------------|----------|------|
| Hello-worlds (all) | ~5 min | $0.01 |
| v1 exploration | ~15 min | $0.05 |
| v2 steering | ~10 min | $0.03 |
| v2 medical pilot (4B) | ~20 min | $0.10 |
| v2 medical rescue | ~25 min | $0.12 |
| v2 generalization | ~60 min | $0.25 |
| v2 flip/distant/combined | ~75 min | $0.30 |
| **Total (full pipeline)** | **~3.5 hours** | **~$1.00** |

## Key Findings (see [FINDINGS.md](FINDINGS.md) for details)

1. **Gemma 3 shows a reversed multilingual gap** -- non-English outperforms English across all tested domains, even CS/coding
2. **Single SAE feature amplification rescues 3-7% of English underperformance**, causally and consistently
3. **The mechanism is universal** across ES, FR, ZH and 5 tested domains
4. **Language-general features work; domain-specific features don't** -- rules out "domain-specific knowledge highway"
5. **Feature composition has a ceiling** -- combining ES+FR features doesn't exceed ES alone
6. **Gap size scales with linguistic distance** -- Romance languages show bigger advantage over English than Chinese does

## Citation

If this work informs your research before we publish, please cite the repository:

```
@misc{fraile-navarro-2026-sae-cross-lingual,
  author = {Fraile Navarro, David},
  title  = {SAE-Guided Cross-Lingual Knowledge Transfer in LLMs},
  year   = {2026},
  url    = {https://github.com/dafraile/SAE_mad}
}
```

(A proper paper is in preparation.)

## License

MIT. See handoff document for context on collaborative attribution.
