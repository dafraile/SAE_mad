# `exploratory/` — Closed prior work (Phase 1)

This directory contains the original SAE-routing project that ran before the
current workshop paper. It is **closed**: a properly scoped null result, not
abandoned — see `FINDINGS.md` for the full retraction-and-validation record.

## Why it's still here

Two reasons.

1. **The medical features identified in v3** (Gemma 3 4B IT, layer 29: features
   893, 12570, 12845) are the seed for the active paper. The workshop paper's
   "medical features" are the v3 features, validated again under the new
   methodology. Keeping the v3 work visible lets readers verify the lineage.
2. **Methodology lessons** distilled from the closed null — particularly the
   `/sanity-check` skill — informed every experiment in the active project.
   That history is part of the paper's provenance.

## Files at this level

- `FINDINGS.md` — full empirical record of the closed null, including the
  retraction of v2-medical and the controlled-replication results in v3.
- `sae_routing_experiment_handoff.md` — original research plan from project
  start.
- `corpus.json`, `corpus_template.json` — parallel EN/ES/FR corpus used in v1
  and the contrastive feature identification in v3.

## Scripts

- `hw1`–`hw5_*.py` — hello-world environment / pipeline checks.
- `v1_exploration.py` — cross-lingual representation analysis (validated).
- `v2_*.py` — language steering on 1B (Feature 857, validated) and the
  v2-medical rescue work (retracted; see FINDINGS for the dataset bug).
- `v3_*.py` — controlled replications and the layer-sweep null.

Run from project root, not from inside this directory:
```bash
python3 exploratory/v3_layer_sweep.py
```

These scripts' outputs are in the project-wide `results/` directory, not a
subdirectory here, because they were originally written when everything was
at root.

## Status

Do not extend this directory with new work. New experiments belong in the
active paper's lineage under `paper/scripts/`. This directory is for
historical reference only.
