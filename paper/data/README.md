# `paper/data/` — Canonical patient vignettes

Two JSON files, 60 cases each, that are the **inputs** to every experiment
in this paper. The same 60 case IDs (E1–E30 + F1–F26 + MH1–MH3 + NH1–NH3)
appear in both files; the difference is the prompt-format variant attached
to each case.

The data are reproduced here verbatim from the parallel paper-faithful
replication project that lives separately at
`github.com/dafraile/nature_triage_expanded_replication` and is mirrored
to `nature_triage_expanded_replication/` in this repo (gitignored — too
big to track directly). Copying these two files into `paper/data/` is
the minimum needed to make every experiment in this paper reproducible
from a fresh clone.

## Files

### `canonical_singleturn_vignettes.json`

60-item list. Each item has fields:

- `id` — short case ID like `E1`, `F12`, `MH3` (E=emergency, F=family-med,
  MH=mental-health, NH=non-clinical/help).
- `title` — short label for the case (e.g. `"E1 TIA"`).
- `gold_standard_triage` — the clinician-assigned correct triage on the
  paper's 4-level scale: `A` (monitor at home), `B` (see doctor in
  weeks), `C` (within 24–48h), `D` (ER now). Some cases list two
  permissible adjacent letters (e.g. `"C/D"`).
- `original_structured` — the paper's original constrained, structured
  prompt; used for the **SL** (structured + forced-letter) condition.
  This is the prompt the original Nature paper scored.
- `patient_realistic` — a free-text rewrite of the same clinical case
  in a "patient writing in to a portal" voice. Used for both the
  **NF** (natural + free-text) condition (no answer-key scaffold; the
  model writes a free-text reply) and as the prefix of the NL
  condition.
- `paper_metadata` — provenance fields (originating paper section,
  edits, etc.); not used by the analysis pipeline.

### `canonical_forced_letter_vignettes.json`

60-item list with the same 60 case IDs as above. Each item:

- `id`, `title`, `gold_standard_triage` — as above.
- `structured_forced_letter` — `original_structured` plus the
  forced-letter scaffold (`Reply with exactly one letter only…
  A = …  B = …  C = …  D = …  Do not include any explanation or
  extra words.`). Used for the original Nature-paper-faithful
  **SL** condition with the explicit answer key spelled out.
- `natural_forced_letter` — `patient_realistic` with the same
  forced-letter scaffold appended. Used for the **NL** (natural +
  forced-letter) condition. **This is the condition that drives the
  format-effect finding** in the paper.
- `paper_metadata` — as above.

## The three behavioral cells

These two files together give us the three behavioral conditions in
§4.1 (Table 1):

| Cell | File | Field used |
|---|---|---|
| **SL** (structured + forced-letter) | `canonical_forced_letter_vignettes.json` | `structured_forced_letter` |
| **NL** (natural + forced-letter)     | `canonical_forced_letter_vignettes.json` | `natural_forced_letter`    |
| **NF** (natural + free-text)         | `canonical_singleturn_vignettes.json`    | `patient_realistic`        |

The clinical content is byte-identical between NF and NL up to the
trailing forced-letter scaffold block — i.e. for any case, removing
"Reply with exactly one letter only…" + the four letter options + "Do
not include any explanation or extra words." from `natural_forced_letter`
yields the `patient_realistic` text. This is what lets us isolate the
format effect: the model sees identical clinical input across NL and
NF, and only the output-format instruction differs.

## How the scripts consume these files

Every analysis script reads from these paths:

- `paper/scripts/phase0_5_three_cells.py` (4B behavioral)
- `paper/scripts/phase3b_12b_pipeline.py` (12B behavioral + SAE)
- `paper/scripts/phase4_qwen_minimal.py` (Qwen3-8B behavioral + SAE)
- `paper/scripts/phase5_top_tokens_and_restricted_random.py`
- `paper/scripts/phase1b_magnitude_matched.py`
- `paper/scripts/phase2b_dilution_check.py`
- `paper/scripts/phase1b_sensitivity.py`
- `paper/scripts/nlaB1_extract_L32.py`

All reference these two JSONs (or the legacy
`nature_triage_expanded_replication/...` paths that mirror them). For a
fresh clone of the repo, the canonical paths are the ones under
`paper/data/`.

## Reproducing model outputs from these vignettes

`paper/scripts/phase0_5_three_cells.py` (4B) and
`paper/scripts/phase3b_12b_pipeline.py` (12B) generate model responses
under all three conditions using `max_new_tokens=2000` (greedy
decoding). Outputs land in `results/_v2/`. See the paper for details
of the generation pipeline.
