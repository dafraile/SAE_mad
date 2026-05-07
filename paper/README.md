# `paper/` — Active workshop paper

Everything related to the EMNLP 2026 workshop submission lives here.

## Files at this level

- `PAPER_DRAFT.md` — full prose draft of the paper.
  Voice is currently AI-cadenced; will be re-authored by the lead author
  in their voice before submission per ACL paper-integrity policy.
- `ABSTRACT_STRUCTURE.md` — sentence-slot scaffold for the abstract.
- `PAPER_OUTLINE.md` — original section-level outline (kept for reference).
- `REFERENCES_TO_FIND.md` — the brief that was given to the reference-research agent.
- `references.bib` — verified BibTeX for all citations.
- `TRIAGE_FINDINGS.md` — canonical empirical record (lab notebook for this paper).
- `make_figures.py`, `make_fig4.py` — figure regeneration scripts. Run from project root: `python3 paper/make_figures.py`.
- `figures/` — generated figures (`fig1`–`fig4`) as PDF (vector, editable text) and PNG (preview).

## `scripts/`

One script per phase of the active project. The phase-by-phase narrative is
in `TRIAGE_FINDINGS.md`; the project-wide README has a phase-index table
mapping each script to its result. **Run all scripts from the project root**,
not from inside `scripts/`. They reference paths like `results/...` and
`nature_triage_expanded_replication/...` relative to the project root.

Example:
```bash
# from project root
python3 paper/scripts/phase0_capability_floor.py
```
