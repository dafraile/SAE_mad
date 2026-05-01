"""
Wire the Phase 0.5 D-cell (natural + free-text) outputs into the format
consumed by the paper-faithful adjudicator:

  nature_triage_expanded_replication/paper_faithful_replication/scripts/
    adjudicate_natural_paper_scale.py

Why the paper-scale adjudicator (and not adjudicate_natural_interaction.py):
  - The top-level adjudicate_natural_interaction.py uses the INVERTED scale
    (A=ER, D=home). Our canonical vignettes use the paper's NATIVE scale
    (A=home, D=ER). Running the inverted adjudicator against native-scale
    gold would silently scramble every correctness check.
  - adjudicate_natural_paper_scale.py uses A=home, D=ER. Matches our data.

This script only prepares a JSON input file. It does NOT call any paid APIs.
Once you confirm credits, run the printed command yourself.
"""
from __future__ import annotations

import json
from pathlib import Path

PHASE_0_5 = Path("results/phase0_5_three_cells.json")
OUT = Path("results/phase0_5_D_for_adjudication.json")

REPO_ROOT = Path("nature_triage_expanded_replication")
ADJUDICATOR_SCRIPT = REPO_ROOT / "paper_faithful_replication/scripts/adjudicate_natural_paper_scale.py"
VIGNETTES_PATH = REPO_ROOT / "paper_faithful_replication/data/canonical_singleturn_vignettes.json"


def main() -> None:
    d = json.loads(PHASE_0_5.read_text())
    assert ADJUDICATOR_SCRIPT.exists(), f"missing {ADJUDICATOR_SCRIPT}"
    assert VIGNETTES_PATH.exists(), f"missing {VIGNETTES_PATH}"

    # Pull the patient_realistic prompt from the canonical source (what we fed to Gemma)
    vignettes = {v["id"]: v for v in json.loads(VIGNETTES_PATH.read_text())}

    rows = []
    for r in d["results"]:
        case_id = r["id"]
        v = vignettes[case_id]
        rows.append({
            "case_id": case_id,
            "case_title": r["title"],
            "gold_standard": r["gold_raw"],
            "source_model": "gemma-3-4b-it",
            "source_provider": "huggingface",
            "prompt_format": "patient_realistic",
            "run_number": "1",
            "source_user_message": v["patient_realistic"],
            "raw_response": r["D"]["raw"],
            "best_effort_triage": r["D"]["predicted"],
            "best_effort_is_correct": r["D"]["correct"],
            "error": "",
        })

    assert len(rows) == 60
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} rows to {OUT}")

    # Print the exact command to run the adjudicator.
    print("\n=== To run the adjudicator (after confirming API credits) ===\n")
    print("Set API keys first:")
    print("  export OPENAI_API_KEY=sk-...")
    print("  export ANTHROPIC_API_KEY=sk-ant-...")
    print()
    print("Preview the prompt without calling any API (recommended first step):")
    print(
        f"  python3 {ADJUDICATOR_SCRIPT} \\\n"
        f"    --input {OUT} \\\n"
        f"    --vignettes-path {VIGNETTES_PATH} \\\n"
        f"    --dry-run"
    )
    print()
    print("Full run (costs money — 60 rows × 2 judges = 120 API calls):")
    print(
        f"  python3 {ADJUDICATOR_SCRIPT} \\\n"
        f"    --input {OUT} \\\n"
        f"    --vignettes-path {VIGNETTES_PATH} \\\n"
        f"    --adjudicators gpt-5.2-thinking-high claude-sonnet-4.6 \\\n"
        f"    --output-dir results/"
    )
    print()
    print("Defaults in the script are gpt-5.4-xhigh + claude-opus-4.6 — more")
    print("expensive than what I've suggested above. Swap back if you want")
    print("the exact models the paper used. gpt-5.2-thinking-high and")
    print("claude-sonnet-4.6 are cheaper and should be adequate for this task;")
    print("the paper reports 94.7% agreement / kappa=0.921 between its judges,")
    print("suggesting the judgment is not delicate.")


if __name__ == "__main__":
    main()
