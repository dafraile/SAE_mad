"""Wire 12B Phase 0.5 D-cell outputs for the paper-faithful adjudicator."""
import json
from pathlib import Path

PHASE_05 = Path("results/phase3b_12b_phase0_5.json")
OUT = Path("results/phase3b_12b_D_for_adjudication.json")
VIGNETTES = Path(
    "nature_triage_expanded_replication/paper_faithful_replication/data/"
    "canonical_singleturn_vignettes.json"
)

p05 = json.loads(PHASE_05.read_text())
vignettes = {v["id"]: v for v in json.loads(VIGNETTES.read_text())}

rows = []
for r in p05["results"]:
    cid = r["id"]
    rows.append({
        "case_id": cid,
        "case_title": r["title"],
        "gold_standard": r["gold_raw"],
        "source_model": "gemma-3-12b-it",
        "source_provider": "huggingface",
        "prompt_format": "patient_realistic",
        "run_number": "1",
        "source_user_message": vignettes[cid]["patient_realistic"],
        "raw_response": r["D"]["raw"],
        "best_effort_triage": None,
        "best_effort_is_correct": None,
        "error": "",
    })

OUT.write_text(json.dumps(rows, indent=2))
print(f"Wrote {len(rows)} rows to {OUT}")
