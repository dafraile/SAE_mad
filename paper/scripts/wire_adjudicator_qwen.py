"""Wire Qwen3-8B Phase 4b D-cell (NF free-text) outputs for the
paper-faithful adjudicator.

Same pattern as wire_adjudicator.py (4B) and wire_adjudicator_12b.py
(12B), pointing at the Phase 4b output.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE_4B = ROOT / "results/phase4b_qwen_behavioral.json"
OUT = ROOT / "results/phase4b_qwen_D_for_adjudication.json"
VIGNETTES = ROOT / "paper/data/canonical_singleturn_vignettes.json"

p4b = json.loads(PHASE_4B.read_text())
vignettes = {v["id"]: v for v in json.loads(VIGNETTES.read_text())}

rows = []
for r in p4b["results"]:
    cid = r["id"]
    rows.append({
        "case_id": cid, "case_title": r["title"], "gold_standard": r["gold_raw"],
        "source_model": "qwen3-8b", "source_provider": "huggingface",
        "prompt_format": "patient_realistic", "run_number": "1",
        "source_user_message": vignettes[cid]["patient_realistic"],
        "raw_response": r["D"]["raw"],
        "best_effort_triage": r["D"]["predicted"],
        "best_effort_is_correct": r["D"]["correct"],
        "error": "",
    })
OUT.write_text(json.dumps(rows, indent=2))
print(f"Wrote {len(rows)} rows to {OUT}")
