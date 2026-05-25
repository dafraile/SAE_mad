"""wire_adjudicator_sf.py -- wire SF (structured + free-text) cell
behavioral outputs for the paper-faithful adjudicator. Generalized to
all three models (4B, 12B, Qwen).

Same pattern as wire_adjudicator{.py,_12b.py,_qwen.py} (which handle the
NF cell), but for SF.

Usage:
  python3 paper/scripts/wire_adjudicator_sf.py --model 4b
  python3 paper/scripts/wire_adjudicator_sf.py --model 12b
  python3 paper/scripts/wire_adjudicator_sf.py --model qwen
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIGNETTES_FL = ROOT / "paper/data/canonical_forced_letter_vignettes.json"

SOURCE_MODELS = {
    "4b":   "gemma-3-4b-it",
    "12b":  "gemma-3-12b-it",
    "qwen": "qwen3-8b",
}

SOURCE_PROVIDERS = {
    "4b":   "huggingface",
    "12b":  "huggingface",
    "qwen": "huggingface",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SOURCE_MODELS), required=True)
    args = ap.parse_args()
    tag = args.model

    sf = json.loads((ROOT / f"results/sf_behavioral_{tag}.json").read_text())
    vignettes = {v["id"]: v for v in json.loads(VIGNETTES_FL.read_text())}

    rows = []
    for r in sf["results"]:
        cid = r["id"]
        rows.append({
            "case_id":            cid,
            "case_title":         r["title"],
            "gold_standard":      r["gold_raw"],
            "source_model":       SOURCE_MODELS[tag],
            "source_provider":    SOURCE_PROVIDERS[tag],
            "prompt_format":      "structured_freetext",
            "run_number":         "1",
            "source_user_message": r.get("SF_prompt") or vignettes[cid]["structured_forced_letter"],
            "raw_response":       r["SF"]["raw"],
            "best_effort_triage": r["SF"]["predicted"],
            "best_effort_is_correct": r["SF"]["correct"],
            "error":              "",
        })

    out_path = ROOT / f"results/sf_{tag}_D_for_adjudication.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
