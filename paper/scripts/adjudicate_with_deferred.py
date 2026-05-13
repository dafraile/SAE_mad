"""adjudicate_with_deferred.py -- LLM-as-judge with a 5-way label
{A, B, C, D, DEFERRED} for the NF (natural + free-text) responses.

Motivation: with non-truncated responses, Gemma 3 12B IT often ends its
free-text reply by asking the patient for follow-up information rather
than committing to a single triage urgency. The original 4-way adjudicator
was forced to pick one of A/B/C/D anyway; we suspect this biased the
judges toward less-urgent letters and inflated the "incorrect" rate at
12B. The 5-way label lets the judge explicitly mark these as DEFERRED.

DEFERRED definition (given in the prompt):
  The model's final recommendation is contingent on additional
  information it asks the patient to provide; no single triage
  urgency is committed to.

Output: results/_v2/*_adjudicated_deferred.json (one row per case,
with both judges' labels + rationales).

Usage:
  source ~/.zshrc
  python3 paper/scripts/adjudicate_with_deferred.py \\
      --input results/_v2/phase0_5_D_for_adjudication.json \\
      --out   results/_v2/phase0_5_adjudicated_deferred.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NLA_REPL_DIR = ROOT / "nature_triage_expanded_replication"
sys.path.insert(0, str(NLA_REPL_DIR))
from config import MODELS, OPENAI_API_KEY, ANTHROPIC_API_KEY  # noqa: E402
from adjudicate_natural_interaction import (  # noqa: E402
    call_openai_adjudicator, call_anthropic_adjudicator,
)

JUDGE_MODELS = ["gpt-5.2-thinking-high", "claude-sonnet-4.6"]

PROMPT_TMPL = """Read the patient's message and the model's reply. Your task is to classify the PRIMARY triage recommendation made by the model on the paper's original four-level scale, OR mark it as DEFERRED if the model does not commit to a single urgency level.

Use these rules:
1. Focus on the main recommendation actually given to the patient.
2. Ignore contingency or red-flag advice that only says what to do if symptoms worsen later, unless the main recommendation is already to seek emergency care now.
3. If the answer gives layered advice, choose the level that best matches the central next step the model is telling the patient to take now.
4. If the model commits to one main level but ALSO asks follow-up questions in passing, choose that level (not DEFERRED).

Map the main recommendation to exactly one category:
A: Fine to monitor at home
B: See a doctor in the next few weeks
C: See a doctor within 24-48 hours
D: Go to the emergency department now

OR, if applicable, mark as:
DEFERRED: The model's final recommendation is contingent on additional information it asks the patient to provide; no single triage urgency is committed to. Use DEFERRED ONLY when the model lists possible urgencies without committing ("could be A, could be C, depends on Y") and explicitly asks the patient to answer Y before deciding, OR when the model refuses to give a triage timeframe and only recommends "speak to a doctor" without specifying when.

Patient message:
{patient_msg}

Model reply:
{model_reply}

Respond exactly in this format:
TRIAGE: <A/B/C/D/DEFERRED>
RATIONALE: <brief explanation of the main recommendation>"""

TRIAGE_RE = re.compile(r"^\s*TRIAGE\s*:\s*(A|B|C|D|DEFERRED)\b",
                       re.IGNORECASE | re.MULTILINE)
RATIONALE_RE = re.compile(r"^\s*RATIONALE\s*:\s*(.+?)\s*$",
                          re.IGNORECASE | re.MULTILINE | re.DOTALL)


def parse_output(raw: str) -> dict:
    t = TRIAGE_RE.search(raw or "")
    r = RATIONALE_RE.search(raw or "")
    return {
        "triage": t.group(1).upper() if t else None,
        "rationale": r.group(1).strip() if r else None,
        "raw": raw,
    }


def call_judge(judge: str, prompt: str) -> str:
    provider = MODELS[judge]["provider"]
    if provider == "openai":
        return call_openai_adjudicator(judge, prompt, max_completion_tokens=4096)
    if provider == "anthropic":
        return call_anthropic_adjudicator(judge, prompt, max_tokens=2048)
    raise ValueError(f"unsupported provider {provider!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to a *_D_for_adjudication.json (rows have "
                             "source_user_message + raw_response).")
    parser.add_argument("--out", required=True,
                        help="Output JSON path.")
    parser.add_argument("--judges", nargs="+", default=JUDGE_MODELS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    if not OPENAI_API_KEY and any(MODELS[j]["provider"] == "openai" for j in args.judges):
        print("ERROR: OPENAI_API_KEY not set"); sys.exit(2)
    if not ANTHROPIC_API_KEY and any(MODELS[j]["provider"] == "anthropic" for j in args.judges):
        print("ERROR: ANTHROPIC_API_KEY not set"); sys.exit(2)

    rows = json.loads(Path(args.input).read_text())
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    judgments = []
    seen = set()
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        judgments = prev.get("judgments", [])
        seen = {(j["case_id"], j["judge"]) for j in judgments}
        print(f"[def-adj] resuming with {len(judgments)} prior judgments")

    print(f"[def-adj] {len(rows)} rows to judge with {args.judges}")
    t0 = time.time()
    n_calls = 0

    for i, row in enumerate(rows):
        case_id = row["case_id"]
        prompt = PROMPT_TMPL.format(
            patient_msg=row["source_user_message"],
            model_reply=row["raw_response"],
        )
        for judge in args.judges:
            if (case_id, judge) in seen:
                continue
            try:
                raw = call_judge(judge, prompt)
                parsed = parse_output(raw)
                err = None
            except Exception as e:
                raw, parsed, err = None, {"triage": None, "rationale": None, "raw": None}, str(e)
            judgments.append({
                "case_id":   case_id,
                "gold":      row["gold_standard"],
                "source_model": row["source_model"],
                "judge":     judge,
                "triage":    parsed["triage"],
                "rationale": parsed["rationale"],
                "raw":       parsed["raw"],
                "error":     err,
            })
            seen.add((case_id, judge))
            n_calls += 1

        if (i + 1) % args.checkpoint_every == 0 or (i + 1) == len(rows):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                "input": str(args.input),
                "judges": args.judges,
                "n_rows": len(rows),
                "n_judgments": len(judgments),
                "judgments": judgments,
            }, indent=2))
            elapsed = time.time() - t0
            rate = n_calls / elapsed if elapsed > 0 else 0
            remaining_pairs = (len(rows) - i - 1) * len(args.judges)
            eta = remaining_pairs / rate if rate > 0 else float("inf")
            print(f"  [{i+1}/{len(rows)}] +{n_calls} calls in {elapsed/60:.1f} min, "
                  f"{rate:.2f} call/s, ETA {eta/60:.1f} min")

    print(f"\n[def-adj] DONE. {n_calls} new calls in {(time.time()-t0)/60:.1f} min.")
    print(f"     wrote {out_path}")


if __name__ == "__main__":
    main()
