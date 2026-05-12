"""nlaB4_judge.py -- LLM-as-judge classification of NLA descriptions.

Reads results/nlaB_descriptions.json (420 NLA-generated natural-language
descriptions of L32 residual-stream activations at 7 token positions
per case), and asks two LLM judges (gpt-5.2-thinking-high and
claude-sonnet-4.6, matching the Phase 0.5 adjudicator stack) to
classify each description on two independent axes:

  MEDICAL:   does this description primarily characterize the
             activation as a clinical / medical representation?
  SCAFFOLD:  does this description primarily characterize the
             activation as a multiple-choice / forced-letter /
             quiz-format scaffold representation?

Each axis is rated PRIMARY / PARTIAL / NO. PRIMARY = the description's
load-bearing content is on this concept. PARTIAL = the concept appears
but the description's primary frame is different (e.g. "Medical Q&A
format" mentions medical but is primarily framing the scaffold).
NO = the concept does not appear meaningfully.

We pre-extract just the structured rating per description; the judges'
free-text rationale is kept for spot-check inspection.

Output: results/nlaB_judge.json (one row per (record, judge)).

Usage (from project root, with OPENAI_API_KEY + ANTHROPIC_API_KEY set):
    python3 paper/scripts/nlaB4_judge.py --limit 5   # smoke test
    python3 paper/scripts/nlaB4_judge.py             # full run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NLA_DESCS = ROOT / "results" / "nlaB_descriptions.json"
OUT_JSON = ROOT / "results" / "nlaB_judge.json"
NLA_REPL_DIR = ROOT / "nature_triage_expanded_replication"

# Bring in the existing MODELS dict + call helpers from the paper-replication repo
sys.path.insert(0, str(NLA_REPL_DIR))
from config import MODELS, OPENAI_API_KEY, ANTHROPIC_API_KEY  # noqa: E402
from adjudicate_natural_interaction import (  # noqa: E402
    call_openai_adjudicator, call_anthropic_adjudicator,
)

JUDGE_MODELS = ["gpt-5.2-thinking-high", "claude-sonnet-4.6"]

PROMPT_TMPL = """You are evaluating natural-language descriptions of LLM
activation vectors. Each description was produced by a Natural Language
Autoencoder (Fraser-Taliente et al. 2026) that maps a single residual-stream
activation vector from Gemma 3 12B IT at layer 32 to a free-text explanation
of what the model encodes at that token position.

Your job: classify the description on TWO INDEPENDENT axes.

AXIS 1 — MEDICAL:
  Does the description primarily characterize the activation as a clinical
  or medical representation? (e.g. patient symptoms, medical reasoning,
  clinical case, diagnostic question, triage advice)

AXIS 2 — SCAFFOLD:
  Does the description primarily characterize the activation as a
  multiple-choice / forced-letter / quiz-style scaffold representation?
  (e.g. lettered answer options, "A = ...", "B = ...", select-one answer,
  quiz format, multiple-choice question, answer label)

Rate each axis on this 3-level scale:
  PRIMARY  — this concept is the description's load-bearing content
  PARTIAL  — this concept appears in the description but is not its
             primary frame (e.g. "Medical Q&A FORMAT" mentions medical
             but the description is primarily about the quiz-format role)
  NO       — this concept does not appear meaningfully

Description to rate:
<<<DESCRIPTION>>>
{description}
<<<END>>>

Respond exactly in this format (no extra commentary before the labels):
MEDICAL: <PRIMARY/PARTIAL/NO>
SCAFFOLD: <PRIMARY/PARTIAL/NO>
RATIONALE: <one sentence explaining the call>"""

LABEL_RE_MED = re.compile(r"^\s*MEDICAL\s*:\s*(PRIMARY|PARTIAL|NO)\b",
                          re.IGNORECASE | re.MULTILINE)
LABEL_RE_SCA = re.compile(r"^\s*SCAFFOLD\s*:\s*(PRIMARY|PARTIAL|NO)\b",
                          re.IGNORECASE | re.MULTILINE)
LABEL_RE_RAT = re.compile(r"^\s*RATIONALE\s*:\s*(.+?)\s*$",
                          re.IGNORECASE | re.MULTILINE | re.DOTALL)


def parse_judge_output(raw: str) -> dict:
    med = LABEL_RE_MED.search(raw or "")
    sca = LABEL_RE_SCA.search(raw or "")
    rat = LABEL_RE_RAT.search(raw or "")
    return {
        "medical": med.group(1).upper() if med else None,
        "scaffold": sca.group(1).upper() if sca else None,
        "rationale": (rat.group(1).strip() if rat else None),
        "raw": raw,
    }


def call_judge(judge: str, prompt: str) -> str:
    provider = MODELS[judge]["provider"]
    if provider == "openai":
        # NLA descriptions are short; 1024-completion budget is plenty,
        # plus thinking tokens for gpt-5.2-thinking-high
        return call_openai_adjudicator(judge, prompt, max_completion_tokens=4096)
    elif provider == "anthropic":
        return call_anthropic_adjudicator(judge, prompt, max_tokens=2048)
    raise ValueError(f"unsupported provider {provider!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N descriptions (smoke test).")
    parser.add_argument("--judges", nargs="+", default=JUDGE_MODELS,
                        help="Judge models to use.")
    parser.add_argument("--checkpoint-every", type=int, default=20,
                        help="Save progress to disk every N records.")
    args = parser.parse_args()

    # Only require the keys that the selected judges actually need.
    need_openai = any(MODELS[j]["provider"] == "openai" for j in args.judges)
    need_anthropic = any(MODELS[j]["provider"] == "anthropic" for j in args.judges)
    if need_openai and not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set (required for selected judges)")
        sys.exit(2)
    if need_anthropic and not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set (required for selected judges)")
        sys.exit(2)

    print(f"[B4] reading NLA descriptions from {NLA_DESCS}")
    d = json.loads(NLA_DESCS.read_text())
    records = d["results"]
    if args.limit:
        records = records[:args.limit]
    print(f"     {len(records)} records to judge with judges={args.judges}")

    # Resume support: load existing file if present
    out_records = []
    seen = set()
    if OUT_JSON.exists():
        existing = json.loads(OUT_JSON.read_text())
        out_records = existing.get("judgments", [])
        seen = {(j["record_id"], j["judge"]) for j in out_records}
        print(f"     resuming: {len(out_records)} prior judgments loaded")

    t0 = time.time()
    n_calls = 0
    for i, rec in enumerate(records):
        desc = rec["samples"][0] if rec.get("samples") else ""
        if not desc.strip():
            continue
        prompt = PROMPT_TMPL.format(description=desc)
        for judge in args.judges:
            if (rec["record_id"], judge) in seen:
                continue
            try:
                raw = call_judge(judge, prompt)
                parsed = parse_judge_output(raw)
                err = None
            except Exception as e:
                raw, parsed, err = None, {"medical": None, "scaffold": None,
                                          "rationale": None, "raw": None}, str(e)
            out_records.append({
                "record_id": rec["record_id"],
                "case_id":   rec["case_id"],
                "format":    rec["format"],
                "kind":      rec["kind"],
                "judge":     judge,
                "medical":   parsed["medical"],
                "scaffold":  parsed["scaffold"],
                "rationale": parsed["rationale"],
                "raw":       parsed["raw"],
                "error":     err,
            })
            seen.add((rec["record_id"], judge))
            n_calls += 1

        # Checkpoint
        if (i + 1) % args.checkpoint_every == 0 or (i + 1) == len(records):
            OUT_JSON.write_text(json.dumps({
                "source": str(NLA_DESCS),
                "n_records": len(records),
                "n_judgments_total": len(out_records),
                "judges": args.judges,
                "judgments": out_records,
            }, indent=2))
            elapsed = time.time() - t0
            rate = n_calls / elapsed if elapsed > 0 else 0
            remaining = len(records) - (i + 1)
            eta = remaining * len(args.judges) / rate if rate > 0 else float("inf")
            print(f"  [{i+1}/{len(records)}] +{n_calls} calls in {elapsed/60:.1f} min, "
                  f"{rate:.2f} call/s, ETA {eta/60:.1f} min")

    print(f"\n[B4] DONE. {n_calls} new calls in {(time.time()-t0)/60:.1f} min.")
    print(f"     {OUT_JSON}")


if __name__ == "__main__":
    main()
