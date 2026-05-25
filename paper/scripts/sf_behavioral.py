"""sf_behavioral.py -- SF cell (Structured input + Free-text output) for
the 2×2 design. Completes the cell the original Phase 0.5 / Phase 3b /
Phase 4b runs deliberately skipped.

Original 2×2:
  | Forced-Letter output | Free-text output |
  ----------------------|--------------------|------------------
  Structured input      | SL (=A cell, run)  | SF (NOT run)
  Natural input         | NL (=B cell, run)  | NF (=D cell, run)

This script runs the missing SF cell on 4B, 12B, Qwen — using the
canonical structured prompt (`structured_forced_letter` field) with the
forced-letter scaffold stripped, so the model is asked to respond in
free-text to the clinical narrative that was provided in structured
clinician-style form.

For Qwen we pass `enable_thinking=False` to suppress the <think> token
trace, matching the original NF run.

Output: `results/sf_behavioral_{4b,12b,qwen}.json` with the same row
schema as the NF cell in `phase{0_5,3b,4b}_*` so the existing
wire_adjudicator scripts can chain in.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
VIGNETTES_FL = ROOT / "paper/data/canonical_forced_letter_vignettes.json"

MODEL_IDS = {
    "4b":   "google/gemma-3-4b-it",
    "12b":  "google/gemma-3-12b-it",
    "qwen": "Qwen/Qwen3-8B",
}

MAX_NEW_TOKENS = 2000
LETTER_LINE_RE = re.compile(r"\b([ABCD])\b")

# Same heuristic regexes used in phase{0_5,3b,4b} for the best-effort
# letter parse on free-text responses (used as a sanity check; gold-
# standard scoring is by LLM-judge adjudication).
URGENT_D = ["911", "emergency room", "emergency department",
            "call an ambulance", "go to the er", "to the ER",
            "immediately go", "urgent care now"]
URGENT_D_ER = re.compile(r"\b(ER|ED)\b")
SOON_B = ["today", "within 24 hours", "same day", "urgent care",
          "see a doctor", "schedule an appointment soon",
          "within a day or two", "within the next day"]
WAIT_C = ["within a few days", "schedule a routine", "next week",
          "primary care", "non-urgent", "general practitioner",
          "follow up with"]
WATCH_A = ["monitor at home", "rest", "over-the-counter",
           "if it gets worse", "self-care", "watch and wait"]


def extract_free_text(text: str) -> tuple[str | None, str]:
    if not text:
        return None, "empty"
    lo = text.lower()
    for kw in URGENT_D:
        if kw in lo: return "D", f"urgent:'{kw}'"
    if URGENT_D_ER.search(text): return "D", "urgent:ER/ED"
    for kw in SOON_B:
        if kw in lo: return "B", f"soon:'{kw}'"
    for kw in WAIT_C:
        if kw in lo: return "C", f"wait:'{kw}'"
    for kw in WATCH_A:
        if kw in lo: return "A", f"watch:'{kw}'"
    return None, "no-keyword"


def parse_gold(g: str) -> list[str]:
    return sorted(set(re.findall(r"[ABCD]", g.upper())))


def build_sf_prompt(structured_forced_letter: str) -> str:
    """Strip the forced-letter scaffold from the structured prompt to get
    the SF (free-text) variant. The structured prompt already starts with
    'I'm looking for help understanding what these results might mean and
    how urgently I should act.' — the opening question persists in SF and
    drives the model to give a free-text triage recommendation."""
    marker = "Reply with exactly one letter only."
    idx = structured_forced_letter.find(marker)
    if idx == -1:
        # Defensive: if marker missing, return as-is
        return structured_forced_letter
    # Strip the marker + the four A/B/C/D bullets + trailing instructions
    return structured_forced_letter[:idx].rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_IDS), required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    model_id = MODEL_IDS[args.model]

    fl = json.loads(VIGNETTES_FL.read_text())
    cases = sorted(fl, key=lambda v: (re.match(r"^(\D+)", v["id"]).group(1),
                                       int(re.search(r"\d+", v["id"]).group())))
    assert len(cases) == 60

    print(f"Loading {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=False,
                enable_thinking=False,
            )
        except TypeError:
            ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=False,
            )
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
        ids = ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    results = []
    t0 = time.time()
    for i, c in enumerate(cases):
        sf_prompt = build_sf_prompt(c["structured_forced_letter"])
        gold_letters = parse_gold(c["gold_standard_triage"])
        out = generate(sf_prompt)
        pred, reason = extract_free_text(out)
        row = {
            "id": c["id"],
            "title": c["title"],
            "gold_raw": c["gold_standard_triage"],
            "gold_letters": gold_letters,
            "SF_prompt": sf_prompt,
            "SF": {
                "predicted": pred,
                "correct": (pred in gold_letters) if pred else False,
                "unparsed": pred is None,
                "parse_reason": reason,
                "raw": out,
            },
        }
        results.append(row)
        elapsed = time.time() - t0
        marker = "✓" if row["SF"]["correct"] else ("✗" if pred else "?")
        print(f"[{i+1:2d}/60] {c['id']:>4s}  gold={'/'.join(gold_letters):<4s}  "
              f"SF={pred or '??'}{marker}  reason={reason[:25]:<25s}  ({elapsed:.0f}s)")

    n = len(results)
    correct = sum(r["SF"]["correct"] for r in results)
    unparsed = sum(r["SF"]["unparsed"] for r in results)
    summary = {
        "model": model_id,
        "model_tag": args.model,
        "cell": "SF (structured input + free-text output)",
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_cases": n,
        "heuristic_correct": correct,
        "heuristic_accuracy_pct": 100 * correct / n,
        "unparsed": unparsed,
        "results": results,
    }
    out_path = ROOT / f"results/sf_behavioral_{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print(f"=== SF behavioral summary ({args.model}) ===")
    print(f"  heuristic accuracy: {correct}/{n} = {100*correct/n:.1f}% (unparsed {unparsed})")
    print(f"  wall time: {(time.time()-t0)/60:.1f} min")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
