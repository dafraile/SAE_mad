"""
Phase 0.5 — Phenomenon presence check (three cells, paper-faithful).

Question: Does Gemma 3 4B show the paper's format effect? Specifically:
- Does input style (structured vs natural) change accuracy, holding output
  constraint fixed (forced-letter)?
- Does output constraint (forced-letter vs free-text) change accuracy,
  holding input style fixed (natural)?

Three cells, all natively prepared in the repo:
  A: structured + forced-letter     -> canonical_forced_letter_vignettes.json
  B: natural    + forced-letter     -> canonical_forced_letter_vignettes.json
  D: natural    + free-text         -> canonical_singleturn_vignettes.json (patient_realistic)

We deliberately omit the fourth "structured + free-text" cell because the
paper did not test it, and constructing it would be a methodology choice
rather than a replication.

Go/no-go for Phase 1: need at least one of
  (A vs B): input effect on forced-letter   >= 5pp
  (B vs D): output effect on natural        >= 5pp

Output: results/phase0_5_three_cells.json
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-4b-it"

FORCED_LETTER_PATH = Path(
    "nature_triage_expanded_replication/paper_faithful_forced_letter/data/"
    "canonical_forced_letter_vignettes.json"
)
SINGLETURN_PATH = Path(
    "nature_triage_expanded_replication/paper_faithful_replication/data/"
    "canonical_singleturn_vignettes.json"
)
OUT_PATH = Path("results/phase0_5_three_cells.json")
MAX_NEW_TOKENS = 400

# --- Parsing ---

LETTER_LINE_RE = re.compile(r"\b([ABCD])\b")

URGENT_D = [
    "911", "emergency room", "emergency department",
    "call an ambulance", "go to the er", "to the ER",
    "immediately go", "urgent care now",
]
URGENT_D_ER = re.compile(r"\b(ER|ED)\b")  # conservative D marker if used as a noun

SOON_B = [
    "today", "within 24 hours", "same day", "urgent care",
]
DAYS_C = [
    "within a few days", "next couple of days", "24 to 48 hours",
    "24-48 hours", "within 24-48", "within a week",
    "see your doctor soon", "in the coming days",
]
HOME_A = [
    "self-care", "no need to see", "watch and wait",
    "rest and fluids", "doesn't require medical attention",
    "no need for a doctor", "does not require",
]

HEDGE_WINDOW = 40  # chars before the urgency phrase to scan for conditional

def extract_forced_letter(output: str) -> str | None:
    # Greedy: first capital letter A-D on its own line, else first in text.
    for line in output.splitlines():
        s = line.strip().strip(".,:;*-")
        if s in ("A", "B", "C", "D"):
            return s
    m = LETTER_LINE_RE.search(output)
    return m.group(1) if m else None


def _is_hedged(text: str, phrase_start: int) -> bool:
    window = text[max(0, phrase_start - HEDGE_WINDOW):phrase_start].lower()
    return any(h in window for h in (" if ", " if,", "should it", "should the",
                                     "if it", "in case", "unless", "worsen"))


def extract_free_text(output: str) -> tuple[str | None, str]:
    """Returns (category, reason). Category in {A,B,C,D} or None."""
    low = output.lower()

    # D first - emergency signals
    for phrase in URGENT_D:
        idx = low.find(phrase)
        if idx != -1 and not _is_hedged(output, idx):
            return "D", f"matched '{phrase}' (unhedged)"
    # Bare ER/ED as a noun, with hedge guard
    for m in URGENT_D_ER.finditer(output):
        if not _is_hedged(output, m.start()):
            return "D", f"matched '{m.group(0)}' as noun (unhedged)"

    # B - same-day
    for phrase in SOON_B:
        if phrase in low:
            return "B", f"matched '{phrase}'"

    # C - within days
    for phrase in DAYS_C:
        if phrase in low:
            return "C", f"matched '{phrase}'"

    # A - self-care
    for phrase in HOME_A:
        if phrase in low:
            return "A", f"matched '{phrase}'"

    return None, "no category keyword matched"


def parse_gold(gold: str) -> list[str]:
    return sorted(set(re.findall(r"[ABCD]", gold.upper())))


# --- Main ---

def build_cells() -> list[dict]:
    fl = json.loads(FORCED_LETTER_PATH.read_text())
    st = json.loads(SINGLETURN_PATH.read_text())
    fl_by_id = {v["id"]: v for v in fl}
    st_by_id = {v["id"]: v for v in st}

    # Assert alignment: all 60 ids present in both files
    assert set(fl_by_id) == set(st_by_id), "ID mismatch between files"

    def _sort_key(s: str) -> tuple:
        m = re.match(r"^(\D+)(\d+)$", s)
        return (m.group(1), int(m.group(2))) if m else (s, 0)

    cells = []
    for id_ in sorted(fl_by_id.keys(), key=_sort_key):
        fl_row = fl_by_id[id_]
        st_row = st_by_id[id_]
        assert fl_row["gold_standard_triage"] == st_row["gold_standard_triage"], (
            f"gold mismatch for {id_}"
        )
        cells.append({
            "id": id_,
            "title": fl_row["title"],
            "gold_raw": fl_row["gold_standard_triage"],
            "gold_letters": parse_gold(fl_row["gold_standard_triage"]),
            "A_prompt": fl_row["structured_forced_letter"],
            "B_prompt": fl_row["natural_forced_letter"],
            "D_prompt": st_row["patient_realistic"],
        })
    return cells


def main() -> None:
    cases = build_cells()
    assert len(cases) == 60

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=False,
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)

    results = []
    t0 = time.time()
    for i, c in enumerate(cases):
        row = {"id": c["id"], "title": c["title"],
               "gold_raw": c["gold_raw"], "gold_letters": c["gold_letters"]}
        # Cell A
        out_a = generate(c["A_prompt"])
        pred_a = extract_forced_letter(out_a)
        row["A"] = {
            "predicted": pred_a, "correct": pred_a in c["gold_letters"] if pred_a else False,
            "unparsed": pred_a is None, "raw": out_a,
        }
        # Cell B
        out_b = generate(c["B_prompt"])
        pred_b = extract_forced_letter(out_b)
        row["B"] = {
            "predicted": pred_b, "correct": pred_b in c["gold_letters"] if pred_b else False,
            "unparsed": pred_b is None, "raw": out_b,
        }
        # Cell D
        out_d = generate(c["D_prompt"])
        pred_d, reason_d = extract_free_text(out_d)
        row["D"] = {
            "predicted": pred_d, "correct": pred_d in c["gold_letters"] if pred_d else False,
            "unparsed": pred_d is None, "parse_reason": reason_d, "raw": out_d,
        }
        results.append(row)
        print(
            f"[{i+1:2d}/60] {c['id']:>4s}  gold={'/'.join(c['gold_letters']):<4s}  "
            f"A={pred_a or '??'}{'ok' if row['A']['correct'] else 'x '}  "
            f"B={pred_b or '??'}{'ok' if row['B']['correct'] else 'x '}  "
            f"D={pred_d or '??'}{'ok' if row['D']['correct'] else 'x '}  "
            f"({time.time()-t0:.0f}s)"
        )

    def agg(cell: str) -> dict:
        correct = sum(r[cell]["correct"] for r in results)
        unparsed = sum(r[cell]["unparsed"] for r in results)
        return {"n": len(results), "correct": correct, "unparsed": unparsed,
                "accuracy": correct / len(results)}

    summary = {
        "model": MODEL_ID,
        "cells": {"A": agg("A"), "B": agg("B"), "D": agg("D")},
        "comparisons": {
            "input_effect_on_forced_letter_A_minus_B": agg("A")["accuracy"] - agg("B")["accuracy"],
            "output_effect_on_natural_B_minus_D": agg("B")["accuracy"] - agg("D")["accuracy"],
        },
        "results": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print(f"\n=== Phase 0.5 three-cell summary ===")
    print(f"Model: {MODEL_ID}")
    for k in ["A", "B", "D"]:
        s = summary["cells"][k]
        label = {"A": "structured+forced-letter",
                 "B": "natural+forced-letter",
                 "D": "natural+free-text"}[k]
        print(f"  {k} {label:<28s}: {s['correct']:>2d}/{s['n']:<2d} = {s['accuracy']:.1%}  "
              f"(unparsed: {s['unparsed']})")
    c = summary["comparisons"]
    print(f"\n  Input effect  (A - B) = {c['input_effect_on_forced_letter_A_minus_B']:+.1%}")
    print(f"  Output effect (B - D) = {c['output_effect_on_natural_B_minus_D']:+.1%}")
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
