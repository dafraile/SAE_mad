"""
Phase 4b -- Qwen3-8B behavioral 4-cell evaluation.

Mirrors phase0_5_three_cells.py (4B) and the 12B equivalent in
phase3b_12b_pipeline.py, but for Qwen3-8B (post-trained, same model the
Phase 4 mechanistic analysis uses).

Three cells per case:
  A: structured + forced-letter  (SL)
  B: natural    + forced-letter  (NL)
  D: natural    + free-text      (NF)

Decoding: greedy, max_new_tokens=2000 (matches v2 4B/12B runs).

Output: results/phase4b_qwen_behavioral.json (parallel to
results/_v2/phase0_5_three_cells.json for 4B and
results/_v2/phase3b_12b_phase0_5.json for 12B).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3-8B"

FORCED_LETTER_PATH = ROOT / "paper/data/canonical_forced_letter_vignettes.json"
SINGLETURN_PATH    = ROOT / "paper/data/canonical_singleturn_vignettes.json"
OUT_PATH           = ROOT / "results/phase4b_qwen_behavioral.json"
MAX_NEW_TOKENS = 2000

LETTER_LINE_RE = re.compile(r"\b([ABCD])\b")

# Reused parsing heuristics from phase0_5_three_cells.py for the
# free-text best-effort letter pick (NF responses get judged by LLM
# downstream; this is just a fallback / sanity check).
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


def extract_forced_letter(text: str) -> str | None:
    """Take first ABCD that appears on its own or as obvious letter answer."""
    if not text: return None
    head = text.lstrip()
    if head and head[0] in "ABCD":
        return head[0]
    m = LETTER_LINE_RE.search(text[:200])
    return m.group(1) if m else None


def extract_free_text(text: str) -> tuple[str | None, str]:
    """Best-effort triage letter from free-text."""
    if not text: return None, "empty"
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


def build_cells():
    fl = json.loads(FORCED_LETTER_PATH.read_text())
    st = json.loads(SINGLETURN_PATH.read_text())
    fl_by_id = {v["id"]: v for v in fl}
    st_by_id = {v["id"]: v for v in st}
    def _key(s):
        m = re.match(r"^(\D+)(\d+)$", s)
        return (m.group(1), int(m.group(2))) if m else (s, 0)
    cells = []
    for cid in sorted(fl_by_id, key=_key):
        fl_row, st_row = fl_by_id[cid], st_by_id[cid]
        cells.append({
            "id": cid, "title": fl_row["title"],
            "gold_raw": fl_row["gold_standard_triage"],
            "gold_letters": parse_gold(fl_row["gold_standard_triage"]),
            "A_prompt": fl_row["structured_forced_letter"],
            "B_prompt": fl_row["natural_forced_letter"],
            "D_prompt": st_row["patient_realistic"],
        })
    return cells


def main():
    cases = build_cells()
    assert len(cases) == 60

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    def generate(prompt: str) -> str:
        # Qwen3 uses its own chat template via apply_chat_template
        messages = [{"role": "user", "content": prompt}]
        # enable_thinking=False to suppress Qwen3's <think>...</think>
        # reasoning trace and get a direct answer
        try:
            ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=False, enable_thinking=False,
            )
        except TypeError:
            # older transformers may not accept enable_thinking kwarg
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
        row = {"id": c["id"], "title": c["title"],
               "gold_raw": c["gold_raw"], "gold_letters": c["gold_letters"]}
        out_a = generate(c["A_prompt"])
        pred_a = extract_forced_letter(out_a)
        row["A"] = {"predicted": pred_a,
                    "correct": (pred_a in c["gold_letters"]) if pred_a else False,
                    "unparsed": pred_a is None, "raw": out_a}
        out_b = generate(c["B_prompt"])
        pred_b = extract_forced_letter(out_b)
        row["B"] = {"predicted": pred_b,
                    "correct": (pred_b in c["gold_letters"]) if pred_b else False,
                    "unparsed": pred_b is None, "raw": out_b}
        out_d = generate(c["D_prompt"])
        pred_d, reason_d = extract_free_text(out_d)
        row["D"] = {"predicted": pred_d,
                    "correct": (pred_d in c["gold_letters"]) if pred_d else False,
                    "unparsed": pred_d is None, "parse_reason": reason_d,
                    "raw": out_d}
        results.append(row)
        elapsed = time.time() - t0
        print(f"[{i+1:2d}/60] {c['id']:>4s}  gold={'/'.join(c['gold_letters']):<4s}  "
              f"A={pred_a or '??'}{'✓' if row['A']['correct'] else '✗'}  "
              f"B={pred_b or '??'}{'✓' if row['B']['correct'] else '✗'}  "
              f"D={pred_d or '??'}{'✓' if row['D']['correct'] else '✗'}  "
              f"({elapsed:.0f}s)")

    def agg(cell: str):
        n = len(results)
        correct = sum(r[cell]["correct"] for r in results)
        unparsed = sum(r[cell]["unparsed"] for r in results)
        return {"n": n, "correct": correct, "unparsed": unparsed,
                "accuracy": correct / n}

    summary = {
        "model": MODEL_ID,
        "max_new_tokens": MAX_NEW_TOKENS,
        "cells": {"A": agg("A"), "B": agg("B"), "D": agg("D")},
        "comparisons": {
            "input_effect_on_forced_letter_A_minus_B":
                agg("A")["accuracy"] - agg("B")["accuracy"],
            "output_effect_on_natural_B_minus_D_heuristic":
                agg("B")["accuracy"] - agg("D")["accuracy"],
        },
        "results": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print()
    print("=== Phase 4b Qwen3-8B behavioral summary ===")
    print(f"Model: {MODEL_ID}")
    for k in ["A", "B", "D"]:
        a = agg(k)
        print(f"  {k}: {a['correct']}/60 = {a['accuracy']:.1%} (unparsed {a['unparsed']})")
    print(f"  A-B input effect:        {summary['comparisons']['input_effect_on_forced_letter_A_minus_B']:+.1%}")
    print(f"  B-D output effect (heur): {summary['comparisons']['output_effect_on_natural_B_minus_D_heuristic']:+.1%}")
    print(f"  Wall time: {(time.time()-t0)/60:.1f} min")
    print(f"  Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
