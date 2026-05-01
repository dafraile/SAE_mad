"""
Phase 0 — Capability floor gate for the SAE-as-detector mechanistic experiment.

Question: Is Gemma 3 4B IT good enough at the paper's original triage task
that any downstream mechanistic claim about its internal representations is
meaningful?

Method: run the model on all 60 canonical paper-faithful structured prompts
(forced A/B/C/D output), extract the letter, score against gold standard.

Benchmarks (for interpretation, decided before running):
  - Paper-reported frontier models: 82-85%
  - <40%: catastrophic - abort or escalate to Gemma 3 12B
  - 40-70%: weak but interpretable, proceed with caveats
  - >70%: matches frontier, proceed cleanly

Output: results/phase0_capability_floor.json
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-4b-it"
VIGNETTES_PATH = Path(
    "nature_triage_expanded_replication/paper_faithful_replication/data/"
    "canonical_singleturn_vignettes.json"
)
OUT_PATH = Path("results/phase0_capability_floor.json")
MAX_NEW_TOKENS = 300

TRIAGE_LINE_RE = re.compile(r"TRIAGE\s*:\s*([ABCD])\b", re.IGNORECASE)


def parse_gold(gold: str) -> list[str]:
    # Accept "C", "C/D", "B/C", etc. Return list of acceptable letters.
    gold = gold.upper().strip()
    letters = re.findall(r"[ABCD]", gold)
    return sorted(set(letters))


def extract_letter(output_text: str) -> str | None:
    m = TRIAGE_LINE_RE.search(output_text)
    if m:
        return m.group(1).upper()
    # Fallback: look for the first bare A/B/C/D on its own line
    for line in output_text.splitlines():
        s = line.strip()
        if s in ("A", "B", "C", "D"):
            return s
    return None


def main() -> None:
    vignettes = json.loads(VIGNETTES_PATH.read_text())
    assert len(vignettes) == 60, f"Expected 60 canonical vignettes, got {len(vignettes)}"

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    results = []
    t0 = time.time()
    for i, v in enumerate(vignettes):
        prompt = v["original_structured"]
        gold = parse_gold(v["gold_standard_triage"])

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
                temperature=None,
                top_p=None,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        letter = extract_letter(gen)
        correct = (letter in gold) if letter else False

        results.append({
            "id": v["id"],
            "title": v["title"],
            "gold_raw": v["gold_standard_triage"],
            "gold_letters": gold,
            "predicted": letter,
            "correct": correct,
            "unparsed": letter is None,
            "raw_output": gen,
        })
        print(
            f"[{i+1:2d}/60] {v['id']:>4s}  gold={'/'.join(gold):<4s}  "
            f"pred={letter or '??':<2s}  {'OK' if correct else 'x '}  "
            f"({time.time()-t0:.0f}s)"
        )

    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    n_unparsed = sum(r["unparsed"] for r in results)

    # Per-gold-category breakdown
    from collections import Counter
    by_gold = Counter()
    by_gold_correct = Counter()
    for r in results:
        key = "/".join(r["gold_letters"]) or "??"
        by_gold[key] += 1
        if r["correct"]:
            by_gold_correct[key] += 1

    summary = {
        "model": MODEL_ID,
        "n_total": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n,
        "n_unparsed": n_unparsed,
        "by_gold": {
            k: {"n": by_gold[k], "correct": by_gold_correct[k],
                "acc": by_gold_correct[k] / by_gold[k] if by_gold[k] else 0}
            for k in sorted(by_gold)
        },
        "results": results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print(f"\n=== Phase 0 capability floor ===")
    print(f"Model: {MODEL_ID}")
    print(f"Accuracy: {n_correct}/{n} = {n_correct/n:.1%}")
    print(f"Unparsed: {n_unparsed}/{n}")
    print(f"By gold category:")
    for k in sorted(summary["by_gold"]):
        row = summary["by_gold"][k]
        print(f"  {k:<5s}: {row['correct']:>2d}/{row['n']:<2d} = {row['acc']:.0%}")
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
