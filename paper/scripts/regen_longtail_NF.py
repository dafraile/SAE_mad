"""regen_longtail_NF.py -- rerun only the NF (D-cell) generations for case
IDs whose responses still hit the truncation cap in the previous regen.

Reads the existing phase0_5_three_cells.json (4B) or phase3b_12b_phase0_5.json
(12B), identifies cases where the NF response ends mid-sentence, regenerates
just those cells at a higher max_new_tokens, and writes the patched file back.

Usage (from project root):
    python3 paper/scripts/regen_longtail_NF.py \\
        --model google/gemma-3-4b-it \\
        --input results/phase0_5_three_cells.json \\
        --output results/phase0_5_three_cells.json \\
        --max-new-tokens 4000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def is_truncated(raw: str) -> bool:
    """Heuristic: response ends in a non-sentence-terminating char."""
    return raw.rstrip()[-1] not in '.!?:"\'\n'


def parse_letter(raw: str) -> tuple[str | None, str]:
    """Mirror of the parsing logic from phase0_5_three_cells.py / phase3b
    for the D-cell heuristic letter (NF is LLM-judge adjudicated downstream,
    but we also write the heuristic in for completeness)."""
    cleaned = raw.strip()
    # Look for first standalone A/B/C/D letter in early text
    m = re.search(r"\b([ABCD])\b", cleaned[:200])
    if m:
        return m.group(1), "regex"
    return None, "no_match"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="HF model id, e.g. google/gemma-3-4b-it")
    parser.add_argument("--input", required=True,
                        help="Input JSON (phase0_5_three_cells or phase3b_12b_phase0_5)")
    parser.add_argument("--output", required=True,
                        help="Output JSON (can be same as input to overwrite)")
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    parser.add_argument("--singleturn-vignettes", default=(
        "nature_triage_expanded_replication/paper_faithful_replication/data/"
        "canonical_singleturn_vignettes.json"))
    args = parser.parse_args()

    print(f"[regen-longtail] reading {args.input}")
    data = json.loads(Path(args.input).read_text())
    rows = data["results"]

    truncated = [r for r in rows if is_truncated(r["D"]["raw"])]
    truncated_ids = [r["id"] for r in truncated]
    print(f"  found {len(truncated)} truncated NF responses: {truncated_ids}")

    if not truncated:
        print("  nothing to do.")
        return

    # Load singleturn vignettes for the prompts
    vignettes = json.loads(Path(args.singleturn_vignettes).read_text())
    vignette_by_id = {v["id"]: v for v in vignettes}

    print(f"\n[regen-longtail] loading {args.model} on cuda (bf16)")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    print(f"[regen-longtail] regenerating with max_new_tokens={args.max_new_tokens}")
    t0 = time.time()
    for i, row in enumerate(truncated):
        cid = row["id"]
        v = vignette_by_id[cid]
        prompt = v["patient_realistic"]
        input_ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=False,
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to(model.device)
        n_in = input_ids.shape[-1]

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen_ids = out[0, n_in:]
        raw = tok.decode(gen_ids, skip_special_tokens=True)

        letter, reason = parse_letter(raw)
        # Recompute correctness vs gold
        gold_letters = sorted(set(re.findall(r"[ABCD]", row["gold_raw"].upper())))
        correct = letter in gold_letters if letter else False

        old_len = len(row["D"]["raw"])
        new_len = len(raw)
        still_truncated = is_truncated(raw)
        print(f"  [{i+1}/{len(truncated)}] {cid}: {old_len} -> {new_len} chars"
              f"{' (STILL TRUNC)' if still_truncated else ''} "
              f"letter={letter} correct={correct}")

        # Patch the row in-place
        row["D"]["raw"] = raw
        row["D"]["predicted"] = letter
        row["D"]["correct"] = correct
        row["D"]["parse_reason"] = reason
        row["D"]["unparsed"] = letter is None

    print(f"\n[regen-longtail] done in {(time.time()-t0)/60:.1f} min")
    print(f"  writing to {args.output}")
    # Recompute the summary block if present (heuristic D-cell only -- LLM judge
    # adjudication is downstream so doesn't need updating here).
    Path(args.output).write_text(json.dumps(data, indent=2))
    print("[regen-longtail] OK")


if __name__ == "__main__":
    main()
