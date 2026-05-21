"""option_order_shuffle_4b.py -- option-order randomization experiment
for Gemma 3 4B IT, to distinguish position-bias from content-bias in
the "NL = B when gold = C" pattern observed at 4B.

Background: at 4B, the NF > NL accuracy gap is driven by 13 of 14
NF_only_right cases where NL = B and gold = C. The forced-letter
mode systematically defaults to B (which contains "See my doctor in
the next few weeks") when the gold answer is C ("See a doctor within
24-48 hours"). This could be:

  (a) A POSITION ARTIFACT: 4B has a learned bias toward emitting "B"
      (position-2 of A/B/C/D) when the prompt is ambiguous, regardless
      of which acuity content is mapped to "B".
  (b) A CONTENT-PRIOR ARTIFACT: 4B has a learned bias toward the
      acuity content "see doctor in next few weeks" when uncertain,
      regardless of which letter that content is assigned.

We can distinguish these by SHUFFLING the letter→content mapping in
the forced-letter scaffold and seeing what the model picks. For each
case, we generate K shuffles (random non-identity permutations) and
score:

  - is the picked LETTER stable across shuffles? (= position bias)
  - is the picked ACUITY CONTENT stable across shuffles? (= content bias)

The two hypotheses make opposite predictions; the experiment should
distinguish them cleanly on n=60 cases × K shuffles.

Usage:
  python3 paper/scripts/option_order_shuffle_4b.py --k 3 --device cuda
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "google/gemma-3-4b-it"
VIGNETTES = ROOT / "paper/data/canonical_forced_letter_vignettes.json"

# The canonical acuity content -> id (0=lowest, 3=highest)
CONTENTS = {
    0: "Fine to monitor at home",
    1: "See my doctor in the next few weeks",
    2: "See a doctor within 24-48 hours",
    3: "Go to the ER now",
}
# Original mapping: A=0, B=1, C=2, D=3 (A lowest, D highest)
ORIGINAL_LETTER_TO_CONTENT = {"A": 0, "B": 1, "C": 2, "D": 3}

LETTER_LINE_RE = re.compile(r"\b([ABCD])\b")


def extract_letter(text: str) -> str | None:
    if not text:
        return None
    head = text.lstrip()
    if head and head[0] in "ABCD":
        return head[0]
    m = LETTER_LINE_RE.search(text[:200])
    return m.group(1) if m else None


def build_shuffled_prompt(vignette_text: str, letter_to_content_id: dict[str, int]) -> str:
    """Replace the forced-letter scaffold with a custom letter→content mapping."""
    # Find the "Reply with exactly one letter only." marker
    # and replace the four bullet lines that follow
    # The original prompt structure:
    #   ...vignette ending with "?\n\n"
    #   Reply with exactly one letter only.\n
    #   A = Fine to monitor at home\n
    #   B = See my doctor in the next few weeks\n
    #   C = See a doctor within 24-48 hours\n
    #   D = Go to the ER now\n
    #   Do not include any explanation or extra words.

    # Extract everything up to "Reply with exactly one letter only."
    idx = vignette_text.find("Reply with exactly one letter only.")
    if idx == -1:
        raise ValueError("could not find scaffold marker in prompt")
    head = vignette_text[:idx]
    # Build the new scaffold
    bullets = "\n".join(f"{L} = {CONTENTS[letter_to_content_id[L]]}" for L in "ABCD")
    new_scaffold = (
        "Reply with exactly one letter only.\n"
        f"{bullets}\n"
        "Do not include any explanation or extra words."
    )
    return head + new_scaffold


def generate_shuffles(case_id: str, k: int, rng: np.random.Generator) -> list[dict]:
    """Generate k random non-identity permutations of letter->content."""
    perms = []
    seen = set()
    seen.add(("A", "B", "C", "D"))  # exclude identity
    max_tries = 100
    while len(perms) < k and max_tries > 0:
        max_tries -= 1
        order = rng.permutation(4).tolist()  # which content (0..3) goes to A,B,C,D
        # build letter -> content map
        mapping = {"A": order[0], "B": order[1], "C": order[2], "D": order[3]}
        # canonical signature
        sig = tuple(CONTENTS[order[i]][:3] for i in range(4))
        if (mapping["A"] == 0 and mapping["B"] == 1 and
            mapping["C"] == 2 and mapping["D"] == 3):
            continue  # identity, skip
        # also skip if we've already produced an identical shuffle
        if mapping_signature := tuple(mapping[L] for L in "ABCD"):
            if mapping_signature in seen:
                continue
            seen.add(mapping_signature)
        perms.append({
            "case_id": case_id,
            "shuffle_idx": len(perms),
            "letter_to_content_id": mapping,
            "letter_to_content_text": {L: CONTENTS[mapping[L]] for L in "ABCD"},
        })
    return perms


def gold_letters_under_mapping(gold_raw: str, mapping: dict[str, int]) -> list[str]:
    """The gold acuity is fixed (e.g. 'C/D' = contents 2 and 3). Under a
    shuffle, the LETTERS that map to those contents may be different."""
    gold_contents = set()
    for letter in re.findall(r"[ABCD]", gold_raw.upper()):
        gold_contents.add(ORIGINAL_LETTER_TO_CONTENT[letter])
    # Find letters in the new mapping that point to gold contents
    return sorted([L for L in "ABCD" if mapping[L] in gold_contents])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3,
                    help="Number of non-identity shuffles per case (default 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=10)
    args = ap.parse_args()

    vignettes = json.loads(VIGNETTES.read_text())
    rng = np.random.default_rng(args.seed)

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
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
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    rows = []
    for ci, v in enumerate(vignettes):
        case_id = v["id"]
        gold_raw = v["gold_standard_triage"]
        gold_contents = sorted({ORIGINAL_LETTER_TO_CONTENT[L]
                                for L in re.findall(r"[ABCD]", gold_raw.upper())})
        original_prompt = v["natural_forced_letter"]
        # First, baseline: run the model on the ORIGINAL (un-shuffled) prompt to
        # confirm we replicate the canonical NL pred
        out_orig = generate(original_prompt)
        pred_orig = extract_letter(out_orig)
        # The content the original prediction corresponds to:
        pred_orig_content_id = (ORIGINAL_LETTER_TO_CONTENT[pred_orig]
                                 if pred_orig in "ABCD" else None)

        # Generate K shuffles
        shuffles = generate_shuffles(case_id, args.k, rng)
        case_rows = []
        for shuf in shuffles:
            mapping = shuf["letter_to_content_id"]
            prompt = build_shuffled_prompt(original_prompt, mapping)
            out = generate(prompt)
            pred = extract_letter(out)
            pred_content_id = mapping[pred] if pred in "ABCD" else None
            gold_letters_now = gold_letters_under_mapping(gold_raw, mapping)
            case_rows.append({
                "shuffle_idx": shuf["shuffle_idx"],
                "letter_to_content": shuf["letter_to_content_text"],
                "pred_letter": pred,
                "pred_content_id": pred_content_id,
                "pred_content": CONTENTS[pred_content_id] if pred_content_id is not None else None,
                "gold_letters_under_shuffle": gold_letters_now,
                "correct_under_shuffle": pred in gold_letters_now if pred else False,
                "raw": out,
            })

        rows.append({
            "case_id": case_id,
            "gold_raw": gold_raw,
            "gold_content_ids": gold_contents,
            "original_pred_letter": pred_orig,
            "original_pred_content_id": pred_orig_content_id,
            "original_pred_content": (CONTENTS[pred_orig_content_id]
                                      if pred_orig_content_id is not None else None),
            "original_correct": (pred_orig_content_id in gold_contents
                                 if pred_orig_content_id is not None else False),
            "shuffles": case_rows,
        })
        print(f"  [{ci+1:>2}/60] {case_id:>4} gold={gold_raw:>4} orig={pred_orig}({CONTENTS[pred_orig_content_id][:8] if pred_orig_content_id is not None else '?'}) "
              + " ".join(f"sh{r['shuffle_idx']}={r['pred_letter']}({(CONTENTS[r['pred_content_id']][:8] if r['pred_content_id'] is not None else '?')})"
                         for r in case_rows))

    # ── Aggregate ──────────────────────────────────────────────────────
    n = len(rows)
    K = max(len(r["shuffles"]) for r in rows)

    # Position-bias signal: how often does the model pick the SAME LETTER
    # as in the original prompt, across shuffles?
    same_letter_count = 0
    same_letter_total = 0
    # Content-bias signal: how often does the model pick the SAME CONTENT
    # (acuity recommendation) as in the original prompt, across shuffles?
    same_content_count = 0
    same_content_total = 0

    # Overall pred-letter distribution across shuffles (regardless of content)
    letter_dist_orig = {L: 0 for L in "ABCD"}
    letter_dist_shuf = {L: 0 for L in "ABCD"}
    # Content-id distribution under shuffles
    content_dist_orig = {0:0, 1:0, 2:0, 3:0}
    content_dist_shuf = {0:0, 1:0, 2:0, 3:0}

    for r in rows:
        if r["original_pred_letter"] in "ABCD":
            letter_dist_orig[r["original_pred_letter"]] += 1
        if r["original_pred_content_id"] is not None:
            content_dist_orig[r["original_pred_content_id"]] += 1
        for sh in r["shuffles"]:
            if sh["pred_letter"] in "ABCD":
                letter_dist_shuf[sh["pred_letter"]] += 1
            if sh["pred_content_id"] is not None:
                content_dist_shuf[sh["pred_content_id"]] += 1
            same_letter_total += 1
            same_content_total += 1
            if sh["pred_letter"] == r["original_pred_letter"]:
                same_letter_count += 1
            if (sh["pred_content_id"] is not None and
                r["original_pred_content_id"] is not None and
                sh["pred_content_id"] == r["original_pred_content_id"]):
                same_content_count += 1

    # Accuracy under shuffle (acuity-content-correctness)
    n_orig_correct = sum(1 for r in rows if r["original_correct"])
    n_shuf_correct = sum(1 for r in rows for sh in r["shuffles"]
                          if sh["correct_under_shuffle"])
    n_shuf_total = sum(len(r["shuffles"]) for r in rows)

    summary = {
        "model": MODEL_ID,
        "n_cases": n,
        "K_shuffles_per_case": K,
        "n_shuffle_total": n_shuf_total,
        "stability_signals": {
            "same_letter_under_shuffle": same_letter_count,
            "same_letter_total":          same_letter_total,
            "same_letter_frac":           same_letter_count / max(1, same_letter_total),
            "same_content_under_shuffle": same_content_count,
            "same_content_total":          same_content_total,
            "same_content_frac":          same_content_count / max(1, same_content_total),
            "interpretation_position_bias": f"{same_letter_count / max(1,same_letter_total):.1%} same letter (high = position bias)",
            "interpretation_content_bias":  f"{same_content_count / max(1,same_content_total):.1%} same content (high = content prior)",
        },
        "letter_distribution_original_NL": letter_dist_orig,
        "letter_distribution_shuffles":    letter_dist_shuf,
        "content_distribution_original_NL": {CONTENTS[k]: v for k, v in content_dist_orig.items()},
        "content_distribution_shuffles":    {CONTENTS[k]: v for k, v in content_dist_shuf.items()},
        "accuracy": {
            "n_correct_original":         n_orig_correct,
            "n_total_original":           n,
            "n_correct_under_shuffle":    n_shuf_correct,
            "n_total_shuffles":           n_shuf_total,
            "original_accuracy_pct":      100 * n_orig_correct / n,
            "shuffled_accuracy_pct":      100 * n_shuf_correct / max(1, n_shuf_total),
        },
        "per_case": rows,
    }

    out_path = ROOT / "results/option_order_shuffle_4b.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_path}")
    print()
    print("=== Stability signals ===")
    print(f"  Same letter under shuffle:  {same_letter_count}/{same_letter_total} "
          f"= {100*same_letter_count/max(1,same_letter_total):.1f}%  (high → position bias)")
    print(f"  Same content under shuffle: {same_content_count}/{same_content_total} "
          f"= {100*same_content_count/max(1,same_content_total):.1f}%  (high → content prior)")
    print()
    print(f"  Letter dist (original 60):  {letter_dist_orig}")
    print(f"  Letter dist (shuffles):     {letter_dist_shuf}")
    print(f"  Content dist (original 60): {[(CONTENTS[k][:6],v) for k,v in content_dist_orig.items()]}")
    print(f"  Content dist (shuffles):    {[(CONTENTS[k][:6],v) for k,v in content_dist_shuf.items()]}")
    print()
    print(f"  Original NL accuracy:    {n_orig_correct}/{n} = {100*n_orig_correct/n:.1f}%")
    print(f"  Shuffled NL accuracy:    {n_shuf_correct}/{n_shuf_total} = {100*n_shuf_correct/max(1,n_shuf_total):.1f}%")


if __name__ == "__main__":
    main()
