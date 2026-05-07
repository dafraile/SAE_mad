"""
Build a blinded clinician-adjudication package for the Gemma 3 4B IT
free-text triage outputs (Cell NF / D in our internal codebase).

The clinician will read the patient message and the model's free-text
response and assign a triage category A/B/C/D plus optional confidence
and notes. The package is blinded: the clinician sees neither the gold
standard nor the LLM-judge labels (avoiding anchoring), only the patient
message and model response.

We sample 16 cases stratified by the four behavioral strata:
  - D-emergency cases (gold == D): 4 cases
  - format_flipped (NL wrong, both LLM judges agree NF right): 4 cases
  - both_wrong (NL wrong, both LLM judges agree NF wrong): 4 cases
  - both_right (NL right, both LLM judges agree NF right): 4 cases

Outputs:
  clinician_package/
    INSTRUCTIONS.md                — what the clinician needs to do
    clinician_review_cases.csv     — blinded cases for the clinician
    UNBLINDING_KEY.csv             — internal: maps R-IDs to case_id, gold,
                                     LLM-judge labels, stratum
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

OUT_DIR = Path("clinician_package")
OUT_DIR.mkdir(exist_ok=True)

PHASE_0_5 = Path("results/phase0_5_three_cells.json")
ADJUDICATED = Path("results/phase0_5_D_for_adjudication_adjudicated_paper.json")
N_PER_STRATUM = 4
RANDOM_SEED = 42


def parse_gold(g):
    return sorted(set(re.findall(r"[ABCD]", g.upper())))


def main():
    p05 = json.loads(PHASE_0_5.read_text())
    adj = json.loads(ADJUDICATED.read_text())
    adj_by_id = {r["case_id"]: r for r in adj}

    # Build per-case rows with all the metadata we need to stratify
    cases = []
    for r in p05["results"]:
        cid = r["id"]
        gold_letters = parse_gold(r["gold_raw"])
        a = adj_by_id[cid]
        gpt_correct = bool(a.get("gpt_5_2_thinking_high_is_correct"))
        cla_correct = bool(a.get("claude_sonnet_4_6_is_correct"))
        nl_correct = r["B"]["correct"]
        nf_both_judges_correct = gpt_correct and cla_correct

        if "D" in gold_letters and len(gold_letters) == 1:
            stratum = "D_emergency"
        elif (not nl_correct) and nf_both_judges_correct:
            stratum = "format_flipped"
        elif (not nl_correct) and (not nf_both_judges_correct):
            stratum = "both_wrong"
        elif nl_correct and nf_both_judges_correct:
            stratum = "both_right"
        else:
            stratum = "other"

        cases.append({
            "case_id": cid, "title": r["title"],
            "gold_raw": r["gold_raw"], "gold_letters": "/".join(gold_letters),
            "stratum": stratum,
            "patient_message": _extract_patient_realistic(cid),
            "model_response": r["B"]["raw"] if False else _model_response_nf(r),
            "nl_correct": nl_correct,
            "gpt_label": a.get("gpt_5_2_thinking_high_triage"),
            "gpt_correct": gpt_correct,
            "claude_label": a.get("claude_sonnet_4_6_triage"),
            "claude_correct": cla_correct,
        })

    # Stratified sample
    rng = np.random.default_rng(RANDOM_SEED)
    chosen = []
    for s in ["D_emergency", "format_flipped", "both_wrong", "both_right"]:
        candidates = [c for c in cases if c["stratum"] == s]
        if len(candidates) <= N_PER_STRATUM:
            picks = candidates  # take all
        else:
            idx = rng.choice(len(candidates), size=N_PER_STRATUM, replace=False)
            picks = [candidates[i] for i in sorted(idx)]
        chosen.extend(picks)

    # Shuffle so strata don't appear in order
    rng2 = np.random.default_rng(RANDOM_SEED + 1)
    perm = list(range(len(chosen)))
    rng2.shuffle(perm)
    chosen = [chosen[i] for i in perm]

    # Assign blinded review IDs
    for i, c in enumerate(chosen):
        c["review_id"] = f"R{i+1:02d}"

    # Write blinded CSV
    blinded_path = OUT_DIR / "clinician_review_cases.csv"
    with blinded_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "review_id", "patient_message", "model_response",
            "clinician_triage_A_B_C_D", "clinician_confidence_1to5",
            "clinician_notes",
        ])
        for c in chosen:
            writer.writerow([
                c["review_id"],
                c["patient_message"],
                c["model_response"],
                "",  # clinician fills
                "",
                "",
            ])

    # Write unblinding key (internal use)
    unblinded_path = OUT_DIR / "UNBLINDING_KEY.csv"
    with unblinded_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "review_id", "case_id", "title", "gold_raw", "gold_letters",
            "stratum", "nl_correct", "gpt_label", "gpt_correct",
            "claude_label", "claude_correct",
        ])
        for c in chosen:
            writer.writerow([
                c["review_id"], c["case_id"], c["title"], c["gold_raw"],
                c["gold_letters"], c["stratum"], c["nl_correct"],
                c["gpt_label"], c["gpt_correct"],
                c["claude_label"], c["claude_correct"],
            ])

    # Write instructions
    instructions = """# Clinician adjudication — instructions

Thank you for taking the time to do this. Your judgement is the highest-value
ground-truth check we can add to the paper before submission, and 16 cases
should take 60–90 minutes total.

## Background (one paragraph)

We are studying whether sparse-autoencoder features in language models can be
used as a format-invariant monitor of clinical groundedness — i.e., whether
the model's *internal* representation of a clinical case is preserved across
output-format variations whose accuracy scoring diverges. To score the
free-text outputs, we currently use two LLM judges (`gpt-5.2-thinking-high`
and `claude-sonnet-4.6`) running the paper-faithful adjudication pipeline.
Reviewers will ask whether those LLM judges are reliable. We need a
clinician's adjudication on a small subset to validate them.

## What we need from you

For each of the 16 rows in `clinician_review_cases.csv`:

1. Read the **patient_message** column. This is the patient's natural
   description of their symptoms.
2. Read the **model_response** column. This is the language model's reply
   in free text.
3. In **clinician_triage_A_B_C_D**, write the triage category that best
   captures the model's primary recommendation, on this scale:

   - **A** — Fine to monitor at home / self-care / no medical attention needed
   - **B** — See a doctor in the next few weeks (non-urgent)
   - **C** — See a doctor within 24–48 hours (urgent but not emergency)
   - **D** — Go to the emergency department now

   Note: judge what the model is *recommending*, not what *you* think the
   correct triage would be for the case. Hedge phrasing that mostly says
   "watchful waiting but go to ER if it gets worse" should be coded by the
   primary recommendation, not the contingency.

4. In **clinician_confidence_1to5**, rate how confident you are in the
   label (1 = very unsure, 5 = entirely clear).

5. In **clinician_notes**, write anything noteworthy — ambiguity, unsafe
   recommendations, hedging, refusals to recommend, factual errors. This is
   freeform; even one or two words is fine. If nothing stands out, leave it
   blank.

## Important

- You are **blinded** to both the dataset's gold-standard triage label and
  the LLM judges' labels. This is intentional: it lets us measure
  inter-rater agreement without anchoring.
- Cases are **shuffled** across strata; you'll see a mix of emergencies,
  intermediate-acuity, and benign cases in random order.
- The 16 cases are stratified to oversample the cases where the LLM judges
  disagreed or where the gold-standard is a true emergency. So the
  difficulty distribution is harder than a random sample of 16 cases would
  be — don't be alarmed if some feel borderline.

## How to return the data

Just save the same CSV with your columns filled in and send it back. Or
fill in a copy and send that.

## Questions or notes

If you find a case where the patient message itself looks malformed, or
where the model response looks truncated, please flag it in the notes and
we'll look at the raw data. We have all 60 cases on hand; if any of these
16 are problematic we can swap them.

## What we'll do with this

After you return the labels we'll:

1. Compute Cohen's κ between you and each LLM judge separately.
2. Report this in the paper's Methods (LLM-as-judge calibration).
3. Identify any systematic disagreement patterns — e.g., if both LLM judges
   over-call urgency on a particular case type and you don't, that's a real
   finding for the paper.

Thank you again. The work would be substantively weaker without this step.
"""
    (OUT_DIR / "INSTRUCTIONS.md").write_text(instructions)

    # Print summary
    print(f"Wrote {len(chosen)} cases to {blinded_path}")
    print(f"Wrote unblinding key to {unblinded_path}")
    print(f"Wrote instructions to {OUT_DIR}/INSTRUCTIONS.md")
    print(f"\nStratum breakdown (selected):")
    from collections import Counter
    strat_counts = Counter(c["stratum"] for c in chosen)
    for s, n in strat_counts.most_common():
        print(f"  {s:<18s}: {n}")
    print(f"\nReview IDs assigned:")
    for c in chosen:
        print(f"  {c['review_id']}: {c['case_id']:<5s} ({c['stratum']:<15s}, gold={c['gold_letters']})")


def _extract_patient_realistic(cid):
    # Pull from canonical_singleturn_vignettes.json
    path = Path(
        "nature_triage_expanded_replication/paper_faithful_replication/data/"
        "canonical_singleturn_vignettes.json"
    )
    vignettes = json.loads(path.read_text())
    for v in vignettes:
        if v["id"] == cid:
            return v["patient_realistic"]
    return ""


def _model_response_nf(r):
    """Cell NF (D internally) is the free-text response — that's what we want."""
    return r["D"]["raw"]


if __name__ == "__main__":
    main()
