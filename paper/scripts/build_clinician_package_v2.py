"""build_clinician_package_v2.py -- rebuild the blinded clinician-adjudication
package using the corrected (untruncated NF responses, 5-way LLM judges with
DEFERRED) data in results/_v2/.

Stratification reflects the refined research question: we want Jia's
adjudication to bear on the *interpretability* claim, not just triage
accuracy. The 16 cases are stratified into four strata of 4 each:

  1. **12B both-deferred** (n=4): 12B's L32 representation supports tiered
     clinical reasoning; the model's free-text response asks the patient
     for missing data before committing to an urgency. Jia rates: is the
     deferral clinically appropriate? Implicit triage if forced?
  2. **Clear-correct** (n=4): 12B both judges agree, in {C, D}, correct.
     Sanity: do you also agree?
  3. **Clear-incorrect** (n=4): 12B both judges agree, in a letter not in
     gold. Sanity: was the model genuinely wrong, or did the LLM judge
     misread its recommendation?
  4. **Format-flipped at 4B** (n=4): 4B's NL letter was wrong, but its NF
     free-text was rated correct by both judges (after the v2 regen). This
     is the original "format penalty" stratum that motivated the paper.

Outputs go to `clinician_package_v2/` so the original package directory
stays untouched.
"""
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "clinician_package_v2"
OUT_DIR.mkdir(exist_ok=True)

P05_4B = ROOT / "results/_v2/phase0_5_three_cells.json"
P05_12B = ROOT / "results/_v2/phase3b_12b_phase0_5.json"
ADJ_4B_5WAY = ROOT / "results/_v2/phase0_5_adjudicated_deferred.json"
ADJ_12B_5WAY = ROOT / "results/_v2/phase3b_12b_adjudicated_deferred.json"
VIGNETTES = ROOT / "nature_triage_expanded_replication/paper_faithful_replication/data/canonical_singleturn_vignettes.json"

N_PER_STRATUM = 4
RANDOM_SEED = 42


def gold_letters(s: str) -> set[str]:
    return set(re.findall(r"[ABCD]", s.upper()))


def main():
    p05_4b = json.loads(P05_4B.read_text())
    p05_12b = json.loads(P05_12B.read_text())
    adj_4b = json.loads(ADJ_4B_5WAY.read_text())
    adj_12b = json.loads(ADJ_12B_5WAY.read_text())
    vignettes = {v["id"]: v for v in json.loads(VIGNETTES.read_text())}

    # Per-case adjudication labels
    def labels_by_case(adj_data, judges):
        out = {}
        for j in adj_data["judgments"]:
            out.setdefault(j["case_id"], {})[j["judge"]] = j
        return out

    lab_4b = labels_by_case(adj_4b, adj_4b["judges"])
    lab_12b = labels_by_case(adj_12b, adj_12b["judges"])

    # Build per-case info for 12B (deferred + clear strata come from here)
    cases_12b = []
    for r in p05_12b["results"]:
        cid = r["id"]
        gold = r["gold_raw"]; gold_set = gold_letters(gold)
        ja = lab_12b[cid]
        gpt = ja.get("gpt-5.2-thinking-high", {})
        cla = ja.get("claude-sonnet-4.6", {})
        gpt_t = gpt.get("triage")
        cla_t = cla.get("triage")
        gpt_def = (gpt_t == "DEFERRED")
        cla_def = (cla_t == "DEFERRED")
        gpt_correct = (gpt_t in gold_set) if not gpt_def else None
        cla_correct = (cla_t in gold_set) if not cla_def else None
        cases_12b.append({
            "model": "gemma-3-12b-it", "case_id": cid, "title": r["title"],
            "gold_raw": gold, "gold_letters": "/".join(sorted(gold_set)),
            "patient_message": vignettes[cid]["patient_realistic"],
            "model_response": r["D"]["raw"],
            "nl_correct": r["B"]["correct"],
            "gpt_label": gpt_t, "claude_label": cla_t,
            "gpt_correct": gpt_correct, "claude_correct": cla_correct,
            "both_deferred": gpt_def and cla_def,
            "both_clear_correct": (gpt_t == cla_t and gpt_t in gold_set and not gpt_def),
            "both_clear_incorrect": (gpt_t == cla_t and not gpt_def and not cla_def and gpt_t not in gold_set),
        })

    # Per-case info for 4B (format-flipped stratum)
    cases_4b = []
    for r in p05_4b["results"]:
        cid = r["id"]
        gold = r["gold_raw"]; gold_set = gold_letters(gold)
        ja = lab_4b[cid]
        gpt = ja.get("gpt-5.2-thinking-high", {})
        cla = ja.get("claude-sonnet-4.6", {})
        gpt_t = gpt.get("triage"); cla_t = cla.get("triage")
        gpt_def = (gpt_t == "DEFERRED")
        cla_def = (cla_t == "DEFERRED")
        nf_clear_correct = (
            not gpt_def and not cla_def and
            gpt_t in gold_set and cla_t in gold_set
        )
        cases_4b.append({
            "model": "gemma-3-4b-it", "case_id": cid, "title": r["title"],
            "gold_raw": gold, "gold_letters": "/".join(sorted(gold_set)),
            "patient_message": vignettes[cid]["patient_realistic"],
            "model_response": r["D"]["raw"],
            "nl_correct": r["B"]["correct"],
            "gpt_label": gpt_t, "claude_label": cla_t,
            "gpt_correct": (gpt_t in gold_set) if not gpt_def else None,
            "claude_correct": (cla_t in gold_set) if not cla_def else None,
            "nf_clear_correct": nf_clear_correct,
            "format_flipped": (not r["B"]["correct"]) and nf_clear_correct,
        })

    # Stratum pools
    pool_def     = [c for c in cases_12b if c["both_deferred"]]
    pool_corr    = [c for c in cases_12b if c["both_clear_correct"]]
    pool_incorr  = [c for c in cases_12b if c["both_clear_incorrect"]]
    pool_flipped = [c for c in cases_4b  if c["format_flipped"]]

    print(f"Pool sizes:")
    print(f"  12B both-deferred:        {len(pool_def)} candidates {[c['case_id'] for c in pool_def]}")
    print(f"  12B both clear-correct:   {len(pool_corr)} candidates")
    print(f"  12B both clear-incorrect: {len(pool_incorr)} candidates")
    print(f"  4B  format-flipped:       {len(pool_flipped)} candidates")

    rng = np.random.default_rng(RANDOM_SEED)
    def sample(pool, n, stratum):
        if len(pool) <= n:
            chosen = list(pool)
        else:
            idx = rng.choice(len(pool), size=n, replace=False)
            chosen = [pool[i] for i in sorted(idx)]
        for c in chosen:
            c["stratum"] = stratum
        return chosen

    chosen = (
        sample(pool_def,    N_PER_STRATUM, "12B_both_deferred")
        + sample(pool_corr, N_PER_STRATUM, "12B_clear_correct")
        + sample(pool_incorr, N_PER_STRATUM, "12B_clear_incorrect")
        + sample(pool_flipped, N_PER_STRATUM, "4B_format_flipped")
    )

    # Shuffle so strata don't appear in order
    rng2 = np.random.default_rng(RANDOM_SEED + 1)
    perm = list(range(len(chosen)))
    rng2.shuffle(perm)
    chosen = [chosen[i] for i in perm]

    # Assign blinded review IDs
    for i, c in enumerate(chosen):
        c["review_id"] = f"R{i+1:02d}"

    # Blinded CSV
    blinded_path = OUT_DIR / "clinician_review_cases.csv"
    with blinded_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "review_id", "patient_message", "model_response",
            "clinician_triage_A_B_C_D_or_DEFERRED",
            "clinician_deferral_appropriate_yes_no_NA",
            "clinician_confidence_1to5",
            "clinician_agrees_with_model_yes_no_partial",
            "clinician_notes",
        ])
        for c in chosen:
            w.writerow([c["review_id"], c["patient_message"], c["model_response"],
                        "", "", "", "", ""])

    # Unblinding key
    key_path = OUT_DIR / "UNBLINDING_KEY.csv"
    with key_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "review_id", "model", "case_id", "title", "gold_raw", "gold_letters",
            "stratum", "nl_correct",
            "gpt_label", "gpt_correct", "claude_label", "claude_correct",
        ])
        for c in chosen:
            w.writerow([
                c["review_id"], c["model"], c["case_id"], c["title"],
                c["gold_raw"], c["gold_letters"], c["stratum"], c["nl_correct"],
                c["gpt_label"], c.get("gpt_correct"), c["claude_label"], c.get("claude_correct"),
            ])

    # Instructions
    instructions = """# Clinician adjudication v2 — instructions for Jia

Thanks for sticking with this. **Important update:** the cases in the
previous package were truncated mid-response (a bug on our end). All 16
cases in **this** package have full, untruncated model responses. The work
you did on the truncated version is still useful as a sanity check, so
please send what you have — but you do not need to revisit those.

## What changed since v1

- All model responses are now full-length (mean ~5,900 chars for the 12B
  model, ~4,100 chars for 4B; up from ~1,750 in the truncated v1).
- We added a new triage option, **DEFERRED**, for cases where the model
  declines to commit to a single urgency level and instead asks the
  patient for clarifying information before deciding. We discovered that
  the larger 12B model does this on roughly 7% of cases, and the four
  rows in this package that came from that stratum are deliberately
  included so we can validate that behavior.
- We added a new question — **clinician_agrees_with_model_yes_no_partial**
  — asking whether the model's recommendation matches what you would
  recommend in this case. This is independent of the triage label you
  assign to the model in the first column.

## The research question (one paragraph)

This paper studies what sparse-autoencoder (SAE) features tell us about
*why* language models behave differently under forced-letter triage
prompts than under free-text prompts. The internal-representation story
is the main finding; the triage-accuracy numbers are context for that
story, not the headline. We need your adjudication on a small subset to
(a) calibrate our LLM judges, and (b) tell us whether the model's
"deferral" responses on tiered-decision cases are clinically appropriate
or whether the model is just stalling. Your judgement is the highest-
value ground-truth check we can include before submission.

## What we need from you

For each of the 16 rows in `clinician_review_cases.csv`:

### 1. clinician_triage_A_B_C_D_or_DEFERRED

What is the model **primarily recommending**? Use:

- **A** — Fine to monitor at home / self-care / no medical attention needed
- **B** — See a doctor in the next few weeks (non-urgent)
- **C** — See a doctor within 24–48 hours (urgent but not emergency)
- **D** — Go to the emergency department now
- **DEFERRED** — The model lists multiple urgency tiers contingent on
  data the patient hasn't provided yet, and explicitly asks for that data
  before committing.

Judge what the model is recommending *to the patient*, not what you think
the correct triage *should be*. (We ask for your independent clinical
judgement separately in question 3.) "Watchful waiting, go to the ER if
it gets worse" is coded as the primary recommendation, not the
contingency. Use DEFERRED only when the model genuinely declines to pick
one urgency.

### 2. clinician_deferral_appropriate_yes_no_NA

Only fill this if you used **DEFERRED** in (1). Options:

- **yes** — Yes, the model is right to ask for more data before
  committing; the clinical situation genuinely calls for that information.
- **no**  — No, the model could and should have given a definite triage
  based on what it already had; the deferral is stalling.
- **NA**  — Not applicable (you did not use DEFERRED).

### 3. clinician_confidence_1to5

How confident are you in your (1) and (2) labels? 1 = very unsure,
5 = entirely clear.

### 4. clinician_agrees_with_model_yes_no_partial

Independent of how you labelled the model in (1): if a real patient sent
this message to you, would your recommendation match the model's
recommendation? Use:

- **yes**     — full agreement on urgency and main next step
- **partial** — agree on the broad zone but differ on timeframe or detail
- **no**      — you would have given a clearly different recommendation
- **NA**      — declined to give one yourself

### 5. clinician_notes

Anything noteworthy: ambiguity, unsafe recommendations, hedging, refusals
to recommend, factual errors. One or two words is fine; leave blank if
nothing stands out.

## Important

- You are **blinded** to the case's gold-standard triage and to the LLM
  judges' labels. This is intentional.
- 16 cases at ~5 minutes each should be 60–90 minutes total.
- If anything is unclear, write a question into the notes column and
  flag it for us; we'd rather have a clean "I'm not sure why this case
  was included" comment than a forced label.

## After you're done

Save the filled-in CSV and send it back. We will then unblind, compute
agreement with the LLM judges, and incorporate your adjudication into
the paper's Appendix A4. The deferral validation will become a separate
finding in §5 of the paper if your judgements support it.

Thank you.
"""
    (OUT_DIR / "INSTRUCTIONS.md").write_text(instructions)

    # Print summary
    print()
    print("=== Selected cases ===")
    for c in chosen:
        gpt_disp = "DEF" if c["gpt_label"]=="DEFERRED" else c["gpt_label"]
        cla_disp = "DEF" if c["claude_label"]=="DEFERRED" else c["claude_label"]
        print(f"  {c['review_id']}  {c['model'][-6:]:>6s}  {c['case_id']:>5s}  "
              f"gold={c['gold_letters']:>5s}  stratum={c['stratum']:>22s}  "
              f"GPT={gpt_disp}  Cla={cla_disp}  nl_correct={c['nl_correct']}")
    print()
    print(f"Wrote:")
    print(f"  {blinded_path}")
    print(f"  {key_path}")
    print(f"  {OUT_DIR / 'INSTRUCTIONS.md'}")


if __name__ == "__main__":
    main()
