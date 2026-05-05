# Clinician adjudication — instructions

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
