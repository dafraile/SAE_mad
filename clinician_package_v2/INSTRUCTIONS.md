# Clinician adjudication v2 — instructions for Jia

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
