"""
v2-medical pilot: Baseline viability check.

Goal: Measure the multilingual medical knowledge gap in Gemma 3 1B IT.
If the gap is large enough (>5 points), the SAE intervention experiment is viable.

Method: Answer likelihood scoring on multiple-choice questions.
For each question, compute P(correct_answer_token | question + options).
No free generation needed -- clean, comparable across languages.

Datasets (from OpenAI MMMLU, professionally translated):
  - Medical (EN + ES): anatomy, clinical_knowledge, college_medicine,
    medical_genetics, professional_medicine
  - Control (EN + ES): global_facts, philosophy (non-medical, matched format)

Usage:
    python3 v2_medical_pilot.py
"""
import json
import os
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_ID = "google/gemma-3-4b-it"

MEDICAL_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]

CONTROL_SUBJECTS = [
    "global_facts",
    "philosophy",
    "miscellaneous",
]

MAX_QUESTIONS_PER_SUBJECT = 200  # Cap for pilot speed
# ============================================================


def load_mmmlu(lang_config, subjects, max_per_subject=200):
    """Load MMMLU questions for given subjects.

    Args:
        lang_config: "default" for English, "ES_LA" for Spanish
        subjects: list of subject names to filter
        max_per_subject: cap per subject
    Returns:
        list of dicts with keys: question, options (A/B/C/D), answer, subject
    """
    print(f"  Loading MMMLU ({lang_config})...")
    ds = load_dataset("openai/MMMLU", lang_config, split="test")

    items = []
    counts = defaultdict(int)
    for row in ds:
        subj = row["Subject"]
        if subj not in subjects:
            continue
        if counts[subj] >= max_per_subject:
            continue
        counts[subj] += 1
        items.append({
            "question": row["Question"],
            "options": {
                "A": row["A"],
                "B": row["B"],
                "C": row["C"],
                "D": row["D"],
            },
            "answer": row["Answer"],  # "A", "B", "C", or "D"
            "subject": subj,
        })

    print(f"    Loaded {len(items)} questions across {dict(counts)}")
    return items


def format_mcq_prompt(question, options):
    """Format a multiple-choice question as a prompt."""
    prompt = f"Question: {question}\n"
    for key in ["A", "B", "C", "D"]:
        prompt += f"{key}. {options[key]}\n"
    prompt += "Answer:"
    return prompt


def evaluate_answer_likelihood(model, tokenizer, questions):
    """Evaluate accuracy using answer token likelihoods.

    For each question, compute P(A), P(B), P(C), P(D) as the probability
    the model assigns to each answer letter token given the prompt.
    The predicted answer is the one with highest probability.
    """
    correct = 0
    total = 0
    results = []

    # Get token IDs for answer letters
    answer_tokens = {}
    for letter in ["A", "B", "C", "D"]:
        # Try different tokenizations of the answer letter
        ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        answer_tokens[letter] = ids[-1]  # Take the last token (the letter itself)

    for i, q in enumerate(questions):
        prompt = format_mcq_prompt(q["question"], q["options"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)

        # Get logits for the last token position (where the model predicts the answer)
        last_logits = outputs.logits[0, -1, :]  # [vocab_size]

        # Compute probabilities for each answer letter
        answer_logits = {
            letter: last_logits[tid].item()
            for letter, tid in answer_tokens.items()
        }

        # Softmax over just the answer options
        logit_values = torch.tensor([answer_logits[l] for l in ["A", "B", "C", "D"]])
        probs = torch.softmax(logit_values, dim=0)
        answer_probs = {l: p.item() for l, p in zip(["A", "B", "C", "D"], probs)}

        predicted = max(answer_probs, key=answer_probs.get)
        is_correct = predicted == q["answer"]
        correct += int(is_correct)
        total += 1

        results.append({
            "subject": q["subject"],
            "correct": is_correct,
            "predicted": predicted,
            "actual": q["answer"],
            "probs": answer_probs,
            "confidence": answer_probs[predicted],
        })

        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(questions)}] running accuracy: {correct/total:.1%}")

        del inputs, outputs
        torch.cuda.empty_cache()

    return correct / total if total > 0 else 0, results


def main():
    print("=" * 60)
    print("v2-MEDICAL PILOT: Baseline Viability Check")
    print("=" * 60)

    # Load model
    print("\n--- Loading model ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16)
    print(f"  Loaded {MODEL_ID}")

    # Load datasets
    print("\n--- Loading datasets ---")
    en_medical = load_mmmlu("default", MEDICAL_SUBJECTS, MAX_QUESTIONS_PER_SUBJECT)
    es_medical = load_mmmlu("ES_LA", MEDICAL_SUBJECTS, MAX_QUESTIONS_PER_SUBJECT)
    en_control = load_mmmlu("default", CONTROL_SUBJECTS, MAX_QUESTIONS_PER_SUBJECT)
    es_control = load_mmmlu("ES_LA", CONTROL_SUBJECTS, MAX_QUESTIONS_PER_SUBJECT)

    # Evaluate all four conditions
    print("\n--- Evaluating: English Medical ---")
    en_med_acc, en_med_results = evaluate_answer_likelihood(model, tokenizer, en_medical)

    print("\n--- Evaluating: Spanish Medical ---")
    es_med_acc, es_med_results = evaluate_answer_likelihood(model, tokenizer, es_medical)

    print("\n--- Evaluating: English Control ---")
    en_ctrl_acc, en_ctrl_results = evaluate_answer_likelihood(model, tokenizer, en_control)

    print("\n--- Evaluating: Spanish Control ---")
    es_ctrl_acc, es_ctrl_results = evaluate_answer_likelihood(model, tokenizer, es_control)

    # Results summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n{'Condition':<25s} {'Accuracy':>10s} {'N':>6s}")
    print("-" * 45)
    print(f"{'English Medical':<25s} {en_med_acc:10.1%} {len(en_medical):6d}")
    print(f"{'Spanish Medical':<25s} {es_med_acc:10.1%} {len(es_medical):6d}")
    print(f"{'English Control':<25s} {en_ctrl_acc:10.1%} {len(en_control):6d}")
    print(f"{'Spanish Control':<25s} {es_ctrl_acc:10.1%} {len(es_control):6d}")

    med_gap = en_med_acc - es_med_acc
    ctrl_gap = en_ctrl_acc - es_ctrl_acc

    print(f"\n{'Medical gap (EN - ES):':<25s} {med_gap:+.1%}")
    print(f"{'Control gap (EN - ES):':<25s} {ctrl_gap:+.1%}")
    print(f"{'Medical-specific gap:':<25s} {med_gap - ctrl_gap:+.1%}")

    # Per-subject breakdown
    print(f"\n--- Per-subject breakdown ---")
    for subj in MEDICAL_SUBJECTS + CONTROL_SUBJECTS:
        en_res = [r for r in (en_med_results + en_ctrl_results) if r["subject"] == subj]
        es_res = [r for r in (es_med_results + es_ctrl_results) if r["subject"] == subj]
        if en_res and es_res:
            en_acc = sum(r["correct"] for r in en_res) / len(en_res)
            es_acc = sum(r["correct"] for r in es_res) / len(es_res)
            gap = en_acc - es_acc
            marker = " ***" if abs(gap) > 0.10 else ""
            print(f"  {subj:30s}  EN={en_acc:.1%}  ES={es_acc:.1%}  gap={gap:+.1%}{marker}")

    # Viability assessment
    print(f"\n{'=' * 60}")
    print("VIABILITY ASSESSMENT")
    print(f"{'=' * 60}")

    if en_med_acc < 0.35:
        print("CONCERN: English medical accuracy is below 35%.")
        print("The model may not have enough medical knowledge for intervention to rescue.")
        print("Consider stepping up to Gemma 3 4B.")
    elif med_gap < 0.05:
        print("CONCERN: Medical gap is less than 5 points.")
        print("Effect may be too small to measure via SAE intervention.")
        print("Could still work with larger sample sizes or stronger features.")
    elif med_gap > 0.10 and en_med_acc > 0.45:
        print("GREEN LIGHT: Large medical gap with reasonable English baseline.")
        print("SAE intervention experiment is viable on this model.")
        print("Proceed to contrastive feature analysis.")
    else:
        print(f"MARGINAL: Gap is {med_gap:.1%}, baseline is {en_med_acc:.1%}.")
        print("Experiment is feasible but may need careful statistical analysis.")
        print("Consider running on 4B as well for comparison.")

    if med_gap > ctrl_gap + 0.03:
        print(f"\nMedical-specific gap ({med_gap - ctrl_gap:+.1%}) exceeds general language gap.")
        print("This suggests domain knowledge loss, not just translation quality.")
        print("The SAE medical feature hypothesis is supported.")
    else:
        print(f"\nMedical gap ({med_gap:.1%}) is similar to control gap ({ctrl_gap:.1%}).")
        print("The gap may be general language quality, not medical-specific.")
        print("Feature intervention might not target the right thing.")

    # Save detailed results
    output = {
        "model": MODEL_ID,
        "summary": {
            "en_medical_acc": en_med_acc,
            "es_medical_acc": es_med_acc,
            "en_control_acc": en_ctrl_acc,
            "es_control_acc": es_ctrl_acc,
            "medical_gap": med_gap,
            "control_gap": ctrl_gap,
        },
        "en_medical_results": en_med_results,
        "es_medical_results": es_med_results,
        "en_control_results": en_ctrl_results,
        "es_control_results": es_ctrl_results,
    }

    with open("results/v2_medical_baseline_4b.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to results/v2_medical_baseline_4b.json")

    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
