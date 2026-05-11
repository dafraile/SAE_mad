"""
v2-medical rescue: Can we rescue English medical performance using Spanish features?

Context: Gemma 3 4B is ~11 points BETTER at medical MCQ in Spanish than English
(42.9% vs 53.6%). This is the opposite of the documented multilingual gap but
real on this model. The question: can we improve English performance by
activating Spanish-associated features at inference time?

Method:
1. Run baseline EN and ES on matched medical questions
2. Identify "rescuable" questions: Spanish gets right, English gets wrong
3. Find a clean Spanish language feature on Gemma 3 4B layer 29 SAE
4. On the rescuable English questions, amplify the Spanish feature during
   answer scoring. Does accuracy improve?

This tests whether feature-level intervention can transfer capability across
language boundaries -- the core routing hypothesis in a practical framing.

Usage:
    python3 v2_medical_rescue.py
"""
import json
import os
import torch
import numpy as np
from collections import defaultdict
from contextlib import contextmanager
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sae_lens import SAE

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_ID = "google/gemma-3-4b-it"

# Layer 29 is at 85% depth of Gemma 3 4B (34 layers). This is where
# language-specific features were strongest in the 1B model.
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = "layer_29_width_16k_l0_medium"
STEER_LAYER = 29

MEDICAL_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]

# Load only the paired subset (where both EN and ES have the same question)
# ES_LA has 808 medical questions, EN has 1000. We use the overlap.
STEER_MULTIPLIERS = [0, 1.0, 2.0, 3.0, 5.0]


def get_layer(model, idx):
    """Access layer N for Gemma 3 (handles 1B text-only and 4B multimodal)."""
    # 4B has model.model.language_model.layers, 1B has model.model.layers
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[idx]
    return model.model.layers[idx]

# How many top candidate Spanish features to screen
N_CANDIDATE_FEATURES = 10
N_FEATURES_TO_STEER = 3  # Top-k features to use as routing signal
# ============================================================


def load_paired_medical():
    """Load MMMLU medical questions with paired EN/ES versions."""
    print("Loading MMMLU medical (both languages)...")

    ds_en = load_dataset("openai/MMMLU", "default", split="test")
    ds_es = load_dataset("openai/MMMLU", "ES_LA", split="test")

    # Build EN index
    en_by_q = {}
    for row in ds_en:
        if row["Subject"] in MEDICAL_SUBJECTS:
            # Use question as key
            en_by_q[row["Question"]] = row

    # The ES_LA questions are translations. We don't have a direct ID mapping,
    # but OpenAI's MMMLU preserves question order within subjects, so match by
    # subject + position.
    en_by_pos = defaultdict(list)
    for i, row in enumerate(ds_en):
        if row["Subject"] in MEDICAL_SUBJECTS:
            en_by_pos[row["Subject"]].append(row)

    es_by_pos = defaultdict(list)
    for i, row in enumerate(ds_es):
        if row["Subject"] in MEDICAL_SUBJECTS:
            es_by_pos[row["Subject"]].append(row)

    # Pair them up by subject + position (ES_LA is a subset of EN)
    pairs = []
    for subj in MEDICAL_SUBJECTS:
        en_list = en_by_pos[subj]
        es_list = es_by_pos[subj]
        # The ES version should be the first N questions of EN for that subject
        n = min(len(en_list), len(es_list))
        for i in range(n):
            # Verify answer matches (sanity check)
            if en_list[i]["Answer"] == es_list[i]["Answer"]:
                pairs.append({
                    "subject": subj,
                    "en_question": en_list[i]["Question"],
                    "en_options": {k: en_list[i][k] for k in ["A", "B", "C", "D"]},
                    "es_question": es_list[i]["Question"],
                    "es_options": {k: es_list[i][k] for k in ["A", "B", "C", "D"]},
                    "answer": en_list[i]["Answer"],
                })

    print(f"  {len(pairs)} paired medical questions across {len(MEDICAL_SUBJECTS)} subjects")
    return pairs


def format_mcq(q, options):
    """Format a multiple-choice question."""
    text = f"Question: {q}\n"
    for key in ["A", "B", "C", "D"]:
        text += f"{key}. {options[key]}\n"
    text += "Answer:"
    return text


def get_answer_probs(model, tokenizer, prompt, answer_token_ids):
    """Get probabilities for A/B/C/D given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    last_logits = outputs.logits[0, -1, :]
    letter_logits = torch.tensor([last_logits[tid].item() for tid in answer_token_ids])
    probs = torch.softmax(letter_logits, dim=0)
    return {l: p.item() for l, p in zip(["A", "B", "C", "D"], probs)}


def evaluate_batch(model, tokenizer, pairs, lang, answer_token_ids):
    """Score a set of questions, return per-question results."""
    q_key = f"{lang}_question"
    o_key = f"{lang}_options"

    results = []
    for i, pair in enumerate(pairs):
        prompt = format_mcq(pair[q_key], pair[o_key])
        probs = get_answer_probs(model, tokenizer, prompt, answer_token_ids)
        predicted = max(probs, key=probs.get)
        results.append({
            "subject": pair["subject"],
            "actual": pair["answer"],
            "predicted": predicted,
            "correct": predicted == pair["answer"],
            "probs": probs,
            "confidence": probs[predicted],
        })
        if (i + 1) % 100 == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"    [{i+1}/{len(pairs)}] {lang}: acc={acc:.1%}")
    return results


def identify_spanish_features(model, tokenizer, sae, pairs, layer_idx, n_candidates=10):
    """Find SAE features that discriminate Spanish text from English text.

    Returns a ranked list of (feature_idx, mean_act_ES, mean_act_EN, discrimination_score).
    """
    print(f"\n--- Identifying Spanish-specific features at layer {layer_idx} ---")

    # Sample 30 pairs for feature discovery (fast)
    sample = pairs[:30]

    # Collect activations on EN and ES prompts
    en_acts = []
    es_acts = []

    def hook_fn(storage):
        def fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            storage.append(o.detach().cpu())
        return fn

    for i, pair in enumerate(sample):
        # EN activation
        en_store = []
        h = get_layer(model, layer_idx).register_forward_hook(hook_fn(en_store))
        prompt = format_mcq(pair["en_question"], pair["en_options"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            model(**inputs)
        h.remove()
        en_acts.append(en_store[0])

        # ES activation
        es_store = []
        h = get_layer(model, layer_idx).register_forward_hook(hook_fn(es_store))
        prompt = format_mcq(pair["es_question"], pair["es_options"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            model(**inputs)
        h.remove()
        es_acts.append(es_store[0])

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(sample)}] collected activations")

    # Encode through SAE, compute mean feature activation per language
    en_feat_means = []
    es_feat_means = []

    for en_a, es_a in zip(en_acts, es_acts):
        en_feats = sae.encode(en_a.cuda().to(sae.dtype))
        es_feats = sae.encode(es_a.cuda().to(sae.dtype))
        # Mean over tokens
        en_feat_means.append(en_feats.float().mean(dim=1).squeeze(0).cpu())
        es_feat_means.append(es_feats.float().mean(dim=1).squeeze(0).cpu())

    en_mat = torch.stack(en_feat_means)  # [N, d_sae]
    es_mat = torch.stack(es_feat_means)

    # Discrimination score: how much higher does ES fire than EN?
    en_mean = en_mat.mean(dim=0)
    es_mean = es_mat.mean(dim=0)

    # Only features that fire on most ES examples
    es_fires = (es_mat > 0).float().mean(dim=0)  # fraction of ES examples where it fires
    en_fires = (en_mat > 0).float().mean(dim=0)

    # Score: high ES activation, low EN activation, fires consistently on ES
    # Use a ratio-based score, but guard against div by zero
    score = (es_mean + 0.01) / (en_mean + 0.01) * es_fires

    # Filter: must fire on >= 80% of ES examples
    mask = es_fires > 0.8
    score = score * mask.float()

    top_k = score.topk(n_candidates)
    candidates = []
    for i in range(n_candidates):
        idx = top_k.indices[i].item()
        candidates.append({
            "feature_idx": idx,
            "es_mean_act": es_mean[idx].item(),
            "en_mean_act": en_mean[idx].item(),
            "es_fire_rate": es_fires[idx].item(),
            "en_fire_rate": en_fires[idx].item(),
            "score": score[idx].item(),
        })

    print(f"\n    Top {n_candidates} Spanish-discriminating features:")
    print(f"    {'rank':>4s} {'idx':>6s} {'es_act':>10s} {'en_act':>10s} {'es_fires':>9s} {'en_fires':>9s} {'score':>8s}")
    for i, c in enumerate(candidates):
        print(f"    {i+1:4d} {c['feature_idx']:6d} {c['es_mean_act']:10.2f} {c['en_mean_act']:10.2f} "
              f"{c['es_fire_rate']:9.2f} {c['en_fire_rate']:9.2f} {c['score']:8.2f}")

    return candidates


@contextmanager
def steer_features(model, sae, layer, feature_deltas):
    """Steer multiple features simultaneously.

    feature_deltas: dict of {feature_idx: amount_to_add}
    """
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        resid = hidden.clone()
        resid_for_sae = resid.to(sae.dtype)
        features = sae.encode(resid_for_sae)

        for feat_idx, delta in feature_deltas.items():
            features[:, :, feat_idx] = features[:, :, feat_idx] + delta

        steered = sae.decode(features)
        original_recon = sae.decode(sae.encode(resid_for_sae))
        delta_vec = (steered - original_recon).to(hidden.dtype)
        result = hidden + delta_vec

        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result

    hook = get_layer(model, layer).register_forward_hook(hook_fn)
    try:
        yield
    finally:
        hook.remove()


def main():
    print("=" * 70)
    print("v2-MEDICAL RESCUE: Activate Spanish features on English questions")
    print("=" * 70)

    # Load
    print("\n--- Loading model ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16)

    print(f"\n--- Loading SAE ({SAE_RELEASE}/{SAE_ID}) ---")
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    print(f"  d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    # Answer token IDs
    answer_tids = [tokenizer.encode(f" {l}", add_special_tokens=False)[-1] for l in ["A", "B", "C", "D"]]

    # Load paired medical questions
    pairs = load_paired_medical()

    # Step 1: Baseline accuracy on matched set
    print("\n--- Step 1: Baseline accuracy on paired medical questions ---")
    print("  Evaluating English...")
    en_results = evaluate_batch(model, tokenizer, pairs, "en", answer_tids)
    print("  Evaluating Spanish...")
    es_results = evaluate_batch(model, tokenizer, pairs, "es", answer_tids)

    en_acc = sum(r["correct"] for r in en_results) / len(en_results)
    es_acc = sum(r["correct"] for r in es_results) / len(es_results)
    print(f"\n  English: {en_acc:.1%}")
    print(f"  Spanish: {es_acc:.1%}")
    print(f"  Gap: {es_acc - en_acc:+.1%}")

    # Step 2: Identify rescuable questions (ES right, EN wrong)
    rescuable = [
        (i, pair) for i, (pair, en_r, es_r) in enumerate(zip(pairs, en_results, es_results))
        if es_r["correct"] and not en_r["correct"]
    ]
    print(f"\n  Rescuable questions (ES correct, EN wrong): {len(rescuable)}")
    # Also track "already correct" (EN right) as we don't want to hurt those
    en_correct_idxs = [i for i, r in enumerate(en_results) if r["correct"]]

    # Step 3: Identify Spanish-specific features
    candidates = identify_spanish_features(
        model, tokenizer, sae, pairs, STEER_LAYER,
        n_candidates=N_CANDIDATE_FEATURES
    )

    # Take top N as routing features
    top_features = candidates[:N_FEATURES_TO_STEER]
    print(f"\n  Using top {N_FEATURES_TO_STEER} features: {[f['feature_idx'] for f in top_features]}")

    # Step 4: Steer English inference with Spanish features and measure accuracy
    print("\n--- Step 4: Steering experiment ---")
    print("  For each strength multiplier, re-score English questions with Spanish features amplified")

    # We'll evaluate on the rescuable set + a control set of already-correct EN
    # questions (to check we're not breaking those)
    rescuable_idxs = [i for i, _ in rescuable]
    # Use up to 50 already-correct as a "don't break it" control
    control_idxs = en_correct_idxs[:50]
    test_idxs = sorted(set(rescuable_idxs + control_idxs))

    print(f"  Evaluating {len(rescuable_idxs)} rescuable + {len(control_idxs)} already-correct = {len(test_idxs)} questions")

    steering_results = {}
    for mult in STEER_MULTIPLIERS:
        print(f"\n  Multiplier {mult}x:")

        # Build feature deltas: add mult * es_mean_act for each top feature
        if mult == 0:
            feature_deltas = {}  # baseline (no steering)
        else:
            feature_deltas = {
                f["feature_idx"]: mult * f["es_mean_act"]
                for f in top_features
            }

        n_rescued = 0
        n_broken = 0
        results_at_mult = []

        for idx in test_idxs:
            pair = pairs[idx]
            prompt = format_mcq(pair["en_question"], pair["en_options"])

            if mult == 0:
                probs = get_answer_probs(model, tokenizer, prompt, answer_tids)
            else:
                with steer_features(model, sae, STEER_LAYER, feature_deltas):
                    probs = get_answer_probs(model, tokenizer, prompt, answer_tids)

            predicted = max(probs, key=probs.get)
            correct = predicted == pair["answer"]

            was_correct = en_results[idx]["correct"]

            results_at_mult.append({
                "idx": idx,
                "subject": pair["subject"],
                "correct": correct,
                "was_correct_en": was_correct,
                "is_rescuable": idx in rescuable_idxs,
                "predicted": predicted,
                "actual": pair["answer"],
            })

            if idx in rescuable_idxs and correct:
                n_rescued += 1
            if was_correct and not correct:
                n_broken += 1

        # Summary
        rescuable_now = sum(1 for r in results_at_mult if r["is_rescuable"] and r["correct"])
        control_still = sum(1 for r in results_at_mult if r["was_correct_en"] and r["correct"])

        rescue_rate = rescuable_now / len(rescuable_idxs) if rescuable_idxs else 0
        break_rate = (len(control_idxs) - control_still) / len(control_idxs) if control_idxs else 0

        print(f"    Rescuable questions now correct: {rescuable_now}/{len(rescuable_idxs)} = {rescue_rate:.1%}")
        print(f"    Already-correct questions broken: {len(control_idxs) - control_still}/{len(control_idxs)} = {break_rate:.1%}")

        # Net change
        net = rescuable_now - (len(control_idxs) - control_still)
        print(f"    Net questions fixed: {net:+d}")

        steering_results[mult] = {
            "rescuable_now_correct": rescuable_now,
            "control_still_correct": control_still,
            "rescue_rate": rescue_rate,
            "break_rate": break_rate,
            "net_fixed": net,
            "details": results_at_mult,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBaseline: EN={en_acc:.1%}  ES={es_acc:.1%}  Gap={es_acc-en_acc:+.1%}")
    print(f"Rescuable questions: {len(rescuable_idxs)}")
    print(f"Control (already correct): {len(control_idxs)}")
    print(f"\n{'Mult':>6s} {'Rescued':>10s} {'Broken':>10s} {'Net':>8s}")
    for mult, res in steering_results.items():
        print(f"{mult:>6.1f}x {res['rescuable_now_correct']:>6d}/{len(rescuable_idxs):>3d} "
              f"{len(control_idxs)-res['control_still_correct']:>6d}/{len(control_idxs):>3d} "
              f"{res['net_fixed']:>+8d}")

    # Verdict
    best = max(steering_results.values(), key=lambda r: r["net_fixed"])
    if best["net_fixed"] > 0:
        print(f"\nVERDICT: Spanish feature steering can rescue English performance.")
        print(f"Best net gain: {best['net_fixed']:+d} questions.")
        print(f"This supports the feature-level cross-lingual routing hypothesis.")
    else:
        print(f"\nVERDICT: No net rescue at any strength.")
        print(f"Steering either doesn't transfer knowledge or breaks as much as it fixes.")

    # Save
    output = {
        "model": MODEL_ID,
        "layer": STEER_LAYER,
        "sae_id": SAE_ID,
        "baseline": {"en_acc": en_acc, "es_acc": es_acc},
        "n_rescuable": len(rescuable_idxs),
        "n_control": len(control_idxs),
        "candidate_features": candidates,
        "top_features_used": [f["feature_idx"] for f in top_features],
        "results_by_strength": {
            str(m): {k: v for k, v in r.items() if k != "details"}
            for m, r in steering_results.items()
        },
    }
    with open("results/v2_medical_rescue.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to results/v2_medical_rescue.json")

    del model, sae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
