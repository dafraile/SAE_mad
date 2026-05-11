"""
v2-medical rescue v2: TARGETED contrastive feature identification.

Upgrade over v2_medical_rescue.py: instead of finding "Spanish-ness" features,
find features that are specifically associated with Spanish medical content and
NOT with Spanish non-medical content or English medical content.

Design (per design Claude's framework):
  4 conditions: EN medical, ES medical, EN non-medical, ES non-medical
  Target features: high ES-medical activation, low everywhere else

  Score = (ES_med_act + eps) / (EN_med_act + ES_nonmed_act + eps) * ES_med_fire_rate

This identifies features that encode medical knowledge through a Spanish-preferential
pathway -- candidates for transplanting into English inference.

Also test multiple layers (17, 22, 29) to see where medical routing lives.

Usage:
    python3 v2_medical_rescue_v2.py
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

# Layers to test (early-mid, upper-mid, late in 34-layer model)
CANDIDATE_LAYERS = [17, 22, 29]

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

# Contrastive feature discovery: per-condition sample size
N_CONTRASTIVE = 40

STEER_MULTIPLIERS = [0, 1.0, 2.0, 3.0]
# ============================================================


def get_layer(model, idx):
    """Access layer N for Gemma 3."""
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[idx]
    return model.model.layers[idx]


def load_paired_mmmlu(subjects, tag):
    """Load paired EN/ES MMMLU questions for given subjects."""
    ds_en = load_dataset("openai/MMMLU", "default", split="test")
    ds_es = load_dataset("openai/MMMLU", "ES_LA", split="test")

    en_by_pos = defaultdict(list)
    es_by_pos = defaultdict(list)
    for row in ds_en:
        if row["Subject"] in subjects:
            en_by_pos[row["Subject"]].append(row)
    for row in ds_es:
        if row["Subject"] in subjects:
            es_by_pos[row["Subject"]].append(row)

    pairs = []
    for subj in subjects:
        en_list = en_by_pos[subj]
        es_list = es_by_pos[subj]
        n = min(len(en_list), len(es_list))
        for i in range(n):
            if en_list[i]["Answer"] == es_list[i]["Answer"]:
                pairs.append({
                    "subject": subj,
                    "category": tag,
                    "en_question": en_list[i]["Question"],
                    "en_options": {k: en_list[i][k] for k in ["A", "B", "C", "D"]},
                    "es_question": es_list[i]["Question"],
                    "es_options": {k: es_list[i][k] for k in ["A", "B", "C", "D"]},
                    "answer": en_list[i]["Answer"],
                })
    return pairs


def format_mcq(q, options):
    text = f"Question: {q}\n"
    for key in ["A", "B", "C", "D"]:
        text += f"{key}. {options[key]}\n"
    text += "Answer:"
    return text


def get_answer_probs(model, tokenizer, prompt, answer_token_ids):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    last_logits = outputs.logits[0, -1, :]
    letter_logits = torch.tensor([last_logits[tid].item() for tid in answer_token_ids])
    probs = torch.softmax(letter_logits, dim=0)
    return {l: p.item() for l, p in zip(["A", "B", "C", "D"], probs)}


def evaluate_pairs(model, tokenizer, pairs, lang, answer_token_ids, label=""):
    """Score questions in given language."""
    q_key = f"{lang}_question"
    o_key = f"{lang}_options"
    results = []
    for i, pair in enumerate(pairs):
        prompt = format_mcq(pair[q_key], pair[o_key])
        probs = get_answer_probs(model, tokenizer, prompt, answer_token_ids)
        predicted = max(probs, key=probs.get)
        results.append({
            "subject": pair["subject"],
            "category": pair.get("category", ""),
            "actual": pair["answer"],
            "predicted": predicted,
            "correct": predicted == pair["answer"],
            "probs": probs,
        })
        if (i + 1) % 100 == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"    [{i+1}/{len(pairs)}] {label}: acc={acc:.1%}")
    return results


def collect_sae_activations(model, tokenizer, sae, pairs, lang, layer_idx):
    """Collect mean SAE feature activations for a batch of prompts at given layer."""
    q_key = f"{lang}_question"
    o_key = f"{lang}_options"

    feat_means = []
    for pair in pairs:
        captured = []
        def hook_fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            captured.append(o.detach())
        h = get_layer(model, layer_idx).register_forward_hook(hook_fn)

        prompt = format_mcq(pair[q_key], pair[o_key])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            model(**inputs)
        h.remove()

        resid = captured[0].to(sae.dtype)
        features = sae.encode(resid)
        feat_means.append(features.float().mean(dim=1).squeeze(0).cpu())
        del captured, features, resid
        torch.cuda.empty_cache()

    return torch.stack(feat_means)  # [N, d_sae]


def find_es_medical_features(model, tokenizer, sae, layer_idx,
                              med_pairs, ctrl_pairs,
                              n_contrastive=40, top_k=10):
    """Find features that are specifically ES-medical.

    Score rewards:
    - High mean activation on ES medical
    - Consistent firing on ES medical (high fire rate)
    Penalizes:
    - High activation on EN medical (we want features English doesn't have)
    - High activation on ES non-medical (we want medical, not just Spanish)
    """
    print(f"\n--- Contrastive feature analysis at layer {layer_idx} ---")
    med_sample = med_pairs[:n_contrastive]
    ctrl_sample = ctrl_pairs[:n_contrastive]

    print(f"  Collecting activations (N={n_contrastive} per condition)...")
    es_med = collect_sae_activations(model, tokenizer, sae, med_sample, "es", layer_idx)
    print(f"    [1/4] ES medical done")
    en_med = collect_sae_activations(model, tokenizer, sae, med_sample, "en", layer_idx)
    print(f"    [2/4] EN medical done")
    es_ctrl = collect_sae_activations(model, tokenizer, sae, ctrl_sample, "es", layer_idx)
    print(f"    [3/4] ES non-medical done")
    en_ctrl = collect_sae_activations(model, tokenizer, sae, ctrl_sample, "en", layer_idx)
    print(f"    [4/4] EN non-medical done")

    es_med_mean = es_med.mean(dim=0)
    en_med_mean = en_med.mean(dim=0)
    es_ctrl_mean = es_ctrl.mean(dim=0)
    en_ctrl_mean = en_ctrl.mean(dim=0)

    es_med_fires = (es_med > 0).float().mean(dim=0)

    # Score function: fire consistently on ES-medical, not on the other three
    # conditions -- especially not EN-medical (we want features English lacks)
    # Penalty: activations in other conditions relative to ES-medical
    eps = 0.5
    ratio_vs_enmed = (es_med_mean + eps) / (en_med_mean + eps)
    ratio_vs_esctrl = (es_med_mean + eps) / (es_ctrl_mean + eps)

    # Combined score (require fires consistently on ES med AND dominates other conds)
    score = es_med_fires * torch.minimum(ratio_vs_enmed, ratio_vs_esctrl)

    # Require firing on at least 70% of ES medical examples
    score = score * (es_med_fires > 0.7).float()
    # Require substantial absolute activation
    score = score * (es_med_mean > 1.0).float()

    top = score.topk(top_k)
    candidates = []
    for i in range(top_k):
        idx = top.indices[i].item()
        if top.values[i].item() == 0:
            continue
        candidates.append({
            "feature_idx": idx,
            "layer": layer_idx,
            "es_med_act": es_med_mean[idx].item(),
            "en_med_act": en_med_mean[idx].item(),
            "es_ctrl_act": es_ctrl_mean[idx].item(),
            "en_ctrl_act": en_ctrl_mean[idx].item(),
            "es_med_fire_rate": es_med_fires[idx].item(),
            "score": top.values[i].item(),
        })

    print(f"\n  Top ES-medical-specific features at layer {layer_idx}:")
    print(f"  {'idx':>6s} {'ES_med':>10s} {'EN_med':>10s} {'ES_ctrl':>10s} {'EN_ctrl':>10s} {'fires':>6s} {'score':>10s}")
    for c in candidates:
        print(f"  {c['feature_idx']:6d} {c['es_med_act']:10.2f} {c['en_med_act']:10.2f} "
              f"{c['es_ctrl_act']:10.2f} {c['en_ctrl_act']:10.2f} "
              f"{c['es_med_fire_rate']:6.2f} {c['score']:10.2f}")

    return candidates


@contextmanager
def steer_features(model, sae, layer, feature_deltas):
    """Steer multiple features simultaneously."""
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


def run_intervention(model, tokenizer, sae, layer, features, strength,
                     test_pairs, test_idxs, en_results, rescuable_idxs, control_idxs,
                     answer_tids):
    """Run steering at given strength, return rescue/break stats."""
    feature_deltas = {
        f["feature_idx"]: strength * f["es_med_act"]
        for f in features
    } if strength > 0 else {}

    new_results = []
    for idx in test_idxs:
        pair = test_pairs[idx]
        prompt = format_mcq(pair["en_question"], pair["en_options"])
        if strength > 0:
            with steer_features(model, sae, layer, feature_deltas):
                probs = get_answer_probs(model, tokenizer, prompt, answer_tids)
        else:
            probs = get_answer_probs(model, tokenizer, prompt, answer_tids)
        predicted = max(probs, key=probs.get)
        correct = predicted == pair["answer"]
        new_results.append({
            "idx": idx,
            "correct": correct,
            "was_correct": en_results[idx]["correct"],
            "is_rescuable": idx in rescuable_idxs,
        })

    rescuable_now = sum(1 for r in new_results if r["is_rescuable"] and r["correct"])
    control_still = sum(1 for r in new_results if r["was_correct"] and r["correct"])
    broken = len(control_idxs) - control_still
    return rescuable_now, broken, new_results


def main():
    print("=" * 70)
    print("v2-MEDICAL RESCUE v2: TARGETED contrastive feature identification")
    print("=" * 70)

    # Load model
    print("\n--- Loading model ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16)

    answer_tids = [tokenizer.encode(f" {l}", add_special_tokens=False)[-1] for l in ["A", "B", "C", "D"]]

    # Load paired conditions
    print("\n--- Loading conditions ---")
    med_pairs = load_paired_mmmlu(MEDICAL_SUBJECTS, "medical")
    ctrl_pairs = load_paired_mmmlu(CONTROL_SUBJECTS, "control")
    print(f"  Medical pairs: {len(med_pairs)}")
    print(f"  Control pairs: {len(ctrl_pairs)}")

    # Baseline on medical
    print("\n--- Step 1: Baseline on medical ---")
    print("  Evaluating English medical...")
    en_results = evaluate_pairs(model, tokenizer, med_pairs, "en", answer_tids, "en_med")
    print("  Evaluating Spanish medical...")
    es_results = evaluate_pairs(model, tokenizer, med_pairs, "es", answer_tids, "es_med")

    en_acc = sum(r["correct"] for r in en_results) / len(en_results)
    es_acc = sum(r["correct"] for r in es_results) / len(es_results)
    print(f"\n  EN={en_acc:.1%}  ES={es_acc:.1%}  Gap={es_acc-en_acc:+.1%}")

    rescuable_idxs = [
        i for i, (er, sr) in enumerate(zip(en_results, es_results))
        if sr["correct"] and not er["correct"]
    ]
    en_correct_idxs = [i for i, r in enumerate(en_results) if r["correct"]]
    control_idxs = en_correct_idxs[:50]  # hold out 50 for "don't break it" test
    test_idxs = sorted(set(rescuable_idxs + control_idxs))

    print(f"\n  Rescuable: {len(rescuable_idxs)} questions")
    print(f"  Control (don't-break): {len(control_idxs)}")

    # Feature discovery at each candidate layer
    print("\n" + "=" * 70)
    print("--- Step 2: Feature discovery (contrastive across 4 conditions) ---")
    print("=" * 70)

    all_features_by_layer = {}
    for layer in CANDIDATE_LAYERS:
        sae_id = f"layer_{layer}_width_16k_l0_medium"
        print(f"\n  Loading SAE at layer {layer}...")
        sae = SAE.from_pretrained(release="gemma-scope-2-4b-it-res",
                                   sae_id=sae_id, device="cuda")
        features = find_es_medical_features(model, tokenizer, sae, layer,
                                             med_pairs, ctrl_pairs,
                                             n_contrastive=N_CONTRASTIVE, top_k=10)
        all_features_by_layer[layer] = {"features": features, "sae": sae}
        # Don't delete SAE yet -- keep for steering

    # Step 3: Steering experiments
    print("\n" + "=" * 70)
    print("--- Step 3: Steering experiments ---")
    print("=" * 70)

    all_results = {}
    for layer in CANDIDATE_LAYERS:
        print(f"\n  Layer {layer}:")
        sae = all_features_by_layer[layer]["sae"]
        features = all_features_by_layer[layer]["features"][:5]  # top 5

        if not features:
            print("    No features to steer")
            continue

        layer_results = {}
        for strength in STEER_MULTIPLIERS:
            rescuable_now, broken, _ = run_intervention(
                model, tokenizer, sae, layer, features, strength,
                med_pairs, test_idxs, en_results, rescuable_idxs, control_idxs,
                answer_tids
            )
            net = rescuable_now - broken
            print(f"    {strength}x: rescued {rescuable_now}/{len(rescuable_idxs)}, "
                  f"broken {broken}/{len(control_idxs)}, net {net:+d}")
            layer_results[strength] = {
                "rescued": rescuable_now,
                "broken": broken,
                "net": net,
                "rescue_rate": rescuable_now / max(len(rescuable_idxs), 1),
            }
        all_results[layer] = {
            "features": features,
            "by_strength": layer_results,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBaseline: EN={en_acc:.1%}  ES={es_acc:.1%}  Gap={es_acc-en_acc:+.1%}")
    print(f"Rescuable: {len(rescuable_idxs)}  Control: {len(control_idxs)}")

    print(f"\n{'Layer':>6s} {'Best mult':>10s} {'Rescued':>10s} {'Broken':>8s} {'Net':>6s} {'Rescue rate':>12s}")
    best_overall = None
    for layer, res in all_results.items():
        best_strength = max(res["by_strength"].items(), key=lambda x: x[1]["net"])
        strength, stats = best_strength
        print(f"{layer:6d} {strength:10.1f}x {stats['rescued']:4d}/{len(rescuable_idxs):<4d} "
              f"{stats['broken']:4d}/{len(control_idxs):<3d} {stats['net']:>+6d} {stats['rescue_rate']:11.1%}")
        if best_overall is None or stats["net"] > best_overall["net"]:
            best_overall = {"layer": layer, "strength": strength, **stats}

    if best_overall:
        print(f"\nBest: Layer {best_overall['layer']}, {best_overall['strength']}x, "
              f"net +{best_overall['net']}, rescue rate {best_overall['rescue_rate']:.1%}")

    # Save
    output = {
        "model": MODEL_ID,
        "layers_tested": CANDIDATE_LAYERS,
        "baseline": {"en_acc": en_acc, "es_acc": es_acc, "gap": es_acc - en_acc},
        "n_rescuable": len(rescuable_idxs),
        "n_control": len(control_idxs),
        "by_layer": {
            str(l): {
                "features": res["features"],
                "by_strength": {str(k): v for k, v in res["by_strength"].items()},
            }
            for l, res in all_results.items()
        },
    }
    with open("results/v2_medical_rescue_v2.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to results/v2_medical_rescue_v2.json")

    del model
    for info in all_features_by_layer.values():
        del info["sae"]
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
