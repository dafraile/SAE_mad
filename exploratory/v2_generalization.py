"""
v2 Generalization: Does single-feature rescue generalize across languages and domains?

Motivation: v2 showed +9 rescue using generic Spanish features on EN medical.
Question: Is this Spanish-specific? Medical-specific? Or a general mechanism?

Design:
  Languages tested: Spanish (ES_LA), French (FR_FR)
  Domains tested:   Medical, Philosophy, Global Facts, Miscellaneous, STEM

For each (language, domain) cell:
  1. Baseline: EN accuracy vs Target-language accuracy on matched questions
  2. Identify generic-language feature (fires on many target texts, not on EN)
  3. Rescue: amplify language feature on EN inference, measure rescue rate

This gives us a 2x5 grid of results. If rescue rates are similar across cells,
the mechanism is language-general. If some cells work and others don't, that
tells us which domains have exploitable cross-lingual routing.

Also pushes harder: top-10 features (not top-3), strengths up to 5x.

Usage:
    python3 v2_generalization.py
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

# Best layer from v2 for steering (late layer where language features concentrate)
STEER_LAYER = 29
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = f"layer_{STEER_LAYER}_width_16k_l0_medium"

# Languages to test (besides English)
LANGUAGES = {
    "es": "ES_LA",
    "fr": "FR_FR",
}

# Domains to test
DOMAINS = {
    "medical": ["anatomy", "clinical_knowledge", "college_medicine",
                "medical_genetics", "professional_medicine"],
    "philosophy": ["philosophy", "moral_disputes", "moral_scenarios"],
    "global_facts": ["global_facts", "miscellaneous"],
    "stem": ["high_school_biology", "high_school_chemistry",
             "high_school_physics", "high_school_mathematics"],
    "humanities": ["world_religions", "prehistory", "high_school_world_history"],
}

# Cap per domain to keep runtime reasonable
MAX_PER_DOMAIN = 200

STEER_MULTIPLIERS = [0, 1.0, 2.0, 3.0, 5.0]
N_TOP_FEATURES = 10  # Use top 10 features instead of 3

# How many text samples for language feature identification
N_FEATURE_ID_SAMPLES = 50
# ============================================================


def get_layer(model, idx):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[idx]
    return model.model.layers[idx]


def load_paired(subjects, ds_en, ds_target, tag):
    """Load paired EN/target questions."""
    en_by_pos = defaultdict(list)
    tgt_by_pos = defaultdict(list)
    for row in ds_en:
        if row["Subject"] in subjects:
            en_by_pos[row["Subject"]].append(row)
    for row in ds_target:
        if row["Subject"] in subjects:
            tgt_by_pos[row["Subject"]].append(row)

    pairs = []
    for subj in subjects:
        en_list = en_by_pos[subj]
        tgt_list = tgt_by_pos[subj]
        n = min(len(en_list), len(tgt_list), MAX_PER_DOMAIN)
        for i in range(n):
            if en_list[i]["Answer"] == tgt_list[i]["Answer"]:
                pairs.append({
                    "subject": subj,
                    "domain": tag,
                    "en_question": en_list[i]["Question"],
                    "en_options": {k: en_list[i][k] for k in ["A", "B", "C", "D"]},
                    "tgt_question": tgt_list[i]["Question"],
                    "tgt_options": {k: tgt_list[i][k] for k in ["A", "B", "C", "D"]},
                    "answer": en_list[i]["Answer"],
                })
    return pairs


def format_mcq(q, options):
    text = f"Question: {q}\n"
    for key in ["A", "B", "C", "D"]:
        text += f"{key}. {options[key]}\n"
    text += "Answer:"
    return text


def get_answer_probs(model, tokenizer, prompt, answer_tids):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    last_logits = outputs.logits[0, -1, :]
    letter_logits = torch.tensor([last_logits[tid].item() for tid in answer_tids])
    probs = torch.softmax(letter_logits, dim=0)
    return {l: p.item() for l, p in zip(["A", "B", "C", "D"], probs)}


def eval_pairs(model, tokenizer, pairs, lang_key, answer_tids, label=""):
    q_key = f"{lang_key}_question"
    o_key = f"{lang_key}_options"
    results = []
    for i, pair in enumerate(pairs):
        prompt = format_mcq(pair[q_key], pair[o_key])
        probs = get_answer_probs(model, tokenizer, prompt, answer_tids)
        predicted = max(probs, key=probs.get)
        results.append({
            "subject": pair["subject"],
            "correct": predicted == pair["answer"],
        })
        if (i + 1) % 100 == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"    [{i+1}/{len(pairs)}] {label}: {acc:.1%}")
    return results


def identify_language_features(model, tokenizer, sae, en_pairs, tgt_pairs,
                                layer, n_samples=50, top_k=10):
    """Find features that fire on target-language text but not English.

    We mix domains to find features that are LANGUAGE-general, not domain-specific.
    """
    print(f"\n  Identifying target-language features at layer {layer}...")

    # Take a mixed sample from multiple domains
    en_sample = en_pairs[:n_samples]
    tgt_sample = tgt_pairs[:n_samples]

    def collect(pairs, lang_key):
        feat_means = []
        q_key = f"{lang_key}_question"
        o_key = f"{lang_key}_options"
        for pair in pairs:
            captured = []
            def hook_fn(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                captured.append(o.detach())
            h = get_layer(model, layer).register_forward_hook(hook_fn)
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
        return torch.stack(feat_means)

    en_feats = collect(en_sample, "en")
    tgt_feats = collect(tgt_sample, "tgt")

    en_mean = en_feats.mean(dim=0)
    tgt_mean = tgt_feats.mean(dim=0)
    tgt_fires = (tgt_feats > 0).float().mean(dim=0)
    en_fires = (en_feats > 0).float().mean(dim=0)

    # Score: high target activation, low English, fires consistently on target
    eps = 0.01
    ratio = (tgt_mean + eps) / (en_mean + eps)
    score = ratio * tgt_fires * (tgt_fires > 0.8).float()
    score = score * (tgt_mean > 1.0).float()

    top = score.topk(top_k)
    features = []
    for i in range(top_k):
        idx = top.indices[i].item()
        if top.values[i].item() == 0:
            continue
        features.append({
            "feature_idx": idx,
            "tgt_mean": tgt_mean[idx].item(),
            "en_mean": en_mean[idx].item(),
            "tgt_fire_rate": tgt_fires[idx].item(),
            "en_fire_rate": en_fires[idx].item(),
            "score": top.values[i].item(),
        })

    print(f"    Top {len(features)} language features:")
    for f in features[:5]:
        print(f"      idx={f['feature_idx']}: tgt={f['tgt_mean']:.1f} en={f['en_mean']:.1f} fires={f['tgt_fire_rate']:.2f}")

    return features


@contextmanager
def steer_features(model, sae, layer, feature_deltas):
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


def run_rescue(model, tokenizer, sae, layer, features, strength,
               pairs, rescuable_idxs, control_idxs, en_results, answer_tids):
    """Run steering at given strength on EN questions."""
    feature_deltas = {
        f["feature_idx"]: strength * f["tgt_mean"]
        for f in features
    } if strength > 0 else {}

    test_idxs = sorted(set(rescuable_idxs + control_idxs))
    rescuable_now = 0
    control_still = 0

    for idx in test_idxs:
        pair = pairs[idx]
        prompt = format_mcq(pair["en_question"], pair["en_options"])
        if strength > 0:
            with steer_features(model, sae, layer, feature_deltas):
                probs = get_answer_probs(model, tokenizer, prompt, answer_tids)
        else:
            probs = get_answer_probs(model, tokenizer, prompt, answer_tids)
        predicted = max(probs, key=probs.get)
        correct = predicted == pair["answer"]

        if idx in rescuable_idxs and correct:
            rescuable_now += 1
        if idx in control_idxs and correct:
            control_still += 1

    broken = len(control_idxs) - control_still
    return rescuable_now, broken


def main():
    print("=" * 70)
    print("v2 GENERALIZATION: Cross-language, cross-domain rescue")
    print("=" * 70)

    print("\n--- Loading model and SAE ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16)
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    print(f"  Model: {MODEL_ID}")
    print(f"  SAE layer {STEER_LAYER}, d_sae={sae.cfg.d_sae}")

    answer_tids = [tokenizer.encode(f" {l}", add_special_tokens=False)[-1] for l in ["A", "B", "C", "D"]]

    # Load EN dataset once
    print("\n--- Loading MMMLU EN ---")
    ds_en = load_dataset("openai/MMMLU", "default", split="test")

    all_results = {}

    for lang_key, lang_config in LANGUAGES.items():
        print(f"\n{'=' * 70}")
        print(f"LANGUAGE: {lang_key} ({lang_config})")
        print(f"{'=' * 70}")

        ds_tgt = load_dataset("openai/MMMLU", lang_config, split="test")

        # First, identify target-language features using a MIX of domains
        print(f"\n--- Identifying generic {lang_key} features ---")
        # Mix 5 from each domain for feature ID
        mixed_pairs = []
        for domain, subjects in DOMAINS.items():
            dom_pairs = load_paired(subjects, ds_en, ds_tgt, domain)
            mixed_pairs.extend(dom_pairs[:10])  # 10 per domain

        print(f"  Feature ID sample: {len(mixed_pairs)} pairs (mixed domains)")
        features = identify_language_features(
            model, tokenizer, sae, mixed_pairs, mixed_pairs,
            STEER_LAYER, n_samples=N_FEATURE_ID_SAMPLES, top_k=N_TOP_FEATURES
        )

        lang_results = {"features": features, "domains": {}}

        # Now for each domain: baseline + rescue
        for domain, subjects in DOMAINS.items():
            print(f"\n--- Domain: {domain} ---")
            pairs = load_paired(subjects, ds_en, ds_tgt, domain)
            print(f"  {len(pairs)} paired questions")

            if len(pairs) < 30:
                print("  Too few questions, skipping.")
                continue

            # Baseline
            print(f"  Baseline EN...")
            en_results = eval_pairs(model, tokenizer, pairs, "en", answer_tids, f"en-{domain}")
            print(f"  Baseline {lang_key}...")
            tgt_results = eval_pairs(model, tokenizer, pairs, "tgt", answer_tids, f"{lang_key}-{domain}")

            en_acc = sum(r["correct"] for r in en_results) / len(en_results)
            tgt_acc = sum(r["correct"] for r in tgt_results) / len(tgt_results)
            gap = tgt_acc - en_acc
            print(f"  EN={en_acc:.1%}  {lang_key}={tgt_acc:.1%}  Gap={gap:+.1%}")

            # Identify rescuable + control
            rescuable_idxs = [i for i, (e, t) in enumerate(zip(en_results, tgt_results))
                              if t["correct"] and not e["correct"]]
            control_idxs = [i for i, r in enumerate(en_results) if r["correct"]][:50]

            if len(rescuable_idxs) < 10:
                print(f"  Only {len(rescuable_idxs)} rescuable -- reverse gap, skipping rescue")
                lang_results["domains"][domain] = {
                    "en_acc": en_acc, "tgt_acc": tgt_acc, "gap": gap,
                    "n_rescuable": len(rescuable_idxs),
                    "skipped": True,
                }
                continue

            # Rescue at multiple strengths
            print(f"  Rescuable: {len(rescuable_idxs)}, Control: {len(control_idxs)}")
            strength_results = {}
            for strength in STEER_MULTIPLIERS:
                rescued, broken = run_rescue(
                    model, tokenizer, sae, STEER_LAYER, features, strength,
                    pairs, rescuable_idxs, control_idxs, en_results, answer_tids
                )
                net = rescued - broken
                rate = rescued / len(rescuable_idxs) if rescuable_idxs else 0
                print(f"    {strength}x: rescued {rescued}/{len(rescuable_idxs)} "
                      f"({rate:.1%}), broken {broken}/{len(control_idxs)}, net {net:+d}")
                strength_results[strength] = {
                    "rescued": rescued, "broken": broken, "net": net, "rate": rate
                }

            lang_results["domains"][domain] = {
                "en_acc": en_acc, "tgt_acc": tgt_acc, "gap": gap,
                "n_rescuable": len(rescuable_idxs), "n_control": len(control_idxs),
                "by_strength": strength_results,
            }

        all_results[lang_key] = lang_results

    # Summary matrix
    print("\n" + "=" * 70)
    print("GENERALIZATION MATRIX")
    print("=" * 70)

    for lang_key in LANGUAGES:
        print(f"\n--- {lang_key.upper()} ---")
        print(f"{'Domain':<15s} {'Gap':>7s} {'N resc':>7s} {'Best Mult':>10s} "
              f"{'Rescued':>10s} {'Broken':>8s} {'Net':>6s}")
        lang_res = all_results[lang_key]
        for domain, data in lang_res["domains"].items():
            if data.get("skipped"):
                print(f"{domain:<15s} {data['gap']:+7.1%} {data['n_rescuable']:7d} "
                      f"{'-SKIPPED-':>10s}")
                continue
            # Best strength
            by_s = data["by_strength"]
            best_s, best_data = max(by_s.items(), key=lambda x: x[1]["net"])
            print(f"{domain:<15s} {data['gap']:+7.1%} {data['n_rescuable']:7d} "
                  f"{best_s:9.1f}x {best_data['rescued']:4d}/{data['n_rescuable']:<4d} "
                  f"{best_data['broken']:4d}/{data['n_control']:<3d} {best_data['net']:>+6d}")

    # Save
    with open("results/v2_generalization.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results/v2_generalization.json")

    del model, sae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
