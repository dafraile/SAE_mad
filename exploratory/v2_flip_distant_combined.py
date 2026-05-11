"""
v2 Flip + Distant + Combined: Three hypotheses in one experiment.

Hypothesis 1 (FLIP): Coding/CS domains should favor English (English-dominant
  training data for programming). If true, we should see EN > target-language
  gap, and amplifying ENGLISH features on target-language CS questions should
  rescue performance. This tests whether the rescue mechanism works
  symmetrically in both directions.

Hypothesis 2 (DISTANT): Test Chinese (ZH_CN) as a linguistically distant
  language. Previous results were on Romance languages (ES, FR) which share
  typology with English. Chinese tests whether the mechanism is universal
  or depends on linguistic similarity.

Hypothesis 3 (COMBINED): On English medical (our best rescue domain),
  amplify ES + FR features simultaneously. Does the rescue compound (+6 +4 → +10)
  or do they interfere?

Usage:
    python3 v2_flip_distant_combined.py
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
STEER_LAYER = 29
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = f"layer_{STEER_LAYER}_width_16k_l0_medium"

# Coding/CS domains -- hypothesis: EN should dominate here
CODING_SUBJECTS = [
    "college_computer_science",
    "high_school_computer_science",
    "computer_security",
    "machine_learning",
    "electrical_engineering",
]

# Medical for the combined test (our strongest rescue domain)
MEDICAL_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]

# Mixed domains for feature identification
FEATURE_ID_SUBJECTS = (
    MEDICAL_SUBJECTS[:2] + CODING_SUBJECTS[:2] + ["philosophy", "world_religions"]
)

LANGUAGES = {
    "es": "ES_LA",
    "fr": "FR_FR",
    "zh": "ZH_CN",
}

MAX_PER_DOMAIN = 200
STEER_MULTIPLIERS = [0, 1.0, 2.0, 3.0, 5.0]
N_TOP_FEATURES = 10
N_FEATURE_ID_SAMPLES = 40
# ============================================================


def get_layer(model, idx):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[idx]
    return model.model.layers[idx]


def load_paired(subjects, ds_en, ds_target, tag):
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


def collect_mean_feats(model, tokenizer, sae, pairs, lang_key, layer):
    q_key = f"{lang_key}_question"
    o_key = f"{lang_key}_options"
    feat_means = []
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


def find_preference_features(high_mean, low_mean, high_fires, direction_label, top_k=10):
    """Find features that fire high in one condition, low in another."""
    eps = 0.01
    ratio = (high_mean + eps) / (low_mean + eps)
    score = ratio * high_fires * (high_fires > 0.8).float() * (high_mean > 1.0).float()
    top = score.topk(top_k)
    features = []
    for i in range(top_k):
        idx = top.indices[i].item()
        if top.values[i].item() == 0:
            continue
        features.append({
            "feature_idx": idx,
            f"{direction_label}_mean": high_mean[idx].item(),
            "other_mean": low_mean[idx].item(),
            "fire_rate": high_fires[idx].item(),
            "score": top.values[i].item(),
        })
    return features


def identify_features_bidirectional(model, tokenizer, sae, pairs, layer, lang_key, top_k=10):
    """Find both target-language-preferred and English-preferred features."""
    print(f"\n  Collecting activations at layer {layer}...")
    en_feats = collect_mean_feats(model, tokenizer, sae, pairs, "en", layer)
    tgt_feats = collect_mean_feats(model, tokenizer, sae, pairs, "tgt", layer)

    en_mean = en_feats.mean(dim=0)
    tgt_mean = tgt_feats.mean(dim=0)
    en_fires = (en_feats > 0).float().mean(dim=0)
    tgt_fires = (tgt_feats > 0).float().mean(dim=0)

    # Target-preferred: fire on tgt, not on en
    tgt_prefer = find_preference_features(tgt_mean, en_mean, tgt_fires, "tgt", top_k)
    # English-preferred: fire on en, not on tgt
    en_prefer = find_preference_features(en_mean, tgt_mean, en_fires, "en", top_k)

    print(f"    Top {lang_key}-preferred features:")
    for f in tgt_prefer[:5]:
        print(f"      idx={f['feature_idx']}: tgt={f['tgt_mean']:.1f} en={f['other_mean']:.1f} fires={f['fire_rate']:.2f}")
    print(f"    Top EN-preferred features:")
    for f in en_prefer[:5]:
        print(f"      idx={f['feature_idx']}: en={f['en_mean']:.1f} tgt={f['other_mean']:.1f} fires={f['fire_rate']:.2f}")

    return tgt_prefer, en_prefer


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
               pairs, src_lang, rescuable_idxs, control_idxs, answer_tids, mean_key):
    """Run steering at given strength on source-language questions."""
    feature_deltas = {
        f["feature_idx"]: strength * f[mean_key]
        for f in features
    } if strength > 0 else {}

    test_idxs = sorted(set(rescuable_idxs + control_idxs))
    rescuable_now = 0
    control_still = 0
    q_key = f"{src_lang}_question"
    o_key = f"{src_lang}_options"

    for idx in test_idxs:
        pair = pairs[idx]
        prompt = format_mcq(pair[q_key], pair[o_key])
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


def rescue_experiment(model, tokenizer, sae, pairs, src_lang, tgt_src_lang,
                      features, mean_key, answer_tids, label=""):
    """Generic rescue: steer src_lang inference using features, rescue toward tgt performance."""
    print(f"\n  Baseline {src_lang}...")
    src_results = eval_pairs(model, tokenizer, pairs, src_lang, answer_tids, f"{src_lang}-{label}")
    print(f"  Baseline {tgt_src_lang}...")
    tgt_results = eval_pairs(model, tokenizer, pairs, tgt_src_lang, answer_tids, f"{tgt_src_lang}-{label}")

    src_acc = sum(r["correct"] for r in src_results) / len(src_results)
    tgt_acc = sum(r["correct"] for r in tgt_results) / len(tgt_results)
    gap = tgt_acc - src_acc
    print(f"  {src_lang}={src_acc:.1%}  {tgt_src_lang}={tgt_acc:.1%}  Gap={gap:+.1%}")

    rescuable_idxs = [i for i, (s, t) in enumerate(zip(src_results, tgt_results))
                      if t["correct"] and not s["correct"]]
    control_idxs = [i for i, r in enumerate(src_results) if r["correct"]][:50]

    if len(rescuable_idxs) < 10:
        print(f"  Only {len(rescuable_idxs)} rescuable. Skipping rescue.")
        return {
            "src_acc": src_acc, "tgt_acc": tgt_acc, "gap": gap,
            "n_rescuable": len(rescuable_idxs), "skipped": True,
        }

    print(f"  Rescuable: {len(rescuable_idxs)}, Control: {len(control_idxs)}")
    strength_results = {}
    for strength in STEER_MULTIPLIERS:
        rescued, broken = run_rescue(
            model, tokenizer, sae, STEER_LAYER, features, strength,
            pairs, src_lang, rescuable_idxs, control_idxs, answer_tids, mean_key
        )
        net = rescued - broken
        rate = rescued / len(rescuable_idxs)
        print(f"    {strength}x: rescued {rescued}/{len(rescuable_idxs)} ({rate:.1%}), broken {broken}/{len(control_idxs)}, net {net:+d}")
        strength_results[strength] = {"rescued": rescued, "broken": broken, "net": net, "rate": rate}

    return {
        "src_acc": src_acc, "tgt_acc": tgt_acc, "gap": gap,
        "n_rescuable": len(rescuable_idxs), "n_control": len(control_idxs),
        "by_strength": strength_results,
    }


def main():
    print("=" * 70)
    print("v2 FLIP + DISTANT + COMBINED")
    print("=" * 70)

    print("\n--- Loading model and SAE ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16)
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    answer_tids = [tokenizer.encode(f" {l}", add_special_tokens=False)[-1] for l in ["A", "B", "C", "D"]]
    print(f"  Model: {MODEL_ID}, SAE layer {STEER_LAYER}")

    ds_en = load_dataset("openai/MMMLU", "default", split="test")

    all_results = {"flip": {}, "distant": {}, "combined": {}}
    language_features = {}  # Store per-language features for combined test

    # ========================================================
    # PART 1: FLIP — test coding domains where EN should dominate
    # ========================================================
    print("\n" + "=" * 70)
    print("PART 1: FLIP — Coding domains (hypothesis: EN > target)")
    print("=" * 70)

    for lang_key, lang_config in LANGUAGES.items():
        print(f"\n--- Language: {lang_key} ({lang_config}) ---")
        ds_tgt = load_dataset("openai/MMMLU", lang_config, split="test")

        # Identify features using MIX of domains (generic language features)
        mixed = load_paired(FEATURE_ID_SUBJECTS, ds_en, ds_tgt, "mixed")[:N_FEATURE_ID_SAMPLES]
        print(f"  Identifying features ({len(mixed)} samples, bidirectional)...")
        tgt_feats, en_feats = identify_features_bidirectional(
            model, tokenizer, sae, mixed, STEER_LAYER, lang_key, top_k=N_TOP_FEATURES)

        language_features[lang_key] = {"tgt_features": tgt_feats, "en_features": en_feats}

        # Test on coding domains
        coding_pairs = load_paired(CODING_SUBJECTS, ds_en, ds_tgt, "coding")
        print(f"\n  Coding pairs: {len(coding_pairs)}")

        if len(coding_pairs) < 30:
            continue

        # Baseline: does EN > target here?
        print(f"  Baseline EN...")
        en_results = eval_pairs(model, tokenizer, coding_pairs, "en", answer_tids, f"en-coding")
        print(f"  Baseline {lang_key}...")
        tgt_results = eval_pairs(model, tokenizer, coding_pairs, "tgt", answer_tids, f"{lang_key}-coding")

        en_acc = sum(r["correct"] for r in en_results) / len(en_results)
        tgt_acc = sum(r["correct"] for r in tgt_results) / len(tgt_results)
        gap_en_minus_tgt = en_acc - tgt_acc
        print(f"  EN={en_acc:.1%}  {lang_key}={tgt_acc:.1%}  EN-{lang_key}={gap_en_minus_tgt:+.1%}")

        # If EN dominates here, try amplifying EN features on target language
        if gap_en_minus_tgt > 0.02:  # EN wins by >2pts
            print(f"\n  EN dominates! Amplifying EN features on {lang_key} coding questions...")
            # Find "rescuable" tgt questions (EN right, tgt wrong)
            rescuable_idxs = [i for i, (e, t) in enumerate(zip(en_results, tgt_results))
                              if e["correct"] and not t["correct"]]
            control_idxs = [i for i, r in enumerate(tgt_results) if r["correct"]][:50]

            print(f"    Rescuable {lang_key}: {len(rescuable_idxs)}, Control: {len(control_idxs)}")
            if len(rescuable_idxs) >= 10:
                strength_results = {}
                for strength in STEER_MULTIPLIERS:
                    rescued, broken = run_rescue(
                        model, tokenizer, sae, STEER_LAYER, en_feats, strength,
                        coding_pairs, "tgt", rescuable_idxs, control_idxs,
                        answer_tids, "en_mean"
                    )
                    net = rescued - broken
                    rate = rescued / len(rescuable_idxs)
                    print(f"    {strength}x: rescued {rescued}/{len(rescuable_idxs)} ({rate:.1%}), "
                          f"broken {broken}/{len(control_idxs)}, net {net:+d}")
                    strength_results[strength] = {"rescued": rescued, "broken": broken, "net": net, "rate": rate}

                all_results["flip"][lang_key] = {
                    "en_acc": en_acc, "tgt_acc": tgt_acc, "gap_en_minus_tgt": gap_en_minus_tgt,
                    "n_rescuable": len(rescuable_idxs), "n_control": len(control_idxs),
                    "direction": f"en->{lang_key}",
                    "by_strength": strength_results,
                }
        else:
            print(f"  EN does NOT dominate coding for {lang_key} (gap={gap_en_minus_tgt:+.1%})")
            # Still record baseline
            all_results["flip"][lang_key] = {
                "en_acc": en_acc, "tgt_acc": tgt_acc, "gap_en_minus_tgt": gap_en_minus_tgt,
                "reversed_gap": True,
            }

    # ========================================================
    # PART 2: DISTANT — Chinese on medical (already covered in PART 1 feature ID)
    # ========================================================
    print("\n" + "=" * 70)
    print("PART 2: DISTANT — Chinese medical rescue")
    print("=" * 70)

    if "zh" in language_features:
        ds_zh = load_dataset("openai/MMMLU", "ZH_CN", split="test")
        med_pairs = load_paired(MEDICAL_SUBJECTS, ds_en, ds_zh, "medical")
        print(f"  Medical pairs: {len(med_pairs)}")

        # Standard rescue: amplify ZH features on EN medical
        zh_features = language_features["zh"]["tgt_features"]
        result = rescue_experiment(
            model, tokenizer, sae, med_pairs, "en", "tgt",
            zh_features, "tgt_mean", answer_tids, label="medical-zh"
        )
        all_results["distant"]["zh_on_en_medical"] = result

    # ========================================================
    # PART 3: COMBINED — ES + FR features simultaneously on EN medical
    # ========================================================
    print("\n" + "=" * 70)
    print("PART 3: COMBINED — ES + FR features on EN medical")
    print("=" * 70)

    if "es" in language_features and "fr" in language_features:
        ds_es = load_dataset("openai/MMMLU", "ES_LA", split="test")
        med_pairs = load_paired(MEDICAL_SUBJECTS, ds_en, ds_es, "medical")
        print(f"  Medical pairs: {len(med_pairs)}")

        # Quick baseline
        en_results = eval_pairs(model, tokenizer, med_pairs, "en", answer_tids, "en-med")
        es_results = eval_pairs(model, tokenizer, med_pairs, "tgt", answer_tids, "es-med")
        en_acc = sum(r["correct"] for r in en_results) / len(en_results)
        es_acc = sum(r["correct"] for r in es_results) / len(es_results)
        print(f"  EN={en_acc:.1%}  ES={es_acc:.1%}  Gap={es_acc-en_acc:+.1%}")

        rescuable_idxs = [i for i, (e, s) in enumerate(zip(en_results, es_results))
                          if s["correct"] and not e["correct"]]
        control_idxs = [i for i, r in enumerate(en_results) if r["correct"]][:50]
        print(f"  Rescuable: {len(rescuable_idxs)}, Control: {len(control_idxs)}")

        es_features = language_features["es"]["tgt_features"]
        fr_features = language_features["fr"]["tgt_features"]

        # Three conditions: ES only, FR only, ES+FR combined
        print("\n  Condition: ES features only")
        es_results_by_s = {}
        for s in STEER_MULTIPLIERS:
            rescued, broken = run_rescue(
                model, tokenizer, sae, STEER_LAYER, es_features, s,
                med_pairs, "en", rescuable_idxs, control_idxs, answer_tids, "tgt_mean"
            )
            net = rescued - broken
            print(f"    {s}x: rescued {rescued}/{len(rescuable_idxs)}, broken {broken}/{len(control_idxs)}, net {net:+d}")
            es_results_by_s[s] = {"rescued": rescued, "broken": broken, "net": net}

        print("\n  Condition: FR features only")
        fr_results_by_s = {}
        for s in STEER_MULTIPLIERS:
            rescued, broken = run_rescue(
                model, tokenizer, sae, STEER_LAYER, fr_features, s,
                med_pairs, "en", rescuable_idxs, control_idxs, answer_tids, "tgt_mean"
            )
            net = rescued - broken
            print(f"    {s}x: rescued {rescued}/{len(rescuable_idxs)}, broken {broken}/{len(control_idxs)}, net {net:+d}")
            fr_results_by_s[s] = {"rescued": rescued, "broken": broken, "net": net}

        print("\n  Condition: ES + FR features combined")
        combined_features = es_features + fr_features  # Union
        combined_results_by_s = {}
        for s in STEER_MULTIPLIERS:
            rescued, broken = run_rescue(
                model, tokenizer, sae, STEER_LAYER, combined_features, s,
                med_pairs, "en", rescuable_idxs, control_idxs, answer_tids, "tgt_mean"
            )
            net = rescued - broken
            print(f"    {s}x: rescued {rescued}/{len(rescuable_idxs)}, broken {broken}/{len(control_idxs)}, net {net:+d}")
            combined_results_by_s[s] = {"rescued": rescued, "broken": broken, "net": net}

        all_results["combined"] = {
            "en_acc": en_acc, "es_acc": es_acc, "gap": es_acc - en_acc,
            "n_rescuable": len(rescuable_idxs), "n_control": len(control_idxs),
            "es_only": es_results_by_s,
            "fr_only": fr_results_by_s,
            "combined": combined_results_by_s,
        }

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\n--- PART 1: FLIP (does EN dominate in CS/coding?) ---")
    for lang, res in all_results["flip"].items():
        if res.get("reversed_gap"):
            print(f"  {lang}: EN={res['en_acc']:.1%}, {lang}={res['tgt_acc']:.1%} -- "
                  f"gap {res['gap_en_minus_tgt']:+.1%} (NOT EN-dominant)")
        else:
            best = max(res["by_strength"].items(), key=lambda x: x[1]["net"])
            s, d = best
            print(f"  {lang}: EN={res['en_acc']:.1%}, {lang}={res['tgt_acc']:.1%} "
                  f"(EN dominates +{res['gap_en_minus_tgt']:.1%}) -> "
                  f"rescue with EN feats @ {s}x: +{d['net']}")

    print("\n--- PART 2: DISTANT (Chinese rescuing EN medical) ---")
    if "zh_on_en_medical" in all_results["distant"]:
        res = all_results["distant"]["zh_on_en_medical"]
        if res.get("skipped"):
            print(f"  Skipped: {res['n_rescuable']} rescuable")
        else:
            best = max(res["by_strength"].items(), key=lambda x: x[1]["net"])
            s, d = best
            print(f"  EN={res['src_acc']:.1%}, ZH={res['tgt_acc']:.1%} (gap {res['gap']:+.1%}) -> "
                  f"best rescue @ {s}x: +{d['net']}/{res['n_rescuable']}")

    print("\n--- PART 3: COMBINED (ES + FR features together) ---")
    if all_results["combined"]:
        r = all_results["combined"]
        print(f"  N rescuable: {r['n_rescuable']}")
        for cond in ["es_only", "fr_only", "combined"]:
            best = max(r[cond].items(), key=lambda x: x[1]["net"])
            s, d = best
            print(f"  {cond:12s}: best {s}x -> rescued {d['rescued']}, broken {d['broken']}, net {d['net']:+d}")

    # Save
    with open("results/v2_flip_distant_combined.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results/v2_flip_distant_combined.json")

    del model, sae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
