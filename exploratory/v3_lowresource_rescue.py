"""
v3 Low-Resource Rescue: The ORIGINAL hypothesis, done right.

Premise (from design Claude, now properly framed):
  LLMs are documented to perform worse on domain-specific tasks in low-resource
  languages. Can we rescue weaker-language performance by amplifying features
  from a stronger language (English, Spanish)?

This is the correct direction. Our previous v2 experiment was accidentally
measuring Arabic vs Spanish (because MMMLU "default" is non-English); the
apparent "reverse gap" was Arabic being weaker than Spanish, which IS the
direction we care about -- we were just mislabeling Arabic as English.

Experiment structure:

Phase A: Baselines across languages
  EN (cais/mmlu), ES, AR, SW, YO on medical MCQ. Find biggest gaps.

Phase B: Pick the target -- weakest language with reasonable baseline.
  Hypothesis: Arabic, Swahili, or Yoruba shows largest gap vs English.

Phase C: Identify English features on paired EN-target data.
  Single top English feature used for rescue (top-3 as comparison).

Phase D: Rescue -- amplify English features during target-language inference.
  Full benchmark evaluation (all paired items).
  Bootstrap 95% CIs on accuracy delta.

Phase E: Random-feature control.
  Amplify a random English-firing feature at same strength.
  Establishes the noise floor.

Phase F: Alternative donor.
  Also try Spanish features as donor, if ES baseline > target baseline.

This addresses all 5 criticisms from v2:
  1. Single feature (primary), top-3 as comparison
  2. Full benchmark evaluation (not subset)
  3. Proper pairing: cais/mmlu for real English, MMMLU configs for target languages
  4. English-wrapper protocol kept for comparability, explicitly noted
  5. Random-feature control included

Usage:
    python3 v3_lowresource_rescue.py
"""
import json
import os
import random
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

MEDICAL_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]

# Target languages to baseline (low-resource expected to be worse than EN)
TARGET_LANGS = {
    "ar": "AR_XY",    # Arabic
    "sw": "SW_KE",    # Swahili
    "yo": "YO_NG",    # Yoruba
    "hi": "HI_IN",    # Hindi
    "es": "ES_LA",    # Spanish (reference, should be close to EN)
}

STEER_MULTIPLIERS = [0, 1.0, 2.0, 3.0, 5.0]
SEED = 42
# ============================================================


def get_layer(model, idx):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[idx]
    return model.model.layers[idx]


def build_paired_medical(cais_ds, target_configs):
    """Build paired set across EN and multiple target languages."""
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    # EN indexed by subject, preserving order
    en_by_subj = defaultdict(list)
    for row in cais_ds:
        if row["subject"] in MEDICAL_SUBJECTS:
            en_by_subj[row["subject"]].append(row)

    # Each target indexed by subject, sorted by Unnamed: 0
    target_by_subj = {}
    for lang, ds in target_configs.items():
        by_subj = defaultdict(list)
        for row in ds:
            if row["Subject"] in MEDICAL_SUBJECTS:
                by_subj[row["Subject"]].append(row)
        for s in by_subj:
            by_subj[s].sort(key=lambda r: r["Unnamed: 0"])
        target_by_subj[lang] = by_subj

    pairs = []
    for subj in MEDICAL_SUBJECTS:
        en_list = en_by_subj.get(subj, [])
        # Align by position across all configs
        lens = [len(en_list)] + [len(target_by_subj[l].get(subj, [])) for l in target_configs]
        n = min(lens)

        for i in range(n):
            en_row = en_list[i]
            en_answer_letter = letter_map[en_row["answer"]]

            # Get target versions, verify answers all match
            target_versions = {}
            all_match = True
            for lang in target_configs:
                tgt_row = target_by_subj[lang][subj][i]
                if tgt_row["Answer"] != en_answer_letter:
                    all_match = False
                    break
                target_versions[lang] = {
                    "question": tgt_row["Question"],
                    "options": {k: tgt_row[k] for k in ["A", "B", "C", "D"]},
                }

            if all_match:
                pair = {
                    "subject": subj,
                    "answer": en_answer_letter,
                    "en_question": en_row["question"],
                    "en_options": {letter_map[j]: en_row["choices"][j] for j in range(4)},
                }
                for lang, v in target_versions.items():
                    pair[f"{lang}_question"] = v["question"]
                    pair[f"{lang}_options"] = v["options"]
                pairs.append(pair)

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


def evaluate_all(model, tokenizer, pairs, lang_key, answer_tids, label="",
                 intervention=None):
    q_key = f"{lang_key}_question"
    o_key = f"{lang_key}_options"
    results = []
    for i, pair in enumerate(pairs):
        prompt = format_mcq(pair[q_key], pair[o_key])
        if intervention is not None:
            with intervention():
                probs = get_answer_probs(model, tokenizer, prompt, answer_tids)
        else:
            probs = get_answer_probs(model, tokenizer, prompt, answer_tids)
        predicted = max(probs, key=probs.get)
        results.append({
            "subject": pair["subject"],
            "actual": pair["answer"],
            "predicted": predicted,
            "correct": predicted == pair["answer"],
        })
        if (i + 1) % 100 == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"    [{i+1}/{len(pairs)}] {label}: {acc:.1%}")
    return results


def bootstrap_ci(results, n_boot=1000, ci=0.95):
    correct = [int(r["correct"]) for r in results]
    if not correct:
        return 0.0, (0.0, 0.0)
    acc = float(np.mean(correct))
    rng = np.random.default_rng(SEED)
    resamples = rng.choice(correct, size=(n_boot, len(correct)), replace=True)
    accs = resamples.mean(axis=1)
    lo = float(np.percentile(accs, (1 - ci) / 2 * 100))
    hi = float(np.percentile(accs, (1 + ci) / 2 * 100))
    return acc, (lo, hi)


def identify_features(model, tokenizer, sae, pairs, source_lang, contrast_lang,
                       layer, n_samples=40, top_k=10):
    """Find features that fire on source_lang but not contrast_lang."""
    print(f"  Collecting activations (n={n_samples}) for {source_lang} vs {contrast_lang}...")
    sample = pairs[:n_samples]

    def collect(lang_key):
        feat_means = []
        for pair in sample:
            captured = []
            def hook_fn(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                captured.append(o.detach())
            h = get_layer(model, layer).register_forward_hook(hook_fn)
            prompt = format_mcq(pair[f"{lang_key}_question"], pair[f"{lang_key}_options"])
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

    src_acts = collect(source_lang)
    ctr_acts = collect(contrast_lang)

    src_mean = src_acts.mean(dim=0)
    ctr_mean = ctr_acts.mean(dim=0)
    src_fires = (src_acts > 0).float().mean(dim=0)

    eps = 0.01
    score = (src_mean + eps) / (ctr_mean + eps) * src_fires
    score = score * (src_fires > 0.9).float() * (src_mean > 50).float()

    top = score.topk(top_k)
    features = []
    for i in range(top_k):
        idx = top.indices[i].item()
        if top.values[i].item() == 0:
            continue
        features.append({
            "feature_idx": idx,
            "src_mean": src_mean[idx].item(),
            "ctr_mean": ctr_mean[idx].item(),
            "src_fires": src_fires[idx].item(),
        })
    return features


def pick_random_control(model, tokenizer, sae, pairs, source_lang, layer,
                         target_mean, exclude, n_samples=20):
    """Pick a random feature firing on source_lang with similar activation magnitude."""
    sample = pairs[:n_samples]
    feat_means = []
    for pair in sample:
        captured = []
        def hook_fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            captured.append(o.detach())
        h = get_layer(model, layer).register_forward_hook(hook_fn)
        prompt = format_mcq(pair[f"{source_lang}_question"], pair[f"{source_lang}_options"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            model(**inputs)
        h.remove()
        resid = captured[0].to(sae.dtype)
        features = sae.encode(resid)
        feat_means.append(features.float().mean(dim=1).squeeze(0).cpu())
        del captured, features, resid
        torch.cuda.empty_cache()

    mat = torch.stack(feat_means)
    mean = mat.mean(dim=0)
    fires = (mat > 0).float().mean(dim=0)
    mask = (fires > 0.9) & (mean > target_mean * 0.3) & (mean < target_mean * 3)
    candidates = [i for i in torch.where(mask)[0].tolist() if i not in exclude]
    if not candidates:
        top = mean.topk(50).indices.tolist()
        candidates = [c for c in top if c not in exclude]

    rng = random.Random(SEED)
    chosen = rng.choice(candidates)
    return {
        "feature_idx": chosen,
        "src_mean": mean[chosen].item(),
        "ctr_mean": 0.0,
        "src_fires": fires[chosen].item(),
    }


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


def run_rescue_sweep(model, tokenizer, sae, pairs, victim_lang, features,
                      mean_key, answer_tids, strengths, label, baseline_acc):
    """Sweep over steering strengths, return per-strength results."""
    results = {}
    for s in strengths:
        if s > 0:
            feature_deltas = {f["feature_idx"]: s * f[mean_key] for f in features}
        else:
            feature_deltas = {}
        intervention = (lambda fd=feature_deltas: steer_features(model, sae, STEER_LAYER, fd))
        print(f"  {label} at {s}x...")
        res = evaluate_all(model, tokenizer, pairs, victim_lang, answer_tids,
                            f"{label}-{s}x", intervention=intervention)
        acc, ci = bootstrap_ci(res)
        delta = acc - baseline_acc
        print(f"    Acc={acc:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]  Δ={delta:+.2%}")
        results[s] = {
            "accuracy": acc,
            "ci_95": list(ci),
            "delta_vs_baseline": delta,
            "predictions": [{"correct": r["correct"], "subject": r["subject"]} for r in res],
        }
    return results


def main():
    print("=" * 70)
    print("v3 LOW-RESOURCE RESCUE: EN features to rescue weaker-language performance")
    print("=" * 70)

    # Load model + SAE
    print("\n--- Loading model + SAE ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16)
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    answer_tids = [tokenizer.encode(f" {l}", add_special_tokens=False)[-1] for l in ["A", "B", "C", "D"]]
    print(f"  Model: {MODEL_ID}, SAE layer {STEER_LAYER}")

    # Load datasets
    print("\n--- Loading datasets ---")
    cais_mmlu = load_dataset("cais/mmlu", "all", split="test")
    target_configs = {}
    for lang_key, lang_config in TARGET_LANGS.items():
        print(f"  Loading {lang_config}...")
        target_configs[lang_key] = load_dataset("openai/MMMLU", lang_config, split="test")

    # Build paired set
    print("\n--- Building paired set (EN + all target languages) ---")
    pairs = build_paired_medical(cais_mmlu, target_configs)
    print(f"Paired items: {len(pairs)}")

    # Phase A: baselines
    print("\n" + "=" * 70)
    print("PHASE A: Baselines across all languages")
    print("=" * 70)

    baseline_results = {}
    print("\n  EN (cais/mmlu)...")
    en_res = evaluate_all(model, tokenizer, pairs, "en", answer_tids, "en-med")
    en_acc, en_ci = bootstrap_ci(en_res)
    baseline_results["en"] = {"accuracy": en_acc, "ci_95": list(en_ci)}
    print(f"  EN medical: {en_acc:.1%} [{en_ci[0]:.1%}, {en_ci[1]:.1%}]")

    for lang_key in TARGET_LANGS:
        print(f"\n  {lang_key}...")
        res = evaluate_all(model, tokenizer, pairs, lang_key, answer_tids, f"{lang_key}-med")
        acc, ci = bootstrap_ci(res)
        baseline_results[lang_key] = {"accuracy": acc, "ci_95": list(ci)}
        gap = en_acc - acc
        print(f"  {lang_key} medical: {acc:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]  gap vs EN: {gap:+.1%}")

    # Summary
    print("\n--- Baseline summary ---")
    print(f"{'Lang':>8s} {'Acc':>8s} {'CI_low':>8s} {'CI_hi':>8s} {'Gap (EN-X)':>12s}")
    print(f"{'en':>8s} {en_acc:8.1%} {en_ci[0]:8.1%} {en_ci[1]:8.1%} {'--':>12s}")
    for lang in TARGET_LANGS:
        r = baseline_results[lang]
        gap = en_acc - r["accuracy"]
        print(f"{lang:>8s} {r['accuracy']:8.1%} {r['ci_95'][0]:8.1%} {r['ci_95'][1]:8.1%} {gap:>+12.1%}")

    # Phase B: pick victim (largest gap, but nonzero baseline)
    gaps = {lang: en_acc - baseline_results[lang]["accuracy"] for lang in TARGET_LANGS}
    # Victim must have baseline > 30% (else near random, nothing to rescue)
    valid_victims = [l for l in gaps if baseline_results[l]["accuracy"] > 0.30]
    if not valid_victims:
        print("\nNo valid victim languages (all below 30% baseline).")
        return

    victim_lang = max(valid_victims, key=lambda l: gaps[l])
    victim_baseline = baseline_results[victim_lang]["accuracy"]
    print(f"\nVictim language chosen: {victim_lang} (gap={gaps[victim_lang]:+.1%}, baseline={victim_baseline:.1%})")

    # Phase C: identify English features (EN vs victim)
    print("\n" + "=" * 70)
    print(f"PHASE C: Identify EN features (EN vs {victim_lang})")
    print("=" * 70)

    en_features = identify_features(model, tokenizer, sae, pairs, "en", victim_lang,
                                      STEER_LAYER, n_samples=40, top_k=10)
    if not en_features:
        print("No EN-specific features found.")
        return

    print(f"\n  Top 5 EN features (EN-specific, not in {victim_lang}):")
    for f in en_features[:5]:
        print(f"    idx={f['feature_idx']}: en_mean={f['src_mean']:.1f} "
              f"{victim_lang}_mean={f['ctr_mean']:.1f} en_fires={f['src_fires']:.2f}")

    top_en = en_features[0]

    # Phase D: SINGLE-feature rescue on victim, full benchmark
    print("\n" + "=" * 70)
    print(f"PHASE D: Single EN feature ({top_en['feature_idx']}) on {victim_lang}")
    print("=" * 70)

    single_results = run_rescue_sweep(
        model, tokenizer, sae, pairs, victim_lang, [top_en],
        "src_mean", answer_tids, STEER_MULTIPLIERS, "single-en", victim_baseline
    )

    # Phase E: Random control
    print("\n" + "=" * 70)
    print("PHASE E: Random-feature control")
    print("=" * 70)

    random_feature = pick_random_control(
        model, tokenizer, sae, pairs, "en", STEER_LAYER,
        target_mean=top_en["src_mean"],
        exclude=[f["feature_idx"] for f in en_features],
    )
    print(f"  Random feature chosen: {random_feature['feature_idx']} "
          f"(en_mean={random_feature['src_mean']:.1f})")

    random_results = run_rescue_sweep(
        model, tokenizer, sae, pairs, victim_lang, [random_feature],
        "src_mean", answer_tids, STEER_MULTIPLIERS, "random", victim_baseline
    )

    # Phase F: Top-3 EN features (for comparison)
    print("\n" + "=" * 70)
    print("PHASE F: Top-3 EN features")
    print("=" * 70)

    top3_results = run_rescue_sweep(
        model, tokenizer, sae, pairs, victim_lang, en_features[:3],
        "src_mean", answer_tids, STEER_MULTIPLIERS, "top3-en", victim_baseline
    )

    # Phase G: Spanish features as donor (if ES > victim)
    es_acc = baseline_results["es"]["accuracy"]
    spanish_results = None
    if es_acc > victim_baseline + 0.02:
        print("\n" + "=" * 70)
        print(f"PHASE G: Spanish features as donor (ES={es_acc:.1%} > {victim_lang}={victim_baseline:.1%})")
        print("=" * 70)

        es_features = identify_features(model, tokenizer, sae, pairs, "es", victim_lang,
                                          STEER_LAYER, n_samples=40, top_k=10)
        if es_features:
            top_es = es_features[0]
            print(f"  Top ES feature: {top_es['feature_idx']} (es_mean={top_es['src_mean']:.1f})")
            spanish_results = run_rescue_sweep(
                model, tokenizer, sae, pairs, victim_lang, [top_es],
                "src_mean", answer_tids, STEER_MULTIPLIERS, "single-es", victim_baseline
            )

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nVictim: {victim_lang}, baseline: {victim_baseline:.1%}, gap to EN: {gaps[victim_lang]:+.1%}")
    print(f"\n{'Strength':>10s} {'Single EN':>14s} {'Random':>14s} {'Top-3 EN':>14s}"
          + (" {'Single ES':>14s}" if spanish_results else ""))
    for s in STEER_MULTIPLIERS:
        row = f"{s:>9.1f}x"
        for res, name in [(single_results, "single"), (random_results, "random"), (top3_results, "top3")]:
            r = res[s]
            row += f"  {r['accuracy']:6.1%} ({r['delta_vs_baseline']:+.2%})"
        if spanish_results:
            r = spanish_results[s]
            row += f"  {r['accuracy']:6.1%} ({r['delta_vs_baseline']:+.2%})"
        print(row)

    # Save
    output = {
        "model": MODEL_ID,
        "sae_id": SAE_ID,
        "n_paired": len(pairs),
        "baselines": baseline_results,
        "victim_lang": victim_lang,
        "victim_baseline": victim_baseline,
        "en_features": en_features,
        "random_feature": random_feature,
        "single_en_results": single_results,
        "random_results": random_results,
        "top3_en_results": top3_results,
        "spanish_results": spanish_results,
    }
    with open("/root/results/v3_lowresource_rescue.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to /root/results/v3_lowresource_rescue.json")

    del model, sae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
