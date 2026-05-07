"""
v3 Domain Rescue: Test whether language-AGNOSTIC medical features
can rescue cross-lingual performance gaps.

PREVIOUS FAILURE MODE: We tested language-specific features (e.g., feature
596 fires on ES, 0 on EN). These are pure language detectors with no
domain content. Amplifying them steers output language but doesn't
transplant knowledge.

NEW HYPOTHESIS: There may exist features that:
  - Fire on medical content across ALL languages we trust (EN, ES, FR medical)
  - Do NOT fire on non-medical content (EN, ES, FR philosophy/religion/facts)
  - Fire WEAKLY on target-language medical content (AR, YO) -- the knowledge
    pathway is there but underused

If such features exist, amplifying them during AR/YO inference might
retrieve the medical knowledge that's present but under-activated.

DESIGN:

Phase A: Baselines with real EN (cais/mmlu) and MMMLU for ES, FR, AR, YO.
  Separately on medical vs non-medical subjects.

Phase B: Feature identification (6-condition contrastive).
  Score features by: fires on medical across EN+ES+FR, does NOT fire on
  non-medical across EN+ES+FR. These are candidate domain-specific,
  language-agnostic features.

Phase C: Feature characterization on AR, YO.
  Do these features fire on AR-medical? YO-medical? If yes (weakly), the
  knowledge pathway exists. If no, the knowledge isn't there in that language.

Phase D: Rescue test.
  Amplify the top domain-specific feature during AR-medical and YO-medical
  inference. Full-benchmark evaluation with random-feature control.

All criticisms from v2 addressed:
  1. Single feature primary (top-k as comparison)
  2. Full-benchmark evaluation
  3. Real English from cais/mmlu
  4. English-wrapper protocol (kept for comparability, noted)
  5. Random-feature control included
  6. NEW: feature selection criterion targets domain, not language
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

NONMED_SUBJECTS = [
    "philosophy",
    "world_religions",
    "global_facts",
]

# Languages for feature IDENTIFICATION (we trust these have both med + nonmed signal)
TRUST_LANGS = {"es": "ES_LA", "fr": "FR_FR"}  # EN comes from cais/mmlu

# Languages to TEST rescue on (weaker)
VICTIM_LANGS = {"ar": "AR_XY", "yo": "YO_NG"}

STEER_MULTIPLIERS = [0, 1.0, 2.0, 3.0, 5.0]
N_CONTRAST_SAMPLES = 40
SEED = 42
# ============================================================


def get_layer(model, idx):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[idx]
    return model.model.layers[idx]


def load_paired(cais_ds, config_datasets, subjects):
    """Build paired set across EN + multiple configs for given subjects."""
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    en_by_subj = defaultdict(list)
    for row in cais_ds:
        if row["subject"] in subjects:
            en_by_subj[row["subject"]].append(row)

    cfg_by_subj = {}
    for lang, ds in config_datasets.items():
        by_s = defaultdict(list)
        for row in ds:
            if row["Subject"] in subjects:
                by_s[row["Subject"]].append(row)
        for s in by_s:
            by_s[s].sort(key=lambda r: r["Unnamed: 0"])
        cfg_by_subj[lang] = by_s

    pairs = []
    for subj in subjects:
        en_list = en_by_subj.get(subj, [])
        lens = [len(en_list)] + [len(cfg_by_subj[l].get(subj, [])) for l in config_datasets]
        n = min(lens)

        for i in range(n):
            en_row = en_list[i]
            en_ans = letter_map[en_row["answer"]]

            lang_versions = {}
            all_match = True
            for lang in config_datasets:
                tgt_row = cfg_by_subj[lang][subj][i]
                if tgt_row["Answer"] != en_ans:
                    all_match = False
                    break
                lang_versions[lang] = (tgt_row["Question"],
                                        {k: tgt_row[k] for k in ["A", "B", "C", "D"]})

            if all_match:
                pair = {
                    "subject": subj,
                    "answer": en_ans,
                    "en_question": en_row["question"],
                    "en_options": {letter_map[j]: en_row["choices"][j] for j in range(4)},
                }
                for lang, (q, opts) in lang_versions.items():
                    pair[f"{lang}_question"] = q
                    pair[f"{lang}_options"] = opts
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
            "correct": predicted == pair["answer"],
        })
        if (i + 1) % 200 == 0:
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


def collect_mean_acts(model, tokenizer, sae, pairs, lang_key, layer, n):
    feat_means = []
    for pair in pairs[:n]:
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


def find_domain_features(model, tokenizer, sae, med_pairs, nonmed_pairs,
                          trust_langs, layer, n_samples=40, top_k=20):
    """Find features that fire on medical across trusted languages, NOT on non-medical.

    Scoring: product of
      - min(med_mean across langs) / max(nonmed_mean across langs)  (domain specificity)
      - min(med_fires across langs)                                  (cross-lingual consistency)
    """
    print(f"\n  Collecting activations at layer {layer} across conditions...")

    med_acts = {}
    nonmed_acts = {}
    for lang in ["en"] + list(trust_langs):
        print(f"    {lang} medical...")
        med_acts[lang] = collect_mean_acts(model, tokenizer, sae, med_pairs, lang, layer, n_samples)
        print(f"    {lang} non-medical...")
        nonmed_acts[lang] = collect_mean_acts(model, tokenizer, sae, nonmed_pairs, lang, layer, n_samples)

    # Compute per-feature stats
    med_means = {lang: a.mean(dim=0) for lang, a in med_acts.items()}
    nonmed_means = {lang: a.mean(dim=0) for lang, a in nonmed_acts.items()}
    med_fires = {lang: (a > 0).float().mean(dim=0) for lang, a in med_acts.items()}
    nonmed_fires = {lang: (a > 0).float().mean(dim=0) for lang, a in nonmed_acts.items()}

    # Stack: shape [n_langs, d_sae]
    langs = ["en"] + list(trust_langs)
    med_mean_stack = torch.stack([med_means[l] for l in langs])
    nonmed_mean_stack = torch.stack([nonmed_means[l] for l in langs])
    med_fires_stack = torch.stack([med_fires[l] for l in langs])

    # Domain specificity: min med activation across langs divided by max nonmed
    eps = 0.1
    min_med_mean = med_mean_stack.min(dim=0).values
    max_nonmed_mean = nonmed_mean_stack.max(dim=0).values
    specificity = (min_med_mean + eps) / (max_nonmed_mean + eps)

    # Cross-lingual medical consistency
    min_med_fires = med_fires_stack.min(dim=0).values

    # Combined score
    score = specificity * min_med_fires

    # Filter: must fire on >50% of medical examples in EVERY trust language
    mask = (min_med_fires > 0.5).float()
    # Filter: must have meaningful activation
    mask = mask * (min_med_mean > 2.0).float()
    # Filter: must NOT fire heavily on non-medical in ANY language
    max_nonmed_fires = torch.stack([nonmed_fires[l] for l in langs]).max(dim=0).values
    mask = mask * (max_nonmed_fires < 0.5).float()

    score = score * mask

    top = score.topk(top_k)
    candidates = []
    for i in range(top_k):
        idx = top.indices[i].item()
        if top.values[i].item() == 0:
            continue
        candidates.append({
            "feature_idx": idx,
            "score": float(top.values[i].item()),
            "med_means": {l: float(med_means[l][idx].item()) for l in langs},
            "nonmed_means": {l: float(nonmed_means[l][idx].item()) for l in langs},
            "med_fires": {l: float(med_fires[l][idx].item()) for l in langs},
            "nonmed_fires": {l: float(nonmed_fires[l][idx].item()) for l in langs},
            "specificity": float(specificity[idx].item()),
            "min_med_fires": float(min_med_fires[idx].item()),
        })

    print(f"\n  Top {len(candidates)} language-agnostic medical features:")
    print(f"  {'idx':>6s} {'spec':>8s} {'min_fire':>8s} {'en_med':>8s} {'es_med':>8s} "
          f"{'fr_med':>8s} {'en_nonm':>8s} {'es_nonm':>8s} {'fr_nonm':>8s}")
    for c in candidates[:10]:
        mm = c["med_means"]
        nm = c["nonmed_means"]
        print(f"  {c['feature_idx']:6d} {c['specificity']:8.2f} {c['min_med_fires']:8.2f} "
              f"{mm.get('en',0):8.1f} {mm.get('es',0):8.1f} {mm.get('fr',0):8.1f} "
              f"{nm.get('en',0):8.1f} {nm.get('es',0):8.1f} {nm.get('fr',0):8.1f}")

    return candidates


def characterize_on_victim(model, tokenizer, sae, med_pairs, nonmed_pairs,
                            victim_lang, features, layer, n_samples=40):
    """Check if domain features fire on victim language medical content."""
    print(f"\n  Characterizing features on {victim_lang}...")
    victim_med = collect_mean_acts(model, tokenizer, sae, med_pairs, victim_lang, layer, n_samples)
    victim_nonmed = collect_mean_acts(model, tokenizer, sae, nonmed_pairs, victim_lang, layer, n_samples)

    victim_med_mean = victim_med.mean(dim=0)
    victim_nonmed_mean = victim_nonmed.mean(dim=0)
    victim_med_fires = (victim_med > 0).float().mean(dim=0)

    print(f"    {victim_lang} med activation on top {min(10, len(features))} features:")
    for c in features[:10]:
        idx = c["feature_idx"]
        c[f"{victim_lang}_med_mean"] = float(victim_med_mean[idx].item())
        c[f"{victim_lang}_nonmed_mean"] = float(victim_nonmed_mean[idx].item())
        c[f"{victim_lang}_med_fires"] = float(victim_med_fires[idx].item())
        print(f"      idx={idx}: {victim_lang}_med={c[f'{victim_lang}_med_mean']:.2f} "
              f"(fires {c[f'{victim_lang}_med_fires']:.0%}), "
              f"{victim_lang}_nonmed={c[f'{victim_lang}_nonmed_mean']:.2f}")

    return features


def pick_random_control(model, tokenizer, sae, pairs, lang, layer, target_mean,
                         exclude, n_samples=20):
    acts = collect_mean_acts(model, tokenizer, sae, pairs, lang, layer, n_samples)
    mean = acts.mean(dim=0)
    fires = (acts > 0).float().mean(dim=0)
    mask = (fires > 0.9) & (mean > target_mean * 0.3) & (mean < target_mean * 3)
    candidates = [i for i in torch.where(mask)[0].tolist() if i not in exclude]
    if not candidates:
        candidates = [i for i in mean.topk(50).indices.tolist() if i not in exclude]
    rng = random.Random(SEED)
    return {
        "feature_idx": rng.choice(candidates),
        "mean": float(mean[candidates[0]].item()) if candidates else target_mean,
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


def main():
    print("=" * 70)
    print("v3 DOMAIN RESCUE: language-agnostic medical features")
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
    trust_configs = {lang: load_dataset("openai/MMMLU", cfg, split="test")
                     for lang, cfg in TRUST_LANGS.items()}
    victim_configs = {lang: load_dataset("openai/MMMLU", cfg, split="test")
                      for lang, cfg in VICTIM_LANGS.items()}

    # Paired sets
    all_configs = {**trust_configs, **victim_configs}
    med_pairs = load_paired(cais_mmlu, all_configs, MEDICAL_SUBJECTS)
    nonmed_pairs = load_paired(cais_mmlu, all_configs, NONMED_SUBJECTS)
    print(f"  Medical pairs: {len(med_pairs)}")
    print(f"  Non-medical pairs: {len(nonmed_pairs)}")

    # Phase A: Baselines
    print("\n" + "=" * 70)
    print("PHASE A: Baselines")
    print("=" * 70)
    baselines = {}
    for lang in ["en"] + list(TRUST_LANGS) + list(VICTIM_LANGS):
        print(f"\n  {lang} medical...")
        res = evaluate_all(model, tokenizer, med_pairs, lang, answer_tids, f"{lang}-med")
        acc, ci = bootstrap_ci(res)
        baselines[f"{lang}_medical"] = {"accuracy": acc, "ci_95": list(ci),
                                         "predictions": [{"correct": r["correct"], "subject": r["subject"]} for r in res]}
        print(f"    {lang} medical: {acc:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]")

    print("\n--- Baseline summary ---")
    for lang, data in baselines.items():
        gap = baselines["en_medical"]["accuracy"] - data["accuracy"]
        print(f"  {lang}: {data['accuracy']:.1%} (gap vs EN: {gap:+.1%})")

    # Phase B: Find domain-specific, language-agnostic features
    print("\n" + "=" * 70)
    print("PHASE B: Identify language-agnostic medical features")
    print("=" * 70)
    features = find_domain_features(
        model, tokenizer, sae, med_pairs, nonmed_pairs,
        TRUST_LANGS, STEER_LAYER, n_samples=N_CONTRAST_SAMPLES
    )
    if not features:
        print("\nNo language-agnostic medical features found.")
        print("Relaxing criteria might help, but the absence itself is a finding.")
        return

    # Phase C: Characterize on victim languages
    print("\n" + "=" * 70)
    print("PHASE C: Feature firing on victim languages")
    print("=" * 70)
    for vlang in VICTIM_LANGS:
        features = characterize_on_victim(
            model, tokenizer, sae, med_pairs, nonmed_pairs,
            vlang, features, STEER_LAYER, n_samples=N_CONTRAST_SAMPLES
        )

    top_feature = features[0]
    print(f"\n  Selected top feature: {top_feature['feature_idx']}")
    print(f"  Average medical activation (trust): "
          f"{np.mean(list(top_feature['med_means'].values())):.1f}")

    # Phase D: Rescue experiments
    print("\n" + "=" * 70)
    print("PHASE D: Rescue attempts")
    print("=" * 70)

    all_results = {"features": features, "baselines": baselines, "rescue": {}}

    for vlang in VICTIM_LANGS:
        print(f"\n--- Rescuing {vlang} medical ---")
        vlang_baseline = baselines[f"{vlang}_medical"]["accuracy"]
        # Steering strength based on top feature's trust-language mean
        base_mean = np.mean(list(top_feature["med_means"].values()))

        # Test: single top feature
        print(f"\n  Single feature {top_feature['feature_idx']} on {vlang}:")
        single_results = {}
        for s in STEER_MULTIPLIERS:
            delta = s * base_mean
            feature_deltas = {top_feature["feature_idx"]: delta} if s > 0 else {}
            intervention = lambda fd=feature_deltas: steer_features(model, sae, STEER_LAYER, fd)
            res = evaluate_all(model, tokenizer, med_pairs, vlang, answer_tids,
                                f"{vlang}-single-{s}x", intervention=intervention)
            acc, ci = bootstrap_ci(res)
            delta_acc = acc - vlang_baseline
            print(f"    {s}x: {acc:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]  Δ={delta_acc:+.2%}")
            single_results[s] = {"accuracy": acc, "ci_95": list(ci),
                                  "delta": delta_acc,
                                  "predictions": [{"correct": r["correct"], "subject": r["subject"]} for r in res]}

        # Top-3 features
        print(f"\n  Top-3 features {[f['feature_idx'] for f in features[:3]]} on {vlang}:")
        top3_results = {}
        for s in STEER_MULTIPLIERS:
            feature_deltas = {f["feature_idx"]: s * np.mean(list(f["med_means"].values()))
                              for f in features[:3]} if s > 0 else {}
            intervention = lambda fd=feature_deltas: steer_features(model, sae, STEER_LAYER, fd)
            res = evaluate_all(model, tokenizer, med_pairs, vlang, answer_tids,
                                f"{vlang}-top3-{s}x", intervention=intervention)
            acc, ci = bootstrap_ci(res)
            delta_acc = acc - vlang_baseline
            print(f"    {s}x: {acc:.1%}  Δ={delta_acc:+.2%}")
            top3_results[s] = {"accuracy": acc, "ci_95": list(ci),
                                "delta": delta_acc,
                                "predictions": [{"correct": r["correct"], "subject": r["subject"]} for r in res]}

        # Random feature control (matched magnitude)
        print(f"\n  Random feature control on {vlang}:")
        exclude = [f["feature_idx"] for f in features]
        # Pick a feature that fires on EN-medical at similar magnitude
        random_feat = pick_random_control(
            model, tokenizer, sae, med_pairs, "en", STEER_LAYER,
            target_mean=base_mean, exclude=exclude
        )
        print(f"    Random feature chosen: {random_feat['feature_idx']}")

        random_results = {}
        for s in STEER_MULTIPLIERS:
            delta = s * base_mean
            feature_deltas = {random_feat["feature_idx"]: delta} if s > 0 else {}
            intervention = lambda fd=feature_deltas: steer_features(model, sae, STEER_LAYER, fd)
            res = evaluate_all(model, tokenizer, med_pairs, vlang, answer_tids,
                                f"{vlang}-random-{s}x", intervention=intervention)
            acc, ci = bootstrap_ci(res)
            delta_acc = acc - vlang_baseline
            print(f"    {s}x: {acc:.1%}  Δ={delta_acc:+.2%}")
            random_results[s] = {"accuracy": acc, "ci_95": list(ci),
                                  "delta": delta_acc,
                                  "predictions": [{"correct": r["correct"], "subject": r["subject"]} for r in res]}

        all_results["rescue"][vlang] = {
            "baseline": vlang_baseline,
            "single_feature": top_feature["feature_idx"],
            "single": single_results,
            "top3_features": [f["feature_idx"] for f in features[:3]],
            "top3": top3_results,
            "random_feature": random_feat["feature_idx"],
            "random": random_results,
        }

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for vlang in VICTIM_LANGS:
        r = all_results["rescue"][vlang]
        print(f"\n{vlang}: baseline {r['baseline']:.1%}")
        print(f"  Strength  Single  Top-3   Random")
        for s in STEER_MULTIPLIERS:
            sgl = r["single"][s]["delta"]
            t3 = r["top3"][s]["delta"]
            rnd = r["random"][s]["delta"]
            print(f"  {s:>6.1f}x  {sgl:+.2%}  {t3:+.2%}  {rnd:+.2%}")

    # Save
    with open("/root/results/v3_domain_rescue.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to /root/results/v3_domain_rescue.json")

    del model, sae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
