"""
v3 Feature Validation: Stress-test the language-agnostic medical features.

In v3_domain_rescue we identified features that appear to be language-agnostic
medical features:
  - 12570: fires on EN/ES/FR medical (606/366/330), zero on non-medical
  - 893:   fires on EN/ES/FR medical (834/542/536), zero on non-medical
  - 12845: fires on EN/ES/FR medical (235/149/129), zero on non-medical

Before calling these "medical knowledge features", we need to rule out that
they're just MCQ-format or medical-register detectors. Three tests:

TEST 1 (Top-activating content):
  Run varied texts through the model. Which specific tokens/contexts trigger
  each feature? If top activations are clearly medical content words
  ("patient", "lesion", "diagnosis"), medical. If top activations are MCQ
  markers ("Question:", "A."), format detector. If top activations are
  generic question words, not medical.

TEST 2 (Ablation):
  Zero out each feature during inference. Measure accuracy drop on:
    - EN/ES/FR medical
    - EN/ES/FR non-medical
  If medical drops more than non-medical, the feature contributes to medical
  performance (driver). If equal drop, shared representation. If no drop, the
  feature is a readout only.

TEST 3 (Non-MCQ medical text):
  Free-form medical text (not MCQ-formatted). Do the features fire?
  If yes, domain-general. If no, MCQ-medical-specific.

Outcome interpretation:
  - All three pass → language-agnostic medical knowledge features (real finding)
  - Only test 1 & 3 pass, 2 fails → medical readout features (weaker but real)
  - Test 1 fails → MCQ or register features mislabeled; retract the claim
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
MODEL_ID = "google/gemma-3-4b-it"
STEER_LAYER = 29
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = f"layer_{STEER_LAYER}_width_16k_l0_medium"

TARGET_FEATURES = [12570, 893, 12845]

MEDICAL_SUBJECTS = ["anatomy", "clinical_knowledge", "college_medicine",
                    "medical_genetics", "professional_medicine"]
NONMED_SUBJECTS = ["philosophy", "world_religions", "global_facts"]

TRUST_LANGS = {"es": "ES_LA", "fr": "FR_FR"}

SEED = 42

# Non-MCQ medical texts (free-form, varied length, varied subtopic)
MEDICAL_FREE_TEXTS = [
    # Clinical vignette
    "A 58-year-old man presents with chest pain radiating to the left arm, "
    "diaphoresis, and shortness of breath. ECG shows ST-elevation in leads "
    "V1-V4. The most likely diagnosis is anterior myocardial infarction "
    "requiring immediate revascularization.",
    # Textbook prose
    "The renin-angiotensin-aldosterone system regulates blood pressure and "
    "fluid balance. Renin released from the juxtaglomerular cells cleaves "
    "angiotensinogen to angiotensin I, which is then converted to "
    "angiotensin II by angiotensin-converting enzyme in the lungs.",
    # Pharmacology
    "Beta-blockers reduce cardiac output and renin secretion, making them "
    "useful in treating hypertension, angina, and certain arrhythmias. "
    "Common agents include metoprolol, atenolol, and propranolol. Non-"
    "selective beta-blockers can exacerbate asthma.",
    # Anatomy
    "The brachial plexus is formed by the ventral rami of spinal nerves C5 "
    "through T1 and supplies motor and sensory innervation to the upper limb. "
    "Injury to the upper trunk produces Erb's palsy.",
    # Microbiology
    "Staphylococcus aureus is a gram-positive coccus that appears in clusters. "
    "Methicillin-resistant strains (MRSA) are resistant to most beta-lactam "
    "antibiotics and are treated with vancomycin or linezolid.",
]

NONMED_FREE_TEXTS = [
    # Philosophy
    "Kant's categorical imperative asserts that one should act only according "
    "to that maxim by which one can at the same time will that it should "
    "become a universal law. This principle of universalizability underlies "
    "his deontological ethics.",
    # History
    "The Treaty of Westphalia in 1648 ended the Thirty Years' War and "
    "established the modern system of sovereign states. It introduced the "
    "principle that each state has exclusive authority over its territory.",
    # General knowledge
    "The Amazon rainforest covers approximately 5.5 million square kilometers "
    "across nine countries. It produces about 20 percent of the oxygen in "
    "the Earth's atmosphere and harbors an estimated 10 percent of all "
    "known species.",
    # Economics
    "Inflation is sustained increase in the general price level of goods and "
    "services. Central banks typically target a specific inflation rate by "
    "adjusting monetary policy instruments such as interest rates.",
    # Literature
    "Shakespeare's Hamlet explores themes of revenge, madness, and moral "
    "ambiguity. The protagonist's famous soliloquy, 'To be or not to be', "
    "meditates on existence and the fear of death.",
]
# ============================================================


def get_layer(model, idx):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[idx]
    return model.model.layers[idx]


def format_mcq(q, options):
    text = f"Question: {q}\n"
    for key in ["A", "B", "C", "D"]:
        text += f"{key}. {options[key]}\n"
    text += "Answer:"
    return text


def get_token_activations(model, tokenizer, sae, text, layer, feature_idxs):
    """Get per-token activations for specific features on a text."""
    captured = []
    def hook_fn(module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        captured.append(o.detach())
    h = get_layer(model, layer).register_forward_hook(hook_fn)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**inputs)
    h.remove()

    resid = captured[0].to(sae.dtype)
    features = sae.encode(resid)  # [1, n_tokens, d_sae]
    feat_per_token = features[0].float().cpu()  # [n_tokens, d_sae]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

    # For each target feature, get activation per token
    result = {}
    for fidx in feature_idxs:
        acts = feat_per_token[:, fidx].tolist()
        result[fidx] = list(zip(tokens, acts))

    del captured, features, resid
    torch.cuda.empty_cache()
    return result


@contextmanager
def ablate_features(model, sae, layer, feature_idxs):
    """Zero out specific features by subtracting their contribution from residual."""
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        resid = hidden.clone()
        resid_for_sae = resid.to(sae.dtype)
        features = sae.encode(resid_for_sae)
        # Build steered features with target features zeroed
        features_ablated = features.clone()
        for fidx in feature_idxs:
            features_ablated[:, :, fidx] = 0.0
        # Compute difference in reconstruction (what the feature was contributing)
        orig_recon = sae.decode(features)
        ablated_recon = sae.decode(features_ablated)
        delta = (ablated_recon - orig_recon).to(hidden.dtype)  # subtract feature contribution
        result = hidden + delta
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result
    hook = get_layer(model, layer).register_forward_hook(hook_fn)
    try:
        yield
    finally:
        hook.remove()


def load_paired_mmlu(cais_ds, config_datasets, subjects):
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
            all_match = True
            for lang in config_datasets:
                if cfg_by_subj[lang][subj][i]["Answer"] != en_ans:
                    all_match = False
                    break
            if all_match:
                pair = {
                    "subject": subj,
                    "answer": en_ans,
                    "en_question": en_row["question"],
                    "en_options": {letter_map[j]: en_row["choices"][j] for j in range(4)},
                }
                for lang in config_datasets:
                    r = cfg_by_subj[lang][subj][i]
                    pair[f"{lang}_question"] = r["Question"]
                    pair[f"{lang}_options"] = {k: r[k] for k in ["A", "B", "C", "D"]}
                pairs.append(pair)
    return pairs


def get_answer_probs(model, tokenizer, prompt, answer_tids):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    last_logits = outputs.logits[0, -1, :]
    letter_logits = torch.tensor([last_logits[tid].item() for tid in answer_tids])
    probs = torch.softmax(letter_logits, dim=0)
    return {l: p.item() for l, p in zip(["A", "B", "C", "D"], probs)}


def evaluate(model, tokenizer, pairs, lang_key, answer_tids, label="", intervention=None):
    results = []
    q_key = f"{lang_key}_question"
    o_key = f"{lang_key}_options"
    for i, pair in enumerate(pairs):
        prompt = format_mcq(pair[q_key], pair[o_key])
        if intervention is not None:
            with intervention():
                probs = get_answer_probs(model, tokenizer, prompt, answer_tids)
        else:
            probs = get_answer_probs(model, tokenizer, prompt, answer_tids)
        predicted = max(probs, key=probs.get)
        results.append({"subject": pair["subject"], "correct": predicted == pair["answer"]})
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
    return acc, (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))


def main():
    print("=" * 70)
    print("v3 FEATURE VALIDATION: Testing the language-agnostic medical features")
    print("=" * 70)

    print("\n--- Loading model + SAE ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16)
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    answer_tids = [tokenizer.encode(f" {l}", add_special_tokens=False)[-1] for l in ["A", "B", "C", "D"]]
    print(f"  Model + SAE loaded. Target features: {TARGET_FEATURES}")

    results = {"features": TARGET_FEATURES}

    # ============================================================
    # TEST 1: Top-activating tokens
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 1: Top-activating tokens on free-form text")
    print("=" * 70)

    all_texts = [("MED", t) for t in MEDICAL_FREE_TEXTS] + [("NONMED", t) for t in NONMED_FREE_TEXTS]

    test1_results = {fidx: {"all_activations": [], "top_tokens": []} for fidx in TARGET_FEATURES}
    for label, text in all_texts:
        print(f"\n  Text [{label}]: {text[:80]}...")
        acts = get_token_activations(model, tokenizer, sae, text, STEER_LAYER, TARGET_FEATURES)
        for fidx, token_acts in acts.items():
            # Sort by activation
            sorted_acts = sorted(token_acts, key=lambda x: -x[1])
            top_5 = sorted_acts[:5]
            max_act = top_5[0][1] if top_5 else 0
            test1_results[fidx]["all_activations"].append({
                "text_label": label,
                "text_preview": text[:100],
                "max_activation": max_act,
                "top_5_tokens": [(t, float(a)) for t, a in top_5],
            })
            print(f"    Feature {fidx}: max_act={max_act:.1f}, top tokens: "
                  f"{[(t, round(a,1)) for t,a in top_5[:3]]}")

    # Summary: average max activation on MED vs NONMED
    print("\n  SUMMARY (test 1):")
    for fidx in TARGET_FEATURES:
        med_maxes = [x["max_activation"] for x in test1_results[fidx]["all_activations"] if x["text_label"] == "MED"]
        nonmed_maxes = [x["max_activation"] for x in test1_results[fidx]["all_activations"] if x["text_label"] == "NONMED"]
        print(f"    Feature {fidx}: MED free-text max act avg={np.mean(med_maxes):.1f}, "
              f"NONMED max act avg={np.mean(nonmed_maxes):.1f}, ratio={np.mean(med_maxes)/max(np.mean(nonmed_maxes),0.1):.1f}x")

    results["test1_top_activations"] = test1_results

    # ============================================================
    # Load MMLU for tests 2
    # ============================================================
    print("\n--- Loading MMLU data ---")
    cais_mmlu = load_dataset("cais/mmlu", "all", split="test")
    trust_configs = {lang: load_dataset("openai/MMMLU", cfg, split="test")
                     for lang, cfg in TRUST_LANGS.items()}

    med_pairs = load_paired_mmlu(cais_mmlu, trust_configs, MEDICAL_SUBJECTS)
    nonmed_pairs = load_paired_mmlu(cais_mmlu, trust_configs, NONMED_SUBJECTS)
    print(f"  Medical pairs: {len(med_pairs)}")
    print(f"  Non-medical pairs: {len(nonmed_pairs)}")

    # ============================================================
    # TEST 2: Ablation (does removing features hurt medical more than non-medical?)
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 2: Ablation - zero out target features")
    print("=" * 70)

    intervention = lambda: ablate_features(model, sae, STEER_LAYER, TARGET_FEATURES)

    test2_results = {}
    for lang in ["en"] + list(TRUST_LANGS):
        print(f"\n  Evaluating {lang}...")

        # Medical baseline + ablated
        print(f"    {lang} medical baseline...")
        baseline_med = evaluate(model, tokenizer, med_pairs, lang, answer_tids, f"{lang}-med-base")
        print(f"    {lang} medical with ablation...")
        ablated_med = evaluate(model, tokenizer, med_pairs, lang, answer_tids, f"{lang}-med-ablate",
                                intervention=intervention)

        # Non-medical baseline + ablated
        print(f"    {lang} non-medical baseline...")
        baseline_nonmed = evaluate(model, tokenizer, nonmed_pairs, lang, answer_tids, f"{lang}-nonmed-base")
        print(f"    {lang} non-medical with ablation...")
        ablated_nonmed = evaluate(model, tokenizer, nonmed_pairs, lang, answer_tids, f"{lang}-nonmed-ablate",
                                   intervention=intervention)

        b_med_acc, b_med_ci = bootstrap_ci(baseline_med)
        a_med_acc, a_med_ci = bootstrap_ci(ablated_med)
        b_nonmed_acc, b_nonmed_ci = bootstrap_ci(baseline_nonmed)
        a_nonmed_acc, a_nonmed_ci = bootstrap_ci(ablated_nonmed)

        print(f"    {lang} MED:    {b_med_acc:.1%} → {a_med_acc:.1%}  Δ={a_med_acc-b_med_acc:+.2%}")
        print(f"    {lang} NONMED: {b_nonmed_acc:.1%} → {a_nonmed_acc:.1%}  Δ={a_nonmed_acc-b_nonmed_acc:+.2%}")

        test2_results[lang] = {
            "medical": {
                "baseline": {"accuracy": b_med_acc, "ci_95": list(b_med_ci)},
                "ablated": {"accuracy": a_med_acc, "ci_95": list(a_med_ci)},
                "delta": a_med_acc - b_med_acc,
            },
            "nonmedical": {
                "baseline": {"accuracy": b_nonmed_acc, "ci_95": list(b_nonmed_ci)},
                "ablated": {"accuracy": a_nonmed_acc, "ci_95": list(a_nonmed_ci)},
                "delta": a_nonmed_acc - b_nonmed_acc,
            },
        }

    results["test2_ablation"] = test2_results

    # ============================================================
    # TEST 3: Non-MCQ medical text activations
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 3: Free-form medical text activations")
    print("=" * 70)
    print("(Already collected in Test 1 -- summary here)")

    test3_summary = {}
    for fidx in TARGET_FEATURES:
        med_maxes = [x["max_activation"] for x in test1_results[fidx]["all_activations"] if x["text_label"] == "MED"]
        nonmed_maxes = [x["max_activation"] for x in test1_results[fidx]["all_activations"] if x["text_label"] == "NONMED"]
        med_avg = float(np.mean(med_maxes))
        nonmed_avg = float(np.mean(nonmed_maxes))
        ratio = med_avg / max(nonmed_avg, 0.1)
        fires_on_med = sum(1 for m in med_maxes if m > 5) / len(med_maxes)
        test3_summary[fidx] = {
            "med_free_text_avg_max_act": med_avg,
            "nonmed_free_text_avg_max_act": nonmed_avg,
            "med_vs_nonmed_ratio": ratio,
            "fires_on_med_free_text_frac": fires_on_med,
        }
        interpretation = "MEDICAL" if ratio > 3 and fires_on_med > 0.5 else (
            "AMBIGUOUS" if ratio > 1.5 else "NOT MEDICAL"
        )
        print(f"  Feature {fidx}: med avg max={med_avg:.1f}, nonmed avg max={nonmed_avg:.1f}, "
              f"ratio={ratio:.1f}x, fires on MED free text={fires_on_med:.0%} → {interpretation}")

    results["test3_free_text"] = test3_summary

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    # Test 1: did medical have higher activations than non-medical in top tokens?
    med_higher_count = 0
    for fidx in TARGET_FEATURES:
        med_maxes = [x["max_activation"] for x in test1_results[fidx]["all_activations"] if x["text_label"] == "MED"]
        nonmed_maxes = [x["max_activation"] for x in test1_results[fidx]["all_activations"] if x["text_label"] == "NONMED"]
        if np.mean(med_maxes) > 3 * np.mean(nonmed_maxes):
            med_higher_count += 1
    test1_pass = med_higher_count >= 2
    print(f"\nTest 1 (free-text medical activation): {'PASS' if test1_pass else 'FAIL'} "
          f"({med_higher_count}/3 features activate preferentially on medical text)")

    # Test 2: did medical drop more than non-medical under ablation (averaged across langs)?
    avg_med_drop = np.mean([test2_results[l]["medical"]["delta"] for l in test2_results])
    avg_nonmed_drop = np.mean([test2_results[l]["nonmedical"]["delta"] for l in test2_results])
    test2_pass = avg_med_drop < avg_nonmed_drop - 0.005  # medical accuracy drops more (delta more negative)
    print(f"\nTest 2 (ablation hurts medical more): {'PASS' if test2_pass else 'FAIL'}")
    print(f"  Avg medical Δ under ablation: {avg_med_drop:+.2%}")
    print(f"  Avg non-medical Δ under ablation: {avg_nonmed_drop:+.2%}")

    # Test 3: do they fire on non-MCQ medical text?
    avg_fires = np.mean([test3_summary[f]["fires_on_med_free_text_frac"] for f in TARGET_FEATURES])
    test3_pass = avg_fires > 0.6
    print(f"\nTest 3 (fires on non-MCQ medical text): {'PASS' if test3_pass else 'FAIL'} "
          f"(avg fire rate on medical free-text: {avg_fires:.0%})")

    # Final interpretation
    print("\n" + "-" * 70)
    if test1_pass and test2_pass and test3_pass:
        print("INTERPRETATION: Medical knowledge features (all three tests pass).")
        print("These are domain-specific features that contribute causally to medical MCQ.")
    elif test1_pass and test3_pass and not test2_pass:
        print("INTERPRETATION: Medical readout features (identified cleanly, but ablation")
        print("doesn't hurt medical more than non-medical). They represent medical content")
        print("but don't drive task performance.")
    elif not test1_pass:
        print("INTERPRETATION: Features are NOT medical-specific in free-form text. ")
        print("Previous identification was confounded by MCQ format or training artifacts.")
        print("Retract the 'language-agnostic medical features' claim.")
    else:
        print("INTERPRETATION: Mixed evidence. Inspect individual test results carefully.")

    results["verdict"] = {
        "test1_pass": test1_pass,
        "test2_pass": test2_pass,
        "test3_pass": test3_pass,
    }

    # Save
    with open("results/v3_feature_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/v3_feature_validation.json")

    del model, sae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
