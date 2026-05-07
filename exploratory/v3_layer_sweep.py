"""
v3 Layer Sweep: Test whether the ablation null at layer 29 also holds
at earlier layers, or whether there's a causal pathway layer we missed.

MOTIVATION: v3 domain rescue picked layer 29 by inertia from v1/v2, where
layer 22 (~85% depth on 1B) showed the strongest language-specific features.
But knowledge retrieval may happen much earlier. v1 showed middle layers
(13, 17 on 1B) are the most language-agnostic, which is consistent with
conceptual processing living there. If that's right, ablating medical
features at middle layers should hurt medical accuracy even when layer-29
ablation doesn't.

EXPERIMENT:
  For each layer L in {9, 17, 22, 29} (all Gemma Scope 2 'res' SAE layers
  for Gemma 3 4B):
    1. Identify language-agnostic medical features using 6-condition contrastive
       at that layer
    2. Compute full-benchmark baseline accuracy: EN/ES/FR × med/non-med (once)
    3. Ablate top-3 medical features at layer L during inference
    4. Report ablation delta per condition

Interpretation:
  - If layer L shows medical accuracy drop under ablation AND non-medical
    stable → causal pathway for medical knowledge at layer L
  - If all conditions drop equally → general representation disruption, not
    medical-specific
  - If no effect at any layer → features are readouts at every layer;
    the broader null is confirmed

All claims scoped: single-feature ablation at individual layers.
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
SAE_RELEASE = "gemma-scope-2-4b-it-res"

# All layers with 16k / L0-medium SAEs in the 'res' release
SWEEP_LAYERS = [9, 17, 22, 29]

MEDICAL_SUBJECTS = ["anatomy", "clinical_knowledge", "college_medicine",
                    "medical_genetics", "professional_medicine"]
NONMED_SUBJECTS = ["philosophy", "world_religions", "global_facts"]

TRUST_LANGS = {"es": "ES_LA", "fr": "FR_FR"}

N_CONTRAST_SAMPLES = 40
N_FEATURES_TO_ABLATE = 3
SEED = 42
# ============================================================


def get_layer(model, idx):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[idx]
    return model.model.layers[idx]


def load_paired(cais_ds, config_datasets, subjects):
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
        if (i + 1) % 400 == 0:
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


def find_domain_features_at_layer(model, tokenizer, sae, med_pairs, nonmed_pairs,
                                    layer, n_samples, top_k):
    """Same 6-condition contrastive as v3_domain_rescue, parameterized by layer."""
    print(f"  Identifying medical features at layer {layer}...")
    med_acts = {}
    nonmed_acts = {}
    for lang in ["en"] + list(TRUST_LANGS):
        med_acts[lang] = collect_mean_acts(model, tokenizer, sae, med_pairs, lang, layer, n_samples)
        nonmed_acts[lang] = collect_mean_acts(model, tokenizer, sae, nonmed_pairs, lang, layer, n_samples)

    med_means = {l: a.mean(dim=0) for l, a in med_acts.items()}
    nonmed_means = {l: a.mean(dim=0) for l, a in nonmed_acts.items()}
    med_fires = {l: (a > 0).float().mean(dim=0) for l, a in med_acts.items()}
    nonmed_fires_max = torch.stack([(a > 0).float().mean(dim=0) for a in nonmed_acts.values()]).max(dim=0).values

    langs = ["en"] + list(TRUST_LANGS)
    med_mean_stack = torch.stack([med_means[l] for l in langs])
    nonmed_mean_stack = torch.stack([nonmed_means[l] for l in langs])
    med_fires_stack = torch.stack([med_fires[l] for l in langs])

    eps = 0.1
    min_med_mean = med_mean_stack.min(dim=0).values
    max_nonmed_mean = nonmed_mean_stack.max(dim=0).values
    specificity = (min_med_mean + eps) / (max_nonmed_mean + eps)
    min_med_fires = med_fires_stack.min(dim=0).values

    score = specificity * min_med_fires
    mask = (min_med_fires > 0.5).float()
    mask = mask * (min_med_mean > 2.0).float()
    mask = mask * (nonmed_fires_max < 0.5).float()
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
            "en_med": float(med_means["en"][idx].item()),
            "es_med": float(med_means["es"][idx].item()),
            "fr_med": float(med_means["fr"][idx].item()),
            "en_nonmed": float(nonmed_means["en"][idx].item()),
            "es_nonmed": float(nonmed_means["es"][idx].item()),
            "fr_nonmed": float(nonmed_means["fr"][idx].item()),
            "min_med_fires": float(min_med_fires[idx].item()),
        })
    print(f"    Found {len(candidates)} features; top: "
          f"{[(c['feature_idx'], round(c['en_med'],1)) for c in candidates[:3]]}")
    return candidates


@contextmanager
def ablate_features(model, sae, layer, feature_idxs):
    """Zero out feature_idxs by subtracting their SAE contribution from residual."""
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        resid = hidden.clone()
        resid_for_sae = resid.to(sae.dtype)
        features = sae.encode(resid_for_sae)
        features_ablated = features.clone()
        for fidx in feature_idxs:
            features_ablated[:, :, fidx] = 0.0
        orig_recon = sae.decode(features)
        ablated_recon = sae.decode(features_ablated)
        delta = (ablated_recon - orig_recon).to(hidden.dtype)
        result = hidden + delta
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
    print("v3 LAYER SWEEP: ablation across layers 9, 17, 22, 29")
    print("=" * 70)

    print("\n--- Loading model ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16)
    answer_tids = [tokenizer.encode(f" {l}", add_special_tokens=False)[-1] for l in ["A", "B", "C", "D"]]

    print("\n--- Loading datasets ---")
    cais_mmlu = load_dataset("cais/mmlu", "all", split="test")
    trust_configs = {lang: load_dataset("openai/MMMLU", cfg, split="test")
                     for lang, cfg in TRUST_LANGS.items()}
    med_pairs = load_paired(cais_mmlu, trust_configs, MEDICAL_SUBJECTS)
    nonmed_pairs = load_paired(cais_mmlu, trust_configs, NONMED_SUBJECTS)
    print(f"  Medical pairs: {len(med_pairs)}")
    print(f"  Non-medical pairs: {len(nonmed_pairs)}")

    # ============================================================
    # Step 1: Baselines (computed once, reused across layers)
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 1: Baselines (no intervention)")
    print("=" * 70)
    baselines = {}
    for lang in ["en"] + list(TRUST_LANGS):
        print(f"\n  {lang} medical...")
        r_med = evaluate(model, tokenizer, med_pairs, lang, answer_tids, f"{lang}-med")
        print(f"  {lang} non-medical...")
        r_nonmed = evaluate(model, tokenizer, nonmed_pairs, lang, answer_tids, f"{lang}-nonmed")
        acc_med, ci_med = bootstrap_ci(r_med)
        acc_nonmed, ci_nonmed = bootstrap_ci(r_nonmed)
        baselines[lang] = {
            "medical": {"accuracy": acc_med, "ci_95": list(ci_med),
                        "predictions": [{"correct": r["correct"], "subject": r["subject"]} for r in r_med]},
            "nonmedical": {"accuracy": acc_nonmed, "ci_95": list(ci_nonmed),
                            "predictions": [{"correct": r["correct"], "subject": r["subject"]} for r in r_nonmed]},
        }
        print(f"    {lang} med: {acc_med:.1%} [{ci_med[0]:.1%}, {ci_med[1]:.1%}]")
        print(f"    {lang} nonmed: {acc_nonmed:.1%} [{ci_nonmed[0]:.1%}, {ci_nonmed[1]:.1%}]")

    # ============================================================
    # Step 2: For each layer, identify features + run ablation
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2: Layer-by-layer feature identification + ablation")
    print("=" * 70)

    layer_results = {}
    for layer in SWEEP_LAYERS:
        print(f"\n{'=' * 50}")
        print(f"LAYER {layer}")
        print(f"{'=' * 50}")

        print(f"\n  Loading SAE for layer {layer}...")
        sae_id = f"layer_{layer}_width_16k_l0_medium"
        sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device="cuda")

        # Identify features
        features = find_domain_features_at_layer(
            model, tokenizer, sae, med_pairs, nonmed_pairs,
            layer, N_CONTRAST_SAMPLES, top_k=N_FEATURES_TO_ABLATE
        )

        if not features:
            print(f"  No clean medical features at layer {layer}. Skipping ablation.")
            layer_results[layer] = {"features": [], "ablation": None}
            del sae
            torch.cuda.empty_cache()
            continue

        feature_idxs = [f["feature_idx"] for f in features]
        print(f"\n  Ablating features {feature_idxs} at layer {layer}...")

        intervention = lambda: ablate_features(model, sae, layer, feature_idxs)

        abl_results = {}
        for lang in ["en"] + list(TRUST_LANGS):
            print(f"\n    Evaluating {lang} with ablation...")
            r_med = evaluate(model, tokenizer, med_pairs, lang, answer_tids,
                              f"L{layer}-{lang}-med-abl", intervention=intervention)
            r_nonmed = evaluate(model, tokenizer, nonmed_pairs, lang, answer_tids,
                                 f"L{layer}-{lang}-nonmed-abl", intervention=intervention)
            a_med, ci_med = bootstrap_ci(r_med)
            a_nonmed, ci_nonmed = bootstrap_ci(r_nonmed)
            b_med = baselines[lang]["medical"]["accuracy"]
            b_nonmed = baselines[lang]["nonmedical"]["accuracy"]
            print(f"      {lang} med:    {b_med:.1%} → {a_med:.1%}  Δ={a_med-b_med:+.2%}")
            print(f"      {lang} nonmed: {b_nonmed:.1%} → {a_nonmed:.1%}  Δ={a_nonmed-b_nonmed:+.2%}")
            abl_results[lang] = {
                "medical": {"accuracy": a_med, "ci_95": list(ci_med),
                             "delta": a_med - b_med,
                             "predictions": [{"correct": r["correct"], "subject": r["subject"]} for r in r_med]},
                "nonmedical": {"accuracy": a_nonmed, "ci_95": list(ci_nonmed),
                                "delta": a_nonmed - b_nonmed,
                                "predictions": [{"correct": r["correct"], "subject": r["subject"]} for r in r_nonmed]},
            }

        layer_results[layer] = {
            "features": features,
            "ablation": abl_results,
        }

        del sae
        torch.cuda.empty_cache()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Ablation deltas by layer and condition")
    print("=" * 70)
    print(f"\n{'Layer':>6s} | {'EN med':>10s} | {'EN non-med':>12s} | "
          f"{'ES med':>10s} | {'ES non-med':>12s} | {'FR med':>10s} | {'FR non-med':>12s}")
    print("-" * 90)
    for layer in SWEEP_LAYERS:
        r = layer_results[layer]
        if r["ablation"] is None:
            print(f"{layer:>6d} | (no features)")
            continue
        abl = r["ablation"]
        row = f"{layer:>6d} |"
        for lang in ["en", "es", "fr"]:
            dm = abl[lang]["medical"]["delta"]
            dn = abl[lang]["nonmedical"]["delta"]
            row += f" {dm:+.2%} |" + f" {dn:+.2%} |"
        print(row)

    # Identify layers with meaningful effect
    print("\n--- Interpretation ---")
    interesting_layers = []
    for layer in SWEEP_LAYERS:
        r = layer_results[layer]
        if r["ablation"] is None:
            continue
        abl = r["ablation"]
        med_deltas = [abl[l]["medical"]["delta"] for l in ["en", "es", "fr"]]
        nonmed_deltas = [abl[l]["nonmedical"]["delta"] for l in ["en", "es", "fr"]]
        avg_med = np.mean(med_deltas)
        avg_nonmed = np.mean(nonmed_deltas)
        specificity = avg_nonmed - avg_med  # positive if medical drops more
        if abs(avg_med) > 0.01 or abs(avg_nonmed) > 0.01:
            interesting_layers.append((layer, avg_med, avg_nonmed, specificity))
            print(f"  Layer {layer}: avg med Δ={avg_med:+.2%}, avg nonmed Δ={avg_nonmed:+.2%}, "
                  f"specificity={specificity:+.2%}")

    if not interesting_layers:
        print("  No layer shows ablation effect above 1%. Features are")
        print("  readouts at every tested layer; the broader null is confirmed.")
    else:
        print(f"\n  Layers showing measurable ablation effect: {[l[0] for l in interesting_layers]}")
        print("  Follow-up candidates for amplification/deeper intervention.")

    # Save
    output = {
        "model": MODEL_ID,
        "sae_release": SAE_RELEASE,
        "layers_tested": SWEEP_LAYERS,
        "n_med_pairs": len(med_pairs),
        "n_nonmed_pairs": len(nonmed_pairs),
        "baselines": baselines,
        "by_layer": {str(l): layer_results[l] for l in SWEEP_LAYERS},
    }
    with open("/root/results/v3_layer_sweep.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to /root/results/v3_layer_sweep.json")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
