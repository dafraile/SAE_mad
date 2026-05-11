"""
v3 Replication: Rigorous re-run addressing 5 criticisms of v2 results.

CRITICISMS BEING ADDRESSED:
1. "Single feature" claim was actually multi-feature (top-3 / top-10). FIX: test single feature.
2. Net rescue measured on subset, not full benchmark. FIX: evaluate ALL paired items.
3. MMMLU "default" config was NOT English (mostly Arabic). FIX: use cais/mmlu for real English.
4. English-wrapper protocol. FIX: caveat in analysis (we keep the protocol for comparability).
5. Random-feature control missing. FIX: steer a random non-language feature as control.

EXPERIMENT STRUCTURE:

Phase A: Re-establish baselines with REAL English
  Use cais/mmlu (true English MMLU) paired by position with openai/MMMLU ES_LA / FR_FR.
  Verify pairing by matching answer letters. Report EN, ES, FR accuracy and gaps.

Phase B: Single-feature rescue, full-benchmark
  Identify Spanish feature (should be 596 again). Amplify ONLY feature 596 at multiple
  strengths during English inference. Evaluate ALL paired English questions (not just
  rescuable). Report mean accuracy delta.

Phase C: Random-feature control
  Amplify a random feature (low activation on both languages, similar magnitude to 596)
  at the same strengths. Ensures rescue is specific to the language feature.

Phase D: Top-k comparison
  Also run top-3 and top-10 feature steering for comparison with v2 results.

Output: JSON with per-item predictions, baseline accuracy, steered accuracy at each
strength for each condition. Mean accuracy delta with bootstrap CIs.
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

# Use only medical subjects for the replication (where we previously saw biggest gap)
MEDICAL_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]

# Control domain - non-medical
CONTROL_SUBJECTS = ["philosophy", "world_religions", "global_facts"]

# Strengths for rescue
STEER_MULTIPLIERS = [0, 1.0, 2.0, 3.0, 5.0]

# Seed for random feature selection
SEED = 42
# ============================================================


def get_layer(model, idx):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[idx]
    return model.model.layers[idx]


def build_paired_set(cais_ds, mmlu_es_ds, mmlu_fr_ds, subjects):
    """Pair questions across cais/mmlu (real EN) and openai/MMMLU (ES, FR).

    cais/mmlu column format: question, choices (list), answer (0-3), subject
    openai/MMMLU format: Question, A, B, C, D, Answer (letter A-D), Subject, Unnamed: 0

    Pairing strategy:
      - Filter by subject in each dataset
      - Sort by stable order (Unnamed: 0 for MMMLU, original order for cais/mmlu)
      - Pair by position within subject
      - Filter: answer letter must match (cais answer index maps to A/B/C/D)
      - Spot-check a few pairs manually for alignment
    """
    print("Building paired set with verification...")

    # Index cais/mmlu by subject
    en_by_subj = defaultdict(list)
    for row in cais_ds:
        if row["subject"] in subjects:
            en_by_subj[row["subject"]].append(row)

    # Index MMMLU configs by subject, sorted by Unnamed: 0
    def index_mmmlu(ds):
        by_subj = defaultdict(list)
        for row in ds:
            if row["Subject"] in subjects:
                by_subj[row["Subject"]].append(row)
        for s in by_subj:
            by_subj[s].sort(key=lambda r: r["Unnamed: 0"])
        return by_subj

    es_by_subj = index_mmmlu(mmlu_es_ds)
    fr_by_subj = index_mmmlu(mmlu_fr_ds)

    # Map cais answer index (0-3) to letter
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    pairs = []
    subj_stats = {}
    for subj in subjects:
        en_list = en_by_subj.get(subj, [])
        es_list = es_by_subj.get(subj, [])
        fr_list = fr_by_subj.get(subj, [])
        n = min(len(en_list), len(es_list), len(fr_list))

        matched = 0
        for i in range(n):
            en_answer_letter = letter_map[en_list[i]["answer"]]
            es_answer_letter = es_list[i]["Answer"]
            fr_answer_letter = fr_list[i]["Answer"]

            if en_answer_letter == es_answer_letter == fr_answer_letter:
                pairs.append({
                    "subject": subj,
                    "answer": en_answer_letter,
                    "en_question": en_list[i]["question"],
                    "en_options": {
                        letter_map[j]: en_list[i]["choices"][j]
                        for j in range(4)
                    },
                    "es_question": es_list[i]["Question"],
                    "es_options": {k: es_list[i][k] for k in ["A", "B", "C", "D"]},
                    "fr_question": fr_list[i]["Question"],
                    "fr_options": {k: fr_list[i][k] for k in ["A", "B", "C", "D"]},
                })
                matched += 1

        subj_stats[subj] = {"available": n, "matched": matched}
        print(f"  {subj}: {matched}/{n} matched")

    # Spot-check first 3 pairs
    print("\nSpot-checks (first 3 pairs):")
    for p in pairs[:3]:
        print(f"  Subject: {p['subject']}, Answer: {p['answer']}")
        print(f"    EN: {p['en_question'][:80]}")
        print(f"    ES: {p['es_question'][:80]}")
        print(f"    FR: {p['fr_question'][:80]}")

    return pairs, subj_stats


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
    """Evaluate ALL pairs in the given language.

    intervention: optional context manager factory. If provided, it's called for
    each forward pass (via `with intervention():`).
    """
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
            "confidence": probs[predicted],
        })
        if (i + 1) % 100 == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"    [{i+1}/{len(pairs)}] {label}: {acc:.1%}")
    return results


def bootstrap_ci(results, n_boot=1000, ci=0.95):
    """Bootstrap CI on mean accuracy."""
    correct = [int(r["correct"]) for r in results]
    if len(correct) == 0:
        return 0.0, (0.0, 0.0)
    n = len(correct)
    acc = np.mean(correct)
    rng = np.random.default_rng(SEED)
    resamples = rng.choice(correct, size=(n_boot, n), replace=True)
    accs = resamples.mean(axis=1)
    lo = np.percentile(accs, (1 - ci) / 2 * 100)
    hi = np.percentile(accs, (1 + ci) / 2 * 100)
    return acc, (lo, hi)


def identify_spanish_feature(model, tokenizer, sae, pairs, layer, n_samples=40):
    """Find the top Spanish feature on the paired set."""
    print(f"\n  Identifying top Spanish feature at layer {layer}...")
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

    en_acts = collect("en")
    es_acts = collect("es")

    en_mean = en_acts.mean(dim=0)
    es_mean = es_acts.mean(dim=0)
    es_fires = (es_acts > 0).float().mean(dim=0)
    en_fires = (en_acts > 0).float().mean(dim=0)

    eps = 0.01
    score = (es_mean + eps) / (en_mean + eps) * es_fires
    score = score * (es_fires > 0.9).float() * (es_mean > 100).float()

    top = score.topk(10)
    candidates = []
    for i in range(10):
        idx = top.indices[i].item()
        if top.values[i].item() == 0:
            continue
        candidates.append({
            "feature_idx": idx,
            "es_mean": es_mean[idx].item(),
            "en_mean": en_mean[idx].item(),
            "es_fires": es_fires[idx].item(),
            "en_fires": en_fires[idx].item(),
        })

    print(f"  Top 5 ES features:")
    for c in candidates[:5]:
        print(f"    idx={c['feature_idx']}: es_mean={c['es_mean']:.1f} en_mean={c['en_mean']:.1f} "
              f"es_fires={c['es_fires']:.2f} en_fires={c['en_fires']:.2f}")

    return candidates


def pick_random_control_feature(model, tokenizer, sae, pairs, layer, target_mean,
                                   exclude_top_k, n_samples=20):
    """Pick a random feature that fires but isn't language-specific, matching target magnitude.

    We want a feature that:
    - Fires on English text (not a Spanish feature)
    - Has similar mean activation to the language feature (for fair comparison)
    - Is NOT in the top Spanish features list
    """
    print(f"\n  Picking random control feature (target magnitude ~{target_mean:.0f})...")
    sample = pairs[:n_samples]

    feat_means_en = []
    for pair in sample:
        captured = []
        def hook_fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            captured.append(o.detach())
        h = get_layer(model, layer).register_forward_hook(hook_fn)
        prompt = format_mcq(pair["en_question"], pair["en_options"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            model(**inputs)
        h.remove()
        resid = captured[0].to(sae.dtype)
        features = sae.encode(resid)
        feat_means_en.append(features.float().mean(dim=1).squeeze(0).cpu())
        del captured, features, resid
        torch.cuda.empty_cache()

    en_mat = torch.stack(feat_means_en)
    en_mean = en_mat.mean(dim=0)
    en_fires = (en_mat > 0).float().mean(dim=0)

    # Find features that fire consistently on EN (>90%) and have activation close to target
    # but are NOT in our excluded list
    mask = (en_fires > 0.9) & (en_mean > target_mean * 0.3) & (en_mean < target_mean * 3)
    candidates = torch.where(mask)[0].tolist()
    candidates = [c for c in candidates if c not in exclude_top_k]

    if not candidates:
        print("  WARNING: No matching features. Fall back to first high-activation feature.")
        top = en_mean.topk(50).indices.tolist()
        candidates = [c for c in top if c not in exclude_top_k]

    rng = random.Random(SEED)
    chosen = rng.choice(candidates)
    print(f"  Chose random feature {chosen}: en_mean={en_mean[chosen].item():.1f}, "
          f"en_fires={en_fires[chosen].item():.2f}")
    return {
        "feature_idx": chosen,
        "es_mean": target_mean,  # We'll amplify by the same absolute amount
        "en_mean": en_mean[chosen].item(),
        "en_fires": en_fires[chosen].item(),
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
    print("v3 REPLICATION: Rigorous re-run with real English")
    print("=" * 70)

    # Load model
    print("\n--- Loading model + SAE ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16)
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    answer_tids = [tokenizer.encode(f" {l}", add_special_tokens=False)[-1] for l in ["A", "B", "C", "D"]]
    print(f"  Model: {MODEL_ID}, SAE layer {STEER_LAYER}")

    # Load datasets - REAL English from cais/mmlu
    print("\n--- Loading datasets (cais/mmlu for English) ---")
    cais_mmlu = load_dataset("cais/mmlu", "all", split="test")
    mmmlu_es = load_dataset("openai/MMMLU", "ES_LA", split="test")
    mmmlu_fr = load_dataset("openai/MMMLU", "FR_FR", split="test")
    print(f"  cais/mmlu test: {len(cais_mmlu)}")
    print(f"  MMMLU ES_LA: {len(mmmlu_es)}")
    print(f"  MMMLU FR_FR: {len(mmmlu_fr)}")

    # Build paired set (medical + control domains combined)
    all_subjects = MEDICAL_SUBJECTS + CONTROL_SUBJECTS
    pairs, subj_stats = build_paired_set(cais_mmlu, mmmlu_es, mmmlu_fr, all_subjects)
    print(f"\nTotal paired: {len(pairs)}")

    # Separate by domain for analysis
    med_pairs = [p for p in pairs if p["subject"] in MEDICAL_SUBJECTS]
    ctrl_pairs = [p for p in pairs if p["subject"] in CONTROL_SUBJECTS]
    print(f"  Medical: {len(med_pairs)}")
    print(f"  Control: {len(ctrl_pairs)}")

    # ========================================================
    # PHASE A: Real baselines
    # ========================================================
    print("\n" + "=" * 70)
    print("PHASE A: Baselines with REAL English (cais/mmlu)")
    print("=" * 70)

    print("\nMedical baselines:")
    print("  EN (cais/mmlu)...")
    en_med = evaluate_all(model, tokenizer, med_pairs, "en", answer_tids, "en-med")
    print("  ES...")
    es_med = evaluate_all(model, tokenizer, med_pairs, "es", answer_tids, "es-med")
    print("  FR...")
    fr_med = evaluate_all(model, tokenizer, med_pairs, "fr", answer_tids, "fr-med")

    print("\nControl (non-medical) baselines:")
    print("  EN (cais/mmlu)...")
    en_ctrl = evaluate_all(model, tokenizer, ctrl_pairs, "en", answer_tids, "en-ctrl")
    print("  ES...")
    es_ctrl = evaluate_all(model, tokenizer, ctrl_pairs, "es", answer_tids, "es-ctrl")
    print("  FR...")
    fr_ctrl = evaluate_all(model, tokenizer, ctrl_pairs, "fr", answer_tids, "fr-ctrl")

    def acc(r):
        return sum(x["correct"] for x in r) / max(len(r), 1)

    print("\n--- Baseline summary ---")
    print(f"Medical    EN={acc(en_med):.1%}  ES={acc(es_med):.1%}  FR={acc(fr_med):.1%}")
    print(f"Control    EN={acc(en_ctrl):.1%}  ES={acc(es_ctrl):.1%}  FR={acc(fr_ctrl):.1%}")
    print(f"Med gap:   ES-EN={acc(es_med)-acc(en_med):+.1%}  FR-EN={acc(fr_med)-acc(en_med):+.1%}")
    print(f"Ctrl gap:  ES-EN={acc(es_ctrl)-acc(en_ctrl):+.1%}  FR-EN={acc(fr_ctrl)-acc(en_ctrl):+.1%}")

    # Bootstrap CIs on medical gap
    en_med_acc, en_med_ci = bootstrap_ci(en_med)
    es_med_acc, es_med_ci = bootstrap_ci(es_med)
    print(f"\nMedical EN (bootstrap 95% CI): {en_med_acc:.1%} [{en_med_ci[0]:.1%}, {en_med_ci[1]:.1%}]")
    print(f"Medical ES (bootstrap 95% CI): {es_med_acc:.1%} [{es_med_ci[0]:.1%}, {es_med_ci[1]:.1%}]")

    # ========================================================
    # PHASE B: Feature identification
    # ========================================================
    print("\n" + "=" * 70)
    print("PHASE B: Identify Spanish feature")
    print("=" * 70)

    candidates = identify_spanish_feature(model, tokenizer, sae, med_pairs, STEER_LAYER)
    if not candidates:
        print("\nNo clean Spanish features found. Aborting rescue.")
        return

    top_es_feature = candidates[0]
    print(f"\nTop ES feature: {top_es_feature['feature_idx']} (mean_act={top_es_feature['es_mean']:.0f})")

    # ========================================================
    # PHASE C: Single-feature rescue (TRUE single feature)
    # ========================================================
    print("\n" + "=" * 70)
    print("PHASE C: SINGLE-feature rescue, full-benchmark")
    print("=" * 70)
    print(f"Using ONLY feature {top_es_feature['feature_idx']}")

    single_feature_results = {}
    for strength in STEER_MULTIPLIERS:
        delta = strength * top_es_feature["es_mean"]
        feature_deltas = {top_es_feature["feature_idx"]: delta} if strength > 0 else {}
        intervention = (lambda fd=feature_deltas: steer_features(model, sae, STEER_LAYER, fd))

        print(f"\n  Strength {strength}x (delta={delta:.0f})...")
        steered = evaluate_all(model, tokenizer, med_pairs, "en", answer_tids,
                                f"single-{strength}x", intervention=intervention)
        s_acc, s_ci = bootstrap_ci(steered)
        delta_acc = s_acc - en_med_acc

        single_feature_results[strength] = {
            "accuracy": s_acc,
            "ci_95": list(s_ci),
            "delta_vs_baseline": delta_acc,
            "predictions": [{"correct": r["correct"], "subject": r["subject"]} for r in steered],
        }
        print(f"    Acc={s_acc:.1%} [{s_ci[0]:.1%}, {s_ci[1]:.1%}]  Δ={delta_acc:+.1%}")

    # ========================================================
    # PHASE D: Random-feature control
    # ========================================================
    print("\n" + "=" * 70)
    print("PHASE D: Random-feature control")
    print("=" * 70)

    exclude = [c["feature_idx"] for c in candidates]
    random_feature = pick_random_control_feature(
        model, tokenizer, sae, med_pairs, STEER_LAYER,
        target_mean=top_es_feature["es_mean"],
        exclude_top_k=exclude,
    )

    random_results = {}
    for strength in STEER_MULTIPLIERS:
        delta = strength * top_es_feature["es_mean"]  # Same absolute delta as ES feature
        feature_deltas = {random_feature["feature_idx"]: delta} if strength > 0 else {}
        intervention = (lambda fd=feature_deltas: steer_features(model, sae, STEER_LAYER, fd))

        print(f"\n  Strength {strength}x (delta={delta:.0f}) on random feature {random_feature['feature_idx']}...")
        steered = evaluate_all(model, tokenizer, med_pairs, "en", answer_tids,
                                f"random-{strength}x", intervention=intervention)
        s_acc, s_ci = bootstrap_ci(steered)
        delta_acc = s_acc - en_med_acc
        random_results[strength] = {
            "accuracy": s_acc,
            "ci_95": list(s_ci),
            "delta_vs_baseline": delta_acc,
            "predictions": [{"correct": r["correct"], "subject": r["subject"]} for r in steered],
        }
        print(f"    Acc={s_acc:.1%} [{s_ci[0]:.1%}, {s_ci[1]:.1%}]  Δ={delta_acc:+.1%}")

    # ========================================================
    # PHASE E: Top-3 features (for comparison with v2)
    # ========================================================
    print("\n" + "=" * 70)
    print("PHASE E: Top-3 features (replicating v2 setup)")
    print("=" * 70)

    top3 = candidates[:3]
    print(f"Features: {[f['feature_idx'] for f in top3]}")

    top3_results = {}
    for strength in STEER_MULTIPLIERS:
        if strength > 0:
            feature_deltas = {f["feature_idx"]: strength * f["es_mean"] for f in top3}
        else:
            feature_deltas = {}
        intervention = (lambda fd=feature_deltas: steer_features(model, sae, STEER_LAYER, fd))

        print(f"\n  Strength {strength}x...")
        steered = evaluate_all(model, tokenizer, med_pairs, "en", answer_tids,
                                f"top3-{strength}x", intervention=intervention)
        s_acc, s_ci = bootstrap_ci(steered)
        delta_acc = s_acc - en_med_acc
        top3_results[strength] = {
            "accuracy": s_acc,
            "ci_95": list(s_ci),
            "delta_vs_baseline": delta_acc,
            "predictions": [{"correct": r["correct"], "subject": r["subject"]} for r in steered],
        }
        print(f"    Acc={s_acc:.1%} [{s_ci[0]:.1%}, {s_ci[1]:.1%}]  Δ={delta_acc:+.1%}")

    # ========================================================
    # SUMMARY
    # ========================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\nPaired medical: {len(med_pairs)}")
    print(f"Baseline EN: {en_med_acc:.1%} [{en_med_ci[0]:.1%}, {en_med_ci[1]:.1%}]")
    print(f"Baseline ES: {es_med_acc:.1%} [{es_med_ci[0]:.1%}, {es_med_ci[1]:.1%}]")
    print(f"True gap (ES - EN): {es_med_acc - en_med_acc:+.1%}")

    print(f"\n{'Strength':>10s} {'Single':>18s} {'Random':>18s} {'Top-3':>18s}")
    print(f"{'Baseline':>10s} {en_med_acc:>17.1%}%")
    for s in STEER_MULTIPLIERS:
        if s == 0:
            continue
        print(f"{s:>9.1f}x "
              f"{single_feature_results[s]['accuracy']:>9.1%} ({single_feature_results[s]['delta_vs_baseline']:+.1%})  "
              f"{random_results[s]['accuracy']:>9.1%} ({random_results[s]['delta_vs_baseline']:+.1%})  "
              f"{top3_results[s]['accuracy']:>9.1%} ({top3_results[s]['delta_vs_baseline']:+.1%})")

    # Save everything
    output = {
        "model": MODEL_ID,
        "sae_id": SAE_ID,
        "n_paired_medical": len(med_pairs),
        "n_paired_control": len(ctrl_pairs),
        "subject_stats": subj_stats,
        "baselines": {
            "en_medical": {"accuracy": en_med_acc, "ci_95": list(en_med_ci)},
            "es_medical": {"accuracy": es_med_acc, "ci_95": list(es_med_ci)},
            "fr_medical": {"accuracy": acc(fr_med)},
            "en_control": {"accuracy": acc(en_ctrl)},
            "es_control": {"accuracy": acc(es_ctrl)},
            "fr_control": {"accuracy": acc(fr_ctrl)},
        },
        "top_es_feature": top_es_feature,
        "random_control_feature": random_feature,
        "candidate_features": candidates,
        "single_feature_results": single_feature_results,
        "random_feature_results": random_results,
        "top3_results": top3_results,
    }
    with open("results/v3_replication.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/v3_replication.json")

    del model, sae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
