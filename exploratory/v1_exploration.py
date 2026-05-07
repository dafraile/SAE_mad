"""
v1 SAE Exploration: Cross-lingual representation analysis for Gemma 3 1B IT.

Run after completing all hello-world scripts and editing the constants below.
Requires corpus.json in the same directory.

Usage:
    python3 v1_exploration.py              # Run all analyses
    python3 v1_exploration.py --collect     # Only collect activations
    python3 v1_exploration.py --analyze     # Only run analyses (needs cached activations)
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for remote servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE

# ============================================================
# CONFIGURATION -- Verified from hello-world results
# ============================================================
MODEL_ID = "google/gemma-3-1b-it"
# Layer 7 is in res-all (only has small/big), layers 13/22 are in res (has small/medium/big)
SAE_CONFIG = {
    7:  {"release": "gemma-scope-2-1b-it-res-all", "sae_id": "layer_7_width_16k_l0_small"},
    13: {"release": "gemma-scope-2-1b-it-res",     "sae_id": "layer_13_width_16k_l0_medium"},
    22: {"release": "gemma-scope-2-1b-it-res",     "sae_id": "layer_22_width_16k_l0_medium"},
}
TARGET_LAYERS = [7, 13, 22]  # early (~27%), middle (~50%), late (~85%)
USE_CHAT_TEMPLATE = False  # HookedSAETransformer handles raw text directly
CORPUS_FILE = "corpus.json"
CACHE_FILE = "cached_activations.pt"
RESULTS_DIR = "results"
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# SECTION 1: Model and SAE Loading
# ============================================================

def load_model():
    print("Loading model via HuggingFace transformers (memory-efficient)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16)
    print(f"  Loaded. Hidden size: {model.config.hidden_size}, "
          f"Layers: {model.config.num_hidden_layers}")
    return model, tokenizer


def load_saes():
    print(f"Loading SAEs for layers {TARGET_LAYERS}...")
    saes = {}
    for layer in TARGET_LAYERS:
        cfg = SAE_CONFIG[layer]
        saes[layer] = SAE.from_pretrained(
            release=cfg["release"], sae_id=cfg["sae_id"], device="cuda")
        print(f"  Layer {layer}: d_sae={saes[layer].cfg.d_sae}")
    return saes


# ============================================================
# SECTION 2: Corpus Loading and Validation
# ============================================================

def load_corpus(path):
    print(f"Loading corpus from {path}...")
    with open(path) as f:
        corpus = json.load(f)

    triples = corpus["parallel_triples"]
    cross = corpus.get("cross_cultural", [])

    print(f"  {len(triples)} parallel triples, {len(cross)} cross-cultural items")

    # Validate
    for item in triples:
        for field in ["id", "topic", "cultural_weight", "text_en", "text_es", "text_fr"]:
            assert field in item, f"Missing '{field}' in item {item.get('id', '?')}"
        assert item["cultural_weight"] in ("neutral", "moderate", "strong"), \
            f"Invalid cultural_weight in {item['id']}: {item['cultural_weight']}"

    for item in cross:
        for field in ["id", "topic", "cultural_tag", "language_of_text", "text"]:
            assert field in item, f"Missing '{field}' in cross-cultural item {item.get('id', '?')}"

    return triples, cross


def tokenization_audit(tokenizer, triples):
    """Print token counts to catch problematic length differences."""
    print("\n--- Tokenization Audit ---")
    print(f"{'ID':15s} {'Topic':15s} {'EN':>5s} {'ES':>5s} {'FR':>5s} {'Max/Min':>8s}")
    print("-" * 65)
    for item in triples:
        counts = {}
        for lang in ["en", "es", "fr"]:
            text = item[f"text_{lang}"]
            counts[lang] = len(tokenizer.encode(text))
        ratio = max(counts.values()) / max(min(counts.values()), 1)
        flag = " !!!" if ratio > 2.0 else ""
        print(f"{item['id']:15s} {item['topic']:15s} "
              f"{counts['en']:5d} {counts['es']:5d} {counts['fr']:5d} "
              f"{ratio:7.2f}x{flag}")
    print("(Items with >2x ratio flagged with !!!)")


# ============================================================
# SECTION 3: Activation Collection
# ============================================================

def extract_activations(model, tokenizer, text, layer_indices):
    """Extract residual stream activations using HuggingFace model + manual hooks."""
    captured = {}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            captured[idx] = o.detach().cpu()
        return hook_fn

    for idx in layer_indices:
        h = model.model.layers[idx].register_forward_hook(make_hook(idx))
        hooks.append(h)

    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    n_tokens = inputs["input_ids"].shape[1]
    del inputs
    torch.cuda.empty_cache()
    return captured, n_tokens


def collect_all_activations(model, tokenizer, saes, triples, cross_cultural):
    """Run all texts through model + SAEs and cache results."""
    print("\n=== Collecting Activations ===")

    data = {
        "triples": {},
        "cross_cultural": {},
        "config": {
            "target_layers": TARGET_LAYERS,
            "model_id": MODEL_ID,
            "sae_config": {str(k): v for k, v in SAE_CONFIG.items()},
        }
    }

    # Process parallel triples
    total = len(triples) * 3
    count = 0
    for item in triples:
        for lang in ["en", "es", "fr"]:
            count += 1
            text = item[f"text_{lang}"]
            key = f"{item['id']}_{lang}"
            print(f"  [{count}/{total}] {key}...", end=" ", flush=True)

            raw_acts, n_tokens = extract_activations(
                model, tokenizer, text, TARGET_LAYERS)

            # Encode with SAEs
            layer_data = {}
            for layer in TARGET_LAYERS:
                raw = raw_acts[layer].to(saes[layer].device).to(saes[layer].dtype)
                features = saes[layer].encode(raw).cpu()

                layer_data[layer] = {
                    "raw_mean": raw_acts[layer].float().mean(dim=1).squeeze(0),
                    "raw_last": raw_acts[layer][:, -1, :].float().squeeze(0),
                    "features_mean": features.float().mean(dim=1).squeeze(0),
                    "features_full": features.float().squeeze(0),
                    "l0": (features > 0).float().sum(dim=-1).mean().item(),
                    "n_tokens": n_tokens,
                }

            data["triples"][key] = {
                "id": item["id"],
                "lang": lang,
                "topic": item["topic"],
                "cultural_weight": item["cultural_weight"],
                "cultural_tag": item.get("cultural_tag"),
                "layers": layer_data,
            }
            print(f"{n_tokens} tokens, L0={[layer_data[l]['l0'] for l in TARGET_LAYERS]}")

    # Process cross-cultural items
    for item in cross_cultural:
        key = item["id"]
        print(f"  [cross] {key}...", end=" ", flush=True)

        raw_acts, n_tokens = extract_activations(
            model, tokenizer, item["text"], TARGET_LAYERS)

        layer_data = {}
        for layer in TARGET_LAYERS:
            raw = raw_acts[layer].to(saes[layer].device).to(saes[layer].dtype)
            features = saes[layer].encode(raw).cpu()

            layer_data[layer] = {
                "raw_mean": raw_acts[layer].float().mean(dim=1).squeeze(0),
                "raw_last": raw_acts[layer][:, -1, :].float().squeeze(0),
                "features_mean": features.float().mean(dim=1).squeeze(0),
                "features_full": features.float().squeeze(0),
                "l0": (features > 0).float().sum(dim=-1).mean().item(),
                "n_tokens": n_tokens,
            }

        data["cross_cultural"][key] = {
            "id": item["id"],
            "topic": item["topic"],
            "cultural_tag": item["cultural_tag"],
            "language_of_text": item["language_of_text"],
            "loanwords": item.get("loanwords", True),
            "layers": layer_data,
        }
        print(f"{n_tokens} tokens")

    return data


# ============================================================
# SECTION 4: Analysis 1 -- Cross-lingual Representation Similarity
# ============================================================

def analysis_1_similarity(data, triples):
    """Cosine similarity between language pairs at each layer."""
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Cross-lingual Representation Similarity")
    print("=" * 60)

    results = {layer: {"en_es": [], "en_fr": [], "es_fr": []}
               for layer in TARGET_LAYERS}
    results_by_weight = {w: {layer: {"en_es": [], "en_fr": [], "es_fr": []}
                             for layer in TARGET_LAYERS}
                         for w in ["neutral", "moderate", "strong"]}

    for item in triples:
        iid = item["id"]
        weight = item["cultural_weight"]

        for layer in TARGET_LAYERS:
            en = data["triples"][f"{iid}_en"]["layers"][layer]["raw_mean"]
            es = data["triples"][f"{iid}_es"]["layers"][layer]["raw_mean"]
            fr = data["triples"][f"{iid}_fr"]["layers"][layer]["raw_mean"]

            sim_en_es = F.cosine_similarity(en, es, dim=0).item()
            sim_en_fr = F.cosine_similarity(en, fr, dim=0).item()
            sim_es_fr = F.cosine_similarity(es, fr, dim=0).item()

            results[layer]["en_es"].append(sim_en_es)
            results[layer]["en_fr"].append(sim_en_fr)
            results[layer]["es_fr"].append(sim_es_fr)

            results_by_weight[weight][layer]["en_es"].append(sim_en_es)
            results_by_weight[weight][layer]["en_fr"].append(sim_en_fr)
            results_by_weight[weight][layer]["es_fr"].append(sim_es_fr)

    # Print summary
    print(f"\n{'Layer':>6s}  {'EN-ES':>10s}  {'EN-FR':>10s}  {'ES-FR':>10s}")
    print("-" * 42)
    for layer in TARGET_LAYERS:
        en_es = np.mean(results[layer]["en_es"])
        en_fr = np.mean(results[layer]["en_fr"])
        es_fr = np.mean(results[layer]["es_fr"])
        print(f"{layer:6d}  {en_es:10.4f}  {en_fr:10.4f}  {es_fr:10.4f}")

    # Print by cultural weight
    for weight in ["neutral", "moderate", "strong"]:
        print(f"\n  [{weight}]")
        for layer in TARGET_LAYERS:
            r = results_by_weight[weight][layer]
            if r["en_es"]:
                print(f"    Layer {layer}: EN-ES={np.mean(r['en_es']):.4f}  "
                      f"EN-FR={np.mean(r['en_fr']):.4f}  "
                      f"ES-FR={np.mean(r['es_fr']):.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall
    ax = axes[0]
    for pair, label in [("en_es", "EN-ES"), ("en_fr", "EN-FR"), ("es_fr", "ES-FR")]:
        means = [np.mean(results[l][pair]) for l in TARGET_LAYERS]
        stds = [np.std(results[l][pair]) for l in TARGET_LAYERS]
        ax.errorbar(TARGET_LAYERS, means, yerr=stds, marker="o", label=label, capsize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cross-lingual Similarity vs Layer Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # By cultural weight
    ax = axes[1]
    colors = {"neutral": "blue", "moderate": "orange", "strong": "red"}
    for weight in ["neutral", "moderate", "strong"]:
        # Average across all pairs
        means = []
        for l in TARGET_LAYERS:
            r = results_by_weight[weight][l]
            if r["en_es"]:
                all_sims = r["en_es"] + r["en_fr"] + r["es_fr"]
                means.append(np.mean(all_sims))
            else:
                means.append(float("nan"))
        ax.plot(TARGET_LAYERS, means, marker="o", label=weight, color=colors[weight])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity (avg across pairs)")
    ax.set_title("Similarity by Cultural Weight")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/analysis1_similarity.png", dpi=150)
    print(f"\nSaved plot: {RESULTS_DIR}/analysis1_similarity.png")

    return results


def analysis_1b_lasttoken_comparison(data, triples):
    """Compare mean-pooled vs last-token cosine similarity.

    Design Claude flagged that 0.999+ mean-pooled similarity might be
    artifactually high because concept tokens dominate the average.
    Last-token should show sharper language divergence.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 1b: Mean-pooled vs Last-token Similarity")
    print("=" * 60)

    print(f"\n{'Layer':>6s}  {'Mean-pool':>10s}  {'Last-tok':>10s}  {'Delta':>8s}")
    print("-" * 40)

    for layer in TARGET_LAYERS:
        mean_sims = []
        last_sims = []
        for item in triples:
            iid = item["id"]
            for pair in [("en", "es"), ("en", "fr"), ("es", "fr")]:
                mean_a = data["triples"][f"{iid}_{pair[0]}"]["layers"][layer]["raw_mean"]
                mean_b = data["triples"][f"{iid}_{pair[1]}"]["layers"][layer]["raw_mean"]
                last_a = data["triples"][f"{iid}_{pair[0]}"]["layers"][layer]["raw_last"]
                last_b = data["triples"][f"{iid}_{pair[1]}"]["layers"][layer]["raw_last"]

                mean_sims.append(F.cosine_similarity(mean_a, mean_b, dim=0).item())
                last_sims.append(F.cosine_similarity(last_a, last_b, dim=0).item())

        avg_mean = np.mean(mean_sims)
        avg_last = np.mean(last_sims)
        delta = avg_mean - avg_last
        print(f"{layer:6d}  {avg_mean:10.4f}  {avg_last:10.4f}  {delta:+8.4f}")

    print("\nIf last-token shows lower similarity, mean-pooling was hiding")
    print("language-specific structure. If both are 0.999+, the convergence is real.")


# ============================================================
# SECTION 5: Analysis 2 -- Language-specific vs Language-agnostic Features
# ============================================================

def analysis_2_features(data, triples):
    """Identify features that fire differently across languages."""
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Language-specific vs Language-agnostic Features")
    print("=" * 60)

    lang_features = {}

    for layer in TARGET_LAYERS:
        print(f"\n--- Layer {layer} ---")

        # Collect per-language feature activations
        lang_acts = {"en": [], "es": [], "fr": []}
        for item in triples:
            for lang in ["en", "es", "fr"]:
                key = f"{item['id']}_{lang}"
                feats = data["triples"][key]["layers"][layer]["features_mean"]
                lang_acts[lang].append(feats)

        # Stack: [n_items, d_sae]
        for lang in lang_acts:
            lang_acts[lang] = torch.stack(lang_acts[lang])

        n_items = lang_acts["en"].shape[0]
        d_sae = lang_acts["en"].shape[1]

        # For each feature, compute language specificity
        # Metric: fraction of items where feature fires for this lang but not others
        lang_specific = {"en": [], "es": [], "fr": []}
        lang_agnostic = []

        for feat_idx in range(d_sae):
            en_fires = (lang_acts["en"][:, feat_idx] > 0).float()
            es_fires = (lang_acts["es"][:, feat_idx] > 0).float()
            fr_fires = (lang_acts["fr"][:, feat_idx] > 0).float()

            en_rate = en_fires.mean().item()
            es_rate = es_fires.mean().item()
            fr_rate = fr_fires.mean().item()

            all_rate = min(en_rate, es_rate, fr_rate)
            max_rate = max(en_rate, es_rate, fr_rate)

            if max_rate < 0.1:
                continue  # Feature rarely fires, skip

            # Language-agnostic: fires in all languages at similar rates
            if all_rate > 0.3 and max_rate / max(all_rate, 0.01) < 2.0:
                avg_act = sum(lang_acts[l][:, feat_idx].mean().item() for l in ["en", "es", "fr"]) / 3
                lang_agnostic.append((feat_idx, all_rate, avg_act))

            # Language-specific: fires much more for one language
            for lang, rate, fires in [("en", en_rate, en_fires),
                                       ("es", es_rate, es_fires),
                                       ("fr", fr_rate, fr_fires)]:
                other_rates = [r for l, r in [("en", en_rate), ("es", es_rate), ("fr", fr_rate)] if l != lang]
                avg_other = np.mean(other_rates)
                if rate > 0.3 and rate > 2 * max(avg_other, 0.01):
                    avg_act = lang_acts[lang][:, feat_idx].mean().item()
                    lang_specific[lang].append((feat_idx, rate, avg_other, avg_act))

        # Sort by consistency (fire rate) and print top features
        for lang in ["en", "es", "fr"]:
            lang_specific[lang].sort(key=lambda x: x[1], reverse=True)
            print(f"\n  Top {lang.upper()}-specific features (fire rate, other_avg, mean_act):")
            for feat_idx, rate, other_rate, avg_act in lang_specific[lang][:15]:
                print(f"    Feature {feat_idx:5d}: rate={rate:.2f}, other={other_rate:.2f}, "
                      f"act={avg_act:.4f}")

        lang_agnostic.sort(key=lambda x: x[2], reverse=True)
        print(f"\n  Top language-agnostic features (min_rate, mean_act):")
        for feat_idx, min_rate, avg_act in lang_agnostic[:15]:
            print(f"    Feature {feat_idx:5d}: min_rate={min_rate:.2f}, act={avg_act:.4f}")

        print(f"\n  Summary: {sum(len(v) for v in lang_specific.values())} language-specific, "
              f"{len(lang_agnostic)} language-agnostic features")

        lang_features[layer] = {
            "specific": lang_specific,
            "agnostic": lang_agnostic,
        }

    return lang_features


# ============================================================
# SECTION 6: Analysis 3 -- Culture-Language Entanglement
# ============================================================

def analysis_3_entanglement(data, triples, cross_cultural, lang_features):
    """Test whether language-specific features fire on cross-cultural items."""
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Culture-Language Entanglement (THE KEY ONE)")
    print("=" * 60)

    if not cross_cultural:
        print("No cross-cultural items in corpus. Skipping.")
        return None

    results = []

    for layer in TARGET_LAYERS:
        print(f"\n--- Layer {layer} ---")

        specific = lang_features[layer]["specific"]

        for cross_item in cross_cultural:
            cross_key = cross_item["id"]
            if cross_key not in data["cross_cultural"]:
                continue

            cross_data = data["cross_cultural"][cross_key]
            cultural_tag = cross_data["cultural_tag"]
            text_lang = cross_data["language_of_text"]

            # Map cultural_tag to language
            culture_to_lang = {"spanish": "es", "french": "fr", "english": "en"}
            cultural_lang = culture_to_lang.get(cultural_tag)
            if cultural_lang is None or cultural_lang not in specific:
                continue

            # Get the cultural-language-specific features
            cultural_features = specific[cultural_lang]
            if not cultural_features:
                continue

            # Check: do these features fire on this cross-cultural text?
            cross_feats = cross_data["layers"][layer]["features_mean"]

            # Take top N cultural-language features
            top_n = min(20, len(cultural_features))
            top_feat_indices = [f[0] for f in cultural_features[:top_n]]

            # What fraction of cultural-language features fire?
            fires = sum(1 for idx in top_feat_indices if cross_feats[idx] > 0)
            fire_rate = fires / top_n

            # Baseline: what fraction fire on a neutral text in the same text_lang?
            baseline_fires = []
            for item in triples:
                if item["cultural_weight"] == "neutral":
                    baseline_key = f"{item['id']}_{text_lang}"
                    if baseline_key in data["triples"]:
                        baseline_feats = data["triples"][baseline_key]["layers"][layer]["features_mean"]
                        bf = sum(1 for idx in top_feat_indices if baseline_feats[idx] > 0)
                        baseline_fires.append(bf / top_n)

            baseline_rate = np.mean(baseline_fires) if baseline_fires else 0.0

            result = {
                "cross_id": cross_key,
                "cultural_tag": cultural_tag,
                "text_lang": text_lang,
                "layer": layer,
                "fire_rate": fire_rate,
                "baseline_rate": baseline_rate,
                "ratio": fire_rate / max(baseline_rate, 0.01),
                "top_n": top_n,
            }
            results.append(result)

            entangled = "ENTANGLED" if fire_rate > 2 * baseline_rate else "SEPARABLE"
            print(f"  {cross_key}: {cultural_tag} culture in {text_lang} text")
            print(f"    Cultural features firing: {fires}/{top_n} = {fire_rate:.2%}")
            print(f"    Baseline (neutral {text_lang}): {baseline_rate:.2%}")
            print(f"    Ratio: {result['ratio']:.2f}x --> {entangled}")

    # Overall verdict
    if results:
        avg_ratio = np.mean([r["ratio"] for r in results])
        print(f"\n{'=' * 40}")
        print(f"OVERALL: Average cross-cultural/baseline ratio = {avg_ratio:.2f}x")
        if avg_ratio > 2.0:
            print("VERDICT: Language and culture appear ENTANGLED in feature space.")
            print("v2 implication: Cross-lingual routing won't be clean. Consider")
            print("  entity-property composition or moving to v3 multimodal earlier.")
        elif avg_ratio > 1.3:
            print("VERDICT: PARTIAL entanglement. Some culture leaks into language features.")
            print("v2 implication: May work with careful feature selection.")
        else:
            print("VERDICT: Language and culture appear SEPARABLE in feature space.")
            print("v2 implication: Cross-lingual factual recall task is viable.")
            print("  Can use language features as routing signals independent of content.")
        print(f"{'=' * 40}")

    return results


def analysis_3b_loanword_comparison(data, cross_cultural, lang_features):
    """Compare entanglement for cross-cultural items WITH vs WITHOUT loanwords.

    Tests the design Claude's hypothesis: is the entanglement driven by
    lexical borrowing (corrida, matador, terroir) or by deeper conceptual
    cultural association?
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 3b: Loanword vs No-Loanword Entanglement")
    print("=" * 60)

    lw_ratios = []
    nolw_ratios = []

    for layer in TARGET_LAYERS:
        specific = lang_features[layer]["specific"]
        print(f"\n--- Layer {layer} ---")

        for cross_item in cross_cultural:
            cross_key = cross_item["id"]
            if cross_key not in data["cross_cultural"]:
                continue

            cross_data = data["cross_cultural"][cross_key]
            cultural_tag = cross_data["cultural_tag"]
            has_loanwords = cross_data.get("loanwords", True)

            culture_to_lang = {"spanish": "es", "french": "fr", "english": "en"}
            cultural_lang = culture_to_lang.get(cultural_tag)
            if cultural_lang is None or cultural_lang not in specific:
                continue

            cultural_features = specific[cultural_lang]
            if not cultural_features:
                continue

            cross_feats = cross_data["layers"][layer]["features_mean"]
            top_n = min(20, len(cultural_features))
            top_feat_indices = [f[0] for f in cultural_features[:top_n]]

            fires = sum(1 for idx in top_feat_indices if cross_feats[idx] > 0)
            fire_rate = fires / top_n

            tag = "LOANWORD" if has_loanwords else "NATIVE"
            print(f"  [{tag:8s}] {cross_key}: {cultural_tag} in {cross_data['language_of_text']} "
                  f"-> {fires}/{top_n} = {fire_rate:.0%}")

            if has_loanwords:
                lw_ratios.append(fire_rate)
            else:
                nolw_ratios.append(fire_rate)

    if lw_ratios and nolw_ratios:
        print(f"\n{'=' * 50}")
        print(f"With loanwords:    avg fire rate = {np.mean(lw_ratios):.1%} (n={len(lw_ratios)})")
        print(f"Without loanwords: avg fire rate = {np.mean(nolw_ratios):.1%} (n={len(nolw_ratios)})")
        diff = np.mean(lw_ratios) - np.mean(nolw_ratios)
        if diff > 0.1:
            print(f"Difference: {diff:+.1%} -> Loanwords AMPLIFY entanglement.")
            print("The entanglement is partly lexical routing through borrowed vocabulary.")
        elif diff > -0.05:
            print(f"Difference: {diff:+.1%} -> Similar rates. Entanglement is CONCEPTUAL, not lexical.")
        else:
            print(f"Difference: {diff:+.1%} -> Native vocabulary shows MORE entanglement (?). Unexpected.")
        print(f"{'=' * 50}")
    else:
        print("\nNeed both loanword and no-loanword items to compare.")


def analysis_4_per_token_attribution(data, model, tokenizer, saes, lang_features):
    """Show which tokens drive the top language-specific features.

    Distinguishes 'feature fires on Spanish-ness' from 'feature fires
    on specific Spanish tokens/loanwords'.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 4: Per-Token Feature Attribution")
    print("=" * 60)

    # Focus on layer 22 where language features are strongest
    layer = TARGET_LAYERS[-1]
    specific = lang_features[layer]["specific"]

    print(f"\nLayer {layer} - Top 5 features per language, with their peak tokens:\n")

    for lang in ["en", "es", "fr"]:
        top_feats = specific[lang][:5]
        if not top_feats:
            continue

        print(f"  === {lang.upper()}-specific features ===")
        for feat_idx, rate, other_rate, avg_act in top_feats:
            print(f"\n  Feature {feat_idx} (rate={rate:.0%}, other={other_rate:.0%}):")

            # Find which texts and tokens drive this feature
            for item_key, item_data in data["triples"].items():
                feat_full = item_data["layers"][layer]["features_full"]  # [seq_len, d_sae]
                token_acts = feat_full[:, feat_idx]

                if token_acts.max() > 0:
                    # Get the text and tokenize it
                    parts = item_key.rsplit("_", 1)
                    item_lang = parts[-1]
                    item_id = parts[0]

                    # Find the original text
                    for triple in data["triples"].values():
                        if triple["id"] == item_id and triple["lang"] == item_lang:
                            break

                    # Get tokens from model
                    text_field = f"text_{item_lang}"
                    # We need to find the original text - search in the triples data
                    # For now, show the top 3 token positions with their activations
                    top_positions = token_acts.topk(min(3, len(token_acts)))
                    if top_positions.values[0] > 0:
                        positions_str = ", ".join(
                            f"pos{top_positions.indices[i]}={top_positions.values[i]:.1f}"
                            for i in range(len(top_positions.values))
                            if top_positions.values[i] > 0
                        )
                        print(f"    {item_key}: {positions_str}")

            # Also check cross-cultural items
            for cross_key, cross_data in data["cross_cultural"].items():
                feat_full = cross_data["layers"][layer]["features_full"]
                token_acts = feat_full[:, feat_idx]
                if token_acts.max() > 0:
                    top_pos = token_acts.topk(min(3, len(token_acts)))
                    if top_pos.values[0] > 0:
                        positions_str = ", ".join(
                            f"pos{top_pos.indices[i]}={top_pos.values[i]:.1f}"
                            for i in range(len(top_pos.values))
                            if top_pos.values[i] > 0
                        )
                        lw_tag = "[LW]" if cross_data.get("loanwords") else "[NAT]"
                        print(f"    {lw_tag} {cross_key}: {positions_str}")

        print()


# ============================================================
# SECTION 7: Sanity Checks
# ============================================================

def run_sanity_checks(model, tokenizer, saes, data, triples):
    """Run verification checks before trusting analysis results."""
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # Check 1: SAE reconstruction quality
    print("\n--- Check 1: SAE Reconstruction Quality ---")
    test_texts = [triples[0]["text_en"], triples[0]["text_es"]]
    for text in test_texts:
        raw_acts, _ = extract_activations(model, tokenizer, text, TARGET_LAYERS)

        print(f"  Text: {text[:50]}...")
        for layer in TARGET_LAYERS:
            raw = raw_acts[layer].to(saes[layer].device)
            raw_sae = raw.to(saes[layer].dtype)
            features = saes[layer].encode(raw_sae)
            recon = saes[layer].decode(features)
            cos = F.cosine_similarity(raw_sae.reshape(-1).float(), recon.reshape(-1).float(), dim=0).item()
            mse = ((raw_sae.float() - recon.float()) ** 2).mean().item()
            print(f"    Layer {layer}: cos_sim={cos:.4f}, MSE={mse:.6f}")
            if cos < 0.8:
                print(f"    WARNING: Low reconstruction quality at layer {layer}!")
        del raw_acts

    # Check 2: Feature sparsity
    print("\n--- Check 2: Feature Sparsity ---")
    for layer in TARGET_LAYERS:
        l0s = []
        for item in triples[:5]:
            for lang in ["en", "es", "fr"]:
                key = f"{item['id']}_{lang}"
                if key in data["triples"]:
                    l0s.append(data["triples"][key]["layers"][layer]["l0"])
        avg_l0 = np.mean(l0s)
        print(f"  Layer {layer}: avg L0 = {avg_l0:.1f} features per token")
        if avg_l0 < 1:
            print(f"    WARNING: Very low L0, features may not be activating properly")
        elif avg_l0 > 1000:
            print(f"    WARNING: Very high L0, SAE may not be sparse enough")

    # Check 3: Chat template sensitivity
    print("\n--- Check 3: Chat Template Sensitivity ---")
    text = triples[0]["text_en"]

    # Without template (raw text, which is what we use)
    raw_plain, _ = extract_activations(model, tokenizer, text, TARGET_LAYERS[:1])

    # With template
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=False,
    )
    raw_chat, _ = extract_activations(model, tokenizer, chat_text, TARGET_LAYERS[:1])

    layer = TARGET_LAYERS[0]
    cos = F.cosine_similarity(
        raw_plain[layer].float().mean(dim=1).squeeze(),
        raw_chat[layer].float().mean(dim=1).squeeze(),
        dim=0
    ).item()
    print(f"  Layer {layer}: cos_sim(plain_text, chat_template) = {cos:.4f}")
    if cos < 0.9:
        print(f"  NOTE: Chat template significantly affects representations.")
        print(f"  Consider wrapping corpus texts in chat template for consistency.")

    print("\n--- Sanity checks complete ---")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="v1 SAE Exploration")
    parser.add_argument("--collect", action="store_true", help="Only collect activations")
    parser.add_argument("--analyze", action="store_true", help="Only run analyses")
    args = parser.parse_args()

    do_collect = not args.analyze  # collect unless --analyze only
    do_analyze = not args.collect  # analyze unless --collect only

    # Load corpus
    triples, cross_cultural = load_corpus(CORPUS_FILE)

    # Load model and SAEs
    print("\n--- Loading model and SAEs ---")
    model, tokenizer = load_model()
    saes = load_saes()

    # Tokenization audit
    tokenization_audit(tokenizer, triples)

    # Collect activations
    if do_collect:
        data = collect_all_activations(model, tokenizer, saes, triples, cross_cultural)
        print(f"\nSaving activations to {CACHE_FILE}...")
        torch.save(data, CACHE_FILE)
        print("Saved.")
    else:
        print(f"\nLoading cached activations from {CACHE_FILE}...")
        data = torch.load(CACHE_FILE, weights_only=False)
        print("Loaded.")

    # Run analyses
    if do_analyze:
        # Sanity checks first
        run_sanity_checks(model, tokenizer, saes, data, triples)

        # Analysis 1: Cross-lingual similarity (mean-pooled)
        sim_results = analysis_1_similarity(data, triples)

        # Analysis 1b: Mean-pooled vs last-token comparison
        analysis_1b_lasttoken_comparison(data, triples)

        # Analysis 2: Language-specific features
        lang_features = analysis_2_features(data, triples)

        # Analysis 3: Culture-language entanglement
        entanglement_results = analysis_3_entanglement(
            data, triples, cross_cultural, lang_features)

        # Analysis 3b: Loanword vs no-loanword entanglement
        analysis_3b_loanword_comparison(data, cross_cultural, lang_features)

        # Analysis 4: Per-token feature attribution
        analysis_4_per_token_attribution(data, model, tokenizer, saes, lang_features)

        # Save summary
        print(f"\n=== All analyses complete. Results in {RESULTS_DIR}/ ===")

    # Cleanup
    del model, saes
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
