"""
Hello World 5: Multilingual smoke test.
Runs EN/ES/FR through model + SAE, compares feature activations.

IMPORTANT: Edit the constants below based on hw2/hw3/hw4 output.
"""
import torch
import numpy as np

# ============================================================
# EDIT THESE based on previous hello-world results
# ============================================================
SAE_RELEASE = "gemma-scope-2-1b-it-resid_post"  # From hw3
SAE_ID_TEMPLATE = "layer_{layer}_width_16k_l0_medium"  # From hw3
TARGET_LAYERS = [6, 12, 22]  # Edit based on hw3's available_layers
USE_CHAT_TEMPLATE = True  # Try both True and False
# ============================================================

print("=== HW5: Multilingual Smoke Test ===")

from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it", device_map="cuda", torch_dtype=torch.bfloat16)

# Load SAEs for target layers
print(f"Loading SAEs for layers {TARGET_LAYERS}...")
saes = {}
for layer in TARGET_LAYERS:
    sae_id = SAE_ID_TEMPLATE.format(layer=layer)
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device="cuda")
    saes[layer] = sae
    print(f"  Layer {layer}: d_sae={sae.cfg.d_sae}")

# Parallel triples for testing
triples = [
    {
        "id": "test_01",
        "topic": "Eiffel Tower",
        "en": "The Eiffel Tower is a famous iron structure located in Paris.",
        "es": "La Torre Eiffel es una famosa estructura de hierro ubicada en Paris.",
        "fr": "La Tour Eiffel est une celebre structure en fer situee a Paris.",
    },
    {
        "id": "test_02",
        "topic": "Water",
        "en": "Water boils at one hundred degrees Celsius at sea level.",
        "es": "El agua hierve a cien grados Celsius al nivel del mar.",
        "fr": "L'eau bout a cent degres Celsius au niveau de la mer.",
    },
    {
        "id": "test_03",
        "topic": "Music",
        "en": "Music has the power to bring people together across cultures.",
        "es": "La musica tiene el poder de unir a las personas entre culturas.",
        "fr": "La musique a le pouvoir de rassembler les gens a travers les cultures.",
    },
]


def get_text(text):
    """Optionally wrap in chat template."""
    if USE_CHAT_TEMPLATE:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=False,
        )
    return text


def extract_and_encode(text, layers):
    """Extract residual stream at layers and encode with SAEs."""
    processed_text = get_text(text)
    inputs = tokenizer(processed_text, return_tensors="pt").to("cuda")

    captured = {}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            captured[idx] = out.detach()
        return hook_fn

    for idx in layers:
        h = model.model.layers[idx].register_forward_hook(make_hook(idx))
        hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    # Encode with SAEs
    results = {}
    for idx in layers:
        raw = captured[idx].to(saes[idx].dtype)
        features = saes[idx].encode(raw)

        # Mean-pooled representation (average across tokens)
        mean_resid = captured[idx].float().mean(dim=1).squeeze(0)  # [hidden_size]
        mean_features = features.float().mean(dim=1).squeeze(0)  # [d_sae]

        # Last-token representation
        last_resid = captured[idx][:, -1, :].float().squeeze(0)
        last_features = features[:, -1, :].float().squeeze(0)

        results[idx] = {
            "raw_mean": mean_resid,
            "raw_last": last_resid,
            "features_mean": mean_features,
            "features_last": last_features,
            "features_full": features.float().squeeze(0),  # [seq_len, d_sae]
            "l0": (features > 0).float().sum(dim=-1).mean().item(),
        }

    return results


# Run all triples
print(f"\n--- Running {len(triples)} triples x 3 languages x {len(TARGET_LAYERS)} layers ---")
all_results = {}
for triple in triples:
    for lang in ["en", "es", "fr"]:
        key = (triple["id"], lang)
        all_results[key] = extract_and_encode(triple[lang], TARGET_LAYERS)
        l0s = {l: all_results[key][l]["l0"] for l in TARGET_LAYERS}
        print(f"  {triple['id']}/{lang}: L0={l0s}")

# Analysis: Cross-lingual cosine similarity at each layer
print(f"\n--- Cross-lingual Cosine Similarity (residual stream) ---")
cos_sim = torch.nn.functional.cosine_similarity

for layer in TARGET_LAYERS:
    print(f"\nLayer {layer}:")
    for triple in triples:
        en = all_results[(triple["id"], "en")][layer]["raw_mean"]
        es = all_results[(triple["id"], "es")][layer]["raw_mean"]
        fr = all_results[(triple["id"], "fr")][layer]["raw_mean"]

        sim_en_es = cos_sim(en, es, dim=0).item()
        sim_en_fr = cos_sim(en, fr, dim=0).item()
        sim_es_fr = cos_sim(es, fr, dim=0).item()

        print(f"  {triple['topic']:15s}: EN-ES={sim_en_es:.4f}  EN-FR={sim_en_fr:.4f}  ES-FR={sim_es_fr:.4f}")

# Analysis: Feature overlap
print(f"\n--- Feature Overlap (shared active features) ---")
for layer in TARGET_LAYERS:
    print(f"\nLayer {layer}:")
    for triple in triples:
        # Features that fire on average (mean activation > 0)
        en_active = set((all_results[(triple["id"], "en")][layer]["features_mean"] > 0).nonzero().squeeze(-1).tolist())
        es_active = set((all_results[(triple["id"], "es")][layer]["features_mean"] > 0).nonzero().squeeze(-1).tolist())
        fr_active = set((all_results[(triple["id"], "fr")][layer]["features_mean"] > 0).nonzero().squeeze(-1).tolist())

        all_three = en_active & es_active & fr_active
        any_lang = en_active | es_active | fr_active
        en_only = en_active - es_active - fr_active
        es_only = es_active - en_active - fr_active
        fr_only = fr_active - en_active - es_active

        print(f"  {triple['topic']:15s}: "
              f"shared={len(all_three):3d}  "
              f"EN-only={len(en_only):3d}  "
              f"ES-only={len(es_only):3d}  "
              f"FR-only={len(fr_only):3d}  "
              f"total={len(any_lang):3d}")

print(f"\n=== HW5 PASSED ===")
print("Multilingual pipeline works. Key observations above.")
print("\nNext steps:")
print("  1. Draft your corpus.json (see corpus_template.json)")
print("  2. Run the v1 exploration: python3 exploratory/v1_exploration.py")

del model, saes
torch.cuda.empty_cache()
