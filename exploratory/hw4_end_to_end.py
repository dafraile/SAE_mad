"""
Hello World 4: End-to-end test -- model + SAE, capture feature activations.
Run after hw3 to confirm the full pipeline works.

IMPORTANT: Edit the constants below based on hw2/hw3 output.
"""
import torch
import sys

# ============================================================
# EDIT THESE based on hw2 and hw3 results
# ============================================================
USE_BRIDGE = False  # Set True if hw2_sae_bridge.py succeeded
BRIDGE_METHOD = None  # "HookedSAETransformer", "SAETransformerBridge", or "HookedTransformer"

SAE_RELEASE = "gemma-scope-2-1b-it-resid_post"  # From hw3
SAE_ID_TEMPLATE = "layer_{layer}_width_16k_l0_medium"  # From hw3, {layer} is placeholder
TARGET_LAYER = 12  # Pick one layer from hw3's available_layers
# ============================================================

print("=== HW4: End-to-End Pipeline Test ===")

from sae_lens import SAE

# Load SAE
sae_id = SAE_ID_TEMPLATE.format(layer=TARGET_LAYER)
print(f"Loading SAE: {SAE_RELEASE} / {sae_id}")
sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device="cuda")
print(f"SAE loaded. d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

text = "The Eiffel Tower is a wrought-iron lattice tower in Paris, France."

if USE_BRIDGE:
    # ---- Bridge path ----
    print(f"\nUsing bridge method: {BRIDGE_METHOD}")

    if BRIDGE_METHOD == "HookedSAETransformer":
        from sae_lens import HookedSAETransformer
        model = HookedSAETransformer.from_pretrained(
            "google/gemma-3-1b-it", device="cuda", dtype=torch.bfloat16)

        # Run with SAE and cache
        logits, cache = model.run_with_cache_with_saes(text, saes=[sae])
        print(f"\nCache keys containing '{TARGET_LAYER}':")
        for key in sorted(cache.keys()):
            if str(TARGET_LAYER) in key:
                print(f"  {key}: {cache[key].shape}")

    elif BRIDGE_METHOD == "SAETransformerBridge":
        from sae_lens.analysis.sae_transformer_bridge import SAETransformerBridge
        model = SAETransformerBridge.boot_transformers(
            "google/gemma-3-1b-it", device="cuda", torch_dtype=torch.bfloat16)
        logits, cache = model.run_with_cache_with_saes(text, saes=[sae])
        for key in sorted(cache.keys()):
            if str(TARGET_LAYER) in key:
                print(f"  {key}: {cache[key].shape}")

    elif BRIDGE_METHOD == "HookedTransformer":
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained(
            "google/gemma-3-1b-it", device="cuda", dtype=torch.bfloat16)
        logits, cache = model.run_with_cache(text)
        hook_name = sae.cfg.hook_name
        print(f"SAE hook name: {hook_name}")
        raw_acts = cache[hook_name]
        feature_acts = sae.encode(raw_acts)
        print(f"Feature activations shape: {feature_acts.shape}")

else:
    # ---- Fallback path: manual hooks ----
    print("\nUsing fallback path (manual hooks)")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it", device_map="cuda", torch_dtype=torch.bfloat16)

    # Extract activations
    captured = {}
    def hook_fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        captured["resid"] = out.detach()

    hook = model.model.layers[TARGET_LAYER].register_forward_hook(hook_fn)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**inputs)
    hook.remove()

    raw_acts = captured["resid"]
    print(f"Raw activations shape: {raw_acts.shape}")
    print(f"Raw activations dtype: {raw_acts.dtype}")

    # Pass through SAE -- may need float32
    raw_acts_for_sae = raw_acts.to(sae.dtype)
    feature_acts = sae.encode(raw_acts_for_sae)
    print(f"Feature activations shape: {feature_acts.shape}")

    # Reconstruction quality check
    reconstructed = sae.decode(feature_acts)
    mse = ((raw_acts_for_sae - reconstructed) ** 2).mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        raw_acts_for_sae.reshape(-1), reconstructed.reshape(-1), dim=0
    ).item()
    print(f"\nReconstruction quality:")
    print(f"  MSE: {mse:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")

    if cos_sim < 0.8:
        print("  WARNING: Low reconstruction quality!")
        print("  The hook point may not match the SAE's expected input.")
        print("  Try hooking at a different point (e.g., after layer norm).")
    else:
        print("  Reconstruction quality looks good.")

# Analyze the feature activations
print(f"\n--- Feature Activation Analysis ---")
print(f"Shape: {feature_acts.shape}")  # [1, seq_len, d_sae]
print(f"Total features: {feature_acts.shape[-1]}")

# L0: average number of active features per token
l0 = (feature_acts > 0).float().sum(dim=-1).mean().item()
print(f"Average L0 (active features per token): {l0:.1f}")

# Which features fire most strongly?
max_acts, max_indices = feature_acts.squeeze(0).max(dim=0)  # [d_sae]
top_k = 20
top_features = max_acts.topk(top_k)
print(f"\nTop {top_k} most active features:")
tokens_list = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) if not USE_BRIDGE else text.split()
for i in range(top_k):
    feat_idx = top_features.indices[i].item()
    feat_val = top_features.values[i].item()
    # Find which token activates this feature most
    token_acts = feature_acts[0, :, feat_idx]
    max_token_idx = token_acts.argmax().item()
    token_str = tokens_list[max_token_idx] if max_token_idx < len(tokens_list) else "?"
    print(f"  Feature {feat_idx:5d}: max_act={feat_val:.3f}, peak_token='{token_str}'")

print(f"\n=== HW4 PASSED ===")
print("Pipeline works end-to-end.")
print("Proceed to: python3 hw5_multilingual.py")

del model, sae
torch.cuda.empty_cache()
