"""
Hello World 2: Try loading Gemma 3 via SAELens/TransformerLens bridge.
This is the preferred path. If it fails, we fall back to manual hooks.
"""
import torch
import sys

print("=== HW2: SAELens/TransformerLens Bridge Test ===")

# First, let's see what's available in SAELens
import sae_lens
print(f"SAELens version: {sae_lens.__version__}")

# Try to discover the bridge API
print("\n--- Attempting SAELens bridge approaches ---")

# Approach A: HookedSAETransformer (traditional SAELens approach)
print("\n[Approach A] HookedSAETransformer...")
try:
    from sae_lens import HookedSAETransformer
    model = HookedSAETransformer.from_pretrained(
        "google/gemma-3-1b-it",
        device="cuda",
        dtype=torch.bfloat16,
    )
    print("SUCCESS: HookedSAETransformer loaded!")
    print(f"Type: {type(model)}")

    # Test forward pass
    logits = model("The capital of France is")
    print(f"Logits shape: {logits.shape}")

    # Test with cache
    logits, cache = model.run_with_cache("The capital of France is")
    print(f"Cache keys (first 10): {sorted(cache.keys())[:10]}")
    print("\n=== HW2 PASSED via Approach A (HookedSAETransformer) ===")
    BRIDGE_METHOD = "HookedSAETransformer"

except Exception as e:
    print(f"FAILED: {e}")
    model = None

# Approach B: TransformerBridge (newer SAELens API)
if model is None:
    print("\n[Approach B] SAETransformerBridge...")
    try:
        from sae_lens.analysis.sae_transformer_bridge import SAETransformerBridge
        model = SAETransformerBridge.boot_transformers(
            "google/gemma-3-1b-it",
            device="cuda",
            torch_dtype=torch.bfloat16,
        )
        print("SUCCESS: SAETransformerBridge loaded!")
        print(f"Type: {type(model)}")
        logits = model("The capital of France is")
        print(f"Logits shape: {logits.shape}")
        print("\n=== HW2 PASSED via Approach B (SAETransformerBridge) ===")
        BRIDGE_METHOD = "SAETransformerBridge"

    except Exception as e:
        print(f"FAILED: {e}")
        model = None

# Approach C: HookedTransformer directly from TransformerLens
if model is None:
    print("\n[Approach C] TransformerLens HookedTransformer directly...")
    try:
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained(
            "google/gemma-3-1b-it",
            device="cuda",
            dtype=torch.bfloat16,
        )
        print("SUCCESS: HookedTransformer loaded!")
        logits = model("The capital of France is")
        print(f"Logits shape: {logits.shape}")
        print("\n=== HW2 PASSED via Approach C (HookedTransformer) ===")
        BRIDGE_METHOD = "HookedTransformer"

    except Exception as e:
        print(f"FAILED: {e}")
        model = None

if model is None:
    print("\n" + "=" * 60)
    print("ALL BRIDGE APPROACHES FAILED.")
    print("This is expected -- Gemma 3 support in TransformerLens is incomplete.")
    print("We will use the FALLBACK PATH: HuggingFace transformers + manual hooks.")
    print("Proceed to: python3 hw2b_fallback.py")
    print("=" * 60)
else:
    print(f"\nBridge method that worked: {BRIDGE_METHOD}")
    print("Proceed to: python3 hw3_load_sae.py")

    # Cleanup
    del model
    torch.cuda.empty_cache()
