"""
Hello World 3: Load a Gemma Scope 2 SAE and inspect it.
Discovers the correct SAE ID naming convention at runtime.
"""
import torch

print("=== HW3: Loading Gemma Scope 2 SAE ===")

from sae_lens import SAE

# Step 1: Discover available SAE IDs
# We try to load with an invalid ID to trigger an informative error
print("--- Step 1: Discovering available SAE IDs ---")

release_names = [
    "gemma-scope-2-1b-it-resid_post",
    "gemma-scope-2-1b-it-resid-post",
    "gemma-scope-2-1b-it-res-post",
]

working_release = None
for release in release_names:
    print(f"\nTrying release: {release}")
    try:
        SAE.from_pretrained(release=release, sae_id="INVALID_DISCOVER")
    except Exception as e:
        error_msg = str(e)
        if "INVALID_DISCOVER" in error_msg or "not found" in error_msg.lower():
            print(f"  Release '{release}' exists! Error about SAE ID (expected).")
            # Try to extract available IDs from error message
            if "available" in error_msg.lower() or "valid" in error_msg.lower():
                print(f"  Available IDs hint: {error_msg[:500]}")
            working_release = release
            break
        else:
            print(f"  Release not found: {error_msg[:200]}")

if working_release is None:
    # Try listing from pretrained_saes
    print("\nTrying to list all available releases...")
    try:
        from sae_lens import pretrained_saes
        if hasattr(pretrained_saes, 'get_pretrained_saes_directory'):
            directory = pretrained_saes.get_pretrained_saes_directory()
            gemma_releases = [k for k in directory.keys() if 'gemma' in k.lower() and '1b' in k.lower()]
            print(f"Found Gemma 1B releases: {gemma_releases}")
            if gemma_releases:
                working_release = gemma_releases[0]
        elif hasattr(pretrained_saes, 'PRETRAINED_SAES'):
            gemma_keys = [k for k in pretrained_saes.PRETRAINED_SAES.keys()
                         if 'gemma' in k.lower() and '1b' in k.lower()]
            print(f"Found Gemma 1B keys: {gemma_keys}")
    except Exception as e:
        print(f"Could not list releases: {e}")

    # Direct approach: list what SAELens knows about
    try:
        from sae_lens.toolkit.pretrained_saes import get_pretrained_saes_directory
        directory = get_pretrained_saes_directory()
        gemma_releases = {k: v for k, v in directory.items()
                         if 'gemma' in k.lower() and ('1b' in k.lower() or 'scope-2' in k.lower())}
        print(f"\nAll Gemma/1B releases in SAELens directory:")
        for k in sorted(gemma_releases.keys()):
            print(f"  {k}")
        if gemma_releases:
            # Pick the resid_post one
            resid_keys = [k for k in gemma_releases.keys() if 'resid' in k.lower()]
            working_release = resid_keys[0] if resid_keys else list(gemma_releases.keys())[0]
    except Exception as e:
        print(f"Could not get directory: {e}")

if working_release is None:
    print("\nERROR: Could not find any Gemma Scope 2 1B IT release in SAELens.")
    print("The SAE might need to be loaded manually from HuggingFace.")
    print("Try: pip install --upgrade sae-lens")
    import sys
    sys.exit(1)

print(f"\nUsing release: {working_release}")

# Step 2: Discover available SAE IDs within the release
print("\n--- Step 2: Finding available SAE IDs ---")

# Try common naming patterns
candidate_ids = [
    # Pattern: layer_N_width_Xk_l0_Y
    "layer_12_width_16k_l0_medium",
    "layer_12_width_16k_l0_small",
    "layer_12_width_16384_l0_medium",
    # Pattern: layer_N/width_Xk/l0_Y
    "layer_12/width_16k/l0_medium",
    # Pattern with actual L0 values
    "layer_12_width_16k_l0_30",
    "layer_12_width_16k_average_l0_30",
]

working_sae = None
for sae_id in candidate_ids:
    try:
        print(f"  Trying: {sae_id}...")
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=working_release,
            sae_id=sae_id,
            device="cuda",
        )
        print(f"  SUCCESS with sae_id='{sae_id}'!")
        working_sae = sae_id
        break
    except Exception as e:
        error_msg = str(e)
        # Check if error reveals valid IDs
        if any(hint in error_msg.lower() for hint in ['available', 'valid', 'choose', 'options']):
            print(f"  Hint from error: {error_msg[:300]}")
        else:
            print(f"  Failed: {error_msg[:150]}")

if working_sae is None:
    print("\nCould not find working SAE ID. Let's try to list the repo contents...")
    try:
        from huggingface_hub import list_repo_tree
        files = list(list_repo_tree("google/gemma-scope-2-1b-it", path_in_repo="resid_post"))
        print("Files in resid_post/:")
        for f in files[:20]:
            print(f"  {f}")
    except Exception as e:
        print(f"Could not list repo: {e}")

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        siblings = api.model_info("google/gemma-scope-2-1b-it").siblings
        resid_files = [s.rfilename for s in siblings if 'resid_post' in s.rfilename]
        print(f"\nResid post files (first 20):")
        for f in resid_files[:20]:
            print(f"  {f}")
    except Exception as e:
        print(f"Could not query HF API: {e}")

    import sys
    sys.exit(1)

# Step 3: Inspect the loaded SAE
print(f"\n--- Step 3: SAE Inspection ---")
print(f"SAE config:")
print(f"  d_in (model hidden size): {sae.cfg.d_in}")
print(f"  d_sae (dictionary size): {sae.cfg.d_sae}")
print(f"  hook_name: {sae.cfg.hook_name}")
print(f"  dtype: {sae.dtype}")

# Step 4: Try loading SAEs at different layers to find what's available
print(f"\n--- Step 4: Scanning available layers ---")
# Extract the naming pattern from the working ID
# Try layers across the network
available_layers = []
for layer_num in range(0, 26):
    test_id = working_sae.replace(
        f"layer_{working_sae.split('layer_')[1].split('_')[0]}",
        f"layer_{layer_num}"
    )
    try:
        test_sae, _, _ = SAE.from_pretrained(
            release=working_release,
            sae_id=test_id,
            device="cpu",  # Use CPU to save VRAM
        )
        available_layers.append(layer_num)
        del test_sae
    except:
        pass

print(f"Available layers: {available_layers}")
print(f"Total layers in model: 26 (for Gemma 3 1B)")

# Recommend layers for v1
if available_layers:
    n = len(available_layers)
    suggested = [
        available_layers[n // 4],       # ~25% depth (early)
        available_layers[n // 2],       # ~50% depth (middle)
        available_layers[3 * n // 4],   # ~75% depth (late)
    ]
    print(f"Suggested layers for v1: {suggested}")

print(f"\n=== HW3 PASSED ===")
print(f"Working release: {working_release}")
print(f"Working sae_id pattern: {working_sae}")
print(f"Available layers: {available_layers}")
print(f"\nProceed to: python3 hw4_end_to_end.py")

del sae
torch.cuda.empty_cache()
