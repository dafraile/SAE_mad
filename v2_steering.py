"""
v2 Steering Experiment: Do SAE features behave as causal control signals?

The simplest possible test of the routing hypothesis precondition:
Take Feature 857 (Spanish detector, 100% precision, 6% noise at layer 22),
clamp it high during generation on neutral English prompts, and observe
whether the output shifts toward Spanish.

If it does: features are causal control signals, green light for routing.
If it doesn't: the routing hypothesis is built on sand at this scale.

Usage:
    python3 v2_steering.py
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from contextlib import contextmanager

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_ID = "google/gemma-3-1b-it"

# Layer 22 SAE -- where language features are strongest
SAE_RELEASE = "gemma-scope-2-1b-it-res"
SAE_ID = "layer_22_width_16k_l0_medium"
STEER_LAYER = 22

# Features to test (from v1 analysis 2)
FEATURES = {
    "spanish_857": {
        "idx": 857,
        "desc": "Spanish detector (100% ES, 6% other, mean_act=383)",
        "expected": "Output should shift toward Spanish",
    },
    "french_1207": {
        "idx": 1207,
        "desc": "French detector (100% FR, 14% other, mean_act=1782)",
        "expected": "Output should shift toward French",
    },
    "french_3201": {
        "idx": 3201,
        "desc": "French detector (100% FR, 6% other, mean_act=192)",
        "expected": "Output should shift toward French",
    },
}

# Neutral prompts that have no language bias
PROMPTS = [
    "The weather today is",
    "Water is made of",
    "The capital of Japan is",
    "Birds can fly because",
    "The number seven is",
    "Mountains are formed when",
    "A good breakfast includes",
    "The moon appears bright because",
]

# Steering strengths to test (multiples of the feature's mean activation)
STEER_MULTIPLIERS = [0.5, 1.0, 2.0, 5.0]

MAX_NEW_TOKENS = 50
# ============================================================


def load_model_and_sae():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16)

    print("Loading SAE...")
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    print(f"  Layer {STEER_LAYER}: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    return model, tokenizer, sae


@contextmanager
def steer_feature(model, sae, layer, feature_idx, steer_value):
    """Context manager that hooks into a layer and steers a specific SAE feature.

    During each forward pass:
    1. Get the residual stream at the target layer
    2. Encode with SAE to get sparse features
    3. Add steer_value to the target feature
    4. Decode back to residual space
    5. Replace the original residual with the steered version
    """
    def hook_fn(module, input, output):
        # output is (hidden_states, ...) tuple for Gemma 3
        hidden = output[0] if isinstance(output, tuple) else output

        # Only steer the last token position (during generation)
        # For the prefill pass, steer all positions
        resid = hidden.clone()

        # Encode -> modify -> decode
        resid_for_sae = resid.to(sae.dtype)
        features = sae.encode(resid_for_sae)

        # Add the steering value to the target feature
        features[:, :, feature_idx] = features[:, :, feature_idx] + steer_value

        # Decode back
        steered_resid = sae.decode(features)

        # Compute the steering vector (difference between steered and original reconstruction)
        original_recon = sae.decode(sae.encode(resid_for_sae))
        steering_delta = (steered_resid - original_recon).to(hidden.dtype)

        # Add the delta to the original residual (preserves info not captured by SAE)
        result = hidden + steering_delta

        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result

    hook = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        hook.remove()


def generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            temperature=1.0,
        )
    # Only return the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def measure_feature_activation(model, tokenizer, sae, text, layer, feature_idx):
    """Measure how much a feature activates on given text."""
    captured = {}

    def hook_fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        captured["resid"] = out.detach()

    hook = model.model.layers[layer].register_forward_hook(hook_fn)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**inputs)
    hook.remove()

    resid = captured["resid"].to(sae.dtype)
    features = sae.encode(resid)
    feat_act = features[0, :, feature_idx]
    return feat_act.mean().item(), feat_act.max().item()


def run_steering_experiment(model, tokenizer, sae):
    """Main experiment: steer features and observe output changes."""

    print("\n" + "=" * 70)
    print("v2 STEERING EXPERIMENT: Do SAE features behave as causal controls?")
    print("=" * 70)

    for feat_name, feat_info in FEATURES.items():
        feat_idx = feat_info["idx"]
        print(f"\n{'=' * 70}")
        print(f"FEATURE: {feat_name} (index {feat_idx})")
        print(f"  {feat_info['desc']}")
        print(f"  {feat_info['expected']}")
        print(f"{'=' * 70}")

        # First, measure the feature's natural activation on a Spanish/French text
        # to calibrate steering strength
        if "spanish" in feat_name:
            ref_text = "La Torre Eiffel es una famosa estructura de hierro ubicada en París."
        else:
            ref_text = "La Tour Eiffel est une célèbre structure en fer située à Paris."

        mean_act, max_act = measure_feature_activation(
            model, tokenizer, sae, ref_text, STEER_LAYER, feat_idx)
        print(f"\n  Natural activation on reference text:")
        print(f"    mean={mean_act:.1f}, max={max_act:.1f}")

        for prompt in PROMPTS:
            print(f"\n  Prompt: \"{prompt}\"")

            # Baseline (no steering)
            baseline = generate(model, tokenizer, prompt)
            print(f"    [baseline]  {baseline[:120]}")

            # Steered at different strengths
            for mult in STEER_MULTIPLIERS:
                steer_val = mean_act * mult
                with steer_feature(model, sae, STEER_LAYER, feat_idx, steer_val):
                    steered = generate(model, tokenizer, prompt)
                print(f"    [steer {mult:4.1f}x] {steered[:120]}")

            # Also try negative steering (suppress the feature)
            with steer_feature(model, sae, STEER_LAYER, feat_idx, -mean_act):
                suppressed = generate(model, tokenizer, prompt)
            print(f"    [suppress]  {suppressed[:120]}")


def run_multi_feature_experiment(model, tokenizer, sae):
    """Bonus: steer Spanish AND suppress French simultaneously."""
    print("\n" + "=" * 70)
    print("MULTI-FEATURE: Steer Spanish(857) + Suppress French(1207)")
    print("=" * 70)

    sp_idx = FEATURES["spanish_857"]["idx"]
    fr_idx = FEATURES["french_1207"]["idx"]

    # Measure natural activations for calibration
    sp_mean, _ = measure_feature_activation(
        model, tokenizer, sae,
        "La Torre Eiffel es una famosa estructura de hierro.",
        STEER_LAYER, sp_idx)
    fr_mean, _ = measure_feature_activation(
        model, tokenizer, sae,
        "La Tour Eiffel est une célèbre structure en fer.",
        STEER_LAYER, fr_idx)

    def multi_hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        resid = hidden.clone()
        resid_for_sae = resid.to(sae.dtype)
        features = sae.encode(resid_for_sae)

        # Boost Spanish, suppress French
        features[:, :, sp_idx] = features[:, :, sp_idx] + sp_mean * 2.0
        features[:, :, fr_idx] = features[:, :, fr_idx] - fr_mean

        steered = sae.decode(features)
        original_recon = sae.decode(sae.encode(resid_for_sae))
        delta = (steered - original_recon).to(hidden.dtype)
        result = hidden + delta

        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result

    for prompt in PROMPTS[:4]:
        print(f"\n  Prompt: \"{prompt}\"")

        baseline = generate(model, tokenizer, prompt)
        print(f"    [baseline]     {baseline[:120]}")

        hook = model.model.layers[STEER_LAYER].register_forward_hook(multi_hook_fn)
        steered = generate(model, tokenizer, prompt)
        hook.remove()
        print(f"    [ES+, FR-]     {steered[:120]}")


def main():
    model, tokenizer, sae = load_model_and_sae()

    # Main experiment
    run_steering_experiment(model, tokenizer, sae)

    # Multi-feature experiment
    run_multi_feature_experiment(model, tokenizer, sae)

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
If steering works (output language shifts):
  -> Features are causal control signals, not just correlates
  -> The routing hypothesis precondition is MET
  -> Green light for v3: composition test with multiple features

If steering produces noise/gibberish:
  -> Feature activation is a readout, not a control
  -> The routing hypothesis needs rethinking
  -> May need to train better SAEs or use a different scale

If steering partially works (some Spanish words mixed in):
  -> Features have causal influence but aren't clean switches
  -> Routing would need to be softer (weighted mixing, not binary gating)
  -> Still informative for the brain analogy (biological routing is soft too)
""")

    del model, sae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
