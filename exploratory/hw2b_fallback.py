"""
Hello World 2B (Fallback): Extract activations using HuggingFace transformers + PyTorch hooks.
Run this if hw2_sae_bridge.py failed.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=== HW2B: Fallback -- Manual Activation Extraction ===")

model_id = "google/gemma-3-1b-it"
print(f"Loading {model_id} via HuggingFace transformers...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

# Inspect model structure to find the right hook points
print("\n--- Model layer structure ---")
print(f"Number of layers: {model.config.num_hidden_layers}")
print(f"Hidden size: {model.config.hidden_size}")

# Print the structure of one layer to understand hook points
layer = model.model.layers[0]
print(f"\nLayer 0 structure:")
for name, module in layer.named_children():
    print(f"  {name}: {type(module).__name__}")

# Test activation extraction at multiple layers
target_layers = [6, 12, 17, 22]  # early, middle, upper-middle, late
print(f"\n--- Extracting activations at layers {target_layers} ---")

activations = {}
hooks = []

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        # Gemma 3 layers return (hidden_states, ...) tuple
        out = output[0] if isinstance(output, tuple) else output
        activations[layer_idx] = out.detach()
    return hook_fn

for idx in target_layers:
    h = model.model.layers[idx].register_forward_hook(make_hook(idx))
    hooks.append(h)

# Run a forward pass
text = "The Eiffel Tower is located in Paris, France."
inputs = tokenizer(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs)

# Remove hooks
for h in hooks:
    h.remove()

# Check results
print(f"\nInput text: {text}")
print(f"Input tokens: {inputs['input_ids'].shape[1]}")
print(f"\nCaptured activations:")
for idx in target_layers:
    act = activations[idx]
    print(f"  Layer {idx}: shape={act.shape}, dtype={act.dtype}")
    print(f"    mean={act.float().mean().item():.4f}, std={act.float().std().item():.4f}")

# Verify the hidden size matches what Gemma Scope 2 SAEs expect
hidden_size = activations[target_layers[0]].shape[-1]
print(f"\nHidden size from activations: {hidden_size}")
print(f"Expected for Gemma 3 1B: 1152")
assert hidden_size == model.config.hidden_size, "Hidden size mismatch!"

# Test that activations are deterministic
print("\n--- Determinism check ---")
activations2 = {}
hooks2 = []
for idx in target_layers[:1]:  # Just check one layer
    h = model.model.layers[idx].register_forward_hook(make_hook(idx))
    hooks2.append(h)

activations_run1 = activations.copy()
activations = {}  # Reset
with torch.no_grad():
    outputs2 = model(**inputs)
for h in hooks2:
    h.remove()

diff = (activations_run1[target_layers[0]] - activations[target_layers[0]]).abs().max().item()
print(f"Max difference between two runs: {diff}")
assert diff == 0.0, f"Activations not deterministic! Max diff: {diff}"
print("Determinism check PASSED.")

# Helper function for reuse
print("\n--- Saving helper function ---")
helper_code = '''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_id="google/gemma-3-1b-it"):
    """Load Gemma 3 1B IT."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.bfloat16,
    )
    return model, tokenizer

def extract_activations(model, tokenizer, text, layer_indices):
    """Extract residual stream activations at specified layers.

    Returns dict mapping layer_idx -> tensor of shape [1, seq_len, hidden_size].
    """
    captured = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = out.detach()
        return hook_fn

    for idx in layer_indices:
        h = model.model.layers[idx].register_forward_hook(make_hook(idx))
        hooks.append(h)

    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return captured

def extract_activations_chat(model, tokenizer, text, layer_indices):
    """Same as extract_activations but wraps text in chat template first."""
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return extract_activations(model, tokenizer, chat_text, layer_indices)
'''

with open("/workspace/sae_mad/utils.py", "w") as f:
    f.write(helper_code)
print("Saved utils.py with helper functions.")

print("\n=== HW2B PASSED ===")
print("Fallback path works. Proceed to: python3 hw3_load_sae.py")

# Cleanup
del model, tokenizer
torch.cuda.empty_cache()
