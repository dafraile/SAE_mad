"""
Hello World 1: Load Gemma 3 1B IT and run a simple generation.
Confirms the model works on this hardware.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=== HW1: Loading Gemma 3 1B IT ===")
print(f"VRAM free before load: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB")

model_id = "google/gemma-3-1b-it"
print(f"Loading {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"VRAM used: {(torch.cuda.mem_get_info(0)[1] - torch.cuda.mem_get_info(0)[0]) / 1e9:.1f} GB")

# Print model architecture overview
print(f"\nModel architecture:")
print(f"  Layers: {model.config.num_hidden_layers}")
print(f"  Hidden size: {model.config.hidden_size}")
print(f"  Vocab size: {model.config.vocab_size}")

# Simple generation test
print("\n--- Generation test ---")
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Output: {result}")

# Multilingual quick check
print("\n--- Multilingual check ---")
for lang, prompt in [
    ("ES", "La capital de Francia es"),
    ("FR", "La capitale de la France est"),
]:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[{lang}] {result}")

# Show tokenization differences (important for later analysis)
print("\n--- Tokenization comparison ---")
texts = {
    "EN": "The Eiffel Tower is located in Paris, France.",
    "ES": "La Torre Eiffel se encuentra en Paris, Francia.",
    "FR": "La Tour Eiffel se trouve a Paris, en France.",
}
for lang, text in texts.items():
    tokens = tokenizer.encode(text)
    print(f"[{lang}] {len(tokens)} tokens: {tokenizer.convert_ids_to_tokens(tokens)}")

print("\n=== HW1 PASSED ===")
print(f"Key info for next steps:")
print(f"  num_hidden_layers = {model.config.num_hidden_layers}")
print(f"  hidden_size = {model.config.hidden_size}")
print(f"\nCleaning up GPU memory...")
del model, tokenizer
torch.cuda.empty_cache()
print(f"VRAM free after cleanup: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB")
