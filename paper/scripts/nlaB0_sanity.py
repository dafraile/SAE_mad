"""nlaB0_sanity.py -- pre-flight sanity check on NLA released checkpoint.

Pulls only the nla_meta.yaml sidecar from kitft/nla-gemma3-12b-L32-av and
asserts the alignment constants we expect to see, BEFORE burning any GPU time.

Catches the two most-common silent failures from docs/inference.md:
  - tokenizer drift on the injection character
  - prompt-template drift breaking the neighbor-id check

Run locally:
    python3 paper/scripts/nlaB0_sanity.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write("pip install pyyaml\n"); sys.exit(2)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.stderr.write("pip install huggingface_hub\n"); sys.exit(2)

try:
    from transformers import AutoTokenizer, AutoConfig
except ImportError:
    sys.stderr.write("pip install transformers\n"); sys.exit(2)

NLA_REPO = "kitft/nla-gemma3-12b-L32-av"
BASE_REPO = "google/gemma-3-12b-it"
EXPECT_D_MODEL = 3840
EXPECT_INJ_SCALE = 80000.0  # docs say 80k for Gemma-3
EXPECT_LAYER = 32

print(f"[B0] sanity check for NLA repo {NLA_REPO!r}")
print(f"     against base model       {BASE_REPO!r}")
print(f"     expected d_model={EXPECT_D_MODEL}, injection_scale={EXPECT_INJ_SCALE}, layer={EXPECT_LAYER}")
print()

# -------- 1. Pull just the sidecar yaml ---------------------------------
print("[B0/1] downloading nla_meta.yaml ...")
sidecar_path = hf_hub_download(
    repo_id=NLA_REPO, filename="nla_meta.yaml", repo_type="model",
)
meta = yaml.safe_load(Path(sidecar_path).read_text())
print(f"       OK -> {sidecar_path}")

# -------- 2. Assert structure --------------------------------------------
print("\n[B0/2] sidecar contents (key fields):")
d_model = meta.get("d_model") or meta.get("extraction", {}).get("d_model")
inj_scale = meta.get("extraction", {}).get("injection_scale")
# In schema_version=2 the layer is top-level; older sidecars use extraction.layer
layer = meta.get("extraction_layer_index") or meta.get("extraction", {}).get("layer")
tokens = meta["tokens"]
tmpl = (meta["prompt_templates"].get("av")
        or meta["prompt_templates"]["actor"])

print(f"  d_model              = {d_model}")
print(f"  extraction.layer     = {layer}")
print(f"  extraction.injection_scale = {inj_scale}")
print(f"  tokens.injection_char      = {tokens['injection_char']!r}")
print(f"  tokens.injection_token_id  = {tokens['injection_token_id']}")
print(f"  tokens.left_neighbor_id    = {tokens['injection_left_neighbor_id']}")
print(f"  tokens.right_neighbor_id   = {tokens['injection_right_neighbor_id']}")
print(f"  actor_prompt_template (first 200 chars):")
print(f"    {tmpl[:200]!r}")

assert d_model == EXPECT_D_MODEL, f"d_model mismatch: sidecar={d_model} vs expected {EXPECT_D_MODEL}"
assert int(inj_scale) == int(EXPECT_INJ_SCALE), f"injection_scale mismatch: sidecar={inj_scale}"
assert int(layer) == EXPECT_LAYER, f"layer mismatch: sidecar={layer}"
print("\n[B0/2] OK: d_model, injection_scale, layer all match expectation.")

# -------- 3a. Live tokenizer cross-check on the AV tokenizer ------------
# NLAClient uses the AV checkpoint's own tokenizer. The AV is fine-tuned
# from Gemma 3 12B IT, so the tokenizer is functionally the same vocab,
# but its chat template may differ. We use the AV tokenizer because that
# is what NLAClient will use at inference time.
print("\n[B0/3a] cross-check with AV tokenizer (kitft/nla-gemma3-12b-L32-av)...")
av_tok = AutoTokenizer.from_pretrained(NLA_REPO, trust_remote_code=True)
inj_char = tokens["injection_char"]
live_inj = av_tok.encode(inj_char, add_special_tokens=False)
assert live_inj == [tokens["injection_token_id"]], (
    f"tokenizer drift: encode({inj_char!r}) = {live_inj}, sidecar expects "
    f"[{tokens['injection_token_id']}]. Wrong tokenizer or vocab changed."
)
print(f"  OK: AV tokenizer encode({inj_char!r}) -> [{tokens['injection_token_id']}]")

content = tmpl.format(injection_char=inj_char)
out = av_tok.apply_chat_template(
    [{"role": "user", "content": content}],
    tokenize=True, add_generation_prompt=True,
)
# Transformers v5 returns BatchEncoding with input_ids; v4 returns flat list.
ids = out["input_ids"] if hasattr(out, "keys") and "input_ids" in out else out
matches = [i for i, t in enumerate(ids) if t == tokens["injection_token_id"]]
assert len(matches) == 1, (
    f"expected 1 injection site in canonical prompt, got {len(matches)}. "
    f"prompt template / chat template drift."
)
p = matches[0]
assert ids[p - 1] == tokens["injection_left_neighbor_id"], (
    f"left neighbor drift at pos {p}: live={ids[p-1]} vs sidecar={tokens['injection_left_neighbor_id']}"
)
assert ids[p + 1] == tokens["injection_right_neighbor_id"], (
    f"right neighbor drift at pos {p}: live={ids[p+1]} vs sidecar={tokens['injection_right_neighbor_id']}"
)
print(f"  OK: injection site at token {p}/{len(ids)}; left/right neighbors match sidecar.")

# -------- 3b. Confirm base-model hidden_size matches --------------------
print("\n[B0/3b] cross-check base model dimensions (google/gemma-3-12b-it)...")
if not os.environ.get("HF_TOKEN") and not (Path.home() / ".cache" / "huggingface" / "token").exists():
    print("  WARN: no HF_TOKEN env var or cached token; Gemma base is gated. Skipping.")
    print("        Expected hidden_size=3840 confirmed via NLA sidecar instead.")
else:
    cfg = AutoConfig.from_pretrained(BASE_REPO, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", cfg)
    base_d = text_cfg.hidden_size
    assert base_d == EXPECT_D_MODEL, f"base model hidden_size={base_d} != expected {EXPECT_D_MODEL}"
    sqrt_d = math.sqrt(base_d)
    print(f"  OK: base hidden_size={base_d}; √d = {sqrt_d:.4f} (Gemma post-lookup embed scale).")

print("\n[B0] ALL CHECKS PASSED.")
print(f"  - Use injection_scale={EXPECT_INJ_SCALE} when calling NLAClient.")
print(f"  - Extract residual stream from layer {EXPECT_LAYER}.")
print(f"  - Gemma needs embed_scale=√d_model={math.sqrt(EXPECT_D_MODEL):.4f}; NLAClient handles this.")
