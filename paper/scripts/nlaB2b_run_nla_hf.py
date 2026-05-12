"""nlaB2b_run_nla_hf.py -- NLA AV inference using HuggingFace transformers
directly (bypassing SGLang).

Rationale: SGLang 0.4/0.5 has a transformers/huggingface_hub/kernels
dependency-resolution conflict on this stack. We don't actually need
SGLang's continuous batching for 420 sequential one-shot calls --
that's ~7 minutes at 1 s/call, fully acceptable.

This script replicates the EXACT injection math from nla_inference.py
(NLAClient):
  1. load AV checkpoint + sidecar
  2. tokenize the canonical actor prompt
  3. embed via the model's input embedding (×√d for Gemma)
  4. L2-rescale the activation to injection_scale=80,000
  5. overwrite the marker-token embedding
  6. generate from inputs_embeds; decode <explanation>...</explanation>

Run on the remote A100 (model fits in bf16 at ~24 GB):

    python3 paper/scripts/nlaB2b_run_nla_hf.py \\
        --checkpoint kitft/nla-gemma3-12b-L32-av \\
        --activations results/nlaB_L32_activations.parquet \\
        --out results/nlaB_descriptions.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


def load_sidecar(checkpoint, tokenizer):
    """Return (cfg dict, prompt template, injection token id, neighbors)
    by reading nla_meta.yaml from a local checkpoint dir OR HF repo id."""
    from huggingface_hub import hf_hub_download
    meta_path = hf_hub_download(repo_id=checkpoint, filename="nla_meta.yaml")
    meta = yaml.safe_load(Path(meta_path).read_text())
    inj_scale = float(meta["extraction"]["injection_scale"])
    tokens = meta["tokens"]
    tmpl = meta["prompt_templates"].get("av") or meta["prompt_templates"]["actor"]
    d_model = meta["d_model"]
    # Cross-check with tokenizer
    live_inj = tokenizer.encode(tokens["injection_char"], add_special_tokens=False)
    assert live_inj == [tokens["injection_token_id"]], (
        f"tokenizer drift: {tokens['injection_char']!r} -> {live_inj}, "
        f"sidecar expects [{tokens['injection_token_id']}]"
    )
    return {
        "d_model": d_model,
        "injection_scale": inj_scale,
        "injection_char": tokens["injection_char"],
        "injection_token_id": tokens["injection_token_id"],
        "left_neighbor_id": tokens["injection_left_neighbor_id"],
        "right_neighbor_id": tokens["injection_right_neighbor_id"],
        "actor_prompt_template": tmpl,
    }


def normalize(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    """L2-rescale to target_scale (norm in fp32 for numerics)."""
    n = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (n / target_scale).to(v.dtype)


def render_chat_ids(tok, content):
    """Return flat list[int] for the actor prompt template."""
    out = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
    )
    if hasattr(out, "keys") and "input_ids" in out:
        return list(out["input_ids"])
    return list(out)


def find_injection_idx(ids, inj_id, left_id, right_id):
    """Return the unique position p where ids[p]==inj_id, ids[p-1]==left_id,
    ids[p+1]==right_id."""
    matches = []
    for i in range(1, len(ids) - 1):
        if ids[i] == inj_id and ids[i-1] == left_id and ids[i+1] == right_id:
            matches.append(i)
    assert len(matches) == 1, f"expected 1 injection site, got {len(matches)}"
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="kitft/nla-gemma3-12b-L32-av")
    parser.add_argument("--activations",
                        default="results/nlaB_L32_activations.parquet")
    parser.add_argument("--out", default="results/nlaB_descriptions.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    print(f"[B2b] loading AV checkpoint {args.checkpoint!r} on {args.device} (bf16)")
    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    cfg = load_sidecar(args.checkpoint, tok)
    print(f"  d_model={cfg['d_model']}, inj_scale={cfg['injection_scale']}, "
          f"inj_char={cfg['injection_char']!r} (tok {cfg['injection_token_id']})")

    # Determine embed-scale for Gemma (×√d post-lookup). We use the model's
    # OWN embed layer in forward(inputs_embeds=...) so the scale is applied
    # internally — BUT we need to know it for the injection vector.
    text_cfg = getattr(model.config, "text_config", model.config)
    is_gemma = "gemma" in (text_cfg.model_type or "").lower()
    embed_scale = math.sqrt(text_cfg.hidden_size) if is_gemma else 1.0
    print(f"  embed_scale={embed_scale:.4f} (gemma={is_gemma})")

    # Strategy: use the model's get_input_embeddings() to compute embeds,
    # which for Gemma 3 applies the √d scale internally. Then replace the
    # marker-token embed with the (already-scaled) injection vector. The
    # injection vector lives in the SAME magnitude space as scaled embeds
    # (the residual stream norm of Gemma 3 L32 is ~60k, comparable to
    # √d * embed_table_value).
    embed_layer = model.get_input_embeddings()

    # Build the canonical prompt template ids (constant across all calls)
    content = cfg["actor_prompt_template"].format(injection_char=cfg["injection_char"])
    base_ids = render_chat_ids(tok, content)
    base_ids_t = torch.tensor(base_ids, dtype=torch.long,
                              device=args.device).unsqueeze(0)
    inj_idx = find_injection_idx(
        base_ids, cfg["injection_token_id"],
        cfg["left_neighbor_id"], cfg["right_neighbor_id"],
    )
    print(f"  canonical prompt has {len(base_ids)} tokens; injection slot @ idx {inj_idx}")

    # Pre-compute the base embeds (with √d already applied for Gemma)
    with torch.no_grad():
        base_embeds = embed_layer(base_ids_t)  # [1, T, d]
    print(f"  base_embeds dtype={base_embeds.dtype} shape={tuple(base_embeds.shape)}")

    # ─── Load activations ────────────────────────────────────────────────
    print(f"[B2b] reading activations from {args.activations}")
    table = pq.read_table(args.activations)
    n_records = len(table)
    if args.limit:
        n_records = min(n_records, args.limit)
        print(f"  limited to first {n_records} records")
    print(f"  {n_records} records to process")

    results = []
    t0 = time.time()

    for i in range(n_records):
        row = {k: table.column(k)[i].as_py() for k in table.column_names}
        v_raw = torch.tensor(row["activation_vector"], dtype=torch.float32,
                             device=args.device).unsqueeze(0)  # [1, d]
        v_scaled = normalize(v_raw, cfg["injection_scale"]).to(base_embeds.dtype)

        embeds = base_embeds.clone()
        embeds[0, inj_idx] = v_scaled[0]

        with torch.no_grad():
            gen_kwargs = dict(
                inputs_embeds=embeds,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                pad_token_id=tok.eos_token_id,
            )
            if args.temperature > 0:
                gen_kwargs["temperature"] = args.temperature
            out = model.generate(**gen_kwargs)
        # `out` contains ONLY generated tokens (since we used inputs_embeds,
        # not input_ids -- generate's prefix is empty by default).
        text = tok.decode(out[0], skip_special_tokens=True)
        m = EXPLANATION_RE.search(text)
        explanation = m.group(1) if m else text.strip()

        results.append({
            "record_id": row["record_id"],
            "case_id":   row["case_id"],
            "format":    row["format"],
            "kind":      row["kind"],
            "token_id":  row["token_id"],
            "token_str": row["token_str"],
            "chat_tok_idx": row["chat_tok_idx"],
            "raw_text":  text,
            "samples":   [explanation],
        })

        if (i + 1) % 30 == 0 or (i + 1) == n_records:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_records - i - 1) / rate if rate > 0 else float("inf")
            print(f"  [{i+1}/{n_records}] {elapsed/60:.1f} min, "
                  f"{rate:.2f} rec/s, ETA {eta/60:.1f} min")

    print(f"\n[B2b] writing to {args.out}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "transport": "hf_transformers_direct",
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "n_records": len(results),
        "results": results,
    }, indent=2))
    print(f"[B2b] DONE. {len(results)} records, "
          f"elapsed {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
