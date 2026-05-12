"""nlaB1_extract_L32.py -- extract Gemma 3 12B IT residual-stream activations
at layer 32 on the paper's 60 NL+NF prompts, at specific token positions, for
later injection into the released NLA AV (kitft/nla-gemma3-12b-L32-av).

Token positions extracted per case:
  - NL  "content":  last clinical-content token ("?" of "follow up?",
                    immediately before "Reply with exactly one letter only.")
  - NL  "pregen":   last user-message token (just before <end_of_turn>),
                    i.e. the token whose L32 hidden state drives the model's
                    first generated letter. This is AFTER the answer-key
                    scaffold block has been read.
  - NL  "letter_A/B/C/D": the four answer-key letter tokens A, B, C, D
                          in the "\\nA = ... / \\nB = ..." block
  - NF  "content":  last user-message content token (= last token before
                    <end_of_turn>; NF prompts have no scaffold so this is
                    the same char position as NL "content"). Under causal
                    attention NL "content" and NF "content" should have
                    byte-identical L32 activations — we still extract both
                    for a sanity check.

Output: results/nlaB_L32_activations.parquet  (and a .json index of which
record id corresponds to which (case_id, format, kind) tuple).

Compute: one forward pass per prompt, hidden_states=True, captures L32.
60 cases x {NL, NF} = 120 forward passes. ~20-30 min on H100-80GB at bf16.

Run from project root.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import torch
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Constants ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "google/gemma-3-12b-it"
LAYER = 32
D_MODEL = 3840

FORCED_LETTER_PATH = ROOT / (
    "nature_triage_expanded_replication/paper_faithful_forced_letter/data/"
    "canonical_forced_letter_vignettes.json"
)
SINGLETURN_PATH = ROOT / (
    "nature_triage_expanded_replication/paper_faithful_replication/data/"
    "canonical_singleturn_vignettes.json"
)
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_PARQUET = RESULTS_DIR / "nlaB_L32_activations.parquet"
OUT_INDEX = RESULTS_DIR / "nlaB_L32_activations_index.json"

ANCHOR = "Reply with exactly one letter only."   # marks start of answer-key block in NL
LETTER_PATTERNS = [
    ("letter_A", r"\nA = "),
    ("letter_B", r"\nB = "),
    ("letter_C", r"\nC = "),
    ("letter_D", r"\nD = "),
]


# ─── Helpers ─────────────────────────────────────────────────────────────
def char_offset_to_token_index(tok, content, char_offset):
    """Tokenize raw `content` with offsets, return the token index whose
    char range contains `char_offset`. Returns the FIRST token whose
    end > char_offset.
    """
    enc = tok(content, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    for i, (s, e) in enumerate(offsets):
        if s <= char_offset < e:
            return i
    # If exact match wasn't found (e.g. char_offset == e), return the token ending there
    for i, (s, e) in enumerate(offsets):
        if e == char_offset:
            return i
    return None


def chat_prefix_token_count(tok):
    """How many tokens does the chat-template prefix add before the user
    message content tokens? For Gemma 3 IT this is the number of tokens in
    "<bos><start_of_turn>user\\n".
    """
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "X"}],
        tokenize=False, add_generation_prompt=True,
    )
    # Find the "X" in the rendered string and count tokens before it
    x_pos = rendered.index("X")
    prefix_text = rendered[:x_pos]
    # Tokenize prefix as a continuation (no double-BOS):
    # apply chat template gave us BOS already baked in; we encode the
    # rendered prefix verbatim.
    prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
    return len(prefix_ids)


def build_cases():
    fl = json.loads(FORCED_LETTER_PATH.read_text())
    st = json.loads(SINGLETURN_PATH.read_text())
    fl_by_id = {v["id"]: v for v in fl}
    st_by_id = {v["id"]: v for v in st}

    def _key(s):
        m = re.match(r"^(\D+)(\d+)$", s)
        return (m.group(1), int(m.group(2))) if m else (s, 0)

    cases = []
    for cid in sorted(fl_by_id, key=_key):
        cases.append({
            "id": cid,
            "title": fl_by_id[cid]["title"],
            "NL_prompt": fl_by_id[cid]["natural_forced_letter"],
            "NF_prompt": st_by_id[cid]["patient_realistic"],
        })
    return cases


def render_chat_ids(tok, prompt):
    """Apply chat template + generation prompt; return flat list of ints
    (works on both transformers v4 and v5)."""
    out = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True,
    )
    if hasattr(out, "keys") and "input_ids" in out:
        return list(out["input_ids"])
    return list(out)


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    print(f"[B1] loading {MODEL_ID} on cuda (bf16) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    # Sanity-check hidden_size and layer count
    text_cfg = getattr(model.config, "text_config", model.config)
    assert text_cfg.hidden_size == D_MODEL, (
        f"base hidden_size={text_cfg.hidden_size} != expected {D_MODEL}"
    )
    print(f"  hidden_size={text_cfg.hidden_size}, num_hidden_layers={text_cfg.num_hidden_layers}")
    assert LAYER <= text_cfg.num_hidden_layers, f"LAYER={LAYER} > num_hidden_layers"

    prefix_n = chat_prefix_token_count(tok)
    print(f"  chat-template prefix token count = {prefix_n}")

    cases = build_cases()
    print(f"  loaded {len(cases)} cases")
    assert len(cases) == 60

    rows = []      # records for the parquet
    index_rows = []  # JSON-friendly metadata per row (no activation_vector)
    t0 = time.time()

    for i, c in enumerate(cases):
        for fmt in ("NL", "NF"):
            prompt = c[f"{fmt}_prompt"]

            # 1) Render chat template -> full input_ids
            ids_list = render_chat_ids(tok, prompt)
            ids = torch.tensor(ids_list, dtype=torch.long).unsqueeze(0).to(model.device)

            # 2) Forward pass with hidden states
            with torch.no_grad():
                out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            # hidden_states is a tuple of (num_layers+1) tensors; index 0 = embeddings
            h_L = out.hidden_states[LAYER][0]   # [seq_len, d_model]
            assert h_L.shape[-1] == D_MODEL

            # 3) Identify token positions
            positions_to_extract = []  # list of (kind, content_char_offset)

            if fmt == "NL":
                # 1) "content" -- token just before "Reply with exactly one letter only."
                anchor_pos = prompt.index(ANCHOR)
                last_content_char = anchor_pos - 1
                while last_content_char > 0 and prompt[last_content_char] in (" ", "\n", "\t"):
                    last_content_char -= 1
                positions_to_extract.append(("content", last_content_char))

                # 2) "pregen" -- last user-message token (the one whose hidden
                #     state drives generation of the first model token)
                pregen_char = len(prompt) - 1
                while pregen_char > 0 and prompt[pregen_char] in (" ", "\n", "\t"):
                    pregen_char -= 1
                positions_to_extract.append(("pregen", pregen_char))

                # 3) Letter tokens A/B/C/D
                for kind, pat in LETTER_PATTERNS:
                    m = re.search(pat, prompt)
                    assert m is not None, f"case {c['id']}: no {pat!r} in NL prompt"
                    letter_char_offset = m.start() + 1  # +1 to skip the "\n"
                    positions_to_extract.append((kind, letter_char_offset))

            else:  # NF
                # "content" -- last token before <end_of_turn>; same char
                # content as NL "content" but extracted from the NF forward
                # pass for the byte-identity sanity check.
                last_content_char = len(prompt) - 1
                while last_content_char > 0 and prompt[last_content_char] in (" ", "\n", "\t"):
                    last_content_char -= 1
                positions_to_extract.append(("content", last_content_char))

            # 4) Map char offsets -> chat-template token positions
            for kind, char_offset in positions_to_extract:
                content_tok_idx = char_offset_to_token_index(tok, prompt, char_offset)
                assert content_tok_idx is not None, (
                    f"case {c['id']}/{fmt}/{kind}: could not map char_offset={char_offset}"
                )
                chat_tok_idx = prefix_n + content_tok_idx
                assert chat_tok_idx < len(ids_list), (
                    f"case {c['id']}/{fmt}/{kind}: chat_tok_idx={chat_tok_idx} >= seq_len={len(ids_list)}"
                )
                token_id = ids_list[chat_tok_idx]
                token_str = tok.decode([token_id], skip_special_tokens=False)
                # Pull the activation vector
                act = h_L[chat_tok_idx].float().cpu().numpy()
                assert act.shape == (D_MODEL,)

                rid = f"{c['id']}_{fmt}_{kind}"
                rows.append({
                    "record_id": rid,
                    "case_id": c["id"],
                    "format": fmt,
                    "kind": kind,
                    "token_id": int(token_id),
                    "token_str": token_str,
                    "chat_tok_idx": int(chat_tok_idx),
                    "seq_len": len(ids_list),
                    "char_offset_in_content": int(char_offset),
                    "activation_vector": act.tolist(),
                })
                index_rows.append({
                    k: rows[-1][k] for k in
                    ("record_id", "case_id", "format", "kind",
                     "token_id", "token_str", "chat_tok_idx", "seq_len")
                })

            del out, h_L

        if (i + 1) % 10 == 0 or (i + 1) == len(cases):
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(cases)}] {elapsed/60:.1f} min elapsed; "
                  f"{len(rows)} activation records so far")

    # ─── Save ────────────────────────────────────────────────────────────
    print(f"\n[B1] writing parquet: {OUT_PARQUET}")
    table = pa.table({
        "record_id":      [r["record_id"]      for r in rows],
        "case_id":        [r["case_id"]        for r in rows],
        "format":         [r["format"]         for r in rows],
        "kind":           [r["kind"]           for r in rows],
        "token_id":       [r["token_id"]       for r in rows],
        "token_str":      [r["token_str"]      for r in rows],
        "chat_tok_idx":   [r["chat_tok_idx"]   for r in rows],
        "seq_len":        [r["seq_len"]        for r in rows],
        "char_offset_in_content":
                          [r["char_offset_in_content"] for r in rows],
        # parquet expects list[float] for variable-length floats column
        "activation_vector":
                          [r["activation_vector"] for r in rows],
    })
    pq.write_table(table, OUT_PARQUET)

    OUT_INDEX.write_text(json.dumps({
        "model_id": MODEL_ID,
        "layer": LAYER,
        "d_model": D_MODEL,
        "n_records": len(rows),
        "n_cases": len(cases),
        "records": index_rows,
    }, indent=2))
    print(f"[B1] wrote {len(rows)} records ({len(cases)} cases) to:")
    print(f"     {OUT_PARQUET}")
    print(f"     {OUT_INDEX}")
    print(f"     elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
