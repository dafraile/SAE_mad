"""
Phase 2b — Dilution-controlled ActAdd projection.

Phase 2 found a systematic negative-sign alignment of every medical feature
with the (B - D) direction. The negative sign is the dilution-artifact
signature: B has ~50 extra forced-letter-instruction tokens that don't fire
medical features and therefore depress the mean. To check whether the
medical-feature alignment we saw in Phase 2 is real or an artifact, we
re-run the projection with two aggregation strategies that are immune to
the prompt-length asymmetry:

  (a) length-controlled mean-pool:
      truncate B's content range at the start of the forced-letter block
      (the literal "Reply with exactly one letter") so that B and D pool
      over identical clinical content (modulo paraphrase).

  (b) max-pool:
      take the max per-dimension residual across content tokens.
      Length-invariant by construction.

If medical-feature alignment ranks change substantially between Phase 2's
mean-pool and (a)/(b), the dilution explanation is confirmed and Phase 2's
medical-feature alignment is an artifact. If ranks stay similar, the
medical-feature alignment is real and needs explanation.

Output: results/phase2b_dilution_check.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
import safetensors.torch as sft
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-4b-it"
SAE_REPO = "google/gemma-scope-2-4b-it"
LAYERS = [17, 29]

MEDICAL_FEATURES = {
    9:  [139, 9909, 956],
    17: [9854, 368, 1539],
    22: [1181, 365, 8389],
    29: [12570, 893, 12845],
}

FORCED_LETTER_PATH = Path(
    "nature_triage_expanded_replication/paper_faithful_forced_letter/data/"
    "canonical_forced_letter_vignettes.json"
)
SINGLETURN_PATH = Path(
    "nature_triage_expanded_replication/paper_faithful_replication/data/"
    "canonical_singleturn_vignettes.json"
)
PHASE_0_5_PATH = Path("results/phase0_5_three_cells.json")
ADJUDICATED_PATH = Path("results/phase0_5_D_for_adjudication_adjudicated_paper.json")
OUT_PATH = Path("results/phase2b_dilution_check.json")
END_OF_TURN_ID = 106
# String that marks the start of the forced-letter instruction block in B prompts
FORCED_LETTER_MARKER = "Reply with exactly one letter only"


def load_sae(layer, device="cuda"):
    sub = f"resid_post/layer_{layer}_width_16k_l0_medium/params.safetensors"
    path = hf_hub_download(SAE_REPO, sub)
    p = sft.load_file(path)
    return {k: v.to(device) for k, v in p.items()}


def parse_gold(g):
    return sorted(set(re.findall(r"[ABCD]", g.upper())))


def build_cases():
    fl = json.loads(FORCED_LETTER_PATH.read_text())
    st = json.loads(SINGLETURN_PATH.read_text())
    fl_by_id = {v["id"]: v for v in fl}
    st_by_id = {v["id"]: v for v in st}
    phase05 = json.loads(PHASE_0_5_PATH.read_text())
    p_by_id = {r["id"]: r for r in phase05["results"]}
    adj = json.loads(ADJUDICATED_PATH.read_text())
    adj_by_id = {r["case_id"]: r for r in adj}

    def _key(s):
        m = re.match(r"^(\D+)(\d+)$", s)
        return (m.group(1), int(m.group(2))) if m else (s, 0)

    cases = []
    for cid in sorted(fl_by_id, key=_key):
        fl_row = fl_by_id[cid]; st_row = st_by_id[cid]
        ph = p_by_id[cid]; ad = adj_by_id[cid]
        b_right = ph["B"]["correct"]
        d_right = bool(ad.get("gpt_5_2_thinking_high_is_correct")
                       and ad.get("claude_sonnet_4_6_is_correct"))
        if b_right and d_right: stratum = "both_right"
        elif (not b_right) and d_right: stratum = "format_flipped"
        elif (not b_right) and (not d_right): stratum = "both_wrong"
        else: stratum = "B_only_right"
        cases.append({
            "id": cid, "title": fl_row["title"],
            "B_prompt": fl_row["natural_forced_letter"],
            "D_prompt": st_row["patient_realistic"],
            "stratum": stratum,
        })
    return cases


def find_marker_token_idx(input_ids: list[int], tokenizer, prompt: str, marker: str) -> int | None:
    """Find the token index where the forced-letter marker starts in the
    chat-templated input_ids. Returns the index of the *first token* of the
    marker, or None if not present.

    Strategy: tokenize the substring "...prompt up to marker..." and find
    where it lands in the full ids. Cheap, robust to chat template wrapping.
    """
    if marker not in prompt:
        return None
    cut = prompt.index(marker)
    # Keep a couple of newlines of slack before the marker to avoid edge tokens
    head = prompt[:cut].rstrip()
    # Tokenize the head as a user message and use the *length* as a proxy
    head_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": head}], add_generation_prompt=False,
        return_tensors=None, return_dict=False,
    )
    if isinstance(head_ids, dict):
        head_ids = head_ids["input_ids"]
    head_ids = list(head_ids[0]) if isinstance(head_ids, list) and isinstance(head_ids[0], list) else list(head_ids)
    # head ends with <end_of_turn>; the marker tokens come AFTER it in the full prompt.
    # Look for first <end_of_turn> in input_ids and use that as the cut point.
    try:
        eot_idx = input_ids.index(END_OF_TURN_ID)
        # We want clinical content tokens [4 : eot_of_head_idx]. The head's
        # eot is at the same position the FULL prompt's content "ends" in
        # the truncated version. Find its position by length of head_ids minus
        # post-eot tokens.
        # Simpler: count tokens of head up to first <end_of_turn>.
        head_eot = head_ids.index(END_OF_TURN_ID)
        return head_eot  # index in full sequence where clinical content ends
    except ValueError:
        return None


def collect_residuals_with_aggregations(model, tok, prompt, layer, truncate_marker=None):
    """Forward pass; return three aggregations of content-token residuals:
    full_mean, max, and (if marker provided) truncated_mean.
    """
    messages = [{"role": "user", "content": prompt}]
    input_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=False,
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    input_ids_dev = input_ids.to(model.device)
    captured = {}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()
    if hasattr(model.model, "language_model"):
        target = model.model.language_model.layers[layer]
    else:
        target = model.model.layers[layer]
    handle = target.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids_dev)
    finally:
        handle.remove()

    h = captured["h"][0]  # [seq_len, d_model]
    ids = input_ids[0].tolist()
    try:
        eot_idx = ids.index(END_OF_TURN_ID)
    except ValueError:
        eot_idx = len(ids)
    start = 4
    if start >= eot_idx:
        start = min(4, len(ids) - 1)
        eot_idx = len(ids)

    content = h[start:eot_idx].float().cpu()  # [n_content, d_model]
    full_mean = content.mean(0)
    full_max = content.max(0).values

    # Truncated mean: cut at marker position, if present
    truncated_mean = None
    truncated_n = None
    if truncate_marker is not None and truncate_marker in prompt:
        # Find marker's token position by re-tokenizing the prefix-only prompt
        prefix = prompt.split(truncate_marker, 1)[0].rstrip()
        prefix_ids = tok.apply_chat_template(
            [{"role": "user", "content": prefix}], add_generation_prompt=True,
            return_tensors="pt", return_dict=False,
        )
        if not isinstance(prefix_ids, torch.Tensor):
            prefix_ids = prefix_ids["input_ids"]
        prefix_ids_list = prefix_ids[0].tolist()
        try:
            prefix_eot = prefix_ids_list.index(END_OF_TURN_ID)
            # Use prefix_eot as the truncation point in the FULL input_ids,
            # because content is identical up to that point.
            truncated_end = min(prefix_eot, eot_idx)
            if truncated_end > start:
                truncated_content = h[start:truncated_end].float().cpu()
                truncated_mean = truncated_content.mean(0)
                truncated_n = truncated_end - start
        except ValueError:
            pass

    return {
        "full_mean": full_mean, "full_max": full_max,
        "truncated_mean": truncated_mean, "truncated_n": truncated_n,
        "n_content": eot_idx - start,
    }


def project_and_rank(diff: torch.Tensor, w_enc: torch.Tensor, med_feats: list[int],
                     n_features: int):
    diff_unit = diff / (diff.norm() + 1e-8)
    w_enc_norms = w_enc.norm(dim=0)
    alignments = (diff_unit.to(w_enc.device) @ w_enc) / w_enc_norms.clamp(min=1e-8)
    alignments_cpu = alignments.cpu()
    abs_align = alignments_cpu.abs()
    sorted_idx = torch.argsort(abs_align, descending=True).tolist()
    rank_of = {f: sorted_idx.index(f) + 1 for f in med_feats}
    align_of = {f: float(alignments_cpu[f]) for f in med_feats}
    abs_align_of = {f: float(abs_align[f]) for f in med_feats}
    top10 = []
    for f in sorted_idx[:10]:
        top10.append({"feature": int(f), "alignment": float(alignments_cpu[f]),
                      "abs_alignment": float(abs_align[f]),
                      "is_medical": f in med_feats})
    return {"diff_norm": float(diff.norm()), "ranks": rank_of,
            "alignments": align_of, "abs_alignments": abs_align_of,
            "top10": top10}


def main():
    cases = build_cases()
    assert len(cases) == 60
    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    layer_results = {}
    for layer in LAYERS:
        print(f"\n=== Layer {layer} ===")
        sae = load_sae(layer)
        w_enc = sae["w_enc"].float()
        d_model = w_enc.shape[0]
        d_sae = w_enc.shape[1]

        # Per-case aggregations
        B_full = torch.zeros(len(cases), d_model)
        B_max  = torch.zeros(len(cases), d_model)
        B_trunc = torch.zeros(len(cases), d_model)
        D_full = torch.zeros(len(cases), d_model)
        D_max  = torch.zeros(len(cases), d_model)

        truncated_ns = []
        for i, c in enumerate(cases):
            B = collect_residuals_with_aggregations(model, tok, c["B_prompt"], layer,
                                                     truncate_marker=FORCED_LETTER_MARKER)
            D = collect_residuals_with_aggregations(model, tok, c["D_prompt"], layer)
            B_full[i] = B["full_mean"]
            B_max[i] = B["full_max"]
            if B["truncated_mean"] is not None:
                B_trunc[i] = B["truncated_mean"]
                truncated_ns.append((B["truncated_n"], B["n_content"]))
            else:
                B_trunc[i] = B["full_mean"]  # fallback
                truncated_ns.append((B["n_content"], B["n_content"]))
            D_full[i] = D["full_mean"]
            D_max[i] = D["full_max"]
            if (i + 1) % 15 == 0:
                print(f"  collected {i+1}/60 (B trunc tokens / full tokens: "
                      f"{truncated_ns[-1][0]}/{truncated_ns[-1][1]})")

        med_feats = MEDICAL_FEATURES[layer]

        # Three (B - D) directions
        diff_full = (B_full - D_full).mean(0)
        diff_trunc = (B_trunc - D_full).mean(0)
        diff_max  = (B_max - D_max).mean(0)

        proj_full = project_and_rank(diff_full, w_enc, med_feats, d_sae)
        proj_trunc = project_and_rank(diff_trunc, w_enc, med_feats, d_sae)
        proj_max = project_and_rank(diff_max, w_enc, med_feats, d_sae)

        avg_truncated_n = float(np.mean([t for t, _ in truncated_ns]))
        avg_full_n = float(np.mean([f for _, f in truncated_ns]))
        layer_results[layer] = {
            "medical_features": med_feats,
            "n_features_total": d_sae,
            "avg_n_content_tokens_full": avg_full_n,
            "avg_n_content_tokens_truncated_B": avg_truncated_n,
            "tokens_dropped_per_B_prompt": avg_full_n - avg_truncated_n,
            "full_mean_pool":     proj_full,
            "truncated_mean_pool": proj_trunc,
            "max_pool":           proj_max,
        }

        del sae
        torch.cuda.empty_cache()

    out = {
        "model": MODEL_ID, "sae_repo": SAE_REPO, "layers": LAYERS,
        "n_cases": len(cases), "by_layer": layer_results,
        "note": ("Three aggregation strategies tested per layer: full mean-pool "
                 "(reproduces Phase 2), length-controlled mean-pool "
                 "(B truncated at 'Reply with exactly one letter only'), "
                 "and max-pool over content tokens."),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")

    print("\n=== Phase 2b summary ===")
    for layer in LAYERS:
        info = layer_results[layer]
        print(f"\nL{layer}: avg B tokens full/truncated = "
              f"{info['avg_n_content_tokens_full']:.0f} / "
              f"{info['avg_n_content_tokens_truncated_B']:.0f} "
              f"(dropped {info['tokens_dropped_per_B_prompt']:.0f})")
        print(f"  Medical-feature ranks (out of {info['n_features_total']}):")
        print(f"  {'feature':>8s}  {'full mean':>10s}  {'trunc mean':>10s}  {'max pool':>10s}")
        for f in info["medical_features"]:
            r_full = info["full_mean_pool"]["ranks"][f]
            r_trunc = info["truncated_mean_pool"]["ranks"][f]
            r_max = info["max_pool"]["ranks"][f]
            a_full = info["full_mean_pool"]["alignments"][f]
            a_trunc = info["truncated_mean_pool"]["alignments"][f]
            a_max = info["max_pool"]["alignments"][f]
            print(f"  {f:>8d}  {r_full:>5d}({a_full:+.3f})  "
                  f"{r_trunc:>5d}({a_trunc:+.3f})  "
                  f"{r_max:>5d}({a_max:+.3f})")


if __name__ == "__main__":
    main()
