"""
Phase 2 — ActAdd-style projection of the format-difference direction onto
SAE features.

Phase 1b shows medical features are invariant under the format change. This
is evidence that the format effect does NOT live in the medical-feature
subspace. But it doesn't say where it DOES live. To localize it we use the
ActAdd methodology: take the activation difference between conditions as a
steering direction, then ask which SAE features align with that direction.

If the medical features (12570, 893, 12845 at L29) are NOT in the top-K
features by alignment with the (B - D) direction, we have stronger evidence
that the format effect operates orthogonally to medical-content encoding.
That is the strongest version of the Version B claim in TRIAGE_FINDINGS.md.

Steps:
1. For each case c in {60 cases}, for each condition in {B, D}, capture the
   mean-pooled residual over user content tokens at layer L.
2. Compute diff_c = residual_B(c) - residual_D(c). Average across cases
   (and per-stratum) to get the format-difference direction.
3. For each SAE feature f, compute the cosine alignment between diff and
   the encoder direction w_enc[:, f]. Rank features by |alignment|.
4. Report where the medical features land in the ranking.

Runs as one script: model + SAE, residual collection, then local analysis.
~25 min wall on a small GPU + immediate analysis.

Output: results/phase2_actadd_projection.json
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
LAYERS = [17, 29]  # primary L29, confirmation L17

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
OUT_PATH = Path("results/phase2_actadd_projection.json")
END_OF_TURN_ID = 106


def load_sae(layer: int, device: str = "cuda"):
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
        fl_row, st_row = fl_by_id[cid], st_by_id[cid]
        ph, ad = p_by_id[cid], adj_by_id[cid]
        b_right = ph["B"]["correct"]
        d_right = bool(ad.get("gpt_5_2_thinking_high_is_correct")
                       and ad.get("claude_sonnet_4_6_is_correct"))
        if b_right and d_right: stratum = "both_right"
        elif (not b_right) and d_right: stratum = "format_flipped"
        elif (not b_right) and (not d_right): stratum = "both_wrong"
        else: stratum = "B_only_right"
        cases.append({
            "id": cid, "title": fl_row["title"],
            "gold_letters": parse_gold(fl_row["gold_standard_triage"]),
            "B_prompt": fl_row["natural_forced_letter"],
            "D_prompt": st_row["patient_realistic"],
            "B_correct": b_right, "D_correct_both_judges": d_right,
            "stratum": stratum,
        })
    return cases


def get_content_mean_residual(model, tok, prompt, layer):
    """Mean-pool residual at given layer over user content tokens."""
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
    h = captured["h"][0]
    ids = input_ids[0].tolist()
    try:
        eot_idx = ids.index(END_OF_TURN_ID)
    except ValueError:
        eot_idx = len(ids)
    start = 4
    if start >= eot_idx:
        start = min(4, len(ids) - 1)
        eot_idx = len(ids)
    return h[start:eot_idx].float().mean(0).cpu()  # [d_model]


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
        w_enc = sae["w_enc"].float()  # [d_model, d_sae]
        d_model, d_sae = w_enc.shape

        # Collect per-case mean residuals
        res_B = torch.zeros(len(cases), d_model)
        res_D = torch.zeros(len(cases), d_model)
        for i, c in enumerate(cases):
            res_B[i] = get_content_mean_residual(model, tok, c["B_prompt"], layer)
            res_D[i] = get_content_mean_residual(model, tok, c["D_prompt"], layer)
            if (i + 1) % 15 == 0:
                print(f"  collected {i+1}/60")

        # ActAdd-style direction: mean over cases of (B - D)
        diff = (res_B - res_D).mean(0)  # [d_model]
        diff_unit = diff / (diff.norm() + 1e-8)

        # Project onto each feature's encoder direction (cosine alignment)
        w_enc_norms = w_enc.norm(dim=0)  # [d_sae]
        # Cosine = (diff . w_enc[:, f]) / (||diff|| * ||w_enc[:, f]||)
        # = (diff_unit . w_enc[:, f] / ||w_enc[:, f]||)
        diff_unit_dev = diff_unit.to(w_enc.device)
        alignments = (diff_unit_dev @ w_enc) / w_enc_norms.clamp(min=1e-8)  # [d_sae]
        alignments_cpu = alignments.cpu()
        abs_align = alignments_cpu.abs()

        # Rankings
        med_feats = MEDICAL_FEATURES[layer]
        # Position in |alignment| ranking (1 = highest absolute alignment)
        sorted_abs_idx = torch.argsort(abs_align, descending=True).tolist()
        rank_of = {f: sorted_abs_idx.index(f) + 1 for f in med_feats}
        med_alignments = {f: alignments_cpu[f].item() for f in med_feats}
        med_abs_alignments = {f: abs_align[f].item() for f in med_feats}

        # Top-20 features by |alignment|
        top20 = []
        for f in sorted_abs_idx[:20]:
            top20.append({
                "feature": int(f),
                "alignment": float(alignments_cpu[f]),
                "abs_alignment": float(abs_align[f]),
                "is_medical": f in med_feats,
            })

        # Per-stratum direction (in case strata diverge)
        from collections import defaultdict
        strat_idx = defaultdict(list)
        for i, c in enumerate(cases):
            strat_idx[c["stratum"]].append(i)
        per_stratum = {}
        for s, idxs in strat_idx.items():
            if not idxs: continue
            if len(idxs) < 2: continue
            d_s = (res_B[idxs] - res_D[idxs]).mean(0)
            d_s_unit = d_s / (d_s.norm() + 1e-8)
            align_s = (d_s_unit.to(w_enc.device) @ w_enc) / w_enc_norms.clamp(min=1e-8)
            align_s_cpu = align_s.cpu()
            abs_s = align_s_cpu.abs()
            sorted_s = torch.argsort(abs_s, descending=True).tolist()
            per_stratum[s] = {
                "n": len(idxs),
                "diff_norm": float(d_s.norm()),
                "med_ranks": {str(f): sorted_s.index(f) + 1 for f in med_feats},
                "med_alignments": {str(f): float(align_s_cpu[f]) for f in med_feats},
                "top10": [
                    {"feature": int(f), "alignment": float(align_s_cpu[f]),
                     "abs_alignment": float(abs_s[f]), "is_medical": f in med_feats}
                    for f in sorted_s[:10]
                ],
            }

        layer_results[layer] = {
            "diff_norm": float(diff.norm()),
            "medical_features": med_feats,
            "medical_ranks_in_abs_alignment": rank_of,
            "medical_alignments": med_alignments,
            "medical_abs_alignments": med_abs_alignments,
            "top20_by_abs_alignment": top20,
            "per_stratum": per_stratum,
            "n_features_total": d_sae,
        }

        # Save raw residuals too (small, useful for further analysis)
        np.savez_compressed(
            f"results/phase2_residuals_L{layer}.npz",
            res_B=res_B.numpy(), res_D=res_D.numpy(),
            case_ids=np.array([c["id"] for c in cases]),
            strata=np.array([c["stratum"] for c in cases]),
        )

        del sae
        torch.cuda.empty_cache()

    out = {
        "model": MODEL_ID,
        "sae_repo": SAE_REPO,
        "layers": LAYERS,
        "n_cases": len(cases),
        "by_layer": layer_results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")

    print(f"\n=== Phase 2 ActAdd projection summary ===")
    for layer in LAYERS:
        info = layer_results[layer]
        print(f"\nL{layer}: ||(B - D)|| = {info['diff_norm']:.2f}, "
              f"d_sae = {info['n_features_total']}")
        print(f"  medical features and their |alignment| rank (out of {info['n_features_total']}):")
        for f in info["medical_features"]:
            r = info["medical_ranks_in_abs_alignment"][f]
            a = info["medical_abs_alignments"][f]
            pct = 100 * r / info['n_features_total']
            print(f"    feature {f:>5d}: rank {r:>5d} ({pct:.1f}%-ile), "
                  f"|alignment| = {a:.4f}")
        print(f"  top 5 most-aligned features (any kind):")
        for entry in info["top20_by_abs_alignment"][:5]:
            mark = " [MEDICAL]" if entry["is_medical"] else ""
            print(f"    feature {entry['feature']:>5d}: |alignment| = {entry['abs_alignment']:.4f}{mark}")


if __name__ == "__main__":
    main()
