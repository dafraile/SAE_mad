"""
Phase 3b — Run the full mechanistic pipeline on Gemma 3 12B IT.

Reads the medical features identified in Phase 3 (results/phase3_12b_features.json)
and runs:

  Phase 0  — Capability floor on `original_structured` triage prompts
  Phase 0.5 — Three-cell behavioral test (A struct+letter, B nat+letter, D nat+free)
  Phase 1b — Magnitude-matched feature invariance test
  Phase 2b — Dilution-controlled ActAdd projection

All in one script. After this lands, run the paper-scale adjudicator on the
D-cell free-text outputs (separate API call), then the analyses are complete.

Output:
  results/phase3b_12b_phase0.json
  results/phase3b_12b_phase0_5.json
  results/phase3b_12b_phase1b.json
  results/phase3b_12b_phase2b.json
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import safetensors.torch as sft
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-12b-it"
SAE_REPO = "google/gemma-scope-2-12b-it"
LAYERS = [12, 24, 31, 41]

FEATURE_ID_PATH = Path("results/phase3_12b_features.json")
SINGLETURN_PATH = Path(
    "nature_triage_expanded_replication/paper_faithful_replication/data/"
    "canonical_singleturn_vignettes.json"
)
FORCED_LETTER_PATH = Path(
    "nature_triage_expanded_replication/paper_faithful_forced_letter/data/"
    "canonical_forced_letter_vignettes.json"
)

OUT_PHASE0 = Path("results/phase3b_12b_phase0.json")
OUT_PHASE0_5 = Path("results/phase3b_12b_phase0_5.json")
OUT_PHASE1B = Path("results/phase3b_12b_phase1b.json")
OUT_PHASE2B = Path("results/phase3b_12b_phase2b.json")

END_OF_TURN_ID = 106
FORCED_LETTER_MARKER = "Reply with exactly one letter only"
N_RANDOM_FEATURES = 30
RANDOM_SEED = 42
MAG_LO = 0.5
MAG_HI = 2.0


# === SAE ===

class JumpReLUSAE:
    def __init__(self, w_enc, w_dec, b_enc, b_dec, threshold, device):
        self.w_enc = w_enc.to(device)
        self.w_dec = w_dec.to(device)
        self.b_enc = b_enc.to(device)
        self.b_dec = b_dec.to(device)
        self.threshold = threshold.to(device)
        self.d_sae = w_enc.shape[1]
        self.d_model = w_enc.shape[0]

    @classmethod
    def from_hf(cls, repo, layer, width="16k", l0="medium", device="cuda"):
        sub = f"resid_post/layer_{layer}_width_{width}_l0_{l0}/params.safetensors"
        path = hf_hub_download(repo, sub)
        p = sft.load_file(path)
        return cls(p["w_enc"], p["w_dec"], p["b_enc"], p["b_dec"], p["threshold"], device)

    def encode(self, x):
        pre = x.float() @ self.w_enc + self.b_enc
        return pre * (pre > self.threshold).float()


# === Helpers ===

LETTER_LINE_RE = re.compile(r"\b([ABCD])\b")


def parse_gold(g):
    return sorted(set(re.findall(r"[ABCD]", g.upper())))


def extract_forced_letter(output: str):
    for line in output.splitlines():
        s = line.strip().strip(".,:;*-")
        if s in ("A", "B", "C", "D"): return s
    m = LETTER_LINE_RE.search(output)
    return m.group(1) if m else None


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
        fl_row, st_row = fl_by_id[cid], st_by_id[cid]
        cases.append({
            "id": cid, "title": fl_row["title"],
            "gold_raw": fl_row["gold_standard_triage"],
            "gold_letters": parse_gold(fl_row["gold_standard_triage"]),
            "structured_orig": st_row["original_structured"],
            "A_prompt": fl_row["structured_forced_letter"],
            "B_prompt": fl_row["natural_forced_letter"],
            "D_prompt": st_row["patient_realistic"],
        })
    return cases


def get_target_layer(model, layer):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer]
    return model.model.layers[layer]


def hook_layer(model, layer):
    captured = {}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()
    handle = get_target_layer(model, layer).register_forward_hook(hook)
    return handle, captured


def chat_template_ids(tok, prompt):
    input_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        return_tensors="pt", return_dict=False,
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    return input_ids


def get_residuals_per_layer(model, tok, prompt, layers):
    """Single forward pass; capture residuals at all requested layers.
    Returns dict[layer] -> [seq_len, d_model] (cpu, float32).
    """
    input_ids = chat_template_ids(tok, prompt).to(model.device)
    captured = {}
    handles = []
    for L in layers:
        cap = {}
        def hook(_m, _i, out, cap=cap):
            h = out[0] if isinstance(out, tuple) else out
            cap["h"] = h.detach()
        handles.append(get_target_layer(model, L).register_forward_hook(hook))
        captured[L] = cap
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h in handles: h.remove()
    out = {}
    for L, cap in captured.items():
        out[L] = cap["h"][0]  # GPU float; do not move to cpu (saves time)
    ids = input_ids[0].tolist()
    try:
        eot_idx = ids.index(END_OF_TURN_ID)
    except ValueError:
        eot_idx = len(ids)
    return out, ids, eot_idx


def truncated_eot_for_marker(prompt, tok, marker):
    if marker not in prompt: return None
    prefix = prompt.split(marker, 1)[0].rstrip()
    pid = chat_template_ids(tok, prefix)[0].tolist()
    try: return pid.index(END_OF_TURN_ID)
    except ValueError: return None


# === Phase 0 ===

def phase0(model, tok, cases):
    print("\n## Phase 0 — Capability floor")
    results = []
    t0 = time.time()
    for i, c in enumerate(cases):
        ids = chat_template_ids(tok, c["structured_orig"]).to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids, max_new_tokens=400, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        letter = extract_forced_letter(gen)
        ok = letter in c["gold_letters"] if letter else False
        results.append({"id": c["id"], "gold": c["gold_raw"],
                        "predicted": letter, "correct": ok,
                        "raw": gen})
        if (i + 1) % 10 == 0:
            print(f"  Phase 0: {i+1}/60 ({time.time()-t0:.0f}s)")
    n = len(results); n_correct = sum(r["correct"] for r in results)
    summary = {"model": MODEL_ID, "n": n, "correct": n_correct,
               "accuracy": n_correct/n, "results": results}
    OUT_PHASE0.parent.mkdir(parents=True, exist_ok=True)
    OUT_PHASE0.write_text(json.dumps(summary, indent=2))
    print(f"  Accuracy: {n_correct}/{n} = {n_correct/n:.1%}")
    return summary


# === Phase 0.5 ===

def phase0_5(model, tok, cases):
    print("\n## Phase 0.5 — Three cells")
    rows = []
    t0 = time.time()
    for i, c in enumerate(cases):
        row = {"id": c["id"], "title": c["title"], "gold_raw": c["gold_raw"],
               "gold_letters": c["gold_letters"]}
        for cell, prompt in [("A", c["A_prompt"]), ("B", c["B_prompt"]), ("D", c["D_prompt"])]:
            ids = chat_template_ids(tok, prompt).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=ids, max_new_tokens=400, do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            if cell in ("A", "B"):
                letter = extract_forced_letter(gen)
                row[cell] = {"predicted": letter,
                             "correct": (letter in c["gold_letters"]) if letter else False,
                             "unparsed": letter is None, "raw": gen}
            else:
                row[cell] = {"raw": gen}  # adjudicated separately later
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  Phase 0.5: {i+1}/60 ({time.time()-t0:.0f}s)")
    a_corr = sum(r["A"]["correct"] for r in rows)
    b_corr = sum(r["B"]["correct"] for r in rows)
    summary = {"model": MODEL_ID, "results": rows,
               "cells": {"A": {"correct": a_corr, "n": 60, "acc": a_corr/60},
                         "B": {"correct": b_corr, "n": 60, "acc": b_corr/60}}}
    OUT_PHASE0_5.write_text(json.dumps(summary, indent=2))
    print(f"  A: {a_corr}/60 = {a_corr/60:.1%}")
    print(f"  B: {b_corr}/60 = {b_corr/60:.1%}")
    print(f"  D: pending adjudication")
    return summary


# === Phase 1b helpers ===

def get_content_features_per_token(sae, residuals, ids, eot_idx, start=4):
    if start >= eot_idx:
        start = min(4, len(ids) - 1)
        eot_idx = len(ids)
    content = residuals[start:eot_idx].to(sae.w_enc.dtype).to(sae.w_enc.device)
    with torch.no_grad():
        feats = sae.encode(content)
    return feats.float()  # [n_content, d_sae]


def pick_random_magnitude_matched(ref_acts, med_feats, n, seed,
                                   mag_lo=MAG_LO, mag_hi=MAG_HI):
    mean_per_feat = ref_acts.float().mean(0)
    med_means = mean_per_feat[med_feats]
    lo = mag_lo * med_means.min().item()
    hi = mag_hi * med_means.max().item()
    in_band = ((mean_per_feat >= lo) & (mean_per_feat <= hi))
    in_band[med_feats] = False
    pool = in_band.nonzero(as_tuple=True)[0].tolist()
    rng = np.random.default_rng(seed)
    if len(pool) < n:
        return pool
    return sorted(rng.choice(pool, size=n, replace=False).tolist())


# === Phase 1b + 2b combined ===

def phase1b_2b(model, tok, cases, medical_per_layer):
    print("\n## Phase 1b + 2b combined")
    p1b = {}
    p2b = {}
    for layer in LAYERS:
        print(f"\n  Layer {layer}")
        sae = JumpReLUSAE.from_hf(SAE_REPO, layer)
        d_sae = sae.d_sae

        # Per-case aggregations
        all_feat_B_mean = torch.zeros(len(cases), d_sae)
        all_feat_D_mean = torch.zeros(len(cases), d_sae)
        all_feat_B_max = torch.zeros(len(cases), d_sae)
        all_feat_D_max = torch.zeros(len(cases), d_sae)
        # Residual aggregations for Phase 2b
        d_model = sae.d_model
        res_B_full = torch.zeros(len(cases), d_model)
        res_B_trunc = torch.zeros(len(cases), d_model)
        res_B_max = torch.zeros(len(cases), d_model)
        res_D_full = torch.zeros(len(cases), d_model)
        res_D_max = torch.zeros(len(cases), d_model)

        t0 = time.time()
        for i, c in enumerate(cases):
            # B prompt
            res_layers, ids_b, eot_b = get_residuals_per_layer(model, tok, c["B_prompt"], [layer])
            r_b = res_layers[layer]  # [seq_len, d_model] gpu
            # truncated eot for B
            trunc_eot_b = truncated_eot_for_marker(c["B_prompt"], tok, FORCED_LETTER_MARKER)
            start = 4
            content_b = r_b[start:eot_b]
            feats_b = sae.encode(content_b.to(sae.w_enc.dtype)).float()
            all_feat_B_mean[i] = feats_b.mean(0).cpu()
            all_feat_B_max[i] = feats_b.max(0).values.cpu()
            res_B_full[i] = content_b.float().mean(0).cpu()
            res_B_max[i] = content_b.float().max(0).values.cpu()
            if trunc_eot_b is not None and trunc_eot_b > start:
                truncated_end = min(trunc_eot_b, eot_b)
                if truncated_end > start:
                    res_B_trunc[i] = r_b[start:truncated_end].float().mean(0).cpu()
                else:
                    res_B_trunc[i] = res_B_full[i]
            else:
                res_B_trunc[i] = res_B_full[i]
            # D prompt
            res_layers, ids_d, eot_d = get_residuals_per_layer(model, tok, c["D_prompt"], [layer])
            r_d = res_layers[layer]
            content_d = r_d[start:eot_d]
            feats_d = sae.encode(content_d.to(sae.w_enc.dtype)).float()
            all_feat_D_mean[i] = feats_d.mean(0).cpu()
            all_feat_D_max[i] = feats_d.max(0).values.cpu()
            res_D_full[i] = content_d.float().mean(0).cpu()
            res_D_max[i] = content_d.float().max(0).values.cpu()
            del r_b, r_d, content_b, content_d, feats_b, feats_d
            if (i + 1) % 15 == 0:
                print(f"    L{layer}: collected {i+1}/60 ({time.time()-t0:.0f}s)")

        med_feats = medical_per_layer[layer]
        ref = torch.cat([all_feat_B_mean, all_feat_D_mean], dim=0)
        rand_feats = pick_random_magnitude_matched(ref, med_feats, N_RANDOM_FEATURES, RANDOM_SEED)

        # Phase 1b: per-case mod-index, cosine
        per_case = []
        for i, c in enumerate(cases):
            entry = {"id": c["id"], "gold_letters": c["gold_letters"]}
            for kind, feats in [("medical", med_feats), ("random", rand_feats)]:
                v_B = all_feat_B_mean[i][feats]
                v_D = all_feat_D_mean[i][feats]
                v_B_max = all_feat_B_max[i][feats]
                v_D_max = all_feat_D_max[i][feats]
                denom = v_B.norm() * v_D.norm()
                cos = (v_B @ v_D / denom).item() if denom > 0 else float("nan")
                num = (v_D - v_B).abs()
                den = (v_B.abs() + v_D.abs()) / 2
                mod = (num / den.clamp(min=1e-8)).mean().item()
                entry[f"{kind}_acts_B_mean"] = v_B.tolist()
                entry[f"{kind}_acts_D_mean"] = v_D.tolist()
                entry[f"{kind}_acts_B_max"] = v_B_max.tolist()
                entry[f"{kind}_acts_D_max"] = v_D_max.tolist()
                entry[f"{kind}_cosine"] = cos
                entry[f"{kind}_mod_index"] = mod
            per_case.append(entry)
        p1b[layer] = {
            "medical_features": med_feats,
            "random_features": rand_feats,
            "per_case": per_case,
        }

        # Phase 2b: 3 diff-norm projections
        w_enc = sae.w_enc.float()
        w_enc_norms = w_enc.norm(dim=0)
        def project(diff):
            diff_unit = diff / (diff.norm() + 1e-8)
            alignments = (diff_unit.to(w_enc.device) @ w_enc) / w_enc_norms.clamp(min=1e-8)
            alignments_cpu = alignments.cpu()
            abs_align = alignments_cpu.abs()
            sorted_idx = torch.argsort(abs_align, descending=True).tolist()
            return {
                "diff_norm": float(diff.norm()),
                "ranks": {str(f): sorted_idx.index(f) + 1 for f in med_feats},
                "alignments": {str(f): float(alignments_cpu[f]) for f in med_feats},
                "abs_alignments": {str(f): float(abs_align[f]) for f in med_feats},
                "top10": [
                    {"feature": int(f), "alignment": float(alignments_cpu[f]),
                     "abs_alignment": float(abs_align[f]),
                     "is_medical": f in med_feats}
                    for f in sorted_idx[:10]
                ],
            }
        diff_full = (res_B_full - res_D_full).mean(0)
        diff_trunc = (res_B_trunc - res_D_full).mean(0)
        diff_max = (res_B_max - res_D_max).mean(0)
        p2b[layer] = {
            "medical_features": med_feats,
            "n_features_total": d_sae,
            "full_mean_pool": project(diff_full),
            "truncated_mean_pool": project(diff_trunc),
            "max_pool": project(diff_max),
        }

        del sae, all_feat_B_mean, all_feat_D_mean, all_feat_B_max, all_feat_D_max
        del res_B_full, res_B_trunc, res_B_max, res_D_full, res_D_max
        torch.cuda.empty_cache()

    OUT_PHASE1B.write_text(json.dumps({"model": MODEL_ID, "by_layer": p1b}, indent=2, default=lambda o: int(o) if hasattr(o, "item") else str(o)))
    OUT_PHASE2B.write_text(json.dumps({"model": MODEL_ID, "by_layer": p2b}, indent=2, default=lambda o: int(o) if hasattr(o, "item") else str(o)))
    print(f"\n  Wrote {OUT_PHASE1B}")
    print(f"  Wrote {OUT_PHASE2B}")


def main():
    print(f"=== 12B mechanistic pipeline ===")
    print(f"Reading features from {FEATURE_ID_PATH}")
    feat_data = json.loads(FEATURE_ID_PATH.read_text())
    medical_per_layer = {}
    for layer in LAYERS:
        info = feat_data["by_layer"][str(layer)]
        # Prefer filtered, fall back to unfiltered if filter is too strict
        if info["top_filtered"]:
            top = info["top_filtered"][:3]
        else:
            top = info["top10_unfiltered"][:3]
        medical_per_layer[layer] = [t["feature"] for t in top]
        print(f"  L{layer}: {medical_per_layer[layer]}")

    cases = build_cases()
    print(f"Loaded {len(cases)} cases")

    print(f"\nLoading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    phase0(model, tok, cases)
    phase0_5(model, tok, cases)
    phase1b_2b(model, tok, cases, medical_per_layer)


if __name__ == "__main__":
    main()
