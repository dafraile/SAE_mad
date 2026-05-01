"""
Phase 1b — Magnitude-matched random-feature control for Phase 1.

Phase 1 finding: medical features show lower modulation index than random
features at L17 and L29 under the format change. Random features were drawn
from "fires on this content" without matching activation magnitude. This
follow-up replaces that picker so random features have mean activation
magnitudes in the same band as the medical features.

Question: Does the L17/L29 invariance signal survive magnitude matching?

If YES: medical features are specifically preserved (not just that
"high-magnitude features look invariant").
If NO: the apparent effect was driven by scale, not feature identity.

Output: results/phase1b_magnitude_matched.json
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import safetensors.torch as sft
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-4b-it"
SAE_REPO = "google/gemma-scope-2-4b-it"
LAYERS = [9, 17, 22, 29]

MEDICAL_FEATURES = {
    9:  [139, 9909, 956],
    17: [9854, 368, 1539],
    22: [1181, 365, 8389],
    29: [12570, 893, 12845],
}

N_RANDOM_FEATURES = 30
RANDOM_SEED = 42
# Magnitude band for matching: random features must have mean activation
# in [MAG_LO * min(med_means), MAG_HI * max(med_means)]
MAG_LO = 0.5
MAG_HI = 2.0

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
OUT_PATH = Path("results/phase1b_magnitude_matched.json")

END_OF_TURN_ID = 106  # Gemma 3 chat-template token


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


def parse_gold(gold):
    return sorted(set(re.findall(r"[ABCD]", gold.upper())))


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
    for cid in sorted(fl_by_id.keys(), key=_key):
        fl_row = fl_by_id[cid]
        st_row = st_by_id[cid]
        ph = p_by_id[cid]
        ad = adj_by_id[cid]
        b_right = ph["B"]["correct"]
        d_right = bool(
            ad.get("gpt_5_2_thinking_high_is_correct")
            and ad.get("claude_sonnet_4_6_is_correct")
        )
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


def get_content_residuals(model, tokenizer, prompt, layer):
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
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
    return h[start:eot_idx].float().cpu()


def pick_random_magnitude_matched(ref_acts, med_feats, n, seed,
                                   mag_lo=MAG_LO, mag_hi=MAG_HI):
    """ref_acts: [n_cases, d_sae] of mean-pooled feature activations.

    Pick n features whose per-feature mean across cases is in
    [mag_lo * min(med_means), mag_hi * max(med_means)]. Excludes the
    medical features themselves.
    """
    mean_per_feat = ref_acts.float().mean(0)  # [d_sae]
    med_means = mean_per_feat[med_feats]
    lo = mag_lo * med_means.min().item()
    hi = mag_hi * med_means.max().item()
    # Pool: features in the magnitude band, excluding medical features
    in_band = ((mean_per_feat >= lo) & (mean_per_feat <= hi))
    in_band[med_feats] = False
    pool = in_band.nonzero(as_tuple=True)[0].tolist()
    rng = np.random.default_rng(seed)
    if len(pool) < n:
        return pool, {"pool_size": len(pool), "lo": lo, "hi": hi,
                      "med_means": med_means.tolist()}
    chosen = sorted(rng.choice(pool, size=n, replace=False).tolist())
    return chosen, {"pool_size": len(pool), "lo": lo, "hi": hi,
                    "med_means": med_means.tolist()}


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
        sae = JumpReLUSAE.from_hf(SAE_REPO, layer)
        med_feats = MEDICAL_FEATURES[layer]

        # Pass 1: collect activations across all cases for both conditions.
        all_acts_B_mean, all_acts_D_mean = [], []
        all_acts_B_max, all_acts_D_max = [], []
        for i, c in enumerate(cases):
            rB = get_content_residuals(model, tok, c["B_prompt"], layer).to("cuda")
            rD = get_content_residuals(model, tok, c["D_prompt"], layer).to("cuda")
            with torch.no_grad():
                fB = sae.encode(rB).float()
                fD = sae.encode(rD).float()
            all_acts_B_mean.append(fB.mean(0).cpu())
            all_acts_D_mean.append(fD.mean(0).cpu())
            all_acts_B_max.append(fB.max(0).values.cpu())
            all_acts_D_max.append(fD.max(0).values.cpu())
            if (i + 1) % 15 == 0:
                print(f"  collected {i+1}/60")

        all_acts_B = torch.stack(all_acts_B_mean)
        all_acts_D = torch.stack(all_acts_D_mean)
        all_acts_B_max_t = torch.stack(all_acts_B_max)
        all_acts_D_max_t = torch.stack(all_acts_D_max)
        ref = torch.cat([all_acts_B, all_acts_D], dim=0)

        # Magnitude-matched random features
        rand_feats, picker_info = pick_random_magnitude_matched(
            ref, med_feats, n=N_RANDOM_FEATURES, seed=RANDOM_SEED,
        )
        # Diagnostic: actual mean activation of medical and random features
        med_means_actual = ref.mean(0)[med_feats].tolist()
        rnd_means_actual = ref.mean(0)[rand_feats].tolist()
        print(f"  medical features: {med_feats} mean acts: "
              f"{[round(x, 1) for x in med_means_actual]}")
        print(f"  band [{picker_info['lo']:.1f}, {picker_info['hi']:.1f}], "
              f"pool size {picker_info['pool_size']}")
        print(f"  random features: {rand_feats[:6]}... mean acts: "
              f"{[round(x, 1) for x in rnd_means_actual[:6]]}...")

        # Per-case metrics
        per_case = []
        for i, c in enumerate(cases):
            aB = all_acts_B[i]
            aD = all_acts_D[i]
            entry = {
                "id": c["id"], "stratum": c["stratum"],
                "B_correct": c["B_correct"],
                "D_correct_both_judges": c["D_correct_both_judges"],
                "gold_letters": c["gold_letters"],
            }
            for kind, feats in [("medical", med_feats), ("random", rand_feats)]:
                v_B = aB[feats]
                v_D = aD[feats]
                v_B_max = all_acts_B_max_t[i][feats]
                v_D_max = all_acts_D_max_t[i][feats]
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

        # Stratum summary
        agg = defaultdict(lambda: {"medical_cos": [], "medical_mod": [],
                                   "random_cos": [], "random_mod": []})
        for e in per_case:
            s = e["stratum"]
            agg[s]["medical_cos"].append(e["medical_cosine"])
            agg[s]["medical_mod"].append(e["medical_mod_index"])
            agg[s]["random_cos"].append(e["random_cosine"])
            agg[s]["random_mod"].append(e["random_mod_index"])

        def _stat(xs):
            xs = [x for x in xs if x == x]
            if not xs: return None
            arr = np.array(xs)
            return {"n": len(xs), "mean": float(arr.mean()),
                    "std": float(arr.std()), "median": float(np.median(arr))}

        summary = {s: {k: _stat(vs) for k, vs in d.items()} for s, d in agg.items()}

        layer_results[layer] = {
            "medical_features": med_feats,
            "medical_means_actual": med_means_actual,
            "random_features": rand_feats,
            "random_means_actual": rnd_means_actual,
            "picker_info": picker_info,
            "per_case": per_case,
            "stratum_summary": summary,
        }

        del sae
        torch.cuda.empty_cache()

    out = {
        "model": MODEL_ID,
        "sae_repo": SAE_REPO,
        "layers": LAYERS,
        "n_cases": len(cases),
        "magnitude_band": {"lo_factor": MAG_LO, "hi_factor": MAG_HI},
        "by_layer": layer_results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")

    print(f"\n=== Phase 1b summary (magnitude-matched random features) ===")
    print(f"{'L':<3s}{'stratum':<18s}{'n':>4s}  {'med_mod':>8s} {'rnd_mod':>8s} {'diff':>8s}")
    for layer in LAYERS:
        for s in ["format_flipped", "both_right", "both_wrong", "B_only_right"]:
            d = layer_results[layer]["stratum_summary"].get(s, {})
            mm = d.get("medical_mod") or {}
            rm = d.get("random_mod") or {}
            if not mm: continue
            print(f"L{layer:<2d}{s:<18s}{mm.get('n','?'):>4}  "
                  f"{mm.get('mean', float('nan')):>8.3f} "
                  f"{rm.get('mean', float('nan')):>8.3f} "
                  f"{(mm.get('mean',0)-rm.get('mean',0)):>+8.3f}")


if __name__ == "__main__":
    main()
