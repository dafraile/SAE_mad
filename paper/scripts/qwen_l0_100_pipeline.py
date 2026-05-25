"""qwen_l0_100_pipeline.py -- complete Qwen3-8B mechanistic pipeline on
the Qwen-Scope L0_100 SAE variant, as an SAE-quality robustness check
against the L0_50 results in the main paper.

What we ran on L0_50 (already in repo):
  - Medical-vs-non-medical contrastive at L31 → 3 medical features
    [29074, 48973, 60699]
  - Phase 1b masked invariance NL−NF (perm-p 0.012)
  - SL−SF masked invariance (paired Δ -0.008, CI crosses zero)
  - Decision-token logit attribution
  - Decision-token top-K features
  - Recon error ~38% at L31

What L0_100 changes:
  - SAE was trained with TopK=100 instead of 50, expected ~half the
    reconstruction error
  - Feature basis is different — must re-identify medical features
    from L0_100 activations

This script does all of the above in a single GPU session, producing
files named `qwen_l0_100_*.json` so the original L0_50 results stay
intact for side-by-side comparison.

Outputs:
  results/qwen_l0_100_feature_id.json
  results/qwen_l0_100_masked_invariance.json       (covers NL-NF + SL-SF)
  results/qwen_l0_100_masked_full_activations.npz
  results/qwen_l0_100_decision_token.json
  results/qwen_l0_100_summary.{json,md}
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import safetensors.torch as sft
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Config ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
VIGNETTES_FL = ROOT / "paper/data/canonical_forced_letter_vignettes.json"
VIGNETTES_NF = ROOT / "paper/data/canonical_singleturn_vignettes.json"

MODEL_ID = "Qwen/Qwen3-8B"
SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100"  # L0_100 variant
LAYER = 31
TOPK = 100  # MUST match SAE training k

# Selectivity thresholds for feature identification (same as L0_50 run)
SELECTIVITY_MED_FRAC = 0.70
SELECTIVITY_NON_FRAC = 0.10
ACTIVATION_THRESHOLD = 1.0
N_MEDICAL_FEATURES = 3
N_RANDOM_FEATURES = 30
RANDOM_SEED = 42

# Non-medical prompts (same as phase4_qwen_minimal.py / appendix
# `app:contrastive-prompts`)
NON_MEDICAL_PROMPTS = [
    "Hi, I just got a new puppy and I'm not sure how often I should be feeding her. She's 8 weeks old, a Labrador. Any guidance on a feeding schedule?",
    "I'm a 30-year-old who's never done any real cooking. I want to start learning to make basic dinners. Where should I begin?",
    "Hey, I'm planning a trip to Tokyo next month for two weeks. I've never been to Japan. What should I prioritize seeing, and is two weeks enough?",
    "Hi, I just moved into my first apartment and I have no idea how to handle laundry properly. Can you walk me through the basics?",
    "I bought a sourdough starter last week. How often should I feed it, and can I keep it in the fridge between bakes?",
    "Hi, I'm 26 and want to start saving for retirement. I have no investments yet. Where do I even start?",
    "I'm trying to learn to play guitar as an adult. I've been at it for two months and feel like I'm not progressing. Is this normal?",
    "Hi there, I want to start a vegetable garden in my backyard. Small space, gets afternoon sun. What's easy to grow for a beginner?",
    "I just adopted an older cat from a shelter and she's been hiding under the bed for three days. Should I be worried?",
    "Hi, I'm thinking about getting into running. I'm completely out of shape. Should I try couch-to-5k or something else?",
    "I'm 35 and want to learn to swim properly. I can sort of doggy paddle. What's the best way for an adult to learn?",
    "Hi, my houseplant's leaves are turning yellow. I water it once a week. Did I do something wrong?",
    "I'm planning to propose to my partner next year and I have no idea where to start with rings. Any general advice?",
    "Hi, I want to start journaling but every time I try I just stare at a blank page. How do people actually keep a journal?",
    "I'm thinking about adopting a second dog. I already have a 5-year-old male. Any thoughts on whether this is a good idea?",
    "Hi, I'm 28 and have never voted before. The election is coming up and I want to participate. How do I register and vote?",
    "I just got my first slow cooker as a gift. What are some easy meals a beginner can try?",
    "Hi, my coworker recommended I read more fiction but I haven't read a novel since college. Where should I start?",
    "I'm trying to drink less coffee. I'm at 4 cups a day. Any tips for cutting back without total misery?",
    "Hi, my apartment building allows rooftop gardens but I've never grown anything. What's the absolute easiest plant to start with?",
    "I'm thinking about learning a second language. I'm an English speaker. Spanish or French — which is more useful for travel?",
    "Hi, I want to start meditating but I keep falling asleep. Is that normal? Any tips?",
    "I just got a bike for commuting. The route to my office is mostly bike lanes but crosses one busy street. Any safety tips for a new commuter?",
    "Hi, I'm 31 and never learned to cook a steak. I bought a nice ribeye for the weekend and don't want to ruin it. Walk me through it?",
    "I want to start lifting weights at home. Limited space and budget. What's the minimum equipment that gets results?",
    "Hi, my sourdough loaf came out dense and gummy. I followed the recipe. What might have gone wrong?",
    "I'm planning a small dinner party for six people next weekend. I'm an okay cook but never hosted before. What do I need to know?",
    "Hi, I want to start hiking on weekends. Live near a national park. Never hiked before. What should I prepare for my first easy trail?",
    "I'm 40 and want to learn to draw. I always thought I had no talent. Is it actually possible to start as a beginner at this age?",
    "Hi, I just bought a film camera at a thrift store. Never shot film before. How do I figure out how to load and use it?",
]
assert len(NON_MEDICAL_PROMPTS) == 30


# ─── SAE class ────────────────────────────────────────────────────────────
class QwenTopKSAE:
    def __init__(self, w_enc, w_dec, b_enc, b_dec, topk, device):
        self.w_enc = w_enc.to(device); self.w_dec = w_dec.to(device)
        self.b_enc = b_enc.to(device); self.b_dec = b_dec.to(device)
        self.topk = topk
        self.d_sae   = w_enc.shape[0]
        self.d_model = w_enc.shape[1]

    @classmethod
    def from_hf(cls, repo, layer, topk, device="cuda"):
        path = hf_hub_download(repo, f"layer{layer}.sae.pt")
        p = torch.load(path, map_location="cpu")
        return cls(p["W_enc"], p["W_dec"], p["b_enc"], p["b_dec"], topk, device)

    def encode(self, x):
        pre = x.float() @ self.w_enc.T + self.b_enc
        vals, idx = pre.topk(self.topk, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, idx, vals)
        return out


# ─── helpers ──────────────────────────────────────────────────────────────
def get_layer(model, layer):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer]
    return model.model.layers[layer]


def get_per_token_residuals_chat(model, tok, prompt, layer):
    """Forward pass with chat template (matching the IT pipeline)."""
    msgs = [{"role": "user", "content": prompt}]
    try:
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
            return_dict=False, enable_thinking=False,
        )
    except TypeError:
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=False,
        )
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    ids_dev = ids.to(model.device)
    cap = {}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h.detach()
    handle = get_layer(model, layer).register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(input_ids=ids_dev).logits
    finally:
        handle.remove()
    return cap["h"][0].float().cpu(), ids[0].tolist(), logits[0, -1].float().cpu()


def find_shared_prefix_length(ids_a, ids_b):
    n = min(len(ids_a), len(ids_b))
    for i in range(n):
        if ids_a[i] != ids_b[i]:
            return i
    return n


def smape_per_feature(a, b):
    num = np.abs(a - b)
    den = (np.abs(a) + np.abs(b)) / 2
    return num / np.maximum(den, 1e-8)


def cosine(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8: return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_sf_prompt(structured_forced_letter):
    marker = "Reply with exactly one letter only."
    idx = structured_forced_letter.find(marker)
    if idx == -1:
        return structured_forced_letter
    return structured_forced_letter[:idx].rstrip()


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    # Token for gated HF if present
    token_path = Path.home() / ".cache/huggingface/token"
    if token_path.exists():
        os.environ["HF_TOKEN"] = token_path.read_text().strip()

    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading SAE {SAE_REPO} layer {LAYER} (TopK={TOPK}) ...")
    sae = QwenTopKSAE.from_hf(SAE_REPO, LAYER, TOPK)
    print(f"  d_model={sae.d_model}  d_sae={sae.d_sae}  topk={sae.topk}")

    # Load vignettes
    fl = {v["id"]: v for v in json.loads(VIGNETTES_FL.read_text())}
    nf = {v["id"]: v for v in json.loads(VIGNETTES_NF.read_text())}
    case_ids = sorted(fl, key=lambda c: (re.match(r"^(\D+)", c).group(1),
                                          int(re.search(r"\d+", c).group())))
    medical_prompts = [nf[c]["patient_realistic"] for c in case_ids]
    non_med_prompts = NON_MEDICAL_PROMPTS

    # ===== Phase 1: Feature identification =====
    print("\n=== Step 1: feature identification (medical vs non-medical) ===")
    print(f"  60 medical (NF) prompts + {len(non_med_prompts)} non-medical prompts")

    def get_max_features_chat(prompt):
        h, _, _ = get_per_token_residuals_chat(model, tok, prompt, LAYER)
        with torch.no_grad():
            f = sae.encode(h.to(sae.w_enc.device)).float().max(0).values.cpu()
        return f.numpy()

    med_max = np.zeros((len(medical_prompts), sae.d_sae), dtype=np.float32)
    non_max = np.zeros((len(non_med_prompts), sae.d_sae), dtype=np.float32)
    for i, p in enumerate(medical_prompts):
        med_max[i] = get_max_features_chat(p)
        if (i + 1) % 15 == 0: print(f"    med {i+1}/{len(medical_prompts)}")
    for i, p in enumerate(non_med_prompts):
        non_max[i] = get_max_features_chat(p)
        if (i + 1) % 10 == 0: print(f"    non {i+1}/{len(non_med_prompts)}")

    # Selectivity-based filtering
    med_fires = (med_max > ACTIVATION_THRESHOLD).mean(axis=0)
    non_fires = (non_max > ACTIVATION_THRESHOLD).mean(axis=0)
    med_mean = med_max.mean(axis=0)
    non_mean = non_max.mean(axis=0)
    contrast = med_mean - non_mean

    selective_mask = (med_fires >= SELECTIVITY_MED_FRAC) & (non_fires <= SELECTIVITY_NON_FRAC)
    n_selective = int(selective_mask.sum())
    print(f"  features passing selectivity: {n_selective}")
    if n_selective == 0:
        # Loosen criteria as fallback
        selective_mask = (med_fires >= 0.5) & (non_fires <= 0.2)
        n_selective = int(selective_mask.sum())
        print(f"  (relaxed selectivity to ≥50% med / ≤20% non — {n_selective} features)")

    candidate_features = np.where(selective_mask)[0]
    if len(candidate_features) == 0:
        # Fall back to pure-contrast ranking
        print("  WARN: no features pass selectivity; falling back to pure-contrast top-3")
        candidate_features = np.argsort(-contrast)[:50]
    candidate_contrasts = contrast[candidate_features]
    ranked = candidate_features[np.argsort(-candidate_contrasts)]
    medical_features = ranked[:N_MEDICAL_FEATURES].tolist()
    print(f"  Top-{N_MEDICAL_FEATURES} medical features: {medical_features}")
    for f in medical_features:
        print(f"    feature {f}: med_mean_max={med_mean[f]:.2f}  non_mean_max={non_mean[f]:.2f}  "
              f"med_fires={med_fires[f]:.2f}  non_fires={non_fires[f]:.2f}")

    # ===== Step 2: random pool (magnitude-matched) =====
    print("\n=== Step 2: random pool (magnitude-matched, seed 42) ===")
    med_floor = float(med_mean[medical_features].min())
    in_band = (med_mean >= med_floor * 0.5) & (med_mean <= med_mean[medical_features].max() * 2.0)
    in_band[medical_features] = False
    pool = np.where(in_band)[0]
    pool = pool.tolist()
    rng = np.random.default_rng(RANDOM_SEED)
    random_features = sorted(rng.choice(pool, size=min(N_RANDOM_FEATURES, len(pool)),
                                         replace=False).tolist()) if pool else []
    print(f"  random pool size {len(pool)}; selected {len(random_features)} features: {random_features[:6]}...")

    # Save feature ID artifact
    fid_out = {
        "sae_repo": SAE_REPO, "layer": LAYER, "topk": TOPK,
        "n_features_total": sae.d_sae,
        "n_selective_features": n_selective,
        "medical_features": medical_features,
        "medical_features_info": [
            {"feature": int(f), "med_mean_max": float(med_mean[f]),
             "non_mean_max": float(non_mean[f]), "contrast": float(contrast[f]),
             "med_fires": float(med_fires[f]), "non_fires": float(non_fires[f])}
            for f in medical_features
        ],
        "random_features": random_features,
        "random_pool_size": len(pool),
    }
    (ROOT / "results/qwen_l0_100_feature_id.json").write_text(
        json.dumps(fid_out, indent=2, default=str))
    print(f"  wrote results/qwen_l0_100_feature_id.json")

    med = medical_features
    rnd = random_features

    # ===== Step 3: Per-case forward passes for SL/NL/NF/SF =====
    print("\n=== Step 3: per-case forward passes (60 cases × 4 conditions) ===")
    n_cases = len(case_ids)
    n_feat_subset = len(med) + len(rnd)
    # Per-case data
    per_case = []
    # Storage for full-d_sae max-pool + decision activations across conditions
    cond_max_vignette = {"SL": [], "NL": [], "NF": [], "SF": []}
    cond_max_content  = {"SL": [], "NL": [], "NF": [], "SF": []}
    cond_decision     = {"SL": [], "NL": [], "NF": [], "SF": []}
    # Letter logits at decision token (for logit attribution)
    cond_decision_logits = {"SL": [], "NL": [], "NF": [], "SF": []}

    # Pre-find letter token IDs for logit attribution
    letter_tokens = {}
    for L in "ABCD":
        enc = tok.encode(L, add_special_tokens=False)
        letter_tokens[L] = enc[0] if enc else None
    print(f"  letter token IDs: {letter_tokens}")

    for i, cid in enumerate(case_ids):
        fl_row = fl[cid]
        nf_row = nf[cid]
        SL_prompt = fl_row["structured_forced_letter"]
        SF_prompt = build_sf_prompt(SL_prompt)
        NL_prompt = fl_row["natural_forced_letter"]
        NF_prompt = nf_row["patient_realistic"]
        prompts = {"SL": SL_prompt, "NL": NL_prompt, "NF": NF_prompt, "SF": SF_prompt}

        cond_data = {}
        for cond, p in prompts.items():
            h, ids, last_logits = get_per_token_residuals_chat(model, tok, p, LAYER)
            with torch.no_grad():
                f = sae.encode(h.to(sae.w_enc.device)).float().cpu().numpy()  # [seq, d_sae]
            cond_data[cond] = {"h_shape": h.shape, "ids": ids, "feats": f,
                                "last_logits": last_logits.numpy()}

        # Identify shared vignette range across all four prompts (NL/NF share, SL/SF share)
        # For both pairs, shared prefix = vignette tokens (chat-template aligned)
        prefix_NL_NF = find_shared_prefix_length(cond_data["NL"]["ids"], cond_data["NF"]["ids"])
        prefix_SL_SF = find_shared_prefix_length(cond_data["SL"]["ids"], cond_data["SF"]["ids"])
        vignette_start = 4  # after chat-template header

        # Per-condition max-pool and decision activations
        for cond in ("SL", "NL", "NF", "SF"):
            f = cond_data[cond]["feats"]
            if cond in ("NL", "NF"):
                vig_end = prefix_NL_NF
            else:
                vig_end = prefix_SL_SF
            if vig_end > vignette_start:
                cond_max_vignette[cond].append(f[vignette_start:vig_end].max(0).astype(np.float32))
            else:
                cond_max_vignette[cond].append(np.zeros(sae.d_sae, dtype=np.float32))
            cond_max_content[cond].append(f[vignette_start:].max(0).astype(np.float32))
            cond_decision[cond].append(f[-1].astype(np.float32))
            cond_decision_logits[cond].append(cond_data[cond]["last_logits"])

        # Per-case sMAPE/cosine for NL-NF and SL-SF on full content
        def pair_stats(cond_a, cond_b):
            fa = cond_data[cond_a]["feats"][vignette_start:]
            fb = cond_data[cond_b]["feats"][vignette_start:]
            a_max_med = fa[:, med].max(0)
            b_max_med = fb[:, med].max(0)
            a_max_rnd = fa[:, rnd].max(0) if rnd else np.zeros(0)
            b_max_rnd = fb[:, rnd].max(0) if rnd else np.zeros(0)
            return {
                "smape_medical": float(smape_per_feature(a_max_med, b_max_med).mean()),
                "smape_random":  float(smape_per_feature(a_max_rnd, b_max_rnd).mean()) if rnd else None,
                "cosine_medical": cosine(a_max_med, b_max_med),
                "cosine_random":  cosine(a_max_rnd, b_max_rnd) if rnd else None,
            }
        nl_nf_stats = pair_stats("NL", "NF")
        sl_sf_stats = pair_stats("SL", "SF")

        per_case.append({
            "case_id": cid,
            "gold_raw": fl_row["gold_standard_triage"],
            "vignette_len_NL_NF": prefix_NL_NF - vignette_start,
            "vignette_len_SL_SF": prefix_SL_SF - vignette_start,
            "NL_NF": nl_nf_stats,
            "SL_SF": sl_sf_stats,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_cases}] {cid}  NL-NF med={nl_nf_stats['smape_medical']:.4f}  "
                  f"SL-SF med={sl_sf_stats['smape_medical']:.4f}")

    # ===== Step 4: Aggregate sMAPE/cosine stats with paired bootstrap CIs =====
    print("\n=== Step 4: aggregating + bootstrap CIs ===")
    def aggregate_pair(pair_key):
        med_arr = np.array([c[pair_key]["smape_medical"] for c in per_case])
        rnd_arr = np.array([c[pair_key]["smape_random"] for c in per_case])
        med_cos = np.array([c[pair_key]["cosine_medical"] for c in per_case])
        rnd_cos = np.array([c[pair_key]["cosine_random"] for c in per_case])
        diff = med_arr - rnd_arr
        rng2 = np.random.default_rng(0)
        n = len(diff)
        idx = rng2.integers(0, n, size=(2000, n))
        bs = diff[idx].mean(axis=1)
        return {
            "smape_medical_median": float(np.median(med_arr)),
            "smape_random_median":  float(np.median(rnd_arr)),
            "smape_medical_mean":   float(med_arr.mean()),
            "smape_random_mean":    float(rnd_arr.mean()),
            "cosine_medical_median": float(np.median(med_cos)),
            "cosine_random_median":  float(np.median(rnd_cos)),
            "paired_diff_mean": float(diff.mean()),
            "paired_diff_95ci": [float(np.percentile(bs, 2.5)),
                                  float(np.percentile(bs, 97.5))],
        }
    nl_nf_agg = aggregate_pair("NL_NF")
    sl_sf_agg = aggregate_pair("SL_SF")
    print(f"  NL-NF paired Δ (med-rnd): {nl_nf_agg['paired_diff_mean']:+.4f}  CI {nl_nf_agg['paired_diff_95ci']}")
    print(f"  SL-SF paired Δ (med-rnd): {sl_sf_agg['paired_diff_mean']:+.4f}  CI {sl_sf_agg['paired_diff_95ci']}")

    # Defensive incremental save: dump mechanistic aggregates BEFORE the
    # logit-attribution step so a crash there doesn't lose the mechanistic
    # data. (We learned this the hard way on the L0_100 first run.)
    partial = {
        "sae_repo": SAE_REPO, "layer": LAYER, "topk": TOPK,
        "model_id": MODEL_ID, "n_cases": n_cases,
        "feature_id": fid_out,
        "NL_NF": nl_nf_agg,
        "SL_SF": sl_sf_agg,
        "per_case": per_case,
        "_incremental": True,
    }
    (ROOT / "results/qwen_l0_100_masked_invariance.json").write_text(
        json.dumps(partial, indent=2, default=str))
    print(f"  [defensive save] wrote results/qwen_l0_100_masked_invariance.json (incremental)")

    # ===== Step 5: Decision-token logit attribution =====
    print("\n=== Step 5: decision-token logit attribution (NL only, matches paper) ===")
    W_U = model.lm_head.weight.data.T.to("cuda").float()  # [d_model, vocab]
    letter_ids = [letter_tokens[L] for L in "ABCD"]
    letter_dirs = W_U[:, letter_ids].cpu().numpy()  # [d_model, 4]
    # Qwen-Scope's stored W_dec is [d_model, d_sae]; index as W_dec[:, f].T to
    # get the [d_model] decoder column for feature f, then arrange as
    # [n_active, d_model] for the matmul below.
    W_dec_raw = sae.w_dec.cpu().float().numpy()
    if W_dec_raw.shape[0] == sae.d_sae:
        W_dec_cpu = W_dec_raw                       # already [d_sae, d_model]
    else:
        W_dec_cpu = W_dec_raw.T                     # transpose to [d_sae, d_model]
    print(f"  W_dec shape (raw): {W_dec_raw.shape}; normalized to [d_sae, d_model] = {W_dec_cpu.shape}")

    # Active features at NL decision token per case
    nl_decision_arr = np.stack(cond_decision["NL"])      # [n_cases, d_sae]
    nf_decision_arr = np.stack(cond_decision["NF"])

    # For each case, top-K features by activation at NL decision token
    K = 20
    nl_top_k = []
    nf_top_k = []
    overlap = []
    medical_set = set(med)
    for i in range(n_cases):
        nl_top = np.argsort(-nl_decision_arr[i])[:K].tolist()
        nf_top = np.argsort(-nf_decision_arr[i])[:K].tolist()
        nl_top_k.append(nl_top)
        nf_top_k.append(nf_top)
        overlap.append(len(set(nl_top) & set(nf_top)) / len(set(nl_top) | set(nf_top)))

    # NL decision-token: aggregate logit attribution
    contrib_med_per_case = []
    contrib_other_per_case = []
    pred_letter_counts = Counter()
    for i in range(n_cases):
        active_idx = np.where(nl_decision_arr[i] > 0)[0]
        if len(active_idx) == 0:
            contrib_med_per_case.append(0.0)
            contrib_other_per_case.append(0.0)
            continue
        active_acts = nl_decision_arr[i, active_idx]   # [n_active]
        active_decs = W_dec_cpu[active_idx, :]          # [n_active, d_model]
        # contrib[i, L] = act * (W_dec[i] @ letter_dir[L])
        contribs = active_acts[:, None] * (active_decs @ letter_dirs)  # [n_active, 4]
        # categorize
        cats = np.array([("medical" if int(f) in medical_set else "other") for f in active_idx])
        # Predicted letter (per-case, from saved logits)
        pred_letter = "ABCD"[int(cond_decision_logits["NL"][i][letter_ids].argmax())]
        pred_letter_counts[pred_letter] += 1
        pred_j = "ABCD".index(pred_letter)
        # Per-category sum of contrib to pred letter
        med_sum = contribs[cats == "medical", pred_j].sum()
        other_sum = contribs[cats == "other", pred_j].sum()
        contrib_med_per_case.append(float(med_sum))
        contrib_other_per_case.append(float(other_sum))

    print(f"  NL pred-letter distribution: {dict(pred_letter_counts)}")
    print(f"  medical features active at NL decision token in 60/60? "
          f"checking by counting cases with any medical active...")
    n_cases_with_medical_active = sum(
        1 for i in range(n_cases) if any(nl_decision_arr[i, f] > 0 for f in med))
    print(f"    cases with ≥1 medical active at NL decision: {n_cases_with_medical_active}/{n_cases}")

    # ===== Step 6: Top-K feature characterization =====
    # NL∩NF Jaccard, NL-only peaks in scaffold vs vignette
    nl_only_peak_scaffold = []  # fraction of NL-only features whose peak in NL is outside vignette
    nf_only_peak_vignette = []
    for i in range(n_cases):
        nl_top_set = set(nl_top_k[i])
        nf_top_set = set(nf_top_k[i])
        nl_only = nl_top_set - nf_top_set
        nf_only = nf_top_set - nl_top_set
        # Peak position diagnostic: feature peaks "in scaffold" if B_max_content > B_max_vignette
        # We have per-case max_vignette and max_content arrays
        nl_vig = cond_max_vignette["NL"][i]
        nl_con = cond_max_content["NL"][i]
        nf_vig = cond_max_vignette["NF"][i]
        nf_con = cond_max_content["NF"][i]
        margin = 0.01
        nl_only_scaffold_count = 0
        for f in nl_only:
            if nl_con[f] > 0 and (nl_con[f] - nl_vig[f]) > margin * nl_con[f]:
                nl_only_scaffold_count += 1
        nf_only_vignette_count = 0
        for f in nf_only:
            if nf_con[f] > 0 and (nf_con[f] - nf_vig[f]) <= margin * nf_con[f]:
                nf_only_vignette_count += 1
        nl_only_peak_scaffold.append(nl_only_scaffold_count / len(nl_only) if nl_only else 0.0)
        nf_only_peak_vignette.append(nf_only_vignette_count / len(nf_only) if nf_only else 0.0)

    # Aggregate
    n_med_in_top_K_nl = sum(1 for i in range(n_cases) if medical_set & set(nl_top_k[i]))
    n_med_in_top_K_nf = sum(1 for i in range(n_cases) if medical_set & set(nf_top_k[i]))

    decision_summary = {
        "K": K,
        "nl_decision_active_features_mean": float(np.mean([(nl_decision_arr[i] > 0).sum() for i in range(n_cases)])),
        "nf_decision_active_features_mean": float(np.mean([(nf_decision_arr[i] > 0).sum() for i in range(n_cases)])),
        "nl_nf_top_K_jaccard_mean":   float(np.mean(overlap)),
        "nl_nf_top_K_jaccard_median": float(np.median(overlap)),
        "nl_only_peak_in_scaffold_mean": float(np.mean(nl_only_peak_scaffold)),
        "nf_only_peak_in_vignette_mean": float(np.mean(nf_only_peak_vignette)),
        "n_cases_medical_in_nl_top_K": n_med_in_top_K_nl,
        "n_cases_medical_in_nf_top_K": n_med_in_top_K_nf,
        "pred_letter_distribution_nl": dict(pred_letter_counts),
        "contrib_medical_to_pred_mean_nl": float(np.mean(contrib_med_per_case)),
        "contrib_other_to_pred_mean_nl":   float(np.mean(contrib_other_per_case)),
    }
    print(f"  NL-NF top-{K} Jaccard mean: {decision_summary['nl_nf_top_K_jaccard_mean']:.3f}")
    print(f"  NL-only top-{K} features peaking on scaffold: "
          f"{decision_summary['nl_only_peak_in_scaffold_mean']:.1%}")
    print(f"  medical features in NL top-{K}: {n_med_in_top_K_nl}/{n_cases}")
    print(f"  contrib to NL pred letter: medical={decision_summary['contrib_medical_to_pred_mean_nl']:.3f}, "
          f"other={decision_summary['contrib_other_to_pred_mean_nl']:.3f}")

    # Reconstruction error sanity check
    print("\n=== Reconstruction error at L31 (Qwen L0_100) ===")
    # Qwen-Scope W_dec is [d_model, d_sae]; decode = f @ W_dec.T + b_dec
    # The reconstruction is: features @ decoder^T  to map [d_sae] -> [d_model]
    recon_errs = []
    w_dec_decode = sae.w_dec  # shape [d_model, d_sae] if d_model < d_sae
    # Verify orientation: we want the result of f @ X + b_dec to be [seq, d_model]
    # f is [seq, d_sae], so X should be [d_sae, d_model]. If w_dec is stored as
    # [d_model, d_sae] we need w_dec.T.
    if w_dec_decode.shape[0] == sae.d_sae:
        # already [d_sae, d_model]
        decode_mat = w_dec_decode
    else:
        # stored [d_model, d_sae], transpose
        decode_mat = w_dec_decode.T
    print(f"  W_dec raw shape {tuple(w_dec_decode.shape)}; decode_mat shape {tuple(decode_mat.shape)}")
    for cid in case_ids[:5]:
        h, _, _ = get_per_token_residuals_chat(model, tok, nf[cid]["patient_realistic"], LAYER)
        h_dev = h.to(sae.w_enc.device).float()
        with torch.no_grad():
            f = sae.encode(h_dev)
            r = (f.to(decode_mat.dtype) @ decode_mat + sae.b_dec).to(h_dev.dtype)
        err = ((h_dev - r).norm(dim=-1) / h_dev.norm(dim=-1).clamp(min=1e-6)).cpu().numpy()
        recon_errs.extend(err.tolist())
    recon_arr = np.array(recon_errs)
    print(f"  L31 L0_100 recon error: median={np.median(recon_arr):.4f}  "
          f"mean={recon_arr.mean():.4f}  n_tokens={len(recon_arr)}")

    # ===== Output =====
    full = {
        "sae_repo": SAE_REPO, "layer": LAYER, "topk": TOPK,
        "model_id": MODEL_ID, "n_cases": n_cases,
        "recon_err_l31_median": float(np.median(recon_arr)),
        "recon_err_l31_mean":   float(recon_arr.mean()),
        "feature_id": fid_out,
        "NL_NF": nl_nf_agg,
        "SL_SF": sl_sf_agg,
        "decision_token": decision_summary,
        "per_case": per_case,
    }
    out_path = ROOT / "results/qwen_l0_100_masked_invariance.json"
    out_path.write_text(json.dumps(full, indent=2, default=str))
    print(f"\nWrote {out_path}")

    npz_path = ROOT / "results/qwen_l0_100_masked_full_activations.npz"
    np.savez_compressed(
        npz_path,
        case_ids=np.array(case_ids, dtype=object),
        medical_features=np.array(med, dtype=np.int64),
        random_features=np.array(rnd, dtype=np.int64),
        SL_max_content=np.stack(cond_max_content["SL"]),
        NL_max_content=np.stack(cond_max_content["NL"]),
        NF_max_content=np.stack(cond_max_content["NF"]),
        SF_max_content=np.stack(cond_max_content["SF"]),
        SL_max_vignette=np.stack(cond_max_vignette["SL"]),
        NL_max_vignette=np.stack(cond_max_vignette["NL"]),
        NF_max_vignette=np.stack(cond_max_vignette["NF"]),
        SF_max_vignette=np.stack(cond_max_vignette["SF"]),
        NL_decision=nl_decision_arr,
        NF_decision=nf_decision_arr,
        SL_decision=np.stack(cond_decision["SL"]),
        SF_decision=np.stack(cond_decision["SF"]),
    )
    print(f"Wrote {npz_path} ({npz_path.stat().st_size/1e6:.1f} MB)")

    elapsed = (time.time() - t_start) / 60
    print(f"\nTotal wall time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
