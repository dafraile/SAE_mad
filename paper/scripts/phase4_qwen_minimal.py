"""
Phase 4 — Truncated cross-family validation: Qwen3-8B + Qwen Scope SAE at L31.

Minimum-viable cross-family experiment to forestall the "single family"
reviewer concern. We accept the SAE's intrinsic ~40% reconstruction error
(Qwen Scope's k=50 TopK choice) and ask:

  Despite the noisier SAE, do Qwen Scope's medical-content features at L31
  show the same magnitude-invariance pattern across B and D that Gemma
  Scope's features show at the deepest layer?

Setup:
  - Model: Qwen3-8B (post-trained); B and D prompts fed as raw text
    (no chat template).
  - SAE: Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50, layer 31 only.
  - 60 paper-canonical cases, B = natural_forced_letter, D = patient_realistic.
  - Feature identification: medical-vs-non-medical contrastive on the 60
    patient_realistic prompts vs 30 non-medical prompts (same as Phase 3),
    score = mean_max(med) - mean_max(non) under firing-reliability filter,
    take top 3.
  - Phase 1b magnitude-matched mod-index (medical vs 30 random features in
    same magnitude band).
  - Phase 2b max-pool projection (medical-feature alignment with B-D
    direction).

Output: results/phase4_qwen_L31.json
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"
SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
LAYER = 31
TOPK = 50
N_RANDOM_FEATURES = 30
RANDOM_SEED = 42
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
OUT_PATH = Path("results/phase4_qwen_L31.json")
FORCED_LETTER_MARKER = "Reply with exactly one letter only"

# Same 30 non-medical prompts as Phase 3
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


def parse_gold(g):
    return sorted(set(re.findall(r"[ABCD]", g.upper())))


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
            "B_prompt": fl_by_id[cid]["natural_forced_letter"],
            "D_prompt": st_by_id[cid]["patient_realistic"],
            "gold_letters": parse_gold(fl_by_id[cid]["gold_standard_triage"]),
        })
    return cases


def get_target(model, layer):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer]
    return model.model.layers[layer]


def collect_residual_raw_text(model, tok, text, layer):
    """Forward a raw text (no chat template), capture residual at given layer.
    Returns [seq, d_model]."""
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    captured = {}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()
    handle = get_target(model, layer).register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        handle.remove()
    return captured["h"][0]  # [seq, d_model]


class TopKSAE:
    def __init__(self, W_enc, W_dec, b_enc, b_dec, topk, device):
        self.W_enc = W_enc.to(device)             # (d_sae, d_model)
        self.W_dec = W_dec.to(device)
        self.b_enc = b_enc.to(device)
        self.b_dec = b_dec.to(device)
        self.topk = topk
        self.d_sae = W_enc.shape[0]
        self.d_model = W_enc.shape[1]

    @classmethod
    def from_hf(cls, repo, layer, topk=TOPK, device="cuda"):
        path = hf_hub_download(repo, f"layer{layer}.sae.pt")
        sae = torch.load(path, map_location="cpu")
        return cls(sae["W_enc"], sae["W_dec"], sae["b_enc"], sae["b_dec"], topk, device)

    def encode(self, x):
        pre = x.float() @ self.W_enc.T + self.b_enc
        topk_vals, topk_idx = pre.topk(self.topk, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    def decode(self, features):
        return features @ self.W_dec.T + self.b_dec


def find_marker_truncation(prompt, tok, marker):
    """Return token index in B's tokenized prompt where forced-letter starts.
    Compares B's tokens to B's prefix-only tokens to find the divergence point.
    Since the prefix is identical to D, this also = number of D content tokens.
    """
    if marker not in prompt:
        return None
    prefix = prompt.split(marker, 1)[0].rstrip()
    b_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    p_ids = tok(prefix, return_tensors="pt").input_ids[0].tolist()
    # Find the position where the two diverge
    n = min(len(b_ids), len(p_ids))
    for i in range(n):
        if b_ids[i] != p_ids[i]:
            return i
    return n


def boot_ci(xs, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.array([x for x in xs if x == x])
    if len(arr) == 0:
        return None, None
    res = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(1)
    return float(np.quantile(res, 0.025)), float(np.quantile(res, 0.975))


def main():
    cases = build_cases()
    assert len(cases) == 60

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    print(f"Loading SAE for L{LAYER}...")
    sae = TopKSAE.from_hf(SAE_REPO, LAYER)
    print(f"  d_model={sae.d_model} d_sae={sae.d_sae} topk={sae.topk}")

    # ==== Step 1: Feature identification ====
    print("\n=== Feature identification (medical vs non-medical) ===")
    medical_prompts = [c["D_prompt"] for c in cases]  # patient_realistic = clinical content
    non_med_prompts = NON_MEDICAL_PROMPTS

    def get_max_features(text):
        residual = collect_residual_raw_text(model, tok, text, LAYER)
        residual_dev = residual.to(sae.W_enc.dtype).to(sae.W_enc.device)
        with torch.no_grad():
            feats = sae.encode(residual_dev)
        return feats.max(0).values.float().cpu()

    med_max = torch.zeros(len(medical_prompts), sae.d_sae)
    for i, p in enumerate(medical_prompts):
        med_max[i] = get_max_features(p)
        if (i + 1) % 15 == 0: print(f"  med {i+1}/60")
    non_max = torch.zeros(len(non_med_prompts), sae.d_sae)
    for i, p in enumerate(non_med_prompts):
        non_max[i] = get_max_features(p)
        if (i + 1) % 10 == 0: print(f"  non {i+1}/{len(non_med_prompts)}")

    med_mean = med_max.mean(0)
    non_mean = non_max.mean(0)
    fires_med = (med_max > 1.0).float().mean(0)
    fires_non = (non_max > 1.0).float().mean(0)
    score = med_mean - non_mean

    good = (fires_med >= 0.7) & (fires_non <= 0.1)
    ranked = torch.argsort(score * good.float(), descending=True).tolist()
    top_filtered = []
    for f in ranked[:20]:
        if not bool(good[f]): break
        top_filtered.append({
            "feature": int(f),
            "score": float(score[f]),
            "med_mean_max": float(med_mean[f]),
            "non_mean_max": float(non_mean[f]),
            "fires_med": float(fires_med[f]),
            "fires_non": float(fires_non[f]),
        })
    if not top_filtered:
        # fallback: take top-3 by score
        for f in torch.argsort(score, descending=True).tolist()[:3]:
            top_filtered.append({
                "feature": int(f), "score": float(score[f]),
                "med_mean_max": float(med_mean[f]),
                "non_mean_max": float(non_mean[f]),
                "fires_med": float(fires_med[f]),
                "fires_non": float(fires_non[f]),
                "note": "from unfiltered (filter pool empty)",
            })
    medical_features = [t["feature"] for t in top_filtered[:3]]
    print(f"  Filter-passing features: {int(good.sum())}")
    print(f"  Top 3: {medical_features}")
    for entry in top_filtered[:3]:
        print(f"    feat {entry['feature']:>5d}  score={entry['score']:>7.2f}  "
              f"med_max_mean={entry['med_mean_max']:>7.1f}  non_max_mean={entry['non_mean_max']:>5.2f}  "
              f"fires med/non = {entry['fires_med']:.2f}/{entry['fires_non']:.2f}")

    # ==== Step 2: Phase 1b — magnitude-matched mod-index ====
    print("\n=== Phase 1b — magnitude-matched mod-index ===")
    all_feat_B_mean = torch.zeros(len(cases), sae.d_sae)
    all_feat_D_mean = torch.zeros(len(cases), sae.d_sae)
    all_feat_B_max = torch.zeros(len(cases), sae.d_sae)
    all_feat_D_max = torch.zeros(len(cases), sae.d_sae)
    res_B_max = torch.zeros(len(cases), sae.d_model)
    res_D_max = torch.zeros(len(cases), sae.d_model)

    for i, c in enumerate(cases):
        # B prompt — full text including forced-letter block
        rB = collect_residual_raw_text(model, tok, c["B_prompt"], LAYER)
        rB_dev = rB.to(sae.W_enc.dtype).to(sae.W_enc.device)
        with torch.no_grad():
            fB = sae.encode(rB_dev).float()
        all_feat_B_mean[i] = fB.mean(0).cpu()
        all_feat_B_max[i] = fB.max(0).values.cpu()
        res_B_max[i] = rB.float().max(0).values.cpu()

        # D prompt
        rD = collect_residual_raw_text(model, tok, c["D_prompt"], LAYER)
        rD_dev = rD.to(sae.W_enc.dtype).to(sae.W_enc.device)
        with torch.no_grad():
            fD = sae.encode(rD_dev).float()
        all_feat_D_mean[i] = fD.mean(0).cpu()
        all_feat_D_max[i] = fD.max(0).values.cpu()
        res_D_max[i] = rD.float().max(0).values.cpu()

        if (i + 1) % 15 == 0:
            print(f"  case {i+1}/60 (B tokens={rB.shape[0]} D tokens={rD.shape[0]})")

    # Magnitude-matched random features
    ref = torch.cat([all_feat_B_mean, all_feat_D_mean], dim=0)
    mean_per_feat = ref.float().mean(0)
    med_means = mean_per_feat[medical_features]
    lo = MAG_LO * med_means.min().item()
    hi = MAG_HI * med_means.max().item()
    in_band = (mean_per_feat >= lo) & (mean_per_feat <= hi)
    in_band[medical_features] = False
    pool = in_band.nonzero(as_tuple=True)[0].tolist()
    rng = np.random.default_rng(RANDOM_SEED)
    if len(pool) >= N_RANDOM_FEATURES:
        random_features = sorted(rng.choice(pool, size=N_RANDOM_FEATURES, replace=False).tolist())
    else:
        random_features = sorted(pool)
    print(f"  Magnitude band [{lo:.2f}, {hi:.2f}], pool size {len(pool)}")
    print(f"  Random features (n={len(random_features)}): {random_features[:6]}...")

    per_case = []
    for i, c in enumerate(cases):
        entry = {"id": c["id"], "gold_letters": c["gold_letters"]}
        for kind, feats in [("medical", medical_features), ("random", random_features)]:
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

    med_mods = [c["medical_mod_index"] for c in per_case]
    rnd_mods = [c["random_mod_index"] for c in per_case]
    diff = [m - r for m, r in zip(med_mods, rnd_mods)]
    ci = boot_ci(diff)
    print(f"\n  Phase 1b summary at L{LAYER}:")
    print(f"    medical mod-index: {np.mean(med_mods):.3f}")
    print(f"    random mod-index:  {np.mean(rnd_mods):.3f}")
    print(f"    diff: {np.mean(diff):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")

    # ==== Step 3: Phase 2b max-pool projection ====
    print("\n=== Phase 2b max-pool projection ===")
    diff_max = (res_B_max - res_D_max).mean(0)
    diff_unit = diff_max / (diff_max.norm() + 1e-8)
    w_enc = sae.W_enc.float()              # (d_sae, d_model)
    w_enc_norms = w_enc.norm(dim=-1)        # (d_sae,)
    alignments = (w_enc @ diff_unit.to(w_enc.device)) / w_enc_norms.clamp(min=1e-8)
    alignments_cpu = alignments.cpu()
    abs_align = alignments_cpu.abs()
    sorted_idx = torch.argsort(abs_align, descending=True).tolist()
    medical_ranks = {f: sorted_idx.index(f) + 1 for f in medical_features}
    print(f"  Medical-feature ranks (max-pool, of {sae.d_sae}):")
    for f in medical_features:
        r = medical_ranks[f]
        print(f"    feat {f:>5d}: rank {r:>5d} ({100*r/sae.d_sae:>4.1f}%-ile) "
              f"|align|={abs_align[f]:.4f}")

    # Save
    out = {
        "model": MODEL_ID,
        "sae_repo": SAE_REPO,
        "layer": LAYER,
        "topk": TOPK,
        "n_features_total": sae.d_sae,
        "medical_features": medical_features,
        "medical_features_info": top_filtered[:3],
        "random_features": random_features,
        "magnitude_band": {"lo": lo, "hi": hi, "pool_size": len(pool)},
        "phase1b": {
            "medical_mod_index_mean": float(np.mean(med_mods)),
            "random_mod_index_mean": float(np.mean(rnd_mods)),
            "diff_mean": float(np.mean(diff)),
            "diff_ci_95": list(ci),
            "per_case": per_case,
        },
        "phase2b_max_pool": {
            "diff_norm": float(diff_max.norm()),
            "medical_ranks": {str(f): medical_ranks[f] for f in medical_features},
            "medical_alignments": {str(f): float(alignments_cpu[f]) for f in medical_features},
            "top10": [
                {"feature": int(f), "alignment": float(alignments_cpu[f]),
                 "is_medical": f in medical_features}
                for f in sorted_idx[:10]
            ],
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
