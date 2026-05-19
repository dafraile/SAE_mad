"""phase1b_sensitivity.py -- robustness check on Phase 1b modulation index
with feature-set size K ∈ {3, 5, 10, 20}.

Question: does our paper's 3-medical-feature result reflect a population-
level pattern (medical features more invariant than magnitude-matched
random features), or is it sensitive to the specific small subset?

Procedure:
  1. Identify the top-20 medical features at the target layer via the
     same contrastive procedure used in phase3_12b_feature_id.py (60
     medical patient_realistic prompts vs 30 non-medical prompts).
     Score = med_mean_max - non_mean_max. Filter to features that fire
     reliably on medical content (>=70%) and rarely on non-medical
     (<=10%).
  2. For each NL+NF prompt (60 × 2), encode the residual at the target
     layer through the SAE and record max-pooled activations on the
     content tokens for: top-20 medical features + 30 magnitude-matched
     random features (in the band [0.5×med_mean, 2×med_mean]).
  3. Compute the modulation index (NF→NL fold-change) per feature per
     case, then for K ∈ {3, 5, 10, 20} report:
       - mean medical modulation index (bootstrap mean + 95% CI)
       - mean random modulation index (matched K)
       - Δ = medical − random (lower is more invariant)

Usage:
  python3 paper/scripts/phase1b_sensitivity.py \\
      --model google/gemma-3-4b-it \\
      --sae-repo google/gemma-scope-2-4b-it \\
      --layer 29 \\
      --out results/phase1b_sensitivity_4b_L29.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import torch as sft
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SINGLETURN_PATH = ROOT / "nature_triage_expanded_replication/paper_faithful_replication/data/canonical_singleturn_vignettes.json"
FORCED_LETTER_PATH = ROOT / "nature_triage_expanded_replication/paper_faithful_forced_letter/data/canonical_forced_letter_vignettes.json"

# Same non-medical corpus as phase3_12b_feature_id.py.
NON_MEDICAL_PROMPTS = json.loads(
    (ROOT / "results/phase3_12b_features.json").read_text()
)["non_medical_prompts"]

FIRE_THRESHOLD = 1.0
END_OF_TURN_ID = 106  # Gemma chat-template <end_of_turn> token
RANDOM_SEED = 42
N_RANDOM = 30
BAND_LO, BAND_HI = 0.5, 2.0
N_BOOTSTRAP = 1000

KS_TO_TEST = [3, 5, 10, 20]


class JumpReLUSAE:
    def __init__(self, w_enc, w_dec, b_enc, b_dec, threshold, device):
        self.w_enc = w_enc.to(device)
        self.w_dec = w_dec.to(device)
        self.b_enc = b_enc.to(device)
        self.b_dec = b_dec.to(device)
        self.threshold = threshold.to(device)
        self.d_sae = w_enc.shape[1]

    @classmethod
    def from_hf(cls, repo, layer, width="16k", l0="medium", device="cuda"):
        sub = f"resid_post/layer_{layer}_width_{width}_l0_{l0}/params.safetensors"
        path = hf_hub_download(repo, sub)
        p = sft.load_file(path)
        return cls(p["w_enc"], p["w_dec"], p["b_enc"], p["b_dec"], p["threshold"], device)

    def encode(self, x):
        pre = x.float() @ self.w_enc + self.b_enc
        return pre * (pre > self.threshold).float()


def chat_ids(tok, prompt, device):
    out = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    if isinstance(out, dict): out = out["input_ids"]
    return out.to(device)


def target_layer(model, layer):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer]
    return model.model.layers[layer]


def get_residual_at_layer(model, tok, prompt, layer):
    """Forward pass; return residual stream output of `layer`, content tokens only.
    Skips the first 4 chat-template tokens and trims at <end_of_turn>."""
    ids = chat_ids(tok, prompt, model.device)
    h_box = {}
    def hook(_m, _i, out):
        h_box["h"] = (out[0] if isinstance(out, tuple) else out).detach()
    handle = target_layer(model, layer).register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        handle.remove()
    h = h_box["h"][0]  # [seq, d_model]
    raw = ids[0].tolist()
    try:
        eot = raw.index(END_OF_TURN_ID)
    except ValueError:
        eot = len(raw)
    return h[4:eot].contiguous() if eot > 4 else h[:eot]


def max_pool_features(model, tok, prompt, layer, sae):
    """Per-feature max activation over content tokens (for contrastive ID)."""
    h = get_residual_at_layer(model, tok, prompt, layer).to(sae.w_enc.dtype).to(sae.w_enc.device)
    with torch.no_grad():
        feats = sae.encode(h)
    return feats.max(0).values.float().cpu()


def mean_pool_features(model, tok, prompt, layer, sae):
    """Per-feature MEAN activation over content tokens (matches Phase 1b)."""
    h = get_residual_at_layer(model, tok, prompt, layer).to(sae.w_enc.dtype).to(sae.w_enc.device)
    with torch.no_grad():
        feats = sae.encode(h)
    return feats.mean(0).float().cpu()


def identify_top20_medical(model, tok, layer, sae):
    """Replicate the contrastive identification used by phase3_12b_feature_id.py."""
    medical_prompts = [v["patient_realistic"] for v in
                       json.loads(SINGLETURN_PATH.read_text())]
    assert len(medical_prompts) == 60

    d_sae = sae.d_sae
    med_max = torch.zeros(len(medical_prompts), d_sae)
    for i, p in enumerate(medical_prompts):
        med_max[i] = max_pool_features(model, tok, p, layer, sae)
        if (i+1) % 15 == 0: print(f"  contrastive med {i+1}/60")
    non_max = torch.zeros(len(NON_MEDICAL_PROMPTS), d_sae)
    for i, p in enumerate(NON_MEDICAL_PROMPTS):
        non_max[i] = max_pool_features(model, tok, p, layer, sae)
        if (i+1) % 10 == 0: print(f"  contrastive non {i+1}/30")

    med_mean = med_max.mean(0)
    non_mean = non_max.mean(0)
    fires_med = (med_max > FIRE_THRESHOLD).float().mean(0)
    fires_non = (non_max > FIRE_THRESHOLD).float().mean(0)
    score = med_mean - non_mean
    good = (fires_med >= 0.7) & (fires_non <= 0.1)
    ranked = torch.argsort(score * good.float(), descending=True).tolist()

    out = []
    for f in ranked[:20]:
        if not bool(good[f]): break
        out.append({
            "feature": int(f),
            "score": float(score[f]),
            "med_mean_max": float(med_mean[f]),
            "non_mean_max": float(non_mean[f]),
            "fires_med": float(fires_med[f]),
            "fires_non": float(fires_non[f]),
        })
    return out, med_mean.numpy()


def pick_random_magnitude_matched(ref_means, med_feats, n=N_RANDOM,
                                  mag_lo=BAND_LO, mag_hi=BAND_HI, seed=RANDOM_SEED):
    """MATCHES paper/scripts/phase1b_magnitude_matched.py::pick_random_magnitude_matched.

    ref_means: [d_sae] of MEAN-pooled feature activations averaged across
               the NL+NF corpus.
    med_feats: list of medical feature indices.
    band: [mag_lo * min(med_means), mag_hi * max(med_means)]
    Excludes the medical features themselves.
    """
    med_means = ref_means[med_feats]
    lo = mag_lo * float(med_means.min())
    hi = mag_hi * float(med_means.max())
    in_band = (ref_means >= lo) & (ref_means <= hi)
    candidates = [int(c) for c in np.where(in_band)[0] if c not in set(med_feats)]
    rng = np.random.default_rng(seed)
    if len(candidates) <= n:
        return candidates, {"pool_size": len(candidates), "lo": lo, "hi": hi}
    chosen = sorted(rng.choice(len(candidates), size=n, replace=False).tolist())
    return [candidates[i] for i in chosen], {"pool_size": len(candidates), "lo": lo, "hi": hi}


def build_cases():
    fl = json.loads(FORCED_LETTER_PATH.read_text())
    st = json.loads(SINGLETURN_PATH.read_text())
    fl_by = {v["id"]: v for v in fl}
    st_by = {v["id"]: v for v in st}
    def _key(s):
        m = re.match(r"^(\D+)(\d+)$", s)
        return (m.group(1), int(m.group(2))) if m else (s, 0)
    cases = []
    for cid in sorted(fl_by, key=_key):
        cases.append({
            "id": cid,
            "NL": fl_by[cid]["natural_forced_letter"],
            "NF": st_by[cid]["patient_realistic"],
        })
    return cases


def modulation(activ_nl, activ_nf):
    """Per-feature symmetric percentage difference, matching the original
    Phase 1b formula (paper/scripts/phase1b_magnitude_matched.py:261-263):
        mod = |NF - NL| / ((|NL| + |NF|) / 2)
    Bounded by 2; lower = more invariant. Inputs are mean-pooled activations.
    """
    num = np.abs(activ_nf - activ_nl)
    den = (np.abs(activ_nl) + np.abs(activ_nf)) / 2.0
    return num / np.maximum(den, 1e-8)


def boot_mean_ci(x, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0: return {"mean": None, "lo": None, "hi": None, "n": 0}
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
        "n": int(len(x)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sae-repo", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top20-cache", default=None,
                        help="If set, load top-20 medical features from this JSON "
                             "instead of re-running contrastive ID. Useful when the "
                             "ranking is already in results/phase3_12b_features.json")
    args = parser.parse_args()

    print(f"Loading {args.model} on cuda (bf16)")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading SAE {args.sae_repo} L{args.layer}")
    sae = JumpReLUSAE.from_hf(args.sae_repo, args.layer)

    # 1. Top-20 medical features
    if args.top20_cache:
        cached = json.loads(Path(args.top20_cache).read_text())
        # Try a few likely shapes
        top20 = (cached.get("by_layer", {}).get(str(args.layer), {}).get("top_filtered")
                 or cached.get("top_filtered")
                 or cached.get("top20"))
        if not top20:
            raise SystemExit(f"could not find top-20 in {args.top20_cache}")
        print(f"Loaded top-20 medical features from cache: {[t['feature'] for t in top20]}")
    else:
        print("Running contrastive medical-feature identification (60 med vs 30 non-med)...")
        top20, _ = identify_top20_medical(model, tok, args.layer, sae)
        print(f"Identified top-20 medical features: {[t['feature'] for t in top20]}")

    if len(top20) < 20:
        print(f"WARN: only {len(top20)} filter-passing features; padding with unfiltered")

    medical_ids = [t["feature"] for t in top20[:20]]

    # 2. Encode NL + NF prompts with MEAN-pool to get the data we need for
    #    both (a) the magnitude-matched random picker, and (b) the modulation
    #    analysis. We store the FULL 60 × 2 × d_sae array (mean-pooled) so the
    #    random picker uses the same magnitude scale as the modulation metric.
    cases = build_cases()
    assert len(cases) == 60
    d_sae = sae.d_sae
    nl_mean_full = np.zeros((60, d_sae), dtype=np.float32)
    nf_mean_full = np.zeros((60, d_sae), dtype=np.float32)
    print("Encoding NL+NF prompts (mean-pool over content tokens)...")
    for ci, c in enumerate(cases):
        nl_mean_full[ci] = mean_pool_features(model, tok, c["NL"], args.layer, sae).numpy()
        nf_mean_full[ci] = mean_pool_features(model, tok, c["NF"], args.layer, sae).numpy()
        if (ci+1) % 10 == 0: print(f"  encode case {ci+1}/60")

    # 3. Random-baseline picker uses MEAN of NL+NF mean-pool activations
    #    across all 60 cases (same as paper's phase1b_magnitude_matched).
    ref_means = ((nl_mean_full + nf_mean_full) / 2.0).mean(axis=0)  # [d_sae]
    random_ids, picker_info = pick_random_magnitude_matched(ref_means, medical_ids)
    print(f"Picked {len(random_ids)} random features for baseline; "
          f"band=[{picker_info['lo']:.2f}, {picker_info['hi']:.2f}]; "
          f"pool size {picker_info['pool_size']}")

    # 4. Slice out the medical + random columns for the modulation analysis
    all_ids = medical_ids + random_ids
    nl_acts = nl_mean_full[:, all_ids]
    nf_acts = nf_mean_full[:, all_ids]

    # 4. Modulation index per (case, feature)
    nl_med = nl_acts[:, :len(medical_ids)]
    nf_med = nf_acts[:, :len(medical_ids)]
    nl_rnd = nl_acts[:, len(medical_ids):]
    nf_rnd = nf_acts[:, len(medical_ids):]

    mod_med = modulation(nl_med, nf_med)  # [60, n_med]
    mod_rnd = modulation(nl_rnd, nf_rnd)  # [60, n_rnd]

    # 5. For each K, compute per-case mean modulation (across features), then
    #    bootstrap over the 60 cases. This matches the original Phase 1b
    #    aggregation: per-case scalar, then case-level bootstrap.
    results = {"K_results": {}}
    # Use first K medical features (top-K of the contrastive ranking).
    # Random picker returned a single set of 30 features; for each K we slice
    # the first K of the random ids (deterministic given RANDOM_SEED).
    for K in KS_TO_TEST:
        K_med = min(K, mod_med.shape[1])
        K_rnd = min(K, mod_rnd.shape[1])
        per_case_med = mod_med[:, :K_med].mean(axis=1)
        per_case_rnd = mod_rnd[:, :K_rnd].mean(axis=1)
        med_stats = boot_mean_ci(per_case_med)
        rnd_stats = boot_mean_ci(per_case_rnd)
        delta_stats = boot_mean_ci(per_case_med - per_case_rnd)
        results["K_results"][str(K)] = {
            "K_medical": K_med,
            "K_random":  K_rnd,
            "medical": med_stats,
            "random":  rnd_stats,
            "delta_paired_per_case": delta_stats,
        }
        sign = "<<<" if delta_stats["hi"] < 0 else (">>>" if delta_stats["lo"] > 0 else " ~ ")
        print(f"K={K_med}: med={med_stats['mean']:.4f} [{med_stats['lo']:.4f},{med_stats['hi']:.4f}]   "
              f"rnd={rnd_stats['mean']:.4f} [{rnd_stats['lo']:.4f},{rnd_stats['hi']:.4f}]   "
              f"Δ_paired={delta_stats['mean']:+.4f} [{delta_stats['lo']:+.4f},{delta_stats['hi']:+.4f}] {sign}")

    results["model"] = args.model
    results["sae_repo"] = args.sae_repo
    results["layer"] = args.layer
    results["medical_features"] = medical_ids
    results["random_features"] = random_ids
    results["top20_table"] = top20[:20]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
