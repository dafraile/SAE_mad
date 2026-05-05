"""
Phase 5 — Two follow-up analyses on Gemma 3 4B IT, L29:

  1. Top-token analysis of the format-direction features.
     In Phase 2b, the SAE features most aligned with the (NL − NF) max-pool
     direction at 4B L29 were 3833, 10012, 980, 9485, 755. We claim these
     are 'output-instruction' features that fire on the forced-letter
     instruction tokens. This phase verifies by finding the top-activating
     tokens across all 60 cases × {NL, NF} prompts.

     If the top tokens for these features are 'Reply', 'letter', 'A=', 'B=',
     'D=', 'exactly', 'Do not', etc., Version B sharpens substantially.

  2. Restricted-random-pool re-run of Phase 1b at L29.
     Phase 1b's sanity-check flagged that magnitude-matched random features
     were not restricted to those that fire on clinical content. Tightening
     this:
        pool = features with mean activation > 5 on the union of NL+NF
               clinical content (i.e., they actually fire here),
               magnitude-matched to medical features.
     Re-run the mod-index analysis at L29 with the stricter random pool.
     Headline cell: format_flipped stratum.

Output:
  results/phase5_top_tokens.json
  results/phase5_restricted_random.json
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
LAYER = 29

# Format-direction features identified in Phase 2b max-pool at 4B L29
FORMAT_DIRECTION_FEATURES = [3833, 10012, 980, 9485, 755]

# v3-validated medical features at L29
MEDICAL_FEATURES_L29 = [12570, 893, 12845]

N_RANDOM = 30
RANDOM_SEED = 42
TOP_TOKENS_PER_FEATURE = 25
CONTEXT_WINDOW = 8  # tokens of context around the top-activating token

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
OUT_TOPTOKENS = Path("results/phase5_top_tokens.json")
OUT_RESTRICTED = Path("results/phase5_restricted_random.json")
END_OF_TURN_ID = 106


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


def get_target_layer(model, layer):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer]
    return model.model.layers[layer]


def chat_template_ids(tok, prompt):
    input_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        return_tensors="pt", return_dict=False,
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    return input_ids


def get_per_token_residuals(model, tok, prompt, layer):
    input_ids = chat_template_ids(tok, prompt).to(model.device)
    captured = {}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()
    handle = get_target_layer(model, layer).register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        handle.remove()
    h = captured["h"][0]  # [seq, d_model]
    ids = input_ids[0].tolist()
    try:
        eot_idx = ids.index(END_OF_TURN_ID)
    except ValueError:
        eot_idx = len(ids)
    start = 4
    return h[start:eot_idx], ids[start:eot_idx]


def boot_ci(xs, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.array([x for x in xs if x == x])
    if len(arr) == 0: return float("nan"), float("nan"), float("nan")
    res = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(1)
    return float(arr.mean()), float(np.quantile(res, 0.025)), float(np.quantile(res, 0.975))


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
    sae = JumpReLUSAE.from_hf(SAE_REPO, LAYER)

    # ============================================================
    # Pass 1: per-token feature activations across all 60 × {B, D}
    # ============================================================
    # We want:
    #   - For each format-direction feature: top-K activating (token, context) pairs across all prompts
    #   - For each medical feature: top-K activating (token, context) pairs (sanity)
    #   - Per-case mean and max activations over user content (for restricted random Phase 1b)
    print("\n=== Pass 1: collecting per-token feature activations ===")
    target_features = list(set(FORMAT_DIRECTION_FEATURES + MEDICAL_FEATURES_L29))
    # We'll keep a heap of top-K (activation, token_id, context_str, case_id, condition) per feature
    top_per_feature = {f: [] for f in target_features}
    K = TOP_TOKENS_PER_FEATURE

    # Also collect mean/max per case for the restricted-random analysis
    per_case_mean_features = []  # list of dicts: {id, condition, feature_means [d_sae]}
    per_case_max_features = []

    # We'll also track which features fire on the union of B+D content
    # for restricted-random pool selection.
    # Use a running 'has any nonzero activation across cases' tally.
    fires_on_content_count = torch.zeros(sae.d_sae, device="cuda")

    for i, c in enumerate(cases):
        for cond, prompt_key in [("B", "B_prompt"), ("D", "D_prompt")]:
            residuals, content_ids = get_per_token_residuals(
                model, tok, c[prompt_key], LAYER
            )
            with torch.no_grad():
                feats = sae.encode(residuals)  # [n_tokens, d_sae]
            feats_cpu = feats.float().cpu()

            # Track firing on content (any token had activation > 0.5)
            fires_on_content_count += (feats > 0.5).any(dim=0).float()

            # Mean and max per feature (for restricted-random Phase 1b)
            per_case_mean_features.append({
                "id": c["id"], "condition": cond,
                "stratum": c["stratum"],
                "mean_features": feats_cpu.mean(0),  # [d_sae]
            })
            per_case_max_features.append({
                "id": c["id"], "condition": cond,
                "max_features": feats_cpu.max(0).values,
            })

            # Top-tokens per target feature
            for f in target_features:
                col = feats_cpu[:, f]  # [n_tokens]
                # Find top tokens in this prompt for this feature
                if col.max() <= 0: continue
                # We'll just record every token with positive activation; sort later
                for tok_pos, act in enumerate(col):
                    a = act.item()
                    if a <= 0: continue
                    # Build context window
                    lo = max(0, tok_pos - CONTEXT_WINDOW)
                    hi = min(len(content_ids), tok_pos + CONTEXT_WINDOW + 1)
                    target_token = tok.decode([content_ids[tok_pos]])
                    context = tok.decode(content_ids[lo:hi])
                    top_per_feature[f].append({
                        "activation": a,
                        "case_id": c["id"], "condition": cond,
                        "token_pos": tok_pos,
                        "target_token": target_token,
                        "context": context,
                    })
        if (i + 1) % 15 == 0:
            print(f"  case {i+1}/60")

    # Keep top-K per feature
    for f in target_features:
        top_per_feature[f].sort(key=lambda x: -x["activation"])
        top_per_feature[f] = top_per_feature[f][:K]

    # ============================================================
    # Tier 1B: Save top-token analysis
    # ============================================================
    print("\n=== Tier 1B: top tokens per feature ===")
    summary_1b = {
        "model": MODEL_ID, "layer": LAYER,
        "format_direction_features": FORMAT_DIRECTION_FEATURES,
        "medical_features": MEDICAL_FEATURES_L29,
        "top_tokens": {str(f): top_per_feature[f] for f in target_features},
    }
    OUT_TOPTOKENS.parent.mkdir(parents=True, exist_ok=True)
    OUT_TOPTOKENS.write_text(json.dumps(summary_1b, indent=2))
    print(f"  wrote {OUT_TOPTOKENS}")
    for f in FORMAT_DIRECTION_FEATURES:
        print(f"\n  Feature {f} (format-direction) top 5 tokens:")
        for entry in top_per_feature[f][:5]:
            ctx = entry["context"].replace("\n", " ")[:120]
            print(f"    act={entry['activation']:>7.1f}  '{entry['target_token']}'  in: «{ctx}»")
    for f in MEDICAL_FEATURES_L29:
        print(f"\n  Feature {f} (medical, sanity) top 3 tokens:")
        for entry in top_per_feature[f][:3]:
            ctx = entry["context"].replace("\n", " ")[:120]
            print(f"    act={entry['activation']:>7.1f}  '{entry['target_token']}'  in: «{ctx}»")

    # ============================================================
    # Tier 1C: Restricted-random Phase 1b
    # ============================================================
    print("\n=== Tier 1C: restricted-random Phase 1b at L29 ===")
    # Aggregate per-case feature means by case_id + condition
    means_by_case_cond = {}
    for entry in per_case_mean_features:
        means_by_case_cond[(entry["id"], entry["condition"])] = entry["mean_features"]

    # Build [60, d_sae] matrices for B and D
    B_mean = torch.stack([means_by_case_cond[(c["id"], "B")] for c in cases])
    D_mean = torch.stack([means_by_case_cond[(c["id"], "D")] for c in cases])

    # Restricted random pool: features that fire on content in at least 25% of all
    # 120 prompts (60 cases × 2 conditions), AND are magnitude-matched to medical features.
    fires_threshold_count = 0.25 * (60 * 2)  # appears in at least ~30 of 120 prompts
    fires_enough = fires_on_content_count >= fires_threshold_count
    fires_enough = fires_enough.cpu()

    ref = torch.cat([B_mean, D_mean], dim=0)
    mean_per_feat = ref.float().mean(0)
    med_means = mean_per_feat[MEDICAL_FEATURES_L29]
    lo = 0.5 * med_means.min().item()
    hi = 2.0 * med_means.max().item()
    in_band = (mean_per_feat >= lo) & (mean_per_feat <= hi)
    # exclude medical features
    for f in MEDICAL_FEATURES_L29:
        in_band[f] = False
    # restrict to features that fire on content
    pool_mask = in_band & fires_enough
    pool = pool_mask.nonzero(as_tuple=True)[0].tolist()
    rng = np.random.default_rng(RANDOM_SEED)
    if len(pool) >= N_RANDOM:
        random_features = sorted(rng.choice(pool, size=N_RANDOM, replace=False).tolist())
    else:
        random_features = sorted(pool)
    print(f"  Magnitude band [{lo:.2f}, {hi:.2f}], "
          f"firing-on-content threshold {int(fires_threshold_count)}/120, "
          f"pool size {len(pool)}")
    print(f"  Random features (n={len(random_features)}): {random_features[:6]}...")

    # Per-case mod-index for medical and random
    per_case = []
    for i, c in enumerate(cases):
        entry = {"id": c["id"], "stratum": c["stratum"],
                 "gold_letters": c["gold_letters"]}
        for kind, feats in [("medical", MEDICAL_FEATURES_L29),
                            ("random_restricted", random_features)]:
            v_B = B_mean[i][feats]
            v_D = D_mean[i][feats]
            denom = v_B.norm() * v_D.norm()
            cos = (v_B @ v_D / denom).item() if denom > 0 else float("nan")
            num = (v_D - v_B).abs()
            den = (v_B.abs() + v_D.abs()) / 2
            mod = (num / den.clamp(min=1e-8)).mean().item()
            entry[f"{kind}_cosine"] = cos
            entry[f"{kind}_mod_index"] = mod
            entry[f"{kind}_acts_B_mean"] = v_B.tolist()
            entry[f"{kind}_acts_D_mean"] = v_D.tolist()
        per_case.append(entry)

    # Stratum summaries
    agg = defaultdict(lambda: {"med_mod": [], "rnd_mod": []})
    for e in per_case:
        s = e["stratum"]
        agg[s]["med_mod"].append(e["medical_mod_index"])
        agg[s]["rnd_mod"].append(e["random_restricted_mod_index"])

    summary_1c = {
        "model": MODEL_ID, "layer": LAYER,
        "medical_features": MEDICAL_FEATURES_L29,
        "random_features_restricted": random_features,
        "magnitude_band": {"lo": lo, "hi": hi},
        "firing_threshold_pct_of_prompts": 0.25,
        "pool_size": len(pool),
        "per_case": per_case,
        "stratum_summary": {},
    }
    print(f"\n  Stratum-level results (medical vs restricted-random mod-index):")
    print(f"  {'stratum':<18s}{'n':>4s}  {'med':>8s}  {'rnd':>8s}  {'diff':>8s}  {'95% CI':>20s}")
    for s, d in agg.items():
        diff = [a - b for a, b in zip(d["med_mod"], d["rnd_mod"])]
        med_mean, _, _ = boot_ci(d["med_mod"])
        rnd_mean, _, _ = boot_ci(d["rnd_mod"])
        diff_mean, lo_ci, hi_ci = boot_ci(diff)
        summary_1c["stratum_summary"][s] = {
            "n": len(d["med_mod"]),
            "med_mod_mean": med_mean, "rnd_mod_mean": rnd_mean,
            "diff_mean": diff_mean, "diff_ci_95": [lo_ci, hi_ci],
        }
        print(f"  {s:<18s}{len(d['med_mod']):>4d}  {med_mean:>8.3f}  {rnd_mean:>8.3f}  "
              f"{diff_mean:>+8.3f}  [{lo_ci:+.3f}, {hi_ci:+.3f}]")

    OUT_RESTRICTED.write_text(json.dumps(summary_1c, indent=2))
    print(f"\n  wrote {OUT_RESTRICTED}")


if __name__ == "__main__":
    main()
