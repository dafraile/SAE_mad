"""
Phase 1 — Format-invariance test of medical SAE features.

Question: Does adding "Reply with exactly one letter only" to a natural
clinical input change Gemma 3 4B's internal representation of the case at
the medical-feature subspace, or only the output layer?

Inputs per case:
  B prompt = canonical natural_forced_letter (natural input + forced-letter
             output instruction)
  D prompt = canonical patient_realistic    (natural input + open-ended
             question, no output instruction)

The two prompts share the same clinical content. They differ only in the
trailing output instruction. Activations are taken at the LAST TOKEN of the
full prompt, before generation. Any difference between B and D activations
at this point reflects how the output instruction modulates upstream
representation.

Layers swept: 9, 17, 22, 29.

Per layer we record:
  - activation of the v3-validated medical features for that layer
  - activation of 30 random features (frozen seed)
  - SAE reconstruction error at the same token

Stratification (using Phase 0.5 + adjudication results):
  S1: B wrong, D right (format-flipped)         — primary
  S2: both right
  S3: both wrong
  S4: B right, D wrong

Pre-registered metrics (medical features vs random features):
  - Cosine similarity of activation vector across B vs D, per case
  - Modulation index = mean |a_D - a_B| / mean (|a_B|+|a_D|)/2, per case
  Compared against random-feature distribution for the same metric.

Output: results/phase1_activation_invariance.json
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
SAE_REPO = "google/gemma-scope-2-4b-it"  # JumpReLU SAEs as safetensors
LAYERS = [9, 17, 22, 29]


class JumpReLUSAE:
    """Minimal JumpReLU SAE — loads Gemma Scope 2 params directly.

    encode(x): pre = x @ w_enc + b_enc; features = pre * (pre > threshold)
    decode(features): features @ w_dec + b_dec
    """

    def __init__(self, w_enc, w_dec, b_enc, b_dec, threshold, device):
        self.w_enc = w_enc.to(device)
        self.w_dec = w_dec.to(device)
        self.b_enc = b_enc.to(device)
        self.b_dec = b_dec.to(device)
        self.threshold = threshold.to(device)
        self.d_sae = w_enc.shape[1]
        self.d_model = w_enc.shape[0]

    @classmethod
    def from_hf(cls, repo: str, layer: int, width: str = "16k", l0: str = "medium",
                device: str = "cuda"):
        sub = f"resid_post/layer_{layer}_width_{width}_l0_{l0}/params.safetensors"
        path = hf_hub_download(repo, sub)
        p = sft.load_file(path)
        return cls(p["w_enc"], p["w_dec"], p["b_enc"], p["b_dec"], p["threshold"], device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x.float() @ self.w_enc + self.b_enc
        return pre * (pre > self.threshold).float()

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.w_dec + self.b_dec

# v3-validated medical features per layer (from v3_layer_sweep.json)
MEDICAL_FEATURES = {
    9:  [139, 9909, 956],
    17: [9854, 368, 1539],
    22: [1181, 365, 8389],
    29: [12570, 893, 12845],
}

N_RANDOM_FEATURES = 30
RANDOM_SEED = 42

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
OUT_PATH = Path("results/phase1_activation_invariance.json")


def parse_gold(gold: str) -> list[str]:
    return sorted(set(re.findall(r"[ABCD]", gold.upper())))


def build_cases() -> list[dict]:
    fl = json.loads(FORCED_LETTER_PATH.read_text())
    st = json.loads(SINGLETURN_PATH.read_text())
    fl_by_id = {v["id"]: v for v in fl}
    st_by_id = {v["id"]: v for v in st}

    phase05 = json.loads(PHASE_0_5_PATH.read_text())
    p_by_id = {r["id"]: r for r in phase05["results"]}

    adj = json.loads(ADJUDICATED_PATH.read_text())
    adj_by_id = {r["case_id"]: r for r in adj}

    def _key(s: str):
        m = re.match(r"^(\D+)(\d+)$", s)
        return (m.group(1), int(m.group(2))) if m else (s, 0)

    cases = []
    for cid in sorted(fl_by_id.keys(), key=_key):
        fl_row = fl_by_id[cid]
        st_row = st_by_id[cid]
        ph = p_by_id[cid]
        ad = adj_by_id[cid]

        # Stratum based on Phase 0.5 B-cell correctness vs adjudicated D-cell.
        # Use BOTH judges agree as a conservative D-correctness signal.
        b_right = ph["B"]["correct"]
        d_right = bool(
            ad.get("gpt_5_2_thinking_high_is_correct")
            and ad.get("claude_sonnet_4_6_is_correct")
        )

        if b_right and d_right:        stratum = "both_right"
        elif (not b_right) and d_right: stratum = "format_flipped"      # primary
        elif (not b_right) and (not d_right): stratum = "both_wrong"
        else:                          stratum = "B_only_right"

        cases.append({
            "id": cid,
            "title": fl_row["title"],
            "gold_letters": parse_gold(fl_row["gold_standard_triage"]),
            "B_prompt": fl_row["natural_forced_letter"],
            "D_prompt": st_row["patient_realistic"],
            "B_correct": b_right,
            "D_correct_both_judges": d_right,
            "stratum": stratum,
        })
    return cases


END_OF_TURN_ID = 106  # Gemma 3 chat-template token

def get_content_residuals(
    model, tokenizer, prompt: str, layer: int,
) -> torch.Tensor:
    """Forward pass, return residuals at *user content* tokens only:
    positions [4 : first <end_of_turn>] in the chat-templated input.
    Returns a tensor [n_content_tokens, d_model] on CPU.
    """
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=False,
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    input_ids_dev = input_ids.to(model.device)

    captured = {}
    def hook(_mod, _inp, out):
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

    # Find user content range: tokens 4 .. first END_OF_TURN_ID exclusive
    ids = input_ids[0].tolist()
    try:
        eot_idx = ids.index(END_OF_TURN_ID)
    except ValueError:
        eot_idx = len(ids)
    start = 4  # after <bos><start_of_turn>user\n
    if start >= eot_idx:
        # Fallback: use full sequence minus header
        start = min(4, len(ids) - 1)
        eot_idx = len(ids)
    return h[start:eot_idx].float().cpu()


def pick_random_features(
    sae: "JumpReLUSAE", ref_activations: torch.Tensor, exclude: list[int], n: int, seed: int,
) -> list[int]:
    """Pick n features that fire on the reference content (mean act > 0),
    excluding the medical features. Frozen by seed.
    """
    # ref_activations: [n_ref, d_sae] of feature activations
    mean_act = ref_activations.float().mean(0)  # [d_sae]
    firing = (mean_act > 0).nonzero(as_tuple=True)[0].tolist()
    pool = [f for f in firing if f not in exclude]
    rng = np.random.default_rng(seed)
    if len(pool) < n:
        return pool
    return sorted(rng.choice(pool, size=n, replace=False).tolist())


def main() -> None:
    cases = build_cases()
    assert len(cases) == 60, f"expected 60 cases, got {len(cases)}"

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    layer_results = {}
    for layer in LAYERS:
        print(f"\n=== Layer {layer} ===")
        sae_id = f"resid_post/layer_{layer}_width_16k_l0_medium"
        sae = JumpReLUSAE.from_hf(SAE_REPO, layer)

        med_feats = MEDICAL_FEATURES[layer]
        d_sae = sae.d_sae

        # Pass 1: collect last-token residuals for both conditions, encode,
        # then pick random features after we know what fires.
        # Activations: per-token on content range, then aggregate to (mean, max)
        all_acts_B_mean = []
        all_acts_D_mean = []
        all_acts_B_max = []
        all_acts_D_max = []
        for i, c in enumerate(cases):
            rB = get_content_residuals(model, tok, c["B_prompt"], layer).to("cuda")  # [nB, d_model]
            rD = get_content_residuals(model, tok, c["D_prompt"], layer).to("cuda")  # [nD, d_model]
            with torch.no_grad():
                fB_per_tok = sae.encode(rB).float()  # [nB, d_sae]
                fD_per_tok = sae.encode(rD).float()
            all_acts_B_mean.append(fB_per_tok.mean(0).cpu())
            all_acts_D_mean.append(fD_per_tok.mean(0).cpu())
            all_acts_B_max.append(fB_per_tok.max(0).values.cpu())
            all_acts_D_max.append(fD_per_tok.max(0).values.cpu())
            if (i + 1) % 15 == 0:
                print(f"  collected {i+1}/60 (B_tokens={rB.shape[0]} D_tokens={rD.shape[0]})")
        all_acts_B = torch.stack(all_acts_B_mean)  # mean over content tokens
        all_acts_D = torch.stack(all_acts_D_mean)
        all_acts_B_max_t = torch.stack(all_acts_B_max)
        all_acts_D_max_t = torch.stack(all_acts_D_max)
        # Reference set for random feature selection: union of B+D mean activations
        ref = torch.cat([all_acts_B, all_acts_D], dim=0)
        rand_feats = pick_random_features(
            sae, ref, exclude=med_feats, n=N_RANDOM_FEATURES, seed=RANDOM_SEED,
        )
        print(f"  medical features: {med_feats}")
        print(f"  random features (n={len(rand_feats)}): {rand_feats[:6]}...")

        # Reconstruction error per token: ||r - sae.decode(sae.encode(r))||_2 / ||r||_2
        # Compute on B residuals (one per case)
        recon_errors_B = []
        recon_errors_D = []
        for i, c in enumerate(cases):
            rB = get_content_residuals(model, tok, c["B_prompt"], layer).to("cuda")  # [nB, d_model]
            rD = get_content_residuals(model, tok, c["D_prompt"], layer).to("cuda")
            with torch.no_grad():
                xB = sae.decode(sae.encode(rB))
                xD = sae.decode(sae.encode(rD))
            # mean per-token reconstruction error (relative L2)
            eB = ((rB - xB).norm(dim=-1) / rB.norm(dim=-1).clamp(min=1e-6)).mean()
            eD = ((rD - xD).norm(dim=-1) / rD.norm(dim=-1).clamp(min=1e-6)).mean()
            recon_errors_B.append(eB.item())
            recon_errors_D.append(eD.item())

        # Per-case metrics for medical and random features
        per_case = []
        for i, c in enumerate(cases):
            aB = all_acts_B[i]
            aD = all_acts_D[i]
            entry = {
                "id": c["id"],
                "stratum": c["stratum"],
                "B_correct": c["B_correct"],
                "D_correct_both_judges": c["D_correct_both_judges"],
                "gold_letters": c["gold_letters"],
                "recon_err_B": recon_errors_B[i],
                "recon_err_D": recon_errors_D[i],
            }
            for kind, feats in [("medical", med_feats), ("random", rand_feats)]:
                v_B = aB[feats]
                v_D = aD[feats]
                v_B_max = all_acts_B_max_t[i][feats]
                v_D_max = all_acts_D_max_t[i][feats]
                # cosine similarity (mean-pooled)
                denom = v_B.norm() * v_D.norm()
                cos = (v_B @ v_D / denom).item() if denom > 0 else float("nan")
                # modulation index per feature, then mean
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

        # Stratum summaries
        from collections import defaultdict
        agg = defaultdict(lambda: {"medical_cos": [], "medical_mod": [],
                                   "random_cos": [], "random_mod": []})
        for e in per_case:
            s = e["stratum"]
            agg[s]["medical_cos"].append(e["medical_cosine"])
            agg[s]["medical_mod"].append(e["medical_mod_index"])
            agg[s]["random_cos"].append(e["random_cosine"])
            agg[s]["random_mod"].append(e["random_mod_index"])

        def _stat(xs):
            xs = [x for x in xs if x == x]  # drop NaN
            if not xs: return None
            arr = np.array(xs)
            return {"n": len(xs), "mean": float(arr.mean()), "std": float(arr.std()),
                    "median": float(np.median(arr))}
        summary = {
            s: {k: _stat(vs) for k, vs in d.items()} for s, d in agg.items()
        }

        layer_results[layer] = {
            "sae_id": sae_id,
            "medical_features": med_feats,
            "random_features": rand_feats,
            "per_case": per_case,
            "stratum_summary": summary,
            "recon_err_summary": {
                "B_mean": float(np.mean(recon_errors_B)),
                "D_mean": float(np.mean(recon_errors_D)),
                "B_max":  float(np.max(recon_errors_B)),
                "D_max":  float(np.max(recon_errors_D)),
            },
        }

        # Free the SAE before loading next
        del sae
        torch.cuda.empty_cache()

    out = {
        "model": MODEL_ID,
        "sae_repo": SAE_REPO,
        "layers": LAYERS,
        "n_cases": len(cases),
        "by_layer": layer_results,
        "cases_meta": [{k: v for k, v in c.items() if k not in ("B_prompt", "D_prompt")} for c in cases],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")

    print(f"\n=== Phase 1 layer-by-layer summary ===")
    print(f"{'layer':<6s}{'stratum':<18s}{'n':>4s}  "
          f"{'med_cos':>8s} {'med_mod':>8s}  {'rnd_cos':>8s} {'rnd_mod':>8s}")
    for layer in LAYERS:
        for s in ["format_flipped", "both_right", "both_wrong", "B_only_right"]:
            d = layer_results[layer]["stratum_summary"].get(s, {})
            mc = d.get("medical_cos") or {}
            mm = d.get("medical_mod") or {}
            rc = d.get("random_cos") or {}
            rm = d.get("random_mod") or {}
            print(f"L{layer:<5d}{s:<18s}{mc.get('n','?'):>4}  "
                  f"{mc.get('mean',float('nan')):>8.3f} {mm.get('mean',float('nan')):>8.3f}  "
                  f"{rc.get('mean',float('nan')):>8.3f} {rm.get('mean',float('nan')):>8.3f}")


if __name__ == "__main__":
    main()
