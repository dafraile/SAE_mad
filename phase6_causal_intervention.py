"""
Phase 6 — Tier 2D: causal intervention on format-direction features.

Phase 5 named the format-direction features at Gemma 3 4B IT, L29 as
3833, 10012, 980 (the three features whose top activations are exclusively
on the forced-letter answer-key scaffold tokens — "next", "the", "=" inside
"B = See my doctor in the next few weeks", "D = Go to the ER now Do not
include any explanation", etc.).

This phase asks the causal counterpart of Phase 1b's correlational result:

  If we ablate (subtract the SAE reconstruction of) features 3833, 10012,
  980 at L29 during NL (natural + forced-letter) generation, does the
  behavioral NL accuracy on the 60 paper-canonical cases shift?

The clean prediction under Version B:
  - Ablation should NOT meaningfully change behavior (the format effect
    we measured was downstream of clinical encoding; ablating an
    upstream feature that fires on the forced-letter scaffold may not
    propagate to the output token distribution if downstream output
    circuits are doing the work).
  - This would corroborate Basu et al. 2026's null on SAE feature
    steering and our own v3 null.

The alternative under a stronger Version B:
  - Ablating those three features removes the "this prompt expects a
    constrained letter answer" signal, and NL accuracy drifts toward the
    NF (free-text) baseline.

Either result is informative. We run it as a clean three-arm comparison
on the 60 NL prompts:
  arm 1: vanilla generation (already done, 56.7%)
  arm 2: ablate format-direction features (3833, 10012, 980) at L29
  arm 3: ablate three magnitude-matched random features (control)

Output: results/phase6_causal_intervention.json
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
LAYER = 29

# Format-direction features identified in Phase 5 (top-token analysis confirms
# these fire on the forced-letter scaffold tokens themselves)
FORMAT_DIRECTION_FEATURES = [3833, 10012, 980]

# Three random features in the same magnitude band as the format-direction
# features, drawn for the control arm. Frozen seed.
RANDOM_SEED = 42
N_RANDOM_CONTROL = 3

FORCED_LETTER_PATH = Path(
    "nature_triage_expanded_replication/paper_faithful_forced_letter/data/"
    "canonical_forced_letter_vignettes.json"
)
SINGLETURN_PATH = Path(
    "nature_triage_expanded_replication/paper_faithful_replication/data/"
    "canonical_singleturn_vignettes.json"
)
OUT_PATH = Path("results/phase6_causal_intervention.json")
END_OF_TURN_ID = 106
LETTER_LINE_RE = re.compile(r"\b([ABCD])\b")


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

    def decode_features(self, features, indices):
        """Reconstruct the contribution of a subset of features.
        features: [seq, d_sae]
        indices: list[int] of feature indices to include
        Returns: [seq, d_model] partial reconstruction.
        """
        sub = features[:, indices]                              # [seq, k]
        sub_w_dec = self.w_dec[indices].to(sub.device)          # [k, d_model]
        return sub.to(sub_w_dec.dtype) @ sub_w_dec


def parse_gold(g):
    return sorted(set(re.findall(r"[ABCD]", g.upper())))


def extract_letter(out: str):
    for line in out.splitlines():
        s = line.strip().strip(".,:;*-")
        if s in ("A", "B", "C", "D"): return s
    m = LETTER_LINE_RE.search(out)
    return m.group(1) if m else None


def build_cases():
    fl = json.loads(FORCED_LETTER_PATH.read_text())
    st = json.loads(SINGLETURN_PATH.read_text())
    fl_by_id = {v["id"]: v for v in fl}
    def _key(s):
        m = re.match(r"^(\D+)(\d+)$", s)
        return (m.group(1), int(m.group(2))) if m else (s, 0)
    cases = []
    for cid in sorted(fl_by_id, key=_key):
        cases.append({
            "id": cid,
            "title": fl_by_id[cid]["title"],
            "gold_raw": fl_by_id[cid]["gold_standard_triage"],
            "gold_letters": parse_gold(fl_by_id[cid]["gold_standard_triage"]),
            "B_prompt": fl_by_id[cid]["natural_forced_letter"],  # NL
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


def make_ablation_hook(sae: JumpReLUSAE, ablate_features: list[int]):
    """Returns a forward hook that ablates `ablate_features` from the
    layer's residual stream by subtracting their SAE-reconstructed
    contribution.

    On a forward pass through the hooked layer, for each token position:
      1. Encode the residual through the SAE.
      2. Reconstruct only the contribution of `ablate_features`.
      3. Subtract that from the residual.
    """
    feats_t = torch.tensor(ablate_features, dtype=torch.long, device=sae.w_enc.device)

    def hook(_mod, _inp, output):
        if isinstance(output, tuple):
            h, *rest = output
        else:
            h, rest = output, None
        # h: [batch, seq, d_model]
        # Encode through SAE per token
        original_dtype = h.dtype
        h_flat = h.reshape(-1, h.shape[-1])  # [batch*seq, d_model]
        feats = sae.encode(h_flat)  # [batch*seq, d_sae]
        # Reconstruct only the targeted features
        sub = feats[:, feats_t]  # [batch*seq, k]
        sub_w_dec = sae.w_dec[feats_t]  # [k, d_model]
        contribution = sub.to(sub_w_dec.dtype) @ sub_w_dec  # [batch*seq, d_model]
        h_new = h_flat - contribution.to(original_dtype)
        h_new = h_new.reshape(h.shape)
        if rest is not None:
            return (h_new, *rest)
        return h_new

    return hook


def pick_random_control_features(sae: JumpReLUSAE, model, tok, cases,
                                   target_features: list[int],
                                   n: int, seed: int):
    """Pick n random features in the same magnitude band as the target
    features, on the union of B-prompt residuals.
    """
    # Quick scan: encode mean residuals from a small subset of B prompts
    # to estimate per-feature mean activations.
    sample_ids = list(range(0, len(cases), 4))[:15]  # 15 cases
    means = []
    for i in sample_ids:
        ids = chat_template_ids(tok, cases[i]["B_prompt"]).to(model.device)
        captured = {}
        def cap_hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["h"] = h.detach()
        handle = get_target_layer(model, LAYER).register_forward_hook(cap_hook)
        try:
            with torch.no_grad():
                model(input_ids=ids)
        finally:
            handle.remove()
        h = captured["h"][0]  # [seq, d_model]
        # Find content range
        ids_list = ids[0].tolist()
        try:
            eot = ids_list.index(END_OF_TURN_ID)
        except ValueError:
            eot = len(ids_list)
        h = h[4:eot]
        with torch.no_grad():
            f = sae.encode(h.to(sae.w_enc.dtype))
        means.append(f.float().mean(0).cpu())
    mean_per_feat = torch.stack(means).mean(0)
    # Target features' mean activation band
    tf_means = mean_per_feat[target_features]
    lo = 0.5 * tf_means.min().item()
    hi = 2.0 * tf_means.max().item()
    in_band = (mean_per_feat >= lo) & (mean_per_feat <= hi)
    for f in target_features: in_band[f] = False
    pool = in_band.nonzero(as_tuple=True)[0].tolist()
    rng = np.random.default_rng(seed)
    if len(pool) < n: return pool, lo, hi
    return sorted(rng.choice(pool, size=n, replace=False).tolist()), lo, hi


def run_arm(model, tok, cases, hook_target_features=None, sae=None, label=""):
    """Generate one letter per case under the given ablation. Returns
    list of {id, gold, predicted, correct, raw}."""
    results = []
    if hook_target_features is not None and len(hook_target_features) > 0:
        hook_fn = make_ablation_hook(sae, hook_target_features)
        handle = get_target_layer(model, LAYER).register_forward_hook(hook_fn)
    else:
        handle = None
    try:
        for i, c in enumerate(cases):
            ids = chat_template_ids(tok, c["B_prompt"]).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=ids, max_new_tokens=20,
                    do_sample=False, pad_token_id=tok.eos_token_id,
                )
            gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            letter = extract_letter(gen)
            ok = letter in c["gold_letters"] if letter else False
            results.append({
                "id": c["id"], "gold": c["gold_raw"],
                "predicted": letter, "correct": ok, "raw": gen,
            })
            if (i + 1) % 15 == 0:
                print(f"  [{label}] {i+1}/60")
    finally:
        if handle is not None:
            handle.remove()
    return results


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

    print(f"\nFormat-direction features: {FORMAT_DIRECTION_FEATURES}")
    print("Picking random control features (magnitude-matched)...")
    random_features, lo, hi = pick_random_control_features(
        sae, model, tok, cases, FORMAT_DIRECTION_FEATURES, N_RANDOM_CONTROL, RANDOM_SEED,
    )
    print(f"  Magnitude band [{lo:.2f}, {hi:.2f}]")
    print(f"  Random control features: {random_features}")

    print("\n=== Arm 1: vanilla generation (no ablation) ===")
    arm1 = run_arm(model, tok, cases, hook_target_features=None, label="vanilla")
    n_correct_1 = sum(r["correct"] for r in arm1)
    print(f"  Vanilla NL accuracy: {n_correct_1}/60 = {n_correct_1/60:.1%}")

    print("\n=== Arm 2: ablate format-direction features ===")
    arm2 = run_arm(model, tok, cases,
                   hook_target_features=FORMAT_DIRECTION_FEATURES, sae=sae,
                   label="ablate_fmt")
    n_correct_2 = sum(r["correct"] for r in arm2)
    print(f"  Ablated NL accuracy: {n_correct_2}/60 = {n_correct_2/60:.1%}")

    print("\n=== Arm 3: ablate random control features ===")
    arm3 = run_arm(model, tok, cases,
                   hook_target_features=random_features, sae=sae,
                   label="ablate_rnd")
    n_correct_3 = sum(r["correct"] for r in arm3)
    print(f"  Random-ablated NL accuracy: {n_correct_3}/60 = {n_correct_3/60:.1%}")

    # Per-case effects
    case_changes = []
    for c, r1, r2, r3 in zip(cases, arm1, arm2, arm3):
        case_changes.append({
            "id": c["id"], "gold": c["gold_raw"],
            "vanilla_pred": r1["predicted"], "vanilla_correct": r1["correct"],
            "fmt_ablated_pred": r2["predicted"], "fmt_ablated_correct": r2["correct"],
            "rnd_ablated_pred": r3["predicted"], "rnd_ablated_correct": r3["correct"],
        })

    summary = {
        "model": MODEL_ID, "layer": LAYER,
        "format_direction_features": FORMAT_DIRECTION_FEATURES,
        "random_control_features": random_features,
        "magnitude_band": {"lo": lo, "hi": hi},
        "arm_summaries": {
            "vanilla":         {"correct": n_correct_1, "n": 60, "acc": n_correct_1 / 60},
            "ablate_fmt_dir":  {"correct": n_correct_2, "n": 60, "acc": n_correct_2 / 60},
            "ablate_random":   {"correct": n_correct_3, "n": 60, "acc": n_correct_3 / 60},
        },
        "deltas": {
            "fmt_minus_vanilla": (n_correct_2 - n_correct_1) / 60,
            "rnd_minus_vanilla": (n_correct_3 - n_correct_1) / 60,
            "fmt_minus_rnd":     (n_correct_2 - n_correct_3) / 60,
        },
        "per_case": case_changes,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_PATH}")
    print("\n=== Phase 6 summary ===")
    print(f"  Vanilla NL:               {summary['arm_summaries']['vanilla']['acc']:.1%}")
    print(f"  Ablate format direction:  {summary['arm_summaries']['ablate_fmt_dir']['acc']:.1%}  (Δ={summary['deltas']['fmt_minus_vanilla']:+.1%})")
    print(f"  Ablate random control:    {summary['arm_summaries']['ablate_random']['acc']:.1%}  (Δ={summary['deltas']['rnd_minus_vanilla']:+.1%})")
    print(f"  Difference of differences (fmt − rnd): {summary['deltas']['fmt_minus_rnd']:+.1%}")


if __name__ == "__main__":
    main()
