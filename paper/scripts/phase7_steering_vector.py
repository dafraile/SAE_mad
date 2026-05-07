"""
Phase 7 — ActAdd-style steering vector at L29 of Gemma 3 4B IT.

Tests whether the residual-stream direction between NL (forced-letter)
and NF (free-text) prompts has causal weight on the model's letter output.
Phase 6 ablated only the contribution of 3 specific SAE features
(0.4% of residual norm) and found no behavioral change. Phase 7 tests the
full residual difference vector — the entire direction along which NL and
NF residuals differ — by ActAdd-style steering.

Setup:
  v = mean(NL_residual) - mean(NF_residual) at L29, averaged across 60 cases
  During NL inference, hook L29 to ADD -alpha * v to the residual at all
    content tokens. With alpha > 0 this subtracts the "NL-ness" from the
    NL residual, in principle shifting it toward NF.
  Run 60 NL prompts at alpha in [0, 0.5, 1.0, 2.0, 4.0]. Compare letter
    accuracy at each alpha to vanilla NL (~55%) and to NF baseline
    (~70% from adjudicated cell).

Predictions:
  - Accuracy shifts toward NF baseline as alpha increases:
    causal weight of the format direction confirmed.
  - Accuracy stable across alpha:
    null result; format effect is more distributed than a single
    layer-29 direction can capture. Reinforces readout-not-driver framing.
  - Accuracy degrades / outputs corrupt at high alpha:
    perturbation too aggressive; report low-alpha results only.

Output: results/phase7_steering_vector.json
"""
from __future__ import annotations
import json, re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-4b-it"
LAYER = 29
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0]

FORCED_LETTER_PATH = Path(
    "nature_triage_expanded_replication/paper_faithful_forced_letter/data/"
    "canonical_forced_letter_vignettes.json"
)
SINGLETURN_PATH = Path(
    "nature_triage_expanded_replication/paper_faithful_replication/data/"
    "canonical_singleturn_vignettes.json"
)
RESIDUALS_PATH = Path("results/phase2_residuals_L29.npz")
OUT_PATH = Path("results/phase7_steering_vector.json")
END_OF_TURN_ID = 106
LETTER_LINE_RE = re.compile(r"\b([ABCD])\b")


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
            "B_prompt": fl_by_id[cid]["natural_forced_letter"],
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


def make_steering_hook(direction: torch.Tensor, alpha: float):
    """Forward hook that adds (-alpha * direction) to the layer's output
    at every token position.

    direction: [d_model] vector, on the same device as the model.
    alpha: scalar coefficient. With alpha > 0, subtracts the direction.
    """
    delta = (-alpha) * direction  # [d_model]

    def hook(_mod, _inp, output):
        if isinstance(output, tuple):
            h, *rest = output
        else:
            h, rest = output, None
        # h: [batch, seq, d_model]
        h_new = h + delta.to(h.dtype).unsqueeze(0).unsqueeze(0)
        if rest is not None:
            return (h_new, *rest)
        return h_new

    return hook


def run_arm(model, tok, cases, hook_fn=None, label=""):
    if hook_fn is not None:
        handle = get_target_layer(model, LAYER).register_forward_hook(hook_fn)
    else:
        handle = None
    results = []
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
                "predicted": letter, "correct": ok,
                "raw": gen[:50],  # truncate for storage
            })
            if (i + 1) % 15 == 0:
                print(f"  [{label}] {i+1}/60", flush=True)
    finally:
        if handle is not None:
            handle.remove()
    return results


def main():
    cases = build_cases()
    assert len(cases) == 60

    # Load saved residuals from Phase 2 (mean-pooled per case at L29)
    print(f"Loading saved residuals from {RESIDUALS_PATH}...")
    rdata = np.load(RESIDUALS_PATH)
    res_B = torch.from_numpy(rdata["res_B"]).float()  # [60, d_model] — mean L29 residual on NL prompts
    res_D = torch.from_numpy(rdata["res_D"]).float()  # [60, d_model] — mean L29 residual on NF prompts
    print(f"  res_B shape: {res_B.shape}, res_D shape: {res_D.shape}")

    # Format direction: average of (NL - NF) per case, then mean across cases
    diff_per_case = res_B - res_D  # [60, d_model]
    v = diff_per_case.mean(0)      # [d_model]
    v_norm = v.norm().item()
    print(f"  ||v|| (format direction at L29) = {v_norm:.2f}")
    # Per-case diff norms for comparison
    per_case_norms = diff_per_case.norm(dim=-1)
    print(f"  per-case diff norms: mean={per_case_norms.mean().item():.2f}, "
          f"std={per_case_norms.std().item():.2f}, max={per_case_norms.max().item():.2f}")

    print(f"\nLoading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    direction = v.to(model.device)

    # Run all alphas
    arm_results = {}
    for alpha in ALPHAS:
        print(f"\n=== alpha = {alpha} ===")
        if alpha == 0.0:
            res = run_arm(model, tok, cases, hook_fn=None, label=f"a{alpha}")
        else:
            hook_fn = make_steering_hook(direction, alpha)
            res = run_arm(model, tok, cases, hook_fn=hook_fn, label=f"a{alpha}")
        n_correct = sum(r["correct"] for r in res)
        n_unparsed = sum(1 for r in res if r["predicted"] is None)
        print(f"  alpha={alpha}: {n_correct}/60 = {n_correct/60:.1%}  (unparsed: {n_unparsed})")
        arm_results[str(alpha)] = {
            "alpha": alpha,
            "correct": n_correct, "n": 60, "acc": n_correct / 60,
            "unparsed": n_unparsed,
            "results": res,
        }

    summary = {
        "model": MODEL_ID,
        "layer": LAYER,
        "direction_norm": v_norm,
        "alphas": ALPHAS,
        "arm_summaries": {a: {"correct": d["correct"], "n": d["n"],
                              "acc": d["acc"], "unparsed": d["unparsed"]}
                          for a, d in arm_results.items()},
        "deltas_vs_alpha0": {a: arm_results[a]["acc"] - arm_results["0.0"]["acc"]
                             for a in arm_results},
        "per_arm_per_case": arm_results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_PATH}")
    print("\n=== Phase 7 summary ===")
    print(f"  Direction norm ||v|| = {v_norm:.1f}")
    for a in ALPHAS:
        s = arm_results[str(a)]
        print(f"  alpha = {a:>4.1f}:  {s['correct']:>3d}/60 = {s['acc']:.1%}  "
              f"(unparsed: {s['unparsed']})")


if __name__ == "__main__":
    main()
