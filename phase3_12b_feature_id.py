"""
Phase 3 — Identify medical features in Gemma 3 12B IT for the workshop's
intra-family scale-generality result.

Method (medical-vs-non-medical contrastive, English-only):
  1. Forward pass the 60 patient_realistic prompts through Gemma 3 12B IT,
     capture residuals at candidate layers (12, 24, 31, 41), SAE-encode,
     take max activation per (case, feature) over user content tokens.
  2. Forward pass 30 non-medical patient-style prompts through the same
     model, same encoding.
  3. Per layer, score each feature:
        score = mean_med_max - mean_nonmed_max
        subject to:
          fires_in_medical >= 0.7   (max > FIRE_THRESHOLD on >=70% of cases)
          fires_in_nonmed  <= 0.1   (max > FIRE_THRESHOLD on <=10% of cases)
  4. Top-3 features per layer.

Output: results/phase3_12b_features.json
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

MODEL_ID = "google/gemma-3-12b-it"
SAE_REPO = "google/gemma-scope-2-12b-it"
LAYERS = [12, 24, 31, 41]  # matched depths to 4B's 9/17/22/29

SINGLETURN_PATH = Path(
    "nature_triage_expanded_replication/paper_faithful_replication/data/"
    "canonical_singleturn_vignettes.json"
)
OUT_PATH = Path("results/phase3_12b_features.json")
END_OF_TURN_ID = 106
FIRE_THRESHOLD = 1.0
TOP_K = 5  # save top 5 per layer; pick top 3 in downstream scripts

# 30 non-medical patient-style prompts. Same conversational register as the
# patient_realistic prompts: a person describing a non-medical situation,
# asking for guidance.
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


def get_content_max_features(model, tok, prompt, layer, sae):
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
    content = h[start:eot_idx].to(sae.w_enc.dtype).to(sae.w_enc.device)
    with torch.no_grad():
        feats = sae.encode(content)  # [n_content, d_sae]
    return feats.max(0).values.float().cpu()  # [d_sae]


def main():
    print("Loading medical prompts (60 patient_realistic)...")
    st = json.loads(SINGLETURN_PATH.read_text())
    medical_prompts = [v["patient_realistic"] for v in st]
    assert len(medical_prompts) == 60

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
        d_sae = sae.d_sae

        med_max = torch.zeros(len(medical_prompts), d_sae)
        for i, p in enumerate(medical_prompts):
            med_max[i] = get_content_max_features(model, tok, p, layer, sae)
            if (i + 1) % 15 == 0: print(f"  med {i+1}/60")
        non_max = torch.zeros(len(NON_MEDICAL_PROMPTS), d_sae)
        for i, p in enumerate(NON_MEDICAL_PROMPTS):
            non_max[i] = get_content_max_features(model, tok, p, layer, sae)
            if (i + 1) % 10 == 0: print(f"  non {i+1}/{len(NON_MEDICAL_PROMPTS)}")

        # Scoring
        med_mean = med_max.mean(0)        # [d_sae]
        non_mean = non_max.mean(0)
        fires_med = (med_max > FIRE_THRESHOLD).float().mean(0)  # [d_sae]
        fires_non = (non_max > FIRE_THRESHOLD).float().mean(0)
        score = med_mean - non_mean       # higher = more medical-specific

        # Filter by firing reliability
        good = (fires_med >= 0.7) & (fires_non <= 0.1)
        # In case the filter is too strict, also keep the unfiltered top-K
        ranked_filtered = torch.argsort(score * good.float(), descending=True).tolist()
        ranked_unfiltered = torch.argsort(score, descending=True).tolist()

        top_filtered = []
        for f in ranked_filtered[:20]:
            if not bool(good[f]): break
            top_filtered.append({
                "feature": int(f),
                "score": float(score[f]),
                "med_mean_max": float(med_mean[f]),
                "non_mean_max": float(non_mean[f]),
                "fires_med": float(fires_med[f]),
                "fires_non": float(fires_non[f]),
            })
        top_unfiltered = [
            {"feature": int(f),
             "score": float(score[f]),
             "med_mean_max": float(med_mean[f]),
             "non_mean_max": float(non_mean[f]),
             "fires_med": float(fires_med[f]),
             "fires_non": float(fires_non[f])}
            for f in ranked_unfiltered[:10]
        ]

        layer_results[layer] = {
            "n_features_total": d_sae,
            "n_filter_passing": int(good.sum()),
            "top_filtered": top_filtered,
            "top10_unfiltered": top_unfiltered,
        }

        print(f"  Filter-passing features: {int(good.sum())}")
        print(f"  Top 5 filter-passing:")
        for entry in top_filtered[:5]:
            print(f"    feat {entry['feature']:>5d}  score={entry['score']:>7.2f}  "
                  f"med_mean={entry['med_mean_max']:>6.1f}  non_mean={entry['non_mean_max']:>5.2f}  "
                  f"fires med/non = {entry['fires_med']:.2f}/{entry['fires_non']:.2f}")

        del sae
        torch.cuda.empty_cache()

    out = {
        "model": MODEL_ID, "sae_repo": SAE_REPO, "layers": LAYERS,
        "n_medical_prompts": len(medical_prompts),
        "n_nonmedical_prompts": len(NON_MEDICAL_PROMPTS),
        "fire_threshold": FIRE_THRESHOLD,
        "non_medical_prompts": NON_MEDICAL_PROMPTS,
        "by_layer": layer_results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
