#!/usr/bin/env python3
"""Acuity-tier probes on cached SAE feature vectors (rebuttal, reviewer C92e W1).

Question: is acuity-RELEVANT information (not just medical-domain content)
decodable from the model's representations of the shared clinical vignette,
and equally so under forced-letter (NL) and free-text (NF) prompts?

Representations (from results/phase1b_masked_full_activations_{model}.npz,
headline late layer per model: 4B L29, 12B L31, Qwen L31):
  - vignette:  SAE features max-pooled over the byte-identical vignette tokens
  - decision:  SAE features at the pre-generation decision token

Targets:
  - emergency (primary, binary): gold label includes D (28 vs 32)
  - most_urgent (secondary, 4-class): most urgent gold letter (A8/B10/C14/D28)

Probe mirrors the paper's Section 3.6 configuration: L2 logistic regression
(C=0.05, liblinear, balanced class weights), leave-one-out CV with the scaler
fit inside each fold, ROC-AUC on held-out probabilities, empirical p from
label-permutation with full LOO refit per permutation.

Outputs: results/acuity_probes.json, results/acuity_probes.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    "4b": {"npz": "results/phase1b_masked_full_activations_4b.npz", "layer": 29},
    "12b": {"npz": "results/phase1b_masked_full_activations_12b.npz", "layer": 31},
    "qwen": {"npz": "results/phase1b_masked_full_activations_qwen.npz", "layer": 31},
}
# npz key prefixes: B = NL (forced-letter), D = NF (free-text)
CONDITIONS = {"NL": "B", "NF": "D"}
POSITIONS = {"vignette": "max_vignette", "decision": "decision"}


def loo_scores(X, y):
    """Held-out P(class) per case under LOO; multiclass returns [n, k]."""
    n = len(y)
    classes = np.unique(y)
    out = np.zeros((n, len(classes)))
    for i in range(n):
        tr = np.arange(n) != i
        sc = StandardScaler().fit(X[tr])
        base = LogisticRegression(C=0.05, solver="liblinear",
                                  class_weight="balanced", max_iter=2000)
        clf = base if len(classes) == 2 else OneVsRestClassifier(base)
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[i:i + 1]))[0]
        for j, c in enumerate(clf.classes_):
            out[i, np.searchsorted(classes, c)] = p[j]
    return out


def loo_auc_binary(X, y):
    return roc_auc_score(y, loo_scores(X, y)[:, 1])


def perm_p(X, y, observed, n_perm, seed=0, n_jobs=-1):
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(y) for _ in range(n_perm)]
    null = Parallel(n_jobs=n_jobs)(delayed(loo_auc_binary)(X, yp) for yp in perms)
    null = np.asarray(null)
    return float((np.sum(null >= observed) + 1) / (n_perm + 1)), null


def paired_auc_bootstrap(y, s_a, s_b, n_boot=2000, seed=0):
    """Case-level bootstrap CI on AUC(a) - AUC(b) from paired held-out scores."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    while len(diffs) < n_boot:
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], s_a[idx]) - roc_auc_score(y[idx], s_b[idx]))
    diffs = np.asarray(diffs)
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    gold = {v["id"]: v["gold_standard_triage"]
            for v in json.load(open(ROOT / "paper/data/canonical_forced_letter_vignettes.json"))}

    results = {"config": {"n_perm": args.n_perm, "probe": "LogReg C=0.05 liblinear balanced, LOO",
                          "layers": {m: c["layer"] for m, c in MODELS.items()}},
               "runs": {}, "paired": {}}

    for model, cfg in MODELS.items():
        d = np.load(ROOT / cfg["npz"], allow_pickle=True)
        ids = [str(x) for x in d["case_ids"]]
        y_bin = np.array([1 if "D" in gold[i] else 0 for i in ids])
        y_mc = np.array([ord(max(gold[i].split("/"))) - ord("A") for i in ids])

        scores = {}
        for pos, key_suffix in POSITIONS.items():
            for cond, prefix in CONDITIONS.items():
                X = d[f"{prefix}_{key_suffix}"].astype(np.float64)
                tag = f"{model}/{cond}/{pos}"
                print(f"[{tag}] LOO ...", flush=True)
                s = loo_scores(X, y_bin)[:, 1]
                auc = roc_auc_score(y_bin, s)
                p, null = perm_p(X, y_bin, auc, args.n_perm, n_jobs=args.n_jobs)
                # secondary: 4-class most-urgent, descriptive macro-OVR AUC
                s_mc = loo_scores(X, y_mc)
                auc_mc = roc_auc_score(y_mc, s_mc, multi_class="ovr", average="macro")
                scores[(cond, pos)] = s
                results["runs"][tag] = {
                    "auc_emergency": float(auc),
                    "perm_p": p,
                    "null_auc_mean": float(null.mean()),
                    "null_auc_p95": float(np.percentile(null, 95)),
                    "auc_most_urgent_macro_ovr": float(auc_mc),
                }
                print(f"[{tag}] AUC={auc:.3f} p={p:.4f} 4class={auc_mc:.3f}", flush=True)

        for pos in POSITIONS:
            ci = paired_auc_bootstrap(y_bin, scores[("NL", pos)], scores[("NF", pos)])
            results["paired"][f"{model}/{pos}/NL-NF"] = {
                "delta_auc": float(roc_auc_score(y_bin, scores[("NL", pos)])
                                   - roc_auc_score(y_bin, scores[("NF", pos)])),
                "ci95": ci,
            }
        for cond in CONDITIONS:
            ci = paired_auc_bootstrap(y_bin, scores[(cond, "vignette")], scores[(cond, "decision")])
            results["paired"][f"{model}/{cond}/vignette-decision"] = {
                "delta_auc": float(roc_auc_score(y_bin, scores[(cond, "vignette")])
                                   - roc_auc_score(y_bin, scores[(cond, "decision")])),
                "ci95": ci,
            }

    out = ROOT / "results/acuity_probes.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")

    lines = ["# Acuity-tier probes (rebuttal, C92e W1)", "",
             "Emergency target: gold includes D (28 vs 32). LOO ROC-AUC, "
             f"{args.n_perm}-permutation p (full LOO refit per permutation).", "",
             "| Model | Cond | Position | AUC (emergency) | perm p | 4-class macro-OVR AUC |",
             "|---|---|---|---|---|---|"]
    for tag, r in results["runs"].items():
        m, c, pos = tag.split("/")
        lines.append(f"| {m} | {c} | {pos} | {r['auc_emergency']:.3f} | "
                     f"{r['perm_p']:.4f} | {r['auc_most_urgent_macro_ovr']:.3f} |")
    lines += ["", "## Paired contrasts (case-bootstrap 95% CI on delta AUC)", "",
              "| Contrast | delta AUC | 95% CI |", "|---|---|---|"]
    for tag, r in results["paired"].items():
        lines.append(f"| {tag} | {r['delta_auc']:+.3f} | "
                     f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] |")
    (ROOT / "results/acuity_probes.md").write_text("\n".join(lines) + "\n")
    print("wrote results/acuity_probes.md")


if __name__ == "__main__":
    main()
