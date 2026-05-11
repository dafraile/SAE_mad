"""
Generate the workshop paper figures from results/*.json.

Outputs to paper/figures/ as PDF (vector) + PNG (preview).

Figures:
  fig1: behavioral phenomenon — three-cell accuracy on 4B and 12B
  fig2: Phase 1b magnitude-matched mod-index — medical vs random,
        all three models, bootstrap 95% CI
  fig3: Phase 2b direction analysis — ||(B-D)|| across aggregations,
        and per-feature alignment ranking with medical features marked
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


def load_result(name: str):
    with (RESULTS_DIR / name).open() as f:
        return json.load(f)

# Paper-friendly style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
mpl.rcParams["pdf.fonttype"] = 42  # editable text in PDFs
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

# Consistent palette
PALETTE = {
    "medical": "#1f5f8b",   # deep blue
    "random":  "#d97a3c",   # warm orange
    "A": "#666666",
    "B": "#3a76a8",
    "D": "#74b04d",
}


def boot_ci(xs, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.array([x for x in xs if x == x])
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    res = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(1)
    return float(arr.mean()), float(np.quantile(res, 0.025)), float(np.quantile(res, 0.975))


def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return centre - half, centre + half


# ============================================================================
# Figure 1 — Behavioral phenomenon (4B vs 12B, three cells)
# ============================================================================

def fig1_behavioral():
    # 4B from results/phase0_5_three_cells.json + adjudicated paper-scale
    p4b = load_result("phase0_5_three_cells.json")
    adj4b = load_result("phase0_5_D_for_adjudication_adjudicated_paper.json")

    # 12B from results/phase3b_12b_phase0_5.json + adjudicated
    p12b = load_result("phase3b_12b_phase0_5.json")
    adj12b = load_result("phase3b_12b_D_for_adjudication_adjudicated_paper.json")

    def stats_for(cell_results, judge_results=None, cell="A"):
        if cell in ("A", "B"):
            n = len(cell_results)
            k = sum(r[cell]["correct"] for r in cell_results)
            return k, n
        # D-cell: prefer "both judges agree" (conservative)
        n = len(judge_results)
        k = sum(1 for r in judge_results
                if r.get("gpt_5_2_thinking_high_is_correct")
                   and r.get("claude_sonnet_4_6_is_correct"))
        return k, n

    cells_4b = {
        "A": stats_for(p4b["results"], cell="A"),
        "B": stats_for(p4b["results"], cell="B"),
        "D": stats_for(None, adj4b, cell="D"),
    }
    cells_12b = {
        "A": stats_for(p12b["results"], cell="A"),
        "B": stats_for(p12b["results"], cell="B"),
        "D": stats_for(None, adj12b, cell="D"),
    }

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    cells = ["A", "B", "D"]  # internal codebase keys (kept for back-compat)
    cell_labels = [
        "SL\nstructured\n+ letter",
        "NL\nnatural\n+ letter",
        "NF\nnatural\n+ free-text",
    ]
    x = np.arange(len(cells))
    width = 0.36

    for i, (model_name, cells_data, offset, color) in enumerate([
        ("Gemma 3 4B IT",  cells_4b,  -width/2, "#5b8db8"),
        ("Gemma 3 12B IT", cells_12b, +width/2, "#1f4e6e"),
    ]):
        accs = [cells_data[c][0] / cells_data[c][1] for c in cells]
        ci_los, ci_his = zip(*[wilson_ci(*cells_data[c]) for c in cells])
        yerr_lo = [a - lo for a, lo in zip(accs, ci_los)]
        yerr_hi = [hi - a for a, hi in zip(accs, ci_his)]
        ax.bar(x + offset, accs, width, label=model_name, color=color,
               yerr=[yerr_lo, yerr_hi], capsize=3, edgecolor="white", linewidth=0.5)
        for xi, acc, hi in zip(x + offset, accs, ci_his):
            # Place label above the upper CI, not on top of the bar
            ax.text(xi, hi + 0.025, f"{acc:.0%}", ha="center", va="bottom",
                    fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(cell_labels)
    ax.set_ylabel("Triage accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{int(t*100)}%" for t in np.arange(0, 1.01, 0.2)])
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Behavioral phenomenon: format effect attenuates with scale",
                 fontsize=11, pad=8)
    ax.text(0.02, -0.32, "Error bars: Wilson 95% CI. NF (free-text) scored by paper-faithful LLM-as-judge (both judges agree).",
            transform=ax.transAxes, fontsize=8, style="italic", color="#555555")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_behavioral.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig1_behavioral.png", bbox_inches="tight", dpi=160)
    plt.close()
    print("Wrote fig1_behavioral.{pdf,png}")


# ============================================================================
# Figure 2 — Phase 1b magnitude-matched mod-index across models
# ============================================================================

def fig2_modindex():
    # Load 4B Phase 1b magnitude-matched
    p1b_4b = load_result("phase1b_magnitude_matched.json")
    p1b_12b = load_result("phase3b_12b_phase1b.json")
    p4q = load_result("phase4_qwen_L31.json")

    # Aggregate per (model, layer, stratum) the bootstrap-mean med, rnd, and diff.
    # For Gemma we use the format_flipped + both_right strata as the headline;
    # for Qwen we have all 60 (no stratification by adjudicator since same model).
    rows = []

    def collect_gemma(p1b_data, model, layer):
        pc = p1b_data["by_layer"][str(layer)]["per_case"]
        med = [c["medical_mod_index"] for c in pc]
        rnd = [c["random_mod_index"] for c in pc]
        m_med, m_med_lo, m_med_hi = boot_ci(med)
        m_rnd, m_rnd_lo, m_rnd_hi = boot_ci(rnd)
        diff = [a - b for a, b in zip(med, rnd)]
        m_d, m_d_lo, m_d_hi = boot_ci(diff)
        return {"model": model, "layer": layer,
                "med": m_med, "med_lo": m_med_lo, "med_hi": m_med_hi,
                "rnd": m_rnd, "rnd_lo": m_rnd_lo, "rnd_hi": m_rnd_hi,
                "diff": m_d, "diff_lo": m_d_lo, "diff_hi": m_d_hi}

    for L in [9, 17, 22, 29]:
        rows.append(collect_gemma(p1b_4b, "Gemma 3 4B IT", L))
    for L in [12, 24, 31, 41]:
        rows.append(collect_gemma(p1b_12b, "Gemma 3 12B IT", L))

    # Qwen L31
    pc_q = p4q["phase1b"]["per_case"]
    med_q = [c["medical_mod_index"] for c in pc_q]
    rnd_q = [c["random_mod_index"] for c in pc_q]
    m_med, m_med_lo, m_med_hi = boot_ci(med_q)
    m_rnd, m_rnd_lo, m_rnd_hi = boot_ci(rnd_q)
    diff_q = [a - b for a, b in zip(med_q, rnd_q)]
    m_d, m_d_lo, m_d_hi = boot_ci(diff_q)
    rows.append({"model": "Qwen3-8B", "layer": 31,
                 "med": m_med, "med_lo": m_med_lo, "med_hi": m_med_hi,
                 "rnd": m_rnd, "rnd_lo": m_rnd_lo, "rnd_hi": m_rnd_hi,
                 "diff": m_d, "diff_lo": m_d_lo, "diff_hi": m_d_hi})

    # Plot — three subplots side-by-side, one per model
    models = ["Gemma 3 4B IT", "Gemma 3 12B IT", "Qwen3-8B"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5),
                              sharey=True, gridspec_kw={"width_ratios": [4, 4, 1.5]})
    for ax, model in zip(axes, models):
        sub = [r for r in rows if r["model"] == model]
        layers = [r["layer"] for r in sub]
        x = np.arange(len(layers))
        width = 0.36

        med = np.array([r["med"] for r in sub])
        med_err = np.array([[r["med"] - r["med_lo"] for r in sub],
                             [r["med_hi"] - r["med"] for r in sub]])
        rnd = np.array([r["rnd"] for r in sub])
        rnd_err = np.array([[r["rnd"] - r["rnd_lo"] for r in sub],
                             [r["rnd_hi"] - r["rnd"] for r in sub]])

        ax.bar(x - width/2, med, width, color=PALETTE["medical"], label="medical",
               yerr=med_err, capsize=3, edgecolor="white", linewidth=0.5)
        ax.bar(x + width/2, rnd, width, color=PALETTE["random"], label="random (mag-matched)",
               yerr=rnd_err, capsize=3, edgecolor="white", linewidth=0.5)

        # Annotate diff with significance
        for xi, r in zip(x, sub):
            sig = "*" if (r["diff_lo"] > 0 or r["diff_hi"] < 0) else ""
            ax.text(xi, max(r["med_hi"], r["rnd_hi"]) + 0.02,
                    f"Δ={r['diff']:+.2f}{sig}", ha="center", fontsize=8.5,
                    color="#444")

        ax.set_xticks(x)
        ax.set_xticklabels([f"L{L}" for L in layers], fontsize=9)
        ax.set_title(model, fontsize=10)
        ax.set_ylim(0, 0.55)
        if ax is axes[0]:
            ax.set_ylabel("Modulation index\n(lower = more invariant)")
            ax.legend(loc="upper right", frameon=False, fontsize=9)
        ax.set_xlabel("Layer")

    fig.suptitle("Phase 1b: magnitude-matched feature invariance across models",
                 fontsize=12, y=1.02)
    fig.text(0.5, -0.04,
             "Bars: bootstrap mean. Error bars: 95% CI on the mean (n=60 cases per layer). "
             "Δ = (medical − random) mod-index; * indicates 95% CI excludes zero.",
             ha="center", fontsize=8, style="italic", color="#555")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_modindex.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig2_modindex.png", bbox_inches="tight", dpi=160)
    plt.close()
    print("Wrote fig2_modindex.{pdf,png}")


# ============================================================================
# Figure 3 — Phase 2b direction analysis
# ============================================================================

def fig3_direction():
    # Use 4B L29 Phase 2b
    p2b = load_result("phase2b_dilution_check.json")
    info = p2b["by_layer"]["29"]
    med_feats = info["medical_features"]
    n_total = info["full_mean_pool"]["top10"][0]  # to derive — use saved data length
    # Diff norms for the three aggregations
    diff_norms = {
        "Full mean-pool\n(length-confounded)": info["full_mean_pool"]["diff_norm"],
        "Length-controlled\nmean-pool": info["truncated_mean_pool"]["diff_norm"],
        "Max-pool\n(length-invariant)": info["max_pool"]["diff_norm"],
    }

    # Get medical feature ranks per aggregation
    ranks = {}
    for key_data, key_label in [
        ("full_mean_pool", "Full mean-pool"),
        ("truncated_mean_pool", "Length-controlled"),
        ("max_pool", "Max-pool"),
    ]:
        ranks[key_label] = info[key_data]["ranks"]

    # Two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.6))

    # LEFT: diff norms across aggregations
    labels = list(diff_norms.keys())
    norms = list(diff_norms.values())
    colors = ["#5b8db8", "#d97a3c", "#1f4e6e"]
    bars = ax1.bar(labels, norms, color=colors, edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, norms):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + max(norms)*0.01,
                 f"{v:.1f}" if v > 0.01 else "0.00",
                 ha="center", va="bottom", fontsize=9, fontweight="bold" if v < 0.1 else "normal")
    ax1.set_ylabel(r"$\|\langle r_{NL} - r_{NF}\rangle\|_2$", fontsize=11)
    ax1.set_title("Residual difference norm by aggregation (Gemma 4B L29)",
                  fontsize=10)
    ax1.tick_params(axis="x", labelsize=8.5)
    ax1.text(0.02, 0.97, "NL and NF contain byte-identical clinical content;\n"
                          "the only difference is whether the forced-letter\n"
                          "instruction block is appended.",
             transform=ax1.transAxes, fontsize=8, va="top", color="#444",
             style="italic")

    # RIGHT: medical feature ranks under max-pool, plotted as percentiles
    # For all 3 models (4B L29, 12B L31, Qwen L31)
    rows_ranks = []
    # 4B L29
    m4b_max = info["max_pool"]
    n_total_4b = 16384  # gemma-scope-2 width
    for f in med_feats:
        r = m4b_max["ranks"][str(f)]
        rows_ranks.append({"model": "Gemma 4B\nL29", "feature": str(f),
                           "rank_pct": 100 * r / n_total_4b})

    # 12B L31
    p12b_2b = load_result("phase3b_12b_phase2b.json")
    info12 = p12b_2b["by_layer"]["31"]
    for f in info12["medical_features"]:
        r = info12["max_pool"]["ranks"][str(f)]
        rows_ranks.append({"model": "Gemma 12B\nL31", "feature": str(f),
                           "rank_pct": 100 * r / 16384})

    # Qwen L31
    p4q = load_result("phase4_qwen_L31.json")
    for f in p4q["medical_features"]:
        r = p4q["phase2b_max_pool"]["medical_ranks"][str(f)]
        rows_ranks.append({"model": "Qwen3-8B\nL31", "feature": str(f),
                           "rank_pct": 100 * r / p4q["n_features_total"]})

    # Strip plot
    import pandas as pd
    df = pd.DataFrame(rows_ranks)
    sns.stripplot(data=df, x="model", y="rank_pct", hue="model",
                  size=10, ax=ax2, alpha=0.85, palette=["#5b8db8", "#3a76a8", "#1f4e6e"],
                  legend=False)
    # Annotate features with per-model collision-aware offsets
    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("rank_pct")
        last_y = -100.0
        for _, row in sub.iterrows():
            y = row["rank_pct"]
            # If too close to previous label, offset upward by a fixed pixel amount
            if y - last_y < 6:
                offset_y = 12  # push label up
            else:
                offset_y = 0
            last_y = max(last_y, y) + (1.0 if offset_y else 0.0)
            ax2.annotate(row["feature"], xy=(row["model"], y),
                         xytext=(8, offset_y), textcoords="offset points",
                         fontsize=7, color="#444",
                         arrowprops=dict(arrowstyle="-",
                                         color="#bbb", lw=0.4) if offset_y else None)
    ax2.axhline(50, color="#999", linestyle="--", alpha=0.7, linewidth=1)
    ax2.set_ylim(-2, 102)
    ax2.set_ylabel("Rank percentile of medical features\nin |alignment| with (NL−NF) max-pool")
    ax2.set_xlabel("")
    ax2.set_title("Medical features are far from the format direction",
                  fontsize=10)
    ax2.text(0.02, 0.97,
             "Higher percentile = farther from format direction.\n"
             "Median (50th) shown as dashed line.\n"
             "Most medical features rank below the top-aligned set.",
             transform=ax2.transAxes, fontsize=8, va="top", color="#444",
             style="italic")
    ax2.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_direction.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig3_direction.png", bbox_inches="tight", dpi=160)
    plt.close()
    print("Wrote fig3_direction.{pdf,png}")


# ============================================================================

if __name__ == "__main__":
    fig1_behavioral()
    fig2_modindex()
    fig3_direction()
    print(f"\nAll figures in {OUT_DIR}/")
