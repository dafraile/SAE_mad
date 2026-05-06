"""
Figure 4 — Top-activating tokens for the format-direction features and the
v3-validated medical features in Gemma 3 4B IT, L29.

Two panels side-by-side:
  Left:  format-direction features (3833, 10012, 980) — top 3 activations each.
         All top activations are in NL (forced-letter) prompts on the
         literal answer-key scaffold tokens.
  Right: medical features (12570, 893, 12845) — top 3 activations each.
         Top activations on clinical-content tokens, fire identically across
         NL and NF.

Output: figures/fig4_top_tokens.{pdf,png}
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="white", context="paper", font_scale=1.0)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

PALETTE = {
    "format_dir": "#d97a3c",  # warm orange — same as 'random' in fig2 (the "format-effect" arm)
    "medical":    "#1f5f8b",  # deep blue — same as 'medical' in fig2
    "highlight":  "#ffd76e",  # warm yellow for token highlight
    "highlight_med": "#cfe1f0",  # light blue for medical token highlight
    "context":    "#222222",
    "muted":      "#777777",
}

FORMAT_DIRECTION_FEATURES = [3833, 10012, 980]
MEDICAL_FEATURES = [12570, 893, 12845]
N_PER_FEATURE = 3
CONTEXT_TRUNCATE = 90  # characters around target token


def _truncate_context(ctx: str, target: str, max_len: int = CONTEXT_TRUNCATE) -> str:
    ctx = ctx.replace("\n", " ").strip()
    if len(ctx) <= max_len:
        return ctx
    # try to center on target
    idx = ctx.find(target.strip())
    if idx == -1:
        return ctx[:max_len] + "…"
    half = max_len // 2
    lo = max(0, idx - half)
    hi = min(len(ctx), lo + max_len)
    out = ctx[lo:hi]
    if lo > 0: out = "…" + out
    if hi < len(ctx): out = out + "…"
    return out


def _split_around_target(ctx: str, target: str) -> tuple[str, str, str]:
    """Return (before, target, after) where target is the trimmed token text.

    Falls back to ('', ctx, '') if not found.
    """
    target_clean = target.strip()
    if not target_clean: return "", ctx, ""
    idx = ctx.find(target_clean)
    if idx == -1:
        # Try with surrounding space
        if ctx.startswith(target):
            return "", target, ctx[len(target):]
        return "", ctx, ""
    return ctx[:idx], target_clean, ctx[idx + len(target_clean):]


def render_panel(ax, title, features, data, color, hl_color):
    """Render one panel: list of features × top tokens."""
    ax.set_xlim(0, 1)
    ax.set_axis_off()

    # Header
    ax.text(0.0, 1.0, title, fontsize=12, fontweight="bold",
            color=color, va="top", transform=ax.transAxes)

    # Determine y positions: leave space for header, then equal spacing
    n_features = len(features)
    rows_per_feat = N_PER_FEATURE + 1  # +1 for the feature header
    total_rows = n_features * rows_per_feat
    line_h = 0.92 / total_rows
    y = 0.95 - line_h * 1.2  # below header

    for f in features:
        # Feature header
        feat_data = data[str(f)][:N_PER_FEATURE]
        ax.text(0.0, y, f"feature {f}", fontsize=10, fontweight="bold",
                family="monospace", color=color, va="top",
                transform=ax.transAxes)
        # Right-align the firing summary
        if feat_data:
            n_in_NL = sum(1 for e in feat_data if e["condition"] == "B")
            n_in_NF = sum(1 for e in feat_data if e["condition"] == "D")
            summary = f"top-{N_PER_FEATURE}: {n_in_NL} NL · {n_in_NF} NF"
            ax.text(1.0, y, summary, fontsize=8, color=PALETTE["muted"],
                    style="italic", va="top", ha="right", transform=ax.transAxes)
        y -= line_h * 1.0

        for entry in feat_data:
            act = entry["activation"]
            cond = entry["condition"]
            cond_label = "NL" if cond == "B" else "NF"
            target = entry["target_token"]
            ctx = _truncate_context(entry["context"], target)
            before, target_str, after = _split_around_target(ctx, target)

            # Activation prefix (light)
            ax.text(0.02, y, f"{act:>5.0f}", fontsize=8, family="monospace",
                    color=PALETTE["muted"], va="top", transform=ax.transAxes)
            # Condition label
            ax.text(0.10, y, cond_label, fontsize=8, family="monospace",
                    color=PALETTE["muted"], va="top", transform=ax.transAxes)
            # Context with highlighted target token — render as a single line of mixed text
            full_text = f'« {before}'
            ax.text(0.18, y, full_text, fontsize=9, family="serif",
                    color=PALETTE["context"], va="top", transform=ax.transAxes)
            # Estimate target x position — use a monospace-equivalent width estimate
            # We can't perfectly word-fit in matplotlib, so we'll approximate by
            # rendering the target with a colored background at an offset that
            # matches the rendered length of the prefix.
            # Practical compromise: render the entire string in one ax.text with
            # the target tokens inside backticks, since per-character coloring
            # is too brittle. Instead we use simpler rendering:
            y -= line_h * 0.95

        # Small gap between features
        y -= line_h * 0.3


def render_panel_simple(ax, title, features, data, color):
    """Cleaner version: monospace context with target wrapped in [highlight]."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Header bar
    ax.text(0.0, 0.99, title, fontsize=11.5, fontweight="bold",
            color=color, va="top", transform=ax.transAxes)
    ax.axhline(0.965, color=color, linewidth=1.2)

    n_features = len(features)
    rows = []
    for f in features:
        feat_data = data[str(f)][:N_PER_FEATURE]
        n_NL = sum(1 for e in feat_data if e["condition"] == "B")
        n_NF = sum(1 for e in feat_data if e["condition"] == "D")
        rows.append(("feature", f, f"top-{N_PER_FEATURE}: {n_NL} NL · {n_NF} NF"))
        for entry in feat_data:
            rows.append(("entry", entry, None))
        rows.append(("gap", None, None))
    if rows and rows[-1][0] == "gap":
        rows = rows[:-1]

    # Allocate vertical space
    total = sum(1.2 if r[0] == "feature" else (0.4 if r[0] == "gap" else 1.0) for r in rows)
    line_h = 0.93 / total
    y = 0.95

    for kind, val, summary in rows:
        if kind == "feature":
            ax.text(0.0, y, f"feature {val}", fontsize=10, fontweight="bold",
                    family="monospace", color=color, va="top",
                    transform=ax.transAxes)
            ax.text(1.0, y, summary, fontsize=8.5, color=PALETTE["muted"],
                    style="italic", va="top", ha="right", transform=ax.transAxes)
            y -= line_h * 1.2
        elif kind == "entry":
            entry = val
            cond_label = "NL" if entry["condition"] == "B" else "NF"
            target = entry["target_token"]
            ctx = _truncate_context(entry["context"], target)
            # Render activation + condition + a stylized line:
            #   « ...context... [TARGET] ...context... »
            tgt_clean = target.strip()
            ctx_with_brackets = ctx.replace(tgt_clean, f"[{tgt_clean}]", 1)
            ax.text(0.0, y, f"{entry['activation']:>6.0f}", fontsize=8.5,
                    family="monospace", color=PALETTE["muted"], va="top",
                    transform=ax.transAxes)
            ax.text(0.10, y, cond_label, fontsize=8.5, family="monospace",
                    color=PALETTE["muted"], va="top", transform=ax.transAxes)
            ax.text(0.18, y, f'« {ctx_with_brackets} »', fontsize=8.8,
                    family="serif", color=PALETTE["context"],
                    va="top", transform=ax.transAxes)
            y -= line_h
        else:
            y -= line_h * 0.4


def main():
    d = json.load(open("results/phase5_top_tokens.json"))
    top = d["top_tokens"]

    # Two-panel figure
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.5))
    render_panel_simple(
        ax_l,
        "Format-direction features (top-aligned with NL−NF max-pool direction)",
        FORMAT_DIRECTION_FEATURES, top, PALETTE["format_dir"],
    )
    render_panel_simple(
        ax_r,
        "Medical features (v3-validated; cross-lingual cross-format)",
        MEDICAL_FEATURES, top, PALETTE["medical"],
    )

    # Footer note
    fig.text(0.5, -0.01,
             "Top 3 activating (token, context) pairs per feature, on the union of NL and NF prompts (60 cases × 2 conditions = 120 prompts). "
             "Target token shown in [brackets]. Activation values are JumpReLU SAE feature values on the layer-29 residual stream of Gemma 3 4B IT.",
             ha="center", fontsize=8, style="italic", color=PALETTE["muted"])
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_top_tokens.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig4_top_tokens.png", bbox_inches="tight", dpi=160)
    plt.close()
    print("Wrote fig4_top_tokens.{pdf,png}")


if __name__ == "__main__":
    main()
