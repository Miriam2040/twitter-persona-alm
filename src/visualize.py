"""
visualize.py — generate all analysis charts.

Run:
    python src/visualize.py

Outputs:
    assets/full_analysis.png     — 4-panel: heatmap, KDE, scatter, bar
    assets/results_all_personas.png  — 3-panel distribution comparison
    assets/finetune_validation.png   — ALM vs base GPT-2
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import gaussian_kde

BG     = "#0d1117"
CARD   = "#161b22"
BORDER = "#30363d"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"

COLORS = {
    "trump":                "#e63946",
    "musk":                 "#457b9d",
    "democrat_senators":    "#2d6a4f",
    "republican_senators":  "#a8dadc",
}
LABELS = {
    "trump":                "Trump",
    "musk":                 "Musk",
    "democrat_senators":    "Dem senators",
    "republican_senators":  "Rep senators",
}
PERSONAS = ["trump", "musk", "democrat_senators", "republican_senators"]
SHORT    = ["Trump", "Musk", "Dem\nsenators", "Rep\nsenators"]


def style_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(color=BORDER, linestyle="--", alpha=0.4)


def load_scores():
    scored = {}
    for p in PERSONAS:
        path = f"results/{p}_eval_scores.csv"
        scored[p] = pd.read_csv(path)["perplexity"].dropna()
    return scored


# ─── CHART 1: full 4-panel analysis ─────────────────────────────────────────

def full_analysis():
    scored = load_scores()
    matrix_raw = pd.read_csv("results/cross_score_matrix.csv", index_col="author")
    norm = matrix_raw.copy()
    for author in matrix_raw.index:
        norm.loc[author] = matrix_raw.loc[author] / matrix_raw.loc[author, author]

    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35)

    # ── Panel 1: heatmap ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    norm_arr = norm.loc[PERSONAS, PERSONAS].values
    cmap = cm.get_cmap("RdYlGn_r")
    im = ax1.imshow(norm_arr, cmap=cmap, vmin=1.0, vmax=3.1, aspect="auto")
    ax1.set_xticks(range(4)); ax1.set_yticks(range(4))
    ax1.set_xticklabels(SHORT, color=TEXT, fontsize=10)
    ax1.set_yticklabels(SHORT, color=TEXT, fontsize=10)
    ax1.set_xlabel("Model used for scoring")
    ax1.set_ylabel("Author of tweets")
    ax1.set_title("Cross-Scoring Matrix\n(ratio vs home score — 1.0=same, higher=more foreign)")
    ax1.set_facecolor(CARD)
    for sp in ax1.spines.values(): sp.set_color(BORDER)
    ax1.tick_params(colors=TEXT)
    for i in range(4):
        for j in range(4):
            val = norm_arr[i, j]
            txt = f"{val:.2f}×\n(home)" if i == j else f"{val:.2f}×"
            col = "black" if 1.3 < val < 2.1 else TEXT
            ax1.text(j, i, txt, ha="center", va="center", color=col,
                     fontsize=9, fontweight="bold")
    cb = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=TEXT)
    cb.outline.set_color(BORDER)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=8)
    cb.set_label("Surprise ratio", color=MUTED, fontsize=9)

    # ── Panel 2: KDE log-PPL ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2)
    x = np.linspace(1.5, 8.0, 600)
    for p in PERSONAS:
        log_ppls = np.log(scored[p].clip(upper=5000))
        kde = gaussian_kde(log_ppls, bw_method=0.15)
        y   = kde(x)
        ax2.fill_between(x, y, alpha=0.2, color=COLORS[p])
        ax2.plot(x, y, color=COLORS[p], linewidth=2.5, label=LABELS[p])
        ax2.axvline(np.log(np.median(scored[p])), color=COLORS[p],
                    linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.set_xlabel("log(Perplexity)")
    ax2.set_ylabel("Density")
    ax2.set_title("Perplexity Distributions (log scale)\nDashed = median per persona")
    ax2.legend(facecolor=CARD, labelcolor=TEXT, edgecolor=BORDER, fontsize=9)
    ax2.set_xlim(1.5, 8.0)

    # ── Panel 3: individual senator scatter ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3)
    for csv, color, party in [
        ("data/processed/democrat_senators_scored.csv",   COLORS["democrat_senators"],   "Democrat"),
        ("data/processed/republican_senators_scored.csv", COLORS["republican_senators"], "Republican"),
    ]:
        df = pd.read_csv(csv)
        users = []
        for user, grp in df.groupby("username"):
            if len(grp) >= 50:
                ppls = grp["perplexity"].dropna()
                users.append((
                    user.replace("Rep", "").replace("Sen", ""),
                    float(np.median(ppls)),
                    float(np.percentile(ppls, 75) - np.percentile(ppls, 25)),
                    len(grp),
                ))
        meds = [u[1] for u in users]
        iqrs = [u[2] for u in users]
        ns   = [u[3] for u in users]
        ax3.scatter(meds, iqrs, alpha=0.55, color=color,
                    s=[max(n * 0.35, 20) for n in ns],
                    label=party, edgecolors="none")
        for uname, med, iqr, n in users:
            if iqr > 28 or iqr < 8.5:
                ax3.annotate(uname, (med, iqr), fontsize=6.5, color=color,
                             xytext=(4, 2), textcoords="offset points")
    # Trump & Musk reference stars
    ax3.scatter([30.2], [22.3], color=COLORS["trump"], s=220, marker="*",
                zorder=6, label="Trump (individual)")
    ax3.annotate("Trump", (30.2, 22.3), fontsize=8.5, color=COLORS["trump"],
                 xytext=(4, 2), textcoords="offset points", fontweight="bold")
    ax3.scatter([32.5], [37.5], color=COLORS["musk"], s=220, marker="*",
                zorder=6, label="Musk (individual)")
    ax3.annotate("Musk", (32.5, 37.5), fontsize=8.5, color=COLORS["musk"],
                 xytext=(4, 2), textcoords="offset points", fontweight="bold")
    ax3.set_xlabel("Median Perplexity")
    ax3.set_ylabel("IQR — consistency spread")
    ax3.set_title("Individual Senator Consistency\n(★ = Trump & Musk for reference; bubble = tweet count)")
    ax3.legend(facecolor=CARD, labelcolor=TEXT, edgecolor=BORDER, fontsize=9)

    # ── Panel 4: style distance bars ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4)
    groups = [
        ("Trump\ntweets →",  [("Rep senators", float(norm.loc["trump","republican_senators"])),
                               ("Dem senators", float(norm.loc["trump","democrat_senators"])),
                               ("Musk",         float(norm.loc["trump","musk"]))]),
        ("Musk\ntweets →",   [("Trump",         float(norm.loc["musk","trump"])),
                               ("Rep senators", float(norm.loc["musk","republican_senators"])),
                               ("Dem senators", float(norm.loc["musk","democrat_senators"]))]),
        ("Dem\nsenator →",   [("Rep senators", float(norm.loc["democrat_senators","republican_senators"])),
                               ("Trump",        float(norm.loc["democrat_senators","trump"])),
                               ("Musk",         float(norm.loc["democrat_senators","musk"]))]),
        ("Rep\nsenator →",   [("Dem senators", float(norm.loc["republican_senators","democrat_senators"])),
                               ("Trump",        float(norm.loc["republican_senators","trump"])),
                               ("Musk",         float(norm.loc["republican_senators","musk"]))]),
    ]
    mc = {"Trump": COLORS["trump"], "Musk": COLORS["musk"],
          "Dem senators": COLORS["democrat_senators"],
          "Rep senators": COLORS["republican_senators"]}
    x_pos = np.arange(len(groups))
    width = 0.26
    for gi, (_, bars) in enumerate(groups):
        for bi, (model_name, ratio) in enumerate(bars):
            offset = (bi - 1) * width
            ax4.bar(gi + offset, ratio, width, color=mc[model_name],
                    alpha=0.85, edgecolor=BORDER, linewidth=0.5)
            ax4.text(gi + offset, ratio + 0.05, f"{ratio:.2f}",
                     ha="center", va="bottom", color=TEXT, fontsize=7.5)
    ax4.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2, alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([g[0] for g in groups], color=TEXT, fontsize=9)
    ax4.set_ylabel("Surprise ratio (1.0 = home style)")
    ax4.set_title("Style Distances — How foreign does each author\nsound when scored by other models?")
    ax4.set_ylim(0.8, 3.3)
    from matplotlib.patches import Patch
    leg = [Patch(fc=v, label=k, ec=BORDER) for k, v in mc.items()]
    leg.append(plt.Line2D([0], [0], color=MUTED, linestyle="--", label="Home (1.0)"))
    ax4.legend(handles=leg, facecolor=CARD, labelcolor=TEXT, edgecolor=BORDER, fontsize=8)

    fig.suptitle("Twitter Persona ALM — Full Analysis",
                 fontsize=16, color=TEXT, y=1.01, fontweight="bold")
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/full_analysis.png", dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print("Saved assets/full_analysis.png")
    plt.close()


# ─── CHART 2: results_all_personas ──────────────────────────────────────────

def results_all_personas():
    scored = load_scores()
    medians = [float(np.median(scored[p])) for p in PERSONAS]
    iqrs    = [float(np.percentile(scored[p],75) - np.percentile(scored[p],25)) for p in PERSONAS]
    names   = [LABELS[p] for p in PERSONAS]
    cols    = [COLORS[p] for p in PERSONAS]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
    for ax in axes:
        style_ax(ax)

    # Bar: median PPL
    ax = axes[0]
    bars = ax.bar(names, medians, color=cols, edgecolor=BORDER, linewidth=0.8, zorder=3)
    ax.set_title("Median Perplexity\n(lower = more predictable)")
    ax.set_ylabel("Perplexity")
    for bar, val in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", color=TEXT,
                fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(medians) * 1.28)
    ax.tick_params(axis="x", rotation=12)

    # Bar: IQR
    ax = axes[1]
    bars = ax.bar(names, iqrs, color=cols, edgecolor=BORDER, linewidth=0.8, zorder=3)
    ax.set_title("IQR — Consistency Index\n(lower = more consistent)")
    ax.set_ylabel("IQR")
    for bar, val in zip(bars, iqrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", color=TEXT,
                fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(iqrs) * 1.28)
    ax.tick_params(axis="x", rotation=12)

    # Boxplot log-PPL
    ax = axes[2]
    log_data = [np.log(scored[p].clip(upper=500)) for p in PERSONAS]
    bp = ax.boxplot(log_data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED),
                    flierprops=dict(marker=".", color=MUTED, markersize=2, alpha=0.3))
    for patch, color in zip(bp["boxes"], cols):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(names, rotation=12, fontsize=9, color=TEXT)
    ax.set_title("log(PPL) Distribution\n(PPL capped at 500 for display)")
    ax.set_ylabel("log(Perplexity)")

    fig.suptitle("ALM Perplexity Comparison — All 4 Personas",
                 fontsize=14, color=TEXT, y=1.01, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/results_all_personas.png", dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print("Saved assets/results_all_personas.png")
    plt.close()


# ─── CHART 3: finetune_validation ───────────────────────────────────────────

def finetune_validation():
    base_med  = [64.5, 76.6, 69.7, 74.7]
    alm_med   = [30.2, 32.5, 25.6, 25.7]
    alm_beats = [99.2, 96.2, 99.8, 99.7]
    names     = [LABELS[p] for p in PERSONAS]
    cols      = [COLORS[p] for p in PERSONAS]
    improvement = [b/a for b, a in zip(base_med, alm_med)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    for ax in axes:
        style_ax(ax)

    x = np.arange(len(PERSONAS))
    w = 0.35

    # Grouped bar: base vs ALM
    ax = axes[0]
    b1 = ax.bar(x - w/2, base_med, w, label="Base GPT-2",     color="#6e7681", edgecolor=BORDER)
    b2 = ax.bar(x + w/2, alm_med,  w, label="Fine-tuned ALM", color="#238636", edgecolor=BORDER)
    ax.set_title("Median PPL: Base GPT-2 vs Fine-tuned ALM\n(lower = better)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, fontsize=9)
    ax.set_ylabel("Median Perplexity")
    ax.legend(facecolor=CARD, labelcolor=TEXT, edgecolor=BORDER)
    ax.set_ylim(0, 95)
    for bar, val in zip(b1, base_med):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}",
                ha="center", va="bottom", color=MUTED, fontsize=8)
    for bar, val in zip(b2, alm_med):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}",
                ha="center", va="bottom", color="#3fb950", fontsize=8, fontweight="bold")

    # Improvement factor
    ax = axes[1]
    bars = ax.bar(names, improvement, color=cols, edgecolor=BORDER, zorder=3)
    ax.set_title("Fine-tuning Improvement\n(how many × better than base GPT-2)")
    ax.set_ylabel("Improvement factor (×)")
    ax.set_ylim(0, max(improvement) * 1.3)
    ax.tick_params(axis="x", rotation=12)
    for bar, val, beats in zip(bars, improvement, alm_beats):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{val:.1f}×\n({beats}% win)", ha="center", va="bottom",
                color=TEXT, fontsize=9, fontweight="bold")

    fig.suptitle("Fine-tuning Validation — All 4 Personas vs Base GPT-2",
                 fontsize=13, color=TEXT, y=1.01, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/finetune_validation.png", dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print("Saved assets/finetune_validation.png")
    plt.close()


if __name__ == "__main__":
    full_analysis()
    results_all_personas()
    finetune_validation()
    print("All charts saved to assets/")
