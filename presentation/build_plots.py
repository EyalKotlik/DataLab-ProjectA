#!/usr/bin/env python3
"""Render static PNG backups of every deck chart from data.json.

The deck (index.html) draws these live with Chart.js; this script renders the
SAME numbers with matplotlib so the PNGs in assets/img/ are a guaranteed-matching
fallback (for the README, slides, or if the browser is unavailable).

Usage:
    python build_plots.py            # writes presentation/assets/img/*.png

Only dependency beyond the stdlib is matplotlib. (This is presentation-only
tooling — it is NOT part of the graded retrieval pipeline or its package rules.)
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401  (kept for explicitness)

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "assets" / "img"
OUT.mkdir(parents=True, exist_ok=True)

DATA = json.loads((HERE / "data.json").read_text())

# ---- dark-technical palette (mirrors css/theme.css) -------------------------
BG = "#0d1117"
PANEL = "#161b22"
TEXT = "#e6edf3"
MUTED = "#9aa7b4"
ACCENT = "#2dd4bf"
ACCENT2 = "#f59e0b"
BAD = "#f87171"
BLUE = "#3b6ea5"
BORDER = "#30363d"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": MUTED, "axes.edgecolor": BORDER,
    "xtick.color": TEXT, "ytick.color": MUTED, "grid.color": BORDER,
    "font.family": "monospace", "font.size": 12, "axes.titlecolor": TEXT,
    "figure.dpi": 160,
})


def _style(ax, title=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    if title:
        ax.set_title(title, color=TEXT, fontsize=14, fontweight="bold", pad=12, loc="left")


def _save(fig, name):
    fig.tight_layout()
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(HERE)}")


def length_contrast():
    d = DATA["length_contrast"]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(d["labels"], d["words"], color=[ACCENT, BAD], width=0.55)
    for b, v in zip(bars, d["words"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 15, f"{v}w", ha="center", color=TEXT, fontsize=11)
    ax.set_ylabel("median words")
    ax.set_ylim(0, max(d["words"]) * 1.18)
    _style(ax, d["title"])
    _save(fig, "length_contrast.png")


def recall():
    d = DATA["recall_at_depth"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = list(range(len(d["depths"])))
    ax.plot(x, d["recall"], color=ACCENT, lw=3, zorder=2)
    ax.fill_between(x, d["recall"], color=ACCENT, alpha=0.12, zorder=1)
    for i, (dp, rc) in enumerate(zip(d["depths"], d["recall"])):
        hi = dp == d["highlight_depth"]
        ax.scatter(i, rc, s=130 if hi else 55, color=ACCENT2 if hi else ACCENT,
                   edgecolor=BG, linewidth=2, zorder=3)
        if hi:
            ax.annotate(f"top-{dp}: {int(rc*100)}%", (i, rc), textcoords="offset points",
                        xytext=(10, -22), color=ACCENT2, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in d["depths"]])
    ax.set_xlabel("rank depth (BM25)")
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.4)
    _style(ax, d["title"])
    _save(fig, "recall_at_depth.png")


def progression():
    d = DATA["progression"]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    labels = [l.replace("\n", "\n") for l in d["labels"]]
    colors = [ACCENT if i == d["final_index"] else BLUE for i in range(len(d["ndcg"]))]
    bars = ax.bar(labels, d["ndcg"], color=colors, width=0.62)
    for b, v in zip(bars, d["ndcg"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", color=TEXT, fontsize=11)
    ax.set_ylabel("mean NDCG@10")
    ax.set_ylim(0, 0.5)
    ax.grid(axis="y", alpha=0.4)
    ax.tick_params(axis="x", labelsize=10)
    _style(ax, d["title"])
    _save(fig, "progression.png")


def beta_sweep():
    d = DATA["beta_sweep"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    betas = d["betas"]
    prior = d["prior_only"]
    fused = d["fused_dense_w_0_8"]
    ax.plot(betas, prior, color=BAD, lw=3, marker="o", label="prior only (no BM25)")
    fx = [b for b, v in zip(betas, fused) if v is not None]
    fy = [v for v in fused if v is not None]
    ax.plot(fx, fy, color=ACCENT, lw=3, marker="o", label="fused with BM25 (dense_w=0.8)")
    ax.axvline(d["operating_beta"], color=MUTED, ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("β  (length-prior strength)")
    ax.set_ylabel("mean NDCG@10")
    ax.set_ylim(0.25, 0.46)
    ax.grid(axis="y", alpha=0.4)
    leg = ax.legend(facecolor=PANEL, edgecolor=BORDER, fontsize=10, loc="lower left")
    for t in leg.get_texts():
        t.set_color(TEXT)
    _style(ax, d["title"])
    _save(fig, "beta_sweep.png")


def ablation():
    d = DATA["ablation"]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    labels = d["labels"][::-1]   # top-to-bottom reads best-first
    vals = d["ndcg"][::-1]
    base_i = len(vals) - 1 - d["baseline_index"]

    def color(i, v):
        if i == base_i:
            return ACCENT
        return ACCENT2 if v >= 0.40 else (MUTED if v >= 0.30 else BAD)

    colors = [color(i, v) for i, v in enumerate(vals)]
    bars = ax.barh(labels, vals, color=colors, height=0.6)
    for b, v in zip(bars, vals):
        ax.text(v + 0.006, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", color=TEXT, fontsize=11)
    ax.set_xlabel("mean NDCG@10")
    ax.set_xlim(0, 0.5)
    ax.grid(axis="x", alpha=0.4)
    ax.tick_params(axis="y", labelsize=10)
    _style(ax, d["title"])
    _save(fig, "ablation.png")


def grid_heatmap():
    d = DATA["grid"]
    z = d["ndcg"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    im = ax.imshow(z, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(d["beta"])))
    ax.set_xticklabels([f"β={b}" for b in d["beta"]])
    ax.set_yticks(range(len(d["dense_w"])))
    ax.set_yticklabels([f"w={w}" for w in d["dense_w"]])
    for r in range(len(d["dense_w"])):
        for c in range(len(d["beta"])):
            ax.text(c, r, f"{z[r][c]:.4f}", ha="center", va="center", color="white", fontsize=9)
    # mark the peak
    pr = d["dense_w"].index(d["peak"]["dense_w"])
    pc = d["beta"].index(d["peak"]["beta"])
    ax.add_patch(plt.Rectangle((pc - 0.5, pr - 0.5), 1, 1, fill=False, edgecolor=ACCENT2, lw=3))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=MUTED)
    _style(ax, d["title"])
    ax.grid(False)
    _save(fig, "grid_heatmap.png")


def cand_n():
    d = DATA["cand_n"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(range(len(d["cand_n"])), d["ndcg"], color=ACCENT2, lw=3, marker="o")
    ax.set_xticks(range(len(d["cand_n"])))
    ax.set_xticklabels([str(v) for v in d["cand_n"]])
    ax.set_xlabel("ZFUSE_CAND_N")
    ax.set_ylabel("mean NDCG@10")
    ax.set_ylim(0.39, 0.44)
    ax.grid(axis="y", alpha=0.4)
    _style(ax, d["title"])
    _save(fig, "cand_n.png")


def main():
    print(f"Rendering PNGs from data.json -> {OUT.relative_to(HERE)}/")
    length_contrast()
    recall()
    progression()
    beta_sweep()
    ablation()
    grid_heatmap()
    cand_n()
    print("done.")


if __name__ == "__main__":
    main()
