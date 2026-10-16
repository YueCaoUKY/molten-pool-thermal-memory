#!/usr/bin/env python3
"""Model-estimation workflow (manuscript Figure 2, Section 4).

Panel (a) identification; panel (b) the division of one record into an estimation interval and
a retained interval (adaptation was removed from the paper on 2026-09-29); panel (c) the two
evaluation routes. Notation follows Section 4: R = {(u_i, z_i)}, loss (4), accuracy (5).

Writes Figure_2_model_estimation_workflow.{svg,pdf,png} next to this script; copy the png and
svg into Manuscript/figures/ as Figure_2_model_estimation_workflow_draft.*.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent
NAME = "Figure_2_model_estimation_workflow"

BG = "#F5F7FA"
PANEL_EDGE = "#C9D1DB"
INK, MUTED = "#1E293B", "#4A5568"
GREY = ("#9AA8BA", "#F7F9FB")
BLUE = ("#2F6FE0", "#EBF2FE")
GREEN = ("#1BA672", "#E9F7F2")
ORANGE = ("#E8900C", "#FFF6EC")
ARROW = "#1E293B"

plt.rcParams.update({
    "font.family": ["DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
})


def panel(ax, x0, y0, x1, y1, title=None, fill="white", title_fs=17):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=1.4",
                                lw=1.6, edgecolor=PANEL_EDGE, facecolor=fill, zorder=0))
    if title:
        ax.text(x0 + 3.0, y1 - 4.4, title, ha="left", va="center", fontsize=title_fs,
                fontweight="bold", color=INK, zorder=1)


def box(ax, x0, y0, x1, y1, style, lines):
    """lines: list of (text, fontsize, bold, colour) placed from top to bottom."""
    edge, fill = style
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                lw=2.0, edgecolor=edge, facecolor=fill, zorder=2))
    n = len(lines)
    cy, h = (y0 + y1) / 2, (y1 - y0)
    step = min(h / (n + 0.6), 4.4)
    for k, (text, fs, bold, colour) in enumerate(lines):
        y = cy + (n - 1) / 2 * step - k * step
        ax.text((x0 + x1) / 2, y, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=colour, zorder=3)


def arrow(ax, a, b, color=ARROW, lw=1.8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=lw,
                                 color=color, shrinkA=0, shrinkB=0, zorder=4))


def small(t):
    return (t, 11.5, False, MUTED)


def label(t):
    return (t, 12.5, False, MUTED)


def main(t, fs=14.5):
    return (t, fs, True, INK)


def build():
    fig = plt.figure(figsize=(20, 11.2), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 112)
    ax.axis("off")

    # (a) Model identification
    panel(ax, 4.8, 58.1, 95.8, 106.3, "(a) Model identification")
    box(ax, 9.2, 72.9, 31.5, 88.1, GREY,
        [label("one process record"), main(r"$\mathcal{R}=\{(\mathbf{u}_i,\mathbf{z}_i)\}$"), small("pool present throughout")])
    arrow(ax, (32.0, 80.5), (39.2, 80.5))
    box(ax, 39.9, 69.6, 66.1, 91.4, BLUE,
        [(r"identify all model", 13, False, INK), main(r"parameters $\boldsymbol{\theta}$"),
         small("multi-step open-loop"), small("prediction error")])
    arrow(ax, (66.7, 80.5), (72.7, 80.5))
    box(ax, 73.4, 72.9, 90.8, 88.1, GREEN,
        [label("identified"), main(r"model $\boldsymbol{\theta}^{*}$")])
    ax.text(49.6, 63.3, r"$\delta$ is inferred; no thermal label is used", ha="center",
            va="center", fontsize=12.5, color=MUTED)

    # (b) Estimation and retained intervals of one record
    panel(ax, 99.8, 58.1, 195.2, 106.3, "(b) Estimation and evaluation intervals")
    x0, xm, x1, yb, hb = 106.0, 147.0, 188.0, 73.5, 6.5
    ax.add_patch(Rectangle((x0, yb), xm - x0, hb, facecolor="#F8C26A", edgecolor=GREY[0], lw=1.2, zorder=3))
    ax.add_patch(Rectangle((xm, yb), x1 - xm, hb, facecolor="#DCE5F0", edgecolor=GREY[0], lw=1.2, zorder=3))
    ax.text((x0 + xm) / 2, yb + hb / 2, "estimation interval", ha="center", va="center",
            fontsize=12.5, color=INK, zorder=4)
    ax.text((xm + x1) / 2, yb + hb / 2, "retained interval", ha="center", va="center",
            fontsize=12.5, color=INK, zorder=4)
    ax.plot([xm, xm], [69.0, 93.5], ls="--", lw=1.4, color=MUTED, zorder=4)
    arrow(ax, (x0, 86.0), (x1, 86.0))
    ax.text((x0 + x1) / 2, 89.8, "one open-loop prediction from the first sample; no restart at the division",
            ha="center", va="center", fontsize=11.5, color=INK)
    ax.text((x0 + xm) / 2, 69.6, "standardization, pre-fit and loss (4)", ha="center", va="center",
            fontsize=11, color=MUTED)
    ax.text((xm + x1) / 2, 69.6, r"accuracy (5) and $\rho$", ha="center", va="center",
            fontsize=11, color=MUTED)
    ax.text(147.0, 62.3, "the retained interval is never used for estimation",
            ha="center", va="center", fontsize=11.5, color=MUTED)

    # links to evaluation
    arrow(ax, (50.5, 57.8), (50.5, 51.8), color="#7A8BA0", lw=1.5)
    arrow(ax, (147.8, 57.8), (147.8, 51.8), color="#7A8BA0", lw=1.5)

    # (c) Evaluation
    panel(ax, 4.8, 4.8, 195.2, 52.9, "(c) Evaluation")

    panel(ax, 9.7, 9.8, 93.6, 44.8, "Finite-volume simulation", fill="#F8FAFC", title_fs=15)
    box(ax, 13.0, 24.6, 35.4, 36.5, BLUE,
        [label("pool geometry"), (r"observed $\mathbf{z}$", 14, False, INK), small("used for estimation")])
    box(ax, 13.0, 11.9, 35.4, 23.9, ORANGE,
        [label("workpiece temperature"), (r"thermal reference", 14, False, INK),
         ("withheld from identification", 10.5, False, MUTED)])
    arrow(ax, (36.0, 30.6), (46.2, 28.9))
    arrow(ax, (36.0, 17.9), (46.2, 20.3))
    box(ax, 47.0, 13.5, 84.8, 36.5, GREEN,
        [label("assessment"), main("pool-state accuracy"), main("hidden-state fidelity"),
         small("thermal-effect analysis")])

    panel(ax, 97.6, 9.8, 190.3, 44.8, "Experimental datasets", fill="#F8FAFC", title_fs=15)
    ax.add_patch(FancyBboxPatch((101.4, 19.1), 13.6, 11.9, boxstyle="round,pad=0,rounding_size=1.2",
                                lw=1.6, edgecolor=GREY[0], facecolor="#263244", zorder=2))
    ax.add_patch(Circle((108.2, 25.1), 2.7, facecolor="#FF9636", edgecolor="none", zorder=3))
    ax.text(108.2, 16.6, "images", ha="center", va="center", fontsize=12.5, color=MUTED)
    arrow(ax, (115.8, 25.1), (121.0, 25.1))
    box(ax, 121.8, 18.0, 141.4, 31.7, GREEN,
        [label("image encoder"), main(r"pool state $\mathbf{z}$")])
    arrow(ax, (142.0, 25.1), (147.3, 25.1))
    box(ax, 148.1, 16.3, 171.6, 34.8, BLUE,
        [label("dynamic model"), main("open-loop", 13.5), main("pool-state prediction", 13)])
    arrow(ax, (172.2, 25.1), (176.0, 25.1))
    ax.text(182.6, 28.4, "observed-state", ha="center", va="center", fontsize=12.5, color=INK)
    ax.text(182.6, 24.4, "accuracy", ha="center", va="center", fontsize=14.5,
            fontweight="bold", color=INK)
    ax.text(182.6, 19.8, "no thermal", ha="center", va="center", fontsize=11, color=MUTED)
    ax.text(182.6, 17.2, "reference", ha="center", va="center", fontsize=11, color=MUTED)
    return fig


if __name__ == "__main__":
    fig = build()
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 164})):
        fig.savefig(OUT / f"{NAME}.{ext}", facecolor=BG, **kw)
    print("written:", *(OUT / f"{NAME}.{e}" for e in ("svg", "pdf", "png")))
