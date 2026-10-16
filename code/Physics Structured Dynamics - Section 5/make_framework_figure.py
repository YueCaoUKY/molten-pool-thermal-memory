#!/usr/bin/env python3
"""Create the paper's physical-motivation and dynamic-model schematics.

Both figures are deliberately schematic. Figure 1 makes the non-Markov
counterexample visible; Figure 5 connects the learned equations to a physical
pool/workpiece energy balance. No simulated or experimental values are
implied by the drawn trajectories.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "Paper writing" / "figures"
SECTION5_ROOT = OUTPUT_ROOT / "section5"

FONT = "Liberation Sans"
INK = "#20262e"
MUTED = "#66717c"
BLUE = "#2f6690"
LIGHT_BLUE = "#eaf2f8"
ORANGE = "#d97706"
LIGHT_ORANGE = "#fff1d6"
GREEN = "#3f7d5a"
RED = "#b43c35"
LINE = "#aeb6bf"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": FONT,
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def arrow(ax, start, end, color=INK, lw=1.0, mutation=8, connection="arc3") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=lw,
            color=color,
            connectionstyle=connection,
            shrinkA=1,
            shrinkB=1,
        )
    )


def rounded_box(ax, xy, width, height, text, face, edge, fontsize=7.2, weight="normal") -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.9,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color=INK,
        linespacing=1.05,
    )


def panel_title(ax, letter, title) -> None:
    ax.text(0.0, 1.01, f"({letter})  {title}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.3, weight="bold", color=INK)


def draw_surface_observation(ax, x, y, w, h, label) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#20252a", edgecolor=LINE, linewidth=0.8))
    ax.add_patch(Ellipse((x + 0.52 * w, y + 0.48 * h), 0.58 * w, 0.42 * h,
                         facecolor="#70777c", edgecolor="#f2f2f2", linewidth=0.9))
    ax.add_patch(Ellipse((x + 0.52 * w, y + 0.48 * h), 0.30 * w, 0.20 * h,
                         facecolor="#a8adb0", edgecolor="none", alpha=0.8))
    ax.text(x + w / 2, y - 0.020, label, ha="center", va="top", fontsize=6.8, color=MUTED)


def draw_cross_section(ax, x, y, w, h, hot: bool) -> None:
    nx, ny = 180, 45
    xx = np.linspace(-1, 1, nx)[None, :]
    yy = np.linspace(0, 1, ny)[:, None]
    base = 0.18 + (0.34 if hot else 0.08) * np.exp(-1.5 * yy)
    local = (0.55 if hot else 0.36) * np.exp(-4.0 * xx**2) * np.exp(-3.2 * yy)
    field = np.clip(base + local, 0, 1)
    ax.imshow(field, extent=(x, x + w, y, y + h), origin="upper", cmap="inferno",
              vmin=0, vmax=1, aspect="auto", interpolation="bilinear", zorder=0)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=LINE, linewidth=0.7, zorder=2))
    ax.add_patch(Ellipse((x + 0.50 * w, y + h), 0.40 * w, 0.34 * h,
                         facecolor="#ffd27a", edgecolor="white", linewidth=0.6,
                         alpha=0.92, zorder=3))


def make_motivation_figure() -> None:
    fig = plt.figure(figsize=(6.85, 2.48))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.30, 0.92, 1.10), wspace=0.25,
                          left=0.035, right=0.985, top=0.88, bottom=0.12)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    panel_title(ax, "a", "Hidden thermal history")
    draw_surface_observation(ax, 0.05, 0.60, 0.36, 0.22, "earlier history")
    draw_surface_observation(ax, 0.59, 0.60, 0.36, 0.22, "heat-soaked history")
    ax.text(0.50, 0.88, r"same measured input $\mathbf{u}_0$", ha="center",
            va="center", color=GREEN, weight="bold", fontsize=7.4)
    ax.text(0.50, 0.50, r"same visible pool state $\mathbf{z}_0$", ha="center",
            va="center", color=BLUE, weight="bold", fontsize=7.4)
    draw_cross_section(ax, 0.05, 0.17, 0.36, 0.19, hot=False)
    draw_cross_section(ax, 0.59, 0.17, 0.36, 0.19, hot=True)
    ax.text(0.23, 0.11, r"low $\delta$", ha="center", va="center",
            color=ORANGE, fontsize=7.4, weight="bold")
    ax.text(0.77, 0.11, r"high $\delta$", ha="center", va="center",
            color=ORANGE, fontsize=7.4, weight="bold")
    ax.text(0.50, 0.01, "different unobserved subsurface fields", ha="center",
            va="bottom", color=MUTED, fontsize=6.8)

    ax = axes[1]
    panel_title(ax, "b", "Different futures")
    plot = ax.inset_axes([0.13, 0.21, 0.80, 0.64])
    t = np.linspace(0, 2.0, 160)
    cold = 0.5 - 0.25 * (1 - np.exp(-1.3 * t))
    hot = 0.5 + 0.30 * (1 - np.exp(-1.0 * t))
    plot.plot(t, cold, color=BLUE, linewidth=1.7)
    plot.plot(t, hot, color=ORANGE, linewidth=1.7)
    plot.scatter([0], [0.5], s=18, color=INK, zorder=4)
    plot.axvline(0, color=LINE, linewidth=0.7)
    plot.set_xlim(-0.05, 2.05)
    plot.set_ylim(0.15, 0.88)
    plot.set_ylabel("pool response")
    plot.set_xticks([0, 1, 2], [r"$t_0$", r"$t_0+\tau$", r"$t_0+2\tau$"])
    plot.set_yticks([])
    plot.spines[["top", "right"]].set_visible(False)
    plot.spines[["left", "bottom"]].set_color(LINE)
    plot.text(1.95, hot[-1] + 0.025, "heat-soaked", color=ORANGE,
              ha="right", va="bottom", fontsize=6.7)
    plot.text(1.95, cold[-1] - 0.025, "cool workpiece", color=BLUE,
              ha="right", va="top", fontsize=6.7)
    ax.text(0.52, 0.035, r"same $(\mathbf{z}_0,\mathbf{u}_0)$, different $\dot{\mathbf{z}}_0$",
            ha="center", va="bottom", color=RED, fontsize=6.8, weight="bold")

    ax = axes[2]
    panel_title(ax, "c", "State closure")
    rounded_box(ax, (0.02, 0.67), 0.40, 0.16, "surface only\n$(\\mathbf{z},\\mathbf{u})$",
                "#f3f4f5", LINE, fontsize=6.7)
    rounded_box(ax, (0.58, 0.67), 0.40, 0.16, "one future\nfor both histories",
                "#fbe9e7", RED, fontsize=6.3)
    arrow(ax, (0.42, 0.75), (0.58, 0.75), color=RED)
    ax.text(0.50, 0.86, "memoryless closure", ha="center", va="bottom",
            color=RED, fontsize=6.6)
    rounded_box(ax, (0.02, 0.36), 0.40, 0.18, "augmented state\n$(\\mathbf{z},\\delta,\\mathbf{u})$",
                LIGHT_ORANGE, ORANGE, fontsize=6.5, weight="bold")
    rounded_box(ax, (0.58, 0.36), 0.40, 0.18, "history-dependent\nMarkov future",
                LIGHT_BLUE, BLUE, fontsize=6.3, weight="bold")
    arrow(ax, (0.42, 0.45), (0.58, 0.45), color=ORANGE, lw=1.2)
    ax.text(0.50, 0.22,
            r"$\dot{\mathbf{z}}=-\gamma\,[\mathbf{z}-\mathbf{z}_{eq}(\mathbf{u},\delta)]$",
            ha="center", va="center", fontsize=7.5, color=BLUE)
    ax.text(0.50, 0.10,
            r"$\dot\delta=\alpha P_{in}(\mathbf{r})-\beta Q_{wp}(\delta)$",
            ha="center", va="center", fontsize=7.5, color=ORANGE)
    ax.text(0.50, 0.01, r"$\gamma>0$; monotone loss; slower thermal time scale",
            ha="center", va="bottom", color=MUTED, fontsize=6.5)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_ROOT / "fig1_framework_overview.svg", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUTPUT_ROOT / "fig1_framework_overview.png", dpi=400,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def make_physical_model_figure() -> None:
    fig, ax = plt.subplots(figsize=(3.46, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.97, "a", ha="left", va="top", weight="bold", fontsize=8.5)
    ax.text(0.08, 0.97, "Coupled pool–workpiece energy balance", ha="left", va="top",
            weight="bold", fontsize=8.2)

    plate_y, plate_h = 0.53, 0.20
    ny, nx = 60, 300
    xx = np.linspace(-1, 1, nx)[None, :]
    yy = np.linspace(0, 1, ny)[:, None]
    field = 0.12 + 0.50 * np.exp(-3.0 * xx**2) * np.exp(-2.8 * yy)
    ax.imshow(field, extent=(0.12, 0.90, plate_y, plate_y + plate_h), origin="upper",
              cmap="inferno", vmin=0, vmax=1, aspect="auto", interpolation="bilinear")
    ax.add_patch(Rectangle((0.12, plate_y), 0.78, plate_h, fill=False,
                           edgecolor=INK, linewidth=0.8))
    ax.add_patch(Ellipse((0.50, plate_y + plate_h), 0.31, 0.18,
                         facecolor="#ffd27a", edgecolor=BLUE, linewidth=1.0))
    ax.text(0.50, plate_y + plate_h + 0.005, r"pool state $\mathbf{z}$",
            ha="center", va="center", color=BLUE, fontsize=7.0, weight="bold")
    ax.text(0.50, plate_y + 0.055, r"workpiece thermal state $\delta$",
            ha="center", va="center", color="white", fontsize=7.0, weight="bold")
    arrow(ax, (0.50, 0.92), (0.50, 0.82), color=GREEN, lw=1.4, mutation=9)
    ax.text(0.52, 0.88, r"absorbed input $P_{in}$", ha="left", va="center",
            color=GREEN, fontsize=6.8)
    arrow(ax, (0.63, 0.74), (0.74, 0.62), color=ORANGE, lw=1.2)
    ax.text(0.72, 0.75, r"pool-to-solid transfer $Q_p$", ha="center", va="bottom",
            color=ORANGE, fontsize=6.3)
    arrow(ax, (0.18, 0.61), (0.07, 0.70), color=MUTED, lw=1.0)
    arrow(ax, (0.82, 0.61), (0.93, 0.70), color=MUTED, lw=1.0)
    ax.text(0.50, 0.48, r"environment and fixture loss $Q_{wp}$",
            ha="center", va="center", color=MUTED, fontsize=6.6)

    ax.text(0.02, 0.43, "b", ha="left", va="top", weight="bold", fontsize=8.5)
    ax.text(0.08, 0.43, "Structured neural realization", ha="left", va="top",
            weight="bold", fontsize=8.2)
    rounded_box(ax, (0.06, 0.15), 0.47, 0.15,
                "fast pool relaxation\n$\\dot{\\mathbf{z}}=-\\gamma(\\mathbf{z}-\\mathbf{z}_{eq})$",
                LIGHT_BLUE, BLUE, fontsize=7.0, weight="bold")
    rounded_box(ax, (0.58, 0.15), 0.36, 0.15,
                "slow thermal balance\n$\\dot\\delta=\\alpha P_{in}-\\beta Q_{wp}$",
                LIGHT_ORANGE, ORANGE, fontsize=6.8, weight="bold")
    arrow(ax, (0.58, 0.225), (0.53, 0.225), color=ORANGE, lw=1.2)
    ax.text(0.61, 0.335, r"$\delta$ shifts equilibrium and rate", ha="center",
            va="bottom", color=ORANGE, fontsize=6.3)
    ax.text(0.16, 0.055, r"$\gamma>0$", color=BLUE, ha="center", va="center",
            fontsize=7.1, weight="bold")
    ax.text(0.49, 0.055, r"$Q'_{wp}(\delta)>0$", color=ORANGE, ha="center", va="center",
            fontsize=7.1, weight="bold")
    ax.text(0.82, 0.055, r"$\tau_{pool}<\tau_{wp}$", color=INK, ha="center", va="center",
            fontsize=7.1, weight="bold")
    ax.plot([0.29, 0.29], [0.025, 0.085], color=LINE, linewidth=0.7)
    ax.plot([0.66, 0.66], [0.025, 0.085], color=LINE, linewidth=0.7)

    SECTION5_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(SECTION5_ROOT / "fig5_dissipative_node.svg", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(SECTION5_ROOT / "fig5_dissipative_node.png", dpi=400,
                bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    configure_style()
    make_motivation_figure()
    make_physical_model_figure()


if __name__ == "__main__":
    main()
