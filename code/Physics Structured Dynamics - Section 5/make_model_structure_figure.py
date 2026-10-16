#!/usr/bin/env python3
"""Structure of the physics-structured neural state-space model (Section 3.3).

House style follows the scientific-figure-making skill: sans-serif (Helvetica-like),
minimalist blocks, blue for the pool path, warm red for the thermal path, neutral for the
fixed structure, vector export with editable text.

Symbols are those of Eq. (3) in the manuscript:
    z            observed pool state          delta        hidden thermal state
    u            process input                z_eq,theta   pool equilibrium
    gamma_theta  pool relaxation rate         phi_theta    thermal target
    lambda_theta thermal relaxation rate
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

OUT = Path(__file__).resolve().parent

POOL, POOL_FILL = "#0F4D92", "#E4ECF5"
THERM, THERM_FILL = "#B64342", "#F6CFCB"
INK, GREY, LIGHT = "#272727", "#767676", "#CFCECE"

plt.rcParams.update({
    "font.family": ["Nimbus Sans", "DejaVu Sans", "sans-serif"],
    "font.size": 13,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic",
    "mathtext.bf": "Nimbus Sans:bold",
    "svg.fonttype": "none",
    "axes.linewidth": 0,
})


# --------------------------------------------------------------------------- primitives
def box(ax, x, y, w, h, text, edge=INK, fill="white", fs=13, lw=1.6, weight="normal",
        radius=1.4, zorder=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                linewidth=lw, edgecolor=edge, facecolor=fill, zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=zorder + 1, fontweight=weight, linespacing=1.5)
    return x + w / 2, y + h / 2


def route(ax, pts, color=INK, lw=1.6, head=True, ls="-", zorder=2):
    """Orthogonal connector through the given points, arrow head on the last segment."""
    for a, b in zip(pts[:-1], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls,
                solid_capstyle="round", zorder=zorder)
    a, b = pts[-2], pts[-1]
    if head:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=lw,
                                     color=color, shrinkA=0, shrinkB=0, zorder=zorder,
                                     linestyle=ls))
    else:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=zorder)


def junction(ax, x, y, r=1.6, color=INK):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color, lw=1.6, zorder=4))
    ax.plot([x - 0.7, x + 0.7], [y, y], color=color, lw=1.3, zorder=5)
    ax.plot([x, x], [y - 0.7, y + 0.7], color=color, lw=1.3, zorder=5)
    return x, y


def gain(ax, x, y, w=7.0, h=7.0, label="", edge=POOL, fill="white", fs=13):
    ax.add_patch(Polygon([(x, y - h / 2), (x, y + h / 2), (x + w, y)], closed=True,
                         facecolor=fill, edgecolor=edge, lw=1.6, zorder=3))
    ax.text(x + w * 0.34, y, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4)
    return x + w, y


def panel(ax, x, y, w, h, title, color=GREY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.0",
                                linewidth=1.2, edgecolor=LIGHT, facecolor="#FBFBFC", zorder=0))
    ax.text(x + 2.2, y + h - 2.6, title, ha="left", va="center", fontsize=13.5,
            fontweight="bold", color=color, zorder=1)


# --------------------------------------------------------------------------- figure
def build():
    fig, ax = plt.subplots(figsize=(14.0, 6.6))
    ax.set_xlim(0, 141)
    ax.set_ylim(0, 68)
    ax.set_axis_off()

    # ---------------------------------------------------------------- observation panel
    panel(ax, 1, 3, 26, 61, "Observed pool state")
    box(ax, 5, 47, 18, 9.5, "image  $I_i$", edge=GREY, fill="white", fs=13)
    route(ax, [(14, 47), (14, 40.5)], color=GREY)
    box(ax, 3.5, 30.5, 21, 9.5, "image encoder", edge=POOL, fill=POOL_FILL, fs=13)
    route(ax, [(14, 30.5), (14, 24)], color=GREY)
    box(ax, 6, 14.5, 16, 9.5, "$\\mathbf{z}_i \\in \\mathbb{R}^{d}$", edge=POOL,
        fill="white", fs=14)
    ax.text(14, 9.4, "or geometric measures\n(width, area)", ha="center", va="center",
            fontsize=11, color=GREY, linespacing=1.4)

    # ---------------------------------------------------------------- model panel
    panel(ax, 30, 3, 110, 61, "Physics-structured neural state-space model")

    y_u, y_p, y_t = 57.0, 45.0, 20.0          # input bus, pool band, thermal band
    y_pfb, y_zc, y_tfb, y_dc = 36.0, 32.0, 11.0, 52.5
    x_tap, x_zc, x_dc = 104.0, 111.0, 114.5

    # process input bus
    ax.text(33.4, y_u, "$\\mathbf{u}_i$", ha="right", va="center", fontsize=15, color=INK)
    ax.plot([35.5, 73], [y_u, y_u], color=INK, lw=1.6, solid_capstyle="round", zorder=2)
    ax.plot([35.5, 35.5], [y_u, y_t], color=INK, lw=1.6, solid_capstyle="round", zorder=2)

    # ---- pool path
    route(ax, [(44, y_u), (44, y_p + 5.2)], color=INK)
    box(ax, 38, y_p - 5.2, 18, 10.4, "$\\mathbf{z}_{\\mathrm{eq},\\theta}(\\mathbf{u},\\delta)$",
        edge=POOL, fill=POOL_FILL, fs=14)
    route(ax, [(56, y_p), (61.4, y_p)], color=POOL)
    junction(ax, 63.5, y_p, color=POOL)
    ax.text(61.0, y_p + 2.3, "$+$", fontsize=13, color=POOL, ha="center", va="center")
    ax.text(65.4, y_p - 3.0, "$-$", fontsize=14, color=POOL, ha="center", va="center")
    route(ax, [(65.6, y_p), (70, y_p)], color=POOL)
    gain(ax, 70, y_p, 9.5, 9.5, "$\\gamma_{\\theta}$", edge=POOL, fill=POOL_FILL, fs=15)
    route(ax, [(73, y_u), (73, y_p + 2.6)], color=INK)
    ax.text(81.7, y_p + 3.4, "$\\dot{\\mathbf{z}}$", fontsize=14, color=POOL,
            ha="center", va="center")
    route(ax, [(79.5, y_p), (84, y_p)], color=POOL)
    box(ax, 84, y_p - 5.2, 12, 10.4, "$\\int \\mathrm{d}t$", edge=INK, fill="white", fs=15)
    ax.text(74.6, y_p - 6.9, "$\\gamma_{\\theta}(\\mathbf{z},\\mathbf{u},\\delta)$",
            fontsize=11, color=POOL, ha="center", va="center")
    route(ax, [(96, y_p), (109, y_p)], color=POOL)
    ax.text(101, y_p + 3.4, "$\\mathbf{z}$", fontsize=15, color=POOL, ha="center", va="center")
    ax.add_patch(Circle((x_tap, y_p), 0.9, color=POOL, zorder=5))
    route(ax, [(x_tap, y_p), (x_tap, y_pfb), (63.5, y_pfb), (63.5, y_p - 1.7)], color=POOL, lw=1.4)

    # ---- thermal path
    route(ax, [(35.5, y_t), (36.5, y_t)], color=INK)
    box(ax, 38, y_t - 5.2, 18, 10.4, "$\\varphi_{\\theta}(\\mathbf{u},\\mathbf{z})$",
        edge=THERM, fill=THERM_FILL, fs=14)
    route(ax, [(56, y_t), (61.4, y_t)], color=THERM)
    junction(ax, 63.5, y_t, color=THERM)
    ax.text(61.0, y_t + 2.3, "$+$", fontsize=13, color=THERM, ha="center", va="center")
    ax.text(65.4, y_t - 3.0, "$-$", fontsize=14, color=THERM, ha="center", va="center")
    route(ax, [(65.6, y_t), (70, y_t)], color=THERM)
    gain(ax, 70, y_t, 9.5, 9.5, "$\\lambda_{\\theta}$", edge=THERM, fill=THERM_FILL, fs=15)
    ax.text(81.7, y_t + 3.4, "$\\dot{\\delta}$", fontsize=14, color=THERM,
            ha="center", va="center")
    route(ax, [(79.5, y_t), (84, y_t)], color=THERM)
    box(ax, 84, y_t - 5.2, 12, 10.4, "$\\int \\mathrm{d}t$", edge=INK, fill="white", fs=15)
    route(ax, [(96, y_t), (109, y_t)], color=THERM)
    ax.text(101, y_t + 3.4, "$\\delta$", fontsize=15, color=THERM, ha="center", va="center")
    ax.add_patch(Circle((x_tap, y_t), 0.9, color=THERM, zorder=5))
    route(ax, [(x_tap, y_t), (x_tap, y_tfb), (63.5, y_tfb), (63.5, y_t - 1.7)], color=THERM, lw=1.4)

    # ---- coupling between the two states
    route(ax, [(x_tap, y_p), (x_zc, y_p), (x_zc, y_zc), (47, y_zc), (47, y_t + 5.2)],
          color=POOL, lw=1.4, ls=(0, (5, 2.5)))
    route(ax, [(x_tap, y_t), (x_dc, y_t), (x_dc, y_dc), (47, y_dc), (47, y_p + 5.2)],
          color=THERM, lw=1.4, ls=(0, (5, 2.5)))
    ax.text(80, y_zc + 2.0, "pool state sets the thermal target", fontsize=11, color=POOL,
            ha="center", va="center")
    ax.text(83, y_dc + 2.0, "thermal memory shifts the pool equilibrium and rate",
            fontsize=11, color=THERM, ha="center", va="center")

    # ---- equations
    ax.text(118, y_p,
            "$\\dot{\\mathbf{z}}=-\\gamma_{\\theta}\\,[\\mathbf{z}-\\mathbf{z}_{\\mathrm{eq},\\theta}]$",
            fontsize=13.5, color=POOL, ha="left", va="center")
    ax.text(118, y_t,
            "$\\dot{\\delta}=\\lambda_{\\theta}\\,[\\varphi_{\\theta}-\\delta]$",
            fontsize=13.5, color=THERM, ha="left", va="center")
    ax.text(85, 6.6,
            "learned: $\\mathbf{z}_{\\mathrm{eq},\\theta}$, $\\gamma_{\\theta}$, "
            "$\\varphi_{\\theta}$ (MLPs) and the scalar $\\lambda_{\\theta}$"
            "        $\\gamma_{\\theta}\\geq\\gamma_{\\min}>0$,  "
            "$\\lambda_{\\theta}>0$,  $\\varphi_{\\theta}\\geq 0$ by parameterization",
            fontsize=11.5, color=GREY, ha="center", va="center")

    fig.tight_layout(pad=0.6)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"Figure_model_structure.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print("saved", OUT / "Figure_model_structure.png")


if __name__ == "__main__":
    build()
