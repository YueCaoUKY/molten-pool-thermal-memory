#!/usr/bin/env python3
"""Figures 9 and 10 of Section 6, from the arrays written by sec6_experiments.py.

The style follows the existing manuscript figures (Figure 5 and Figure 7): the same serif face,
the same colours for the two-state NODE and for the model without the thermal state, and the
same light grid. Everything is exported at double-column width as PNG (600 dpi), PDF and SVG.

    python sec6_figures.py 9 10
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
RES = HERE / "results_sec6"
OUT = HERE.parent.parent / "Manuscript" / "figures"
LOCAL = HERE / "figures_sec6"

try:
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    FACE = "Times New Roman"
except ValueError:
    FACE = "Nimbus Roman"

OBSERVED = "#252525"          # the recorded state, as in Figures 5 and 7
WITH_DELTA = "#2166ac"        # two-state NODE
WITHOUT_DELTA = "#d95f02"     # without the thermal state
THERMAL = "#6b6b6b"
GRID = "#dddddd"
WIDTH_IN = 180.0 / 25.4       # double column


def style() -> None:
    mpl.rcParams.update({
        "font.family": FACE, "mathtext.fontset": "custom", "mathtext.rm": FACE,
        "mathtext.it": f"{FACE}:italic", "mathtext.bf": f"{FACE}:bold",
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0, "legend.frameon": False,
        "savefig.dpi": 600, "savefig.bbox": None, "svg.fonttype": "none",
    })


def clean(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.6, length=2.6)
    ax.grid(axis="both", color=GRID, lw=0.4)
    ax.set_axisbelow(True)


def right_time_label(ax) -> None:
    """The panels above a row of frames carry their time label at the right end, where the
    centred frame titles leave room."""
    ax.set_xlabel("time (s)")
    ax.xaxis.label.set_ha("right")
    ax.xaxis.set_label_coords(1.0, -0.40)


def letter(ax, s: str, dx: float = -0.085, dy: float = 1.06) -> None:
    ax.text(dx, dy, s, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold")


def frame_row(fig, gs, row, images, times, vmin, vmax, label=None, cmap="gray",
              show_time=True):
    """One row of frames with a display range common to the whole figure, time above each."""
    axes = []
    for j, (im, t) in enumerate(zip(images, times)):
        ax = fig.add_subplot(gs[row, j])
        ax.imshow(np.clip(im, vmin, vmax), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        if show_time:
            ax.set_title(f"{t:.0f} s", fontsize=7.5, pad=1.5)
        if j == 0 and label:
            ax.set_ylabel(label, fontsize=8)
        axes.append(ax)
    return axes


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    for folder in (OUT, LOCAL):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(folder / f"{stem}.{ext}")
    plt.close(fig)
    print(f"written {stem}.{{png,pdf,svg}} to {OUT} and {LOCAL}", flush=True)


# --------------------------------------------------------------------------- Figure 9
def figure_9(process="GMAW", sid="exp_05", design=("GTAW", "Iden_02"), n_show=5):
    d = np.load(RES / f"3a_{process}_{sid}.npz")
    g = np.load(RES / f"3d_{design[0]}_{design[1]}.npz")
    t, speed, delta = d["t"], d["speed"], d["delta"]
    z_obs, z_pred = d["z_obs"], d["z_pred"]
    coords = np.argsort(-z_obs.std(0))[:3]          # the three liveliest coordinates

    fig = plt.figure(figsize=(WIDTH_IN, 8.6))
    gs = fig.add_gridspec(7, n_show,
                          height_ratios=[0.72, 1.15, 0.62, 1.22, 1.22, 0.72, 1.45],
                          left=0.100, right=0.985, top=0.960, bottom=0.050,
                          hspace=0.46, wspace=0.07)

    ax = fig.add_subplot(gs[0, :]); clean(ax); letter(ax, "(a)")
    ax.plot(t, speed, color=OBSERVED, lw=1.0)
    ax.set_ylabel("travel speed\n(mm s$^{-1}$)"); ax.set_xlim(t[0], t[-1])
    ax.set_xticklabels([])

    ax = fig.add_subplot(gs[1, :]); clean(ax); letter(ax, "(b)")
    for i, c in enumerate(coords):
        off, m = i * 3.2, z_obs[:, c].mean()
        sd = max(z_obs[:, c].std(), 1e-9)
        ax.plot(t, (z_obs[:, c] - m) / sd + off, color=OBSERVED, lw=0.9)
        ax.plot(t, (z_pred[:, c] - m) / sd + off, color=WITH_DELTA, lw=0.9, ls=(0, (4, 1.6)))
        ax.text(t[-1], off, f"  $z_{{{c + 1}}}$", va="center", fontsize=8, color=OBSERVED)
    ax.plot([], [], color=OBSERVED, lw=0.9, label="recorded")
    ax.plot([], [], color=WITH_DELTA, lw=0.9, ls=(0, (4, 1.6)), label="two-state NODE")
    ax.legend(loc="upper left", ncol=2, borderaxespad=0.2, handlelength=2.2)
    ax.set_ylabel("pool state\n(standardized, offset)"); ax.set_yticks([])
    ax.set_xlim(t[0], t[-1] * 1.05); ax.set_xticklabels([])

    ax = fig.add_subplot(gs[2, :]); clean(ax); letter(ax, "(c)")
    ax.plot(t, delta[:, 0] if delta.ndim > 1 else delta, color=WITH_DELTA, lw=1.0)
    ax.set_ylabel("thermal state\n$\\delta$ (a.u.)")
    right_time_label(ax)
    ax.set_xlim(t[0], t[-1])

    idx = np.linspace(0, len(t) - 1, n_show).round().astype(int)
    vmin, vmax = 0.0, float(max(d["real"].max(), d["gen"].max()))
    frame_row(fig, gs, 3, d["real"][idx], t[idx], vmin, vmax, label="recorded")
    frame_row(fig, gs, 4, d["gen"][idx], t[idx], vmin, vmax, label="generated", show_time=False)
    fig.text(0.012, gs[3, 0].get_position(fig).y1, "(d)", fontsize=9, fontweight="bold",
             va="bottom")

    td, vd = g["t"], g["speed"]
    first, back = g["first"], g["back"]
    ax = fig.add_subplot(gs[5, :]); clean(ax); letter(ax, "(e)")
    ax.plot(td, vd, color=OBSERVED, lw=1.0)
    for w, lab, ha in ((first, "first hold", "center"), (back, "after the return", "right")):
        ax.axvspan(td[w[0]], td[w[-1]], color="#e8edf3", zorder=0)
        x = td[w].mean() if ha == "center" else td[w[-1]]
        ax.text(x, vd.max(), lab, ha=ha, va="bottom", fontsize=7.5, color=THERMAL)
    ax.set_ylabel("travel speed\n(mm s$^{-1}$)")
    right_time_label(ax)
    ax.set_xlim(td[0], td[-1]); ax.set_ylim(vd.min() - 0.4, vd.max() + 1.1)

    panels = []
    for name, colour in (("A", WITH_DELTA), ("B", WITHOUT_DELTA)):
        st = g[f"{name}_struct"]
        panels.append((name, colour, st[first].mean(0), st[back].mean(0),
                       st[back].mean(0) - st[first].mean(0)))
    lim = max(float(np.abs(q[4]).max()) for q in panels)
    smin = min(float(min(q[2].min(), q[3].min())) for q in panels)
    smax = max(float(max(q[2].max(), q[3].max())) for q in panels)
    box = gs[6, 0].get_position(fig)
    sub = fig.add_gridspec(2, 3, left=0.215, right=0.80, top=box.y1, bottom=box.y0,
                           hspace=0.08, wspace=0.06)
    for r, (name, colour, a_im, b_im, dif) in enumerate(panels):
        for c, (im, ttl, cmap, kw) in enumerate((
                (a_im, "first hold", "gray", dict(vmin=smin, vmax=smax)),
                (b_im, "after the return", "gray", dict(vmin=smin, vmax=smax)),
                (dif, "difference", "RdBu_r", dict(vmin=-lim, vmax=lim)))):
            ax = fig.add_subplot(sub[r, c])
            ax.imshow(im, cmap=cmap, **kw)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GRID)
            if r == 0:
                ax.set_title(ttl, fontsize=7.5, pad=1.5)
            if c == 0:
                ax.set_ylabel("two-state\nNODE" if name == "A" else "no thermal\nstate",
                              fontsize=8, color=colour, rotation=0, ha="right",
                              va="center", labelpad=6)
    fig.text(0.012, box.y1, "(f)", fontsize=9, fontweight="bold", va="bottom")
    save(fig, "Figure_9_input_driven_generation")


# --------------------------------------------------------------------------- Figure 10
def figure_10(waam="video_3", freeze=("GTAW", "Iden_02"), n_show=5):
    w = np.load(RES / "3c_WAAM.npz")
    f = np.load(RES / f"3b_{freeze[0]}_{freeze[1]}.npz")
    tw = w[f"{waam}_t"]
    obs, pa, pb = w[f"{waam}_obs"], w[f"{waam}_A_m1_pred"], w[f"{waam}_B_no_delta_pred"]
    da = w[f"{waam}_A_m1_delta"]
    coords = np.argsort(-obs.std(0))[:2]

    t, k, t_f = f["t"], int(f["k"]), float(f["t_f"])
    idx = np.linspace(k, len(t) - 1, n_show).round().astype(int)

    fig = plt.figure(figsize=(WIDTH_IN, 8.5))
    gs = fig.add_gridspec(7, n_show, height_ratios=[1.05, 0.62, 1.22, 1.22, 1.22, 1.22, 0.80],
                          left=0.100, right=0.985, top=0.962, bottom=0.055,
                          hspace=0.44, wspace=0.07)

    ax = fig.add_subplot(gs[0, :]); clean(ax); letter(ax, "(a)")
    for i, c in enumerate(coords):
        off = i * 3.2
        m, s = obs[:, c].mean(), max(obs[:, c].std(), 1e-9)
        ax.plot(tw, (obs[:, c] - m) / s + off, color=OBSERVED, lw=0.9)
        ax.plot(tw, (pa[:, c] - m) / s + off, color=WITH_DELTA, lw=0.9, ls=(0, (4, 1.6)))
        ax.plot(tw, (pb[:, c] - m) / s + off, color=WITHOUT_DELTA, lw=0.9, ls=(0, (2, 1.4)))
        ax.text(tw[-1], off, f"  $z_{{{c + 1}}}$", va="center", fontsize=8, color=OBSERVED)
    ax.plot([], [], color=OBSERVED, lw=0.9, label="measured")
    ax.plot([], [], color=WITH_DELTA, lw=0.9, ls=(0, (4, 1.6)), label="two-state NODE")
    ax.plot([], [], color=WITHOUT_DELTA, lw=0.9, ls=(0, (2, 1.4)), label="without thermal state")
    ax.legend(loc="upper center", ncol=3, borderaxespad=0.1, handlelength=2.2,
              bbox_to_anchor=(0.5, 1.26))
    ax.set_ylabel("pool state\n(standardized, offset)"); ax.set_yticks([])
    ax.set_xlim(tw[0], tw[-1] * 1.05); ax.set_xticklabels([])

    ax = fig.add_subplot(gs[1, :]); clean(ax); letter(ax, "(b)")
    ax.plot(tw, da[:, 0] if da.ndim > 1 else da, color=WITH_DELTA, lw=1.0)
    ax.set_ylabel("thermal state\n$\\delta$ (a.u.)")
    right_time_label(ax)
    ax.set_xlim(tw[0], tw[-1])

    vmin, vmax = 0.0, float(max(f["real"].max(), f["gen_id"].max(), f["gen_frozen"].max()))
    for r, (key, lab) in enumerate((("real", "recorded"), ("gen_id", "identified"),
                                    ("gen_frozen", f"$\\delta$ frozen at {t_f:.0f} s"))):
        frame_row(fig, gs, 2 + r, f[key][idx], t[idx], vmin, vmax, label=lab,
                  show_time=(r == 0))
    fig.text(0.012, gs[2, 0].get_position(fig).y1, "(c)", fontsize=9,
             fontweight="bold", va="bottom")

    diff = f["struct_id"][idx] - f["struct_frozen"][idx]
    lim = float(np.abs(diff).max())
    for j in range(n_show):
        ax = fig.add_subplot(gs[5, j])
        im = ax.imshow(diff[j], cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        if j == 0:
            ax.set_ylabel("identified\n$-$ frozen", fontsize=8, labelpad=2)
    cax = fig.add_axes([0.40, gs[5, 0].get_position(fig).y0 - 0.016, 0.20, 0.006])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-lim, 0, lim]); cb.set_ticklabels([f"$-${lim:.2f}", "0", f"{lim:.2f}"])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.5)
    fig.text(0.012, gs[5, 0].get_position(fig).y1, "(d)", fontsize=9,
             fontweight="bold", va="bottom")

    bot = gs[6, 0].get_position(fig)
    pair = fig.add_gridspec(1, 2, left=0.100, right=0.985, top=bot.y1, bottom=bot.y0,
                            wspace=0.42, width_ratios=[1.35, 1.0])
    ax = fig.add_subplot(pair[0]); clean(ax); letter(ax, "(e)", dx=-0.14)
    ax.plot(t, f["E_id"], color=WITH_DELTA, lw=1.0, label="identified")
    ax.plot(t, f["E_frozen"], color=WITHOUT_DELTA, lw=1.0, label="$\\delta$ frozen")
    ax.axvline(t_f, color=THERMAL, lw=0.7, ls=(0, (3, 2)))
    ax.set_ylabel("$E$ (grey levels)", labelpad=2); ax.set_xlabel("time (s)")
    ax.set_xlim(t[0], t[-1]); ax.legend(loc="upper left", borderaxespad=0.2, handlelength=2.0)

    ax = fig.add_subplot(pair[1]); clean(ax); letter(ax, "(f)", dx=-0.19)
    ax.plot(t, f["delta_id"][:, 0], color=WITH_DELTA, lw=1.0)
    ax.plot(t, f["delta_frozen"][:, 0], color=WITHOUT_DELTA, lw=1.0, ls=(0, (4, 1.6)))
    ax.axvline(t_f, color=THERMAL, lw=0.7, ls=(0, (3, 2)))
    ax.set_ylabel("$\\delta$ (a.u.)", labelpad=2); ax.set_xlabel("time (s)")
    ax.set_xlim(t[0], t[-1])
    save(fig, "Figure_10_thermal_state_role")


if __name__ == "__main__":
    style()
    todo = sys.argv[1:] or ["9", "10"]
    for n in todo:
        {"9": figure_9, "10": figure_10}[n]()
