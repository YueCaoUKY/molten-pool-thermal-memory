#!/usr/bin/env python3
"""Figures 9, 10 and 11 of Section 6: generation from the input alone, one figure per process.

The typography, the grid, the panel captions and the image rows are copied from the script behind
Figure 7 (`step_test_pool_sequence.py`); the two model colours are the ones Figures 5 and 6 use
for the two-state NODE and for the model without the thermal state.

    python sec6_figures_v3.py 9 10 11
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results_sec6"
OUT = HERE.parent.parent / "Manuscript" / "figures"
LOCAL = HERE / "figures_sec6"

OBSERVED = "#252525"        # recorded, as in Figures 5 and 6
WITH_DELTA = "#2166ac"      # two-state NODE
WITHOUT_DELTA = "#d95f02"   # without the thermal state
FRAME_EDGE = "0.55"
WIDTH_IN = 180.0 / 25.4
STATE_NAMES = {0: "z_1", 1: "z_2", 2: "z_3", 3: "z_4", 4: "z_5", 5: "z_6"}


def style() -> None:
    """The rcParams of the Figure 7 script, unchanged except for the export resolution."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 0.9,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "savefig.dpi": 600,
        "savefig.bbox": None,
        "svg.fonttype": "none",
    })


def trace_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="0.88", lw=0.4)
    ax.set_axisbelow(True)


def image_axis(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(FRAME_EDGE)
        sp.set_linewidth(0.5)


def save(fig, stem: str) -> None:
    for d in (OUT, LOCAL):
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(d / f"{stem}.{ext}")
    plt.close(fig)
    print(f"written {stem}.{{png,pdf,svg}}", flush=True)


def numbers(process: str, record: str) -> dict:
    r = pd.read_csv(HERE / "per_record.csv")
    r = r[(r.process == process) & (r.record == record)].set_index("variant")
    s = pd.DataFrame(json.loads((RES / "ssim_per_record.json").read_text()))
    s = s[(s.process == process) & (s.record == record)].iloc[0]
    return dict(r2_A=float(r.loc["A_m1", "r2_smoothed"]),
                r2_B=float(r.loc["B_no_delta", "r2_smoothed"]),
                ssim_A=float(s.ssim_A), ssim_B=float(s.ssim_B))


def pick_times(t, speed, n=5):
    """Five evenly spaced times; where the speed varies, nudge each onto the nearest sample whose
    speed is farthest from the record mean, so that slow and fast segments are both represented."""
    idx = np.linspace(0, len(t) - 1, n).round().astype(int)
    if float(np.ptp(speed)) < 1e-9:
        return idx
    half = max(int(0.04 * len(t)), 1)
    out = []
    for k in idx:
        lo, hi = max(k - half, 0), min(k + half, len(t) - 1)
        w = np.abs(speed[lo:hi + 1] - speed.mean())
        out.append(lo + int(np.argmax(w)))
    return np.array(sorted(set(out)))


def build(process: str, record: str, stem: str, with_speed: bool, with_difference: bool,
          speed_note: str | None = None, n_show: int = 5, frame_mm: float = 20.0) -> None:
    """Every vertical distance is set here in inches, rather than through a shared `hspace`,
    so that the rows of frames sit close together and the figure stays compact."""
    d = np.load(RES / f"gen_{process}_{record}.npz")
    num = numbers(process, record)
    t, speed = d["t"], d["speed"]
    z_obs, zA, zB = d["z_obs"], d["A_z"], d["B_z"]
    coord = int(np.argmax(z_obs.std(0)))
    idx = pick_times(t, speed, n_show)
    n_show = len(idx)

    rows = [("recorded", d["real"]), ("with $\\delta$", d["A_gen"]),
            ("without $\\delta$", d["B_gen"])]
    vmin, vmax = 0.0, float(max(r[1].max() for r in rows))
    n_img = len(rows) + (1 if with_difference else 0)

    # vertical plan, in inches
    cell = frame_mm / 25.4
    TOP, BOT = 0.06, 0.42
    CAP, GAP_IMG, ROW_GAP, TIMES, GAP_TR = 0.16, 0.12, 0.035, 0.145, 0.18
    SPEED_H, TRACE_H, DELTA_H, GAP_D = 0.40, 0.62, 0.46, 0.16
    H = (TOP + (CAP + SPEED_H + GAP_IMG if with_speed else 0) + CAP
         + n_img * cell + (n_img - 1) * ROW_GAP + TIMES + GAP_TR
         + CAP + TRACE_H + GAP_D + DELTA_H + BOT)
    fig = plt.figure(figsize=(WIDTH_IN, H))

    L, R = 0.085, 0.985                       # the traces use the full column width
    band = n_show * cell / WIDTH_IN           # the frames are centred in their own band
    IL = 0.5 - band / 2 + (0.018 if with_difference else 0.0)
    y = 1.0 - TOP / H                         # current top edge, in figure fractions

    def axes(y_top, height, left=L, right=R):
        return fig.add_axes([left, y_top - height / H, right - left, height / H])

    if with_speed:
        fig.text(L, y - CAP * 0.62 / H, "(a) travel speed", ha="left", va="bottom", fontsize=9.5)
        y -= CAP / H
        ax = axes(y, SPEED_H); trace_axis(ax)
        ax.plot(t, speed, color=OBSERVED, lw=0.9)
        ax.set_ylabel("travel speed\n(mm s$^{-1}$)")
        ax.set_xlim(t[0], t[-1]); ax.set_xticklabels([])
        y -= (SPEED_H + GAP_IMG) / H

    fig.text(L, y - CAP * 0.62 / H,
             f"({'b' if with_speed else 'a'}) recorded and generated frames",
             ha="left", va="bottom", fontsize=9.5)
    y -= CAP / H

    series = [r[1] for r in rows] + ([d["A_struct"] - d["B_struct"]] if with_difference else [])
    labels = [r[0] for r in rows] + (["difference"] if with_difference else [])
    lim = float(np.abs(series[-1][idx]).max()) if with_difference else 1.0
    im_diff = None
    for i, (lab, arr) in enumerate(zip(labels, series)):
        for j, k in enumerate(idx):
            ax = fig.add_axes([IL + j * cell / WIDTH_IN, y - cell / H, cell / WIDTH_IN, cell / H])
            if with_difference and i == n_img - 1:
                im_diff = ax.imshow(arr[k], cmap="RdBu_r", vmin=-lim, vmax=lim,
                                    interpolation="none")
            else:
                ax.imshow(np.clip(arr[k], vmin, vmax), cmap="gray", vmin=vmin, vmax=vmax,
                          interpolation="none")
            image_axis(ax)
            if j == 0:
                ax.set_ylabel(lab, labelpad=3)
            if i == n_img - 1:
                ax.text(0.5, -0.10, rf"$t={t[k]:.0f}\,\mathrm{{s}}$", transform=ax.transAxes,
                        ha="center", va="top", fontsize=8.5)
        if im_diff is not None and i == n_img - 1:
            cax = fig.add_axes([IL + n_show * cell / WIDTH_IN + 0.012, y - cell / H,
                                0.009, cell / H])
            cb = fig.colorbar(im_diff, cax=cax)
            cb.set_ticks([-lim, 0, lim])
            cb.set_ticklabels([f"$-${lim:.2f}", "0", f"{lim:.2f}"])
            cb.ax.tick_params(labelsize=7, length=1.6)
            cb.outline.set_linewidth(0.4)
            cb.set_label("grey level", fontsize=7.5, labelpad=2)
        y -= (cell + (ROW_GAP if i < n_img - 1 else 0)) / H
    y -= (TIMES + GAP_TR) / H

    fig.text(L, y - CAP * 0.62 / H,
             f"({'c' if with_speed else 'b'}) pool state and thermal state",
             ha="left", va="bottom", fontsize=9.5)
    y -= CAP / H

    ax_top = axes(y, TRACE_H); trace_axis(ax_top)
    m, sd = z_obs[:, coord].mean(), max(z_obs[:, coord].std(), 1e-9)
    ax_top.plot(t, (z_obs[:, coord] - m) / sd, color=OBSERVED, lw=0.9, label="recorded")
    ax_top.plot(t, (zA[:, coord] - m) / sd, color=WITH_DELTA, lw=0.9, ls=(0, (4, 1.6)),
                label="with $\\delta$")
    ax_top.plot(t, (zB[:, coord] - m) / sd, color=WITHOUT_DELTA, lw=0.9, ls=(0, (1.4, 1.4)),
                label="without $\\delta$")
    ax_top.set_ylabel(f"${STATE_NAMES[coord]}$\n(standardized)")
    ax_top.set_xlim(t[0], t[-1]); ax_top.set_xticklabels([])
    ax_top.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=3, columnspacing=1.5,
                  handletextpad=0.6, borderaxespad=0.0)
    y -= (TRACE_H + GAP_D) / H

    ax_d = axes(y, DELTA_H); trace_axis(ax_d)
    ax_d.plot(t, d["A_delta"][:, 0], color=WITH_DELTA, lw=0.9)
    ax_d.set_ylabel("thermal state\n$\\delta$ (a.u.)")
    ax_d.set_xlabel("time (s)"); ax_d.set_xlim(t[0], t[-1])
    if speed_note:
        ax_d.text(0.995, 0.08, speed_note, transform=ax_d.transAxes, ha="right",
                  va="bottom", fontsize=8, color="0.35")
    save(fig, stem)


def figure_9():
    build("GMAW", "exp_05", "Figure_9_generation_GMAW", with_speed=True, with_difference=False)


def figure_10():
    build("GTAW", "Iden_02", "Figure_10_generation_GTAW", with_speed=True, with_difference=True)


def figure_11():
    build("WAAM", "video_3", "Figure_11_generation_WAAM", with_speed=False, with_difference=True,
          speed_note="constant input: 200 A, 2.7 mm s$^{-1}$")


if __name__ == "__main__":
    style()
    for n in (sys.argv[1:] or ["9", "10", "11"]):
        {"9": figure_9, "10": figure_10, "11": figure_11}[n]()
