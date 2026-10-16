#!/usr/bin/env python3
"""Figure 11: WAAM video_3 at constant input, the two-state NODE against the same model with its
thermal state held fixed from t_f onwards.

Input: results_sec6/3b_WAAM_video_3.npz, the output of the 3b freeze test for WAAM video_3 (one
identified two-state model, rolled out twice from the same first state; in the second run delta is
held at its value at t_f, chosen by the same nearest-recorded-state criterion as the GTAW freeze
test). That file is about 230 MB, so the script also accepts results_sec6/3b_WAAM_video_3_fig11.npz,
which holds the traces and only the seven frames shown; `--write-slim` makes it from the full file.
Nothing is re-identified and nothing is re-run here.

Layout and typography follow Figures 9 and 10 of the manuscript: style() of sec6_figures_v3.py,
160 mm wide, seven frames at the centres of seven equal time bins, panel captions under the
panels, 600 dpi, transparent background.

    python sec6_figure_11_fixed_delta.py              # draw the figure
    python sec6_figure_11_fixed_delta.py --write-slim # full npz -> the small figure-data npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sec6_figures_v3 as F  # noqa: E402  style() and the colours of Figures 9-11

RES = HERE / "results_sec6"
FULL = RES / "3b_WAAM_video_3.npz"
SLIM = RES / "3b_WAAM_video_3_fig11.npz"
FRAMES = ("real", "gen_id", "gen_frozen", "struct_id", "struct_frozen")
TRACES = ("z_obs", "z_id", "z_frozen", "delta_id", "delta_frozen", "E_id", "E_frozen")
OUT = HERE.parent.parent / "Manuscript" / "figures"
LOCAL = HERE / "figures_sec6"
STEM = "Figure_11_generation_WAAM_fixed_delta"

FIXED = "#1b9e77"           # delta held fixed: a third colour, so it is not read as "without delta"
FIXED_LS = (0, (5.0, 1.4, 1.2, 1.4))
TWO_LS = (0, (4.0, 1.6))
GUIDE = "0.62"
WIDTH_IN = 160.0 / 25.4


def bin_centres(t: np.ndarray, n: int = 7) -> np.ndarray:
    """Frame indices at the centres of n equal time bins, as in Figures 9 and 10."""
    T = t[-1] + (t[1] - t[0])
    want = (np.arange(n) + 0.5) * T / n
    return np.array([int(np.argmin(np.abs(t - w))) for w in want])


def image_axis(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(F.FRAME_EDGE); sp.set_linewidth(0.5)


def trace_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", color="0.88", lw=0.4)
    ax.set_axisbelow(True)


def load():
    """Traces over the whole record and the seven frames at the bin centres."""
    if FULL.exists():
        d = np.load(FULL)
        n = min(len(d["real"]), len(d["gen_id"]), len(d["gen_frozen"]))
        t = d["t"][:n]
        idx = bin_centres(t)
        frames = {k: d[k][idx] for k in FRAMES}
        traces = {k: d[k][:n] for k in TRACES}
        meta = dict(t=t, idx=idx, t_f=float(d["t_f"]), k=int(d["k"]), dt=float(d["dt"]))
    else:
        d = np.load(SLIM)
        frames = {k: d[k] for k in FRAMES}
        traces = {k: d[k] for k in TRACES}
        meta = dict(t=d["t"], idx=d["frame_idx"], t_f=float(d["t_f"]), k=int(d["k"]),
                    dt=float(d["dt"]))
    return frames, traces, meta


def write_slim() -> None:
    frames, traces, meta = load()
    np.savez_compressed(SLIM, **frames, **traces, t=meta["t"], frame_idx=meta["idx"],
                        t_f=meta["t_f"], k=meta["k"], dt=meta["dt"])
    print(f"written {SLIM.name} ({SLIM.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    F.style()
    frames, traces, meta = load()
    t, idx, t_f = meta["t"], meta["idx"], meta["t_f"]
    z_obs, z_id, z_fr = traces["z_obs"], traces["z_id"], traces["z_frozen"]
    coord = 1                                    # z_2, the coordinate of the published Figure 11
    assert coord == int(np.argmax(z_obs.std(0))), "z_2 is no longer the most variable coordinate"

    rows = [("recorded", frames["real"]),
            ("two-state\nNODE", frames["gen_id"]),
            ("$\\hat{\\delta}$ fixed\nafter $t_f$", frames["gen_frozen"])]
    vmax = float(max(r[1].max() for r in rows))
    diff = frames["struct_id"] - frames["struct_frozen"]
    lim = float(np.abs(diff).max())

    # vertical plan in inches
    cell = 0.95 * (WIDTH_IN * 0.80) / 7
    gap_c = 0.035
    TOP, ROWG, DIFF_T, TIMES, CAP_A, GAP_B, LEG = 0.04, 0.035, 0.34, 0.20, 0.30, 0.20, 0.22
    TR1, TRG, TR2, XLAB, CAP_B = 0.95, 0.12, 0.70, 0.44, 0.28
    H = (TOP + 3 * cell + 2 * ROWG + DIFF_T + cell + TIMES + CAP_A + GAP_B + LEG
         + TR1 + TRG + TR2 + XLAB + CAP_B)
    fig = plt.figure(figsize=(WIDTH_IN, H))
    fig.patch.set_alpha(0.0)

    L, R = 0.112, 0.902                          # axes extent, in figure fractions
    step = (R - L) / 7
    w = step - gap_c / WIDTH_IN
    y = 1.0 - TOP / H

    def frame_row(y_top, arr, label, cmap="gray", vlim=None):
        ims = []
        for j in range(len(idx)):
            ax = fig.add_axes([L + j * step + (step - w) / 2, y_top - cell / H, w, cell / H])
            if vlim is None:
                ims.append(ax.imshow(np.clip(arr[j], 0, vmax), cmap=cmap, vmin=0, vmax=vmax,
                                     interpolation="none"))
            else:
                ims.append(ax.imshow(arr[j], cmap=cmap, vmin=-vlim, vmax=vlim,
                                     interpolation="none"))
            image_axis(ax)
            if j == 0:
                ax.set_ylabel(label, labelpad=4, fontsize=10, linespacing=1.05)
        return ims

    for i, (lab, arr) in enumerate(rows):
        frame_row(y, arr, lab)
        y -= (cell + (ROWG if i < 2 else 0)) / H

    y -= DIFF_T / H
    fig.text(0.5 * (L + R), y + 0.11 / H,
             "difference of $D(\\hat{\\mathbf{z}})$:  two-state NODE $-$ "
             "$\\hat{\\delta}$ fixed after $t_f$",
             ha="center", va="bottom", fontsize=10)
    ims = frame_row(y, diff, "difference", cmap="RdBu_r", vlim=lim)
    cax = fig.add_axes([R + 0.010, y - cell / H, 0.011, cell / H])
    cb = fig.colorbar(ims[-1], cax=cax)
    cb.set_ticks([-lim, 0, lim])
    cb.set_ticklabels([f"$-${lim:.2f}", "0", f"{lim:.2f}"])
    cb.ax.tick_params(labelsize=9.5, length=2.0, width=0.5)
    cb.outline.set_linewidth(0.4)
    for j, k in enumerate(idx):
        fig.text(L + (j + 0.5) * step, y - cell / H - 0.035 / H, f"$t = {t[k]:.0f}$ s",
                 ha="center", va="top", fontsize=10)
    y -= (cell + TIMES) / H
    fig.text(0.5 * (L + R), y - 0.04 / H, "(a) Recorded and generated frames",
             ha="center", va="top", fontsize=10)
    y -= (CAP_A + GAP_B + LEG) / H

    m, sd = z_obs[:, coord].mean(), max(z_obs[:, coord].std(), 1e-9)
    ax1 = fig.add_axes([L, y - TR1 / H, R - L, TR1 / H]); trace_axis(ax1)
    ax1.plot(t, (z_obs[:, coord] - m) / sd, color=F.OBSERVED, lw=1.0, label="recorded")
    ax1.plot(t, (z_id[:, coord] - m) / sd, color=F.WITH_DELTA, lw=1.0, ls=TWO_LS,
             label="two-state NODE")
    ax1.plot(t, (z_fr[:, coord] - m) / sd, color=FIXED, lw=1.0, ls=FIXED_LS,
             label="$\\hat{\\delta}$ fixed after $t_f$")
    ax1.set_ylabel("$z_2$\n(standardized)", fontsize=10, linespacing=1.05)
    ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, columnspacing=1.6,
               handletextpad=0.6, borderaxespad=0.15, fontsize=10)
    y -= (TR1 + TRG) / H

    ax2 = fig.add_axes([L, y - TR2 / H, R - L, TR2 / H], sharex=ax1); trace_axis(ax2)
    ax2.plot(t, traces["delta_id"][:, 0], color=F.WITH_DELTA, lw=1.0)
    ax2.plot(t, traces["delta_frozen"][:, 0], color=FIXED, lw=1.0, ls=FIXED_LS)
    ax2.set_ylabel("thermal state\n$\\hat{\\delta}$ (a.u.)", fontsize=10, linespacing=1.05)
    ax2.set_xlabel("time (s)", fontsize=10, labelpad=2)

    for ax in (ax1, ax2):
        for k in idx:
            ax.axvline(t[k], color=GUIDE, lw=0.6, ls=(0, (1.5, 1.5)), zorder=0)
        ax.axvline(t_f, color="0.25", lw=0.7, ls="-", zorder=1)
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(labelsize=10)
    ax1.tick_params(labelbottom=False)
    ax2.text(t_f + 0.25, 0.06, f"$t_f = {t_f:.1f}$ s", transform=ax2.get_xaxis_transform(),
             ha="left", va="bottom", fontsize=9.5, color="0.25",
             bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax2.set_ylim(bottom=0)
    y -= (TR2 + XLAB) / H
    fig.text(0.5 * (L + R), y - 0.02 / H, "(b) Pool state and thermal state",
             ha="center", va="top", fontsize=10)

    for dd in (OUT, LOCAL):
        dd.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(dd / f"{STEM}.{ext}", dpi=600, transparent=True)
    plt.close(fig)
    print(f"written {STEM}.{{png,pdf,svg}}; t_f = {t_f:.2f} s, frames at "
          f"{', '.join(f'{t[k]:.1f}' for k in idx)} s, |diff| max {lim:.3f}")


if __name__ == "__main__":
    write_slim() if "--write-slim" in sys.argv[1:] else main()
