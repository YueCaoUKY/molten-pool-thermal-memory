#!/usr/bin/env python3
"""Supplementary videos, one per process, from the arrays behind Figures 9, 10 and 11.

Three views side by side — the recording, the frame the model generates from the input alone, and
the same model without the thermal state — with the travel speed and the thermal state below them
and the time of the recording at the top right. Playback is real time.

The type and the colours are those of Figures 9 to 11.

    python sec6_videos.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import imageio_ffmpeg
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sec6_figures_v3 as F   # noqa: E402  the style and colours of Figures 9-11

RES = HERE / "results_sec6"
OUT = HERE.parent.parent / "Manuscript" / "videos"
TRACK = "#EDEFF1"
RECORDS = [("GMAW", "exp_05", "Video_S1_GMAW", "Gas metal arc welding"),
           ("GTAW", "Iden_02", "Video_S2_GTAW", "Gas tungsten arc welding"),
           ("WAAM", "video_3", "Video_S3_WAAM", "Wire arc additive manufacturing")]
WAAM_NOTE = "constant input: 200 A, 2.7 mm s$^{-1}$"


def bar(ax, label, colour, vmax, unit=""):
    """A labelled bar; the label sits in the left margin and the value beside the bar."""
    ax.set_xlim(0, vmax); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.barh(0.5, vmax, height=0.46, color=TRACK, zorder=1)
    patch = ax.barh(0.5, 0.0, height=0.46, color=colour, zorder=2)[0]
    ax.text(-0.015 * vmax, 0.5, label, ha="right", va="center", fontsize=15, color=F.OBSERVED)
    value = ax.text(vmax * 1.015, 0.5, "", ha="left", va="center", fontsize=15, color=colour)
    return patch, value


def build(process: str, sid: str, stem: str, title: str) -> None:
    d = np.load(RES / f"gen_{process}_{sid}.npz")
    t, speed = d["t"], d["speed"]
    dt = float(d["dt"])
    fps = int(round(1.0 / dt))
    rows = [("recorded", d["real"], F.OBSERVED),
            ("with $\\delta$", d["A_gen"], F.WITH_DELTA),
            ("without $\\delta$", d["B_gen"], F.WITHOUT_DELTA)]
    n = min(len(r[1]) for r in rows)
    vmin, vmax = 0.0, float(max(r[1][:n].max() for r in rows))
    delta = d["A_delta"][:, 0]
    dmax = float(delta.max()) * 1.12
    constant = float(np.ptp(speed)) < 1e-9
    smax = max(float(np.ceil(speed.max())), 1.0)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=150)  # 1920 x 1080
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(4, 3, height_ratios=[0.80, 2.25, 0.40, 0.40],
                          left=0.175, right=0.905, top=0.955, bottom=0.075,
                          hspace=0.25, wspace=0.045)

    ax_t = fig.add_subplot(gs[0, :]); ax_t.set_axis_off()
    ax_t.text(0.0, 0.92, title, ha="left", va="top", fontsize=22, color=F.OBSERVED)
    ax_t.text(0.0, 0.06,
              "generated from the first frame alone; the input never changes" if constant
              else "generated from the recorded travel speed and the first frame alone",
              ha="left", va="bottom", fontsize=14, color="0.45")
    clock = ax_t.text(1.0, 0.92, "", ha="right", va="top", fontsize=20, color="0.45")

    ims = []
    for j, (label, series, colour) in enumerate(rows):
        ax = fig.add_subplot(gs[1, j])
        ims.append(ax.imshow(series[0], cmap="gray", vmin=vmin, vmax=vmax, animated=True,
                             interpolation="none"))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(F.FRAME_EDGE); sp.set_linewidth(1.0)
        ax.set_title(label, fontsize=17, color=colour, pad=7)

    ax_s = fig.add_subplot(gs[2, :])
    if constant:
        ax_s.set_axis_off()
        ax_s.text(0.0, 0.5, WAAM_NOTE, ha="left", va="center", fontsize=15, color="0.45",
                  transform=ax_s.transAxes)
        sp_patch = sp_value = None
    else:
        sp_patch, sp_value = bar(ax_s, "travel speed", F.OBSERVED, smax)

    ax_d = fig.add_subplot(gs[3, :])
    d_patch, d_value = bar(ax_d, "thermal state $\\delta$ (a.u.)", F.WITH_DELTA, dmax)

    OUT.mkdir(parents=True, exist_ok=True)
    still = OUT / f"{stem}.png"
    writer = FFMpegWriter(fps=fps, bitrate=4200, metadata=dict(title=stem))
    t0 = time.time()
    with writer.saving(fig, str(OUT / f"{stem}.mp4"), dpi=150):
        for k in range(n):
            clock.set_text(f"{t[k]:5.1f} s")
            for im, (_, series, _c) in zip(ims, rows):
                im.set_data(series[k])
            if sp_patch is not None:
                sp_patch.set_width(speed[k])
                sp_value.set_text(f"{speed[k]:.1f} mm s" + "$^{-1}$")
            d_patch.set_width(delta[k])
            d_value.set_text(f"{delta[k]:.2f}")
            writer.grab_frame()
            if k == int(0.55 * n):
                fig.savefig(still, dpi=150)
    plt.close(fig)
    print(f"{stem}: {n} frames, {n / fps:.0f} s at {fps} fps, {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    F.style()
    for args in RECORDS:
        build(*args)
