#!/usr/bin/env python3
"""Supplementary videos, one per process: what the thermal state does to the generated images.

Three views side by side: the recording, the frames the identified two-state NODE generates from
the first frame and the recorded input, and the frames the same model generates when its thermal
state is held at its value at t_f. Below them: the travel speed (for WAAM, the constant input) and
the thermal state of both runs. The time of the recording is at the top right; playback is at the
recording rate rounded to a whole number of frames per second.

Inputs are the 3b freeze-test arrays (sec6_experiments.part_3b, or numpy_port/run_3b_numpy.py):
    results_sec6/3b_GMAW_exp_05.npz, 3b_GTAW_Iden_02.npz, 3b_WAAM_video_3.npz

The type and the colours are those of Figures 9-11.

    python sec6_videos_fixed_delta.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
try:
    import imageio_ffmpeg
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:                      # fall back to an ffmpeg on the PATH
    pass
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sec6_figures_v3 as F   # noqa: E402  the style and colours of Figures 9-11

RES = HERE / "results_sec6"
OUT = HERE.parent.parent / "Manuscript" / "videos"
TRACK = "#EDEFF1"
FIXED = "#1b9e77"             # as in Figure 11
RECORDS = [("GMAW", "exp_05", "fixed_delta_GMAW", "Gas metal arc welding"),
           ("GTAW", "Iden_02", "fixed_delta_GTAW", "Gas tungsten arc welding"),
           ("WAAM", "video_3", "fixed_delta_WAAM", "Wire arc additive manufacturing")]
WAAM_NOTE = "constant input: 200 A, 2.7 mm s$^{-1}$"


def bar(ax, label, colour, vmax):
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
    d = np.load(RES / f"3b_{process}_{sid}.npz")
    dt = float(d["dt"])
    t_f = float(d["t_f"])
    fps = int(round(1.0 / dt))
    n = min(len(d["real"]), len(d["gen_id"]), len(d["gen_frozen"]))
    t = np.arange(n) * dt
    speed = d["speed"][:n]
    rows = [("recorded", d["real"][:n], F.OBSERVED),
            ("two-state NODE", d["gen_id"][:n], F.WITH_DELTA),
            (f"$\\hat{{\\delta}}$ fixed after $t_f$ = {t_f:.1f} s", d["gen_frozen"][:n], FIXED)]
    vmax = float(max(r[1].max() for r in rows))
    d_id, d_fr = d["delta_id"][:n, 0], d["delta_frozen"][:n, 0]
    dmax = float(max(d_id.max(), d_fr.max())) * 1.12
    constant = float(np.ptp(speed)) < 1e-9
    smax = max(float(np.ceil(speed.max())), 1.0)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=150)  # 1920 x 1080
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(5, 3, height_ratios=[0.80, 2.25, 0.36, 0.36, 0.36],
                          left=0.175, right=0.905, top=0.955, bottom=0.06,
                          hspace=0.25, wspace=0.045)

    ax_t = fig.add_subplot(gs[0, :]); ax_t.set_axis_off()
    ax_t.text(0.0, 0.92, title, ha="left", va="top", fontsize=22, color=F.OBSERVED)
    ax_t.text(0.0, 0.06,
              "generated from the first frame alone; the input never changes" if constant
              else "generated from the recorded travel speed and the first frame alone",
              ha="left", va="bottom", fontsize=13.5, color="0.45")
    clock = ax_t.text(1.0, 0.92, "", ha="right", va="top", fontsize=20, color="0.45")

    ims = []
    for j, (label, series, colour) in enumerate(rows):
        ax = fig.add_subplot(gs[1, j])
        ims.append(ax.imshow(np.clip(series[0], 0, vmax), cmap="gray", vmin=0, vmax=vmax,
                             animated=True, interpolation="none"))
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

    a_patch, a_value = bar(fig.add_subplot(gs[3, :]), "thermal state $\\hat{\\delta}$",
                           F.WITH_DELTA, dmax)
    f_patch, f_value = bar(fig.add_subplot(gs[4, :]), "$\\hat{\\delta}$ fixed after $t_f$",
                           FIXED, dmax)

    OUT.mkdir(parents=True, exist_ok=True)
    still = OUT / f"{stem}.png"
    k_still = int(0.55 * n)
    # constant quality rather than a fixed bitrate keeps each file at a few megabytes
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1, metadata=dict(title=stem),
                          extra_args=["-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"])
    t0 = time.time()
    with writer.saving(fig, str(OUT / f"{stem}.mp4"), dpi=150):
        for k in range(n):
            clock.set_text(f"{t[k]:5.1f} s")
            for im, (_, series, _c) in zip(ims, rows):
                im.set_data(np.clip(series[k], 0, vmax))
            if sp_patch is not None:
                sp_patch.set_width(speed[k])
                sp_value.set_text(f"{speed[k]:.1f} mm s" + "$^{-1}$")
            a_patch.set_width(d_id[k]); a_value.set_text(f"{d_id[k]:.2f}")
            f_patch.set_width(d_fr[k]); f_value.set_text(f"{d_fr[k]:.2f}")
            writer.grab_frame()
            if k == k_still:
                fig.savefig(still, dpi=150)
    plt.close(fig)
    print(f"{stem}: {n} frames, {n / fps:.0f} s at {fps} fps, t_f {t_f:.2f} s, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    F.style()
    todo = sys.argv[1:] or [r[0] for r in RECORDS]
    for args in RECORDS:
        if args[0] in todo:
            build(*args)
