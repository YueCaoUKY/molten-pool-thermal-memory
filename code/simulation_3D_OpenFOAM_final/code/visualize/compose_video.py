#!/usr/bin/env python3
"""Compose the 3D views rendered by render_pool3d.py and the case signals into
video frames, and encode them with ffmpeg (H.264 MP4 and VP9 WebM).

Frame layout (1920 x 1080): header with the time and the arc state; quarter
cutaway (left); liquid region seen from below and the whole plate (right);
colour bar and legend, process inputs, pool size and temperatures with a time
cursor (bottom). Playback is 2x real time (one 0.1 s snapshot per frame at
20 fps), slowed to 0.5x around the arc switch-off so that the solidification
is visible.

Outputs in results/<case_id>/figures/: <case_id>_pool3d.mp4, .webm and
pool3d_keyframes/<case_id>_t<tag>.png at KEY_TIMES.

    python code/visualize/compose_video.py <case_id> <frame_dir>
"""
from __future__ import annotations

import glob
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mp3d import RESULTS_DIR  # noqa: E402
from mp3d.case_writer import build_material, load_config  # noqa: E402

W, H, FPS = 1920, 1080, 20
T_RANGE = (293.15, 2200.0)
INFERNO_START = 0.08                     # as in render_pool3d.py
CMAP = ListedColormap(plt.cm.inferno(np.linspace(INFERNO_START, 1.0, 256)))
SLOW = (19.75, 20.65, 4)                 # snapshots in (t0, t1] are shown 4 times (0.5x real time)
HOLD_END_S = 2.0
KEY_TIMES = (2.0, 8.0, 14.0, 20.0, 20.2, 30.0)
BOXES = {"cut": (12, 90, 1240, 640), "pool": (1258, 90, 660, 370), "plate": (1258, 460, 660, 270)}
CUT_LABELS = (("centreline section", (250, 560)), ("cross-section 1 mm behind the arc", (905, 585)))
DARK = "#1b1630"


def find_ffmpeg():
    cands = [shutil.which("ffmpeg")]
    try:
        import imageio_ffmpeg
        cands.append(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    cands += sorted(glob.glob(str(Path.home() / "miniconda3/envs/*/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*")))
    for c in cands:
        if c and Path(c).is_file():
            return c
    raise SystemExit("no ffmpeg binary found")


def px_axes(fig, x, y, w, h, **kw):
    return fig.add_axes([x / W, 1.0 - (y + h) / H, w / W, h / H], **kw)


def label(ax, x, y, text, **kw):
    style = dict(fontsize=11, color="white", ha="left", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc=DARK, ec="none", alpha=0.72))
    style.update(kw)
    return ax.text(x, y, text, **style)


def main(case, frame_dir):
    frame_dir = Path(frame_dir)
    cfg = load_config(RESULTS_DIR / case / "config_resolved.yaml")
    mat = build_material(cfg)
    s = pd.read_csv(RESULTS_DIR / case / "signals_0p1s.csv")
    s.index = np.round(s["t_s"], 3)
    tags = sorted(int(Path(p).stem.split("_")[1]) for p in glob.glob(str(frame_dir / "cut_*.png")))
    if not tags:
        raise SystemExit(f"no frames in {frame_dir}")
    out = RESULTS_DIR / case / "figures"
    (out / "pool3d_keyframes").mkdir(parents=True, exist_ok=True)
    src = cfg["source"]
    geo = cfg["geometry"]

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    ims = {}
    for k, (x, y, w, h) in BOXES.items():
        ax = px_axes(fig, x, y, w, h)
        ax.set_axis_off()
        ims[k] = (ax, ax.imshow(np.zeros((h, w, 3), np.uint8), interpolation="none"))
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
    ax_cut, ax_pool, ax_plate = (ims[k][0] for k in ("cut", "pool", "plate"))
    label(ax_cut, 14, 14, "Quarter cutaway following the arc (true scale)", fontsize=13)
    for text, (x, y) in CUT_LABELS:
        label(ax_cut, x, y, text)
    label(ax_pool, 10, 10, f"Liquid region (T ≥ {mat.liquidus:g} K), seen from below", fontsize=12)
    label(ax_pool, 10, 44, "grids: 1 mm on the top face, 2 mm on the bottom face (6 mm lower)", fontsize=9.5)
    pool_txt = label(ax_pool, 10, 360, "", fontsize=11, va="bottom")
    label(ax_plate, 10, 10, f"Whole plate, {geo['Lx'] * 1e3:g} × {geo['Ly'] * 1e3:g} × {geo['H'] * 1e3:g} mm",
          fontsize=12)

    fig.text(12 / W, 1 - 26 / H, f"3D heat-conduction simulation of the reference weld (case {case})",
             fontsize=20, weight="bold", va="center")
    fig.text(12 / W, 1 - 55 / H,
             f"OpenFOAM v14 + AdditiveFOAM 2.0.0, finite volume, 732 900 cells  ·  low-carbon steel plate "
             f"{geo['Lx'] * 1e3:g} × {geo['Ly'] * 1e3:g} × {geo['H'] * 1e3:g} mm  ·  Gaussian arc, "
             f"efficiency {src['eta']:.2f}, radius σ = {src['sigma'] * 1e3:g} mm, {cfg['schedule']['voltage']:g} V",
             fontsize=11.5, va="center", color="0.2")
    fig.text(12 / W, 1 - 77 / H,
             f"Heat conduction only: no fluid flow in the pool (liquid conductivity × "
             f"{cfg['material']['liquid_k_multiplier']:g} stands in for stirring). "
             f"Molten pool = region above the liquidus, {mat.liquidus:g} K.",
             fontsize=11, va="center", color="0.35", style="italic")
    t_txt = fig.text(1 - 14 / W, 1 - 26 / H, "", fontsize=24, ha="right", va="center", family="monospace",
                     weight="bold")
    state_txt = fig.text(1 - 14 / W, 1 - 56 / H, "", fontsize=13, ha="right", va="center")
    speed_txt = fig.text(1 - 14 / W, 1 - 78 / H, "", fontsize=10, ha="right", va="center", color="0.4")

    # colour bar and legend
    cax = px_axes(fig, 40, 790, 330, 22)
    cb = fig.colorbar(ScalarMappable(Normalize(*T_RANGE), CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([300, 800, 1300, 1800, 2200])
    cb.set_label("temperature (K)", fontsize=11)
    cax.axvline(mat.solidus, color=(0.0, 0.85, 1.0), lw=1.5)
    cax.axvline(mat.liquidus, color="white", lw=1.5)
    lax = px_axes(fig, 40, 862, 330, 176)
    lax.set_facecolor(DARK)
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.set_xticks([])
    lax.set_yticks([])
    entries = ((dict(color="white", lw=3), f"liquidus {mat.liquidus:g} K (pool boundary)"),
               (dict(color=(0.0, 0.85, 1.0), lw=2), f"solidus {mat.solidus:g} K"),
               (dict(color="0.8", lw=1, alpha=0.8), "isotherms 600, 1000, 1400 K"))
    for n, (kw, text) in enumerate(entries):
        yy = 0.84 - 0.21 * n
        lax.plot([0.05, 0.2], [yy, yy], **kw)
        lax.text(0.25, yy, text, color="white", fontsize=10.5, va="center")
    lax.fill_between([0.08, 0.17], 0.12, 0.30, color=(0.55, 0.72, 1.0), alpha=0.5, lw=0)
    lax.text(0.25, 0.21, "arc and electrode (schematic)", color="white", fontsize=10.5, va="center")

    # time series
    t = s["t_s"].to_numpy()
    plots = []
    ax = px_axes(fig, 470, 768, 370, 245)
    ax.step(t, s["I_A"], where="post", color="C0", lw=1.5)
    ax.set_ylabel("current (A)", color="C0")
    ax.set_ylim(0, 260)
    ax.set_title("process inputs", fontsize=11)
    ax.set_xlabel("time (s)")
    ax2 = ax.twinx()
    ax2.step(t, s["u_mm_s"], where="post", color="C1", lw=1.5)
    ax2.set_ylabel("travel speed (mm/s)", color="C1")
    ax2.set_ylim(0, 5)
    plots.append((ax, [(ax, "I_A", "C0"), (ax2, "u_mm_s", "C1")]))
    ax = px_axes(fig, 955, 768, 370, 245)
    for col, lab, c in (("top_length_liq_mm", "length", "C2"), ("top_width_liq_mm", "width", "C4"),
                        ("depth_liq_mm", "depth", "C3")):
        ax.plot(t, s[col], color=c, lw=1.5, label=lab)
    ax.set(ylabel="mm", xlabel="time (s)")
    ax.set_title("pool size (top-surface length and width, depth)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    plots.append((ax, [(ax, "top_length_liq_mm", "C2"), (ax, "top_width_liq_mm", "C4"), (ax, "depth_liq_mm", "C3")]))
    ax = px_axes(fig, 1440, 768, 370, 245)
    ax.plot(t, s["T_top_max_K"], color="C3", lw=1.5)
    ax.axhline(mat.liquidus, color="0.6", ls=":", lw=1)
    ax.set_ylabel("peak surface T (K)", color="C3")
    ax.set_xlabel("time (s)")
    ax.set_title("temperatures", fontsize=11)
    ax2 = ax.twinx()
    ax2.plot(t, s["delta_H_K"], color="C5", lw=1.5)
    ax2.set_ylabel("heat stored in plate,\nas mean T rise (K)", color="C5")
    plots.append((ax, [(ax, "T_top_max_K", "C3"), (ax2, "delta_H_K", "C5")]))
    cursors, dots = [], []
    for ax, series in plots:
        ax.set_xlim(0, t[-1])
        ax.grid(alpha=0.3)
        cursors.append(ax.axvline(0, color="k", lw=1))
        for a, col, c in series:
            dots.append((a.plot([], [], "o", color=c, ms=6, mec="k", mew=0.6)[0], col))

    seq = []
    for tag in tags:
        rep = SLOW[2] if SLOW[0] < tag / 100 <= SLOW[1] + 1e-9 else 1
        seq.append((tag, rep))
    seq[-1] = (seq[-1][0], 1 + int(HOLD_END_S * FPS))
    mp4 = out / f"{case}_pool3d.mp4"
    cmd = [find_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(mp4)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n_frames = 0
    for tag, rep in seq:
        tt = tag / 100
        row = s.loc[round(tt, 3)]
        for k in ims:
            ims[k][1].set_data(np.asarray(Image.open(frame_dir / f"{k}_{tag:05d}.png").convert("RGB")))
        t_txt.set_text(f"t = {tt:.1f} s")
        if row["I_A"] > 0:
            state_txt.set_text(f"arc on  ·  {row['I_A']:.0f} A  ·  {row['u_mm_s']:.1f} mm/s  ·  "
                               f"{row['P_abs_nominal_W']:.0f} W absorbed")
            state_txt.set_color("C0")
        else:
            state_txt.set_text("arc off  ·  plate cooling")
            state_txt.set_color("C3")
        speed_txt.set_text("playback 0.5× real time (arc switch-off)" if rep == SLOW[2]
                           else "playback 2× real time")
        if row["top_area_liq_mm2"] > 0:
            pool_txt.set_text(f"length {row['top_length_liq_mm']:.1f} mm  ·  width {row['top_width_liq_mm']:.1f} mm"
                              f"  ·  depth {row['depth_liq_mm']:.2f} mm  ·  volume {row['vol_liquid_mm3']:.0f} mm³")
        else:
            pool_txt.set_text("no liquid")
        for c in cursors:
            c.set_xdata([tt, tt])
        for d, col in dots:
            d.set_data([tt], [row[col]])
        fig.canvas.draw()
        buf = bytes(fig.canvas.buffer_rgba())
        assert len(buf) == W * H * 4, "figure canvas is not W x H pixels"
        for _ in range(rep):
            enc.stdin.write(buf)
        n_frames += rep
        if any(abs(tt - k) < 1e-6 for k in KEY_TIMES):
            Image.frombuffer("RGBA", (W, H), buf, "raw", "RGBA", 0, 1).convert("RGB").save(out / "pool3d_keyframes" / f"{case}_t{tag:05d}.png")
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    webm = mp4.with_suffix(".webm")
    subprocess.run([cmd[0], "-y", "-loglevel", "error", "-i", str(mp4), "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "33",
                    "-row-mt", "1", "-deadline", "good", "-cpu-used", "4", str(webm)], check=True)
    print(f"{n_frames} frames ({n_frames / FPS:.1f} s at {FPS} fps) -> {mp4} ({mp4.stat().st_size / 1e6:.1f} MB), "
          f"{webm.name} ({webm.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
