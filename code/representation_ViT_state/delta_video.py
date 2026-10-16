#!/usr/bin/env python3
"""Video: the recorded weld, the weld the model predicts, and the same prediction with the model's
stored-heat state held fixed.

Top bar      the welding speed, the one input that was recorded.
Three views  the camera recording, the view the model generates from that input alone, and the view
             it generates when its stored-heat state is not allowed to keep rising.
Under each   the stored heat the model is carrying at that moment.
generated
view

The titles avoid process shorthand and record identifiers, so the video can be shown on its own.
"""
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter
from scipy.ndimage import uniform_filter1d

import delta_study as D
import generator_figures as G
import node_visual as N
import visual_generator as V

HERE = Path(__file__).resolve().parent
INK, BLUE, WARM = "#232629", "#1B6CA8", "#C0652B"
SOFT, RULE, TRACK = "#7A8189", "#D8DCE0", "#EDEFF1"
# Two output sizes. "paper" is the 720p version used alongside the manuscript; "web" is 1080p with
# the type enlarged, because a video embedded in a page is usually shown well below its own width.
# Sizes are points in a figure of the given height, so each is a fixed fraction of the frame.
LAYOUT = {
    "paper": dict(figsize=(12.8, 7.2), left=0.135, right=0.905,
                  rows=[0.62, 0.60, 0.42, 3.80, 0.60, 0.26],
                  title=19, sub=12.5, clock=17, head=14.5, note=10.5, tick=9.5,
                  bar_label=11.5, bar_value=11.5, panel_label=10.5, panel_value=11, foot=11,
                  short=False),
    "web": dict(figsize=(19.2, 10.8), left=0.160, right=0.900,
                rows=[1.05, 0.68, 0.80, 4.25, 0.72, 0.36],
                title=38, sub=24, clock=34, head=27, note=21, tick=16,
                bar_label=22, bar_value=24, panel_label=20, panel_value=22, foot=21,
                short=True),
}
# Same layout at 720p: every size is a fixed fraction of the frame, so the type is scaled with the
# frame and stays as legible, at roughly half the file size.
LAYOUT["web720"] = {**{k: (round(v * 2 / 3, 1) if isinstance(v, (int, float)) else v)
                       for k, v in LAYOUT["web"].items()},
                    "figsize": (12.8, 7.2), "rows": LAYOUT["web"]["rows"],
                    "left": LAYOUT["web"]["left"], "right": LAYOUT["web"]["right"],
                    "short": True}
PROCESS_NAME = {"GTAW": "Gas tungsten arc welding",
                "GMAW": "Gas metal arc welding",
                "WAAM": "Wire arc additive manufacturing"}
plt.rcParams.update({
    "font.family": ["Nimbus Sans", "DejaVu Sans", "sans-serif"], "font.size": 11,
    "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold",
    "savefig.bbox": None})


def sequences(process="GTAW", sid="Iden_09", smooth=20, tag="_s20_e1200", flat_at=15.0):
    N.configure(process)
    N.SMOOTH, N.TAG = smooth, tag
    rec = N.records()[sid]
    dt = N.DT
    ck = torch.load(N.OUT / f"fit_{sid}{tag}.pt", weights_only=False)
    model = N.PoolNODE(z_dim=6, u_dim=1, width=16)
    model.load_state_dict(ck["model"])
    model.eval()
    mean, std = ck["mean"], ck["std"]
    u = torch.tensor(rec["u"].astype(np.float32))
    zn = torch.tensor(((rec["z"] - mean) / std).astype(np.float32))
    with torch.no_grad():
        z_base, d_base = model.rollout(zn[0], u)
    k = int(flat_at / dt)
    d_flat = d_base.clone()
    d_flat[k:] = d_base[k]
    z_flat = D.rollout_forced(model, zn[0], u, d_flat)

    V.configure(process)
    frames, _ = V.load_cache()
    codes = torch.load(V.OUT / "codes.pt", weights_only=False)
    zs_all = torch.load(V.OUT / "states.pt", weights_only=False)
    dec, vq, *_ = G.models()
    bank_z = torch.cat([torch.tensor(uniform_filter1d(zs_all[o].numpy(), smooth, axis=0,
                                                      mode="nearest").astype(np.float32))
                        for o in V.IDS])
    bank_c = torch.cat([codes[o] for o in V.IDS])
    bounds = np.cumsum([0] + [len(codes[o]) for o in V.IDS]).tolist()

    def render(z_std):
        raw = torch.tensor((z_std.numpy() * std + mean).astype(np.float32))
        field = bank_c[D.retrieve(raw, bank_z, bank_c, bounds)].to(V.DEVICE)
        out = []
        with torch.no_grad():
            for i in range(0, len(raw), 64):
                out.append((dec(raw[i:i + 64].to(V.DEVICE))
                            + vq.decode_indices(field[i:i + 64])).cpu().numpy()[:, 0])
        return np.concatenate(out)

    real = frames[sid].float().numpy()[:, 0] / 255.0
    return dict(real=real, pred=render(z_base), frozen=render(z_flat),
                speed=rec["speed"], delta=d_base.numpy(), delta_flat=d_flat.numpy(),
                flat_at=flat_at, sid=sid, dt=dt, process=process)


def wide_bar(ax, label, colour, vmax, L, ticks=None):
    """A labelled bar across the full width, with its label in the left margin."""
    ax.set_xlim(0, vmax); ax.set_ylim(0, 1)
    ax.set_yticks([]); ax.tick_params(labelsize=L["tick"], colors=SOFT, length=3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.barh(0.5, vmax, height=0.5, color=TRACK, zorder=1)
    patch = ax.barh(0.5, 0.0, height=0.5, color=colour, zorder=2)[0]
    ax.text(-0.012 * vmax, 0.5, label, ha="right", va="center", fontsize=L["bar_label"], color=INK)
    txt = ax.text(vmax * 1.012, 0.5, "", ha="left", va="center", fontsize=L["bar_value"],
                  color=colour)
    ax.set_xticks(ticks if ticks is not None else [])
    return patch, txt


def panel_bar(ax, label, colour, vmax, L):
    """A bar under one generated view, with its caption and value on the line above it."""
    ax.set_xlim(0, vmax); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.barh(0.28, vmax, height=0.34, color=TRACK, zorder=1)
    patch = ax.barh(0.28, 0.0, height=0.34, color=colour, zorder=2)[0]
    ax.text(0.0, 0.80, label, ha="left", va="center", fontsize=L["panel_label"], color=SOFT)
    txt = ax.text(vmax, 0.80, "", ha="right", va="center", fontsize=L["panel_value"],
                  color=colour)
    return patch, txt


def build(process="GTAW", fps=None, out=None, preview=None, mode="paper", outdir=None,
          speed=1.0, **kw):
    L = LAYOUT[mode]
    d = sequences(process, **kw)
    # Playing faster keeps every frame, so nothing is lost; the clock still reads record time.
    fps = fps or int(round(speed / d["dt"]))
    out = out or f"molten_pool_heat_memory_{process}_v2.mp4"
    root = Path(outdir) if outdir else HERE
    root.mkdir(parents=True, exist_ok=True)
    n = min(len(d["real"]), len(d["pred"]), len(d["speed"]))
    t = np.arange(n) * d["dt"]
    vmax_img = float(d["real"].max())
    smax = max(float(np.ceil(d["speed"].max())), 1.0)
    dmax = float(max(d["delta"].max(), d["delta_flat"].max()) * 1.15)
    hold = d["flat_at"]

    fig = plt.figure(figsize=L["figsize"], dpi=100)
    fig.patch.set_facecolor("white")
    # The headings sit in a row of their own, so they cannot collide with the input scale above.
    gs = fig.add_gridspec(6, 3, height_ratios=L["rows"],
                          left=L["left"], right=L["right"], top=0.955, bottom=0.050,
                          hspace=0.34, wspace=0.055)

    constant = float(np.ptp(d["speed"])) < 1e-6
    ax_t = fig.add_subplot(gs[0, :]); ax_t.set_axis_off()
    ax_t.text(0.0, 1.0,
              "Molten pool predicted by the model" if constant
              else "Molten pool predicted from the welding input",
              ha="left", va="top", fontsize=L["title"], color=INK)
    ax_t.text(0.0, 0.0, PROCESS_NAME.get(d["process"], d["process"]),
              ha="left", va="bottom", fontsize=L["sub"], color=SOFT)
    clock = ax_t.text(1.0, 1.0, "", ha="right", va="top", fontsize=L["clock"], color=SOFT)
    if speed != 1.0:
        ax_t.text(1.0, 0.0, f"playback {speed:g}\u00d7 real time", ha="right", va="bottom",
                  fontsize=L["note"], color=SOFT)

    ax_s = fig.add_subplot(gs[1, :])
    if constant:
        ax_s.set_axis_off()
        ax_s.text(0.0, 0.5,
                  "The settings never change: nothing in the input says the part is heating up."
                  if L["short"] else
                  "The settings are held the same throughout: nothing in the input tells the "
                  "model that the part is heating up.",
                  ha="left", va="center", fontsize=L["sub"], color=SOFT, transform=ax_s.transAxes)
        sp_patch = sp_txt = None
    else:
        # only the ends of the scale; the live value is printed beside the bar
        sp_patch, sp_txt = wide_bar(ax_s, "welding speed", INK, smax, L, ticks=[0.0, smax])

    short = L["short"]
    views = [("real", "Recorded", "high-speed camera", INK),
             ("pred", "Model prediction" if short else "Predicted by the model",
              "from the recorded settings" if constant
              else ("from the welding input" if short else "generated from the welding input"),
              BLUE),
             ("frozen", "Model without heat memory" if short else "Predicted without heat memory",
              f"heat memory frozen at {hold:.0f} s" if short
              else f"stored heat held fixed after {hold:.0f} s", WARM)]
    ims = []
    for j, (key, name, note, colour) in enumerate(views):
        ax_h = fig.add_subplot(gs[2, j]); ax_h.set_axis_off()
        ax_h.text(0.0, 1.0, name, ha="left", va="top", fontsize=L["head"], color=colour)
        ax_h.text(0.0, 0.0, note, ha="left", va="bottom", fontsize=L["note"], color=SOFT)
        ax = fig.add_subplot(gs[3, j])
        ims.append(ax.imshow(d[key][0], cmap="gray", vmin=0, vmax=vmax_img, animated=True))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(RULE)

    labels = ([(BLUE, "heat the model carries"), (WARM, f"heat, frozen at {hold:.0f} s")] if short
              else [(BLUE, "stored heat carried by the model"),
                    (WARM, f"stored heat, held fixed after {hold:.0f} s")])
    bars = []
    for j, (colour, label) in zip((1, 2), labels):
        ax = fig.add_subplot(gs[4, j])
        bars.append(panel_bar(ax, label, colour, dmax, L))

    ax_n = fig.add_subplot(gs[5, :]); ax_n.set_axis_off()
    ax_n.text(0.0, 0.5,
              "Same recorded input for both predictions — only the heat memory differs." if short
              else "Both predictions come from the same recorded input; they differ only in "
                   "whether the model keeps track of the heat accumulating in the part.",
              ha="left", va="center", fontsize=L["foot"], color=SOFT)

    if preview is not None:
        # One frame to a PNG, for checking the layout without rendering the whole video.
        i = min(int(preview / d["dt"]), n - 1)
        clock.set_text(f"{t[i]:5.1f} s")
        if sp_patch is not None:
            sp_patch.set_width(d["speed"][i]); sp_txt.set_text(f"{d['speed'][i]:.1f} mm/s")
        for im, (key, *_rest) in zip(ims, views):
            im.set_data(d[key][i])
        for (patch, txt), series in zip(bars, (d["delta"], d["delta_flat"])):
            patch.set_width(series[i]); txt.set_text(f"{series[i]:.2f}")
        png = root / f"preview_{d['process']}_{mode}.png"
        fig.savefig(png, dpi=100)
        plt.close(fig)
        print("saved", png)
        return

    writer = FFMpegWriter(fps=fps, bitrate=3600, metadata=dict(title="molten pool heat memory"))
    path = root / out
    with writer.saving(fig, str(path), dpi=100):
        for i in range(n):
            clock.set_text(f"{t[i]:5.1f} s")
            if sp_patch is not None:
                sp_patch.set_width(d["speed"][i]); sp_txt.set_text(f"{d['speed'][i]:.1f} mm/s")
            for im, (key, *_rest) in zip(ims, views):
                im.set_data(d[key][i])
            for (patch, txt), series in zip(bars, (d["delta"], d["delta_flat"])):
                patch.set_width(series[i]); txt.set_text(f"{series[i]:.2f}")
            writer.grab_frame()
    plt.close(fig)
    print("saved", path, f"({n} frames, {n / fps:.0f} s at {fps} fps)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", default="GTAW", choices=["GTAW", "GMAW", "WAAM"])
    ap.add_argument("--preview", type=float, default=None,
                    help="save one frame at this time (s) instead of rendering the video")
    ap.add_argument("--mode", default="paper", choices=list(LAYOUT),
                    help="'web' enlarges the type and renders at 1080p")
    ap.add_argument("--out", default=None, help="output file name")
    ap.add_argument("--outdir", default=None, help="directory to write into")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback rate relative to real time; no frames are dropped")
    a = ap.parse_args()
    cfg = {"GTAW": dict(sid="Iden_09", smooth=20, tag="_s20_e1200", flat_at=15.0),
           "GMAW": dict(sid="exp_05", smooth=24, tag="_s24_e1200", flat_at=10.0),
           "WAAM": dict(sid="video_1", smooth=25, tag="_s25_e1200", flat_at=8.0)}[a.process]
    build(a.process, preview=a.preview, mode=a.mode, out=a.out, outdir=a.outdir,
          speed=a.speed, **cfg)
