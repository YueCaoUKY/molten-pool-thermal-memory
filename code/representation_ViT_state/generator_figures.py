#!/usr/bin/env python3
"""Figures for the GTAW visual state extractor and image generator (no dynamic model)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import visual_generator as V

FIGDIR = Path(__file__).resolve().parent / "figures"
DT = V.SPEC.dt
PROC = "GTAW"

# Record used for the trajectory and reconstruction figures, and the pair that is averaged.
CHOICE = {"GTAW": ("Iden_06", "Iden_06", "Iden_13"),
          "GMAW": ("exp_01", "exp_01", "exp_10"),
          "WAAM": ("video_1", "video_1", "video_5")}


def configure(process: str) -> None:
    global FIGDIR, DT, PROC
    V.configure(process)
    FIGDIR = Path(__file__).resolve().parent / "figures"
    DT = V.SPEC.dt
    PROC = process
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 200,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.4})


def models():
    a = torch.load(V.OUT / "stage_a.pt", weights_only=False)
    dec = V.StructureDecoder(mean_frame=a["mean_frame"].to(V.DEVICE),
                             size=a["mean_frame"].shape[-1]).to(V.DEVICE)
    dec.load_state_dict(a["dec"]); dec.eval()
    b = torch.load(V.OUT / "stage_b.pt", weights_only=False)
    vq = V.ResidualVQ().to(V.DEVICE); vq.load_state_dict(b["net"]); vq.eval()
    c = torch.load(V.OUT / "stage_c.pt", weights_only=False)
    prior = V.CodePrior(c["grid"]).to(V.DEVICE); prior.load_state_dict(c["net"]); prior.eval()
    return dec, vq, prior, c["zmean"], c["zstd"]


def show(ax, im, title=None):
    ax.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if title:
        ax.set_title(title, fontsize=7, pad=2)


# --------------------------------------------------------------------- figure 1
def fig_states(sid=None):
    sid = sid or CHOICE[V.PROCESS][0]
    zs = torch.load(V.OUT / "states.pt", weights_only=False)
    z = zs[sid].numpy()
    t = np.arange(len(z)) * DT
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 4.6), sharex=True)
    for k, ax in enumerate(axes.T.flat):
        ax.plot(t, z[:, k], lw=0.8, color="#1f4e79")
        ax.set_ylabel(f"$z_{{{k+1}}}$")
        if k in (2, 5):
            ax.set_xlabel("time (s)")
    fig.suptitle(f"State trajectory, {V.PROCESS} {sid} (six coordinates, extracted frame by frame)", y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGDIR / f"repr_{PROC}_states.png"); plt.close(fig)


# --------------------------------------------------------------------- figure 2
def fig_reconstruction(sid=None, n=6):
    sid = sid or CHOICE[V.PROCESS][0]
    frames, _ = V.load_cache()
    zs = torch.load(V.OUT / "states.pt", weights_only=False)
    dec, vq, *_ = models()
    img = frames[sid].float() / 255.0
    idx = np.linspace(0, len(img) - 1, n).round().astype(int)
    with torch.no_grad():
        y = img[idx].to(V.DEVICE)
        s = dec(zs[sid][idx].to(V.DEVICE))
        r = y - s
        rhat, _, _ = vq(r)
        full = s + rhat
    y, s, full, r, rhat = (x.cpu().numpy()[:, 0] for x in (y, s, full, r, rhat))
    rows = [("original", y, "gray"), ("structure $D(z)$", s, "gray"),
            ("residual $r$", r, "diff"), ("structure + detail", full, "gray")]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.15 * n, 1.22 * len(rows)))
    for i, (name, data, kind) in enumerate(rows):
        for j in range(n):
            ax = axes[i, j]
            if kind == "gray":
                show(ax, data[j], f"t = {idx[j]*DT:.1f} s" if i == 0 else None)
            else:
                ax.imshow(data[j], cmap="coolwarm", vmin=-0.25, vmax=0.25)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if j == 0:
                ax.set_ylabel(name, fontsize=7)
    err = float(np.abs(y - full).mean())
    fig.suptitle(f"{sid}: reconstruction with images available  (mean absolute error {err:.4f})", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"repr_{PROC}_reconstruction.png"); plt.close(fig)
    return err


# --------------------------------------------------------------------- figure 3
@torch.no_grad()
def generate(z_seq, c0, dec, vq, prior, zmean, zstd, temperature=1.0, seed=0):
    """Roll the code field forward under the given state trajectory and decode."""
    g = torch.Generator(device=V.DEVICE).manual_seed(seed)
    c = c0.to(V.DEVICE)[None]
    out = []
    for t in range(len(z_seq)):
        z = ((z_seq[t] - zmean) / zstd).to(V.DEVICE)[None]
        if t > 0:
            logits = prior(c, z) / temperature
            b, h, w, k = logits.shape
            c = torch.multinomial(logits.reshape(-1, k).softmax(-1), 1, generator=g).view(b, h, w)
        detail = vq.decode_indices(c)
        out.append((dec(z_seq[t][None].to(V.DEVICE)) + detail).cpu().numpy()[0, 0])
    return np.stack(out)


def fig_average(a=None, b=None, n=6, temperature=0.5):
    a = a or CHOICE[V.PROCESS][1]
    b = b or CHOICE[V.PROCESS][2]
    frames, _ = V.load_cache()
    zs = torch.load(V.OUT / "states.pt", weights_only=False)
    codes = torch.load(V.OUT / "codes.pt", weights_only=False)
    dec, vq, prior, zmean, zstd = models()
    m = min(len(zs[a]), len(zs[b]))
    za, zb = zs[a][:m], zs[b][:m]
    zbar = 0.5 * (za + zb)
    seq = generate(zbar, codes[a][0], dec, vq, prior, zmean, zstd, temperature=temperature)
    np.save(V.OUT / "generated_average.npy", seq)
    idx = np.linspace(0, m - 1, n).round().astype(int)
    t = np.arange(m) * DT

    fig = plt.figure(figsize=(1.3 * n, 8.6))
    gs = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1.5, 1.5, 1.5], hspace=0.55, wspace=0.18)
    for k in range(6):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ax.plot(t, za[:, k], lw=0.6, label=a, color="#1f4e79")
        ax.plot(t, zb[:, k], lw=0.6, label=b, color="#a33")
        ax.plot(t, zbar[:, k], lw=1.1, label="average", color="k")
        ax.set_ylabel(f"$z_{{{k+1}}}$", labelpad=1)
        ax.tick_params(labelsize=6)
        for x in idx * DT:
            ax.axvline(x, color="0.75", lw=0.4, zorder=0)
        if k == 0:
            ax.legend(ncol=3, fontsize=6, loc="upper center", framealpha=0.9,
                      bbox_to_anchor=(0.5, 1.45))
        if k >= 4:
            ax.set_xlabel("time (s)", fontsize=7)
    with torch.no_grad():
        struct = dec(zbar[idx].to(V.DEVICE)).cpu().numpy()[:, 0]
    ia = frames[a].float().numpy()[idx, 0] / 255.0
    ib = frames[b].float().numpy()[idx, 0] / 255.0
    inner = gs[3:, :].subgridspec(3, n, hspace=0.06, wspace=0.04)
    rows = [(f"real {a}", ia), (f"real {b}", ib), ("generated from\nthe average state", seq[idx])]
    for i, (name, data) in enumerate(rows):
        for j in range(n):
            ax = fig.add_subplot(inner[i, j])
            show(ax, data[j], f"t = {idx[j]*DT:.1f} s" if i == 0 else None)
            if j == 0:
                ax.set_ylabel(name, fontsize=7)
    fig.suptitle("Image sequence generated from the average of two state trajectories\n"
                 f"(detail sampled from the code prior, temperature {temperature})", y=0.955)
    fig.savefig(FIGDIR / f"repr_{PROC}_average_state_generation.png"); plt.close(fig)

    strip = FIGDIR / f"repr_{PROC}_generated_sequence"
    strip.mkdir(exist_ok=True)
    for f in strip.glob("*.png"):
        f.unlink()
    for k in np.linspace(0, m - 1, 24).round().astype(int):
        plt.imsave(strip / f"t_{k*DT:05.1f}s.png", np.clip(seq[k], 0, 1), cmap="gray", vmin=0, vmax=1)
    return m


def fig_temperature(sid=None, m=200, n=6, temps=(1.0, 0.7, 0.4)):
    sid = sid or CHOICE[V.PROCESS][0]
    frames, _ = V.load_cache()
    zs = torch.load(V.OUT / "states.pt", weights_only=False)
    codes = torch.load(V.OUT / "codes.pt", weights_only=False)
    dec, vq, prior, zmean, zstd = models()
    idx = np.linspace(0, m - 1, n).round().astype(int)
    real = frames[sid].float().numpy()[:, 0] / 255.0
    fig, axes = plt.subplots(len(temps) + 1, n, figsize=(1.3 * n, 1.35 * (len(temps) + 1)))
    for j, k in enumerate(idx):
        show(axes[0, j], real[k], f"t = {k*DT:.1f} s")
    axes[0, 0].set_ylabel("real", fontsize=7)
    for i, T in enumerate(temps):
        seq = generate(zs[sid][:m], codes[sid][0], dec, vq, prior, zmean, zstd, temperature=T)
        for j, k in enumerate(idx):
            show(axes[i + 1, j], seq[k])
        axes[i + 1, 0].set_ylabel(f"temperature {T}", fontsize=7)
    fig.suptitle(f"{sid}: detail sampled at three temperatures, true state trajectory", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"repr_{PROC}_sampling_temperature.png"); plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", default="GTAW", choices=list(V.SIZES))
    configure(ap.parse_args().process)
    fig_states()
    err = fig_reconstruction()
    m = fig_average()
    fig_temperature()
    print(f"reconstruction MAE {err:.4f}; generated {m} frames")
