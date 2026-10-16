#!/usr/bin/env python3
"""Manuscript Figure 10 (Section 6.2): one frame per process through the two-stage representation.

Columns, linked by arrows: recorded frame I_i -> pool state z_i (six numbers) -> structure image
D(z_i) -> residual codes c_i (24 x 24 codebook indices) -> reconstruction D(z_i) + r_hat_i.
Rows: GTAW, GMAW, WAAM, each with its own trained model. Uses the stored stage-1 and stage-2
checkpoints; nothing is trained here.

Writes figures/Figure_representation_examples.{png,pdf,svg}.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import ConnectionPatch

import generator_figures as GF
import visual_generator as V

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "Figure_representation_examples"
PROCESSES = ("GTAW", "GMAW", "WAAM")

plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 8, "svg.fonttype": "none",
                     "mathtext.fontset": "dejavusans"})

# Code indices carry no order. Each index is coloured by the position of its codebook vector
# along the first principal component of that process's codebook, so entries that encode
# similar patterns receive similar colours.
CODE_CMAP = plt.cm.viridis


def code_rank(vq) -> np.ndarray:
    e = vq.vq.embedding.detach().cpu().numpy()
    e = e - e.mean(0)
    pc1 = np.linalg.svd(e, full_matrices=False)[2][0]
    score = e @ pc1
    return (np.argsort(np.argsort(score)) / (len(score) - 1)).astype(float)


@torch.no_grad()
def one_frame(process: str) -> dict:
    GF.configure(process)
    sid = GF.CHOICE[process][0]
    frames, _ = V.load_cache()
    zs = torch.load(V.OUT / "states.pt", weights_only=False)
    dec, vq, *_ = GF.models()
    k = len(frames[sid]) // 2
    y = (frames[sid][k:k + 1].float() / 255.0).to(V.DEVICE)
    z = zs[sid][k:k + 1].to(V.DEVICE)
    s = dec(z)
    rhat, _, idx = vq(y - s)
    codes = idx[0].cpu().numpy()
    return dict(image=y[0, 0].cpu().numpy(), z=z[0].cpu().numpy(), structure=s[0, 0].cpu().numpy(),
                codes=codes, code_colour=code_rank(vq)[codes], recon=(s + rhat)[0, 0].cpu().numpy())


def gray(ax, im):
    ax.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])


def arrow(fig, a, b, label=None):
    con = ConnectionPatch(xyA=(1.06, 0.5), coordsA=a.transAxes, xyB=(-0.06, 0.5), coordsB=b.transAxes,
                          arrowstyle="-|>", mutation_scale=10, lw=0.9, color="#333333")
    fig.add_artist(con)
    if label:
        pa, pb = a.get_position(), b.get_position()
        fig.text((pa.x1 + pb.x0) / 2, (pa.y0 + pa.y1) / 2 + 0.012, label, ha="center", va="bottom",
                 fontsize=6.5, color="#333333", linespacing=1.1)


def build():
    data = {p: one_frame(p) for p in PROCESSES}
    zmax = max(np.abs(d["z"]).max() for d in data.values())

    fig = plt.figure(figsize=(7.2, 4.3))
    widths = [1.0, 0.18, 0.30, 1.0, 1.0, 1.0]          # second column is an empty spacer
    cols = (0, 2, 3, 4, 5)
    gs = fig.add_gridspec(3, 6, width_ratios=widths, wspace=0.75, hspace=0.12,
                          left=0.06, right=0.99, top=0.90, bottom=0.03)
    titles = ["Recorded frame $I_i$", "State $\\mathbf{z}_i$", "Structure $D(\\mathbf{z}_i)$",
              "Residual codes $c_i$", "Reconstruction $D(\\mathbf{z}_i)+\\hat r_i$"]
    labels = ["DINOv2\n+ head $h$", "decoder\n$D$", "encode\n$I_i-D(\\mathbf{z}_i)$", "decode\n+ add"]

    for r, p in enumerate(PROCESSES):
        d = data[p]
        axes = [fig.add_subplot(gs[r, c]) for c in cols]
        gray(axes[0], d["image"])
        axes[0].set_ylabel(p, fontsize=9, fontweight="bold", labelpad=6)

        ax = axes[1]
        ax.imshow(d["z"][:, None], cmap="coolwarm", vmin=-zmax, vmax=zmax, aspect="auto")
        for j, v in enumerate(d["z"]):
            ax.text(0, j, f"{v:.2f}", ha="center", va="center", fontsize=6)
        ax.set_xticks([]); ax.set_yticks(range(6)); ax.set_yticklabels([f"$z_{j+1}$" for j in range(6)], fontsize=6)
        ax.tick_params(length=0, pad=1)

        gray(axes[2], d["structure"])
        axes[3].imshow(d["code_colour"], cmap=CODE_CMAP, vmin=0, vmax=1, interpolation="nearest")
        axes[3].set_xticks([]); axes[3].set_yticks([])
        gray(axes[4], d["recon"])

        fig.canvas.draw()                                  # fixes the image axes to their aspect
        p0, pz = axes[0].get_position(), axes[1].get_position()
        axes[1].set_position([pz.x0, p0.y0, pz.width, p0.height])   # state column as tall as the frames
        if r == 0:
            for a, t in zip(axes, titles):
                pa = a.get_position()
                fig.text((pa.x0 + pa.x1) / 2, p0.y1 + 0.02, t, ha="center", va="bottom", fontsize=7.5)
        for c in range(4):
            arrow(fig, axes[c], axes[c + 1], labels[c] if r == 0 else None)
    return fig


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    fig = build()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{OUT}.{ext}", dpi=300)
    print("written:", OUT)
