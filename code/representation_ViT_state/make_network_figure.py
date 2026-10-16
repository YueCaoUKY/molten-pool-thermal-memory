#!/usr/bin/env python3
"""Network structure of the visual representation stage.

Tensors are drawn as isometric plates carrying their real contents: the frame, the pooled
patch-token map, the state, the decoded structure, the residual, and the code field all come
from a trained model (results_generator/figure_tensors.npz, written by this file's companion
extraction step). Colour marks what is trained in which stage.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
OUT = Path(__file__).resolve().parent / "figures"
DATA = HERE / "results_generator" / "figure_tensors.npz"

BLUE, GREY, GREEN, TEAL, RED = "#1F5C9E", "#8E969E", "#3F8F58", "#2E7F8A", "#B64342"
INK, SOFT, LINE = "#232629", "#7A8189", "#B9BFC6"
DX, DY = 0.32, 0.23                      # isometric depth direction, deliberately shallow

plt.rcParams.update({
    "font.family": ["Nimbus Sans", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic",
    "mathtext.bf": "Nimbus Sans:bold",
    "svg.fonttype": "none",
})
FIRE = LinearSegmentedColormap.from_list("fire", ["#07070b", "#3b2350", "#a33b4e", "#e8913f",
                                                 "#ffe9b0"])


def mix(color, white=0.0, black=0.0):
    c = np.array(to_rgb(color))
    c = c + (1 - c) * white
    c = c * (1 - black)
    return tuple(np.clip(c, 0, 1))


# --------------------------------------------------------------------------- primitives
def plate(ax, x, y, w, h, d, base=BLUE, img=None, cmap="gray", vmin=None, vmax=None,
          lw=0.9, z=3, face_white=0.82):
    """One tensor: axis-aligned front face, lit top face, shaded right face."""
    dx, dy = DX * d, DY * d
    front = mix(base, white=face_white)
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, facecolor=mix(base, white=face_white + 0.11),
                         edgecolor=mix(base, black=0.15), lw=lw, zorder=z, joinstyle="round"))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, facecolor=mix(base, white=max(face_white - 0.28, 0)),
                         edgecolor=mix(base, black=0.15), lw=lw, zorder=z, joinstyle="round"))
    ax.add_patch(Rectangle((x, y), w, h, facecolor=front, edgecolor="none", zorder=z))
    if img is not None:
        ax.imshow(img, extent=(x, x + w, y, y + h), cmap=cmap, vmin=vmin, vmax=vmax,
                  zorder=z + 0.5, aspect="auto", interpolation="nearest" if img.shape[0] < 40
                  else "bilinear")
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=mix(base, black=0.15),
                           lw=lw, zorder=z + 1))
    return x + w + dx, y + h / 2 + dy / 2


def plates(ax, x, y, w, h, d, n=3, gap=1.0, base=BLUE, **kw):
    for i in range(n):
        plate(ax, x + i * (w + gap), y, w, h, d, base=base, z=3 + 0.01 * i, **kw)
    return x + (n - 1) * (w + gap) + w + DX * d, y + h / 2 + DY * d / 2


def caption(ax, x0, x1, y, name=None, dims=None, color=INK):
    cx = (x0 + x1) / 2
    if name:
        ax.text(cx, y, name, ha="center", va="bottom", fontsize=10, color=color)
    if dims:
        ax.text(cx, y - 1.0, dims, ha="center", va="top", fontsize=8.4, color=SOFT)


def flow(ax, x0, x1, y, label=None, color=LINE, lw=1.5):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=11,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=6))
    if label:
        ax.text((x0 + x1) / 2, y + 1.3, label, ha="center", va="bottom", fontsize=8.6,
                color=SOFT, style="italic")


def sign(ax, x, y, s, color=INK):
    ax.text(x, y, s, ha="center", va="center", fontsize=15, color=color, zorder=7)


def rule(ax, x, y, w, label, color=INK):
    ax.text(x, y, label, ha="left", va="center", fontsize=11.5, fontweight="bold", color=color)
    ax.plot([x + 0.35 * w, x + w], [y, y], color="#E3E6E9", lw=1.0, zorder=0)


# --------------------------------------------------------------------------- figure
def build():
    t = np.load(DATA)
    img, struct, res = t["image"], t["structure"], t["residual"]
    recon, codes, tok = t["recon"], t["codes"], t["tokens8"]
    tok16 = t["tokens16"]
    zv = t["z"]
    vmax = float(max(img.max(), struct.max(), recon.max()))
    rlim = float(np.abs(res).max())

    fig, ax = plt.subplots(figsize=(13.4, 9.3))
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 118)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # ============================================================ (a) state extraction
    rule(ax, 3, 115, 144, "(a)   Pool state from frozen transformer features")
    ax.text(147, 115, "stage 1", ha="right", va="center", fontsize=10.5, color=BLUE)
    y = 97
    r = plate(ax, 4, y, 13, 13, 2.5, base=GREY, img=img, cmap=FIRE, vmin=0, vmax=vmax)
    caption(ax, 4, r[0], y - 2.2, "frame $I_i$", "192 $\\times$ 192")
    flow(ax, r[0] + 2, r[0] + 9, r[1])
    r2 = plate(ax, r[0] + 11, y - 2.5, 8, 18, 4.5, base=GREY, face_white=0.55)
    ax.text((r[0] + 11 + r2[0]) / 2, y + 6.5, "DINOv2\nViT-B/14", ha="center", va="center",
            fontsize=9.6, color="white", linespacing=1.4, zorder=6)
    flow(ax, r2[0] + 2, r2[0] + 9, r[1])
    r3 = plate(ax, r2[0] + 11, y + 1, 9, 9, 13, base=GREY, img=tok16, cmap="bone")
    caption(ax, r2[0] + 11, r3[0], y - 2.2, "patch tokens", "768 $\\times$ 16 $\\times$ 16")
    flow(ax, r3[0] + 2, r3[0] + 9, r3[1], "pool")
    r4 = plate(ax, r3[0] + 11, y + 2.5, 6, 6, 13, base=GREY, img=tok, cmap="bone")
    caption(ax, r3[0] + 11, r4[0], y - 2.2, None, "768 $\\times$ 8 $\\times$ 8")
    flow(ax, r4[0] + 2, r4[0] + 9, r4[1])
    r5 = plates(ax, r4[0] + 11, y + 2, 4.5, 7, 6, n=2, gap=1.6, base=BLUE)
    caption(ax, r4[0] + 11, r5[0], y - 2.2, "head", "128 $\\times$ 8 $\\times$ 8  $\\rightarrow$  4 $\\times$ 4")
    flow(ax, r5[0] + 2, r5[0] + 9, r5[1], "linear")
    xz = r5[0] + 11
    for i, v in enumerate(zv):
        plate(ax, xz, y + 1 + i * 1.9, 2.4, 1.7, 2.0, base=BLUE,
              face_white=float(np.clip(0.88 - 1.1 * abs(v), 0.05, 0.9)))
    caption(ax, xz, xz + 2.4 + DX * 2, y - 2.2, "state $\\mathbf{z}_i$", "$d=6$")

    # ============================================================ (b) structure decoder
    rule(ax, 3, 82, 144, "(b)   Structure decoder and residual")
    ax.text(147, 82, "stage 1", ha="right", va="center", fontsize=10.5, color=BLUE)
    y = 62
    for i, v in enumerate(zv):
        plate(ax, 5, y + 4 + i * 1.9, 2.4, 1.7, 2.0, base=BLUE,
              face_white=float(np.clip(0.88 - 1.1 * abs(v), 0.05, 0.9)))
    caption(ax, 5, 8, y - 2.2, None, "state $\\mathbf{z}_i$")
    flow(ax, 11, 18, y + 8, "linear")
    r = plate(ax, 20, y + 5, 4, 4, 10, base=BLUE)
    caption(ax, 20, r[0], y - 2.2, None, "256 $\\times$ 6 $\\times$ 6")
    flow(ax, r[0] + 2, r[0] + 7, r[1])
    r = plates(ax, r[0] + 9, y + 2, 4.2, 10, 5, n=5, gap=1.2, base=BLUE)
    caption(ax, r[0] - 26, r[0], y - 2.2, "upsample $\\times$ 5", "to 192 $\\times$ 192")
    flow(ax, r[0] + 2, r[0] + 9, r[1])
    x = r[0] + 11
    r = plate(ax, x, y + 1, 13, 13, 2.5, base=BLUE, img=struct, cmap=FIRE, vmin=0, vmax=vmax)
    caption(ax, x, r[0], y - 2.2, "structure  $D(\\mathbf{z}_i)$", "$\\bar I+\\Delta(\\mathbf{z}_i)$")
    sign(ax, r[0] + 4.5, r[1], "$-$")
    x = r[0] + 9
    r = plate(ax, x, y + 1, 13, 13, 2.5, base=GREY, img=img, cmap=FIRE, vmin=0, vmax=vmax)
    caption(ax, x, r[0], y - 2.2, "frame $I_i$", None)
    sign(ax, r[0] + 4.5, r[1], "$=$")
    x = r[0] + 9
    r = plate(ax, x, y + 1, 13, 13, 2.5, base=RED, img=res, cmap="RdBu_r", vmin=-rlim, vmax=rlim)
    caption(ax, x, r[0], y - 2.2, "residual  $r_i$", "detail the state cannot carry")

    # ============================================================ (c) detail branch
    rule(ax, 3, 43, 144, "(c)   Residual codebook")
    ax.text(147, 43, "stage 2", ha="right", va="center", fontsize=10.5, color=GREEN)
    y = 23
    r = plate(ax, 4, y + 1, 13, 13, 2.5, base=RED, img=res, cmap="RdBu_r", vmin=-rlim, vmax=rlim)
    caption(ax, 4, r[0], y - 2.2, "$r_i$", None)
    flow(ax, r[0] + 2, r[0] + 7, r[1])
    r = plates(ax, r[0] + 9, y + 3.5, 4.2, 8, 5, n=3, gap=1.2, base=GREEN)
    caption(ax, r[0] - 16, r[0], y - 2.2, "stride-2 conv $\\times$ 3", "32 $\\times$ 24 $\\times$ 24")
    flow(ax, r[0] + 2, r[0] + 9, r[1], "quantise")
    xb = r[0] + 11
    for i in range(6):
        plate(ax, xb + i * 1.45, y + 3.5 - i * 0.28, 1.05, 8, 1.1, base=GREEN,
              face_white=0.86 - 0.035 * i, z=3 + 0.1 * i)
    caption(ax, xb, xb + 9.5, y - 2.2, "codebook", "512 $\\times$ 32")
    xb2 = xb + 6 * 1.45 + 0.6
    flow(ax, xb2 + 1.5, xb2 + 7, y + 7.5)
    r = plate(ax, xb2 + 9, y + 3, 9, 9, 3, base=GREEN, img=codes % 20, cmap="tab20")
    caption(ax, xb2 + 9, r[0], y - 2.2, "codes $c_i$", "24 $\\times$ 24 indices")
    flow(ax, r[0] + 2, r[0] + 7, r[1])
    r = plates(ax, r[0] + 9, y + 3.5, 4.2, 8, 5, n=3, gap=1.2, base=GREEN)
    caption(ax, r[0] - 16, r[0], y - 2.2, "upsample $\\times$ 3", None)
    flow(ax, r[0] + 2, r[0] + 9, r[1])
    x = r[0] + 11
    r = plate(ax, x, y + 1, 13, 13, 2.5, base=GREEN, img=recon, cmap=FIRE, vmin=0, vmax=vmax)
    caption(ax, x, r[0], y - 2.2, "$\\hat I_i = D(\\mathbf{z}_i)+\\hat r_i$", "reconstruction")

    ax.text(75, 51.0, "stage 1   minimise  $\\|I_i - D(\\mathbf{z}_i)\\|^2$   over single frames;"
            "   $D(\\mathbf{z})=\\bar I+\\Delta(\\mathbf{z})$ with $\\bar I$ fixed",
            ha="center", va="center", fontsize=11, color=BLUE)
    ax.text(75, 12.0, "stage 2   stage 1 frozen;   minimise  $|r_i-\\hat r_i|_1$ "
            "$+\\,\\beta\\,\\|\\mathbf{w}_i-\\mathrm{sg}[\\mathbf{c}_i]\\|^2$   "
            "over single frames", ha="center", va="center", fontsize=11, color=GREEN)

    # colour key
    for i, (c, txt) in enumerate([(GREY, "frozen"),
                                  (BLUE, "stage 1: head and structure decoder"),
                                  (GREEN, "stage 2: residual codebook")]):
        x = 3 + i * 46
        ax.add_patch(Rectangle((x, 2.5), 3.2, 2.6, facecolor=mix(c, white=0.82),
                               edgecolor=mix(c, black=0.15), lw=0.9))
        ax.text(x + 4.4, 3.8, txt, ha="left", va="center", fontsize=9.2, color=SOFT)

    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"Figure_9_representation_network.{ext}", dpi=320,
                    bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print("saved", OUT / "Figure_9_representation_network.png")


if __name__ == "__main__":
    build()
