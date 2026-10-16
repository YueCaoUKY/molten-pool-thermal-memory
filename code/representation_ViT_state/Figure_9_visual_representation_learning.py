#!/usr/bin/env python3
"""Publication schematic for frame-wise visual representation learning.

The visual hierarchy follows architecture figures used in major vision
conferences: method-defining objects are large; implementation dimensions are
secondary annotations. Text and geometry remain editable in SVG.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "Paper writing" / "development_notes" / "section6" / (
    "Figure_9_visual_representation_learning_data.npz"
)
PNG = HERE / "Figure_9_visual_representation_learning.png"
PDF = HERE / "Figure_9_visual_representation_learning.pdf"
SVG = HERE / "Figure_9_visual_representation_learning.svg"

INK = "#1D2935"
MUTED = "#6E7B87"
LINE = "#AAB5BE"
HAIR = "#DCE3E8"
GREY = "#7B8791"
BLUE = "#1763A5"
BLUE2 = "#4388C2"
CYAN = "#49A7B6"
VIOLET = "#7563A7"
GOLD = "#C78A28"

plt.rcParams.update(
    {
        "font.family": "Nimbus Sans",
        "font.size": 7.0,
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

FIRE = LinearSegmentedColormap.from_list(
    "pool", ["#060911", "#18324E", "#79334D", "#D86439", "#FFE6A5"]
)
FEATURE = LinearSegmentedColormap.from_list(
    "feature", ["#F7FAFC", "#C8DAE8", "#6494BB", "#29496F"]
)


def mix(color: str, white: float = 0.0, black: float = 0.0):
    rgb = np.asarray(to_rgb(color))
    rgb = rgb + (1.0 - rgb) * white
    rgb *= 1.0 - black
    return tuple(np.clip(rgb, 0, 1))


def rounded(ax, x, y, w, h, *, fc="white", ec=HAIR, lw=0.7, radius=1.2,
            zorder=2, alpha=1.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.16,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x0, x1, y, *, color=LINE, lw=1.15):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9,
            linewidth=lw, color=color, shrinkA=0.8, shrinkB=0.8, zorder=20,
        )
    )


def image_card(ax, x, y, w, h, image, *, edge, label, cmap=FIRE,
               vmin=None, vmax=None, patch_grid=False):
    # One shallow back plate gives depth without competing with the network.
    rounded(ax, x + 0.75, y - 0.55, w, h, fc="#E1E7EB", ec="none",
            radius=0.85, zorder=1, alpha=0.75)
    clip = rounded(ax, x, y, w, h, fc="white", ec=edge, lw=0.75,
                   radius=0.85, zorder=3)
    im = ax.imshow(image, extent=(x, x + w, y, y + h), cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="bilinear",
                   aspect="auto", zorder=4)
    im.set_clip_path(clip)
    if patch_grid:
        for k in range(1, 6):
            xx = x + w * k / 6
            yy = y + h * k / 6
            ax.plot([xx, xx], [y, y + h], color="white", lw=0.26,
                    alpha=0.34, zorder=5)
            ax.plot([x, x + w], [yy, yy], color="white", lw=0.26,
                    alpha=0.34, zorder=5)
    rounded(ax, x, y, w, h, fc="none", ec=edge, lw=0.75,
            radius=0.85, zorder=6)
    ax.text(x + w / 2, y - 1.45, label, ha="center", va="top",
            fontsize=6.1, color=INK, zorder=8)


def group_card(ax, x, y, w, h, title, color):
    rounded(ax, x, y, w, h, fc=mix(color, white=0.975), ec=mix(color, white=0.72),
            lw=0.58, radius=1.4, zorder=0)
    ax.text(x + 2.0, y + h - 2.2, title, ha="left", va="center",
            fontsize=6.6, color=INK, fontweight="bold", zorder=10)
    ax.plot([x + 2.0, x + w - 2.0], [y + h - 4.0, y + h - 4.0],
            color=mix(color, white=0.78), lw=0.55, zorder=1)


def frozen_backbone(ax, x, y):
    """One clear backbone object; stacked sheets indicate transformer depth."""
    w, h = 11.2, 13.8
    for k in range(3, 0, -1):
        rounded(ax, x - 0.62 * k, y - 0.42 * k, w, h,
                fc=mix(GREY, white=0.74 + 0.035 * k), ec=mix(GREY, white=0.18),
                lw=0.45, radius=0.75, zorder=3 + 0.05 * k)
    rounded(ax, x, y, w, h, fc=mix(GREY, white=0.68), ec=mix(GREY, black=0.04),
            lw=0.65, radius=0.75, zorder=5)
    ax.text(x + w / 2, y + 8.1, "DINOv2", ha="center", va="center",
            fontsize=7.2, color=INK, fontweight="bold", zorder=8)
    ax.text(x + w / 2, y + 5.8, "ViT-B/14", ha="center", va="center",
            fontsize=5.3, color=MUTED, zorder=8)
    ax.plot([x + 2.4, x + w - 2.4], [y + 3.55, y + 3.55],
            color=mix(GREY, white=0.45), lw=0.48, zorder=8)
    ax.text(x + w / 2, y + 2.25, "frozen backbone", ha="center", va="center",
            fontsize=4.25, color=GREY, fontweight="bold", zorder=8)
    return x + w


def token_tile(ax, x, y, image):
    w = h = 6.6
    # Only one spatial feature tile is needed to establish that the backbone
    # retains a patch grid.
    rounded(ax, x + 0.75, y + 0.55, w, h, fc=mix(GREY, white=0.76),
            ec=mix(GREY, white=0.12), lw=0.45, radius=0.45, zorder=4)
    front = rounded(ax, x, y, w, h, fc="white", ec=mix(GREY, white=0.06),
                    lw=0.55, radius=0.45, zorder=6)
    im = ax.imshow(image, extent=(x, x + w, y, y + h), cmap=FEATURE,
                   interpolation="nearest", aspect="auto", zorder=7)
    im.set_clip_path(front)
    rounded(ax, x, y, w, h, fc="none", ec=mix(GREY, white=0.06),
            lw=0.55, radius=0.45, zorder=8)
    ax.text(x + w / 2 + 0.3, y - 1.15, "patch features", ha="center", va="top",
            fontsize=4.65, color=MUTED, zorder=10)
    return x + w + 0.75


def compression_head(ax, x, y):
    """A tapered block communicates compression more clearly than layer cubes."""
    stages = [
        (x, y, 4.2, 11.0, BLUE2),
        (x + 4.6, y + 1.6, 3.5, 7.8, BLUE),
        (x + 8.5, y + 3.0, 2.8, 5.0, VIOLET),
    ]
    for xx, yy, ww, hh, color in stages:
        ax.add_patch(
            Polygon(
                [(xx, yy), (xx + ww - 0.55, yy + 0.35),
                 (xx + ww - 0.55, yy + hh - 0.35), (xx, yy + hh)],
                closed=True, facecolor=mix(color, white=0.79),
                edgecolor=color, lw=0.62, zorder=6,
            )
        )
        ax.add_patch(
            Polygon(
                [(xx + ww - 0.55, yy + 0.35), (xx + ww, yy + 0.72),
                 (xx + ww, yy + hh), (xx + ww - 0.55, yy + hh - 0.35)],
                closed=True, facecolor=mix(color, white=0.52),
                edgecolor=color, lw=0.48, zorder=5,
            )
        )
    ax.text(x + 5.4, y + 12.3, "compression head", ha="center", va="bottom",
            fontsize=5.1, color=BLUE, fontweight="bold", zorder=9)
    ax.text(x + 5.4, y - 0.15, "trainable", ha="center", va="top",
            fontsize=4.15, color=BLUE, fontweight="bold", zorder=9)
    return x + 11.3


def latent_state(ax, x, y, z):
    """The bottleneck is the most distinctive object in the diagram."""
    w, h = 19.0, 21.0
    for radius, alpha in [(8.0, 0.035), (6.1, 0.045), (4.4, 0.055)]:
        ax.add_patch(Circle((x + w / 2, y + 10.6), radius,
                            facecolor=mix(VIOLET, white=0.54), edgecolor="none",
                            alpha=alpha, zorder=0))
    # Mathematical brackets make the object read as a compact vector rather than
    # another neural-network module.
    bracket_color = mix(VIOLET, white=0.18)
    left, right = x + 0.7, x + w - 0.7
    bottom, top = y + 5.0, y + 16.4
    for xx, direction in [(left, 1), (right, -1)]:
        ax.plot([xx, xx], [bottom, top], color=bracket_color, lw=1.05, zorder=3)
        ax.plot([xx, xx + direction * 1.55], [top, top], color=bracket_color,
                lw=1.05, zorder=3)
        ax.plot([xx, xx + direction * 1.55], [bottom, bottom], color=bracket_color,
                lw=1.05, zorder=3)
    ax.text(x + w / 2, y + h - 1.9, "compact pool state", ha="center", va="center",
            fontsize=6.2, color=VIOLET, fontweight="bold", zorder=8)

    colors = [BLUE, BLUE2, CYAN, VIOLET, GOLD, "#B65352"]
    mag = np.abs(z) / max(float(np.max(np.abs(z))), 1e-6)
    x0 = x + 2.1
    for k, (value, color) in enumerate(zip(mag, colors)):
        xx = x0 + k * 2.48
        rounded(ax, xx, y + 8.1, 1.35, 6.2,
                fc=mix(color, white=0.88 - 0.42 * float(value)),
                ec=mix(color, white=0.28), lw=0.48, radius=0.62, zorder=5)
        ax.text(xx + 0.675, y + 6.65, rf"$z_{k+1}$", ha="center", va="center",
                fontsize=4.4, color=MUTED, zorder=8)
    ax.text(x + w / 2, y + 2.8, r"$\mathbf{z}_t\in\mathbb{R}^{6}$",
            ha="center", va="center", fontsize=6.1, color=INK,
            fontweight="bold", zorder=8)
    return x + w


def tensor_plate(ax, x, y, w, h, depth, color):
    dx, dy = depth * 0.38, depth * 0.21
    ax.add_patch(
        Polygon([(x, y + h), (x + dx, y + h + dy),
                 (x + w + dx, y + h + dy), (x + w, y + h)],
                closed=True, facecolor=mix(color, white=0.91),
                edgecolor=color, lw=0.48, zorder=5)
    )
    ax.add_patch(
        Polygon([(x + w, y), (x + w + dx, y + dy),
                 (x + w + dx, y + h + dy), (x + w, y + h)],
                closed=True, facecolor=mix(color, white=0.59),
                edgecolor=color, lw=0.48, zorder=5)
    )
    rounded(ax, x, y, w, h, fc=mix(color, white=0.81), ec=color,
            lw=0.62, radius=0.45, zorder=6)
    return x + w + dx


def structure_decoder(ax, x, y):
    group_card(ax, x, y, 36.0, 24.0, r"structure decoder  $D$", BLUE)
    ax.text(x + 33.8, y + 21.8, "trainable", ha="right", va="center",
            fontsize=4.35, color=BLUE, fontweight="bold", zorder=10)
    ax.text(x + 18.0, y + 17.0, r"upsampling blocks  $\times5$",
            ha="center", va="center", fontsize=5.0, color=BLUE,
            fontweight="bold", zorder=10)

    specs = [
        (x + 3.0, y + 7.2, 3.6, 5.2, 3.3),
        (x + 10.0, y + 6.1, 4.3, 7.4, 2.8),
        (x + 18.1, y + 4.7, 5.0, 10.3, 2.4),
        (x + 27.2, y + 3.1, 5.5, 13.4, 2.0),
    ]
    ends = []
    for xx, yy, ww, hh, dd in specs:
        ends.append(tensor_plate(ax, xx, yy, ww, hh, dd, BLUE))
    for a, b in zip(ends[:-1], [s[0] for s in specs[1:]]):
        arrow(ax, a + 0.25, b - 0.25, y + 10.0, color=mix(BLUE, white=0.32), lw=0.72)
    ax.text(x + 18.0, y + 1.9,
            r"$256\times6\times6\ \longrightarrow\ 192\times192$",
            ha="center", va="center", fontsize=4.7, color=MUTED, zorder=10)
    return x + 36.0


def reconstruction_objective(ax, x_input, x_output):
    y_top, y_mid = 18.8, 13.1
    color = mix(BLUE, white=0.52)
    ax.plot([x_input, x_input, 69.0], [y_top, y_mid, y_mid],
            color=color, lw=0.75, linestyle=(0, (3, 2)), zorder=1)
    ax.plot([x_output, x_output, 111.0], [y_top, y_mid, y_mid],
            color=color, lw=0.75, linestyle=(0, (3, 2)), zorder=1)
    ax.text(90.0, 13.25,
            r"reconstruction objective   $\mathcal{L}_{\mathrm{state}}="
            r"\Vert I_t-D(\mathbf{z}_t)\Vert_2^2$",
            ha="center", va="center", fontsize=5.2, color=BLUE,
            fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none"))
    ax.text(90.0, 10.75, "single frame  ·  no geometry labels",
            ha="center", va="center", fontsize=4.4, color=MUTED, zorder=5)


def build():
    d = np.load(DATA)
    image = d["image"]
    structure = d["structure"]
    z = d["z"]
    vmax = float(max(np.percentile(image, 99.8), np.percentile(structure, 99.8)))

    fig = plt.figure(figsize=(7.18, 2.02), facecolor="white")
    ax = fig.add_axes([0.012, 0.025, 0.976, 0.955])
    ax.set_xlim(0, 180)
    ax.set_ylim(5.2, 49.2)
    ax.set_aspect("equal")
    ax.axis("off")

    image_card(ax, 4.5, 23.0, 18.0, 18.0, image, edge=GREY,
               label=r"molten-pool frame  $I_t$", vmin=0, vmax=vmax,
               patch_grid=True)

    group_card(ax, 27.5, 20.0, 48.5, 24.0, r"visual encoder  $E_{\psi}$", BLUE)
    backbone_end = frozen_backbone(ax, 31.5, 25.0)
    arrow(ax, backbone_end + 0.7, 48.0, 31.9, color=GREY, lw=0.9)
    token_end = token_tile(ax, 48.7, 28.6, d["tokens16"])
    arrow(ax, token_end + 0.4, 58.0, 31.9, color=BLUE, lw=0.9)
    compression_head(ax, 58.5, 26.4)
    ax.text(51.8, 21.7,
            r"$16\times16\times768\ \rightarrow\ 8\times8\times768\ \rightarrow\ 6$",
            ha="center", va="center", fontsize=4.55, color=MUTED, zorder=10)

    latent_state(ax, 83.0, 21.5, z)
    structure_decoder(ax, 111.0, 20.0)

    image_card(ax, 157.5, 23.0, 18.0, 18.0, structure, edge=BLUE,
               label=r"decoded structure  $D(\mathbf{z}_t)$",
               vmin=0, vmax=vmax)

    # Major inference arrows only.
    arrow(ax, 23.3, 27.0, 32.0, color=GREY)
    arrow(ax, 76.5, 82.5, 32.0, color=VIOLET)
    arrow(ax, 102.5, 110.5, 32.0, color=BLUE)
    arrow(ax, 147.5, 157.0, 32.0, color=BLUE)

    reconstruction_objective(ax, 13.5, 166.5)

    for path in (PNG, PDF, SVG):
        fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
    plt.close(fig)
    print(SVG)


if __name__ == "__main__":
    build()
