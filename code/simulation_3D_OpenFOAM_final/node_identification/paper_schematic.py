#!/usr/bin/env python3
"""Assemble the four ParaView stills into one 2x2 schematic figure (no text anywhere).

The stills come from code/visualize/render_schematic.py:

    pvbatch code/visualize/render_schematic.py PP_steel_4_slow_cycles 150.0 \
        node_identification/figures/paper/render

    python paper_schematic.py
"""
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures" / "paper"
RENDER = FIG / "render"
PANELS = ("scene", "pool", "states", "delta")


def main():
    imgs = [plt.imread(RENDER / f"{n}.png") for n in PANELS]
    h, w = imgs[0].shape[:2]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 7.16 * h / w), gridspec_kw=dict(
        wspace=0.012, hspace=0.012, left=0, right=1, top=1, bottom=0))
    for ax, im in zip(axes.ravel(), imgs):
        ax.imshow(im, interpolation="lanczos")
        ax.set_axis_off()
    fig.savefig(FIG / "schematic_2x2.png", bbox_inches="tight", pad_inches=0.01, dpi=620)
    fig.savefig(FIG / "schematic_2x2.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote", FIG / "schematic_2x2.png")


if __name__ == "__main__":
    main()
