#!/usr/bin/env python3
"""WAAM: what the memory state accounts for when the input never changes.

The deposition runs at fixed settings, so nothing in the input can explain the drift that the
recording shows. The same structure is fitted twice per record, once with the memory state and
once without it. Without it the pool state can only relax towards a fixed equilibrium.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
RES = HERE / "results_node_WAAM"
DT = 0.08
INK, BLUE, RED, SOFT = "#272727", "#1F5C9E", "#B64342", "#7A8189"
plt.rcParams.update({
    "font.family": ["Nimbus Sans", "DejaVu Sans", "sans-serif"], "font.size": 9.5,
    "axes.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.frameon": False,
    "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold",
    "svg.fonttype": "none"})

RECORDS = ("video_1", "video_3", "video_5")
SCORES = {"video_1": (0.968, 0.444), "video_3": (0.904, 0.459), "video_5": (0.944, 0.334)}


def build():
    fig = plt.figure(figsize=(12.6, 9.0))
    outer = fig.add_gridspec(3, 1, hspace=0.42, top=0.90, bottom=0.07, left=0.07, right=0.985)
    for r, sid in enumerate(RECORDS):
        full = np.load(RES / f"fit_{sid}_s25_e1200.npz")
        none = np.load(RES / f"fit_{sid}_s25_e1200_nodelta.npz")
        zt, pf, pn, dl = full["z_true"], full["z_pred"], none["z_pred"], full["delta"]
        t = np.arange(len(zt)) * DT
        inner = outer[r].subgridspec(1, 7, wspace=0.34, width_ratios=[1] * 6 + [1.15])
        for k in range(6):
            ax = fig.add_subplot(inner[k])
            ax.plot(t, zt[:, k], color=INK, lw=0.9, alpha=0.8)
            ax.plot(t, pn[:, k], color=SOFT, lw=1.5, ls=(0, (3, 1.6)))
            ax.plot(t, pf[:, k], color=BLUE, lw=1.6)
            ax.set_xlim(t[0], t[-1])
            ax.set_title(f"$z_{k+1}$", fontsize=10, pad=4)
            ax.tick_params(labelsize=7.5)
            if r == len(RECORDS) - 1:
                ax.set_xlabel("time (s)", fontsize=8.5)
            if k == 0:
                a, b = SCORES[sid]
                ax.set_ylabel(f"{sid}\n$R^2$  {a:.2f}  vs  {b:.2f}", fontsize=9.5,
                              linespacing=1.6)
        ax = fig.add_subplot(inner[6])
        ax.plot(t, dl, color=RED, lw=1.6)
        ax.fill_between(t, 0, dl, color=RED, alpha=0.12)
        ax.set_xlim(t[0], t[-1])
        ax.set_title("$\\delta$", fontsize=10, pad=4, color=RED)
        ax.tick_params(labelsize=7.5)
        if r == len(RECORDS) - 1:
            ax.set_xlabel("time (s)", fontsize=8.5)

    handles = [plt.Line2D([], [], color=INK, lw=1.2, alpha=0.8, label="state read from the images"),
               plt.Line2D([], [], color=BLUE, lw=1.8, label="predicted, with the memory state"),
               plt.Line2D([], [], color=SOFT, lw=1.6, ls=(0, (3, 1.6)),
                          label="predicted, memory state removed")]
    fig.legend(handles=handles, ncol=3, fontsize=10, loc="upper center",
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("WAAM: deposition at fixed settings, so no input change can account for the drift",
                 y=0.995, fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"waam_memory_ablation.{ext}", dpi=280, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("saved", FIGS / "waam_memory_ablation.png")


if __name__ == "__main__":
    build()
