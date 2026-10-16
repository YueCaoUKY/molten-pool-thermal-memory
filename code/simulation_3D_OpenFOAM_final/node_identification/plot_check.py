#!/usr/bin/env python3
"""Inspection figures: one per record and identification share.

Nine panels. (a) the two process inputs, with the identification interval shaded, the division
dashed, and the four constant-input windows marked. (b)-(h) the seven pool-state coordinates, each
showing the finite-volume reference, the two-state NODE and ARX. (i) the identified hidden state
against the withheld thermal reference.

The title carries the held-out R^2 of both models and says so when the NODE is the weaker one.

    python plot_check.py            # both shares, all twelve records
"""
import sys
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import baselines as B
import identify_3d as I
from delta_variants import ring_heat, window_for

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
FIG = HERE / "figures" / "check"
V2 = {"PP_steel_speed_free": "PP_steel_speed_free_v2",
      "PP_steel_speed_cooled": "PP_steel_speed_cooled_v2"}
CASES = [V2.get(c, c) for c in
         (f"PP_{m}_{d}" for m in ("steel", "ss304")
          for d in ("4_slow_cycles", "5_slow_cooled", "speed_free", "speed_cooled",
                    "both_free", "both_cooled"))]
NAMES = ["area (mm$^2$)", "length (mm)", "width (mm)", "tail length (mm)",
         "perimeter (mm)", "width at arc (mm)", "centroid lag (mm)"]
REF, NODE, ARX, HEAT = "#252525", "#2166ac", "#d95f02", "#6b6b6b"
SHADE = "#eef1f5"
W4 = [(50, 60), (110, 120), (170, 180), (230, 240)]
W6 = [(30, 40), (70, 80), (110, 120), (150, 160), (190, 200), (230, 240)]

try:
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    FONT = "Times New Roman"
except ValueError:
    FONT = "Nimbus Roman"


def style():
    mpl.rcParams.update({"font.family": FONT, "mathtext.fontset": "custom", "mathtext.rm": FONT,
                         "mathtext.it": f"{FONT}:italic", "font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
                         "axes.linewidth": 0.6, "lines.linewidth": 1.0, "savefig.dpi": 200})


def r2(z_true, z_pred, held, sigma):
    t, p = z_true[held] / sigma, z_pred[held] / sigma
    sse = np.sum((p - t) ** 2, axis=0)
    sst = np.sum((t - t.mean(axis=0)) ** 2, axis=0)
    return float(1.0 - sse.sum() / max(sst.sum(), 1e-12))


def load(case, tag):
    s, z_raw, u_np, cur, p, v, *_ = I.load_record(case, I.STATE_NAMES, window_for(case), False)
    t = s["t_s"].to_numpy()
    sigma = np.maximum(z_raw.std(axis=0), 1e-9)
    d = np.load(OUT / f"{case}_v_pool_simpleD{tag}" / "fit.npz", allow_pickle=True)
    n = int(d["n_train"])
    held = np.zeros(len(t), bool); held[n:] = True
    a = np.load(OUT / "baselines" / f"{case}_arx{tag}.npz")
    zn = ((z_raw - a["z_mean"]) / a["z_std"]).astype(np.float64)
    pred, _ = B.arx_predict(a["theta"], zn, u_np, int(a["na"]), int(a["nb"]), True)
    z_arx = pred * a["z_std"] + a["z_mean"]
    ring = ring_heat(case, t)
    return dict(t=t, cur=cur, v=v, z=z_raw, node=d["z_pred"], arx=z_arx, delta=d["delta_hat"],
                ring=ring, split=float(t[n]), n=n,
                r2_node=r2(z_raw, d["z_pred"], held, sigma), r2_arx=r2(z_raw, z_arx, held, sigma))


def figure(case, tag, share):
    r = load(case, tag)
    t, split = r["t"], r["split"]
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 7.6), sharex=True,
                             gridspec_kw=dict(hspace=0.42, wspace=0.30))
    ax = axes.ravel()

    def mark(a, windows=False):
        a.axvspan(t[0], split, color=SHADE, zorder=0)
        a.axvline(split, color="0.45", lw=0.8, ls=(0, (3, 2)), zorder=2)
        if windows:
            for w0, w1 in (W6 if case.endswith('_v2') else W4):
                a.axvspan(w0, w1, color="#fdf0d5", zorder=0)
        a.set_xlim(t[0], t[-1])
        a.spines[["top", "right"]].set_visible(False)
        a.grid(color="#dddddd", lw=0.4); a.set_axisbelow(True)

    a0 = ax[0]
    a0.step(t, r["cur"], where="post", color=NODE, lw=1.0)
    a0.set_ylabel("$I$ (A)", color=NODE); a0.tick_params(axis="y", colors=NODE)
    a1 = a0.twinx()
    a1.step(t, r["v"], where="post", color=ARX, lw=1.0)
    a1.set_ylabel("$v$ (mm/s)", color=ARX); a1.tick_params(axis="y", colors=ARX)
    a1.spines[["top"]].set_visible(False)
    mark(a0, windows=True)
    a0.set_title("(a) process inputs", loc="left", fontsize=9)

    for j in range(7):
        a = ax[j + 1]
        a.plot(t, r["z"][:, j], color=REF, lw=1.1, label="finite volume", zorder=4)
        a.plot(t, r["node"][:, j], color=NODE, lw=0.95, ls=(0, (4, 1.8)), label="NODE with $\\delta$", zorder=5)
        a.plot(t, r["arx"][:, j], color=ARX, lw=0.95, ls=(0, (2, 1.6)), label="ARX", zorder=5)
        lo, hi = np.nanmin(r["z"][:, j]), np.nanmax(r["z"][:, j])
        pad = 0.10 * (hi - lo)
        a.set_ylim(lo - pad, hi + pad)
        mark(a)
        a.set_ylabel(NAMES[j])
        a.set_title(f"({chr(98 + j)}) $z_{{{j + 1}}}$", loc="left", fontsize=9)

    a8 = ax[8]
    a8.plot(t, r["delta"], color=NODE, lw=1.1)
    a8.set_ylabel("$\\delta$ (a.u.)", color=NODE); a8.tick_params(axis="y", colors=NODE)
    a9 = a8.twinx()
    a9.plot(t, r["ring"], color=HEAT, lw=1.0, ls=(0, (4, 2)))
    a9.set_ylabel("$\\Delta T_{\\mathrm{w}}$ (K)", color=HEAT); a9.tick_params(axis="y", colors=HEAT)
    a9.spines[["top"]].set_visible(False)
    mark(a8)
    a8.set_title("(i) hidden state and withheld reference", loc="left", fontsize=9)

    for a in axes[2]:
        a.set_xlabel("time (s)")
    h, l = ax[1].get_legend_handles_labels()
    h.append(mpl.patches.Patch(color=SHADE)); l.append("identification interval")
    h.append(mpl.patches.Patch(color="#fdf0d5")); l.append("constant-input windows")
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.015))

    worse = " — NODE below ARX" if r["r2_node"] < r["r2_arx"] else ""
    fig.suptitle(f"{case}   identification {share}   held-out $R^2$:  "
                 f"NODE {r['r2_node']:.3f}   ARX {r['r2_arx']:.3f}{worse}",
                 fontsize=11, y=0.985)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.10)
    d = FIG / f"p{share.rstrip('%')}"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{case}.png", bbox_inches="tight")
    plt.close(fig)
    return r["r2_node"], r["r2_arx"]


if __name__ == "__main__":
    style()
    for tag, share in (("_p60", "60%"), ("_p50", "50%")):
        for c in CASES:
            n, a = figure(c, tag, share)
            print(f"{share} {c:26s} NODE {n:7.3f}  ARX {a:7.3f}{'  <-- NODE worse' if n < a else ''}")
    print("written to", FIG)
