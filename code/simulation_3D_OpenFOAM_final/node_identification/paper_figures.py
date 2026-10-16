#!/usr/bin/env python3
"""Paper figures for identification and adaptation (final design, tag _v_pool_simpleD).

Figure 1 (identification), three rows by two columns, one record:
  row 1  the two process inputs
  row 2  the seven pool-state coordinates, measured and predicted, with and without the
         hidden thermal state
  row 3  what the model carries as thermal information in each case, against the withheld
         thermal reference

Figure 2 (adaptation), four rows by two columns, one transfer: the same three rows for the
adapted and the unchanged source model, plus the calibration history.

  python paper_figures.py
"""
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
FIG = HERE / "figures" / "paper"
TAG = "_v_pool_simpleD"
STEP = 50                                  # 5 s lag for the falling-phase agreement
CASE = "PP_steel_4_slow_cycles"            # largest gap between the model with and without delta
PAIR = ("PP_ss304_4_slow_cycles", "PP_ss304_5_slow_cooled")

MEAS, PRED, SRC, DELTA, REF = "0.15", "#c62828", "#1565c0", "#c62828", "0.45"
SHORT = {"area_mm2": "area", "length_mm": "length", "width_mm": "width",
         "tail_length_mm": "tail length", "perimeter_mm": "perimeter",
         "width_at_arc_mm": "width at arc", "centroid_lag_mm": "centroid lag"}


def style():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 0.9,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "legend.frameon": False, "savefig.dpi": 400,
        "figure.dpi": 130, "axes.grid": False, "legend.handlelength": 1.8,
    })


def ring_heat(case, t):
    s = pd.read_csv(HERE.parent / "results" / case / "signals_0p1s.csv")
    s.index = np.round(s["t_s"], 3)
    return s.reindex(np.round(t, 3))["dHK_ring_5_15mm_K"].to_numpy()


def agreement(delta, heat):
    dx, dy = delta[STEP:] - delta[:-STEP], heat[STEP:] - heat[:-STEP]
    thr = 0.02 * (np.nanmax(heat) - np.nanmin(heat))
    drop = dy < -thr
    s_minus = float(np.mean(dx[drop] < 0)) if drop.any() else np.nan
    return float(spearmanr(delta, heat)[0]), s_minus


def shade(ax, t0, t1):
    ax.axvspan(t0, t1, color="0.88", zorder=0)


def panel_signal(ax, t, y, ylabel, color=MEAS, cal=None):
    ax.plot(t, y, color=color, lw=1.0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("time (s)")
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    if hi - lo < 1e-3 * max(abs(hi), 1e-9):
        ax.set_ylim(0.0, 1.6 * hi)
        ax.set_yticks([0.0, hi])
        ax.annotate("constant", xy=(0.03, 0.88), xycoords="axes fraction", fontsize=7, color=color)
    if cal:
        shade(ax, *cal)


def panel_states_3d(ax, t, z_true, z_pred, names, cal=None):
    """The seven coordinates staggered in depth; measured and predicted superposed."""
    lo = z_true.min(axis=0)
    span = np.maximum(z_true.max(axis=0) - lo, 1e-9)
    for j in range(z_true.shape[1]):
        y = np.full_like(t, j)
        ax.plot(t, y, (z_true[:, j] - lo[j]) / span[j], color=MEAS, lw=0.6, zorder=3)
        ax.plot(t, y, (z_pred[:, j] - lo[j]) / span[j], color=PRED, lw=0.6, ls=(0, (3, 1.6)), zorder=4)
    if cal is not None:
        n = z_true.shape[1] - 1
        for j in range(z_true.shape[1]):
            ax.plot([cal, cal], [j, j], [0, 1], color="0.55", lw=0.7, zorder=5)
        for z0 in (0.0, 1.0):
            ax.plot([cal, cal], [0, n], [z0, z0], color="0.55", lw=0.7, zorder=5)
    ax.set_yticks(range(z_true.shape[1]))
    ax.set_yticklabels([SHORT[n] for n in names], fontsize=5.8, va="center_baseline")
    ax.set_xlabel("time (s)", fontsize=7.5, labelpad=-4)
    ax.set_zlabel("scaled value", fontsize=7.5, labelpad=-6)
    ax.tick_params(labelsize=6.3, pad=-2.5)
    ax.tick_params(axis="y", pad=-3.5)
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((2.3, 1.25, 0.85), zoom=1.32)


def panel_delta(ax, t, delta, heat, label=r"$\delta$ (-)", cal=None):
    ax.plot(t, delta, color=DELTA, lw=1.0)
    ax.set_ylabel(label, color=DELTA)
    ax.tick_params(axis="y", colors=DELTA, width=0.6, length=2.5)
    ax.spines["left"].set_color(DELTA)
    ax.set_xlabel("time (s)")
    ax2 = ax.twinx()
    ax2.plot(t, heat, color=REF, lw=1.0, ls=(0, (4, 2)))
    ax2.set_ylabel(r"$\Delta T_{\mathrm{ring}}$ (K)", color=REF)
    ax2.tick_params(axis="y", colors=REF, width=0.6, length=2.5)
    ax2.spines["right"].set_color(REF)
    ax2.spines["right"].set_linewidth(0.6)
    ax2.spines["left"].set_color(DELTA)
    if cal:
        shade(ax, *cal)
    return ax2


def load_fit(case, tag):
    d = np.load(OUT / (case + tag) / "fit.npz", allow_pickle=True)
    m = json.loads((OUT / (case + tag) / "metrics.json").read_text())
    return d, m


def fig_identification(case=CASE):
    d, m = load_fit(case, TAG)
    dn, mn = load_fit(case, "_nodelta")
    t, names = d["t"], [str(s) for s in d["state_names"]]
    heat = ring_heat(case, t)
    rho, s_minus = agreement(d["delta_hat"], heat)

    fig = plt.figure(figsize=(7.16, 6.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.80, 1.05, 0.90], hspace=0.50, wspace=0.42)
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    panel_signal(a1, t, d["P_abs_W"] * 1e-3, "absorbed power (kW)")
    panel_signal(a2, t, d["speed_mm_s"], "travel speed (mm s$^{-1}$)", color=SRC)
    a1.set_title("(a) heat input", loc="left", pad=5)
    a2.set_title("(b) travel speed", loc="left", pad=5)

    b1 = fig.add_subplot(gs[1, 0], projection="3d")
    b2 = fig.add_subplot(gs[1, 1], projection="3d")
    panel_states_3d(b1, t, d["z_true"], d["z_pred"], names)
    panel_states_3d(b2, t, dn["z_true"], dn["z_pred"], names)
    b1.set_title(f"(c) with the hidden state, eq. (3): $R^2$ = {m['r2_global']:.3f}",
                 loc="left", fontsize=7.6, y=0.94)
    b2.set_title(f"(d) without the hidden state: $R^2$ = {mn['r2_global']:.3f}",
                 loc="left", fontsize=7.6, y=0.94)

    c1, c2 = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    panel_delta(c1, t, d["delta_hat"], heat)
    c1.set_title(f"(e) identified $\\delta$: $\\rho$ = {rho:.2f}, $S^-$ = {s_minus:.2f}",
                 loc="left", pad=5, fontsize=7.6)
    ck = __import__("torch").load(OUT / (case + "_nodelta") / "checkpoint.pt", weights_only=False)
    res = ((dn["z_pred"] - dn["z_true"]) / ck["z_std"]).mean(axis=1)
    c2.plot(t, res, color=PRED, lw=1.0)
    c2.axhline(0.0, color="0.7", lw=0.5)
    c2.set_ylabel("mean scaled residual (-)", color=PRED)
    c2.tick_params(axis="y", colors=PRED, width=0.6, length=2.5)
    c2.spines["left"].set_color(PRED)
    c2.set_xlabel("time (s)")
    c22 = c2.twinx()
    c22.plot(t, heat, color=REF, lw=1.0, ls=(0, (4, 2)))
    c22.set_ylabel(r"$\Delta T_{\mathrm{ring}}$ (K)", color=REF)
    c22.tick_params(axis="y", colors=REF, width=0.6, length=2.5)
    c22.spines["right"].set_color(REF)
    c22.spines["left"].set_color(PRED)
    rr = float(spearmanr(res, heat)[0])
    c2.set_title(f"(f) no hidden state: prediction error vs reference, $\\rho$ = {rr:.2f}",
                 loc="left", pad=5, fontsize=7.6)

    h = [plt.Line2D([], [], color=MEAS, lw=0.9), plt.Line2D([], [], color=PRED, lw=0.9, ls=(0, (3.5, 1.8))),
         plt.Line2D([], [], color=REF, lw=0.9, ls=(0, (4, 2)))]
    fig.legend(h, ["finite-volume simulation", "model prediction from the input sequence only",
                   r"withheld thermal reference $\Delta T_{\mathrm{ring}}$"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.025), columnspacing=2.0)
    fig.subplots_adjust(left=0.09, right=0.91, top=0.96, bottom=0.085)
    fig.savefig(FIG / "identification.png", bbox_inches="tight")
    fig.savefig(FIG / "identification.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_adaptation(pair=PAIR):
    src, tgt = pair
    d = np.load(OUT / f"adapt_{src}_to_{tgt}_mse_reg{TAG}_p40.npz", allow_pickle=True)
    t, names, n_pre = d["t"], [str(s) for s in d["state_names"]], int(d["n_pre"])
    t_cal = float(t[n_pre])
    cal = (float(t[0]), t_cal)
    sc = list(json.loads((OUT / "adapt_parts_p40" / f"{src}_to_{tgt}.json").read_text()).values())[0]
    rho_a, _ = agreement(d["delta_adapted"][n_pre:], d["ring_heat"][n_pre:])
    rho_s, _ = agreement(d["delta_source"][n_pre:], d["ring_heat"][n_pre:])

    fig = plt.figure(figsize=(7.16, 8.8))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.78, 1.00, 0.85, 0.85], hspace=0.58, wspace=0.42)
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    panel_signal(a1, t, d["P_abs_W"] * 1e-3, "absorbed power (kW)", cal=cal)
    panel_signal(a2, t, d["speed_mm_s"], "travel speed (mm s$^{-1}$)", color=SRC, cal=cal)
    a1.set_title("(a) heat input of the target run", loc="left", pad=5)
    a2.set_title("(b) travel speed of the target run", loc="left", pad=5)

    b1 = fig.add_subplot(gs[1, 0], projection="3d")
    b2 = fig.add_subplot(gs[1, 1], projection="3d")
    panel_states_3d(b1, t, d["z_true"], d["z_adapted"], names, cal=t_cal)
    panel_states_3d(b2, t, d["z_true"], d["z_source"], names, cal=t_cal)
    b1.set_title(f"(c) after calibration: held-out $R^2$ = {sc['r2_adapted']:.2f}",
                 loc="left", fontsize=7.6, y=0.94)
    b2.set_title(f"(d) source model unchanged: held-out $R^2$ = {sc['r2_none']:.2f}",
                 loc="left", fontsize=7.6, y=0.94)

    c1, c2 = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    panel_delta(c1, t, d["delta_adapted"], d["ring_heat"], cal=cal)
    panel_delta(c2, t, d["delta_source"], d["ring_heat"], cal=cal)
    c1.set_title(f"(e) $\\delta$ after calibration: held-out $\\rho$ = {rho_a:.2f}",
                 loc="left", pad=5, fontsize=7.6)
    c2.set_title(f"(f) $\\delta$ of the unchanged source model: $\\rho$ = {rho_s:.2f}",
                 loc="left", pad=5, fontsize=7.6)

    tr = d["trace"]
    k = np.arange(1, len(tr) + 1)
    e1, e2 = fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])
    e1.plot(k, tr[:, 1], color=PRED, lw=1.0, label="$a$, drive scale")
    e1.plot(k, tr[:, 2], color=SRC, lw=1.0, ls=(0, (4, 2)), label="$b$, rate scale")
    e1.axhline(1.0, color="0.7", lw=0.5)
    e1.set_ylabel("scaling factor (-)")
    e1.set_xlabel("calibration iteration")
    e1.legend(loc="upper right", fontsize=7)
    e12 = e1.twinx()
    e12.plot(k, tr[:, 0], color="0.35", lw=1.0, ls=(0, (1.4, 1.4)))
    e12.set_ylabel(r"$\delta_0$ (-)", color="0.35")
    e12.tick_params(axis="y", colors="0.35", width=0.6, length=2.5)
    e12.spines["right"].set_color("0.35")
    e1.set_title("(g) calibrated variables, eq. (8)", loc="left", pad=5, fontsize=7.6)
    e2.semilogy(k, tr[:, 3], color=MEAS, lw=1.0)
    e2.set_ylabel("calibration objective (-)")
    e2.set_xlabel("calibration iteration")
    e2.set_title("(h) objective on the calibration interval", loc="left", pad=5, fontsize=7.6)

    h = [plt.Line2D([], [], color=MEAS, lw=0.9), plt.Line2D([], [], color=PRED, lw=0.9, ls=(0, (3.5, 1.8))),
         plt.Line2D([], [], color=REF, lw=0.9, ls=(0, (4, 2))), mpl.patches.Patch(color="0.88")]
    fig.legend(h, ["finite-volume simulation of the target run", "model prediction",
                   r"withheld thermal reference $\Delta T_{\mathrm{ring}}$", "calibration interval"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), columnspacing=1.6)
    fig.subplots_adjust(left=0.09, right=0.91, top=0.965, bottom=0.07)
    fig.savefig(FIG / "adaptation.png", bbox_inches="tight")
    fig.savefig(FIG / "adaptation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    fig_identification()
    fig_adaptation()
    print("figures written to", FIG)
