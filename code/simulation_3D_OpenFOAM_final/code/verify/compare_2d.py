#!/usr/bin/env python3
"""V8: trends and energy of the 3D S0 cases vs the 2D FEniCSx pool_steel_lit run.

The comparison is not an equality test. The 2D model averages temperature over
the 6 mm thickness, deposits the arc power through the whole thickness, loses
heat only through top and bottom (as a volumetric sink), and uses liquid k x5.
The 3D cases resolve depth and lose heat on all six surfaces; S0_k5 uses the
same liquid-k closure as 2D, S0_k1 is the physical baseline. Like-for-like
quantities computed from the 3D field:
  2D peak_T (thickness-averaged max)       <-> 3D Tbar_max_K
  2D pool_area (thickness-averaged T > Tm) <-> 3D Tbar_area_Tm_mm2
  2D delta (volume-mean T rise)            <-> 3D mean_T_rise_K
Physical 3D quantities with no 2D counterpart: top-surface melt area, length,
width, depth, peak surface temperature, delta_H, and the 20 s cooldown.
Absorbed energy must agree (criterion |dE|/E <= 0.5 % at 20 s).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vlib import FIG_DIR, PROJECT_DIR, VER_DIR, record, signals

D2 = PROJECT_DIR.parent / "simulation_2D_FEM" / "Selected Sims" / "pool_steel_lit"
TIMES = [4.0, 8.0, 12.0, 16.0, 20.0]


def load_2d():
    ts = pd.read_csv(D2 / "timeseries.csv")
    mo = pd.read_csv(D2 / "pool_morphology_0.1s.csv")
    ts["E_abs_J"] = np.concatenate([[0.0], np.cumsum(0.8 * 14.0 * ts["I"].to_numpy()[1:] * np.diff(ts["t"]))])
    return ts, mo


def main(cases):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ts, mo = load_2d()
    table, rows = [], []
    S = {c: signals(c) for c in cases}
    for c, s in S.items():
        e3 = float(np.interp(20.0, s["t_s"], s["E_src_J"]))
        e2 = float(np.interp(20.0, ts["t"], ts["E_abs_J"]))
        rows.append(dict(check="V8", case_id=c, metric="absorbed_energy_20s_rel_diff_vs_2D",
                         value=e3 / e2 - 1.0, criterion="|x| <= 0.005",
                         **{"pass": bool(abs(e3 / e2 - 1.0) <= 0.005)},
                         note=f"3D {e3:.1f} J vs 2D {e2:.1f} J"))
        for t in TIMES:
            g = lambda col: float(np.interp(t, s["t_s"], s[col]))  # noqa: E731
            table.append({
                "case": c, "t_s": t,
                "2D_peak_T_K": float(np.interp(t, ts["t"], ts["peak_T"])),
                "3D_Tbar_max_K": g("Tbar_max_K"), "3D_T_top_max_K": g("T_top_max_K"),
                "2D_pool_area_mm2": float(np.interp(t, ts["t"], ts["pool_area"]) * 1e6),
                "3D_Tbar_area_Tm_mm2": g("Tbar_area_Tm_mm2"), "3D_top_area_liq_mm2": g("top_area_liq_mm2"),
                "2D_length_mm": float(np.interp(t, mo["t"], mo["length_mm"])),
                "3D_top_length_liq_mm": g("top_length_liq_mm"),
                "2D_width_mm": float(np.interp(t, mo["t"], mo["width_mm"])),
                "3D_top_width_liq_mm": g("top_width_liq_mm"),
                "3D_depth_liq_mm": g("depth_liq_mm"),
                "2D_delta_K": float(np.interp(t, ts["t"], ts["delta"])),
                "3D_mean_T_rise_K": g("mean_T_rise_K"), "3D_delta_H_K": g("delta_H_K"),
            })
    record(rows)
    tab = pd.DataFrame(table)
    tab.to_csv(VER_DIR / "V8_2D_vs_3D_table.csv", index=False, float_format="%.4g")
    print(tab.to_string(index=False))

    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    ax = ax.flat
    ax[0].plot(ts["t"], ts["peak_T"], "k-", lw=1.5, label="2D peak_T (thickness avg)")
    ax[1].plot(ts["t"], ts["pool_area"] * 1e6, "k-", lw=1.5, label="2D pool_area (Tbar > Tm)")
    ax[2].plot(mo["t"], mo["length_mm"], "k-", lw=1.5, label="2D length (Tbar > TL)")
    ax[3].plot(mo["t"], mo["width_mm"], "k-", lw=1.5, label="2D width (Tbar > TL)")
    ax[5].plot(ts["t"], ts["delta"], "k-", lw=1.5, label="2D delta (mean T rise)")
    for i, (c, s) in enumerate(S.items()):
        col = f"C{i}"
        tag = c.split("_")[1]
        ax[0].plot(s["t_s"], s["Tbar_max_K"], "-", color=col, lw=1, label=f"3D {tag} Tbar max")
        ax[0].plot(s["t_s"], s["T_top_max_K"], "--", color=col, lw=1, label=f"3D {tag} top max")
        ax[1].plot(s["t_s"], s["Tbar_area_Tm_mm2"], "-", color=col, lw=1, label=f"3D {tag} Tbar > Tm")
        ax[1].plot(s["t_s"], s["top_area_liq_mm2"], "--", color=col, lw=1, label=f"3D {tag} top > TL")
        ax[2].plot(s["t_s"], s["top_length_liq_mm"], "--", color=col, lw=1, label=f"3D {tag} top")
        ax[3].plot(s["t_s"], s["top_width_liq_mm"], "--", color=col, lw=1, label=f"3D {tag} top")
        ax[4].plot(s["t_s"], s["depth_liq_mm"], "-", color=col, lw=1, label=f"3D {tag} depth (TL)")
        ax[5].plot(s["t_s"], s["mean_T_rise_K"], "-", color=col, lw=1, label=f"3D {tag} mean T rise")
        ax[5].plot(s["t_s"], s["delta_H_K"], ":", color=col, lw=1, label=f"3D {tag} delta_H")
    for a, lab in zip(ax, ["T (K)", "area (mm2)", "length (mm)", "width (mm)", "depth (mm)", "K"]):
        a.set(xlabel="t (s)", ylabel=lab)
        a.legend(fontsize=6)
    fig.suptitle("2D pool_steel_lit (0-20 s) vs 3D S0 (0-40 s, torch off at 20 s)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "V8_2D_vs_3D.png", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1:] or ["S0_k1_L2_dt0p05", "S0_k5_L2_dt0p05"])
