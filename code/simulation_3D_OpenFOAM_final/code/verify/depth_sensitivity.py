#!/usr/bin/env python3
"""Source-depth sensitivity of S0 (plan section 4): exponential depth
0.25 / 0.5 / 1.0 / 1.5 mm, liquid k x1, mesh L2, dt 0.05 s.

Documented study without pass criteria (check id DS). Time means over
t in [2, 20] s unless stated:
  top melt area, length, width, depth (liquidus); peak and ROI-mean top
  temperature; top area above 3000 K; radiated fraction of the absorbed energy
  over 0-40 s; delta_H at 20 s; maximum bottom temperature under the torch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vlib import FIG_DIR, RESULTS_DIR, VER_DIR, record, signals

CASES = {0.25: "S0_k1_L2_depth0p25mm", 0.5: "S0_k1_L2_dt0p05",
         1.0: "S0_k1_L2_depth1p0mm", 1.5: "S0_k1_L2_depth1p5mm"}
MEANS = [("top_area_liq_mm2", "top melt area (mm2)"), ("top_length_liq_mm", "length (mm)"),
         ("top_width_liq_mm", "width (mm)"), ("depth_liq_mm", "depth (mm)"),
         ("T_top_max_K", "peak top T (K)"), ("T_top_roi_mean_K", "ROI-mean top T (K)"),
         ("top_area_above_3000K_mm2", "top area > 3000 K (mm2)")]


def main():
    rows, table = [], []
    present = {d: c for d, c in CASES.items() if (RESULTS_DIR / c / "signals_0p1s.csv").exists()}
    for d, c in present.items():
        s = signals(c)
        w = (s["t_s"] >= 2.0) & (s["t_s"] <= 20.0)
        rec = {"depth_mm": d, "case": c}
        for col, _ in MEANS:
            rec[col] = float(s.loc[w, col].mean()) if col in s else np.nan
        qr = s["Q_rad_W"].to_numpy() if "Q_rad_W" in s else np.full(len(s), np.nan)
        rec["E_rad_over_E_src"] = float(np.sum(0.5 * (qr[1:] + qr[:-1]) * np.diff(s["t_s"])) / s["E_src_J"].iloc[-1])
        rec["delta_H_20s_K"] = float(np.interp(20.0, s["t_s"], s["delta_H_K"]))
        rec["T_bottom_under_torch_max_K"] = float(s["T_bottom_under_torch_K"].max())
        table.append(rec)
        for k, v in rec.items():
            if k not in ("depth_mm", "case"):
                rows.append(dict(check="DS", case_id=c, metric=k, value=v, criterion="info",
                                 note=f"source depth {d:g} mm; time mean over 2-20 s unless named"))
    record(rows)
    tab = pd.DataFrame(table)
    tab.to_csv(VER_DIR / "DS_source_depth_table.csv", index=False, float_format="%.4g")
    print(tab.to_string(index=False))

    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for d, c in present.items():
        s = signals(c)
        for a, col, lab in zip(ax.flat, ["top_area_liq_mm2", "depth_liq_mm", "top_width_liq_mm",
                                         "T_top_max_K", "top_area_above_3000K_mm2", "delta_H_K"],
                               ["top melt area (mm2)", "depth (mm)", "width (mm)", "peak top T (K)",
                                "top area > 3000 K (mm2)", "delta_H (K)"]):
            if col in s:
                a.plot(s["t_s"], s[col], lw=1, label=f"depth {d:g} mm")
            a.set(xlabel="t (s)", ylabel=lab)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("S0 source-depth sensitivity (liquid k x1, mesh L2, dt 0.05 s; liquidus 1800 K)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "DS_source_depth.png", dpi=150)


if __name__ == "__main__":
    main()
