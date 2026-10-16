#!/usr/bin/env python3
"""Closure sensitivity of the history effect: P0 - S0 over the evaluation
segment with liquid k x5 (dataset) and x1 (sensitivity).

For each closure, both runs are compared at the same schedule time
(P0 at t + 37 s vs S0 at t, 0 <= t <= 20 s; identical inputs):
- the mean and the values at t = 2, 10 and 20 s of the differences in the
  surface coordinates and the hidden heat signals;
- the fraction of the segment in which the P0 pool is larger.

The history effect is closure-independent if the sign and order of magnitude
of the differences agree between x5 and x1.

    python code/analysis/closure_sensitivity.py [C3r|C3]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mp3d import PROJECT_DIR, RESULTS_DIR  # noqa: E402

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "C3r"
PAIRS = {"k x5 (dataset)": (f"{PREFIX}_S0", f"{PREFIX}_P0"), "k x1 (sensitivity)": (f"{PREFIX}_S0_k1", f"{PREFIX}_P0_k1")}
COLS = ["top_area_liq_mm2", "top_length_liq_mm", "top_width_liq_mm", "depth_liq_mm", "T_top_max_K",
        "T_top_roi_mean_K", "delta_H_K", "dHK_ring_5_15mm_K", "T_bottom_under_torch_K"]
T_SHIFT = 37.0
OUT = PROJECT_DIR / "analysis" / PREFIX


def aligned(s0_case, p0_case):
    t = np.round(np.arange(0.0, 20.0 + 1e-9, 0.1), 6)
    s0 = pd.read_csv(RESULTS_DIR / s0_case / "signals_0p1s.csv")
    p0 = pd.read_csv(RESULTS_DIR / p0_case / "signals_0p1s.csv")
    a = s0.set_index(np.round(s0["t_s"], 6)).loc[t]
    b = p0.set_index(np.round(p0["t_s"] - T_SHIFT, 6)).loc[t]
    return t, a, b


def main():
    OUT.mkdir(exist_ok=True)
    rows, curves = [], {}
    for lab, (sc, pc) in PAIRS.items():
        if not all((RESULTS_DIR / c / "signals_0p1s.csv").exists() for c in (sc, pc)):
            print(f"skip {lab}: results missing")
            continue
        t, a, b = aligned(sc, pc)
        curves[lab] = (t, a, b)
        for col in COLS:
            d = b[col].to_numpy() - a[col].to_numpy()
            rows.append(dict(closure=lab, quantity=col, mean_diff=d.mean(), diff_t2=d[20], diff_t10=d[100],
                             diff_t20=d[-1], frac_P0_larger=float(np.mean(d > 0))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "closure_sensitivity_P0_minus_S0.csv", index=False, float_format="%.4g")
    print(df.pivot(index="quantity", columns="closure", values="mean_diff").round(3).to_string())
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, (lab, (t, a, b)) in enumerate(curves.items()):
        for axi, col, yl in zip(ax, ["top_area_liq_mm2", "depth_liq_mm", "dHK_ring_5_15mm_K"],
                                ["top melt area (mm2)", "depth (mm)", "ring 5-15 mm (K)"]):
            axi.plot(t, a[col], "-", color=f"C{i}", lw=1, label=f"S0 {lab}")
            axi.plot(t, b[col], "--", color=f"C{i}", lw=1, label=f"P0 {lab}")
            axi.set(xlabel="time into the S0 schedule (s)", ylabel=yl)
    ax[0].legend(fontsize=7)
    fig.suptitle("History pair: identical inputs, P0 preheated by a previous pass (dashed) vs S0 (solid)")
    fig.tight_layout()
    fig.savefig(OUT / "closure_sensitivity.png", dpi=150)


if __name__ == "__main__":
    main()
