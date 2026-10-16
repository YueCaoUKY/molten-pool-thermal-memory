#!/usr/bin/env python3
"""Check an added record: the identical-input windows, then the identified models.

The four windows are the last 10 s of each 60 s block, where every input sits exactly at the
block's nominal value. Windows 1 and 3 share one nominal, windows 2 and 4 the other, so each pair
compares the same input at different accumulated heat -- evidence that needs no model.

    python check_new.py [case ...]           # windows only
    python check_new.py --fits [case ...]    # windows and the identified models
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import identify_3d as I
from delta_variants import window_for, ring_heat

HERE = Path(__file__).resolve().parent
W4 = [(50.0, 60.0), (110.0, 120.0), (170.0, 180.0), (230.0, 240.0)]
W6 = [(30.0, 40.0), (70.0, 80.0), (110.0, 120.0), (150.0, 160.0),
      (190.0, 200.0), (230.0, 240.0)]


def windows_for(case):
    """The constant-input windows: six 40 s blocks in the v2 records, four 60 s otherwise."""
    return W6 if case.endswith('_v2') else W4


def windows(case):
    f = pd.read_csv(HERE / "features" / f"{case}.csv")
    s = pd.read_csv(HERE.parent / "results" / case / "signals_0p1s.csv")
    s.index = np.round(s["t_s"], 3)
    t = f["t_s"].to_numpy()
    heat = s.reindex(np.round(t, 3))["dHK_ring_5_15mm_K"].to_numpy()
    rows = []
    for a, b in windows_for(case):
        m = (t >= a - 1e-9) & (t <= b + 1e-9)
        rows.append(dict(t=f"{a:.0f}-{b:.0f}", area=float(np.nanmean(f["area_mm2"].to_numpy()[m])),
                         heat=float(np.nanmean(heat[m])),
                         I=float(np.nanmean(s.reindex(np.round(t, 3))["I_A"].to_numpy()[m])),
                         v=float(np.nanmean(s.reindex(np.round(t, 3))["u_mm_s"].to_numpy()[m]))))
    return rows


def report(case, with_fits=False):
    print(f"\n=== {case} ===")
    r = windows(case)
    print(f"{'窗口 s':>10s} {'电流 A':>8s} {'速度':>7s} {'热量 K':>8s} {'面积 mm2':>9s}")
    for x in r:
        print(f"{x['t']:>10s} {x['I']:8.1f} {x['v']:7.3f} {x['heat']:8.0f} {x['area']:9.1f}")
    n = len(r)
    for first, name in ((0, "大熔池"), (1, "小熔池")):      # first and last visit to the same nominal
        a, b = r[first], r[first + 2 * ((n - 1 - first) // 2)]
        print(f"   {name}：同一输入 {a['I']:.1f} A / {b['v']:.3f} mm/s，"
              f"热量 {a['heat']:.0f} -> {b['heat']:.0f} K，"
              f"面积 {a['area']:.1f} -> {b['area']:.1f} mm² ({100*(b['area']-a['area'])/a['area']:+.1f}%)")
    if not with_fits:
        return
    try:
        m = json.loads((HERE / "results" / f"{case}_v_pool_simpleD_p60" / "metrics.json").read_text())
        mn = json.loads((HERE / "results" / f"{case}_nodelta_p60" / "metrics.json").read_text())
    except FileNotFoundError:
        print("   （尚未拟合）")
        return
    d = np.load(HERE / "results" / f"{case}_v_pool_simpleD_p60" / "fit.npz", allow_pickle=True)
    n = int(d["n_train"]); zt = d["z_true"]
    lo, hi = zt[:n].min(axis=0), zt[:n].max(axis=0)
    out = ((zt[n:] < lo) | (zt[n:] > hi)).mean() * 100
    print(f"   留出段 R²  带 δ {m['r2_global_heldout']:6.3f}   无 δ {mn['r2_global_heldout']:6.3f}"
          f"   差 {m['r2_global_heldout']-mn['r2_global_heldout']:5.3f}")
    print(f"   留出段 RMSE 带 δ {m['nrmse_std_heldout']:6.3f}   无 δ {mn['nrmse_std_heldout']:6.3f}")
    print(f"   ρ(δ, ΔT_w) 留出段 {m['delta_spearman_heldout']:.3f}   全段 {m['delta_spearman']:.3f}")
    print(f"   留出段落在训练范围外 {out:.0f}%   （条件 4 是 38%，steel 1 是 81%）")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--fits"]
    fits = "--fits" in sys.argv
    for c in args or [c for c in (HERE / "results" / "new_cases.txt").read_text().split() if c]:
        try:
            report(c, fits)
        except FileNotFoundError as e:
            print(f"\n=== {c} ===  尚未完成 ({Path(str(e).split(': ')[-1]).name})")
