#!/usr/bin/env python3
"""Choosing arc currents without running the solver.

The exact constant-property, loss-free moving-source solution of the benchmark
(mp3d.analytic) is calibrated for each material against one finished run: a
single threshold temperature rise is found so that the analytic top-surface pool
area matches the measured one late in that run, and a depth factor matches the
measured melt depth. The calibrated model then gives, for any travel speed, the
current that produces a wanted pool area, and the melt depth that comes with it.

Calibration references: NI_A_free_const (steel, 190 A at 2.5 mm/s) and
NI304_A_free_const (AISI 304L, 161.5 A at 2.5 mm/s). For steel the calibrated
model reproduced the whole build-up of NI_A within 1.4 mm² and its depth within
0.05 mm, and predicted SL_1's depth within 12 %.

    python source_estimate.py            # calibrate and print the currents of the paper plan
"""
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from mp3d.analytic import MovingSourcePlate, Segment  # noqa: E402

RESULTS = HERE.parent / "results"
ETA, VOLT = 0.70, 14.0
LX, LY, H = 0.2, 0.1, 0.006
SIG, DEP = 2.6e-3, 0.5e-3
X0, Y0 = 0.010, 0.050
DX = 0.25e-3
XR = np.arange(-26e-3, 8e-3, DX)              # window around the arc
YR = np.arange(-9e-3, 9e-3 + 1e-9, DX)
XRZ = np.arange(-8e-3, 1e-3, DX)
ZS = -np.arange(0.0, 6.0e-3 + 1e-9, 0.1e-3)

MATERIALS = {
    "steel": dict(rho=7400.0, cp=700.0, k=32.0, ref="NI_A_free_const", ref_I=190.0, ref_v=2.5e-3, ref_t=52.0),
    "ss304": dict(rho=7582.0, cp=604.0, k=24.0, ref="NI304_A_free_const", ref_I=161.5, ref_v=2.5e-3, ref_t=52.0),
}


def measured(case, t0, t1=None):
    """Mean pool area and melt depth of a finished run over [t0, t1)."""
    s = pd.read_csv(RESULTS / case / "signals_0p1s.csv")
    t1 = t1 if t1 is not None else t0 + 8.0
    m = s[(s["t_s"] >= t0) & (s["t_s"] < t1)]
    return float(m["top_area_liq_mm2"].mean()), float(m["depth_liq_mm"].mean())


def _plate(mat, segs):
    p = MATERIALS[mat]
    return MovingSourcePlate(Lx=LX, Ly=LY, H=H, sigma=SIG, depth=DEP, rho=p["rho"], cp=p["cp"], k=p["k"],
                             segments=segs)


def field(args):
    """Top-surface grid and vertical section for 1 W in one segment, at time t."""
    mat, v, t0, t1, t = args
    seg = Segment(t0=t0, t1=t1, Q=1.0, x0=X0 + v * t0, x1=X0 + v * t1, y=Y0)
    p = _plate(mat, [seg])
    xa = X0 + v * t
    Xg, Yg = np.meshgrid(xa + XR, Y0 + YR)
    top = p.temperature_rise(np.c_[Xg.ravel(), Yg.ravel(), np.zeros(Xg.size)], t).reshape(Xg.shape)
    Xz, Zz = np.meshgrid(xa + XRZ, ZS)
    sec = p.temperature_rise(np.c_[Xz.ravel(), np.full(Xz.size, Y0), Zz.ravel()], t).reshape(Xz.shape)
    return top, sec


def area_depth(top, sec, thr):
    area = (top > thr).sum() * (DX * 1e3) ** 2
    hot = (sec > thr).any(axis=1)
    return area, (-ZS[hot].min() * 1e3 if hot.any() else 0.0)


def calibrate(mat, pool):
    p = MATERIALS[mat]
    area_ref, depth_ref = measured(p["ref"], p["ref_t"])
    top, sec = pool.map(field, [(mat, p["ref_v"], 0.0, 60.0, p["ref_t"])])[0]
    Q = ETA * VOLT * p["ref_I"]
    lo, hi = 100.0, 6000.0
    for _ in range(60):
        thr = 0.5 * (lo + hi)
        lo, hi = (thr, hi) if area_depth(Q * top, Q * sec, thr)[0] > area_ref else (lo, thr)
    d_an = area_depth(Q * top, Q * sec, thr)[1]
    return dict(threshold_K=thr, depth_factor=depth_ref / d_an, area_ref=area_ref, depth_ref=depth_ref)


def current_for(mat, v, duration, targets, cal, pool):
    """Currents giving the target pool areas at 80 % of the run, and their melt depths."""
    t = 0.8 * duration
    top, sec = pool.map(field, [(mat, v, 0.0, duration, t)])[0]
    out = []
    for target in targets:
        lo, hi = 30.0, 320.0
        for _ in range(40):
            I = 0.5 * (lo + hi)
            a = area_depth(ETA * VOLT * I * top, ETA * VOLT * I * sec, cal["threshold_K"])[0]
            lo, hi = (I, hi) if a < target else (lo, I)
        a, d = area_depth(ETA * VOLT * I * top, ETA * VOLT * I * sec, cal["threshold_K"])
        out.append(dict(target_area=target, current_A=round(I, 1), area_mm2=round(a, 1),
                        depth_mm=round(d * cal["depth_factor"], 2)))
    return out


def main():
    plan = {}
    with Pool(12) as pool:
        for mat in MATERIALS:
            cal = calibrate(mat, pool)
            print(f"{mat}: threshold {cal['threshold_K']:.0f} K, depth factor {cal['depth_factor']:.2f} "
                  f"(reference {cal['area_ref']:.1f} mm², {cal['depth_ref']:.2f} mm)", flush=True)
            plan[mat] = dict(calibration=cal, levels={})
            for name, v, dur, targets in (("normal", 1.5e-3, 80.0, [60.0]),
                                          ("intermediate", 1.0e-3, 120.0, [60.0]),
                                          ("extreme", 0.5e-3, 240.0, [60.0, 32.0])):
                rows = current_for(mat, v, dur, targets, cal, pool)
                plan[mat]["levels"][name] = dict(speed_mm_s=v * 1e3, duration_s=dur, rows=rows)
                for r in rows:
                    print(f"  {name:12s} {v * 1e3:.1f} mm/s  {r['current_A']:6.1f} A  -> "
                          f"{r['area_mm2']:5.1f} mm², melt {r['depth_mm']:.2f} mm", flush=True)
    (HERE / "results" / "source_estimate.json").write_text(json.dumps(plan, indent=1) + "\n")


if __name__ == "__main__":
    main()
