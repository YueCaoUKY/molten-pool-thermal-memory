#!/usr/bin/env python3
"""V7 mesh and time-step convergence of the canonical S0 case (plan section 4).

Pairs compared (coarse vs next finer):
  mesh  S0_k1_L1_dt0p05 vs S0_k1_L2_dt0p05   (h 0.50 -> 0.35 mm)
        S0_k1_L2_dt0p05 vs S0_k1_L3_dt0p05   (h 0.35 -> 0.25 mm)
  time  S0_k1_L2_dt0p05 vs S0_k1_L2_dt0p025  (dt 0.05 -> 0.025 s)

Metrics and criteria (plan section 4; fixed before comparison):
  pool    time-mean over t in [2, 20] s of top-surface length and width and
          depth (liquidus): |coarse/fine - 1| < 5 %
  history temperature histories at the 48 fixed probes (x = 20/40/60 mm,
          y offsets 0/2.5/5/10 mm, z = 0/-1/-3/-6 mm) over 0-40 s:
          RMSE(coarse - fine) / max_t(T_fine - T_inf) < 3 % at every probe
The accepted level is the coarsest one whose comparison with the next finer
calculation meets both criteria. Peak surface temperature, delta_H, the energy
residual, wall time and peak memory are reported without criteria.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vlib import FIG_DIR, PROJECT_DIR, RESULTS_DIR, record, signals

PAIRS = [("mesh", "S0_k1_L1_dt0p05", "S0_k1_L2_dt0p05"),
         ("mesh", "S0_k1_L2_dt0p05", "S0_k1_L3_dt0p05"),
         ("time", "S0_k1_L2_dt0p05", "S0_k1_L2_dt0p025")]
POOL = [("length", "top_length_liq_mm"), ("width", "top_width_liq_mm"), ("depth", "depth_liq_mm")]


def probes(case):
    d = np.load(RESULTS_DIR / case / "probes_0p1s.npz")
    return d["t"], d["xyz"], d["T"]


def peak_memory_GB(case):
    """Peak total RSS of additiveFoam processes during the run (logs/memory_additiveFoam.csv)."""
    f = PROJECT_DIR / "logs" / "memory_additiveFoam.csv"
    man = json.loads((RESULTS_DIR / case / "run_manifest.json").read_text())
    if not f.exists() or "timings_s" not in man:
        return float("nan")
    mem = pd.read_csv(f)
    end = pd.Timestamp(man["collected"]).timestamp()
    t_solve = man["timings_s"].get("additiveFoam", 0.0) + man["timings_s"].get("reconstructPar", 0.0) + 60
    sel = (mem["unix_time"] >= end - t_solve) & (mem["unix_time"] <= end)
    return float(mem.loc[sel, "rss_kB_total"].max() / 1024 ** 2) if sel.any() else float("nan")


def compare(kind, coarse, fine):
    sc, sf = signals(coarse), signals(fine)
    win_c = (sc["t_s"] >= 2.0) & (sc["t_s"] <= 20.0)
    win_f = (sf["t_s"] >= 2.0) & (sf["t_s"] <= 20.0)
    rows, ok = [], True
    for name, col in POOL:
        d = float(sc.loc[win_c, col].mean() / sf.loc[win_f, col].mean() - 1.0)
        ok &= abs(d) < 0.05
        rows.append(dict(check="V7", case_id=f"{coarse}|{fine}", metric=f"{kind}_pool_{name}_rel_diff",
                         value=d, criterion="|x| < 0.05", **{"pass": bool(abs(d) < 0.05)},
                         note=f"time-mean t in [2, 20] s: {sc.loc[win_c, col].mean():.3f} vs {sf.loc[win_f, col].mean():.3f} mm"))
    tc, xyz, Tc = probes(coarse)
    tf, _, Tf = probes(fine)
    Tf_on_c = np.array([np.interp(tc, tf, Tf[:, p]) for p in range(Tf.shape[1])]).T
    T_inf = 293.15
    rmse = np.sqrt(np.mean((Tc - Tf_on_c) ** 2, axis=0)) / np.max(Tf_on_c - T_inf, axis=0)
    worst = int(np.argmax(rmse))
    ok &= bool(rmse.max() < 0.03)
    rows.append(dict(check="V7", case_id=f"{coarse}|{fine}", metric=f"{kind}_probe_rmse_rel_max",
                     value=float(rmse.max()), criterion="< 0.03", **{"pass": bool(rmse.max() < 0.03)},
                     note=f"worst probe x,y,z = {tuple(np.round(xyz[worst] * 1e3, 2))} mm; "
                          f"median {np.median(rmse):.4f}"))
    for col in ("T_top_max_K", "delta_H_K", "top_area_liq_mm2"):
        d = float(sc.loc[win_c, col].mean() / sf.loc[win_f, col].mean() - 1.0)
        rows.append(dict(check="V7", case_id=f"{coarse}|{fine}", metric=f"{kind}_{col}_rel_diff_info",
                         value=d, criterion="info"))
    rows.append(dict(check="V7", case_id=f"{coarse}|{fine}", metric=f"{kind}_pair_meets_criteria",
                     value=float(ok), criterion="== 1", **{"pass": bool(ok)}))
    return rows, rmse


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, c, f in PAIRS:
        if not ((RESULTS_DIR / c / "signals_0p1s.csv").exists() and (RESULTS_DIR / f / "signals_0p1s.csv").exists()):
            print(f"skip {c} vs {f}: results missing")
            continue
        r, _ = compare(kind, c, f)
        rows += r
    for case in sorted({p for _, a, b in PAIRS for p in (a, b)}):
        if (RESULTS_DIR / case / "run_manifest.json").exists():
            man = json.loads((RESULTS_DIR / case / "run_manifest.json").read_text())
            meta = json.loads((RESULTS_DIR / case / "mesh_meta.json").read_text())
            rows.append(dict(check="V7", case_id=case, metric="n_cells", value=float(meta["n_cells"]), criterion="info"))
            rows.append(dict(check="V7", case_id=case, metric="solver_wall_time_s",
                             value=float(man["timings_s"].get("additiveFoam", np.nan)), criterion="info",
                             note=f"{man['nprocs']} MPI ranks"))
            rows.append(dict(check="V7", case_id=case, metric="peak_memory_GB", value=peak_memory_GB(case),
                             criterion="info", note="sum of additiveFoam RSS, sampled every 5 s"))
    record(rows)

    cases = [c for c in ("S0_k1_L1_dt0p05", "S0_k1_L2_dt0p05", "S0_k1_L3_dt0p05", "S0_k1_L2_dt0p025")
             if (RESULTS_DIR / c / "signals_0p1s.csv").exists()]
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
    for c in cases:
        s = signals(c)
        for a, col, lab in zip(ax.flat, ["top_length_liq_mm", "top_width_liq_mm", "depth_liq_mm",
                                         "T_top_max_K", "delta_H_K", "residual_rel"],
                               ["top length (mm)", "top width (mm)", "depth (mm)",
                                "peak top T (K)", "delta_H (K)", "energy residual / E_src"]):
            a.plot(s["t_s"], s[col], lw=1, label=c.replace("S0_k1_", ""))
            a.set(xlabel="t (s)", ylabel=lab)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("S0 (liquid k x1): mesh L1/L2/L3 (h = 0.5/0.35/0.25 mm) and dt 0.05/0.025 s; liquidus 1800 K")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "V7_convergence.png", dpi=150)


if __name__ == "__main__":
    main()
