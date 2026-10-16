#!/usr/bin/env python3
"""Checks on a finished S0-type case: V2, V3, energy closure, solver health
and outputs (plan section 5).

  V3  source normalisation: logged absorbed power vs eta*V*I while the torch
      is on (the torch stays >= 10 mm from every edge).
  V2  zero-input cooling over the cooldown (t > t_heat_end): enthalpy and the
      maximum temperature decrease monotonically, no undershoot below T_inf,
      and the enthalpy decrease matches the boundary loss.
  EC  energy closure over the whole run: H - H(0) - (E_src - E_loss).
  SH  solver health: every step's temperature/phase loop converged; no
      momentum or pressure equation solved (thermal-only).
  OUT outputs: complete 0.1 s grid; AdditiveFOAM's meltPoolDimensions vs the
      independent post-processing recomputation of length, width and depth.

Criteria (fixed before comparison):
  V3   max |P_logged / (eta V I) - 1| <= 1 %
  V2   dH/dt <= 0 and dTmax/dt <= 1e-6 K/s for every cooldown interval,
       T_min >= T_inf - 0.01 K, cooldown closure drift <= 0.2 % of E_src
  EC   |residual| <= 2 % of E_src at every time (AdditiveFOAM lags cp by one
       step, so the enthalpy closure is first order in dt; set after the smoke
       test showed ~1 % at 2 s; the dt study must show it shrinking)
  SH   100 % of steps converged; 0 U/p_rgh solves
  OUT  401-point 0.1 s grid without gaps or NaN; time-mean |relative
       difference| over t in [1, t_heat_end] <= 3 % for L, W and D (liquidus)
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from vlib import RESULTS_DIR, case_config, record, signals


def check(case_id):
    cfg = case_config(case_id)
    s = signals(case_id)
    st = pd.read_csv(RESULTS_DIR / case_id / "solver_steps.csv")
    from mp3d.schedule import ProcessSchedule
    t_heat = float(ProcessSchedule.from_config(cfg).t_heat_end)
    T_inf = float(cfg["boundary"]["T_inf"])
    rows = []

    on = st["P_abs_nominal_W"] > 0
    dev = float(np.max(np.abs(st.loc[on, "P_rel_dev"])))
    rows.append(dict(check="V3", case_id=case_id, metric="max_rel_dev_absorbed_power", value=dev,
                     criterion="<= 0.01", **{"pass": bool(dev <= 0.01)},
                     note=f"{int(on.sum())} steps with the torch on"))
    off = st["P_abs_nominal_W"] == 0
    rows.append(dict(check="V3", case_id=case_id, metric="max_power_after_switch_off_W",
                     value=float(st.loc[off, "absorbed_power"].abs().max()) if off.any() else 0.0,
                     criterion="== 0", **{"pass": bool((st.loc[off, "absorbed_power"] == 0).all())}))

    c = s[s["t_s"] >= t_heat - 1e-9]
    dH = np.diff(c["H_J"].to_numpy())
    dTm = np.diff(c["T_max_3d_K"].to_numpy()) / np.diff(c["t_s"].to_numpy())
    Tmin = float(s["T_min_3d_K"].min()) if "T_min_3d_K" in s else float("nan")
    drift = float((c["residual_J"].max() - c["residual_J"].min()) / s["E_src_J"].iloc[-1])
    rows += [
        dict(check="V2", case_id=case_id, metric="max_dH_per_interval_cooldown_J", value=float(dH.max()),
             criterion="<= 0", **{"pass": bool(dH.max() <= 0)}),
        dict(check="V2", case_id=case_id, metric="max_dTmax_dt_cooldown_K_s", value=float(dTm.max()),
             criterion="<= 1e-6", **{"pass": bool(dTm.max() <= 1e-6)}),
        dict(check="V2", case_id=case_id, metric="T_min_minus_T_inf_K", value=Tmin - T_inf,
             criterion=">= -0.01", **{"pass": bool(Tmin - T_inf >= -0.01)}),
        dict(check="V2", case_id=case_id, metric="closure_drift_cooldown_rel", value=drift,
             criterion="<= 0.002", **{"pass": bool(abs(drift) <= 0.002)}),
    ]

    res = float(np.max(np.abs(s["residual_J"] / s["E_src_J"].iloc[-1])))
    rows += [
        dict(check="EC", case_id=case_id, metric="max_abs_residual_rel_Esrc", value=res,
             criterion="<= 0.02", **{"pass": bool(res <= 0.02)}),
        dict(check="EC", case_id=case_id, metric="residual_rel_at_t_heat_end",
             value=float(np.interp(t_heat, s["t_s"], s["residual_J"]) / s["E_src_J"].iloc[-1]),
             criterion="info"),
        dict(check="EC", case_id=case_id, metric="loss_fraction_of_Esrc_at_end",
             value=float(s["E_loss_J"].iloc[-1] / s["E_src_J"].iloc[-1]), criterion="info"),
    ]

    conv = float(st["thermo_converged"].mean())
    log = (RESULTS_DIR / case_id / "logs" / "log.additiveFoam").read_text(errors="replace")
    n_flow = len(re.findall(r"Solving for (Ux|Uy|Uz|p_rgh)", log))
    rows += [
        dict(check="SH", case_id=case_id, metric="fraction_steps_thermo_converged", value=conv,
             criterion="== 1", **{"pass": bool(conv == 1.0)},
             note=f"max iterations {int(st['n_thermo'].max())}, mean {st['n_thermo'].mean():.2f}"),
        dict(check="SH", case_id=case_id, metric="n_flow_equation_solves", value=float(n_flow),
             criterion="== 0", **{"pass": bool(n_flow == 0)}),
        dict(check="SH", case_id=case_id, metric="max_T_final_residual", value=float(st["T_final_residual"].max()),
             criterion="info"),
        dict(check="SH", case_id=case_id, metric="solver_wall_time_s", value=float(st["clock_time"].iloc[-1]),
             criterion="info"),
    ]

    t = s["t_s"].to_numpy()
    expect = np.round(np.arange(0.0, float(cfg["time"]["t_end"]) + 1e-9, 0.1), 6)
    grid_ok = len(t) == len(expect) and np.allclose(t, expect, atol=1e-6)
    key = ["T_top_max_K", "top_area_liq_mm2", "depth_liq_mm", "H_J", "delta_H_K"]
    nan_ok = not s[key].isna().any().any()
    rows.append(dict(check="OUT", case_id=case_id, metric="complete_0p1s_grid_no_nan",
                     value=float(grid_ok and nan_ok), criterion="== 1",
                     **{"pass": bool(grid_ok and nan_ok)}, note=f"{len(t)} snapshots"))
    mpd_file = RESULTS_DIR / case_id / "meltPoolDimensions" / f"{cfg_liquidus(cfg):g}.csv"
    if mpd_file.exists():
        m = pd.read_csv(mpd_file)
        m.columns = ["t", "L", "W", "D"]
        win = (s["t_s"] >= 1.0) & (s["t_s"] <= t_heat)
        ts = s.loc[win, "t_s"].to_numpy()
        for name, col, mc in (("length", "top_length_liq_mm", "L"), ("width", "top_width_liq_mm", "W"),
                              ("depth", "depth_liq_mm", "D")):
            ref = np.interp(ts, m["t"], m[mc] * 1e3)
            ours = s.loc[win, col].to_numpy()
            ok = ref > 0.2                      # a pool exists (excludes pauses between passes)
            d = float(np.mean(np.abs(ours[ok] / ref[ok] - 1.0)))
            rows.append(dict(check="OUT", case_id=case_id, metric=f"meltPoolDimensions_vs_post_{name}",
                             value=d, criterion="<= 0.03", **{"pass": bool(d <= 0.03)},
                             note="time-mean |ours/AdditiveFOAM - 1|, liquidus, t in [1, t_heat_end] "
                                  "where AdditiveFOAM reports a pool (> 0.2 mm)"))
    record(rows)


def cfg_liquidus(cfg):
    from mp3d.case_writer import build_material
    return build_material(cfg).liquidus


if __name__ == "__main__":
    for c in sys.argv[1:]:
        check(c)
