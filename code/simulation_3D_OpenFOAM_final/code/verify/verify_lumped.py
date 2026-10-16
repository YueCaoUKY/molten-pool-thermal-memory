#!/usr/bin/env python3
"""V5 boundary audit and V6 phase audit in the uniform-temperature limit.

V5a  per-patch convection only (h = 10, 20, 40, 80, 160, 320 W/m2/K on
     top, bottom, xmin, xmax, ymin, ymax): T - Tinf must decay as
     exp(-sum(A_p h_p) t / (rho c V)).
V5b  canonical h = 15 W/m2/K, emissivity 0.6 on all patches vs the lumped ODE
     rho c V dT/dt = -A [h (T - Tinf) + eps sigma (T^4 - Tinf^4)].
V6   solidification from 1850 K through the 1750-1800 K mushy range with
     h = 200 W/m2/K: the heat lost while crossing the mushy range must equal
     rho V (cp * 50 K + Lf); T(t) vs the lumped enthalpy ODE.

Criteria (fixed before comparison):
  V5a  max |dT_fv / dT_exact - 1| <= 0.5 %; fitted decay rate within 0.2 %
  V5b  max |dT_fv / dT_ode - 1| <= 0.5 %
  V6   latent + sensible heat released across the mushy range within 1 %;
       max |T_fv - T_ode| <= 2 K; energy-closure residual <= 0.5 % of the
       initial enthalpy
  all  plate uniform: max(T) - mean(T) <= 0.1 K
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from vlib import FIG_DIR, case_config, record, signals

SIGMA = 5.67e-8   # value hard-coded in AdditiveFOAM's mixedTemperature


def geometry(cfg):
    g = cfg["geometry"]
    Lx, Ly, H = g["Lx"], g["Ly"], g["H"]
    areas = {"top": Lx * Ly, "bottom": Lx * Ly, "xmin": Ly * H, "xmax": Ly * H,
             "ymin": Lx * H, "ymax": Lx * H}
    return areas, Lx * Ly * H


def patch_param(cfg, p, key):
    b = cfg["boundary"]
    spec = dict(b["default"])
    spec.update((b.get("patches") or {}).get(p, {}))
    return spec[key]


def uniformity(case, s):
    T_mean = s["mean_T_rise_K"] + case_config(case)["boundary"]["T_inf"]
    return float(np.max(np.abs(s["T_max_3d_K"] - T_mean)))


def v5a(case="V5a_boundary_convection"):
    cfg, s = case_config(case), signals(case)
    areas, V = geometry(cfg)
    m = cfg["material"]
    rhoc = m["rho"] * m["cp"]
    lam = sum(areas[p] * patch_param(cfg, p, "h") for p in areas) / (rhoc * V)
    T_inf, T0 = cfg["boundary"]["T_inf"], cfg["initial"]["T"]
    t, dT = s["t_s"].to_numpy(), s["mean_T_rise_K"].to_numpy()
    exact = (T0 - T_inf) * np.exp(-lam * t)
    err = np.max(np.abs(dT / exact - 1.0))
    k_fit = -np.polyfit(t, np.log(dT), 1)[0]
    rows = [
        dict(check="V5", case_id=case, metric="max_rel_err_dT_vs_exponential", value=err,
             criterion="<= 0.005", **{"pass": bool(err <= 0.005)},
             note=f"lambda = {lam:.6e} 1/s from sum(A_p h_p)/(rho c V)"),
        dict(check="V5", case_id=case, metric="decay_rate_rel_err", value=abs(k_fit / lam - 1),
             criterion="<= 0.002", **{"pass": bool(abs(k_fit / lam - 1) <= 0.002)},
             note=f"fitted {k_fit:.6e} vs {lam:.6e} 1/s"),
        dict(check="V5", case_id=case, metric="nonuniformity_K", value=uniformity(case, s),
             criterion="<= 0.1", **{"pass": bool(uniformity(case, s) <= 0.1)}),
        dict(check="V5", case_id=case, metric="energy_residual_rel_H0",
             value=float(np.max(np.abs(s["residual_J"])) / abs(s["H_J"].iloc[0] + 1e-30)),
             criterion="info", note="closure vs initial enthalpy"),
    ]
    record(rows)
    return t, dT, exact


def v5b(case="V5b_boundary_radiation"):
    cfg, s = case_config(case), signals(case)
    areas, V = geometry(cfg)
    A = sum(areas.values())
    m = cfg["material"]
    rhocV = m["rho"] * m["cp"] * V
    h, eps = cfg["boundary"]["default"]["h"], cfg["boundary"]["default"]["emissivity"]
    T_inf, T0 = cfg["boundary"]["T_inf"], cfg["initial"]["T"]
    t = s["t_s"].to_numpy()
    sol = solve_ivp(lambda _t, T: -A * (h * (T - T_inf) + eps * SIGMA * (T ** 4 - T_inf ** 4)) / rhocV,
                    (0, t[-1]), [T0], t_eval=t, rtol=1e-11, atol=1e-9)
    dT_ode = sol.y[0] - T_inf
    dT = s["mean_T_rise_K"].to_numpy()
    err = np.max(np.abs(dT / dT_ode - 1.0))
    record([
        dict(check="V5", case_id=case, metric="max_rel_err_dT_vs_lumped_ode", value=err,
             criterion="<= 0.005", **{"pass": bool(err <= 0.005)},
             note="convection + radiation on all six patches"),
        dict(check="V5", case_id=case, metric="nonuniformity_K", value=uniformity(case, s),
             criterion="<= 0.1", **{"pass": bool(uniformity(case, s) <= 0.1)}),
        dict(check="V5", case_id=case, metric="loss_rate_rel_err_t0",
             value=float(s["Q_loss_W"].iloc[0] / (A * (h * (T0 - T_inf) + eps * SIGMA * (T0 ** 4 - T_inf ** 4))) - 1),
             criterion="info", note="post-processed patch loss vs analytic at t = 0"),
    ])
    return t, dT, dT_ode


def v6(case="V6_phase_lumped"):
    cfg, s = case_config(case), signals(case)
    areas, V = geometry(cfg)
    A = sum(areas.values())
    m = cfg["material"]
    rho, cp, Lf, Ts, Tl = m["rho"], m["cp"], m["Lf"], m["solidus"], m["liquidus"]
    h = cfg["boundary"]["default"]["h"]
    T_inf, T0 = cfg["boundary"]["T_inf"], cfg["initial"]["T"]
    t = s["t_s"].to_numpy()
    T = s["mean_T_rise_K"].to_numpy() + T_inf
    Q = s["Q_loss_W"].to_numpy()

    # lumped enthalpy ODE, H(T) = rho V [cp (T - Tinf) + Lf (1 - alpha(T))]
    Tg = np.linspace(T_inf, 2500.0, 200001)
    Hg = rho * V * (cp * (Tg - T_inf) + Lf * (1 - np.clip((Tl - Tg) / (Tl - Ts), 0, 1)))
    T_of_H = lambda H: np.interp(H, Hg, Tg)  # noqa: E731
    H0 = np.interp(T0, Tg, Hg)
    sol = solve_ivp(lambda _t, H: [-A * h * (T_of_H(H[0]) - T_inf)], (0, t[-1]), [H0],
                    t_eval=t, rtol=1e-11, atol=1e-6, max_step=0.05)
    T_ode = T_of_H(sol.y[0])

    def crossing(level):
        i = np.flatnonzero((T[:-1] > level) & (T[1:] <= level))[0]
        return t[i] + (T[i] - level) * (t[i + 1] - t[i]) / (T[i] - T[i + 1])

    t1, t2 = crossing(Tl), crossing(Ts)
    tt = np.linspace(t1, t2, 4001)
    released = np.trapezoid(np.interp(tt, t, Q), tt)
    expected = rho * V * (cp * (Tl - Ts) + Lf)
    rel = released / expected - 1.0
    dTmax = float(np.max(np.abs(T - T_ode)))
    res = float(np.max(np.abs(s["residual_J"])) / np.interp(T0, Tg, Hg))
    record([
        dict(check="V6", case_id=case, metric="mushy_heat_released_rel_err", value=rel,
             criterion="|x| <= 0.01", **{"pass": bool(abs(rel) <= 0.01)},
             note=f"released {released:.1f} J vs rho V (cp*50 + Lf) = {expected:.1f} J; "
                  f"mushy crossing {t1:.2f}-{t2:.2f} s"),
        dict(check="V6", case_id=case, metric="max_abs_err_T_vs_lumped_ode_K", value=dTmax,
             criterion="<= 2", **{"pass": bool(dTmax <= 2.0)}),
        dict(check="V6", case_id=case, metric="energy_residual_rel_H0", value=res,
             criterion="<= 0.005", **{"pass": bool(res <= 0.005)}),
        dict(check="V6", case_id=case, metric="nonuniformity_K", value=uniformity(case, s),
             criterion="<= 0.1", **{"pass": bool(uniformity(case, s) <= 0.1)}),
        dict(check="V6", case_id=case, metric="alpha_consistency_max",
             value=float(np.nanmax(s["alpha_consistency"])), criterion="info",
             note="max |alpha.solid - alpha(T)| where alpha.solid was written"),
    ])
    return t, T, T_ode, (t1, t2)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    t, dT, ex = v5a()
    ax[0].plot(t, dT, "o", ms=2, label="AdditiveFOAM")
    ax[0].plot(t, ex, "-", lw=1, label="exp(-sum(A h) t / rho c V)")
    ax[0].set(title="V5a per-patch convection", xlabel="t (s)", ylabel="T - T_inf (K)")
    ax[0].legend(fontsize=7)
    t, dT, ode = v5b()
    ax[1].plot(t, dT, "o", ms=2, label="AdditiveFOAM")
    ax[1].plot(t, ode, "-", lw=1, label="lumped ODE")
    ax[1].set(title="V5b h = 15, eps = 0.6", xlabel="t (s)", ylabel="T - T_inf (K)")
    ax[1].legend(fontsize=7)
    t, T, Tode, (t1, t2) = v6()
    ax[2].plot(t, T, "o", ms=2, label="AdditiveFOAM")
    ax[2].plot(t, Tode, "-", lw=1, label="lumped enthalpy ODE")
    ax[2].axhspan(1750, 1800, color="0.9", label="mushy range")
    ax[2].set(title="V6 solidification", xlabel="t (s)", ylabel="T (K)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "V5_V6_lumped.png", dpi=150)


if __name__ == "__main__":
    main()
