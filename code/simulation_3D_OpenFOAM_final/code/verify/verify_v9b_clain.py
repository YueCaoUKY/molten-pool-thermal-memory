#!/usr/bin/env python3
"""V9gtaw realism check: our model vs Clain, Teixeira & Araujo (2017).

Autogenous GTAW bead-on-plate on SAE 1020, 200 x 100 x 6 mm, 100 A, 15 V,
2.5 mm/s; bottom-face thermocouples at x = 100 mm, y = 50 (TC2) and 50 +- 5 mm
(TC1/TC3). Measured data: verification/V9b_Clain2017/.

Our cases (configs/V9gtaw_*.yaml) use the realistic S0 model (liquid k x5):
  canon      eta 0.80, sigma 1.5 mm  (the S0 source, a prediction)
  s1.9       eta 0.50 / 0.65 / 0.80, sigma 1.9 mm (the authors' radius)
  bcClain    eta 0.65, sigma 1.9 mm with the authors' h 8, eps 0.8
Thermocouple histories come from patchProbes on the bottom patch at every
step. The fusion zone (solidus 1750 K; liquidus 1800 K reported too) is the
widest and deepest extent within 8 mm of x = 100 mm while the torch passes,
t in [28, 38] s.

Time alignment (declared): the measured time axis of Fig. 11 has no stated
origin, so the measured curves are shifted until the TC2 peak times agree.

Declared criteria (fixed before the first comparison), using the sigma 1.9 mm
cases with each observable fitted linearly in eta (extrapolation is flagged):
  C1 thermal consistency: the eta matching the TC2 peak rise and the eta
     matching the TC2 level at the last measured time (199 s aligned) differ
     by <= 0.10;
  C2 at the TC2-peak eta: TC1/TC3 peak rise within 10 %, TC2 level at 199 s
     within 10 %, TC2 cycle RMSE over [peak - 10 s, 199 s] <= 8 % of the
     measured peak rise;
  C3 at the TC2-peak eta: fusion width within 25 % of 3.79 mm and penetration
     within 0.3 mm of 0.78 mm.

Post-hoc calibration (declared after C1-C3 were evaluated, before the runs it
uses; results are labelled "posthoc"):
  eta is fixed at 0.70, the energy-matching value from the sigma 1.9 family.
  sigma takes the values 1.9 (interpolated between eta 0.65 and 0.80), 2.5,
  3.0 and 3.5 mm. sigma* minimises
    J = mean(|TC2 rise err|, |TC1/3 rise err|, |width err| / 3.79, |pen err| / 0.78)
  with the observables interpolated linearly in sigma.
  Criteria at (0.70, sigma*):
    C4 TC2 and TC1/TC3 peak rises within 10 %;
    C5 TC2 level at 199 s within 10 %;
    C6 fusion width within 25 % and penetration within 0.3 mm.
The canonical prediction and the boundary-condition variant are information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vlib import FIG_DIR, RESULTS_DIR, RUN_ROOT, SUMMARY, VER_DIR, record

from mp3d.foam_io import read_field, time_dirs
from mp3d.grid import StructuredMesh

DATA = VER_DIR / "V9b_Clain2017"
CASES = {"canon": "V9gtaw_canon_e0p80_s1p5", "e0.50": "V9gtaw_e0p50_s1p9", "e0.65": "V9gtaw_e0p65_s1p9",
         "e0.80": "V9gtaw_e0p80_s1p9", "bcClain": "V9gtaw_e0p65_s1p9_bcClain",
         "s2.5": "V9gtaw_e0p70_s2p5", "s3.0": "V9gtaw_e0p70_s3p0", "s3.5": "V9gtaw_e0p70_s3p5"}
ETA_CASES = [(0.50, "e0.50"), (0.65, "e0.65"), (0.80, "e0.80")]
SIGMA_CASES = [(2.5, "s2.5"), (3.0, "s3.0"), (3.5, "s3.5")]
ETA_FIXED = 0.70
KEYS = ["pk2", "pk13", "lvl", "rmse", "w_sol", "p_sol", "w_liq", "p_liq"]
T0 = 303.15
FZ_W, FZ_P = 3.79, 0.78          # mm, measured width and penetration (6 mm plate)
X_TC = 0.100


def measured():
    d = pd.read_csv(DATA / "thermal_cycles_fig11.csv")
    d = d[(d.plate_thickness_mm == 6.0) & (d.series == "EXP")]
    tc2 = d[d.curve == "TC2"].sort_values("t_s_as_plotted")
    tc13 = d[d.curve.isin(["TC1_or_TC3", "TC1=TC2=TC3"])].sort_values("t_s_as_plotted")
    pk = pd.read_csv(DATA / "peak_temperatures.csv")
    pk = pk[pk.plate_thickness_mm == 6.0].set_index("tc")
    return (tc2.t_s_as_plotted.to_numpy(), tc2.T_degC.to_numpy() + 273.15,
            tc13.t_s_as_plotted.to_numpy(), tc13.T_degC.to_numpy() + 273.15,
            float(pk.loc["TC2", "T_peak_exp_from_table1_degC"]) + 273.15,
            0.5 * (float(pk.loc["TC1", "T_peak_exp_from_table1_degC"]) + float(pk.loc["TC3", "T_peak_exp_from_table1_degC"])) + 273.15,
            float(pk.loc["TC2", "t_peak_exp_fig11_s_as_plotted"]))


def probe_histories(case):
    """patchProbes output: columns time, TC1 (y 45), TC2 (y 50), TC3 (y 55)."""
    base = RUN_ROOT / case / "postProcessing" / "tcProbes"
    f = sorted(base.glob("*/T"))[0]
    rows = [list(map(float, ln.split())) for ln in f.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    a = np.array(rows)
    return a[:, 0], a[:, 1:4]


def fusion_zone(case, iso):
    mesh = StructuredMesh(RUN_ROOT / case)
    sel_x = np.abs(mesh.xc - X_TC) <= 0.008
    z_nodes = np.concatenate([[0.0], mesh.zc[::-1]])
    width, pen = 0.0, 0.0
    for t, d in time_dirs(RUN_ROOT / case, "T"):
        if not 28.0 <= t <= 38.0:
            continue
        Tf = read_field(d / "T")
        T3 = mesh.cells(Tf.internal)
        Ttop = mesh.patch_grid("top", Tf.boundary["top"]["value"])
        col = np.concatenate([Ttop[None], T3[::-1]], axis=0)[:, :, sel_x]     # [depth, y, x]
        hot = col >= iso
        if not hot.any():
            continue
        k = np.flatnonzero(hot.any(axis=(1, 2))).max()
        if k + 1 < len(z_nodes):
            Ta, Tb = col[k].max(), col[k + 1].max()
            pen = max(pen, -(z_nodes[k] + (iso - Ta) * (z_nodes[k + 1] - z_nodes[k]) / (Tb - Ta)))
        prof = col[0].max(axis=1)                                             # top surface, max over x
        above = np.flatnonzero(prof >= iso)
        yc = mesh.yc
        lo, hi = above[0], above[-1]
        y_lo = yc[lo - 1] + (iso - prof[lo - 1]) * (yc[lo] - yc[lo - 1]) / (prof[lo] - prof[lo - 1])
        y_hi = yc[hi] + (prof[hi] - iso) * (yc[hi + 1] - yc[hi]) / (prof[hi] - prof[hi + 1])
        width = max(width, y_hi - y_lo)
    return width * 1e3, pen * 1e3


def observables(case, meas):
    t2e, T2e, t13e, T13e, pk2e, pk13e, tpk2e = meas
    t, T = probe_histories(case)
    tc2, tc13 = T[:, 1], 0.5 * (T[:, 0] + T[:, 2])
    i2 = int(np.argmax(tc2))
    shift = t[i2] - t2e[np.argmax(T2e)]                     # measured time + shift = model time
    t_last = t2e.max() + shift
    lvl_m = float(np.interp(t_last, t, tc2))
    lvl_e = float(T2e[np.argmax(t2e)])
    w = (t2e + shift >= t[i2] - 10.0)
    rmse = float(np.sqrt(np.mean((np.interp(t2e[w] + shift, t, tc2) - T2e[w]) ** 2)))
    ws, ps = fusion_zone(case, 1750.0)
    wl, pl = fusion_zone(case, 1800.0)
    return dict(case=case, shift_s=shift, pk2=tc2[i2], pk13=tc13.max(), t_pk2=t[i2], lvl=lvl_m, lvl_e=lvl_e,
                rmse=rmse, w_sol=ws, p_sol=ps, w_liq=wl, p_liq=pl, t=t, tc2=tc2, tc13=tc13)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    meas = measured()
    t2e, T2e, t13e, T13e, pk2e, pk13e, tpk2e = meas
    # only runs that finished and were collected (a run still in progress also has probe files)
    obs = {k: observables(c, meas) for k, c in CASES.items() if (RESULTS_DIR / c / "run_manifest.json").exists()}
    rise2e, rise13e = pk2e - T0, pk13e - T0
    lvl_e = next(iter(obs.values()))["lvl_e"]
    rows = []
    for k, o in obs.items():
        note = (f"TC2 peak {o['pk2'] - 273.15:.0f} degC (meas {pk2e - 273.15:.0f}); TC1/3 {o['pk13'] - 273.15:.0f} "
                f"(meas {pk13e - 273.15:.0f}); TC2 at 199 s {o['lvl'] - 273.15:.0f} (meas {o['lvl_e'] - 273.15:.0f}); "
                f"fusion width/pen (solidus) {o['w_sol']:.2f}/{o['p_sol']:.2f} mm (meas {FZ_W}/{FZ_P}); "
                f"liquidus {o['w_liq']:.2f}/{o['p_liq']:.2f} mm; time shift {o['shift_s']:+.2f} s")
        rows.append(dict(check="V9b", case_id=o["case"], metric="summary", value=float((o["pk2"] - T0) / rise2e - 1),
                         criterion="info", note=note))
        print(f"{k:8s} {note}")

    def errors(v):
        return dict(tc2=(v["pk2"] - T0) / rise2e - 1, tc13=(v["pk13"] - T0) / rise13e - 1,
                    lvl=(v["lvl"] - T0) / (lvl_e - T0) - 1, rmse=v["rmse"] / rise2e,
                    w=v["w_sol"] / FZ_W - 1, p=v["p_sol"] - FZ_P)

    have = [(e, obs[k]) for e, k in ETA_CASES if k in obs]
    at_eta = None
    if len(have) >= 2:
        etas = np.array([e for e, _ in have])
        fit = {key: np.polyfit(etas, [o[key] for _, o in have], 1) for key in KEYS}

        def at_eta(eta):
            return {key: float(np.polyval(fit[key], eta)) for key in KEYS}

        def eta_for(key, target):
            vals = np.array([o[key] for _, o in have])
            e = float((target - fit[key][1]) / fit[key][0])
            return e, bool(target < vals.min() or target > vals.max())

        eta_pk, ext_pk = eta_for("pk2", pk2e)
        eta_lvl, ext_lvl = eta_for("lvl", lvl_e)
        rows += [dict(check="V9b", case_id="sigma1.9", metric="eta_matching_TC2_peak", value=eta_pk, criterion="info",
                      note="linear extrapolation below eta 0.50" if ext_pk else "interpolated"),
                 dict(check="V9b", case_id="sigma1.9", metric="eta_matching_TC2_level_199s", value=eta_lvl, criterion="info",
                      note="extrapolated" if ext_lvl else "interpolated"),
                 dict(check="V9b", case_id="sigma1.9", metric="C1_eta_spread_peak_vs_level", value=abs(eta_pk - eta_lvl),
                      criterion="<= 0.10", **{"pass": bool(abs(eta_pk - eta_lvl) <= 0.10)})]
        E = errors(at_eta(eta_pk))
        rows += [dict(check="V9b", case_id="sigma1.9@eta_pk", metric="C2_TC13_peak_rise_rel_err", value=E["tc13"],
                      criterion="|x| <= 0.10", **{"pass": bool(abs(E["tc13"]) <= 0.10)}),
                 dict(check="V9b", case_id="sigma1.9@eta_pk", metric="C2_TC2_level_199s_rel_err", value=E["lvl"],
                      criterion="|x| <= 0.10", **{"pass": bool(abs(E["lvl"]) <= 0.10)}),
                 dict(check="V9b", case_id="sigma1.9@eta_pk", metric="C2_TC2_cycle_rmse_rel_peak_rise", value=E["rmse"],
                      criterion="<= 0.08", **{"pass": bool(E["rmse"] <= 0.08)}),
                 dict(check="V9b", case_id="sigma1.9@eta_pk", metric="C3_fusion_width_rel_err_solidus", value=E["w"],
                      criterion="|x| <= 0.25", **{"pass": bool(abs(E["w"]) <= 0.25)}),
                 dict(check="V9b", case_id="sigma1.9@eta_pk", metric="C3_penetration_err_mm_solidus", value=E["p"],
                      criterion="|x| <= 0.3", **{"pass": bool(abs(E["p"]) <= 0.3)})]
        print(f"sigma 1.9: eta(TC2 peak) = {eta_pk:.3f}{' (extrapolated)' if ext_pk else ''}; eta(level 199 s) = {eta_lvl:.3f}")

    # post-hoc sigma calibration at eta 0.70
    sig_pts = []
    if at_eta is not None:
        sig_pts.append((1.9, at_eta(ETA_FIXED)))
    sig_pts += [(s, {key: obs[k][key] for key in KEYS}) for s, k in SIGMA_CASES if k in obs]
    best = None
    if len(sig_pts) >= 2:
        sig = np.array([s for s, _ in sig_pts])
        grid = np.linspace(sig.min(), sig.max(), 321)
        interp = {key: np.interp(grid, sig, [v[key] for _, v in sig_pts]) for key in KEYS}
        J = np.mean([np.abs((interp["pk2"] - T0) / rise2e - 1), np.abs((interp["pk13"] - T0) / rise13e - 1),
                     np.abs(interp["w_sol"] / FZ_W - 1), np.abs(interp["p_sol"] - FZ_P) / FZ_P], axis=0)
        ib = int(np.argmin(J))
        best = {key: float(interp[key][ib]) for key in KEYS}
        s_star = float(grid[ib])
        E = errors(best)
        cid = f"posthoc_eta{ETA_FIXED:.2f}_sigma_star"
        rows += [dict(check="V9b", case_id=cid, metric="posthoc_sigma_star_mm", value=s_star, criterion="info",
                      note=f"J = {J[ib]:.3f}; sigma points {sig.tolist()} mm; sigma*=edge of range" if ib in (0, len(grid) - 1)
                      else f"J = {J[ib]:.3f}; sigma points {sig.tolist()} mm"),
                 dict(check="V9b", case_id=cid, metric="posthoc_C4_TC2_peak_rise_rel_err", value=E["tc2"],
                      criterion="|x| <= 0.10", **{"pass": bool(abs(E["tc2"]) <= 0.10)}),
                 dict(check="V9b", case_id=cid, metric="posthoc_C4_TC13_peak_rise_rel_err", value=E["tc13"],
                      criterion="|x| <= 0.10", **{"pass": bool(abs(E["tc13"]) <= 0.10)}),
                 dict(check="V9b", case_id=cid, metric="posthoc_C5_TC2_level_199s_rel_err", value=E["lvl"],
                      criterion="|x| <= 0.10", **{"pass": bool(abs(E["lvl"]) <= 0.10)}),
                 dict(check="V9b", case_id=cid, metric="posthoc_C6_fusion_width_rel_err_solidus", value=E["w"],
                      criterion="|x| <= 0.25", **{"pass": bool(abs(E["w"]) <= 0.25)}),
                 dict(check="V9b", case_id=cid, metric="posthoc_C6_penetration_err_mm_solidus", value=E["p"],
                      criterion="|x| <= 0.3", **{"pass": bool(abs(E["p"]) <= 0.3)}),
                 dict(check="V9b", case_id=cid, metric="posthoc_TC2_cycle_rmse_rel_peak_rise", value=E["rmse"],
                      criterion="info")]
        print(f"post-hoc: sigma* = {s_star:.2f} mm at eta {ETA_FIXED}: " + ", ".join(f"{k} {v:+.3f}" for k, v in E.items()))
    if SUMMARY.exists():                       # drop stale post-hoc rows before recording
        old = pd.read_csv(SUMMARY)
        old[~((old["check"] == "V9b") & old["metric"].astype(str).str.startswith("posthoc_"))].to_csv(SUMMARY, index=False)
    record(rows)

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
    for i, (k, o) in enumerate(obs.items()):
        ax[0].plot(o["t"], o["tc2"] - 273.15, color=f"C{i}", lw=1, label=f"{k} TC2")
        ax[0].plot(o["t"], o["tc13"] - 273.15, ":", color=f"C{i}", lw=1)
    ref = obs.get("e0.65") or next(iter(obs.values()))
    ax[0].plot(t2e + ref["shift_s"], T2e - 273.15, "k.", ms=2, label="measured TC2 (aligned)")
    ax[0].plot(t13e + ref["shift_s"], T13e - 273.15, ".", color="0.5", ms=2, label="measured TC1/TC3")
    ax[0].set(xlabel="t (s)", ylabel="T (degC)", title="bottom-face thermocouples, x = 100 mm (dotted: y +- 5 mm)")
    ax[0].legend(fontsize=6)
    for a, pts, xl, title in ((ax[1], [(e, obs[k]) for e, k in ETA_CASES if k in obs], "eta (sigma 1.9 mm)",
                               "sigma 1.9 mm: model / measured vs eta"),
                              (ax[2], sig_pts, "sigma (mm), eta 0.70",
                               "eta 0.70: model / measured vs sigma (1.9 interpolated)")):
        if len(pts) < 2:
            continue
        xs = [p for p, _ in pts]
        for key, ref_v, lab in (("pk2", pk2e, "TC2 peak rise"), ("pk13", pk13e, "TC1/3 peak rise"),
                                ("lvl", lvl_e, "TC2 at 199 s")):
            a.plot(xs, [(v[key] - T0) / (ref_v - T0) for _, v in pts], "o-", label=lab)
        a.plot(xs, [v["w_sol"] / FZ_W for _, v in pts], "s-", label="fusion width")
        a.plot(xs, [v["p_sol"] / FZ_P for _, v in pts], "^-", label="penetration")
        a.axhline(1.0, color="k", lw=0.8)
        a.set(xlabel=xl, ylabel="model / measured", title=title)
        a.legend(fontsize=7)
    if best is not None:
        ax[2].axvline(s_star, color="0.5", ls="--", lw=0.8)
    ks = list(obs)
    ax[3].bar(np.arange(len(ks)) - 0.2, [obs[k]["w_sol"] for k in ks], 0.4, label="width")
    ax[3].bar(np.arange(len(ks)) + 0.2, [obs[k]["p_sol"] for k in ks], 0.4, label="penetration")
    ax[3].axhline(FZ_W, color="C0", ls="--", lw=0.8)
    ax[3].axhline(FZ_P, color="C1", ls="--", lw=0.8)
    ax[3].set_xticks(range(len(ks)), ks, fontsize=7)
    ax[3].set(ylabel="mm", title=f"fusion zone (solidus); measured {FZ_W} / {FZ_P} mm")
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "V9b_clain2017.png", dpi=150)


if __name__ == "__main__":
    main()
