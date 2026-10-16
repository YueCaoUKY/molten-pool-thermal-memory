#!/usr/bin/env python3
"""V9 literature plausibility: Champagne (2020) MIG bead on a 6061-T6 bar.

Reference data: verification/V9_champagne2020/ (digitised Fig. 3.5, README).
The model is linear (constant properties, no latent heat, convection only,
T0 = T_inf), so each candidate source run at eta_ref = 0.5 is rescaled exactly:
T(eta) = T0 + (eta / eta_ref) (T_ref - T0).

Declared calibration (fixed before comparison):
  eta     so that the model's mean thermocouple rise over t in [12, 14] s
          equals the measured mean rise (measured rise relative to each
          thermocouple's mean over t < 1 s);
  source  among V9a-c, the candidate with the smallest fusion-zone error
          |d_pen| / 1.702 mm + mean_i |d_w_i| / w_i at the measured depths,
          650 degC isotherm, envelope over x and t in [3, 6] s near x = 75 mm.
Predictions judged for the selected candidate (criteria fixed before
comparison):
  peak time of every thermocouple within 0.3 s of the measurement;
  peak rise of TC2-TC4 within 15 % (TC1, 3 mm below the top, is reported
  only: the authors' own FEM overpredicts it by 9 %);
  penetration within 0.3 mm of 1.702 mm.
The authors' FEM curves are compared for information.

Post-hoc analysis (added after the declared check failed; information only,
metric names start with "posthoc_"): eta recalibrated (a) on the TC2-TC4 peak
rises and (b) on the macrograph penetration, to show how the late-time level,
the peaks and the fusion zone constrain the absorbed power differently in a
conduction-only model of this experiment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from vlib import FIG_DIR, RUN_ROOT, VER_DIR, case_config, record

from mp3d.foam_io import read_field, time_dirs
from mp3d.grid import StructuredMesh

CASES = ["V9a_champagne_s1p5_d0p5", "V9b_champagne_s2p0_d0p5", "V9c_champagne_s1p5_d1p5"]
TC_X, TC_Z = 0.075, np.array([-0.003, -0.006, -0.009, -0.012])
FZ_DEPTH = np.array([0.27, 0.5, 1.0, 1.3, 1.54]) * 1e-3
FZ_WIDTH = np.array([4.3, 3.6, 2.9, 2.15, 1.5]) * 1e-3
FZ_PEN = 1.702e-3
T_MELT = 923.15
ETA_REF = 0.5
P_ARC = 23.1 * 167.0


def reference_data():
    d = pd.read_csv(VER_DIR / "V9_champagne2020" / "digitized_fig3_5.csv")
    out = {}
    for (series, tc), g in d.groupby(["series", "TC"]):
        g = g.sort_values("t_s")
        out[(series, int(tc))] = (g["t_s"].to_numpy(), g["T_degC"].to_numpy() + 273.15)
    return out


def model_histories(case):
    cfg = case_config(case)
    case_dir = RUN_ROOT / case
    mesh = StructuredMesh(case_dir)
    t_list, tc_T, env = [], [], {}
    for t, d in time_dirs(case_dir, "T"):
        Tf = read_field(d / "T")
        side = mesh.patch_grid("ymax", Tf.boundary["ymax"]["value"])      # [k, i]
        f = RegularGridInterpolator((mesh.zc, mesh.xc), side)
        t_list.append(t)
        tc_T.append(f(np.column_stack([TC_Z, np.full(4, TC_X)])))
        if 3.0 - 1e-9 <= t <= 6.0 + 1e-9:
            env[t] = mesh.cells(Tf.internal), mesh.patch_grid("top", Tf.boundary["top"]["value"])
    return cfg, mesh, np.array(t_list), np.array(tc_T), env


def fusion_zone(mesh, env, scale, T0):
    """Width at each measured depth and penetration of the 650 degC isotherm,
    envelope over the stored snapshots for x within 10 mm of x = 75 mm."""
    sel_x = np.abs(mesh.xc - TC_X) <= 0.010
    z_nodes = np.concatenate([[0.0], mesh.zc[::-1]])                    # top face, then cells downward
    pen, width = 0.0, np.zeros(len(FZ_DEPTH))
    for T3, Ttop in env.values():
        col = np.concatenate([Ttop[None], T3[::-1]], axis=0)[:, :, sel_x]   # [depth, y, x]
        col = T0 + scale * (col - T0)
        hot = col >= T_MELT
        if not hot.any():
            continue
        k = np.flatnonzero(hot.any(axis=(1, 2))).max()
        if k + 1 < len(z_nodes):
            Ta, Tb = col[k].max(), col[k + 1].max()
            zi = z_nodes[k] + (T_MELT - Ta) * (z_nodes[k + 1] - z_nodes[k]) / (Tb - Ta)
            pen = max(pen, -zi)
        for n, zd in enumerate(-FZ_DEPTH):
            kz = np.searchsorted(-z_nodes, -zd) - 1
            w = (zd - z_nodes[kz + 1]) / (z_nodes[kz] - z_nodes[kz + 1])
            prof = (w * col[kz] + (1 - w) * col[kz + 1]).max(axis=1)        # max over x per y
            above = np.flatnonzero(prof >= T_MELT)
            if above.size == 0:
                continue
            yc = mesh.yc
            lo, hi = above[0], above[-1]
            y_lo = yc[lo] if lo == 0 else yc[lo - 1] + (T_MELT - prof[lo - 1]) * (yc[lo] - yc[lo - 1]) / (prof[lo] - prof[lo - 1])
            y_hi = yc[hi] if hi == len(yc) - 1 else yc[hi] + (prof[hi] - T_MELT) * (yc[hi + 1] - yc[hi]) / (prof[hi] - prof[hi + 1])
            width[n] = max(width[n], y_hi - y_lo)
    return pen, width


def peak(t, T):
    i = int(np.argmax(T))
    return t[i], T[i]


def measured(ref):
    """Per thermocouple: baseline, peak time, peak rise, late (12-14 s) rise."""
    out = []
    for n in range(4):
        te, Te = ref[("EXP", n + 1)]
        base = Te[te < 1.0].mean()
        tp, Tp = peak(te, Te)
        w = (te >= 12.0) & (te <= 14.0)
        out.append(dict(t=te, T=Te, base=base, t_pk=tp, rise_pk=Tp - base, rise_late=np.mean(Te[w] - base), w=w))
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ref = reference_data()
    meas = measured(ref)
    raw, results = {}, {}
    for case in CASES:
        cfg, mesh, t, T, env = model_histories(case)
        T0 = cfg["initial"]["T"]
        raw[case] = (cfg, mesh, t, T, env, T0)
        mod_late = [np.mean(np.interp(m["t"][m["w"]], t, T[:, n]) - T0) for n, m in enumerate(meas)]
        scale = float(np.mean([m["rise_late"] for m in meas]) / np.mean(mod_late))
        pen, width = fusion_zone(mesh, env, scale, T0)
        fz_err = abs(pen - FZ_PEN) / FZ_PEN + float(np.mean(np.abs(width - FZ_WIDTH) / FZ_WIDTH))
        results[case] = dict(cfg=cfg, t=t, T=T0 + scale * (T - T0), eta=ETA_REF * scale, pen=pen,
                             width=width, fz_err=fz_err, mod_late=np.array(mod_late))
        print(f"{case}: eta = {ETA_REF * scale:.3f} (absorbed {ETA_REF * scale * P_ARC:.0f} W); "
              f"penetration {pen * 1e3:.2f} mm; widths {np.round(width * 1e3, 2)} mm; fusion-zone error {fz_err:.3f}")
    best = min(results, key=lambda c: results[c]["fz_err"])
    r = results[best]
    rows = [dict(check="V9", case_id=c, metric="calibrated_eta", value=results[c]["eta"], criterion="info",
                 note=f"fusion-zone error {results[c]['fz_err']:.3f}; penetration {results[c]['pen'] * 1e3:.2f} mm; "
                      f"widths {np.round(results[c]['width'] * 1e3, 2).tolist()} mm")
            for c in results]
    rows.append(dict(check="V9", case_id=best, metric="selected_candidate", value=1.0, criterion="info",
                     note="smallest fusion-zone error (declared calibration)"))
    pen_err = r["pen"] - FZ_PEN
    rows.append(dict(check="V9", case_id=best, metric="penetration_error_mm", value=pen_err * 1e3,
                     criterion="|x| <= 0.3", **{"pass": bool(abs(pen_err) <= 0.3e-3)},
                     note=f"model {r['pen'] * 1e3:.2f} vs macrograph 1.702 mm"))
    for n, m in enumerate(meas):
        tpm, Tpm = peak(r["t"], r["T"][:, n])
        dt_pk = tpm - m["t_pk"]
        rise = (Tpm - r["cfg"]["initial"]["T"]) / m["rise_pk"] - 1.0
        rows.append(dict(check="V9", case_id=best, metric=f"TC{n + 1}_peak_time_error_s", value=dt_pk,
                         criterion="|x| <= 0.3", **{"pass": bool(abs(dt_pk) <= 0.3)},
                         note=f"model {tpm:.2f} s vs measured {m['t_pk']:.2f} s"))
        judged = n > 0
        rows.append(dict(check="V9", case_id=best, metric=f"TC{n + 1}_peak_rise_rel_error", value=rise,
                         criterion="|x| <= 0.15" if judged else "info",
                         **({"pass": bool(abs(rise) <= 0.15)} if judged else {}),
                         note=f"model {Tpm - 273.15:.1f} degC vs measured {m['rise_pk'] + m['base'] - 273.15:.1f} degC"))
        w = (m["t"] >= 2.0) & (m["t"] <= 14.0)
        rmse = float(np.sqrt(np.mean((np.interp(m["t"][w], r["t"], r["T"][:, n]) - m["T"][w]) ** 2)))
        rows.append(dict(check="V9", case_id=best, metric=f"TC{n + 1}_rmse_2_14s_K", value=rmse, criterion="info"))
        tf, Tf_ = ref[("FEM", n + 1)]
        wf = (tf >= 2.0) & (tf <= 25.0)
        rows.append(dict(check="V9", case_id=best, metric=f"TC{n + 1}_rmse_vs_authors_FEM_K",
                         value=float(np.sqrt(np.mean((np.interp(tf[wf], r["t"], r["T"][:, n]) - Tf_[wf]) ** 2))),
                         criterion="info", note="authors' FEM uses its own (unreported) source power"))

    # post-hoc: alternative eta calibrations for every candidate
    post = {}
    for case, (cfg, mesh, t, T, env, T0) in raw.items():
        pk_ref = np.array([T[:, n].max() - T0 for n in range(4)])
        s_pk = float(np.mean(np.array([m["rise_pk"] for m in meas])[1:] / pk_ref[1:]))
        lo, hi = 0.05, 3.0
        for _ in range(40):                                   # penetration is monotone in eta
            mid = 0.5 * (lo + hi)
            if fusion_zone(mesh, env, mid, T0)[0] > FZ_PEN:
                hi = mid
            else:
                lo = mid
        s_pen = 0.5 * (lo + hi)
        late = np.mean([m["rise_late"] for m in meas])
        mod_late = results[case]["mod_late"].mean()
        post[case] = {}
        for tag, s in (("peaks", s_pk), ("penetration", s_pen)):
            pen, width = fusion_zone(mesh, env, s, T0)
            pk_err = s * pk_ref / np.array([m["rise_pk"] for m in meas]) - 1.0
            late_err = s * mod_late / late - 1.0
            post[case][tag] = dict(eta=ETA_REF * s, pen=pen, width=width, pk_err=pk_err, late_err=late_err)
            rows.append(dict(check="V9", case_id=case, metric=f"posthoc_eta_from_{tag}", value=ETA_REF * s,
                             criterion="info",
                             note=(f"late 12-14 s rise error {100 * late_err:+.1f} %; peak rise errors TC1-4 "
                                   f"{np.round(100 * pk_err, 1).tolist()} %; penetration {pen * 1e3:.2f} mm; "
                                   f"top width {width[0] * 1e3:.2f} mm")))
    record(rows)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.5, 1.0, 1.1]})
    for n, m in enumerate(meas):
        tf, Tf_ = ref[("FEM", n + 1)]
        ax[0].plot(m["t"], m["T"] - 273.15, "o", ms=2, color=f"C{n}", label=f"TC{n + 1} measured")
        ax[0].plot(tf, Tf_ - 273.15, ":", lw=1, color=f"C{n}")
        ax[0].plot(r["t"], r["T"][:, n] - 273.15, "-", lw=1.2, color=f"C{n}")
    ax[0].set(xlabel="t (s)", ylabel="T (degC)", xlim=(0, 25),
              title=f"{best.split('_')[0]}: solid = this model (eta {r['eta']:.2f}, late-time calibration); dotted = authors' FEM")
    ax[0].legend(fontsize=7)
    for c, res in results.items():
        ax[1].plot(res["width"] * 1e3, -FZ_DEPTH * 1e3, "o-", ms=3, label=f"{c.split('_')[0]} eta {res['eta']:.2f}: pen {res['pen'] * 1e3:.2f} mm")
        pp = post[c]["penetration"]
        ax[1].plot(pp["width"] * 1e3, -FZ_DEPTH * 1e3, "x--", ms=4, label=f"{c.split('_')[0]} eta {pp['eta']:.2f} (post-hoc, pen fitted)")
    ax[1].plot(FZ_WIDTH * 1e3, -FZ_DEPTH * 1e3, "ks-", ms=4, lw=2, label="macrograph (pen 1.70 mm)")
    ax[1].set(xlabel="fusion width (mm)", ylabel="depth (mm)", title="650 degC fusion boundary at x ~ 75 mm")
    ax[1].legend(fontsize=6)
    labels, vals = [], []
    for c in results:
        labels += [f"{c.split('_')[0]}\nlate", f"{c.split('_')[0]}\npeaks", f"{c.split('_')[0]}\npen"]
        vals += [results[c]["eta"], post[c]["peaks"]["eta"], post[c]["penetration"]["eta"]]
    ax[2].bar(range(len(vals)), vals, color=["C0", "C1", "C2"] * len(results))
    ax[2].set_xticks(range(len(vals)), labels, fontsize=7)
    ax[2].set(ylabel="eta (absorbed / 3858 W)", title="eta implied by each observable")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "V9_champagne2020.png", dpi=150)


if __name__ == "__main__":
    main()
