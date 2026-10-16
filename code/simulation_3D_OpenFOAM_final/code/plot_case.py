#!/usr/bin/env python3
"""Overview figures for one case (results/<case_id>/figures/).

  overview_signals.png  inputs, surface and subsurface signals, energy budget
  overview_fields.png   top-surface temperature maps with the liquidus and
                        solidus contours, longitudinal x-z sections through the
                        track, and through-thickness profiles

    python code/plot_case.py <case_id> [<case_id> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mp3d import RESULTS_DIR, RUN_ROOT  # noqa: E402
from mp3d.case_writer import build_material, load_config  # noqa: E402
from mp3d.foam_io import read_field  # noqa: E402
from mp3d.grid import StructuredMesh  # noqa: E402


def signals_figure(case_id, cfg, s, out):
    mat = build_material(cfg)
    fig, ax = plt.subplots(3, 3, figsize=(15, 10))
    a = ax[0, 0]
    a.step(s["t_s"], s["I_A"], where="post", color="C0")
    a.set(ylabel="I (A)", title="inputs")
    b = a.twinx()
    b.step(s["t_s"], s["u_mm_s"], where="post", color="C1")
    b.set_ylabel("u (mm/s)", color="C1")
    a = ax[0, 1]
    for col, lab in (("T_top_max_K", "top max"), ("T_top_roi_mean_K", "top ROI mean (10x10 mm)"),
                     ("Tbar_max_K", "thickness-avg max"), ("T_bottom_under_torch_K", "bottom under torch")):
        a.plot(s["t_s"], s[col], label=lab)
    a.axhline(mat.liquidus, color="0.6", ls=":", lw=0.8)
    a.set(ylabel="T (K)", title="temperatures")
    a.legend(fontsize=7)
    a = ax[0, 2]
    a.plot(s["t_s"], s["top_area_liq_mm2"], label="T > liquidus")
    a.plot(s["t_s"], s["top_area_sol_mm2"], label="T > solidus")
    a.set(ylabel="area (mm2)", title="top-surface melt area")
    a.legend(fontsize=7)
    a = ax[1, 0]
    a.plot(s["t_s"], s["top_length_liq_mm"], label="length (liquidus)")
    a.plot(s["t_s"], s["top_width_liq_mm"], label="width (liquidus)")
    a.set(ylabel="mm", title="top-surface pool extents")
    a.legend(fontsize=7)
    a = ax[1, 1]
    a.plot(s["t_s"], s["depth_liq_mm"], label="depth (liquidus)")
    a.plot(s["t_s"], s["depth_sol_mm"], label="depth (solidus)")
    b = a.twinx()
    b.plot(s["t_s"], s["vol_liquid_mm3"], color="C3", lw=0.8)
    b.set_ylabel("liquid volume (mm3)", color="C3")
    a.set(ylabel="mm", title="penetration and molten volume")
    a.legend(fontsize=7)
    a = ax[1, 2]
    for col, lab in (("delta_H_K", "delta_H (plate)"), ("mean_T_rise_K", "mean T rise"),
                     ("dHK_ring_5_15mm_K", "ring 5-15 mm from torch"),
                     ("dHK_lower_half_K", "lower half of plate")):
        a.plot(s["t_s"], s[col], label=lab)
    a.set(ylabel="K", title="workpiece heat content (enthalpy / rho cp_ref V)")
    a.legend(fontsize=7)
    a = ax[2, 0]
    for col in [c for c in s.columns if c.startswith("Q_loss_") and c != "Q_loss_W"]:
        a.plot(s["t_s"], s[col], label=col[7:-2])
    a.set(xlabel="t (s)", ylabel="W", title="boundary loss by patch")
    a.legend(fontsize=7)
    a = ax[2, 1]
    a.plot(s["t_s"], s["E_src_J"] / 1e3, label="absorbed")
    a.plot(s["t_s"], s["E_loss_J"] / 1e3, label="lost")
    a.plot(s["t_s"], (s["H_J"] - s["H_J"].iloc[0]) / 1e3, "--", label="enthalpy rise")
    a.set(xlabel="t (s)", ylabel="kJ", title="energy budget")
    a.legend(fontsize=7)
    a = ax[2, 2]
    a.plot(s["t_s"], 100 * s["residual_rel"])
    a.set(xlabel="t (s)", ylabel="% of absorbed energy", title="closure residual H - H0 - (E_src - E_loss)")
    from mp3d.schedule import ProcessSchedule
    for a in ax.flat:
        a.axvline(ProcessSchedule.from_config(cfg).t_heat_end, color="0.7", lw=0.6)
    fig.suptitle(f"{case_id}: {cfg.get('description', '')}")
    fig.tight_layout()
    fig.savefig(out / "overview_signals.png", dpi=130)
    plt.close(fig)


def fields_figure(case_id, cfg, s, out):
    mat = build_material(cfg)
    case = RUN_ROOT / case_id
    mesh = StructuredMesh(case)
    y0 = cfg["schedule"]["y0"]
    j0 = int(np.searchsorted(mesh.yc, y0)) - 1
    times = [5.0, 10.0, 20.0, 30.0]
    fig, ax = plt.subplots(3, 4, figsize=(18, 11), gridspec_kw={"height_ratios": [1.0, 0.9, 1.3]})
    levels = np.linspace(300, 2200, 20)
    from mp3d.schedule import ProcessSchedule
    sched = ProcessSchedule.from_config(cfg)
    for n, t in enumerate(times):
        d = next(p for p in case.iterdir() if p.name.replace(".", "", 1).isdigit() and abs(float(p.name) - t) < 1e-9)
        Tf = read_field(d / "T")
        T3 = mesh.cells(Tf.internal)
        Ttop = mesh.patch_grid("top", Tf.boundary["top"]["value"])
        xs = float(sched.position(t))
        sel_x = (mesh.xc > xs - 0.045) & (mesh.xc < xs + 0.015)
        sel_y = np.abs(mesh.yc - y0) < 0.012
        a = ax[0, n]
        X, Y = np.meshgrid(mesh.xc[sel_x] * 1e3, mesh.yc[sel_y] * 1e3)
        Tt = Ttop[np.ix_(sel_y, sel_x)]
        cs = a.contourf(X, Y, np.clip(Tt, 300, 2200), levels=levels, cmap="inferno")
        a.contour(X, Y, Tt, levels=[mat.solidus, mat.liquidus], colors=["c", "w"], linewidths=0.8)
        a.plot(xs * 1e3, y0 * 1e3, "+", color="c", ms=8)
        a.set(title=f"top surface t = {t:g} s (+ = torch)", xlabel="x (mm)", ylabel="y (mm)", aspect="equal")
        a = ax[1, n]
        Xs, Zs = np.meshgrid(mesh.xc[sel_x] * 1e3, mesh.zc * 1e3)
        Ts = 0.5 * (T3[:, j0, :] + T3[:, j0 + 1, :])[:, sel_x]
        a.contourf(Xs, Zs, np.clip(Ts, 300, 2200), levels=levels, cmap="inferno")
        a.contour(Xs, Zs, Ts, levels=[mat.solidus, mat.liquidus], colors=["c", "w"], linewidths=0.8)
        a.set(title=f"section y = {y0 * 1e3:g} mm, t = {t:g} s", xlabel="x (mm)", ylabel="z (mm)")
        a.set_aspect(2.0)
    fig.colorbar(cs, ax=ax[:2, :].ravel().tolist(), shrink=0.6, label="T (K), clipped at 2200 K")
    pr = np.load(RESULTS_DIR / case_id / "profiles_0p1s.npz")
    for n, t in enumerate([5.0, 10.0, 20.0, 30.0]):
        a = ax[2, n]
        it = int(np.argmin(np.abs(pr["t"] - t)))
        for m, off in enumerate(pr["offsets_behind_torch"]):
            a.plot(pr["T"][it, m], pr["z"] * 1e3, label=f"{off * 1e3:g} mm behind torch")
        a.axvline(mat.liquidus, color="0.6", ls=":", lw=0.8)
        a.set(title=f"centreline profiles t = {t:g} s", xlabel="T (K)", ylabel="z (mm)")
        a.legend(fontsize=6)
    fig.suptitle(f"{case_id}: white = liquidus {mat.liquidus:g} K, cyan = solidus {mat.solidus:g} K")
    fig.savefig(out / "overview_fields.png", dpi=130)
    plt.close(fig)


def main(cases):
    for c in cases:
        cfg = load_config(RESULTS_DIR / c / "config_resolved.yaml")
        s = pd.read_csv(RESULTS_DIR / c / "signals_0p1s.csv")
        out = RESULTS_DIR / c / "figures"
        out.mkdir(exist_ok=True)
        signals_figure(c, cfg, s, out)
        fields_figure(c, cfg, s, out)
        print(f"figures -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
