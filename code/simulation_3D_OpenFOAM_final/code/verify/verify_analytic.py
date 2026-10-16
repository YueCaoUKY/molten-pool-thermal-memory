#!/usr/bin/env python3
"""V4 analytic limit: AdditiveFOAM vs the exact moving-source plate solution.

Constant properties, no phase change, adiabatic surfaces (configs/V4_*.yaml).
The reference (mp3d.analytic.MovingSourcePlate) is evaluated exactly at the FV
evaluation locations (top-patch face centres at z = 0, cell centres inside),
so no interpolation enters the comparison.

Evaluation sets at t = 5, 10, 15, 20 s (source on) and 25, 30 s (off):
  top     all top faces with |y - y0| <= 20 mm, x <= 120 mm
  centre  cells of the row just below the track (y = y0 - h/2), all z, x <= 120 mm
  trans   cells of the column nearest x_torch - 2 mm, |y - y0| <= 20 mm, all z
Near field: horizontal distance to the torch axis < 5 mm (> 3 sigma) while the
source is on. Errors are normalised by the maximum analytic temperature rise
of the same set and time (far) or of all sets (near).

Criteria (fixed before comparison), per case and time:
  far field   RMS <= 1 %, max <= 2 %
  near field  reported, with the L1 -> L2 reduction factor
"""
from __future__ import annotations

import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vlib import FIG_DIR, RUN_ROOT, case_config, record

from mp3d.analytic import MovingSourcePlate
from mp3d.foam_io import read_field
from mp3d.grid import StructuredMesh
from mp3d.schedule import ProcessSchedule

TIMES = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
NEAR_R = 5.0e-3


def reference(cfg):
    m, g, s = cfg["material"], cfg["geometry"], cfg["source"]
    sched = ProcessSchedule.from_config(cfg)
    return sched, MovingSourcePlate(Lx=g["Lx"], Ly=g["Ly"], H=g["H"], sigma=s["sigma"],
                                    depth=s["depth"], rho=m["rho"], cp=m["cp"], k=m["k"],
                                    segments=sched.analytic_segments())


def snapshot_dir(case_dir, t):
    for d in case_dir.iterdir():
        try:
            if abs(float(d.name) - t) < 1e-9:
                return d
        except ValueError:
            pass
    raise FileNotFoundError(f"no snapshot at t = {t} in {case_dir}")


def evaluation_sets(mesh, cfg, xs):
    y0 = cfg["schedule"]["y0"]
    j_row = int(np.searchsorted(mesh.yc, y0)) - 1               # centre just below the track
    ix = mesh.xc <= 0.120
    jy = np.abs(mesh.yc - y0) <= 0.020
    i_tr = int(np.argmin(np.abs(mesh.xc - (xs - 2e-3))))
    J, I = np.meshgrid(np.flatnonzero(jy), np.flatnonzero(ix), indexing="ij")
    top = dict(kind="top", j=J.ravel(), i=I.ravel(),
               xyz=np.column_stack([mesh.xc[I.ravel()], mesh.yc[J.ravel()], np.zeros(J.size)]))
    K, I2 = np.meshgrid(np.arange(mesh.nz), np.flatnonzero(ix), indexing="ij")
    centre = dict(kind="cells", k=K.ravel(), j=np.full(K.size, j_row), i=I2.ravel(),
                  xyz=np.column_stack([mesh.xc[I2.ravel()], np.full(K.size, mesh.yc[j_row]), mesh.zc[K.ravel()]]))
    K3, J3 = np.meshgrid(np.arange(mesh.nz), np.flatnonzero(jy), indexing="ij")
    trans = dict(kind="cells", k=K3.ravel(), j=J3.ravel(), i=np.full(K3.size, i_tr),
                 xyz=np.column_stack([np.full(K3.size, mesh.xc[i_tr]), mesh.yc[J3.ravel()], mesh.zc[K3.ravel()]]))
    return {"top": top, "centre": centre, "trans": trans}


def compare_case(case_id):
    cfg = case_config(case_id)
    case_dir = RUN_ROOT / case_id
    mesh = StructuredMesh(case_dir)
    sched, ref = reference(cfg)
    T0 = cfg["initial"]["T"]
    y0 = cfg["schedule"]["y0"]
    out, profiles = [], {}
    for t in TIMES:
        Tf = read_field(snapshot_dir(case_dir, t) / "T")
        T3 = mesh.cells(Tf.internal)
        Ttop = mesh.patch_grid("top", Tf.boundary["top"]["value"]) if "value" in Tf.boundary["top"] else T3[-1]
        xs = float(sched.position(t))
        # the source is on during the step ending at t_heat_end, so the near
        # field still exists at that instant
        on = t <= cfg["schedule"]["t_heat_end"] + 1e-9
        sets = evaluation_sets(mesh, cfg, xs)
        near_err, near_scale = [], 0.0
        for name, st in sets.items():
            fv = Ttop[st["j"], st["i"]] - T0 if st["kind"] == "top" else T3[st["k"], st["j"], st["i"]] - T0
            an = ref.temperature_rise(st["xyz"], t)
            r = np.hypot(st["xyz"][:, 0] - xs, st["xyz"][:, 1] - y0)
            near = (r < NEAR_R) if on else np.zeros(r.size, bool)
            far = ~near
            scale = np.max(an[far])
            e = fv - an
            out.append(dict(case=case_id, t=t, set=name, zone="far",
                            rms=float(np.sqrt(np.mean(e[far] ** 2)) / scale),
                            max=float(np.max(np.abs(e[far])) / scale), scale=float(scale)))
            near_err.append(e[near])
            near_scale = max(near_scale, float(np.max(an)))
            profiles[(t, name)] = (st["xyz"], fv, an)
        ne = np.concatenate(near_err)
        if ne.size:
            out.append(dict(case=case_id, t=t, set="all", zone="near",
                            rms=float(np.sqrt(np.mean(ne ** 2)) / near_scale),
                            max=float(np.max(np.abs(ne)) / near_scale), scale=near_scale))
        print(f"  {case_id} t={t:4.1f}: " + "  ".join(
            f"{o['set']}/{o['zone']} rms={100 * o['rms']:.2f}% max={100 * o['max']:.2f}%"
            for o in out if o["t"] == t))
    return out, profiles, mesh


def main(cases):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for c in cases:
        results[c] = compare_case(c)
    rows = []
    for c, (out, _, _) in results.items():
        far = [o for o in out if o["zone"] == "far"]
        near = [o for o in out if o["zone"] == "near"]
        rms, mx = max(o["rms"] for o in far), max(o["max"] for o in far)
        rows += [
            dict(check="V4", case_id=c, metric="far_rms_rel_max_over_t", value=rms,
                 criterion="<= 0.01", **{"pass": bool(rms <= 0.01)},
                 note="r >= 5 mm from torch axis (all points after power-off); top, centre-plane and transverse sets"),
            dict(check="V4", case_id=c, metric="far_max_rel_max_over_t", value=mx,
                 criterion="<= 0.02", **{"pass": bool(mx <= 0.02)}),
            dict(check="V4", case_id=c, metric="near_rms_rel_max_over_t",
                 value=max(o["rms"] for o in near), criterion="info",
                 note="r < 5 mm while the source is on, relative to peak analytic rise"),
            dict(check="V4", case_id=c, metric="near_max_rel_max_over_t",
                 value=max(o["max"] for o in near), criterion="info"),
            dict(check="V4", case_id=c, metric="peak_rise_analytic_t20_K",
                 value=max(o["scale"] for o in near if o["t"] == 20.0), criterion="info"),
        ]
    record(rows)

    # figure: top centreline and transverse profiles at t = 10 and 20 s, finest case
    c = cases[-1]
    _, prof, mesh = results[c]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for t, col in ((10.0, "C0"), (20.0, "C1"), (30.0, "C2")):
        xyz, fv, an = prof[(t, "top")]
        sel = np.abs(xyz[:, 1] - xyz[np.argmin(np.abs(xyz[:, 1] - 0.05)), 1]) < 1e-9
        o = np.argsort(xyz[sel, 0])
        ax[0].plot(xyz[sel, 0][o] * 1e3, an[sel][o], "-", color=col, lw=1, label=f"exact t={t:g} s")
        ax[0].plot(xyz[sel, 0][o][::3] * 1e3, fv[sel][o][::3], "o", color=col, ms=2, label=f"FV t={t:g} s")
        xyz, fv, an = prof[(t, "trans")]
        top_k = xyz[:, 2] == xyz[:, 2].max()
        o = np.argsort(xyz[top_k, 1])
        ax[1].plot(xyz[top_k, 1][o] * 1e3, an[top_k][o], "-", color=col, lw=1)
        ax[1].plot(xyz[top_k, 1][o] * 1e3, fv[top_k][o], "o", color=col, ms=2)
        xyz, fv, an = prof[(t, "centre")]
        xi = xyz[np.argmin(np.abs(xyz[:, 0] - (xyz[:, 0][np.argmax(an)]))), 0]
        sel = np.abs(xyz[:, 0] - xi) < 1e-12
        ax[2].plot(an[sel], xyz[sel, 2] * 1e3, "-", color=col, lw=1)
        ax[2].plot(fv[sel], xyz[sel, 2] * 1e3, "o", color=col, ms=2)
    ax[0].set(xlabel="x (mm)", ylabel="T - T0 (K)", title=f"{c}: top surface near y = 50 mm")
    ax[0].legend(fontsize=6, ncol=2)
    ax[1].set(xlabel="y (mm)", ylabel="T - T0 (K)", title="top cell row, 2 mm behind torch")
    ax[2].set(xlabel="T - T0 (K)", ylabel="z (mm)", title="column at the hottest centre-plane x")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"V4_analytic_{c}.png", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1:] or ["V4_analytic_L1", "V4_analytic_L2"])
