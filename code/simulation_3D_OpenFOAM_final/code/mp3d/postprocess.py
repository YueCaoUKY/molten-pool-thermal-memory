"""Research signals every snapshot (0.1 s) from a finished case.

Outputs in results/<case_id>/:
  signals_0p1s.csv      one row per snapshot; columns defined in COLUMNS
  solver_steps.csv      one row per solver step, from log.additiveFoam
  probes_0p1s.npz       temperature histories at fixed probe points
  profiles_0p1s.npz     through-thickness centreline profiles at the torch and
                        trailing distances
  top_surface_0p1s.npz  top-surface temperature on a fixed 0.5 mm grid (float32)
  signals_columns.json  units and definitions of every CSV column

Thresholds: liquidus and solidus of the case material (1800 / 1750 K for
steel); both are reported. Energy uses the material model exactly as
AdditiveFOAM evaluates it (mp3d.materials.AFMaterial).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .case_writer import build_material, patch_specs
from .foam_io import parse_additivefoam_log, read_field, time_dirs
from .grid import StructuredMesh
from .materials import SIGMA_SB_AF
from .schedule import ProcessSchedule

ROI_HALF = 5.0e-3              # m, half-width of the torch-centred ROI square
RING = (5.0e-3, 15.0e-3)       # m, torch-following ring for the subsurface signal
WINDOW = (-25e-3, 10e-3, 10e-3)  # m, pool window: x from -25 to +10 mm, |y| <= 10 mm
WINDOW_DX = 5.0e-5             # m, bilinear resolution for pool metrics
TOP_GRID_DX = 5.0e-4           # m, fixed ML grid
PROFILE_OFFSETS = [0.0, 2.0e-3, 5.0e-3, 10.0e-3, 20.0e-3]   # m behind the torch
PROBE_X = [0.020, 0.040, 0.060]
PROBE_DY = [0.0, 2.5e-3, 5.0e-3, 10.0e-3]
PROBE_Z = [0.0, -1.0e-3, -3.0e-3, -6.0e-3]

COLUMNS = {
    "t_s": "time (s)",
    "I_A": "arc current on [t, t+dt) (A); 0 during cooldown",
    "V_V": "arc voltage (V)",
    "u_mm_s": "travel speed on [t, t+dt) (mm/s)",
    "P_incident_W": "V*I (W)",
    "P_abs_nominal_W": "eta*V*I (W)",
    "P_abs_logged_W": "AdditiveFOAM domain-integrated source averaged over (t-0.1 s, t] (W)",
    "x_torch_mm": "torch x (mm)", "y_torch_mm": "torch y (mm)",
    "T_top_max_K": "maximum top-surface (patch face) temperature (K)",
    "T_top_roi_mean_K": "area-weighted mean top temperature in a 10 mm x 10 mm square centred on the torch (K)",
    "T_max_3d_K": "maximum cell temperature (K)",
    "T_min_3d_K": "minimum over cells and boundary faces (K); must not fall below T_inf",
    "Q_rad_W": "radiative part of Q_loss_W (W)",
    "top_area_above_3000K_mm2": "top-surface area above 3000 K (mm^2); superheat the heat-only model cannot limit (steel boils near 3100 K)",
    "top_area_liq_mm2": "top-surface area with T > liquidus, bilinear on 0.05 mm (mm^2)",
    "top_length_liq_mm": "x extent of the top-surface liquidus region (mm)",
    "top_width_liq_mm": "y extent of the top-surface liquidus region (mm)",
    "top_centroid_x_liq_mm": "centroid x of the top-surface liquidus region (mm)",
    "top_centroid_y_liq_mm": "centroid y of the top-surface liquidus region (mm)",
    "top_area_sol_mm2": "as top_area_liq_mm2 for T > solidus",
    "top_length_sol_mm": "as top_length_liq_mm for T > solidus",
    "top_width_sol_mm": "as top_width_liq_mm for T > solidus",
    "depth_liq_mm": "maximum depth below the top of T >= liquidus, linear in z per column (mm)",
    "depth_sol_mm": "as depth_liq_mm for solidus",
    "vol_liquid_mm3": "sum of V*(1 - alpha.solid) (mm^3)",
    "vol_above_liq_mm3": "volume of cells with T > liquidus (mm^3)",
    "vol_above_sol_mm3": "volume of cells with T > solidus (mm^3)",
    "Tbar_max_K": "maximum of the thickness-averaged temperature (comparable to 2D peak_T)",
    "Tbar_area_Tm_mm2": "area where the thickness-averaged T > (solidus+liquidus)/2 (2D pool_area definition)",
    "Tbar_area_liq_mm2": "area where the thickness-averaged T > liquidus (mm^2)",
    "T_bottom_under_torch_K": "bottom-surface temperature below the torch (K)",
    "T_bottom_max_K": "maximum bottom-surface temperature (K)",
    "H_J": "enthalpy rise of the plate rel. solid at T_inf, sensible + latent (J)",
    "delta_H_K": "H / (rho cp_ref V_plate), cp_ref = model cp at T_inf (K)",
    "mean_T_rise_K": "volume-mean T - T_inf (comparable to 2D delta) (K)",
    "dHK_ring_5_15mm_K": "enthalpy rise per rho cp_ref V in the full-thickness ring 5-15 mm from the torch axis (K)",
    "dHK_lower_half_K": "enthalpy rise per rho cp_ref V of the lower half of the plate, z < -H/2 (K)",
    "E_src_J": "cumulative absorbed energy from the solver log (J)",
    "Q_loss_W": "convective + radiative loss rate over all patches (W)",
    "E_loss_J": "cumulative loss, trapezoidal on snapshots (J)",
    "residual_J": "H - H(0) - (E_src - E_loss) (J)",
    "residual_rel": "residual_J / total absorbed energy of the run",
    "alpha_consistency": "max |alpha.solid - alpha(T)| over cells (thermoPath consistency)",
}


def _extent(mask_axis_any, coords):
    idx = np.flatnonzero(mask_axis_any)
    if idx.size == 0:
        return 0.0
    return float(coords[idx[-1]] - coords[idx[0]] + (coords[1] - coords[0]))


class _Frame:
    """Signals from one snapshot."""

    def __init__(self, mesh: StructuredMesh, mat, sched, specs, T_inf, t, d):
        self.mesh, self.mat, self.t = mesh, mat, t
        Tf = read_field(d / "T")
        self.T = mesh.cells(Tf.internal)
        self.faces = {}
        for p in specs:
            v = Tf.boundary[p].get("value")
            if v is None:           # zeroGradient: face = adjacent cell
                v = self._adjacent(p)
            else:
                v = mesh.patch_grid(p, v)
            self.faces[p] = v
        a_path = d / "alpha.solid"
        if a_path.exists() or (d / "alpha.solid.gz").exists():
            self.alpha = mesh.cells(read_field(a_path).internal)
        else:
            self.alpha = mat.solid_fraction(self.T)

    def _adjacent(self, p):
        T = self.T
        return {"top": T[-1], "bottom": T[0], "xmin": T[:, :, 0], "xmax": T[:, :, -1],
                "ymin": T[:, 0, :], "ymax": T[:, -1, :]}[p]


def _pool_top(Ttop_interp, xs, ys, iso, window=WINDOW):
    x0, x1, yh = window
    gx = np.arange(xs + x0, xs + x1 + 1e-12, WINDOW_DX)
    gy = np.arange(ys - yh, ys + yh + 1e-12, WINDOW_DX)
    Y, X = np.meshgrid(gy, gx, indexing="ij")
    Tw = Ttop_interp(np.column_stack([Y.ravel(), X.ravel()])).reshape(Y.shape)
    out = {}
    for tag, T_iso in iso.items():
        m = Tw > T_iso
        if not m.any():
            out.update({f"top_area_{tag}_mm2": 0.0, f"top_length_{tag}_mm": 0.0,
                        f"top_width_{tag}_mm": 0.0, f"top_centroid_x_{tag}_mm": np.nan,
                        f"top_centroid_y_{tag}_mm": np.nan})
            continue
        out[f"top_area_{tag}_mm2"] = float(m.sum() * WINDOW_DX ** 2 * 1e6)
        out[f"top_length_{tag}_mm"] = _extent(m.any(axis=0), gx) * 1e3
        out[f"top_width_{tag}_mm"] = _extent(m.any(axis=1), gy) * 1e3
        out[f"top_centroid_x_{tag}_mm"] = float(X[m].mean() * 1e3)
        out[f"top_centroid_y_{tag}_mm"] = float(Y[m].mean() * 1e3)
        if m[:, 0].any() or m[:, -1].any() or m[0].any() or m[-1].any():
            out[f"window_clipped_{tag}"] = 1.0
    return out


def _depth(mesh, T3, Ttop, Tbot, iso):
    """Max depth of T >= iso connected to the top surface, per column, linear in z."""
    z = np.concatenate([[0.0], mesh.zc[::-1], [mesh.ze[0]]])       # top -> bottom
    col = np.concatenate([Ttop[None], T3[::-1], Tbot[None]], axis=0)
    hot = col >= iso
    if not hot[0].any():
        return 0.0
    first_cold = np.argmax(~hot, axis=0)                              # 0 if top is cold
    full = hot.all(axis=0)
    sel = hot[0] & ~full
    d = 0.0
    if sel.any():
        m = first_cold[sel]
        jj, ii = np.nonzero(sel)
        Ta, Tb = col[m - 1, jj, ii], col[m, jj, ii]
        za, zb = z[m - 1], z[m]
        zi = za + (iso - Ta) * (zb - za) / (Tb - Ta)
        d = float(np.max(-zi))
    if full.any():
        d = float(mesh.H)
    return d


def process_case(case_dir, cfg: dict, out_dir, frames: slice | None = None) -> pd.DataFrame:
    case, out = Path(case_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mesh = StructuredMesh(case)
    mat = build_material(cfg)
    sched = ProcessSchedule.from_config(cfg)
    specs = patch_specs(cfg)
    T_inf = float(cfg["boundary"]["T_inf"])
    rho = mat.rho
    cp_ref = float(mat.cp(T_inf))
    V = mesh.V
    V_tot = float(V.sum())
    iso = {"liq": mat.liquidus, "sol": mat.solidus}
    y0 = float(cfg["schedule"]["y0"])

    steps = parse_additivefoam_log(case / "log.additiveFoam")
    if len(steps):
        mid = steps["time"] - 0.5 * steps["dt"]
        steps["P_abs_nominal_W"] = sched.absorbed_power(mid.to_numpy())
        with np.errstate(invalid="ignore", divide="ignore"):
            steps["P_rel_dev"] = np.where(steps["P_abs_nominal_W"] > 0,
                                          steps["absorbed_power"] / steps["P_abs_nominal_W"] - 1.0, np.nan)
        steps["thermo_converged"] = steps["thermo_residual"] < float(cfg["solver"]["thermoTolerance"])
        steps["E_src_J"] = (steps["absorbed_power"] * steps["dt"]).cumsum()
        steps.to_csv(out / "solver_steps.csv", index=False)
    E_total = float(steps["E_src_J"].iloc[-1]) if len(steps) else sched.absorbed_energy(cfg["time"]["t_end"])

    snaps = time_dirs(case, "T")
    if frames is not None:
        snaps = snaps[frames]
    lower = (mesh.zc < mesh.ze[0] + 0.5 * mesh.H)[:, None, None]
    Xc, Yc = np.meshgrid(mesh.xc, mesh.yc, indexing="xy")          # [j, i]
    gx = np.arange(0.5 * TOP_GRID_DX, float(cfg["geometry"]["Lx"]), TOP_GRID_DX)
    gy = np.arange(0.5 * TOP_GRID_DX, float(cfg["geometry"]["Ly"]), TOP_GRID_DX)
    GY, GX = np.meshgrid(gy, gx, indexing="ij")
    probes = np.array([(x, y0 + dy, z) for x in PROBE_X for dy in PROBE_DY for z in PROBE_Z])
    z_prof = np.concatenate([[mesh.ze[0]], mesh.zc, [0.0]])
    rows, top_maps, probe_T, prof_T = [], [], [], []

    for t, d in snaps:
        f = _Frame(mesh, mat, sched, specs, T_inf, t, d)
        T3, Ttop, Tbot = f.T, f.faces["top"], f.faces["bottom"]
        xs = float(sched.position(t))
        d = int(sched.direction(t))                 # travel direction, +1 or -1 (back-and-forth passes)
        r = {"t_s": round(t, 10), "I_A": float(sched.current(t)), "V_V": float(cfg["schedule"]["voltage"]),
             "u_mm_s": float(sched.speed(t)) * 1e3, "P_incident_W": float(sched.incident_power(t)),
             "P_abs_nominal_W": float(sched.absorbed_power(t)),
             "x_torch_mm": xs * 1e3, "y_torch_mm": y0 * 1e3}
        if len(steps):
            sel = (steps["time"] > t - 0.1 + 1e-9) & (steps["time"] <= t + 1e-9)
            w = steps.loc[sel, "dt"].sum()
            r["P_abs_logged_W"] = float((steps.loc[sel, "absorbed_power"] * steps.loc[sel, "dt"]).sum() / w) if w > 0 else 0.0
            r["E_src_J"] = float(steps.loc[steps["time"] <= t + 1e-9, "E_src_J"].iloc[-1]) if (steps["time"] <= t + 1e-9).any() else 0.0
        top_int = RegularGridInterpolator((mesh.yc, mesh.xc), Ttop, bounds_error=False, fill_value=None)
        A_top = mesh.patch_area["top"]
        roi = (np.abs(Xc - xs) <= ROI_HALF) & (np.abs(Yc - y0) <= ROI_HALF)
        r["T_top_max_K"] = float(Ttop.max())
        r["T_top_roi_mean_K"] = float((Ttop * A_top)[roi].sum() / A_top[roi].sum())
        r["T_max_3d_K"] = float(T3.max())
        r["T_min_3d_K"] = float(min(T3.min(), min(v.min() for v in f.faces.values())))
        r.update(_pool_top(top_int, xs, y0, iso, WINDOW if d > 0 else (-WINDOW[1], -WINDOW[0], WINDOW[2])))
        r["depth_liq_mm"] = _depth(mesh, T3, Ttop, Tbot, mat.liquidus) * 1e3
        r["depth_sol_mm"] = _depth(mesh, T3, Ttop, Tbot, mat.solidus) * 1e3
        r["vol_liquid_mm3"] = float((V * (1.0 - f.alpha)).sum() * 1e9)
        r["vol_above_liq_mm3"] = float(V[T3 > mat.liquidus].sum() * 1e9)
        r["vol_above_sol_mm3"] = float(V[T3 > mat.solidus].sum() * 1e9)
        Tbar = (T3 * mesh.dz[:, None, None]).sum(axis=0) / mesh.H
        r["Tbar_max_K"] = float(Tbar.max())
        r["Tbar_area_Tm_mm2"] = float(A_top[Tbar > 0.5 * (mat.solidus + mat.liquidus)].sum() * 1e6)
        r["Tbar_area_liq_mm2"] = float(A_top[Tbar > mat.liquidus].sum() * 1e6)
        bot_int = RegularGridInterpolator((mesh.yc, mesh.xc), Tbot, bounds_error=False, fill_value=None)
        r["T_bottom_under_torch_K"] = float(bot_int([[y0, xs]])[0])
        r["T_bottom_max_K"] = float(Tbot.max())

        h = rho * mat.enthalpy(T3, f.alpha, T_ref=T_inf)                  # J/m^3
        H = float((h * V).sum())
        r["H_J"] = H
        r["delta_H_K"] = H / (rho * cp_ref * V_tot)
        r["mean_T_rise_K"] = float(((T3 - T_inf) * V).sum() / V_tot)
        rr = np.hypot(Xc - xs, Yc - y0)
        ring = ((rr >= RING[0]) & (rr <= RING[1]))[None, :, :]
        Vr = (V * ring).sum()
        r["dHK_ring_5_15mm_K"] = float((h * V * ring).sum() / (rho * cp_ref * Vr))
        r["dHK_lower_half_K"] = float((h * V * lower).sum() / (rho * cp_ref * (V * lower).sum()))
        Q = Qr = 0.0
        for p, spec in specs.items():
            if spec["type"] != "mixedTemperature":
                continue
            Tp = f.faces[p]
            qr = spec["emissivity"] * SIGMA_SB_AF * (Tp ** 4 - T_inf ** 4)
            q = spec["h"] * (Tp - T_inf) + qr
            Qp = float((q * mesh.patch_area[p]).sum())
            r[f"Q_loss_{p}_W"] = Qp
            Q += Qp
            Qr += float((qr * mesh.patch_area[p]).sum())
        r["Q_loss_W"] = Q
        r["Q_rad_W"] = Qr
        # superheat indicator: steel boils near 3100 K, which this heat-only model cannot represent
        r["top_area_above_3000K_mm2"] = float(A_top[Ttop > 3000.0].sum() * 1e6)
        r["alpha_consistency"] = float(np.max(np.abs(f.alpha - mat.solid_fraction(T3))))
        rows.append(r)

        interp = mesh.interpolator(T3, Ttop, Tbot)
        probe_T.append(interp(probes[:, [2, 1, 0]]))
        pts = np.array([(z, y0, xs - d * off) for off in PROFILE_OFFSETS for z in z_prof])
        prof_T.append(interp(pts).reshape(len(PROFILE_OFFSETS), len(z_prof)))
        top_maps.append(top_int(np.column_stack([GY.ravel(), GX.ravel()])).reshape(GY.shape).astype(np.float32))

    df = pd.DataFrame(rows)
    if len(df):
        H0 = df["H_J"].iloc[0]
        tt = df["t_s"].to_numpy()
        Ql = df["Q_loss_W"].to_numpy()
        df["E_loss_J"] = np.concatenate([[0.0], np.cumsum(0.5 * (Ql[1:] + Ql[:-1]) * np.diff(tt))])
        if "E_src_J" not in df:
            df["E_src_J"] = [sched.absorbed_energy(x) for x in tt]
        df["residual_J"] = df["H_J"] - H0 - (df["E_src_J"] - df["E_loss_J"])
        df["residual_rel"] = df["residual_J"] / E_total
    df.to_csv(out / "signals_0p1s.csv", index=False, float_format="%.10g")
    np.savez_compressed(out / "probes_0p1s.npz", t=df["t_s"].to_numpy(), xyz=probes,
                        T=np.array(probe_T))
    np.savez_compressed(out / "profiles_0p1s.npz", t=df["t_s"].to_numpy(), z=z_prof,
                        offsets_behind_torch=np.array(PROFILE_OFFSETS), y=y0, T=np.array(prof_T))
    if cfg.get("outputs", {}).get("top_map", True):
        np.savez_compressed(out / "top_surface_0p1s.npz", t=df["t_s"].to_numpy(), x=gx, y=gy,
                            T=np.array(top_maps))
    cols = dict(COLUMNS)
    cols.update({f"Q_loss_{p}_W": f"loss rate through patch {p} (W)" for p in specs})
    cols["window_clipped_liq"] = "1 if the liquidus region touched the pool-metric window edge"
    cols["window_clipped_sol"] = "1 if the solidus region touched the pool-metric window edge"
    meta = {"columns": cols, "T_inf": T_inf, "cp_ref": cp_ref, "rho": rho, "V_plate_m3": V_tot,
            "liquidus": mat.liquidus, "solidus": mat.solidus,
            "roi_half_width_m": ROI_HALF, "ring_m": RING, "pool_window_m": WINDOW,
            "pool_window_dx_m": WINDOW_DX, "top_grid_dx_m": TOP_GRID_DX,
            "profile_offsets_behind_torch_m": PROFILE_OFFSETS,
            "probes_xyz_m": probes.tolist(), "E_src_total_J": E_total}
    (out / "signals_columns.json").write_text(json.dumps(meta, indent=1))
    return df
