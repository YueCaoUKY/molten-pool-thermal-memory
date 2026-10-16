#!/usr/bin/env python3
"""Package the checkpoint-3 cases as a learning/evaluation dataset (dataset/).

For every case: dataset/<case_id>.csv and .npz on the 0.1 s grid with
  inputs          t_s, I_A, V_V, u_mm_s, P_abs_nominal_W, x_torch_mm, y_torch_mm
  surface state   the five observed coordinates of plan §6
                  (log_top_area, top_length_liq_mm, top_width_liq_mm,
                  T_top_max_K, T_top_roi_mean_K) plus solidus-based variants
  withheld refs   delta_H_K, mean_T_rise_K, depth_liq_mm, depth_sol_mm,
                  vol_liquid_mm3, dHK_ring_5_15mm_K, dHK_lower_half_K,
                  T_bottom_under_torch_K (physical references; never used for
                  training, adaptation or checkpoint selection)
The top-surface temperature maps (0.5 mm grid, float32) stay in
results/<case_id>/top_surface_0p1s.npz and are listed in dataset/index.csv.

    python code/dataset/build_dataset.py [C3r|C3] [output folder]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mp3d import PROJECT_DIR, RESULTS_DIR  # noqa: E402
from mp3d.case_writer import load_config  # noqa: E402
from mp3d.schedule import ProcessSchedule  # noqa: E402

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "C3r"
OUT = PROJECT_DIR / (sys.argv[2] if len(sys.argv) > 2 else "dataset")
ROLES_BY_SUFFIX = {
    "S0": ("identification", "canonical schedule from 293.15 K; the only case used to identify the dynamic model"),
    "H0": ("holdout-initial-condition", "uniform 448.15 K preheat, S0 schedule"),
    "P0": ("holdout-thermal-history", "preheating pass (150 A, 4 mm/s) + 20 s pause, then the S0 input at 37-57 s"),
    "Iminus": ("holdout-power", "currents x0.85"),
    "Iplus": ("holdout-power", "currents x1.15"),
    "Vminus": ("holdout-speed", "speeds x0.75"),
    "Vplus": ("holdout-speed", "speeds x1.25"),
    "S0_k1": ("closure-sensitivity (not for learning)", "S0 with liquid k x1"),
    "P0_k1": ("closure-sensitivity (not for learning)", "P0 with liquid k x1"),
}
ROLES = {f"{PREFIX}_{k}": v for k, v in ROLES_BY_SUFFIX.items()}
INPUTS = ["t_s", "I_A", "V_V", "u_mm_s", "P_abs_nominal_W", "x_torch_mm", "y_torch_mm"]
SURFACE = ["log_top_area", "top_length_liq_mm", "top_width_liq_mm", "T_top_max_K", "T_top_roi_mean_K"]
SURFACE_EXTRA = ["top_area_liq_mm2", "top_area_sol_mm2", "top_length_sol_mm", "top_width_sol_mm",
                 "top_centroid_x_liq_mm", "top_centroid_y_liq_mm"]
WITHHELD = ["delta_H_K", "mean_T_rise_K", "depth_liq_mm", "depth_sol_mm", "vol_liquid_mm3",
            "dHK_ring_5_15mm_K", "dHK_lower_half_K", "T_bottom_under_torch_K"]
AREA_FLOOR_MM2 = 0.01   # log area uses max(area, floor); 0 area means no pool


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for case, (role, desc) in ROLES.items():
        f = RESULTS_DIR / case / "signals_0p1s.csv"
        if not f.exists():
            print(f"missing {case}")
            continue
        s = pd.read_csv(f)
        cfg = load_config(RESULTS_DIR / case / "config_resolved.yaml")
        sched = ProcessSchedule.from_config(cfg)
        s["log_top_area"] = np.log(np.maximum(s["top_area_liq_mm2"], AREA_FLOOR_MM2))
        cols = INPUTS + SURFACE + SURFACE_EXTRA + WITHHELD
        s[cols].to_csv(OUT / f"{case}.csv", index=False, float_format="%.6g")
        np.savez_compressed(OUT / f"{case}.npz", **{c: s[c].to_numpy() for c in cols})
        t_end = float(cfg["time"]["t_end"])
        man = json.loads((RESULTS_DIR / case / "run_manifest.json").read_text())
        rows.append(dict(case_id=case, role=role, description=desc, t_end_s=t_end,
                         t_heat_end_s=sched.t_heat_end, n_samples=len(s),
                         adapt_window_s=f"[0, {0.2 * t_end:g}]",
                         eval_segment_s=str(cfg["schedule"].get("eval_segment", "")),
                         liquid_k_multiplier=cfg["material"]["liquid_k_multiplier"],
                         T0_K=cfg["initial"]["T"],
                         absorbed_energy_J=round(sched.absorbed_energy(t_end), 1),
                         top_map_file=f"results/{case}/top_surface_0p1s.npz",
                         openfoam14=man["versions"]["openfoam14_package"],
                         additivefoam=man["versions"]["additivefoam_commit"][:10]))
    idx = pd.DataFrame(rows)
    idx.to_csv(OUT / "index.csv", index=False)
    s0 = pd.read_csv(OUT / f"{PREFIX}_S0.csv")
    ref = s0[(s0["I_A"] > 0) & (s0["top_area_liq_mm2"] > 1.0)]
    std = {"columns": SURFACE, "mean": ref[SURFACE].mean().to_dict(), "std": ref[SURFACE].std().to_dict(),
           "basis": f"{PREFIX}_S0 samples with the arc on and top melt area > 1 mm2"}
    (OUT / "surface_standardisation.json").write_text(json.dumps(std, indent=1))
    print(idx[["case_id", "role", "t_end_s", "n_samples", "absorbed_energy_J"]].to_string(index=False))


if __name__ == "__main__":
    main()
