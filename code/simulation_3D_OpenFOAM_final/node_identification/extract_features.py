#!/usr/bin/env python3
"""Top-view pool features for the NODE identification: what a camera above the
pool sees.

For every 0.1 s snapshot, the pool outline is the liquidus contour (1800 K) of
the top-surface face temperatures. It is found with contourpy, by linear
interpolation on the native face grid (0.35 mm). The features are measured
in the travel direction; the arc centre and the direction (+x or -x, for
back-and-forth passes; a pause keeps the previous pass's direction) are taken
from the schedule:

  area_mm2          area inside the outline
  length_mm         extent along the track
  width_mm          extent across the track
  tail_length_mm    from the arc centre back to the rear end of the pool
  perimeter_mm      length of the outline
  width_at_arc_mm   chord of the outline across the arc centre
  centroid_lag_mm   distance of the pool centroid behind the arc centre

Output: features/<case>.csv on the 0.1 s grid (NaN where there is no pool),
with the travel direction.

    python extract_features.py <case> [<case> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import contourpy
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from mp3d import RESULTS_DIR, RUN_ROOT  # noqa: E402
from mp3d.case_writer import load_config  # noqa: E402
from mp3d.foam_io import read_field, time_dirs  # noqa: E402
from mp3d.grid import StructuredMesh  # noqa: E402
from mp3d.schedule import ProcessSchedule  # noqa: E402

FEATURES = ["area_mm2", "length_mm", "width_mm", "tail_length_mm", "perimeter_mm", "width_at_arc_mm",
            "centroid_lag_mm"]
CROP_MM = (-30.0, 15.0, 12.0)          # window behind / ahead of the arc (travel direction), half-width across


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def top_window(mesh, time_dir, xa, ya, direction=1):
    """Top-face temperatures in the window around the arc, with x, y in mm."""
    xc, yc = mesh.xc * 1e3, mesh.yc * 1e3
    lo, hi = ((xa + CROP_MM[0], xa + CROP_MM[1]) if direction > 0 else (xa - CROP_MM[1], xa - CROP_MM[0]))
    ix = (xc > lo) & (xc < hi)
    iy = np.abs(yc - ya) < CROP_MM[2]
    T_top = mesh.patch_grid("top", read_field(Path(time_dir) / "T").boundary["top"]["value"])
    X, Y = np.meshgrid(xc[ix], yc[iy])
    return X, Y, T_top[np.ix_(iy, ix)]


def pool_outline(X, Y, T, level, xa, ya):
    """Closed liquidus outline around the arc (or the largest one); None if there is no pool."""
    if T.max() <= level:
        return None
    lines = contourpy.contour_generator(X, Y, T, line_type=contourpy.LineType.Separate).lines(level)
    closed = [p for p in lines if len(p) > 3 and np.allclose(p[0], p[-1])]
    if not closed:
        return None
    around_arc = [p for p in closed if MplPath(p).contains_point((xa, ya))]
    return max(around_arc or closed, key=_area)


def outline_features(p, xa, direction=1):
    x, y = p[:, 0], p[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    a_signed = 0.5 * cross.sum()
    cx = ((x[:-1] + x[1:]) * cross).sum() / (6.0 * a_signed)
    cy = ((y[:-1] + y[1:]) * cross).sum() / (6.0 * a_signed)
    ys = [y[i] + (xa - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i])
          for i in range(len(x) - 1) if (x[i] - xa) * (x[i + 1] - xa) <= 0 and x[i] != x[i + 1]]
    f = dict(area_mm2=abs(a_signed), length_mm=x.max() - x.min(), width_mm=y.max() - y.min(),
             tail_length_mm=(xa - x.min()) if direction > 0 else (x.max() - xa),
             perimeter_mm=float(np.hypot(np.diff(x), np.diff(y)).sum()),
             width_at_arc_mm=(max(ys) - min(ys)) if len(ys) >= 2 else 0.0,
             centroid_lag_mm=direction * (xa - cx))
    return f, (cx, cy), ((min(ys), max(ys)) if len(ys) >= 2 else None)


def _case_data(case):
    liquidus = json.loads((RESULTS_DIR / case / "mesh_meta.json").read_text())["material"]["liquidus"]
    s = pd.read_csv(RESULTS_DIR / case / "signals_0p1s.csv")
    s.index = np.round(s["t_s"], 3)
    sched = ProcessSchedule.from_config(load_config(RESULTS_DIR / case / "config_resolved.yaml"))
    return StructuredMesh(RUN_ROOT / case), liquidus, s, sched


def outline_at(case, t):
    """Outline (mm), arc centre, features, centroid and chord at the arc for one snapshot (for figures)."""
    mesh, liquidus, s, sched = _case_data(case)
    d = next(d for tt, d in time_dirs(RUN_ROOT / case) if abs(tt - t) < 1e-6)
    xa, ya = float(s.at[round(t, 3), "x_torch_mm"]), float(s.at[round(t, 3), "y_torch_mm"])
    direction = int(sched.direction(t))
    p = pool_outline(*top_window(mesh, d, xa, ya, direction), liquidus, xa, ya)
    f, centroid, chord = outline_features(p, xa, direction)
    return p, (xa, ya), f, centroid, chord


def process(case):
    mesh, liquidus, s, sched = _case_data(case)
    rows = []
    for t, tdir in time_dirs(RUN_ROOT / case):
        t = round(t, 3)
        if t not in s.index:
            continue
        xa, ya = float(s.at[t, "x_torch_mm"]), float(s.at[t, "y_torch_mm"])
        direction = int(sched.direction(t))
        p = pool_outline(*top_window(mesh, tdir, xa, ya, direction), liquidus, xa, ya)
        f = outline_features(p, xa, direction)[0] if p is not None else {k: np.nan for k in FEATURES}
        rows.append({"t_s": t, "x_arc_mm": xa, "travel_dir": direction, **f})
    df = pd.DataFrame(rows).sort_values("t_s")
    out = HERE / "features"
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"{case}.csv", index=False, float_format="%.5g")
    # consistency with the independent post-processing (bilinear on 0.05 mm)
    m = df.set_index("t_s").join(s[["top_area_liq_mm2", "top_length_liq_mm", "top_width_liq_mm"]], how="inner")
    m = m[m["top_area_liq_mm2"] > 5.0]
    dev = {k: float(np.median(np.abs(m[a] / m[b] - 1.0)))
           for k, a, b in (("area", "area_mm2", "top_area_liq_mm2"), ("length", "length_mm", "top_length_liq_mm"),
                           ("width", "width_mm", "top_width_liq_mm"))}
    print(f"{case}: {len(df)} frames, pool in {int(df['area_mm2'].notna().sum())}; median |rel. diff| vs "
          f"post-processing: " + ", ".join(f"{k} {100 * v:.2f} %" for k, v in dev.items()), flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        process(c)
