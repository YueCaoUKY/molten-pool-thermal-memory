#!/usr/bin/env pvbatch
"""Render the 3D temperature field and the liquid region of one case, frame by
frame, with ParaView 5.11 (pvbatch). Coordinates are shown in mm.

Three images per saved time t (tag = round(100 t), five digits):
  cut_<tag>.png    close-up quarter cutaway following the torch: the block
                   {x < x_torch - 1 mm, y < y0} is removed, exposing the
                   longitudinal section through the weld centreline (behind the
                   cut) and the transverse section through the pool;
                   colour = T, white line = liquidus, cyan line = solidus,
                   grey lines = 600 / 1000 / 1400 K isotherms
  pool_<tag>.png   the region above the liquidus as a closed 3D body (clip by
                   scalar), seen from below the plate; grids with 1 mm spacing
                   on the top face and 2 mm spacing on the bottom face
  plate_<tag>.png  the whole plate, surface temperature

The torch position and arc state come from results/<case>/signals_0p1s.csv.
The arc cone and electrode are schematic indicators, not the heat source.
The colour map is matplotlib inferno restricted to [INFERNO_START, 1] on
T_RANGE, so that a cold plate is dark blue rather than black; the compositing
script uses the same map.

    pvbatch code/visualize/render_pool3d.py <case_id> <frame_dir> [t_first t_last stride]
"""
import csv
import json
import sys
import time
from pathlib import Path

from paraview.simple import (Clip, Cone, Contour, CreateView, Cylinder, ExtractSurface, GenerateSurfaceNormals,
                             GetColorTransferFunction, MergeBlocks, OpenFOAMReader, Plane, SaveScreenshot, Show,
                             Transform)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mp3d import RESULTS_DIR, RUN_ROOT  # noqa: E402

CASE = sys.argv[1]
OUT = Path(sys.argv[2])
T_FIRST, T_LAST, STRIDE = (float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0.1, 1e9, 1)
SIZE = {"cut": [1240, 640], "pool": [660, 370], "plate": [660, 270]}
T_RANGE = (293.15, 2200.0)
INFERNO_START = 0.08
CUT_BEHIND_TORCH = 1.0                           # mm, position of the transverse section
BG = ([1.0, 1.0, 1.0], [0.80, 0.83, 0.87])     # top, bottom of the gradient

meta = json.loads((RESULTS_DIR / CASE / "mesh_meta.json").read_text())
TS, TL = meta["material"]["solidus"], meta["material"]["liquidus"]
LX, LY = (1e3 * meta[f"edges_{a}"][-1] for a in "xy")
Z_BOT = 1e3 * meta["edges_z"][0]
with open(RESULTS_DIR / CASE / "signals_0p1s.csv") as f:
    torch = {round(float(r["t_s"]), 3): (float(r["x_torch_mm"]), float(r["y_torch_mm"]), float(r["I_A"]) > 0)
             for r in csv.DictReader(f)}
Y0 = next(iter(torch.values()))[1]
X_TRACK = (min(v[0] for v in torch.values()), max(v[0] for v in torch.values()))

reader = OpenFOAMReader(FileName=str(RUN_ROOT / CASE / "system" / "controlDict"))
reader.MeshRegions = ["internalMesh"]
reader.CellArrays = ["T"]
reader.Createcelltopointfiltereddata = 1
reader.Cachemesh = 1
times = [t for t in reader.TimestepValues if T_FIRST - 1e-6 <= t <= T_LAST + 1e-6]
times = times[::STRIDE]
mm = Transform(Input=MergeBlocks(Input=reader))
mm.Transform.Scale = [1000.0, 1000.0, 1000.0]

lut = GetColorTransferFunction("T")
lut.ApplyPreset("Inferno (matplotlib)", True)
pts = list(lut.RGBPoints)
x0, x1 = pts[0], pts[-4]
rgb_points = []
for i in range(0, len(pts), 4):
    s = (pts[i] - x0) / (x1 - x0)
    if s >= INFERNO_START or i == len(pts) - 4:
        u = max(0.0, (s - INFERNO_START) / (1.0 - INFERNO_START))
        rgb_points += [T_RANGE[0] + u * (T_RANGE[1] - T_RANGE[0])] + pts[i + 1:i + 4]
rgb_points[0] = T_RANGE[0]
lut.RGBPoints = rgb_points
lut.AutomaticRescaleRangeMode = "Never"


def new_view(name):
    v = CreateView("RenderView")
    v.ViewSize = SIZE[name]
    v.OrientationAxesVisibility = 0
    v.UseColorPaletteForBackground = 0
    v.BackgroundColorMode = "Gradient"
    v.Background2, v.Background = BG
    v.UseFXAA = 1
    v.CameraViewUp = [0.0, 0.0, 1.0]
    return v


def show_T(src, view, specular=0.15):
    d = Show(src, view)
    d.Representation = "Surface"
    d.ColorArrayName = ["POINTS", "T"]
    d.LookupTable = lut
    d.Ambient, d.Diffuse = 0.35, 0.7
    d.Specular, d.SpecularPower = specular, 40.0
    return d


def show_solid(src, view, rgb, opacity=1.0, line_width=None, representation="Surface"):
    d = Show(src, view)
    d.Representation = representation
    d.ColorArrayName = ["POINTS", ""]
    d.DiffuseColor = d.AmbientColor = rgb
    d.Opacity = opacity
    if line_width:
        d.LineWidth = line_width
        d.RenderLinesAsTubes = 1
    return d


def grid_plane(z, x_range, y_half, spacing):
    nx, ny = round((x_range[1] - x_range[0]) / spacing), round(2 * y_half / spacing)
    return Plane(Origin=[x_range[0], Y0 - y_half, z], Point1=[x_range[1], Y0 - y_half, z],
                 Point2=[x_range[0], Y0 + y_half, z], XResolution=nx, YResolution=ny)


# --- arc and electrode (schematic) -----------------------------------------------------------
ARC_H, ARC_R = 7.0, 2.6
arc = Cone(Center=[0.0, Y0, 0.5 * ARC_H], Direction=[0.0, 0.0, 1.0], Height=ARC_H, Radius=ARC_R, Resolution=64)
arc.Capping = 0
electrode = Transform(Input=Cylinder(Radius=1.2, Height=14.0, Resolution=32))
electrode.Transform.Rotate = [90.0, 0.0, 0.0]
arc_big = Cone(Center=[0.0, Y0, 10.0], Direction=[0.0, 0.0, 1.0], Height=20.0, Radius=6.0, Resolution=48)

# --- view 1: close-up quarter cutaway ---------------------------------------------------------
v_cut = new_view("cut")
block = Clip(Input=mm, ClipType="Box")
block.ClipType.UseReferenceBounds = 0
block.ClipType.Length = [500.0, 500.0, 100.0]
block.Invert = 0                                  # keep everything outside the removed block
show_T(block, v_cut)
block_surf = ExtractSurface(Input=block)
for val, rgb, w in ((TL, [1.0, 1.0, 1.0], 3.0), (TS, [0.0, 0.85, 1.0], 2.0)):
    c = Contour(Input=block_surf, ContourBy=["POINTS", "T"], Isosurfaces=[val])
    c.ComputeScalars = 0
    show_solid(c, v_cut, rgb, line_width=w)
iso_lines = Contour(Input=block_surf, ContourBy=["POINTS", "T"], Isosurfaces=[600.0, 1000.0, 1400.0])
iso_lines.ComputeScalars = 0
show_solid(iso_lines, v_cut, [0.8, 0.8, 0.8], opacity=0.6, line_width=1.0)
arc_d1 = show_solid(arc, v_cut, [0.55, 0.72, 1.0], opacity=0.15)
el_d1 = show_solid(electrode, v_cut, [0.55, 0.55, 0.58])
el_d1.Specular = 0.6

# --- view 2: the liquid region as a 3D body ---------------------------------------------------
v_pool = new_view("pool")
liquid = Clip(Input=mm, ClipType="Scalar", Scalars=["POINTS", "T"], Value=TL)
liquid.Invert = 0                                 # keep T >= liquidus
liquid_surf = GenerateSurfaceNormals(Input=ExtractSurface(Input=liquid))
liquid_surf.FeatureAngle = 60.0
pool_d = show_T(liquid_surf, v_pool, specular=0.5)
x_grid = (X_TRACK[0] - 20.0, X_TRACK[1] + 20.0)
show_solid(grid_plane(0.0, x_grid, 8.0, 1.0), v_pool, [0.35, 0.37, 0.42], opacity=0.22, representation="Wireframe")
show_solid(grid_plane(Z_BOT, x_grid, 8.0, 2.0), v_pool, [0.35, 0.37, 0.42], opacity=0.15, representation="Wireframe")
arc_d2 = show_solid(arc, v_pool, [0.55, 0.72, 1.0], opacity=0.2)

# --- view 3: whole plate ----------------------------------------------------------------------
v_plate = new_view("plate")
show_T(mm, v_plate, specular=0.1)
arc_d3 = show_solid(arc_big, v_plate, [0.55, 0.72, 1.0], opacity=0.45)
v_plate.CameraFocalPoint = [0.5 * LX, 0.5 * LY, -3.0]
v_plate.CameraPosition = [0.5 * LX - 35.0, 0.5 * LY - 175.0, 125.0]
v_plate.CameraViewAngle = 28.0


def place_camera(view, focal, direction, dist, angle):
    n = sum(c * c for c in direction) ** 0.5
    view.CameraFocalPoint = focal
    view.CameraPosition = [f + dist * c / n for f, c in zip(focal, direction)]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraViewAngle = angle


OUT.mkdir(parents=True, exist_ok=True)
print(f"{CASE}: {len(times)} frames, liquidus {TL} K, solidus {TS} K, y0 {Y0} mm", flush=True)
for t in times:
    t0 = time.time()
    xs, ys, on = torch[round(t, 3)]
    for v in (v_cut, v_pool, v_plate):
        v.ViewTime = t
    xc = xs - CUT_BEHIND_TORCH
    block.ClipType.Position = [xc - 500.0, ys - 500.0, -50.0]
    arc.Center = [xs, ys, 0.5 * ARC_H]
    arc_big.Center = [xs, ys, 10.0]
    electrode.Transform.Translate = [xs, ys, ARC_H + 7.0]
    for d in (arc_d1, arc_d2, arc_d3):
        d.Visibility = int(on)
    place_camera(v_cut, [xc - 0.5, ys + 0.5, -1.8], [-0.8, -1.0, 0.58], 30.0, 30.0)
    place_camera(v_pool, [xs - 2.5, ys, -1.5], [-0.35, -1.0, -0.42], 29.0, 30.0)
    tag = f"{round(100 * t):05d}"
    for name, v in (("cut", v_cut), ("pool", v_pool), ("plate", v_plate)):
        SaveScreenshot(str(OUT / f"{name}_{tag}.png"), v, ImageResolution=SIZE[name])
    print(f"t = {t:6.2f} s  {time.time() - t0:5.2f} s", flush=True)
