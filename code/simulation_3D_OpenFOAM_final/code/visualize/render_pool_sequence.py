#!/usr/bin/env pvbatch
"""The liquid region as a three-dimensional body, frame by frame, in the style of the case videos.

The camera follows the arc at a fixed offset, distance and angle, so every frame has the same
scale and pool sizes can be compared directly; the 1 mm (top face) and 2 mm (bottom face)
reference grids are fixed to the workpiece. One image per time, named by the time.

    pvbatch render_pool_sequence.py <case_id> <out_dir> <t_first> <t_last> <step>
"""
import csv
import json
import sys
from pathlib import Path

from paraview.simple import (Clip, CreateView, ExtractSurface, GenerateSurfaceNormals,
                             GetColorTransferFunction, MergeBlocks, OpenFOAMReader, Plane,
                             SaveScreenshot, Show, Transform)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mp3d import RESULTS_DIR, RUN_ROOT  # noqa: E402

CASE, OUT = sys.argv[1], Path(sys.argv[2])
T0, T1, STEP = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
T_RANGE, INFERNO_START = (293.15, 2200.0), 0.08
BG = ([1.0, 1.0, 1.0], [0.80, 0.83, 0.87])
SIZE = [2000, 1500]
BEHIND, DIST, DIRECTION = 0.8, 17.0, [-0.28, -1.0, 0.46]

meta = json.loads((RESULTS_DIR / CASE / "mesh_meta.json").read_text())
TL = meta["material"]["liquidus"]
Z_BOT = 1e3 * meta["edges_z"][0]
with open(RESULTS_DIR / CASE / "signals_0p1s.csv") as f:
    torch = {round(float(r["t_s"]), 3): (float(r["x_torch_mm"]), float(r["y_torch_mm"]))
             for r in csv.DictReader(f)}
x_start, y0 = torch[min(torch)]
x_end = torch[max(torch)][0]

reader = OpenFOAMReader(FileName=str(RUN_ROOT / CASE / "system" / "controlDict"))
reader.MeshRegions = ["internalMesh"]
reader.CellArrays = ["T"]
reader.Createcelltopointfiltereddata = 1
mm = Transform(Input=MergeBlocks(Input=reader))
mm.Transform.Scale = [1000.0, 1000.0, 1000.0]

lut = GetColorTransferFunction("T")
lut.ApplyPreset("Inferno (matplotlib)", True)
pts = list(lut.RGBPoints)
p0, p1 = pts[0], pts[-4]
rgb = []
for i in range(0, len(pts), 4):
    s = (pts[i] - p0) / (p1 - p0)
    if s >= INFERNO_START or i == len(pts) - 4:
        u = max(0.0, (s - INFERNO_START) / (1.0 - INFERNO_START))
        rgb += [T_RANGE[0] + u * (T_RANGE[1] - T_RANGE[0])] + pts[i + 1:i + 4]
rgb[0] = T_RANGE[0]
lut.RGBPoints = rgb
lut.AutomaticRescaleRangeMode = "Never"

v = CreateView("RenderView")
v.ViewSize = SIZE
v.OrientationAxesVisibility = 0
v.UseColorPaletteForBackground = 0
v.BackgroundColorMode = "Gradient"
v.Background2, v.Background = BG
v.UseFXAA = 1

liquid = Clip(Input=mm, ClipType="Scalar", Scalars=["POINTS", "T"], Value=TL)
liquid.Invert = 0
body = GenerateSurfaceNormals(Input=ExtractSurface(Input=liquid))
body.FeatureAngle = 60.0
d = Show(body, v)
d.Representation = "Surface"
d.ColorArrayName = ["POINTS", "T"]
d.LookupTable = lut
d.Ambient, d.Diffuse, d.Specular, d.SpecularPower = 0.35, 0.7, 0.5, 40.0

xa0, xa1 = x_start - 15.0, x_end + 15.0
for z, spacing, opacity in ((0.0, 1.0, 0.28), (Z_BOT, 2.0, 0.18)):
    g = Plane(Origin=[xa0, y0 - 12.0, z], Point1=[xa1, y0 - 12.0, z], Point2=[xa0, y0 + 12.0, z],
              XResolution=round((xa1 - xa0) / spacing), YResolution=round(24.0 / spacing))
    dg = Show(g, v)
    dg.Representation = "Wireframe"
    dg.ColorArrayName = ["POINTS", ""]
    dg.DiffuseColor = dg.AmbientColor = [0.35, 0.37, 0.42]
    dg.Opacity = opacity

n = sum(c * c for c in DIRECTION) ** 0.5
OUT.mkdir(parents=True, exist_ok=True)
t = T0
while t <= T1 + 1e-6:
    xa, ya = torch[round(t, 3)]
    v.ViewTime = t
    focal = [xa - BEHIND, ya, -0.9]
    v.CameraFocalPoint = focal
    v.CameraPosition = [f + DIST * c / n for f, c in zip(focal, DIRECTION)]
    v.CameraViewUp = [0.0, 0.0, 1.0]
    v.CameraViewAngle = 30.0
    SaveScreenshot(str(OUT / f"t_{t:05.1f}s.png"), v, ImageResolution=SIZE)
    print(f"t = {t:6.1f} s", flush=True)
    t = round(t + STEP, 6)
