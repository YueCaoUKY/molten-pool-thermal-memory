#!/usr/bin/env pvbatch
"""Four schematic stills of the finite-volume experiment, in the style of the case videos.

  scene.png   the whole plate with its surface temperature and the arc
  pool.png    quarter cutaway following the arc: the pool section and the fusion lines
  states.png  the liquid region as a three-dimensional body over the reference grids,
              seen from above, with the pool-state definitions drawn as dimension lines
  delta.png   the cutaway without the fusion lines, with the two circles that bound the
              region whose stored heat defines the thermal reference

No text in any of them.

    pvbatch render_schematic.py <case_id> <time_s> <out_dir>
"""
import csv
import json
import sys
from pathlib import Path

from paraview.simple import (Clip, Cone, Contour, CreateView, Cylinder, ExtractSurface,
                             GenerateSurfaceNormals, GetColorTransferFunction, Line, MergeBlocks,
                             OpenFOAMReader, Plane, SaveScreenshot, Show, Slice, Transform)
from paraview import servermanager as sm
from vtk.util.numpy_support import vtk_to_numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mp3d import RESULTS_DIR, RUN_ROOT  # noqa: E402

CASE, T_SHOW, OUT = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])
T_RANGE, INFERNO_START = (293.15, 2200.0), 0.08
R_IN, R_OUT = 5.0, 15.0
RING_RGB = [0.24, 0.87, 0.78]
ARC_RGB = [0.55, 0.72, 1.0]
CUT_BEHIND_TORCH = 1.0
BG = ([1.0, 1.0, 1.0], [0.80, 0.83, 0.87])
SIZE = [2480, 1880]
DIM_RGB = {"length": [0.12, 0.50, 1.00], "width": [0.00, 0.77, 0.54],
           "width_at_arc": [1.00, 0.18, 0.60], "tail": [0.48, 0.36, 1.00],
           "lag": [0.00, 0.82, 1.00]}

meta = json.loads((RESULTS_DIR / CASE / "mesh_meta.json").read_text())
TS, TL = meta["material"]["solidus"], meta["material"]["liquidus"]
LX, LY = (1e3 * meta[f"edges_{a}"][-1] for a in "xy")
Z_BOT = 1e3 * meta["edges_z"][0]
with open(RESULTS_DIR / CASE / "signals_0p1s.csv") as f:
    torch = {round(float(r["t_s"]), 3): (float(r["x_torch_mm"]), float(r["y_torch_mm"]))
             for r in csv.DictReader(f)}
xa, ya = torch[round(T_SHOW, 3)]

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


def new_view():
    v = CreateView("RenderView")
    v.ViewSize = SIZE
    v.ViewTime = T_SHOW
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


def show_solid(src, view, colour, opacity=1.0, line_width=None, representation="Surface", flat=False):
    d = Show(src, view)
    d.Representation = representation
    d.ColorArrayName = ["POINTS", ""]
    d.DiffuseColor = d.AmbientColor = colour
    d.Opacity = opacity
    if line_width:
        d.LineWidth = line_width
        d.RenderLinesAsTubes = 1
    if flat:
        d.Ambient, d.Diffuse, d.Specular = 1.0, 0.0, 0.0
    return d


def place(view, focal, direction, dist, angle):
    n = sum(c * c for c in direction) ** 0.5
    view.CameraFocalPoint = focal
    view.CameraPosition = [f + dist * c / n for f, c in zip(focal, direction)]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraViewAngle = angle


def arc_cone(height=7.0, radius=2.6):
    c = Cone(Center=[xa, ya, 0.5 * height], Direction=[0.0, 0.0, 1.0], Height=height,
             Radius=radius, Resolution=64)
    c.Capping = 0
    return c


def cut_block():
    xc = xa - CUT_BEHIND_TORCH
    box = ([xc - 500.0, ya - 500.0, -50.0], [500.0, 500.0, 100.0])
    b = Clip(Input=mm, ClipType="Box")
    b.ClipType.UseReferenceBounds = 0
    b.ClipType.Position, b.ClipType.Length = box
    b.Invert = 0
    return b, box, xc


def flat_annulus(r_in, r_out, z, view, opacity, box):
    """A flat annulus lying on the top face, built by clipping a plane with two cylinders."""
    pl = Plane(Origin=[xa - r_out, ya - r_out, z], Point1=[xa + r_out, ya - r_out, z],
               Point2=[xa - r_out, ya + r_out, z], XResolution=300, YResolution=300)
    inner = Clip(Input=pl, ClipType="Cylinder")
    inner.ClipType.Center, inner.ClipType.Axis, inner.ClipType.Radius = [xa, ya, z], [0.0, 0.0, 1.0], r_in
    inner.Invert = 0
    outer = Clip(Input=inner, ClipType="Cylinder")
    outer.ClipType.Center, outer.ClipType.Axis, outer.ClipType.Radius = [xa, ya, z], [0.0, 0.0, 1.0], r_out
    outer.Invert = 1
    k = Clip(Input=outer, ClipType="Box")
    k.ClipType.UseReferenceBounds = 0
    k.ClipType.Position, k.ClipType.Length = box
    k.Invert = 0
    show_solid(k, view, RING_RGB, opacity=opacity, flat=True)


def pool_outline():
    """Points of the liquidus contour on the top surface, in mm."""
    sl = Slice(Input=mm, SliceType="Plane")
    sl.SliceType.Origin = [0.0, 0.0, -0.05]
    sl.SliceType.Normal = [0.0, 0.0, 1.0]
    c = Contour(Input=sl, ContourBy=["POINTS", "T"], Isosurfaces=[TL])
    c.ComputeScalars = 0
    data = sm.Fetch(c)
    if hasattr(data, "GetBlock"):
        data = data.GetBlock(0)
    p = vtk_to_numpy(data.GetPoints().GetData())
    near = ((p[:, 0] - xa) ** 2 + (p[:, 1] - ya) ** 2) ** 0.5 < 25.0
    return p[near]


def dim3d(view, a, b, off, colour, z=0.15, head=0.45, gap=0.35, ext=True):
    """A dimension line between a and b, offset by off, with two arrow heads."""
    a = [a[0] + off[0], a[1] + off[1], z]
    b = [b[0] + off[0], b[1] + off[1], z]
    d = [b[i] - a[i] for i in range(3)]
    n = sum(c * c for c in d) ** 0.5
    u = [c / n for c in d]
    inner_a = [a[i] + u[i] * head for i in range(3)]
    inner_b = [b[i] - u[i] * head for i in range(3)]
    show_solid(Line(Point1=inner_a, Point2=inner_b), view, colour, line_width=5.0, flat=True)
    for pt, sign in ((a, 1.0), (b, -1.0)):
        cone = Cone(Center=[pt[i] + sign * u[i] * head * 0.5 for i in range(3)],
                    Direction=[-sign * u[i] for i in range(3)], Height=head, Radius=0.20, Resolution=32)
        show_solid(cone, view, colour, flat=True)
    if ext and (off[0] or off[1]):
        e = [off[0], off[1], 0.0]
        m = sum(c * c for c in e) ** 0.5
        ue = [c / m for c in e]
        for pt in (a, b):
            s = [pt[i] - e[i] + ue[i] * gap for i in range(3)]
            t = [pt[i] + ue[i] * 0.7 for i in range(3)]
            show_solid(Line(Point1=s, Point2=t), view, colour, line_width=2.0, flat=True)


def marker(view, x, y, colour, size=0.55, ring=False):
    if ring:
        c = Cylinder(Radius=size, Height=0.02, Resolution=64)
        c.Capping = 0
        t = Transform(Input=c)
        t.Transform.Rotate = [90.0, 0.0, 0.0]
        t.Transform.Translate = [x, y, 0.18]
        show_solid(t, view, colour, line_width=4.0, flat=True)
    else:
        for d in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]):
            show_solid(Line(Point1=[x - size * d[0], y - size * d[1], 0.18],
                            Point2=[x + size * d[0], y + size * d[1], 0.18]),
                       view, colour, line_width=5.0, flat=True)


# --- 1. whole plate ----------------------------------------------------------------------------
v_scene = new_view()
show_T(mm, v_scene, specular=0.1)
show_solid(arc_cone(20.0, 6.0), v_scene, ARC_RGB, opacity=0.45)
place(v_scene, [0.5 * LX, 0.5 * LY, -3.0], [-0.20, -1.0, 0.50], 430.0, 26.0)

# --- 2. cutaway: pool section and fusion lines --------------------------------------------------
v_pool = new_view()
block, box, xc = cut_block()
show_T(block, v_pool)
surf = ExtractSurface(Input=block)
for val, colour, w in ((TL, [1.0, 1.0, 1.0], 4.0), (TS, [0.0, 0.85, 1.0], 2.5)):
    c = Contour(Input=surf, ContourBy=["POINTS", "T"], Isosurfaces=[val])
    c.ComputeScalars = 0
    show_solid(c, v_pool, colour, line_width=w)
iso = Contour(Input=surf, ContourBy=["POINTS", "T"], Isosurfaces=[600.0, 1000.0, 1400.0])
iso.ComputeScalars = 0
show_solid(iso, v_pool, [0.82, 0.82, 0.82], opacity=0.6, line_width=1.2)
show_solid(arc_cone(), v_pool, ARC_RGB, opacity=0.18)
place(v_pool, [xc - 0.5, ya + 0.5, -1.6], [-0.75, -1.0, 0.55], 34.0, 30.0)

# --- 3. the liquid region as a body, with the state definitions ---------------------------------
v_states = new_view()
liquid = Clip(Input=mm, ClipType="Scalar", Scalars=["POINTS", "T"], Value=TL)
liquid.Invert = 0
body = GenerateSurfaceNormals(Input=ExtractSurface(Input=liquid))
body.FeatureAngle = 60.0
show_T(body, v_states, specular=0.5)
for z, spacing, opacity in ((0.0, 1.0, 0.28), (Z_BOT, 2.0, 0.18)):
    nx, ny = round(40.0 / spacing), round(32.0 / spacing)
    g = Plane(Origin=[xa - 22.0, ya - 16.0, z], Point1=[xa + 18.0, ya - 16.0, z],
              Point2=[xa - 22.0, ya + 16.0, z], XResolution=nx, YResolution=ny)
    show_solid(g, v_states, [0.35, 0.37, 0.42], opacity=opacity, representation="Wireframe")

p = pool_outline()
x_min, x_max = float(p[:, 0].min()), float(p[:, 0].max())
y_min, y_max = float(p[:, 1].min()), float(p[:, 1].max())
cx, cy = float(p[:, 0].mean()), float(p[:, 1].mean())
chord = p[abs(p[:, 0] - xa) < 0.12]
y_lo, y_hi = (float(chord[:, 1].min()), float(chord[:, 1].max())) if len(chord) else (y_min, y_max)

dim3d(v_states, [x_min, y_min], [x_max, y_min], [0.0, -2.4], DIM_RGB["length"])
dim3d(v_states, [x_max, y_min], [x_max, y_max], [2.0, 0.0], DIM_RGB["width"])
dim3d(v_states, [xa, y_lo], [xa, y_hi], [0.0, 0.0], DIM_RGB["width_at_arc"], z=0.25, ext=False)
dim3d(v_states, [x_min, y_max], [xa, y_max], [0.0, 2.2], DIM_RGB["tail"])
dim3d(v_states, [cx, cy], [xa, cy], [0.0, -1.4], DIM_RGB["lag"], head=0.30)
marker(v_states, xa, ya, [1.0, 1.0, 1.0], size=1.1)
marker(v_states, cx, cy, [1.0, 1.0, 1.0], size=0.65, ring=True)
place(v_states, [cx + 0.6, cy - 0.3, -0.9], [-0.28, -1.0, 0.46], 27.0, 30.0)

# --- 4. the region behind the thermal reference -------------------------------------------------
v_delta = new_view()
block_d, box_d, _ = cut_block()
show_T(block_d, v_delta)
iso_d = Contour(Input=ExtractSurface(Input=block_d), ContourBy=["POINTS", "T"],
                Isosurfaces=[600.0, 1000.0, 1400.0])
iso_d.ComputeScalars = 0
show_solid(iso_d, v_delta, [0.85, 0.85, 0.85], opacity=0.35, line_width=1.0)
flat_annulus(R_IN, R_OUT, 0.05, v_delta, 0.16, box_d)
for r in (R_IN, R_OUT):
    flat_annulus(r - 0.20, r + 0.20, 0.10, v_delta, 1.0, box_d)

xc_d = xa - CUT_BEHIND_TORCH
half = 0.22
for r in (R_IN, R_OUT):                      # traces of the two cylinders on the cut faces
    y_face = ya - 0.08
    for x0 in (xa - r,):
        show_solid(Plane(Origin=[x0 - half, y_face, Z_BOT], Point1=[x0 + half, y_face, Z_BOT],
                         Point2=[x0 - half, y_face, 0.0]), v_delta, RING_RGB, flat=True)
    x_face = xc_d - 0.08
    for y0 in (ya - r,):
        show_solid(Plane(Origin=[x_face, y0 - half, Z_BOT], Point1=[x_face, y0 + half, Z_BOT],
                         Point2=[x_face, y0 - half, 0.0]), v_delta, RING_RGB, flat=True)
    band = 0.16                              # a faint wall between the two traces
    if r == R_OUT:
        show_solid(Plane(Origin=[xa - R_OUT, y_face + 0.01, Z_BOT],
                         Point1=[xa - R_IN, y_face + 0.01, Z_BOT],
                         Point2=[xa - R_OUT, y_face + 0.01, 0.0]), v_delta, RING_RGB,
                   opacity=band, flat=True)
        show_solid(Plane(Origin=[x_face + 0.01, ya - R_OUT, Z_BOT],
                         Point1=[x_face + 0.01, ya - R_IN, Z_BOT],
                         Point2=[x_face + 0.01, ya - R_OUT, 0.0]), v_delta, RING_RGB,
                   opacity=band, flat=True)
show_solid(arc_cone(), v_delta, ARC_RGB, opacity=0.18)
place(v_delta, [xa - 1.5, ya + 1.5, -1.0], [-0.55, -1.0, 0.70], 72.0, 30.0)

OUT.mkdir(parents=True, exist_ok=True)
for name, view in (("scene", v_scene), ("pool", v_pool), ("states", v_states), ("delta", v_delta)):
    SaveScreenshot(str(OUT / f"{name}.png"), view, ImageResolution=SIZE)
print(f"{CASE} t={T_SHOW}: arc at ({xa}, {ya}) mm; pool {x_max - x_min:.2f} x {y_max - y_min:.2f} mm")
