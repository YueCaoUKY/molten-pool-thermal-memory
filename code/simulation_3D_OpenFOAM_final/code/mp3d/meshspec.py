"""Single-block structured hex mesh with a refined corridor along the track.

One blockMesh block keeps the global i-j-k cell ordering (x fastest, then y,
then z), so post-processing can reshape fields without searching. Refinement is
by multi-section grading: uniform cells of size h in the corridor and geometric
growth (ratio <= q_max) outside it, sized so that the cells adjacent to the
corridor equal h. blockMesh rounds each section to label(frac*N + 0.5) cells and
places geometric sections with factor ratio**(1/(n-1)) (lineDivide.C); the
fractions written here reproduce the intended counts exactly, and `edges`
returns the resulting node coordinates for checking against the mesh.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Section:
    length: float
    n: int
    ratio: float      # last/first cell size within the section (blockMesh convention)


def _geometric_section(L: float, h: float, q_max: float, small_end_first: bool) -> Section:
    """n cells over length L whose smallest cell (at one end) is exactly h."""
    if L <= h * 1.000001:
        return Section(L, 1, 1.0)
    n = int(np.ceil(np.log(1.0 + L * (q_max - 1.0) / h) / np.log(q_max)))
    n = max(n, 2)
    if n * h >= L:                                   # uniform fits: no growth needed
        n = max(1, int(round(L / h)))
        return Section(L, n, 1.0)
    lo, hi = 1.0 + 1e-12, q_max                      # solve h (q^n - 1)/(q - 1) = L for q
    for _ in range(200):
        q = 0.5 * (lo + hi)
        if h * (q ** n - 1.0) / (q - 1.0) > L:
            hi = q
        else:
            lo = q
    q = 0.5 * (lo + hi)
    ratio = q ** (n - 1)
    return Section(L, n, ratio if small_end_first else 1.0 / ratio)


@dataclass
class Axis:
    sections: list

    @property
    def n(self) -> int:
        return sum(s.n for s in self.sections)

    @property
    def length(self) -> float:
        return sum(s.length for s in self.sections)

    def grading(self) -> str:
        if len(self.sections) == 1:
            return repr(self.sections[0].ratio)
        L, N = self.length, self.n
        parts = [f"({s.length / L!r} {s.n / N!r} {s.ratio!r})" for s in self.sections]
        return "(" + " ".join(parts) + ")"

    def edges(self, origin: float = 0.0) -> np.ndarray:
        """Node coordinates exactly as blockMesh's lineDivide places them."""
        pts = [origin]
        start = origin
        for s in self.sections:
            if s.ratio == 1.0 or s.n == 1:
                d = start + s.length * np.arange(1, s.n + 1) / s.n
            else:
                e = s.ratio ** (1.0 / (s.n - 1))
                d = start + s.length * (1.0 - e ** np.arange(1, s.n + 1)) / (1.0 - e ** s.n)
            pts.extend(d)
            start += s.length
        return np.array(pts)


def uniform_axis(L: float, n: int) -> Axis:
    return Axis([Section(L, n, 1.0)])


def plate_axes(*, Lx, Ly, H, y_track, h, hz, x_fine_end, y_half, z_fine_depth,
               q_xy=1.1, q_z=1.2):
    """Corridor x in [0, ~x_fine_end], |y - y_track| <= ~y_half, top ~z_fine_depth."""
    nxf = int(round(x_fine_end / h))
    ax_x = Axis([Section(nxf * h, nxf, 1.0),
                 _geometric_section(Lx - nxf * h, h, q_xy, small_end_first=True)])
    m = int(round(y_half / h))
    y_lo, y_hi = y_track - m * h, y_track + m * h
    ax_y = Axis([_geometric_section(y_lo, h, q_xy, small_end_first=False),
                 Section(2 * m * h, 2 * m, 1.0),
                 _geometric_section(Ly - y_hi, h, q_xy, small_end_first=True)])
    nzf = int(round(z_fine_depth / hz))
    ax_z = Axis([_geometric_section(H - nzf * hz, hz, q_z, small_end_first=False),
                 Section(nzf * hz, nzf, 1.0)])
    return ax_x, ax_y, ax_z


PATCH_FACES = {
    "bottom": "(0 3 2 1)",
    "top": "(4 5 6 7)",
    "xmin": "(0 4 7 3)",
    "xmax": "(1 2 6 5)",
    "ymin": "(0 1 5 4)",
    "ymax": "(3 7 6 2)",
}


def block_mesh_dict(Lx, Ly, H, ax_x: Axis, ax_y: Axis, ax_z: Axis) -> str:
    v = [(0, 0, -H), (Lx, 0, -H), (Lx, Ly, -H), (0, Ly, -H),
         (0, 0, 0), (Lx, 0, 0), (Lx, Ly, 0), (0, Ly, 0)]
    verts = "\n".join(f"    ({a!r} {b!r} {c!r})" for a, b, c in v)
    patches = "\n".join(
        f"    {name}\n    {{\n        type wall;\n        faces ({faces});\n    }}"
        for name, faces in PATCH_FACES.items())
    return f"""vertices
(
{verts}
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({ax_x.n} {ax_y.n} {ax_z.n})
    simpleGrading
    (
        {ax_x.grading()}
        {ax_y.grading()}
        {ax_z.grading()}
    )
);

boundary
(
{patches}
);
"""
