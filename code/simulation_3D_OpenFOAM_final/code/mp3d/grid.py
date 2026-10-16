"""Structured-grid view of a single-block case.

Cells are indexed [k, j, i] (z, y, x). The point ordering of the blockMesh
block gives the edge coordinates; the cell-centre field written by
`writeCellCentres` (0/C) is used to verify the internal ordering and to map
every boundary patch onto a 2D grid, so no face ordering is assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .foam_io import read_field, read_points

# patch -> (axis names of the 2D grid rows, cols)
PATCH_AXES = {"top": ("y", "x"), "bottom": ("y", "x"),
              "xmin": ("z", "y"), "xmax": ("z", "y"),
              "ymin": ("z", "x"), "ymax": ("z", "x")}


class StructuredMesh:
    def __init__(self, case_dir):
        case = Path(case_dir)
        meta = json.loads((case / "mesh_meta.json").read_text())
        self.meta = meta
        nx, ny, nz = meta["n"]
        self.nx, self.ny, self.nz = nx, ny, nz
        P = read_points(case)
        if len(P) != (nx + 1) * (ny + 1) * (nz + 1):
            raise ValueError("point count does not match a single structured block")
        self.xe = P[:nx + 1, 0].copy()
        self.ye = P[0:(nx + 1) * (ny + 1):nx + 1, 1].copy()
        self.ze = P[::(nx + 1) * (ny + 1), 2].copy()
        for name, a, b in (("x", self.xe, meta["edges_x"]), ("y", self.ye, meta["edges_y"]),
                           ("z", self.ze, meta["edges_z"])):
            if np.max(np.abs(a - np.asarray(b))) > 1e-9:
                raise ValueError(f"{name} edges differ from mesh_meta.json")
        self.xc = 0.5 * (self.xe[1:] + self.xe[:-1])
        self.yc = 0.5 * (self.ye[1:] + self.ye[:-1])
        self.zc = 0.5 * (self.ze[1:] + self.ze[:-1])
        self.dx, self.dy, self.dz = np.diff(self.xe), np.diff(self.ye), np.diff(self.ze)
        self.V = self.dz[:, None, None] * self.dy[None, :, None] * self.dx[None, None, :]
        self.H = self.ze[-1] - self.ze[0]

        C = read_field(case / "0" / "C")
        Cc = np.asarray(C.internal).reshape(nz, ny, nx, 3)
        err = max(np.max(np.abs(Cc[..., 0] - self.xc[None, None, :])),
                  np.max(np.abs(Cc[..., 1] - self.yc[None, :, None])),
                  np.max(np.abs(Cc[..., 2] - self.zc[:, None, None])))
        if err > 1e-9:
            raise ValueError(f"cell ordering is not [k, j, i] (max centre error {err:.3e} m)")

        centres = {"x": self.xc, "y": self.yc, "z": self.zc}
        widths = {"x": self.dx, "y": self.dy, "z": self.dz}
        self._pmap, self.patch_area = {}, {}
        for patch, (ra, ca) in PATCH_AXES.items():
            fc = np.asarray(C.boundary[patch]["value"])
            col = {"x": 0, "y": 1, "z": 2}
            r = np.searchsorted(centres[ra], fc[:, col[ra]])
            c = np.searchsorted(centres[ca], fc[:, col[ca]])
            r = np.clip(r, 0, len(centres[ra]) - 1)
            c = np.clip(c, 0, len(centres[ca]) - 1)
            # searchsorted gives the right neighbour; pick the nearest centre
            r -= (r > 0) & (np.abs(centres[ra][r - 1] - fc[:, col[ra]]) < np.abs(centres[ra][r] - fc[:, col[ra]]))
            c -= (c > 0) & (np.abs(centres[ca][c - 1] - fc[:, col[ca]]) < np.abs(centres[ca][c] - fc[:, col[ca]]))
            shape = (len(centres[ra]), len(centres[ca]))
            if len(np.unique(r * shape[1] + c)) != shape[0] * shape[1]:
                raise ValueError(f"patch {patch} faces do not tile a {shape} grid")
            self._pmap[patch] = (r, c, shape)
            self.patch_area[patch] = np.outer(widths[ra], widths[ca])

    # -- field access ----------------------------------------------------------
    def cells(self, internal) -> np.ndarray:
        a = np.asarray(internal)
        if a.ndim == 0:
            return np.full((self.nz, self.ny, self.nx), float(a))
        return a.reshape(self.nz, self.ny, self.nx)

    def patch_grid(self, patch: str, values) -> np.ndarray:
        r, c, shape = self._pmap[patch]
        out = np.empty(shape)
        v = np.asarray(values)
        out[r, c] = float(v) if v.ndim == 0 else v
        return out

    def extended_column_field(self, T3, T_top, T_bottom):
        """T on (z_ext, y, x) with bottom and top face values appended."""
        z_ext = np.concatenate([[self.ze[0]], self.zc, [self.ze[-1]]])
        T_ext = np.concatenate([T_bottom[None], T3, T_top[None]], axis=0)
        return z_ext, T_ext

    def interpolator(self, T3, T_top, T_bottom):
        z_ext, T_ext = self.extended_column_field(T3, T_top, T_bottom)
        return RegularGridInterpolator((z_ext, self.yc, self.xc), T_ext,
                                       bounds_error=False, fill_value=None)
