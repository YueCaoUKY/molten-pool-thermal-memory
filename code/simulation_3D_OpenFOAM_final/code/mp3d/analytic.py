"""Exact reference solution for the constant-property, loss-free limit.

Temperature rise in an adiabatic plate x in [0, Lx], y in [0, Ly], z in [-H, 0]
(top surface z = 0) heated by the source used in this benchmark, AdditiveFOAM's
`projected` model: the planar Gaussian exp(-r^2 / (2 sigma^2)) multiplied by the
one-sided depth weight exp(-3 |z| / d), normalised to the absorbed power Q.
With constant rho, c and k (alpha = k / (rho c)) and no phase change,

    dT(x, t) = int_0^t Q(tau) / (rho c) * Gx * Gy * Gz  dtau,   s = t - tau,

where Gx and Gy are Gaussians of variance sigma^2 + 2 alpha s with mirror images
for the adiabatic side walls, and Gz is the adiabatic-slab Green's function
applied to the normalised depth weight a exp(-a zeta), a = 3 / d, zeta = -z.
Gz is evaluated by images with closed-form (erfcx) convolutions for small s and
by the cosine eigen-series for large s; `self_test` checks that the two forms
agree and that each factor integrates to one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import erfc, erfcx

_GL_X, _GL_W = leggauss(16)


@dataclass
class Segment:
    """Constant absorbed power Q (W) while the source moves linearly
    from (x0, y) at t0 to (x1, y) at t1."""
    t0: float
    t1: float
    Q: float
    x0: float
    x1: float
    y: float

    def x(self, tau):
        return self.x0 + (self.x1 - self.x0) * (tau - self.t0) / (self.t1 - self.t0)


class MovingSourcePlate:
    def __init__(self, *, Lx, Ly, H, sigma, depth, rho, cp, k, segments,
                 s_switch=0.02, n_images=1):
        self.Lx, self.Ly, self.H = Lx, Ly, H
        self.sigma, self.a = sigma, 3.0 / depth
        self.rho_c = rho * cp
        self.alpha = k / (rho * cp)
        self.segments = [s for s in segments if s.Q != 0.0]
        self.s_switch = s_switch
        self.images = range(-n_images, n_images + 1)

    # -- kernels -------------------------------------------------------------
    def gxy(self, u, us, S2, L):
        """Gaussian of variance S2 centred at us, reflected at 0 and L."""
        out = np.zeros_like(u)
        for m in self.images:
            out += np.exp(-(u - us - 2 * m * L) ** 2 / (2 * S2))
            out += np.exp(-(u + us - 2 * m * L) ** 2 / (2 * S2))
        return out / np.sqrt(2 * np.pi * S2)

    def _conv(self, u, D):
        """int_0^inf a exp(-a z') g_D(u - z') dz', g_D = heat kernel of variance 2D."""
        a = self.a
        rt = np.sqrt(D)
        v = (2 * a * D - u) / (2 * rt)
        out = np.empty_like(u)
        pos = v >= 0
        out[pos] = 0.5 * a * erfcx(v[pos]) * np.exp(-u[pos] ** 2 / (4 * D))
        neg = ~pos
        out[neg] = 0.5 * a * np.exp(a * a * D - a * u[neg]) * erfc(v[neg])
        return out

    def gz_images(self, zeta, s):
        D = self.alpha * s
        H = self.H
        out = np.zeros_like(zeta)
        for m in (-1, 0, 1):
            out += self._conv(zeta - 2 * m * H, D) + self._conv(2 * m * H - zeta, D)
        return out

    def gz_series(self, zeta, s):
        a, H = self.a, self.H
        nmax = int(np.ceil(H / np.pi * np.sqrt(60.0 / (self.alpha * s)))) + 1
        n = np.arange(1, nmax + 1)
        kn = n * np.pi / H
        cn = a * a * (1.0 - (-1.0) ** n * np.exp(-a * H)) / (a * a + kn ** 2)
        c0 = 1.0 - np.exp(-a * H)
        terms = (cn * np.exp(-self.alpha * kn ** 2 * s))[None, :] * np.cos(np.outer(zeta, kn))
        return (c0 + 2.0 * terms.sum(axis=1)) / H

    def gz(self, zeta, s):
        return self.gz_images(zeta, s) if s < self.s_switch else self.gz_series(zeta, s)

    # -- solution ------------------------------------------------------------
    @staticmethod
    def _s_edges(s_lo, s_hi):
        start = max(s_lo, 1e-6)
        if s_hi <= start:
            return np.array([s_lo, s_hi])
        n = max(2, int(np.ceil(8 * np.log10(s_hi / start))) + 1)
        edges = np.geomspace(start, s_hi, n)
        if s_lo < start:
            edges = np.concatenate([[s_lo], edges])
        return edges

    def temperature_rise(self, pts, t):
        """Temperature rise (K) at points pts (N, 3) [m], z <= 0, at time t [s]."""
        pts = np.asarray(pts, dtype=float)
        x, y, zeta = pts[:, 0], pts[:, 1], -pts[:, 2]
        dT = np.zeros(len(pts))
        for seg in self.segments:
            ta, tb = seg.t0, min(seg.t1, t)
            if tb <= ta:
                continue
            edges = self._s_edges(t - tb, t - ta)
            for sa, sb in zip(edges[:-1], edges[1:]):
                half = 0.5 * (sb - sa)
                for xi, wi in zip(_GL_X, _GL_W):
                    s = sa + half * (xi + 1.0)
                    tau = t - s
                    S2 = self.sigma ** 2 + 2 * self.alpha * s
                    g = (self.gxy(x, seg.x(tau), S2, self.Lx)
                         * self.gxy(y, seg.y, S2, self.Ly)
                         * self.gz(zeta, s))
                    dT += wi * half * seg.Q / self.rho_c * g
        return dT

    def absorbed_energy(self, t):
        return sum(s.Q * max(0.0, min(s.t1, t) - s.t0) for s in self.segments)

    # -- self-checks -----------------------------------------------------------
    def self_test(self):
        """Return a dict of checks of the kernels (all should be ~0)."""
        z = np.linspace(0.0, self.H, 20001)
        out = {}
        for s in (1e-4, self.s_switch, 1.0, 20.0):
            gz = self.gz_images(z, s) if s <= self.s_switch else self.gz_series(z, s)
            out[f"int_Gz_minus_1_s{s:g}"] = np.trapezoid(gz, z) - (1 - np.exp(-self.a * self.H))
        zz = np.linspace(0.0, self.H, 401)
        out["Gz_images_vs_series_rel_at_switch"] = float(
            np.max(np.abs(self.gz_images(zz, self.s_switch) - self.gz_series(zz, self.s_switch)))
            / np.max(self.gz_series(zz, self.s_switch)))
        xx = np.linspace(0.0, self.Lx, 40001)
        S2 = self.sigma ** 2 + 2 * self.alpha * 20.0
        out["int_Gx_minus_1_edge_source"] = np.trapezoid(self.gxy(xx, 0.010, S2, self.Lx), xx) - 1.0
        return out
