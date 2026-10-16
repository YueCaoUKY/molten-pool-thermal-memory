"""Material model for the 3D AdditiveFOAM benchmark.

The 2D FEniCSx baseline (`simulation_2D_FEM/code/materials.py`,
`low_carbon_steel`) uses piecewise-linear k(T) and cp(T) tables. AdditiveFOAM
2.0.0 accepts only quadratic polynomials per phase (`readTransportProperties.H`,
`Polynomial<3>`), evaluated at T clamped to [300 K, T_solidus] for the solid and
[T_liquidus, 2 T_liquidus] for the liquid, and mixes the two linearly in the
solid fraction (`updateProperties.H`). Decision D1 (2026-09-13): represent the
2D tables by fitted quadratics within the pinned code and report the deviation.

Fit used by `fit_steel`:
  cp_solid  least squares on the sensible enthalpy over [300 K, T_solidus],
            constrained so that the enthalpy at T_solidus equals the table
            value (the heat needed to reach the solidus is exact);
  k_solid   least squares on k over [300 K, T_solidus];
  liquid    the table's constant values above T_liquidus (820 J kg-1 K-1,
            35 W m-1 K-1). The Kou-Wang multiplier scales the liquid k only,
            as in the 2D model; mushy cells mix linearly in solid fraction in
            both codes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 2D baseline tables, copied from simulation_2D_FEM/code/materials.py.
STEEL_T = np.array([293, 500, 800, 1000, 1200, 1500, 1700, 1800, 2000], dtype=float)
STEEL_CP = np.array([460, 560, 700, 900, 700, 700, 800, 820, 820], dtype=float)
STEEL_K = np.array([60, 50, 35, 30, 30, 30, 33, 35, 35], dtype=float)
STEEL_RHO = 7400.0
STEEL_TS = 1750.0
STEEL_TL = 1800.0
STEEL_LF = 270e3

T_POLY_MIN = 300.0      # lower clamp of the solid polynomial (updateProperties.H)
SIGMA_SB_AF = 5.67e-8   # Stefan-Boltzmann constant hard-coded in mixedTemperature


def _poly(c, T):
    return c[0] + c[1] * T + c[2] * T * T


def _poly_int(c, T):
    return c[0] * T + c[1] * T ** 2 / 2.0 + c[2] * T ** 3 / 3.0


@dataclass
class AFMaterial:
    """A material exactly as AdditiveFOAM 2.0.0 evaluates it."""
    name: str
    rho: float
    solidus: float
    liquidus: float
    Lf: float
    k_solid: tuple
    cp_solid: tuple
    k_liquid: tuple
    cp_liquid: tuple
    emissivity: float = 0.6
    notes: str = ""

    def solid_fraction(self, T):
        """Solid fraction on the linear thermoPath ((solidus 1) (liquidus 0))."""
        T = np.asarray(T, dtype=float)
        return np.clip((self.liquidus - T) / (self.liquidus - self.solidus), 0.0, 1.0)

    def _mix(self, cs, cl, T, alpha):
        T = np.asarray(T, dtype=float)
        a = self.solid_fraction(T) if alpha is None else np.asarray(alpha, dtype=float)
        T1 = np.clip(T, T_POLY_MIN, self.solidus)
        T2 = np.clip(T, self.liquidus, 2.0 * self.liquidus)
        return a * _poly(cs, T1) + (1.0 - a) * _poly(cl, T2)

    def cp(self, T, alpha=None):
        return self._mix(self.cp_solid, self.cp_liquid, T, alpha)

    def k(self, T, alpha=None):
        return self._mix(self.k_solid, self.k_liquid, T, alpha)

    def _F(self, T):
        """Antiderivative of cp(T, alpha(T)) with F(300 K) = 0 (exact, piecewise)."""
        T = np.asarray(T, dtype=float)
        Ts, Tl, Tm = self.solidus, self.liquidus, 2.0 * self.liquidus
        cs, cl = self.cp_solid, self.cp_liquid
        A, B = _poly(cs, Ts), _poly(cl, Tl)
        F_Ts = _poly_int(cs, Ts) - _poly_int(cs, T_POLY_MIN)
        F_Tl = F_Ts + (Tl - Ts) * (A + B) / 2.0
        F_Tm = F_Tl + _poly_int(cl, Tm) - _poly_int(cl, Tl)
        cp_mushy = A + (B - A) * (T - Ts) / (Tl - Ts)
        return np.select(
            [T <= T_POLY_MIN, T <= Ts, T <= Tl, T <= Tm],
            [_poly(cs, T_POLY_MIN) * (T - T_POLY_MIN),
             _poly_int(cs, T) - _poly_int(cs, T_POLY_MIN),
             F_Ts + (T - Ts) * (A + cp_mushy) / 2.0,
             F_Tl + _poly_int(cl, T) - _poly_int(cl, Tl)],
            F_Tm + _poly(cl, Tm) * (T - Tm))

    def sensible_enthalpy(self, T, T_ref=293.15):
        """int_{T_ref}^{T} cp(T', alpha(T')) dT'  [J kg-1]."""
        return self._F(T) - self._F(T_ref)

    def enthalpy(self, T, alpha, T_ref=293.15):
        """Sensible plus latent enthalpy relative to solid at T_ref [J kg-1]."""
        return self.sensible_enthalpy(T, T_ref) + self.Lf * (1.0 - np.asarray(alpha, dtype=float))


def _steel_table_enthalpy(x):
    cp_tab = np.interp(x, STEEL_T, STEEL_CP)
    return np.concatenate([[0.0], np.cumsum(0.5 * (cp_tab[1:] + cp_tab[:-1]) * np.diff(x))])


def fit_steel(liquid_k_multiplier: float = 1.0) -> AFMaterial:
    """Quadratic representation of the 2D `low_carbon_steel` tables."""
    x = np.linspace(T_POLY_MIN, STEEL_TS, 5001)
    h_tab = _steel_table_enthalpy(x)
    B = np.vstack([x - T_POLY_MIN, (x ** 2 - T_POLY_MIN ** 2) / 2.0,
                   (x ** 3 - T_POLY_MIN ** 3) / 3.0]).T
    kkt = np.block([[2.0 * B.T @ B, B[-1:].T], [B[-1:], np.zeros((1, 1))]])
    rhs = np.concatenate([2.0 * B.T @ h_tab, [h_tab[-1]]])
    cp_s = tuple(float(v) for v in np.linalg.solve(kkt, rhs)[:3])
    V = np.vstack([np.ones_like(x), x, x ** 2]).T
    k_s = tuple(float(v) for v in np.linalg.lstsq(V, np.interp(x, STEEL_T, STEEL_K), rcond=None)[0])
    cp_l = (float(np.interp(STEEL_TL, STEEL_T, STEEL_CP)), 0.0, 0.0)
    k_l = (float(np.interp(STEEL_TL, STEEL_T, STEEL_K)) * liquid_k_multiplier, 0.0, 0.0)
    return AFMaterial(
        name=f"low_carbon_steel_quadratic_kl{liquid_k_multiplier:g}",
        rho=STEEL_RHO, solidus=STEEL_TS, liquidus=STEEL_TL, Lf=STEEL_LF,
        k_solid=k_s, cp_solid=cp_s, k_liquid=k_l, cp_liquid=cp_l, emissivity=0.6,
        notes=("Quadratic fit of the 2D pool_steel_lit tables: cp by constrained "
               "least squares on enthalpy (exact h at solidus), k by least squares; "
               f"liquid k multiplier {liquid_k_multiplier:g}."))


# AISI 304L (Kim recommends the same values for 304): C. S. Kim, "Thermophysical
# properties of stainless steels", ANL-75-55, Argonne National Laboratory, 1975.
CAL_PER_G = 4184.0                     # J/kg per cal/g (thermochemical calorie)
SS304_TS, SS304_TL = 1670.0, 1730.0    # melting range, Sec. II.A
SS304_LF = 64.0 * CAL_PER_G            # heat of fusion 64.0 cal/g, Sec. II.A
SS304_RHO = 7582.0                     # mean of Table 7 over 300-1700 K (AdditiveFOAM takes one density)


def ss304_kim(liquid_k_multiplier: float = 1.0) -> AFMaterial:
    """AISI 304L from Kim (1975). All correlations are linear or constant in T,
    so AdditiveFOAM represents them exactly (no fit):
      cp_solid  = 0.1122 + 3.222e-5 T  cal/(g K)     eq. (5)
      cp_liquid = 0.1900 cal/(g K)                    Sec. II.A
      k_solid   = 8.116e-2 + 1.618e-4 T  W/(cm K)     eq. (28)
      k_liquid  = 1.229e-1 + 3.248e-5 T  W/(cm K)     eq. (29), times the Kou-Wang multiplier
    """
    m = float(liquid_k_multiplier)
    return AFMaterial(
        name=f"ss304L_kim1975_kl{m:g}", rho=SS304_RHO, solidus=SS304_TS, liquidus=SS304_TL, Lf=SS304_LF,
        k_solid=(8.116, 1.618e-2, 0.0), cp_solid=(0.1122 * CAL_PER_G, 3.222e-5 * CAL_PER_G, 0.0),
        k_liquid=(12.29 * m, 3.248e-3 * m, 0.0), cp_liquid=(0.1900 * CAL_PER_G, 0.0, 0.0), emissivity=0.6,
        notes=f"AISI 304L, Kim (1975) ANL-75-55 eqs. (5), (28), (29), Table 7 mean density; liquid k multiplier {m:g}.")


def constant_material(name: str, rho: float, cp: float, k: float, *,
                      Lf: float = 0.0, solidus: float = 1.0e6, liquidus: float = 1.1e6,
                      emissivity: float = 0.0) -> AFMaterial:
    """Constant properties; the default phase-change range is out of reach."""
    return AFMaterial(name=name, rho=rho, solidus=solidus, liquidus=liquidus, Lf=Lf,
                      k_solid=(k, 0.0, 0.0), cp_solid=(cp, 0.0, 0.0),
                      k_liquid=(k, 0.0, 0.0), cp_liquid=(cp, 0.0, 0.0),
                      emissivity=emissivity, notes="constant-property test material")


def fit_report(mat: AFMaterial) -> dict:
    """Deviation of the fitted solid properties from the 2D tables."""
    x = np.linspace(T_POLY_MIN, STEEL_TS, 5001)
    cp_err = mat.cp(x) - np.interp(x, STEEL_T, STEEL_CP)
    k_err = mat.k(x) - np.interp(x, STEEL_T, STEEL_K)
    h_tab = _steel_table_enthalpy(x)
    h_fit = mat.sensible_enthalpy(x, T_ref=T_POLY_MIN)
    i_cp = int(np.argmax(np.abs(cp_err)))
    i_h = int(np.argmax(np.abs(h_fit - h_tab)))
    return {
        "cp_solid_coeffs": mat.cp_solid, "k_solid_coeffs": mat.k_solid,
        "cp_liquid": mat.cp_liquid[0], "k_liquid": mat.k_liquid[0],
        "cp_max_abs_err": float(abs(cp_err[i_cp])), "cp_max_err_at_T": float(x[i_cp]),
        "cp_at_solidus_fit": float(mat.cp(STEEL_TS - 1e-9)),
        "cp_at_solidus_table": float(np.interp(STEEL_TS, STEEL_T, STEEL_CP)),
        "k_max_abs_err": float(np.max(np.abs(k_err))),
        "h_max_abs_err_kJkg": float(abs(h_fit[i_h] - h_tab[i_h]) / 1e3),
        "h_max_err_at_T": float(x[i_h]),
        "h_err_at_solidus_kJkg": float((h_fit[-1] - h_tab[-1]) / 1e3),
        "h_solidus_table_kJkg": float(h_tab[-1] / 1e3),
    }
