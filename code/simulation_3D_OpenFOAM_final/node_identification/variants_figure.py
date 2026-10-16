#!/usr/bin/env python3
"""Memory state against the heat around the pool: the proposed δ and one variant
(delta_variants.py), one panel per simulation. Each state is mapped onto the
heat by a least-squares scale and offset, for display only (a hidden state has
no unit or zero). Panel titles give the rank correlation and the share of heat
drops that the state follows.

    python variants_figure.py pool [--cases ...]
"""
import argparse
import json

import matplotlib
import numpy as np

from delta_variants import I, TAG, WINDOWS, ring_heat, score_one

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#1a1a19", "#6f6e69", "#e4e3dd"
C_NEW, C_BASE = "#2a78d6", "#eb6834"          # categorical slots 1-2, fixed order
MAT = {"steel": "Low-carbon steel", "ss304": "AISI 304L"}
COND = {"1_normal_var": "1.5 mm s⁻¹, multi-step current",
        "2_normal_const": "1.5 mm s⁻¹, constant current",
        "3_mid_var": "1.0 mm s⁻¹, multi-step current",
        "4_slow_cycles": "0.5 mm s⁻¹, cyclic current, free plate",
        "5_slow_cooled": "0.5 mm s⁻¹, cyclic current, cooled backing"}
LABEL = {f"PP_{m}_{c}": f"{MAT[m]}, {t}" for m in MAT for c, t in COND.items()}


def mapped(case, variant):
    d = np.load(I.OUT / (case + TAG[variant]) / "fit.npz", allow_pickle=True)
    t = d["t"]
    fit = d["fit"].astype(bool) if "fit" in d.files else np.ones(len(t), bool)
    x, y = d["delta_hat"][fit], ring_heat(case, t)[fit]
    A = np.c_[x, np.ones_like(x)]
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return t[fit], y, A @ coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--cases", nargs="+", default=list(WINDOWS))
    a = ap.parse_args()
    n = len(a.cases)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(11, 2.9 * rows), squeeze=False)
    summary = {}
    for ax, case in zip(axs.ravel(), a.cases):
        t, heat, base = mapped(case, "base")
        _, _, new = mapped(case, a.variant)
        sb, sn = score_one(case, "base"), score_one(case, a.variant)
        summary[case] = {"base": sb, a.variant: sn}
        ax.plot(t, heat, color=INK, lw=1.8,
                label="ΔT$_{ring}$, ring-averaged temperature rise 5–15 mm from arc (FVM reference)")
        ax.plot(t, base, color=C_BASE, lw=1.4, ls="--", label="δ, input-driven closure, Eq. (3)")
        ax.plot(t, new, color=C_NEW, lw=1.4, label="δ, pool-coupled closure, Eq. (5)")
        fall = ("not defined, ΔT$_{ring}$ monotonic" if np.isnan(sn["drop_agree"])
                else f"{100 * sb['drop_agree']:.0f} → {100 * sn['drop_agree']:.0f} %")
        ax.set_title(f"({chr(97 + list(a.cases).index(case))}) {LABEL.get(case, case)}\n"
                     f"ρ(δ, ΔT$_{{ring}}$) = {sb['spearman']:.2f} → {sn['spearman']:.2f};   "
                     f"falling-phase agreement S⁻: {fall}",
                     fontsize=8.5, color=INK, loc="left")
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_ylabel("temperature rise (K)", color=INK, fontsize=8)
    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel("time (s)", color=INK, fontsize=8.5)
    axs[0, 0].legend(fontsize=7.5, frameon=False, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    out = I.HERE / "figures" / f"variants_{a.variant}.png"
    fig.savefig(out, dpi=120)
    print(out)
    (I.OUT / f"variants_{a.variant}_figure_scores.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
