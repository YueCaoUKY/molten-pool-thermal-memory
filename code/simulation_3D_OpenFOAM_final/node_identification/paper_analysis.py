#!/usr/bin/env python3
"""Tables and figures for the paper set (PP_* runs).

  summary   one row per run: fit quality with the pool-driven memory state (two
            seeds), with the proposed δ and with no memory state, plus how each
            memory state follows the heat around the pool
  windows   designs 4 and 5: the pool at the four identical-input block ends
            (last 8 s of each 60 s block), simulation against each model
  figures   one panel figure per run: input, the seven states (simulation, with
            the pool-driven state, without any memory state) and the memory
            state against the heat

    python paper_analysis.py [--cases ...]
"""
import argparse
import json

import matplotlib
import numpy as np
import pandas as pd

import identify_3d as I
from delta_variants import ring_heat, score_one, window_for

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = I.HERE
INK, MUTED, GRID = "#1a1a19", "#6f6e69", "#e4e3dd"
C_POOL, C_NONE, C_SEED = "#2a78d6", "#eb6834", "#1baf7a"
BLOCK_ENDS = [((52.0, 60.0), "High 1"), ((112.0, 120.0), "Low 1"), ((172.0, 180.0), "High 2"),
              ((232.0, 240.0), "Low 2")]
UNITS = {"area_mm2": "area (mm²)", "length_mm": "length (mm)", "width_mm": "width (mm)",
         "tail_length_mm": "tail length (mm)", "perimeter_mm": "perimeter (mm)",
         "width_at_arc_mm": "width at arc (mm)", "centroid_lag_mm": "centroid lag (mm)"}


def cases_from_manifest():
    return [c for c in (HERE / "results" / "paper_cases.txt").read_text().split() if c]


def load(case, tag):
    d = dict(np.load(I.OUT / (case + tag) / "fit.npz", allow_pickle=True))
    m = json.loads((I.OUT / (case + tag) / "metrics.json").read_text())
    return d, m


def mean_in(t, y, w):
    k = (t >= w[0]) & (t < w[1]) & np.isfinite(y)
    return float(np.mean(y[k])) if k.any() else float("nan")


def summary(cases):
    rows = []
    for c in cases:
        try:
            _, m_pool = load(c, "_v_pool")
            _, m_s7 = load(c, "_v_pool_s7")
            _, m_base = load(c, "")
            _, m_none = load(c, "_nodelta")
        except FileNotFoundError as e:
            print(f"{c}: missing {e.filename}")
            continue
        sp, sb = score_one(c, "pool"), score_one(c, "base")
        rows.append(dict(case=c, material="304L" if "ss304" in c else "steel", design=c.split("_")[2],
                         window_s=list(window_for(c)),
                         r2_pool=round(m_pool["r2_global"], 3), r2_pool_seed7=round(m_s7["r2_global"], 3),
                         r2_base=round(m_base["r2_global"], 3), r2_none=round(m_none["r2_global"], 3),
                         nrmse_pool=round(m_pool["nrmse_std"], 3), nrmse_none=round(m_none["nrmse_std"], 3),
                         pool_spearman=round(sp["spearman"], 3), base_spearman=round(sb["spearman"], 3),
                         pool_drop=round(sp["drop_agree"], 3), base_drop=round(sb["drop_agree"], 3),
                         pool_trend=round(sp["trend_agree"], 3)))
    df = pd.DataFrame(rows)
    (I.OUT / "paper_summary.json").write_text(json.dumps(rows, indent=1) + "\n")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    if len(df):
        print("\nmeans by material:")
        print(df.groupby("material")[["r2_pool", "r2_base", "r2_none", "pool_spearman", "base_spearman",
                                      "pool_drop", "base_drop"]].mean().round(3).to_string())
    return df


def windows(cases):
    rows = []
    for c in cases:
        if "slow" not in c:
            continue
        try:
            d_pool, _ = load(c, "_v_pool")
            d_base, _ = load(c, "")
            d_none, _ = load(c, "_nodelta")
        except FileNotFoundError:
            continue
        t = d_pool["t"]
        heat = ring_heat(c, t)
        for w, label in BLOCK_ENDS:
            rows.append(dict(case=c, block=label, current_A=round(mean_in(t, d_pool["I_A"], w), 1),
                             heat_K=round(mean_in(t, heat, w)),
                             area_true=round(mean_in(t, d_pool["z_true"][:, 0], w), 1),
                             area_pool=round(mean_in(t, d_pool["z_pred"][:, 0], w), 1),
                             area_base=round(mean_in(t, d_base["z_pred"][:, 0], w), 1),
                             area_none=round(mean_in(t, d_none["z_pred"][:, 0], w), 1)))
    df = pd.DataFrame(rows)
    (I.OUT / "paper_windows_table.json").write_text(json.dumps(rows, indent=1) + "\n")
    if len(df):
        print("\nidentical-input block ends (designs 4 and 5):")
        print(df.to_string(index=False))
    return df


def case_figure(case):
    d_pool, m_pool = load(case, "_v_pool")
    d_none, _ = load(case, "_nodelta")
    t, names = d_pool["t"], [str(s) for s in d_pool["state_names"]]
    heat = ring_heat(case, t)
    fit = d_pool["fit"].astype(bool) if "fit" in d_pool else np.ones(len(t), bool)
    fig, axs = plt.subplots(5, 2, figsize=(11, 12), sharex=True)
    order = ["current", "speed", *names, "memory"]
    for ax, key in zip(axs.ravel(), order):
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        if key == "current":
            ax.step(t, d_pool["I_A"], where="post", color=INK, lw=1.4)
            ax.set_ylabel("current (A)", color=INK)
        elif key == "speed":
            ax.step(t, d_pool["speed_mm_s"], where="post", color=INK, lw=1.4)
            ax.set_ylabel("travel speed (mm/s)", color=INK)
        elif key == "memory":
            x = d_pool["delta_hat"][fit]
            A = np.c_[x, np.ones_like(x)]
            fitted = A @ np.linalg.lstsq(A, heat[fit], rcond=None)[0]
            ax.plot(t, heat, color=INK, lw=1.8, label="ΔT$_{ring}$, 5–15 mm from arc (FVM)")
            ax.plot(t[fit], fitted, color=C_POOL, lw=1.4,
                    label="δ, pool-coupled closure, Eq. (5), affine-mapped")
            ax.set_ylabel("temperature rise (K)", color=INK)
            ax.legend(fontsize=7.5, frameon=False, labelcolor=INK, loc="lower right")
        else:
            j = names.index(key)
            ax.plot(t, d_pool["z_true"][:, j], color=INK, lw=1.8, label="FVM simulation")
            ax.plot(t, d_pool["z_pred"][:, j], color=C_POOL, lw=1.4,
                    label="model with thermal state δ, Eq. (5)")
            ax.plot(t, d_none["z_pred"][:, j], color=C_NONE, lw=1.4, ls="--",
                    label="model without thermal state")
            ax.set_ylabel(UNITS.get(key, key), color=INK)
            if j == 0:
                ax.legend(fontsize=7.5, frameon=False, labelcolor=INK)
    for ax in axs[-1]:
        ax.set_xlabel("time (s)", color=INK)
    from variants_figure import LABEL as CASE_LABEL          # same run descriptions in both figures
    fig.suptitle(f"{CASE_LABEL.get(case, case)} — pool-coupled closure, Eq. (5): R² = {m_pool['r2_global']:.3f}",
                 color=INK, fontsize=11)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / f"{case}.png", dpi=110)
    plt.close(fig)
    return HERE / "figures" / f"{case}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    cases = a.cases or cases_from_manifest()
    summary(cases)
    windows(cases)
    if not a.no_figures:
        for c in cases:
            try:
                print(case_figure(c))
            except FileNotFoundError as e:
                print(f"{c}: no figure ({e.filename})")


if __name__ == "__main__":
    main()
