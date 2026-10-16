#!/usr/bin/env python3
"""Long-term prediction under a held-out power step test.

The absorbed power is stepped to a higher level and then returned to its initial level, so the
same input recurs after the workpiece has taken up heat.

All models were identified on low-carbon steel condition 4 with structures fixed before the new
simulation. Each is run recursively on the new record from the input sequence alone: the NODEs
and the LSTM start from the first measured state, ARX and the Transformer from the 16 and 64
measured states they require. Inputs are normalised and states standardised with the
identification record, as in adaptation. Evaluation starts at the first step.

    python step_test.py [--case PP_steel_6_step_ABA] [--variant poolb4] [--tag _p50]
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch

mpl.use("Agg")
import matplotlib.pyplot as plt

import baselines as B
import identify_3d as I
import paper_figures as PF
from delta_variants import VariantNODE3D, window_for

HERE = Path(__file__).resolve().parent
SRC = "PP_steel_4_slow_cycles"
VARIANT = "poolb4"
T_STEP, FINAL = 60.0, 10.0
# Finite-volume molten-region frames of the step-test record, rendered once for the paper; the
# strip shows them at uniformly spaced times with one common crop, so pool sizes are comparable.
FRAMES = HERE / "figures" / "pool_sequence_step_ABA"
STRIP_TIMES = (20, 50, 80, 110, 140, 170)
# The three comparison structures are named from the files that are loaded, so the legend always
# states the structure that was actually selected.
MODELS = [["Two-state NODE", "#c62828"], ["NODE without $\\delta$", "#ef8a17"],
          ["ARX", "#1565c0"], ["LSTM", "#2e7d32"], ["Transformer", "#6a1b9a"]]


def predict(case, tag=""):
    src = I.load_record(SRC, I.STATE_NAMES, window_for(SRC), False)
    p_ref, v_ref = float(src[4].max()), float(src[5].max())
    s, z_raw, _, _, p, v, _, _ = I.load_record(case, I.STATE_NAMES, window_for(case), False)
    t = s["t_s"].to_numpy()
    u_np = np.column_stack([p / p_ref, v / v_ref]).astype(np.float32)
    u = torch.tensor(u_np)
    out = {}

    def std(zm, zs):
        return torch.tensor(((z_raw - zm) / zs).astype(np.float32))

    with torch.no_grad():
        ck = torch.load(I.OUT / f"{SRC}_v_{VARIANT}_simpleD{tag}" / "checkpoint.pt", weights_only=False)
        m = VariantNODE3D(len(ck["z_mean"]), **ck["settings"])
        m.load_state_dict(ck["model"])
        zh, _ = m.rollout(std(ck["z_mean"], ck["z_std"])[0], u, I.DT)
        out[MODELS[0][0]] = zh.numpy() * ck["z_std"] + ck["z_mean"]

        ck = torch.load(I.OUT / f"{SRC}_nodelta{tag}" / "checkpoint.pt", weights_only=False)
        m = I.DissipativeNODE3D(z_dim=len(ck["z_mean"]), use_delta=False)
        m.load_state_dict(ck["model"])
        zh, _ = m.rollout(std(ck["z_mean"], ck["z_std"])[0], u, I.DT)
        out[MODELS[1][0]] = zh.numpy() * ck["z_std"] + ck["z_mean"]

        a = np.load(B.OUT / f"{SRC}_arx{tag}.npz")
        zn = ((z_raw - a["z_mean"]) / a["z_std"]).astype(np.float64)
        d = zn.shape[1]
        na, nb = int(a["na"]), int(a["nb"])
        MODELS[2][0] = f"ARX ($n_a$={na}, $n_b$={nb})"
        zh = B.arx_rollout(a["theta"], na, nb, zn, u_np, len(zn), np.ones(d), np.zeros(d))
        out[MODELS[2][0]] = zh * a["z_std"] + a["z_mean"]

        ck = torch.load(B.OUT / f"{SRC}_lstm{tag}.pt", weights_only=False)
        m = B.LSTMModel(len(ck["z_mean"]), hidden=ck["size"])
        m.load_state_dict(ck["model"])
        MODELS[3][0] = f"LSTM (width {int(ck['size'])})"
        zh = m.rollout(std(ck["z_mean"], ck["z_std"])[0], u)
        out[MODELS[3][0]] = zh.numpy() * ck["z_std"] + ck["z_mean"]

        ck = torch.load(B.OUT / f"{SRC}_transformer{tag}.pt", weights_only=False)
        m = B.TransformerModel(len(ck["z_mean"]), d_model=ck["size"], window=ck["window"])
        m.load_state_dict(ck["model"])
        MODELS[4][0] = f"Transformer (window {int(ck['window'])}, width {int(ck['size'])})"
        zh = m.rollout(std(ck["z_mean"], ck["z_std"]), u)
        out[MODELS[4][0]] = zh.numpy() * ck["z_std"] + ck["z_mean"]
    return t, p, z_raw, out


def evaluate(t, z_raw, out):
    mean, sd = z_raw.mean(axis=0), np.maximum(z_raw.std(axis=0), 1e-5)
    held = t >= T_STEP
    final = t >= t[-1] - FINAL
    o = (z_raw - mean) / sd
    rows, err = {}, {}
    for name, pred in out.items():
        q = (pred - mean) / sd
        e = np.sqrt(np.mean((q - o) ** 2, axis=1))
        err[name] = e
        ok = np.isfinite(q).all()
        rows[name] = dict(
            held_out_r2=float(I.metrics(o[held], q[held])["latent_r2_global"]) if ok else float("nan"),
            final_steady_state_error=float(np.sqrt(np.mean((q[final] - o[final]) ** 2))) if ok else float("nan"))
    return rows, err


def holds(t):
    """The two windows in which the input sits at its initial level: before the first step and at
    the end of the record. The pool is compared between them, because the input is the same."""
    return (t >= T_STEP - FINAL) & (t <= T_STEP), t >= t[-1] - FINAL


def load_frame(time_s):
    from PIL import Image
    image = np.asarray(Image.open(FRAMES / f"t_{time_s:03d}.0s.png").convert("RGB"))
    return image[55:640, 50:1365]          # identical crop for every time


def figure(t, p, z_raw, out, err, path, strip=False):
    PF.style()
    first, last = holds(t)
    area = z_raw[:, 0]
    reference = float(area[last].mean() - area[first].mean())

    if strip:
        fig = plt.figure(figsize=(7.16, 7.0))
        gs = fig.add_gridspec(4, 1, height_ratios=[0.5, 0.52, 1.25, 0.80], hspace=0.42)
        ax_p = fig.add_subplot(gs[0])
        sub = gs[1].subgridspec(1, len(STRIP_TIMES), wspace=0.035)
        for j, ts in enumerate(STRIP_TIMES):
            ax = fig.add_subplot(sub[0, j])
            ax.imshow(load_frame(ts), interpolation="none")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(rf"$t={ts}\,\mathrm{{s}}$", labelpad=1.0, fontsize=7.5)
            for spine in ax.spines.values():
                spine.set_color("0.55"); spine.set_linewidth(0.5)
            if j == 0:
                ax.set_title("(b) molten region from the finite-volume simulation", loc="left",
                             fontsize=8)
        axes = [ax_p, fig.add_subplot(gs[2]), fig.add_subplot(gs[3])]
        letters = ("(a)", "(c)", "(d)")
    else:
        fig, axes = plt.subplots(3, 1, figsize=(7.16, 6.2),
                                 gridspec_kw=dict(height_ratios=[0.5, 1.25, 0.80], hspace=0.52))
        letters = ("(a)", "(b)", "(c)")
    for ax in axes[:2]:
        for ts in (T_STEP, 2 * T_STEP):
            ax.axvline(ts, color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(t[0], t[-1])
        if strip:
            ax.set_xlim(5.0, 185.0)
            ax.set_xticks(STRIP_TIMES)

    axes[0].plot(t, p * 1e-3, color="0.15", lw=1.1)
    axes[0].set_ylabel("absorbed power\n(kW)")
    axes[0].set_title(f"{letters[0]} absorbed power, stepped up and returned to its initial level "
                      "at 0.5 mm s$^{-1}$", loc="left", fontsize=8)

    axes[1].plot(t, area, color="0.1", lw=1.6, label="finite-volume simulation", zorder=5)
    for name, c in MODELS:
        axes[1].plot(t, out[name][:, 0], color=c, lw=1.0, ls=(0, (4, 1.6)), label=name, zorder=6)
    lo, hi = np.nanmin(area), np.nanmax(area)
    axes[1].set_ylim(0.0, hi + 0.30 * (hi - lo))
    axes[1].set_ylabel("pool area (mm$^2$)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title(f"{letters[1]} pool-area prediction", loc="left", fontsize=8)
    # Arrows to the two instants at which the input is the same; the pool is not.
    for window, dx in ((first, -34), (last, -34)):
        tw, value = float(t[window].mean()), float(area[window].mean())
        axes[1].annotate(f"{value:.1f} mm$^2$", (tw, value), xytext=(dx, -30),
                         textcoords="offset points", ha="center", va="top", fontsize=8,
                         color="0.1", zorder=8,
                         arrowprops=dict(arrowstyle="->", color="0.1", lw=0.9,
                                         shrinkA=1.0, shrinkB=2.0))
    axes[1].legend(loc="upper left", ncol=3, fontsize=7, handlelength=2.4, frameon=False,
                   columnspacing=1.2, borderaxespad=0.2)

    # What the figure is for: at the two marked instants the input is identical, so a model is
    # judged by whether it is right at both. One bar per instant, solid for the first.
    ax = axes[2]
    xs = np.arange(len(MODELS))
    width = 0.34
    for k, (window, hatch, label) in enumerate(((first, "", "before the step"),
                                                (last, "////", "after the return"))):
        target = float(area[window].mean())
        values = [abs(float(out[n][window, 0].mean()) - target) for n, _ in MODELS]
        ax.bar(xs + (k - 0.5) * width, values, width, color=[c for _, c in MODELS],
               edgecolor="white", linewidth=0.7, hatch=hatch, zorder=3)
        for xi, v in zip(xs + (k - 0.5) * width, values):
            ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color="0.25")
    ax.set_xticks(xs)
    ax.set_xticklabels(["two-state\nNODE", "NODE\nwithout $\\delta$", "ARX", "LSTM",
                        "Transformer"], fontsize=8)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("pool-area error\n(mm$^2$)")
    ax.set_ylim(0, 11.2)
    ax.set_title(f"{letters[2]} pool-area error at the two marked times, where the input is the same",
                 loc="left", fontsize=8)
    ax.grid(axis="y", color="#dddddd", lw=0.45, zorder=0)
    ax.set_axisbelow(True)
    proxy = [mpl.patches.Patch(facecolor="0.7", edgecolor="white", hatch=h, label=l)
             for h, l in (("", "before the step"), ("////", "after the return"))]
    ax.legend(handles=proxy, loc="upper left", fontsize=7.5, frameon=False, ncol=2,
              borderaxespad=0.2, handlelength=1.6)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=0.6, length=3)
    axes[2].tick_params(axis="x", length=0)
    axes[2].set_xlim(-0.6, len(MODELS) - 0.4)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=400)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="PP_steel_6_step_ABA")
    ap.add_argument("--out", default="step_test")
    ap.add_argument("--tag", default="", help="suffix of the identified models to load, e.g. _p50")
    ap.add_argument("--variant", default=VARIANT, help="delta_variants structure to load")
    ap.add_argument("--strip", action="store_true",
                    help="add the finite-volume molten-region strip as panel (b)")
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw the figure from the saved <out>.npz without re-running any model")
    a = ap.parse_args()
    if a.figure_only:
        d = np.load(I.OUT / f"{a.out}.npz", allow_pickle=True)
        out = {str(n): d[f"pred_{i}"] for i, n in enumerate(d["model_names"])}
        for i, n in enumerate(d["model_names"]):       # legend names as saved with the results
            MODELS[i][0] = str(n)
        figure(d["t"], d["P_abs_W"], d["z_true"], out, None, HERE / "figures" / "paper" / a.out,
               strip=a.strip)
        return
    globals()["VARIANT"] = a.variant
    torch.set_num_threads(4)
    t, p, z_raw, out = predict(a.case, a.tag)
    rows, err = evaluate(t, z_raw, out)
    res = dict(case=a.case, source=SRC, source_tag=a.tag, first_step_s=T_STEP, final_window_s=FINAL,
               record_s=[float(t[0]), float(t[-1])], models=rows)
    (I.OUT / f"{a.out}.json").write_text(json.dumps(res, indent=1) + "\n")
    np.savez_compressed(I.OUT / f"{a.out}.npz", t=t, P_abs_W=p, z_true=z_raw,
                        **{f"pred_{i}": out[n] for i, (n, _) in enumerate(MODELS)},
                        model_names=np.array([n for n, _ in MODELS]))
    figure(t, p, z_raw, out, err, HERE / "figures" / "paper" / a.out, strip=a.strip)
    for name, r in rows.items():
        print(f"{name:40s} held-out R2 {r['held_out_r2']:9.3f}   final error {r['final_steady_state_error']:8.3f}")


if __name__ == "__main__":
    main()
