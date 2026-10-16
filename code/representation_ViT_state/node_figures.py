#!/usr/bin/env python3
"""Trajectory figures for the dynamic model on the visual state (GTAW)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
import node_visual as _N
import visual_generator as _V

RES, FIG = HERE / "results_node", HERE / "figures"
DT = 0.1


def configure(process: str) -> None:
    """Point the figures at one of the processes."""
    global RES, DT
    _N.configure(process)
    _V.configure(process)
    RES, DT = _N.OUT, _N.DT
INK, BLUE, RED, GREEN, SOFT = "#272727", "#1F5C9E", "#B64342", "#3F8F58", "#7A8189"

plt.rcParams.update({
    "font.family": ["Nimbus Sans", "DejaVu Sans", "sans-serif"],
    "font.size": 9.5,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic",
    "mathtext.bf": "Nimbus Sans:bold",
    "svg.fonttype": "none",
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", FIG / f"{name}.png")


def identification(sid="Iden_08", tag=""):
    d = np.load(RES / f"fit_{sid}{tag}.npz")
    zt, zp, dl, v = d["z_true"], d["z_pred"], d["delta"], d["speed"]
    raw = d["z_raw"] if "z_raw" in d.files else None
    t = np.arange(len(zt)) * DT
    fig = plt.figure(figsize=(9.6, 7.6))
    gs = fig.add_gridspec(5, 2, height_ratios=[0.8, 1, 1, 1, 0.9], hspace=0.42, wspace=0.22,
                          top=0.88)

    ax = fig.add_subplot(gs[0, :])
    ax.step(t, v, where="post", color=INK, lw=1.1)
    ax.set_ylabel("speed\n(mm s$^{-1}$)", linespacing=1.3)
    ax.set_xlim(t[0], t[-1])
    ax.set_xticklabels([])

    for k in range(6):
        ax = fig.add_subplot(gs[1 + k // 2, k % 2])
        if raw is not None:
            ax.plot(t, raw[:, k], color="#C9CDD1", lw=0.5, label="as read")
        ax.plot(t, zt[:, k], color=INK, lw=0.8, alpha=0.85,
                label="filtered state" if raw is not None else "from the images")
        ax.plot(t, zp[:, k], color=BLUE, lw=1.5, label="predicted from the input")
        ax.set_ylabel(f"$z_{k+1}$")
        ax.set_xlim(t[0], t[-1])
        if k < 4:
            ax.set_xticklabels([])
        if k == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, ncol=2, fontsize=9, loc="upper center",
                       bbox_to_anchor=(0.5, 0.925))

    ax = fig.add_subplot(gs[4, :])
    ax.plot(t, dl, color=RED, lw=1.5)
    ax.fill_between(t, 0, dl, color=RED, alpha=0.12)
    ax.set_ylabel("$\\delta$")
    ax.set_xlabel("time (s)")
    ax.set_xlim(t[0], t[-1])
    kind = "filtered state, 2 s moving average" if tag else "state as read"
    fig.suptitle(f"Identification on {sid} ({kind}): the whole record predicted from the "
                 "welding speed and the first state", y=0.975, fontsize=11)
    save(fig, f"identification_{sid}{tag}")


def adaptation(source="Iden_08", target="Iden_01", tag=""):
    d = np.load(RES / f"adapt_{source}_to_{target}_window{tag}.npz")
    zt, zs, za, zo = d["z_true"], d["z_source"], d["z_adapted"], d["z_own"]
    dl, v, n_pre = d["delta_adapted"], d["speed"], int(d["n_pre"])
    t = np.arange(len(zt)) * DT
    fig = plt.figure(figsize=(9.6, 7.6))
    gs = fig.add_gridspec(5, 2, height_ratios=[0.8, 1, 1, 1, 0.9], hspace=0.42, wspace=0.22,
                          top=0.88)

    def window(ax):
        ax.axvspan(t[0], t[n_pre], color="#F0F2F4", zorder=0, lw=0)

    ax = fig.add_subplot(gs[0, :])
    window(ax)
    ax.step(t, v, where="post", color=INK, lw=1.1)
    ax.set_ylabel("speed\n(mm s$^{-1}$)", linespacing=1.3)
    ax.set_xlim(t[0], t[-1])
    ax.set_xticklabels([])
    ax.text(t[n_pre] / 2, ax.get_ylim()[1], "calibration window", ha="center", va="bottom",
            fontsize=8.5, color=SOFT)

    for k in range(6):
        ax = fig.add_subplot(gs[1 + k // 2, k % 2])
        window(ax)
        ax.plot(t, zt[:, k], color=INK, lw=0.7, alpha=0.75, label="from the images")
        ax.plot(t, zs[:, k], color=RED, lw=1.1, ls=(0, (4, 2)), label="source model, unchanged")
        ax.plot(t, za[:, k], color=BLUE, lw=1.5, label="after calibrating three scalars")
        ax.plot(t, zo[:, k], color=GREEN, lw=1.0, ls=(0, (1, 1.6)),
                label="identified on this record")
        ax.set_ylabel(f"$z_{k+1}$")
        ax.set_xlim(t[0], t[-1])
        if k < 4:
            ax.set_xticklabels([])
        if k == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, ncol=4, fontsize=8.6, loc="upper center",
                       bbox_to_anchor=(0.5, 0.925))

    ax = fig.add_subplot(gs[4, :])
    window(ax)
    ax.plot(t, dl, color=RED, lw=1.5)
    ax.fill_between(t, 0, dl, color=RED, alpha=0.12)
    ax.set_ylabel("$\\delta$")
    ax.set_xlabel("time (s)")
    ax.set_xlim(t[0], t[-1])
    kind = "filtered state" if tag else "state as read"
    fig.suptitle(f"Adaptation {source} $\\rightarrow$ {target} ({kind}): networks frozen, three "
                 "scalars fitted on the first 40 %", y=0.975, fontsize=11)
    save(fig, f"adaptation_{source}_to_{target}{tag}")


def state_overview():
    """The six coordinates of every record, to show what the state looks like."""
    import node_visual as N
    data = N.records()
    fig, axes = plt.subplots(6, 1, figsize=(9.6, 8.2), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(N.IDS)))
    for k in range(6):
        ax = axes[k]
        for c, sid in zip(colors, N.IDS):
            z = data[sid]["z"][:, k]
            z = (z - z.mean()) / max(z.std(), 1e-9)
            ax.plot(np.arange(len(z)) * DT, z, color=c, lw=0.55, alpha=0.8)
        ax.set_ylabel(f"$z_{k+1}$")
        ax.set_xlim(0, 70)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("The six state coordinates, all ten records, each standardised",
                 y=0.935, fontsize=11)
    save(fig, "state_overview")


def smoothing(sid="Iden_11", records=("Iden_08", "Iden_09", "Iden_11")):
    """Fitting the filtered state against fitting the state as read."""
    raw = np.load(RES / f"fit_{sid}.npz")
    sm = np.load(RES / f"fit_{sid}_s10.npz")
    zr, zf = sm["z_raw"], sm["z_true"]
    pr, pf = raw["z_pred"], sm["z_pred"]
    t = np.arange(len(zr)) * DT
    ORANGE = "#D98A2B"

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 0.25, 1.15], hspace=0.5, wspace=0.26,
                          top=0.88)
    for k in range(6):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        ax.plot(t, zr[:, k], color="#BFC4C9", lw=0.6, label="state as read")
        ax.plot(t, zf[:, k], color=INK, lw=0.9, label="after a 1 s moving average")
        ax.plot(t, pr[:, k], color=BLUE, lw=1.4, label="fitted on the state as read")
        ax.plot(t, pf[:, k], color=ORANGE, lw=1.4, ls=(0, (4, 2)),
                label="fitted on the filtered state")
        ax.set_ylabel(f"$z_{k+1}$")
        ax.set_xlim(t[0], t[-1])
        if k < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (s)")
        if k == 0:
            h, l = ax.get_legend_handles_labels()
            fig.legend(h, l, ncol=4, fontsize=8.6, loc="upper center", bbox_to_anchor=(0.5, 0.945))

    ax = fig.add_subplot(gs[3, :])
    x = np.arange(len(records))
    def r2(o, p):
        m, s_ = o.mean(0), np.maximum(o.std(0), 1e-9)
        o, p = (o - m) / s_, (p - m) / s_
        return float(1 - ((o - p) ** 2).sum() / (o ** 2).sum())
    v_raw, v_sm, ceil = [], [], []
    from scipy.ndimage import uniform_filter1d
    for r in records:
        d0, d1 = np.load(RES / f"fit_{r}.npz"), np.load(RES / f"fit_{r}_s10.npz")
        z = d1["z_raw"]
        v_raw.append(r2(z, d0["z_pred"]))
        v_sm.append(r2(z, d1["z_pred"]))
        zn = (z - z.mean(0)) / np.maximum(z.std(0), 1e-9)
        ceil.append(float(uniform_filter1d(zn, 10, axis=0).var(0).sum() / zn.var(0).sum()))
    ax.bar(x - 0.17, v_raw, 0.32, color=BLUE, edgecolor="none", label="fitted on the state as read")
    ax.bar(x + 0.17, v_sm, 0.32, color=ORANGE, edgecolor="none", label="fitted on the filtered state")
    for i, c in enumerate(ceil):
        ax.plot([i - 0.4, i + 0.4], [c, c], color=INK, lw=1.3)
        ax.text(i + 0.42, c, "predictable share", fontsize=8, va="center", color=INK)
    ax.set_xticks(x, [r.replace("Iden_", "Iden ") for r in records])
    ax.set_ylabel("$R^2$ against the\nstate as read", linespacing=1.3)
    ax.set_ylim(0, 0.85)
    ax.legend(ncol=2, fontsize=8.6, loc="upper left")
    fig.suptitle("Filtering the state before fitting changes the score, not the model",
                 y=0.985, fontsize=11)
    save(fig, "smoothing_comparison")


def stages(records=("Iden_08", "Iden_09", "Iden_11"), source="Iden_08"):
    """One row per record: the state as read, the identification, and the adaptation."""
    ORANGE = "#D98A2B"
    fig, axes = plt.subplots(len(records), 3, figsize=(12.2, 7.2))
    for r, sid in enumerate(records):
        sm = np.load(RES / f"fit_{sid}_s10.npz")
        zr, zf, pf = sm["z_raw"], sm["z_true"], sm["z_pred"]
        t = np.arange(len(zr)) * DT
        m, sd = zf.mean(0), np.maximum(zf.std(0), 1e-9)
        per = 1 - (((zf - m) / sd - (pf - m) / sd) ** 2).sum(0) / (((zf - m) / sd) ** 2).sum(0)
        k = int(np.argmax(per))                      # the best-fitted coordinate of this record

        ax = axes[r, 0]
        ax.plot(t, zr[:, k], color="#C3C7CB", lw=0.7, label="as read")
        ax.plot(t, zf[:, k], color=INK, lw=1.2, label="1 s moving average")
        ax.set_ylabel(f"{sid.replace('Iden_', 'Iden ')}\n$z_{{{k+1}}}$", linespacing=1.5)
        if r == 0:
            ax.set_title("state read from the images", fontsize=10.5, pad=26)
            ax.legend(fontsize=8.4, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)

        ax = axes[r, 1]
        ax.plot(t, zf[:, k], color=INK, lw=1.0, label="filtered state")
        ax.plot(t, pf[:, k], color=ORANGE, lw=1.7, label="predicted from the input")
        if r == 0:
            ax.set_title("identification on the filtered state", fontsize=10.5, pad=26)
            ax.legend(fontsize=8.4, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
        ax.text(0.015, 0.06, f"$R^2$ = {per[k]:.2f}", transform=ax.transAxes, fontsize=9,
                color=ORANGE)

        ax = axes[r, 2]
        if sid == source:
            ax.text(0.5, 0.86, "source record\n(model identified here)", ha="center", va="top",
                    transform=ax.transAxes, fontsize=10, color=SOFT, linespacing=1.5)
            for col, lab in ((SOFT, "state as read"), (RED, "source model, unchanged"),
                             (BLUE, "after calibrating three scalars")):
                ax.plot([], [], color=col, lw=1.6,
                        ls=(0, (4, 2)) if lab.startswith("source") else "-", label=lab)
            ax.legend(fontsize=8.6, loc="center", bbox_to_anchor=(0.5, 0.36))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        else:
            d = np.load(RES / f"adapt_{source}_to_{sid}_window.npz")
            td = np.arange(len(d["z_true"])) * DT
            n_pre = int(d["n_pre"])
            ax.axvspan(td[0], td[n_pre], color="#F0F2F4", zorder=0, lw=0)
            ax.plot(td, d["z_true"][:, k], color="#C3C7CB", lw=0.7)
            ax.plot(td, d["z_source"][:, k], color=RED, lw=1.2, ls=(0, (4, 2)),
                    )
            ax.plot(td, d["z_adapted"][:, k], color=BLUE, lw=1.7,
                    )
            ax.set_xlim(td[0], td[-1])
        if r == 0:
            ax.set_title("adaptation from " + source.replace("Iden_", "Iden "), fontsize=10.5,
                         pad=26)
        if r == 1:
            ax.text(0.015, 0.06, "shaded: calibration window", transform=ax.transAxes,
                    fontsize=8.4, color=SOFT)
        for c in (0, 1):
            axes[r, c].set_xlim(t[0], t[-1])
        for c in range(3):
            if r < len(records) - 1:
                axes[r, c].set_xticklabels([])
            else:
                axes[r, c].set_xlabel("time (s)")
    fig.suptitle("One coordinate per record: the state as read, the fit to its filtered form, "
                 "and the transfer to another record", y=0.975, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    save(fig, "stages_overview")


def adaptation_pair(source, target, tag="_s15"):
    """One target, with the memory-only calibration and the one that also fits a speed gain."""
    d3 = np.load(RES / f"adapt_{source}_to_{target}_window{tag}_n3.npz")
    d4 = np.load(RES / f"adapt_{source}_to_{target}_window{tag}_n4.npz")
    zt, zs, zo = d4["z_true"], d4["z_source"], d4["z_own"]
    a3, a4 = d3["z_adapted"], d4["z_adapted"]
    dl, v, n_pre = d4["delta_adapted"], d4["speed"], int(d4["n_pre"])
    c = float(d4["calib"][3])
    t = np.arange(len(zt)) * DT
    PURPLE = "#7B5AA6"

    fig = plt.figure(figsize=(9.6, 7.6))
    gs = fig.add_gridspec(5, 2, height_ratios=[0.8, 1, 1, 1, 0.9], hspace=0.42, wspace=0.22,
                          top=0.88)

    def window(ax):
        ax.axvspan(t[0], t[n_pre], color="#F0F2F4", zorder=0, lw=0)

    ax = fig.add_subplot(gs[0, :])
    window(ax)
    ax.step(t, v, where="post", color=INK, lw=1.1)
    ax.set_ylabel("speed\n(mm s$^{-1}$)", linespacing=1.3)
    ax.set_xlim(t[0], t[-1]); ax.set_xticklabels([])
    ax.text(t[n_pre] / 2, ax.get_ylim()[1], "calibration window", ha="center", va="bottom",
            fontsize=8.5, color=SOFT)

    for k in range(6):
        ax = fig.add_subplot(gs[1 + k // 2, k % 2])
        window(ax)
        ax.plot(t, zt[:, k], color=INK, lw=0.8, alpha=0.8, label="filtered state")
        ax.plot(t, zs[:, k], color=RED, lw=1.0, ls=(0, (4, 2)), label="source model, unchanged")
        ax.plot(t, a3[:, k], color=PURPLE, lw=1.2, ls=(0, (2, 1.5)),
                label="calibrated: memory only")
        ax.plot(t, a4[:, k], color=BLUE, lw=1.6, label="calibrated: memory and speed gain")
        ax.plot(t, zo[:, k], color=GREEN, lw=0.9, ls=(0, (1, 1.6)),
                label="identified on this record")
        ax.set_ylabel(f"$z_{k+1}$"); ax.set_xlim(t[0], t[-1])
        if k < 4:
            ax.set_xticklabels([])
        if k == 0:
            h, l = ax.get_legend_handles_labels()
            fig.legend(h, l, ncol=5, fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, 0.945))

    ax = fig.add_subplot(gs[4, :])
    window(ax)
    ax.plot(t, dl, color=RED, lw=1.5)
    ax.fill_between(t, 0, dl, color=RED, alpha=0.12)
    ax.set_ylabel("$\\delta$"); ax.set_xlabel("time (s)"); ax.set_xlim(t[0], t[-1])
    fig.suptitle(f"Adaptation {source} $\\rightarrow$ {target}: three scalars, or four with a "
                 f"speed gain of {c:.3f}", y=0.985, fontsize=11)
    save(fig, f"adaptation_{source}_to_{target}{tag}_pair")


GROUP = ("Iden_11", "Iden_09", "Iden_07", "Iden_06")
SOURCE = "Iden_11"

SETUP = {"GTAW": (("Iden_01", "Iden_02", "Iden_06", "Iden_07", "Iden_08", "Iden_09",
                   "Iden_10", "Iden_11", "Iden_12", "Iden_13"), "Iden_09", "_s20_e1200"),
         "GMAW": (("exp_01", "exp_02", "exp_03", "exp_04", "exp_05", "exp_06", "exp_07",
                   "exp_09", "exp_10"), "exp_05", "_s24_e1200")}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", default="GTAW", choices=list(SETUP))
    proc = ap.parse_args().process
    configure(proc)
    records, best, tag = SETUP[proc]
    for r in records:
        identification(r, tag)
    for t in records:
        if t != best:
            adaptation_pair(best, t, tag)

