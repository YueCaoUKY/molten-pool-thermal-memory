#!/usr/bin/env python3
"""What the thermal memory does: the same input with the memory altered.

The model identified on one GTAW record is rolled out from the recorded input and the first
state. The memory state is then replaced by an altered trajectory and the pool state is rolled
out again, so the difference between the two is the contribution of accumulated heat alone.
The resulting states are passed through the image decoder, which turns the difference into
something that can be looked at.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import generator_figures as G
import node_visual as N
import visual_generator as V

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
RES = HERE / "results_node"
DT = 0.1
INK, BLUE, RED, SOFT = "#272727", "#1F5C9E", "#B64342", "#7A8189"
CASES = [("identified", INK, "-"), ("held flat", "#2E7F8A", (0, (4, 2)))]

plt.rcParams.update({
    "font.family": ["Nimbus Sans", "DejaVu Sans", "sans-serif"], "font.size": 9.5,
    "axes.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.frameon": False,
    "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold",
    "svg.fonttype": "none"})


@torch.no_grad()
def retrieve(z_pred, z_bank, codes_bank, bounds=None, k=16, tol=2.5):
    """For each predicted state, the code field of a frame that was actually recorded.

    Sampling the field from the prior draws each grid position on its own, so the decoder
    receives a blend of several possible fields rather than one of them and the result is
    smooth. A recorded field keeps it on the manifold.

    The frames are followed the way a clip is played: once a frame has been used, its successor
    is taken next as long as its state stays within `tol` times the distance of the best match,
    so the detail moves as it did in the recording. When the match falls too far behind, the
    nearest state is looked up again and a new clip is started. `bounds` gives the first index
    of each record so that a clip never runs past the end of one.
    """
    mu, sd = z_bank.mean(0), z_bank.std(0).clamp(min=1e-6)
    d = torch.cdist((z_pred - mu) / sd, (z_bank - mu) / sd)
    ends = set() if bounds is None else {b - 1 for b in bounds[1:]}
    picked, p = [], None
    for t in range(len(z_pred)):
        best = int(d[t].argmin())
        if p is not None and (p + 1) < len(z_bank) and p not in ends:
            nxt = p + 1
            if d[t, nxt] <= tol * d[t, best]:
                p = nxt
            else:
                p = best
        else:
            p = best
        picked.append(p)
    return torch.tensor(picked)


@torch.no_grad()
def rollout_forced(model, z0, u, delta, dt=DT):
    """The pool equation driven by a memory trajectory that is supplied rather than integrated."""
    z, out = z0, []
    for k in range(u.shape[0]):
        out.append(z)
        if k == u.shape[0] - 1:
            break
        z_eq, gamma, _ = model.terms(z, delta[k], u[k])
        z = z_eq + (z - z_eq) * torch.exp(-gamma * dt)
    return torch.stack(out)


def build(sid="Iden_09", smooth=20, tag="_s20_e1200", flat_at=15.0, fast=0.5, n_show=4,
          process="GTAW"):
    N.configure(process)
    N.SMOOTH, N.TAG = smooth, tag
    data = N.records()
    res = N.OUT
    dt = N.DT
    rec = data[sid]
    ck = torch.load(res / f"fit_{sid}{tag}.pt", weights_only=False)
    model = N.PoolNODE(z_dim=6, u_dim=1, width=16)
    model.load_state_dict(ck["model"])
    model.eval()
    mean, std = ck["mean"], ck["std"]
    u = torch.tensor(rec["u"].astype(np.float32))
    zn = torch.tensor(((rec["z"] - mean) / std).astype(np.float32))
    t = np.arange(len(zn)) * dt

    with torch.no_grad():
        z_base, d_base = model.rollout(zn[0], u)
    k_flat = int(flat_at / dt)
    d_flat = d_base.clone()
    d_flat[k_flat:] = d_base[k_flat]
    deltas = [d_base, d_flat]
    cases = [CASES[0], (f"held flat from {flat_at:.0f} s", CASES[1][1], CASES[1][2])]
    states = [z_base] + [rollout_forced(model, zn[0], u, d) for d in deltas[1:]]

    # images
    V.configure(process)
    frames, _ = V.load_cache()
    codes = torch.load(V.OUT / "codes.pt", weights_only=False)
    dec, vq, prior, zmean, zstd = G.models()
    raw = [torch.tensor((z.numpy() * std + mean).astype(np.float32)) for z in states]
    # each run retrieves its own detail layer, so structure and texture agree within a frame;
    # the difference panel below uses the structure alone, which carries no retrieval
    zs_all = torch.load(V.OUT / "states.pt", weights_only=False)
    from scipy.ndimage import uniform_filter1d
    bank_z, bank_c = [], []
    for other in V.IDS:
        zf = uniform_filter1d(zs_all[other].numpy(), smooth, axis=0, mode="nearest")
        bank_z.append(torch.tensor(zf.astype(np.float32)))
        bank_c.append(codes[other])
    bounds = np.cumsum([0] + [len(c) for c in bank_c]).tolist()
    bank_z, bank_c = torch.cat(bank_z), torch.cat(bank_c)
    seqs, structs = [], []
    with torch.no_grad():
        for r in raw:
            field = bank_c[retrieve(r, bank_z, bank_c, bounds)].to(V.DEVICE)
            full, st = [], []
            for i in range(0, len(r), 64):
                base = dec(r[i:i + 64].to(V.DEVICE))
                st.append(base.cpu().numpy()[:, 0])
                full.append((base + vq.decode_indices(field[i:i + 64])).cpu().numpy()[:, 0])
            seqs.append(np.concatenate(full))
            structs.append(np.concatenate(st))
    real = frames[sid].float().numpy()[:, 0] / 255.0
    idx = np.linspace(k_flat, len(t) - 1, n_show).round().astype(int)

    fig = plt.figure(figsize=(10.6, 12.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.55, 0.7, 2.6, 4.6], hspace=0.30, top=0.945)

    ax = fig.add_subplot(gs[0])
    if np.ptp(rec["speed"]) < 1e-6:
        ax.set_axis_off()
        ax.text(0.5, 0.45, "deposition at fixed settings: the input never changes",
                ha="center", va="center", fontsize=11, color=SOFT, transform=ax.transAxes)
    else:
        ax.step(t, rec["speed"], where="post", color=INK, lw=1.1)
        ax.set_ylabel("speed\n(mm s$^{-1}$)", linespacing=1.3)
        ax.set_xlim(t[0], t[-1]); ax.set_xticklabels([])
    ax.set_title(f"{sid}: the same recorded input, with the memory state frozen part way through",
                 fontsize=11, pad=10)

    ax = fig.add_subplot(gs[1])
    for d, (lab, col, ls) in zip(deltas, cases):
        ax.plot(t, d.numpy(), color=col, ls=ls, lw=1.6, label=lab)
    ax.axvline(flat_at, color=SOFT, lw=0.8, ls=":")
    ax.set_ylabel("$\\delta$"); ax.set_xlim(t[0], t[-1]); ax.set_xticklabels([])
    ax.legend(ncol=3, fontsize=8.6, loc="upper left")

    inner = gs[2].subgridspec(3, 2, hspace=0.35, wspace=0.22)
    for k in range(6):
        ax = fig.add_subplot(inner[k // 2, k % 2])
        for z, (lab, col, ls) in zip(states, cases):
            ax.plot(t, z[:, k].numpy(), color=col, ls=ls, lw=1.4)
        ax.set_ylabel(f"$z_{k+1}$"); ax.set_xlim(t[0], t[-1])
        if k < 4:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (s)")
        ax.axvline(flat_at, color=SOFT, lw=0.8, ls=":")

    diff = structs[1] - structs[0]
    lim = float(np.abs(diff[idx]).max())
    rows = [("recorded frame", real, "gray"),
            ("generated,\nidentified memory", seqs[0], "gray"),
            (f"generated,\nmemory held flat", seqs[1], "gray"),
            ("difference of\nthe structures", diff, "diff")]
    grid = gs[3].subgridspec(len(rows), n_show, hspace=0.08, wspace=0.04)
    for i, (lab, arr, kind) in enumerate(rows):
        for j, kk in enumerate(idx):
            ax = fig.add_subplot(grid[i, j])
            if kind == "gray":
                ax.imshow(np.clip(arr[kk], 0, 1), cmap="gray", vmin=0, vmax=float(real.max()))
            else:
                ax.imshow(arr[kk], cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(lab, fontsize=8.8, linespacing=1.4)
            if i == 0:
                ax.set_title(f"{t[kk]:.0f} s", fontsize=9, pad=3)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"delta_manipulation_{process}_{sid}.{ext}", dpi=280, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("saved", FIGS / f"delta_manipulation_{process}_{sid}.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", default="GTAW", choices=["GTAW", "GMAW", "WAAM"])
    ap.add_argument("--record", default=None)
    ap.add_argument("--smooth", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--flat-at", type=float, default=None)
    a = ap.parse_args()
    cfg = {"GTAW": dict(sid="Iden_09", smooth=20, tag="_s20_e1200", flat_at=15.0),
           "GMAW": dict(sid="exp_05", smooth=24, tag="_s24_e1200", flat_at=10.0),
           "WAAM": dict(sid="video_1", smooth=25, tag="_s25_e1200", flat_at=8.0)}[a.process]
    if a.record: cfg["sid"] = a.record
    if a.smooth: cfg["smooth"] = a.smooth
    if a.tag: cfg["tag"] = a.tag
    if a.flat_at: cfg["flat_at"] = a.flat_at
    build(process=a.process, **cfg)
