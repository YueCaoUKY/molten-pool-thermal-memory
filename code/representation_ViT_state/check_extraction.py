#!/usr/bin/env python3
"""Quality check of the extracted pool boundaries: overlays, and agreement with the dataset labels."""
import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from skimage import measure

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Representation Learning NODE zs - All Datasets" / "paper_section4"))
import experiment as E  # noqa: E402
from pool_segmentation import ALL_IDS, OUT, SIZE, SPEC  # noqa: E402

SCALE_A, SCALE_W = (SIZE / 200) ** 2, SIZE / 200          # labels are in the 200 px crop
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#9a9993", "#e4e3df"


def style():
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
                         "legend.frameon": False, "savefig.dpi": 300, "axes.spines.top": False,
                         "axes.spines.right": False})


def load_masks(sid):
    d = np.load(OUT / f"{sid}_masks.npz")
    shape = tuple(d["shape"])
    masks = np.unpackbits(d["masks"])[: int(np.prod(shape))].reshape(shape).astype(bool)
    return masks, d


def labels(sid):
    folder = E.DATA_ROOT / "GTAW_varying_speed" / sid
    info = pd.read_csv(folder / "info.txt", sep="\t")
    have = {int(p.stem.split("_")[1]) for p in folder.glob("images/cropped_*.jpg")}
    return info.loc[info["index"].isin(have)].sort_values("index").reset_index(drop=True)


def noise(x):
    return float(np.sqrt(np.mean(np.diff(x) ** 2)) / np.std(x))


def agreement():
    rows = []
    for sid in ALL_IDS:
        masks, d = load_masks(sid)
        lab = labels(sid)
        a_ext, a_lab = d["area"].astype(float), lab["mask_area"].to_numpy() * SCALE_A
        w_ext, w_lab = d["width"].astype(float), lab["mask_width"].to_numpy() * SCALE_W
        row = dict(sequence_id=sid, frames=len(a_ext),
                   area_extracted=a_ext.mean(), area_label=a_lab.mean(),
                   width_extracted=w_ext.mean(), width_label=w_lab.mean(),
                   r_area=np.corrcoef(a_ext, a_lab)[0, 1],
                   r_area_2s=np.corrcoef(uniform_filter1d(a_ext, 20), uniform_filter1d(a_lab, 20))[0, 1],
                   r_width=np.corrcoef(w_ext, w_lab)[0, 1],
                   noise_extracted=noise(a_ext), noise_label=noise(a_lab))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "agreement_with_labels.csv", index=False)
    return df


def fig_overlays(n_frames=6):
    sids = ["Iden_06", "Iden_09", "Iden_12", "Iden_13"]
    fig, axes = plt.subplots(len(sids), n_frames, figsize=(7.16, 1.25 * len(sids)))
    for r, sid in enumerate(sids):
        masks, d = load_masks(sid)
        seq = E.load_sequence(SPEC, sid, SIZE)
        pick = np.linspace(10, len(masks) - 10, n_frames).round().astype(int)
        for c, i in enumerate(pick):
            ax = axes[r, c]
            ax.imshow(seq.images.numpy()[i, 0], cmap="gray")
            for cont in measure.find_contours(masks[i].astype(float), 0.5):
                ax.plot(cont[:, 1], cont[:, 0], color=BLUE, lw=1.1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(f"frame {i}", fontsize=7)
        axes[r, 0].set_ylabel(sid.replace("_", " "), fontsize=7.5)
    fig.subplots_adjust(wspace=0.04, hspace=0.08)
    fig.savefig(OUT / "fig_overlays.png", bbox_inches="tight")
    plt.close(fig)


def fig_timeseries(df):
    sids = ["Iden_06", "Iden_12"]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.6), gridspec_kw=dict(width_ratios=[2, 1], hspace=0.45, wspace=0.3))
    for r, sid in enumerate(sids):
        masks, d = load_masks(sid)
        lab = labels(sid)
        t = np.arange(len(d["area"])) * SPEC.dt
        a_ext, a_lab = d["area"].astype(float), lab["mask_area"].to_numpy() * SCALE_A
        ax = axes[r, 0]
        ax.plot(t, a_lab, color=MUTED, lw=1.0, label="dataset label")
        ax.plot(t, a_ext, color=BLUE, lw=1.0, label="extracted boundary")
        ax.set_ylabel("pool area (px)")
        ax.set_title(f"{sid.replace('_', ' ')}   frame-to-frame noise: label {noise(a_lab):.2f}, "
                     f"extracted {noise(a_ext):.2f}", loc="left")
        ax.grid(axis="y", color=GRID, lw=0.5)
        if r == 0:
            ax.legend(ncol=2, loc="upper right")
        if r == 1:
            ax.set_xlabel("time (s)")
        sc = axes[r, 1]
        sc.scatter(a_lab, a_ext, s=3, color=BLUE, alpha=0.3, lw=0)
        lo, hi = np.percentile(np.concatenate([a_lab, a_ext]), [1, 99])
        sc.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8)
        sc.set_xlabel("dataset label (px)")
        sc.set_ylabel("extracted (px)")
        row = df[df.sequence_id == sid].iloc[0]
        sc.set_title(f"r = {row.r_area:.2f} per frame, {row.r_area_2s:.2f} at 2 s", loc="left")
        sc.grid(color=GRID, lw=0.5)
    fig.savefig(OUT / "fig_area_agreement.png", bbox_inches="tight")
    plt.close(fig)


def fig_multipart(n=12):
    """Frames where the pool is extracted as more than one part: the hard cases."""
    from scipy import ndimage as ndi
    picks = []
    for sid in ALL_IDS:
        masks, _ = load_masks(sid)
        parts = np.array([ndi.label(m)[1] for m in masks])
        idx = np.flatnonzero(parts > 1)
        for i in idx[np.linspace(0, len(idx) - 1, min(3, len(idx))).round().astype(int)] if len(idx) else []:
            picks.append((sid, int(i), int(parts[i])))
    picks = picks[:n]
    if not picks:
        return
    cols = 6
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.16, 1.32 * rows))
    axes = np.atleast_2d(axes)
    for ax, (sid, i, k) in zip(axes.ravel(), picks):
        masks, _ = load_masks(sid)
        seq = E.load_sequence(SPEC, sid, SIZE)
        ax.imshow(seq.images.numpy()[i, 0], cmap="gray")
        for cont in measure.find_contours(masks[i].astype(float), 0.5):
            ax.plot(cont[:, 1], cont[:, 0], color=BLUE, lw=1.1)
        ax.set_title(f"{sid.replace('_', ' ')} #{i}, {k} parts", fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(wspace=0.04, hspace=0.25)
    fig.savefig(OUT / "fig_multipart.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    style()
    df = agreement()
    fig_overlays()
    fig_timeseries(df)
    fig_multipart()
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    ck = __import__("torch").load(OUT / "unet.pt", map_location="cpu", weights_only=False)
    json.dump({"val_iou_mean": ck["val_iou_mean"], "val_iou_median": ck["val_iou_median"],
               "n_val_frames": ck["n_val"], "median_r_area_per_frame": float(df.r_area.median()),
               "median_r_area_2s": float(df.r_area_2s.median()),
               "median_noise_extracted": float(df.noise_extracted.median()),
               "median_noise_label": float(df.noise_label.median())},
              open(OUT / "quality.json", "w"), indent=1)
    print("\nfigures and tables in", OUT)
