"""make_xai_figures.py — per-experiment reconstruction-fidelity figure (3 rows x 10 cols).

For each experiment in each dataset (GTAW / GMAW / WAAM_ViD):
  Row 1: 10 evenly-spaced original frames (the 128x128 grayscale the model
         sees), per-frame contrast-stretched so dim pools are visible.
  Row 2: Decoder reconstruction from the encoder's mu_t for the same frame.
         Same contrast stretch as row 1.
  Row 3: Agreement map = elementwise product of normalized original and
         reconstruction, shown with a hot colormap. Bright = both frames
         agree a pool is at this pixel. IoU of top-20% masks printed beneath.

Output: <DATASET>/results/xai_<id>.png

Usage:
  python make_xai_figures.py                       # all datasets, all experiments
  python make_xai_figures.py GTAW                  # GTAW only
  python make_xai_figures.py GTAW Iden_06          # one experiment

The encoder runs on cached DINOv2 CLS tokens (no DINOv2 forward at all),
so this is fast: ~1 s per experiment on CPU, <1 s on GPU.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "code"))
from encoder import Encoder
from decoder import Decoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PANELS = 10
IMG_SIZE = 128
POOL_TOPK = 0.20      # top-20% brightest pixels treated as the pool mask for IoU

CLEAN_ROOT = Path(r"e:/AM World Model/Dataset clear")
RESULTS_ROOT = _HERE
GTAW_ROOT = CLEAN_ROOT / "GTAW_varying_speed"
GMAW_ROOT = CLEAN_ROOT / "GMAW_varying_speed"
WAAM_FRAMES = RESULTS_ROOT / "WAAM_ViD" / "results" / "_frames"

GTAW_IDS = ["Iden_01", "Iden_02", "Iden_06", "Iden_07", "Iden_08",
            "Iden_09", "Iden_10", "Iden_11", "Iden_12", "Iden_13"]
GMAW_IDS = ["exp_01", "exp_02", "exp_03", "exp_04", "exp_05",
            "exp_06", "exp_07", "exp_09", "exp_10"]
WAAM_IDS = ["video_1", "video_3", "video_5"]

_IMAGE_TX = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.Grayscale(1),
    T.ToTensor(),
])


# ---------------------------------------------------------------------------
# Frame loaders -- match the trained-model timeline (same ordering / stride).
# ---------------------------------------------------------------------------
def _gtaw_files(iden: str) -> List[Path]:
    root = GTAW_ROOT / iden
    info = pd.read_csv(root / "info.txt", sep="\t")
    img_dir = root / "images"
    files = sorted(img_dir.glob("cropped_*.jpg"),
                   key=lambda p: int(p.stem.split("_")[1]))
    idx_to_file = {int(p.stem.split("_")[1]): p for p in files}
    info = (info[info["index"].isin(idx_to_file.keys())]
            .sort_values("index").reset_index(drop=True))
    return [idx_to_file[int(i)] for i in info["index"].values]


def _gmaw_files(exp: str) -> List[Path]:
    root = GMAW_ROOT / exp
    info = pd.read_csv(root / "data.csv")
    img_dir = root / "images"
    files = sorted(img_dir.glob("received_image_*.jpg"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    idx_to_file = {int(p.stem.split("_")[-1]): p for p in files}
    info = (info[info["Index"].isin(idx_to_file.keys())]
            .sort_values("Index").reset_index(drop=True))
    files = [idx_to_file[int(i)] for i in info["Index"].values]
    # frame_stride=2 to match training preprocessing
    return files[::2]


def _waam_frames(vid: str) -> np.ndarray:
    return np.load(WAAM_FRAMES / f"{vid}.npy", mmap_mode="r")


def load_frames(dataset: str, exp_id: str, indices: List[int]) -> torch.Tensor:
    """Return (N, 1, 128, 128) grayscale in [0,1] for the given indices."""
    out = []
    if dataset == "GTAW":
        files = _gtaw_files(exp_id)
        for i in indices:
            out.append(_IMAGE_TX(Image.open(files[i]).convert("RGB")))
    elif dataset == "GMAW":
        files = _gmaw_files(exp_id)
        for i in indices:
            out.append(_IMAGE_TX(Image.open(files[i]).convert("RGB")))
    elif dataset == "WAAM_ViD":
        arr = _waam_frames(exp_id)
        for i in indices:
            x = np.array(arr[i], copy=True).astype(np.float32) / 255.0
            out.append(torch.from_numpy(x).unsqueeze(0))
    else:
        raise ValueError(dataset)
    return torch.stack(out, dim=0)


def load_cached_cls(dataset: str, exp_id: str, indices: List[int]) -> torch.Tensor:
    """Return (N, 768) CLS tokens from the cached features for these indices."""
    cache = RESULTS_ROOT / dataset / "results" / "_features" / f"{exp_id}.pt"
    blob = torch.load(cache, map_location="cpu", weights_only=False)
    cls = blob["cls"]               # (T, 768)
    return cls[indices].clone()


def num_frames(dataset: str, exp_id: str) -> int:
    cache = RESULTS_ROOT / dataset / "results" / "_features" / f"{exp_id}.pt"
    blob = torch.load(cache, map_location="cpu", weights_only=False)
    return int(blob["cls"].shape[0])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_models(dataset: str):
    deploy_path = RESULTS_ROOT / dataset / "results" / "deploy.pt"
    blob = torch.load(deploy_path, map_location=DEVICE, weights_only=False)
    cfg = blob["cfg"]
    enc = Encoder(in_dim=cfg["dinov2_dim"],
                  hidden=cfg["encoder_hidden"],
                  d_s=cfg["d_s"]).to(DEVICE)
    enc.load_state_dict(blob["enc"])
    enc.eval()
    dec_block_ch = tuple(cfg.get("dec_block_ch", (128, 64, 32, 16)))
    dec_init_ch = int(cfg.get("dec_init_ch", 128))
    dec = Decoder(d_s=cfg["d_s"], init_ch=dec_init_ch,
                  block_ch=dec_block_ch).to(DEVICE)
    dec.load_state_dict(blob["dec"])
    dec.eval()
    return enc, dec


# ---------------------------------------------------------------------------
# Per-frame normalisation / pool-mask IoU
# ---------------------------------------------------------------------------
def _per_frame_stretch(x: np.ndarray, lo_q: float = 0.01, hi_q: float = 0.99) -> np.ndarray:
    """Scale each frame to roughly [0, 1] using (lo, hi) percentiles, then clip.
    Brings dim GTAW frames into the same display range as bright GMAW frames."""
    N = x.shape[0]
    flat = x.reshape(N, -1)
    lo = np.quantile(flat, lo_q, axis=-1).reshape(N, 1, 1)
    hi = np.quantile(flat, hi_q, axis=-1).reshape(N, 1, 1)
    return np.clip((x - lo) / np.clip(hi - lo, 1e-6, None), 0, 1)


def _agreement_map(orig: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """Elementwise product; per-frame percentile-normalised for display."""
    prod = orig * recon
    N = prod.shape[0]
    flat = prod.reshape(N, -1)
    hi = np.quantile(flat, 0.99, axis=-1).reshape(N, 1, 1)
    return np.clip(prod / np.clip(hi, 1e-6, None), 0, 1)


def _pool_iou(orig: np.ndarray, recon: np.ndarray, top_frac: float = POOL_TOPK) -> np.ndarray:
    """Per-frame IoU of top-top_frac brightest pixels in original vs reconstruction.
    Returns (N,) array of IoU values in [0, 1]."""
    N, H, W = orig.shape
    K = max(1, int(top_frac * H * W))
    out = np.zeros(N, dtype=np.float32)
    for i in range(N):
        thr_o = np.partition(orig[i].ravel(), -K)[-K]
        thr_r = np.partition(recon[i].ravel(), -K)[-K]
        mo = orig[i] >= thr_o
        mr = recon[i] >= thr_r
        inter = (mo & mr).sum()
        union = (mo | mr).sum()
        out[i] = inter / max(union, 1)
    return out


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
@torch.no_grad()
def reconstruct(cls: torch.Tensor, enc: Encoder, dec: Decoder) -> np.ndarray:
    """cls: (N, 768). Returns recon (N, 128, 128) float32 in [0, 1]."""
    cls = cls.to(DEVICE)
    mu, _ = enc(cls)
    logits = dec(mu)
    return torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def make_figure(orig: np.ndarray, recon: np.ndarray, agree: np.ndarray,
                indices: List[int], iou: np.ndarray,
                title: str, save_path: Path):
    N = orig.shape[0]
    fig, axes = plt.subplots(3, N, figsize=(1.5 * N, 4.8))
    if N == 1:
        axes = axes[:, None]

    for j in range(N):
        ax = axes[0, j]
        ax.imshow(orig[j], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t={indices[j]}", fontsize=9)
        if j == 0:
            ax.set_ylabel("original", fontsize=10)

        ax = axes[1, j]
        ax.imshow(recon[j], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("reconstruction", fontsize=10)

        ax = axes[2, j]
        ax.imshow(agree[j], cmap="hot", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"IoU={iou[j]:.2f}", fontsize=8)
        if j == 0:
            ax.set_ylabel("agreement\n(orig x recon)", fontsize=10)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_dataset(dataset: str, ids: List[str]):
    enc, dec = load_models(dataset)
    out_dir = RESULTS_ROOT / dataset / "results"
    for exp_id in ids:
        T_ = num_frames(dataset, exp_id)
        indices = [int(i) for i in np.linspace(0, T_ - 1, NUM_PANELS).round().astype(int)]
        t0 = time.time()
        frames = load_frames(dataset, exp_id, indices).squeeze(1).numpy()  # (N, 128, 128)
        cls = load_cached_cls(dataset, exp_id, indices)
        recon = reconstruct(cls, enc, dec)                                  # (N, 128, 128)

        orig_disp = _per_frame_stretch(frames)
        recon_disp = _per_frame_stretch(recon)
        agree = _agreement_map(orig_disp, recon_disp)
        iou = _pool_iou(orig_disp, recon_disp)

        save = out_dir / f"xai_{exp_id}.png"
        make_figure(orig_disp, recon_disp, agree, indices, iou,
                    title=f"{dataset} / {exp_id}  (T={T_},  mean IoU={iou.mean():.2f})",
                    save_path=save)
        print(f"  {dataset}/{exp_id}: {time.time()-t0:.1f}s  mean IoU={iou.mean():.2f}  -> {save}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", choices=["GTAW", "GMAW", "WAAM_ViD"],
                    default=None)
    ap.add_argument("exp_id", nargs="?", default=None)
    args = ap.parse_args()

    print(f"device: {DEVICE}")
    t0 = time.time()
    plan = [("GTAW", GTAW_IDS), ("GMAW", GMAW_IDS), ("WAAM_ViD", WAAM_IDS)]
    if args.dataset is not None:
        plan = [(d, ids) for d, ids in plan if d == args.dataset]
        if args.exp_id is not None:
            plan = [(d, [args.exp_id]) for d, _ in plan]

    for dataset, ids in plan:
        print(f"\n=== {dataset} ({len(ids)} experiments) ===")
        run_dataset(dataset, ids)

    print(f"\ntotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
