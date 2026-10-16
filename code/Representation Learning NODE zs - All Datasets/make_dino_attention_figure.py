"""make_dino_attention_figure.py — Emergent DINOv2 saliency on welding pools.

Produces two kinds of output:

  (1) Presentation/dino_attention_emergent.png  — the slide figure.
      Two rows, six columns: hand-picked frames from GTAW / GMAW / WAAM-ViD
      (two each). Row 1 = raw frame, row 2 = same frame with the DINOv2
      saliency heatmap overlaid.

  (2) <DATASET>/results/dino_saliency_<id>.png  — per-experiment sweep.
      One figure per experiment in every dataset (10 evenly-spaced frames
      across the trained-model timeline). Useful for picking the best
      visual variant or for backup slides.

Implementation: the saliency is the L2 norm of the last-layer DINOv2
patch tokens (out['x_norm_patchtokens']). The user's encoder consumes
only the CLS token, but the CLS token is itself a transformer-mixed
summary of the patch tokens, so the patch-norm map directly visualises
the spatial substrate that feeds the encoder. We avoid raw CLS-to-patch
attention because DINOv2 ViT-B/14 (no register tokens) has a documented
"high-norm outlier" artifact (Darcet et al. 2024) that concentrates CLS
attention on a few content-irrelevant patches.

Usage:
    python make_dino_attention_figure.py            # both (1) and (2)
    python make_dino_attention_figure.py slide      # only (1) -- the slide
    python make_dino_attention_figure.py sweep      # only (2) -- per-experiment
    python make_dino_attention_figure.py sweep GTAW # sweep one dataset
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "code"))
from dinov2_wrapper import build_dinov2, Preprocess
from encoder import Encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128
DINOV2_INPUT = 224
PATCH_GRID = 16
NUM_PANELS_SWEEP = 10

CLEAN_ROOT = Path(r"e:/AM World Model/Dataset clear")
GTAW_ROOT = CLEAN_ROOT / "GTAW_varying_speed"
GMAW_ROOT = CLEAN_ROOT / "GMAW_varying_speed"
WAAM_FRAMES = _HERE / "WAAM_ViD" / "results" / "_frames"
SLIDE_OUT = _HERE.parent / "Presentation" / "dino_attention_emergent.png"
COMBINED_SLIDE_OUT = _HERE.parent / "Presentation" / "dino_latent_emergent.png"

GTAW_IDS = ["Iden_01", "Iden_02", "Iden_06", "Iden_07", "Iden_08",
            "Iden_09", "Iden_10", "Iden_11", "Iden_12", "Iden_13"]
GMAW_IDS = ["exp_01", "exp_02", "exp_03", "exp_04", "exp_05",
            "exp_06", "exp_07", "exp_09", "exp_10"]
WAAM_IDS = ["video_1", "video_3", "video_5"]

# Two hand-picked, well-developed-pool frames per dataset for the slide.
SLIDE_SAMPLES = [
    ("GTAW",     "Iden_06", 230),
    ("GTAW",     "Iden_06", 480),
    ("GMAW",     "exp_01",  120),
    ("GMAW",     "exp_01",  290),
    ("WAAM_ViD", "video_1",  90),
    ("WAAM_ViD", "video_1", 220),
]

_IMAGE_TX = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.Grayscale(1),
    T.ToTensor(),
])


# ---------------------------------------------------------------------------
# Frame loaders -- match the trained-model timeline.
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
    return [idx_to_file[int(i)] for i in info["Index"].values][::2]


def load_frame(dataset: str, exp_id: str, idx: int) -> torch.Tensor:
    if dataset == "GTAW":
        return _IMAGE_TX(Image.open(_gtaw_files(exp_id)[idx]).convert("RGB"))
    if dataset == "GMAW":
        return _IMAGE_TX(Image.open(_gmaw_files(exp_id)[idx]).convert("RGB"))
    if dataset == "WAAM_ViD":
        arr = np.load(WAAM_FRAMES / f"{exp_id}.npy", mmap_mode="r")
        x = np.array(arr[idx], copy=True).astype(np.float32) / 255.0
        return torch.from_numpy(x).unsqueeze(0)
    raise ValueError(dataset)


def num_frames(dataset: str, exp_id: str) -> int:
    cache = _HERE / dataset / "results" / "_features" / f"{exp_id}.pt"
    blob = torch.load(cache, map_location="cpu", weights_only=False)
    return int(blob["cls"].shape[0])


# ---------------------------------------------------------------------------
# Saliency extraction
# ---------------------------------------------------------------------------
def load_encoder_and_pc1(dataset: str):
    """Load the dataset's trained encoder + PCA mean + PC1 direction from
    deploy.pt. Returns (encoder, pca_mean, pc1) on DEVICE, encoder in eval mode."""
    deploy = _HERE / dataset / "results" / "deploy.pt"
    blob = torch.load(deploy, map_location=DEVICE, weights_only=False)
    cfg = blob["cfg"]
    enc = Encoder(in_dim=cfg["dinov2_dim"],
                  hidden=cfg["encoder_hidden"],
                  d_s=cfg["d_s"]).to(DEVICE)
    enc.load_state_dict(blob["enc"])
    enc.eval()
    pca_mean = torch.as_tensor(blob["pca_mean"], dtype=torch.float32, device=DEVICE)
    pc1 = torch.as_tensor(blob["pca_components"][0], dtype=torch.float32, device=DEVICE)
    return enc, pca_mean, pc1


@torch.no_grad()
def extract_latent_saliency_maps(model, preproc, frames: torch.Tensor,
                                  encoder, pca_mean, pc1) -> np.ndarray:
    """Per-patch saliency using the TRAINED bottleneck.

    For each patch token p, treat it as a 768-D summary, run it through the
    learned encoder MLP to get a 6-D mu_p, project on PC1 to get a scalar
    z_pc1_p. Subtract per-frame median (background level) and take the
    magnitude: patches whose features fall far from the typical patch on the
    PC1 axis are the patches the trained latent finds informative.

    This differs from the raw DINOv2 patch-L2-norm saliency in that it asks
    'which patches push the LEARNED latent off its background level', not
    'which patches have rich raw features'. The map directly reflects what
    the encoder + PC1 axis has learned to attend to."""
    x = preproc(frames.to(DEVICE))
    out = model.forward_features(x)
    patches = out["x_norm_patchtokens"]                  # (B, 256, 768)
    B, N, D = patches.shape
    mu_patches, _ = encoder(patches.reshape(B * N, D))   # (B*N, 6)
    z_per_patch = (mu_patches - pca_mean) @ pc1          # (B*N,)
    z_per_patch = z_per_patch.reshape(B, N)              # (B, 256)
    med = z_per_patch.median(dim=-1, keepdim=True).values
    z_mag = (z_per_patch - med).abs()                    # (B, 256)
    grid = z_mag.reshape(B, PATCH_GRID, PATCH_GRID)
    grid = F.interpolate(grid.unsqueeze(1).float(),
                         size=(IMG_SIZE, IMG_SIZE),
                         mode="bilinear", align_corners=False).squeeze(1)
    flat = grid.reshape(B, -1)
    lo = torch.quantile(flat, 0.05, dim=-1).view(B, 1, 1)
    hi = torch.quantile(flat, 0.98, dim=-1).view(B, 1, 1)
    grid = ((grid - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1)
    return grid.cpu().numpy().astype(np.float32)


@torch.no_grad()
def extract_saliency_maps(model, preproc, frames: torch.Tensor) -> np.ndarray:
    """frames: (B, 1, 128, 128) grayscale in [0, 1].
    Returns (B, 128, 128) float32, per-frame DINOv2 patch-feature-norm saliency.

    Capped at the 98th percentile per frame to suppress DINOv2 ViT-B/14's
    high-norm outlier patches (a known artifact in the no-register model);
    rescaled to [0, 1] over the 5th-98th percentile band so the pool
    region dominates the display."""
    x = preproc(frames.to(DEVICE))
    out = model.forward_features(x)
    patches = out["x_norm_patchtokens"]                     # (B, 256, 768)
    norms = patches.norm(dim=-1)                            # (B, 256)
    B = norms.shape[0]
    grid = norms.reshape(B, PATCH_GRID, PATCH_GRID)
    grid = F.interpolate(grid.unsqueeze(1).float(),
                         size=(IMG_SIZE, IMG_SIZE),
                         mode="bilinear", align_corners=False).squeeze(1)
    flat = grid.reshape(B, -1)
    lo = torch.quantile(flat, 0.05, dim=-1).view(B, 1, 1)
    hi = torch.quantile(flat, 0.98, dim=-1).view(B, 1, 1)
    grid = ((grid - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1)
    return grid.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def _stretch(frame: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(frame, [0.01, 0.99])
    return np.clip((frame - lo) / max(hi - lo, 1e-6), 0, 1)


def _stretch_batch(frames: np.ndarray) -> np.ndarray:
    return np.stack([_stretch(f) for f in frames], axis=0)


def _overlay_rgba(saliency: np.ndarray, cmap_name: str = "hot",
                  max_alpha: float = 0.50) -> np.ndarray:
    """Build an (H, W, 4) RGBA overlay where alpha is proportional to
    saliency. Low-saliency pixels are fully transparent (original frame
    shows through); high-saliency pixels are colored by the cmap with up
    to max_alpha opacity. Standard CAM-style overlay."""
    import matplotlib
    s = np.clip(saliency, 0.0, 1.0)
    rgba = matplotlib.colormaps[cmap_name](s)   # (H, W, 4)
    rgba[..., 3] = np.clip(s * max_alpha, 0.0, 1.0)
    return rgba


def make_figure_3row(frames: np.ndarray, dino_sal: np.ndarray, latent_sal: np.ndarray,
                     col_titles: List[str], save_path: Path, suptitle: str):
    """Three rows: raw / DINOv2-alone saliency overlay / DINOv2+trained-latent
    saliency overlay. Lets the audience compare what frozen DINOv2 produces
    against what DINOv2 + the trained bottleneck produces, side by side."""
    N = frames.shape[0]
    fig, axes = plt.subplots(3, N, figsize=(1.7 * N, 6.2))
    if N == 1:
        axes = axes[:, None]
    for j in range(N):
        ax = axes[0, j]
        ax.imshow(frames[j], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(col_titles[j], fontsize=9)
        if j == 0:
            ax.set_ylabel("raw frame", fontsize=10)

        ax = axes[1, j]
        ax.imshow(frames[j], cmap="gray", vmin=0, vmax=1)
        ax.imshow(_overlay_rgba(dino_sal[j]))
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("DINOv2 alone\n(patch L2 norm)", fontsize=10)

        ax = axes[2, j]
        ax.imshow(frames[j], cmap="gray", vmin=0, vmax=1)
        ax.imshow(_overlay_rgba(latent_sal[j]))
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("DINOv2 + trained latent\n(per-patch |z_pc1 − median|)", fontsize=10)

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_figure(frames: np.ndarray, saliency: np.ndarray,
                col_titles: List[str], save_path: Path, suptitle: str):
    N = frames.shape[0]
    fig, axes = plt.subplots(2, N, figsize=(1.7 * N, 4.2))
    if N == 1:
        axes = axes[:, None]
    for j in range(N):
        ax = axes[0, j]
        ax.imshow(frames[j], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(col_titles[j], fontsize=9)
        if j == 0:
            ax.set_ylabel("raw frame", fontsize=11)
        ax = axes[1, j]
        ax.imshow(frames[j], cmap="gray", vmin=0, vmax=1)
        ax.imshow(_overlay_rgba(saliency[j]))
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("DINOv2 saliency overlay\n(patch-token L2 norm)", fontsize=10)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def render_slide(model, preproc):
    frames = torch.stack([load_frame(d, e, i) for d, e, i in SLIDE_SAMPLES], dim=0)
    sal = extract_saliency_maps(model, preproc, frames)
    titles = [f"{d}  {e}\nt={i}" for d, e, i in SLIDE_SAMPLES]
    frames_disp = _stretch_batch(frames.squeeze(1).numpy())
    make_figure(frames_disp, sal, titles, SLIDE_OUT,
                "Frozen DINOv2 spotlights the molten pool "
                "— without ever being told what a pool is")
    print(f"saved slide -> {SLIDE_OUT}")


def render_combined_slide(model, preproc):
    """3-row keynote figure: raw / DINOv2 alone / DINOv2 + trained latent."""
    from collections import defaultdict
    frames = torch.stack([load_frame(d, e, i) for d, e, i in SLIDE_SAMPLES], dim=0)

    dino_sal = extract_saliency_maps(model, preproc, frames)

    # Latent saliency: each dataset uses its OWN trained encoder + PC1.
    latent_sal = np.empty((len(SLIDE_SAMPLES), IMG_SIZE, IMG_SIZE), dtype=np.float32)
    by_dataset: dict = defaultdict(list)
    for idx, (d, _e, _i) in enumerate(SLIDE_SAMPLES):
        by_dataset[d].append(idx)
    for dataset, indices in by_dataset.items():
        enc, pca_mean, pc1 = load_encoder_and_pc1(dataset)
        sub = extract_latent_saliency_maps(model, preproc, frames[indices],
                                            enc, pca_mean, pc1)
        for k, global_idx in enumerate(indices):
            latent_sal[global_idx] = sub[k]

    titles = [f"{d}  {e}\nt={i}" for d, e, i in SLIDE_SAMPLES]
    frames_disp = _stretch_batch(frames.squeeze(1).numpy())
    make_figure_3row(frames_disp, dino_sal, latent_sal, titles,
                     COMBINED_SLIDE_OUT,
                     "Frozen DINOv2 finds the pool. The trained bottleneck "
                     "sharpens onto the pool features that drive the latent state.")
    print(f"saved combined slide -> {COMBINED_SLIDE_OUT}")


def render_combined_sweep_one(model, preproc, dataset: str, exp_id: str,
                               enc, pca_mean, pc1):
    """3-row per-experiment sweep figure."""
    T_ = num_frames(dataset, exp_id)
    indices = [int(i) for i in np.linspace(0, T_ - 1, NUM_PANELS_SWEEP).round().astype(int)]
    frames = torch.stack([load_frame(dataset, exp_id, i) for i in indices], dim=0)
    dino_sal = extract_saliency_maps(model, preproc, frames)
    latent_sal = extract_latent_saliency_maps(model, preproc, frames, enc, pca_mean, pc1)
    titles = [f"t={i}" for i in indices]
    frames_disp = _stretch_batch(frames.squeeze(1).numpy())
    save = _HERE / dataset / "results" / f"dino_latent_saliency_{exp_id}.png"
    make_figure_3row(frames_disp, dino_sal, latent_sal, titles, save,
                     f"DINOv2 vs DINOv2+trained-latent saliency  —  "
                     f"{dataset} / {exp_id}  (T={T_})")
    return save


def render_combined_sweep(model, preproc, datasets: List[Tuple[str, List[str]]]):
    for dataset, ids in datasets:
        print(f"\n=== {dataset} ({len(ids)} experiments) ===")
        enc, pca_mean, pc1 = load_encoder_and_pc1(dataset)
        for exp_id in ids:
            t0 = time.time()
            save = render_combined_sweep_one(model, preproc, dataset, exp_id,
                                              enc, pca_mean, pc1)
            print(f"  {dataset}/{exp_id}: {time.time()-t0:.1f}s  -> {save}")


def render_sweep_one(model, preproc, dataset: str, exp_id: str):
    T_ = num_frames(dataset, exp_id)
    indices = [int(i) for i in np.linspace(0, T_ - 1, NUM_PANELS_SWEEP).round().astype(int)]
    frames = torch.stack([load_frame(dataset, exp_id, i) for i in indices], dim=0)
    sal = extract_saliency_maps(model, preproc, frames)
    titles = [f"t={i}" for i in indices]
    frames_disp = _stretch_batch(frames.squeeze(1).numpy())
    save = _HERE / dataset / "results" / f"dino_saliency_{exp_id}.png"
    make_figure(frames_disp, sal, titles, save,
                f"DINOv2 patch-norm saliency  —  {dataset} / {exp_id}  (T={T_})")
    return save


def render_sweep(model, preproc, datasets: List[Tuple[str, List[str]]]):
    for dataset, ids in datasets:
        print(f"\n=== {dataset} ({len(ids)} experiments) ===")
        for exp_id in ids:
            t0 = time.time()
            save = render_sweep_one(model, preproc, dataset, exp_id)
            print(f"  {dataset}/{exp_id}: {time.time()-t0:.1f}s  -> {save}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    valid = ("all", "slide", "sweep", "combined-slide", "combined-sweep", "combined")
    if mode not in valid:
        raise SystemExit(f"unknown mode {mode!r}; use one of: {', '.join(valid)}")

    print(f"device: {DEVICE}")
    model = build_dinov2("dinov2_vitb14").to(DEVICE).eval()
    preproc = Preprocess(DINOV2_INPUT).to(DEVICE)

    if mode in ("all", "slide"):
        render_slide(model, preproc)

    if mode in ("all", "sweep"):
        plan = [("GTAW", GTAW_IDS), ("GMAW", GMAW_IDS), ("WAAM_ViD", WAAM_IDS)]
        if len(sys.argv) > 2:
            plan = [(d, ids) for d, ids in plan if d == sys.argv[2]]
        render_sweep(model, preproc, plan)

    if mode in ("combined", "combined-slide"):
        render_combined_slide(model, preproc)

    if mode in ("combined", "combined-sweep"):
        plan = [("GTAW", GTAW_IDS), ("GMAW", GMAW_IDS), ("WAAM_ViD", WAAM_IDS)]
        if len(sys.argv) > 2:
            plan = [(d, ids) for d, ids in plan if d == sys.argv[2]]
        render_combined_sweep(model, preproc, plan)


if __name__ == "__main__":
    main()
