#!/usr/bin/env python3
"""Re-crop the WAAM videos around the pool and rebuild the caches for visual_generator.py.

The crop is chosen once per video, not per frame: the brightest region is tracked through the
record, the track is median filtered, and its median over the record fixes a single 768 x 768
window, placed so that the bright region sits at 35 % of the window from the left and the top.
The window then stays put, so nothing in the cropped video moves that did not move in the
original. Frames are resized to 192 x 192 and encoded with the same frozen DINOv2 ViT-B/14.

    python waam_recrop.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "Representation Learning NODE zs - All Datasets"
sys.path.insert(0, str(ROOT / "code"))
from dinov2_wrapper import Preprocess, build_dinov2, extract_tokens  # noqa: E402

VIDEOS = HERE.parent / "Dataset clear" / "WAAM_ViD_constant_params" / "videos"
FILES = {"video_1": "250312-110206-video_1.mp4",
         "video_3": "250312-110523-video_3.mp4",
         "video_5": "250312-111515-video_5.mp4"}
BOX, FX, FY = 768, 0.35, 0.35      # window size and where the bright region sits inside it
STRIDE, SIZE, GRID = 2, 192, 8     # as in the original cache; output resolution; patch grid kept
MEDIAN, SKIP = 25, 12              # ~2 s median filter; frames skipped while the arc strikes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def window(path: Path) -> tuple[tuple[int, int], np.ndarray]:
    """One fixed window per video, from the brightest region over the whole record."""
    cap = cv2.VideoCapture(str(path))
    grey, track = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grey.append(g)
        m = g >= np.percentile(g, 99.5)
        ys, xs = np.nonzero(m)
        w = g[ys, xs].astype(np.float32)
        track.append(((xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()))
    cap.release()
    track = np.asarray(track)
    smooth = np.stack([median_filter(track[:, k], MEDIAN, mode="nearest") for k in (0, 1)], 1)
    cx, cy = np.median(smooth[SKIP:], axis=0)
    h, w_ = grey[0].shape
    x0 = int(np.clip(round(cx - FX * BOX), 0, w_ - BOX))
    y0 = int(np.clip(round(cy - FY * BOX), 0, h - BOX))
    return (x0, y0), np.stack(grey[::STRIDE])


def main() -> None:
    model = build_dinov2("dinov2_vitb14").to(DEVICE)
    pre = Preprocess(224).to(DEVICE)
    frames_out, feats_out, boxes = {}, {}, {}
    for vid, name in FILES.items():
        t0 = time.time()
        (x0, y0), grey = window(VIDEOS / name)
        boxes[vid] = (x0, y0)
        cropped = np.stack([np.asarray(Image.fromarray(g[y0:y0 + BOX, x0:x0 + BOX])
                                       .resize((SIZE, SIZE), Image.BILINEAR)) for g in grey])
        frames_out[vid] = torch.from_numpy(cropped).unsqueeze(1)          # (T, 1, S, S) uint8
        parts = []
        with torch.no_grad():
            for i in range(0, len(cropped), 32):
                x = torch.from_numpy(cropped[i:i + 32]).float().div(255).unsqueeze(1).to(DEVICE)
                _, patch = extract_tokens(model, pre(x))                   # (b, 16, 16, 768)
                p = patch.permute(0, 3, 1, 2)
                parts.append(F.avg_pool2d(p, 16 // GRID).half().cpu())
        feats_out[vid] = torch.cat(parts).contiguous()
        print(f"{vid}: window ({x0}, {y0}) {BOX} px, {len(cropped)} frames, "
              f"{time.time() - t0:.0f}s", flush=True)
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True)
    torch.save(frames_out, cache / f"frames{SIZE}_WAAM.pt")
    torch.save(feats_out, cache / f"patch{GRID}_WAAM.pt")
    torch.save(boxes, cache / "waam_windows.pt")
    print("windows:", boxes)


if __name__ == "__main__":
    main()
