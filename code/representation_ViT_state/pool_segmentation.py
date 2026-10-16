#!/usr/bin/env python3
"""Pool-boundary extraction for GTAW, to be used as the evaluation reference.

The dataset ships 141 hand-labelled frames, but they were recorded with a wider field of view:
the pool covers 1.5 % of those frames against 15.5 % of the 200 x 200 experiment crops. Each
labelled frame is therefore re-cropped around its own mask so that the pool covers the same
fraction as in the experiments, with random scale and centre jitter, which also serves as
augmentation. A small U-Net is trained on those crops and applied to every experiment frame.

    python pool_segmentation.py train          # train and report validation IoU
    python pool_segmentation.py apply          # masks, area, width and length per experiment
    python pool_segmentation.py check          # figures and agreement with the dataset labels
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
SECTION4 = HERE.parent / "Representation Learning NODE zs - All Datasets" / "paper_section4"
sys.path.insert(0, str(SECTION4))
import experiment as E  # noqa: E402

DATA = E.DATA_ROOT / "GTAW_varying_speed"
OUT = HERE / "pool_extraction"
SIZE = 128                      # same grid as the representation pipeline
POOL_FRACTION = 0.155           # share of the experiment crop covered by the pool, from info.txt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPEC = E.SPECS["GTAW"]
ALL_IDS = (SPEC.train_id, *SPEC.heldout_ids)


# --------------------------------------------------------------------------- labelled data
def labelled_pairs():
    pairs = []
    for image_path in sorted((DATA / "segmentation" / "images").glob("*.jpg")):
        mask_path = DATA / "segmentation" / "masks" / f"{image_path.stem}_mask_0.png"
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def load_pair(image_path, mask_path):
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    return image, mask


def crop_to_experiment_scale(image, mask, scale=1.0, jitter=(0.0, 0.0)):
    """Crop around the mask so the pool covers POOL_FRACTION of the window, as in the experiments."""
    cy, cx = ndi.center_of_mass(mask)
    side = np.sqrt(mask.sum() / POOL_FRACTION) * scale
    cy += jitter[0] * side
    cx += jitter[1] * side
    box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    i = Image.fromarray((image * 255).astype(np.uint8)).crop(box).resize((SIZE, SIZE), Image.BILINEAR)
    m = Image.fromarray(mask.astype(np.uint8) * 255).crop(box).resize((SIZE, SIZE), Image.NEAREST)
    return np.asarray(i, dtype=np.float32) / 255.0, np.asarray(m) > 127


def sample_batch(pairs, rng, batch=16):
    images, masks = [], []
    for _ in range(batch):
        image, mask = load_pair(*pairs[rng.integers(len(pairs))])
        i, m = crop_to_experiment_scale(image, mask, scale=rng.uniform(0.7, 1.5),
                                        jitter=(rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)))
        if rng.random() < 0.5:
            i, m = i[:, ::-1].copy(), m[:, ::-1].copy()
        i = np.clip(i ** rng.uniform(0.7, 1.4) * rng.uniform(0.8, 1.2), 0, 1)     # exposure jitter
        images.append(i)
        masks.append(m)
    x = torch.tensor(np.stack(images))[:, None]
    y = torch.tensor(np.stack(masks).astype(np.float32))[:, None]
    return x.to(DEVICE), y.to(DEVICE)


# --------------------------------------------------------------------------- model
class UNet(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        block = lambda a, b: nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU(),
                                           nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU())
        self.d1, self.d2, self.d3 = block(1, base), block(base, 2 * base), block(2 * base, 4 * base)
        self.bottom = block(4 * base, 8 * base)
        self.u3, self.u2, self.u1 = block(12 * base, 4 * base), block(6 * base, 2 * base), block(3 * base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(F.max_pool2d(d1, 2))
        d3 = self.d3(F.max_pool2d(d2, 2))
        b = self.bottom(F.max_pool2d(d3, 2))
        u3 = self.u3(torch.cat([F.interpolate(b, scale_factor=2), d3], 1))
        u2 = self.u2(torch.cat([F.interpolate(u3, scale_factor=2), d2], 1))
        u1 = self.u1(torch.cat([F.interpolate(u2, scale_factor=2), d1], 1))
        return self.out(u1)


def dice_bce(logits, target):
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum((1, 2, 3)) + 1) / ((p + target).sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + dice.mean()


MIN_RELATIVE, MIN_PIXELS, MAX_DISTANCE = 0.10, 30, 30.0


def clean(mask, min_relative=MIN_RELATIVE, min_pixels=MIN_PIXELS, max_distance=MAX_DISTANCE):
    """Keep the largest component, plus any component that is at least `min_relative` of it,
    at least `min_pixels`, and whose centroid lies within `max_distance` of it, so a pool that
    breaks into nearby parts is kept whole while distant blobs are dropped. Holes are filled.
    `max_distance=None` keeps every component above the size thresholds, 0 keeps only the largest."""
    labels, n = ndi.label(mask)
    if n == 0:
        return mask
    ids = np.arange(1, n + 1)
    sizes = ndi.sum(mask, labels, ids)
    main = int(np.argmax(sizes))
    keep = {ids[main]}
    if max_distance is None or max_distance > 0:
        centres = np.array(ndi.center_of_mass(mask, labels, ids))
        far = np.linalg.norm(centres - centres[main], axis=1)
        for k in ids[(sizes >= max(min_pixels, min_relative * sizes[main])) &
                     (far <= (np.inf if max_distance is None else max_distance))]:
            keep.add(int(k))
    out = np.isin(labels, sorted(keep))
    if len(keep) > 1:                       # bridge parts that are only a few pixels apart
        out = ndi.binary_closing(out, structure=np.ones((9, 9)))
    return ndi.binary_fill_holes(out)


@torch.no_grad()
def predict(net, frames):
    """Probability map averaged over the frame and its mirror image."""
    x = torch.tensor(frames)[:, None].to(DEVICE)
    p = torch.sigmoid(net(x)) + torch.sigmoid(net(torch.flip(x, dims=[3]))).flip(dims=[3])
    return (p / 2)[:, 0].cpu().numpy()


# --------------------------------------------------------------------------- train / apply
def train(steps=2500, seed=0, base=24):
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    pairs = labelled_pairs()
    order = rng.permutation(len(pairs))
    n_val = max(20, len(pairs) // 5)
    val = [pairs[i] for i in order[:n_val]]
    train_pairs = [pairs[i] for i in order[n_val:]]
    print(f"{len(train_pairs)} training frames, {len(val)} validation frames", flush=True)
    torch.manual_seed(seed)
    net = UNet(base).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = sample_batch(train_pairs, rng)
        loss = dice_bce(net(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step}/{steps} loss {float(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
    net.eval()
    ious = []
    with torch.no_grad():
        for image_path, mask_path in val:
            image, mask = load_pair(image_path, mask_path)
            i, m = crop_to_experiment_scale(image, mask)
            p = clean(predict(net, i[None].astype(np.float32))[0] > 0.5)
            ious.append(np.logical_and(p, m).sum() / max(np.logical_or(p, m).sum(), 1))
    ious = np.array(ious)
    torch.save({"model": net.state_dict(), "base": base, "val_iou_mean": float(ious.mean()),
                "val_iou_median": float(np.median(ious)), "val_iou_worst": float(ious.min()),
                "n_val": len(val), "steps": steps}, OUT / "unet.pt")
    print(f"validation IoU: mean {ious.mean():.3f}, median {np.median(ious):.3f}, "
          f"worst {ious.min():.3f} over {len(ious)} held-out labelled frames", flush=True)
    return float(ious.mean())


@torch.no_grad()
def apply_to_experiments():
    ck = torch.load(OUT / "unet.pt", map_location=DEVICE, weights_only=False)
    net = UNet(ck.get("base", 16)).to(DEVICE)
    net.load_state_dict(ck["model"])
    net.eval()
    rows = []
    for sid in ALL_IDS:
        seq = E.load_sequence(SPEC, sid, SIZE)
        frames = seq.images.numpy()[:, 0].astype(np.float32) / 255.0
        masks = np.zeros((len(frames), SIZE, SIZE), dtype=bool)
        for s in range(0, len(frames), 64):
            p = predict(net, frames[s:s + 64]) > 0.5
            masks[s:s + 64] = np.stack([clean(m) for m in p])
        area = masks.reshape(len(masks), -1).sum(1)
        width = np.array([m.any(axis=0).sum() for m in masks])
        length = np.array([m.any(axis=1).sum() for m in masks])
        centroid = np.array([ndi.center_of_mass(m) if m.any() else (np.nan, np.nan) for m in masks])
        parts = np.array([ndi.label(m)[1] for m in masks])
        np.savez_compressed(OUT / f"{sid}_masks.npz", masks=np.packbits(masks, axis=None),
                            shape=np.array(masks.shape), area=area, width=width, length=length,
                            centroid=centroid, parts=parts, u=seq.u.numpy())
        pd.DataFrame(dict(frame=np.arange(len(masks)), t_s=np.arange(len(masks)) * SPEC.dt,
                          welding_speed=seq.u.numpy() * (SPEC.u_max - SPEC.u_min) + SPEC.u_min,
                          area_px=area, width_px=width, length_px=length,
                          centroid_row=centroid[:, 0], centroid_col=centroid[:, 1],
                          parts=parts)).to_csv(OUT / f"{sid}_pool_state.csv", index=False)
        rows.append(dict(sequence_id=sid, frames=len(masks), empty=int((area == 0).sum()),
                         area_mean=float(area.mean()), width_mean=float(width.mean()),
                         multi_part_frames=int((parts > 1).sum()), parts_max=int(parts.max())))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=2500)
    t.add_argument("--base", type=int, default=24)
    sub.add_parser("apply")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.steps, base=a.base)
    else:
        apply_to_experiments()


if __name__ == "__main__":
    main()
