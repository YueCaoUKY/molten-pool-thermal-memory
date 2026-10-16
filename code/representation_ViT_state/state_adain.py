#!/usr/bin/env python3
"""Does removing the frame-level statistics of the patch tokens make the state transfer?

One change to the state extractor of visual_generator.py. The patch-token map of a frame is
split into its per-channel spatial mean and spread, which carry how the frame looks, and the
pattern that remains, which carries where things are:

    t~[c,i,j] = (t[c,i,j] - mu[c]) / sd[c]

The state is read from the pattern only. The statistics are not thrown away: they reach the
decoder through a path that can only scale and shift the whole image,

    I^ = a(n) * D(z) + b(n),

so nothing spatial can travel that way and the state is free of frame-level appearance by
construction. No loss term, no weight, no augmentation is added.

    python state_adain.py train --variant adain
    python state_adain.py compare

Segmentation masks appear only in the evaluation; training reads the feature cache alone.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d

import visual_generator as V

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_adain"
VARIANTS = ("plain", "adain", "mean_only")


def split(feats: torch.Tensor):
    """Frame-level statistics of the patch map, and what is left after removing them."""
    mu = feats.mean(dim=(2, 3), keepdim=True)
    sd = feats.std(dim=(2, 3), keepdim=True).clamp(min=1e-5)
    return (feats - mu) / sd, torch.cat([mu.flatten(1), sd.flatten(1)], 1)


class Decoder(V.StructureDecoder):
    """The structure decoder with a path for the frame statistics that is affine in the image."""

    def __init__(self, dim=V.DIM, base=256, mean_frame=None, size=None, nuis_dim=0):
        super().__init__(dim=dim, base=base, mean_frame=mean_frame, size=size)
        self.gain = None
        if nuis_dim:
            self.gain = nn.Sequential(nn.Linear(nuis_dim, 32), nn.SiLU(), nn.Linear(32, 2))
            nn.init.zeros_(self.gain[-1].weight)
            nn.init.zeros_(self.gain[-1].bias)

    def forward(self, z, n=None):
        img = super().forward(z)
        if self.gain is None or n is None:
            return img
        g = self.gain(n)
        return (1.0 + g[:, 0]).view(-1, 1, 1, 1) * img + g[:, 1].view(-1, 1, 1, 1)


def prepare(feats: torch.Tensor, variant: str):
    if variant == "plain":
        return feats, None
    norm, stats = split(feats)
    if variant == "mean_only":                    # weaker version: keep the spread
        mu = feats.mean(dim=(2, 3), keepdim=True)
        return feats - mu, mu.flatten(1)
    return norm, stats


def train(process="GTAW", variant="adain", steps=8000, batch=32, lr=2e-4, seed=0):
    assert variant in VARIANTS
    V.configure(process)
    torch.manual_seed(seed)
    OUT.mkdir(exist_ok=True)
    frames, feats = V.load_cache()
    img, spans = V.flatten(frames)
    fea, _ = V.flatten(feats)
    n_frames = len(img)
    mean_frame = (img.float() / 255.0).mean(0, keepdim=True).to(V.DEVICE)
    # the split depends only on the frame, so it is done once for the whole cache
    parts, stat_parts = [], []
    for i in range(0, n_frames, 256):
        x, n = prepare(fea[i:i + 256].to(V.DEVICE).float(), variant)
        parts.append(x.half().cpu())
        if n is not None:
            stat_parts.append(n.half().cpu())
    fea = torch.cat(parts)
    nuis = torch.cat(stat_parts) if stat_parts else None
    del parts, stat_parts
    stats = None if nuis is None else nuis[:2]
    head = V.Head().to(V.DEVICE)
    dec = Decoder(mean_frame=mean_frame, nuis_dim=0 if stats is None else stats.shape[1]).to(V.DEVICE)
    opt = torch.optim.AdamW(list(head.parameters()) + list(dec.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n_frames, (batch,))
        x = fea[idx].to(V.DEVICE).float()
        n = None if nuis is None else nuis[idx].to(V.DEVICE).float()
        y = img[idx].to(V.DEVICE).float() / 255.0
        loss = F.mse_loss(dec(head(x), n), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"{variant} {step:5d}  mse {loss.item():.5f}  {time.time()-t0:.0f}s", flush=True)
    head.eval()
    zs, recon = {}, {}
    with torch.no_grad():
        for sid in V.IDS:
            a, b = spans[sid]
            out, err = [], []
            for i in range(a, b, 128):
                j = min(i + 128, b)
                x = fea[i:j].to(V.DEVICE).float()
                n = None if nuis is None else nuis[i:j].to(V.DEVICE).float()
                z = head(x)
                out.append(z.cpu())
                y = img[i:j].to(V.DEVICE).float() / 255.0
                err.append((dec(z, n) - y).abs().mean().item() * (j - i))
            zs[sid] = torch.cat(out)
            recon[sid] = sum(err) / (b - a)
    torch.save({"head": head.state_dict(), "dec": dec.state_dict(), "variant": variant,
                "mean_frame": mean_frame.cpu()}, OUT / f"stage_a_{variant}.pt")
    torch.save(zs, OUT / f"states_{variant}.pt")
    json.dump(recon, open(OUT / f"recon_{variant}.json", "w"), indent=1)
    print(variant, "reconstruction MAE:", {k: round(v, 4) for k, v in recon.items()}, flush=True)


# --------------------------------------------------------------------------- evaluation
def _design(u, smooth=10):
    """Speed, its square, and first-order responses to the heat input at four time constants.
    Only the recorded input is used; nothing measured from the images enters here."""
    u = np.asarray(u, dtype=float)
    h = 1.0 - u
    cols = [np.ones_like(u), u, u ** 2]
    for tau in (2.0, 5.0, 10.0, 20.0):
        y = np.zeros_like(h)
        acc = 0.0
        for k, v in enumerate(h):
            acc += 0.1 / tau * (v - acc)
            y[k] = acc
        cols.append(y)
    return np.column_stack(cols)


def _r2(o, p):
    o = np.atleast_2d(o.T).T
    p = np.atleast_2d(p.T).T
    m, s = o.mean(0), np.maximum(o.std(0), 1e-9)
    o, p = (o - m) / s, (p - m) / s
    return float(1 - ((o - p) ** 2).sum() / (o ** 2).sum())


def matrix(series: dict, inputs: dict, ids, prefix=0.4):
    """Transfer matrix: fit on the source record, predict the held-out part of the target."""
    A = {s: _design(inputs[s]) for s in ids}
    M = np.zeros((len(ids), len(ids)))
    for i, s in enumerate(ids):
        y = series[s]
        ys = (y - y.mean(0)) / np.maximum(y.std(0), 1e-9)
        th = np.linalg.lstsq(A[s], ys, rcond=None)[0]
        for j, t in enumerate(ids):
            yt = series[t]
            npre = max(20, int(prefix * len(yt)))
            m, sd = yt[:npre].mean(0), np.maximum(yt[:npre].std(0), 1e-9)
            M[i, j] = _r2(((yt - m) / sd)[npre:], (A[t] @ th)[npre:])
    return M


def compare(process="GTAW"):
    import node_visual as N
    V.configure(process)
    data = N.records()
    ids = list(N.IDS)
    inputs, area, states = {}, {}, {}
    for sid in ids:
        df = pd.read_csv(HERE / "pool_extraction" / f"{sid}_pool_state.csv")
        n = min(len(df), len(data[sid]["u"]))
        inputs[sid] = data[sid]["u"][:n, 0]
        area[sid] = uniform_filter1d(df["area_px"].to_numpy(float)[:n], 10)   # evaluation only
    rows = []
    mats = {"pool area (reference)": matrix(area, inputs, ids)}
    for variant in VARIANTS:
        f = OUT / f"states_{variant}.pt"
        if not f.exists():
            continue
        z = torch.load(f, weights_only=False)
        states[variant] = {s: uniform_filter1d(z[s].numpy()[:len(inputs[s])], 10, axis=0) for s in ids}
        mats[variant] = matrix(states[variant], inputs, ids)
    for name, M in mats.items():
        off = M.copy()
        np.fill_diagonal(off, np.nan)
        rows.append(dict(series=name, own=round(float(np.mean(np.diag(M))), 3),
                         transfer_mean=round(float(np.nanmean(off)), 3),
                         positive_pairs=int(np.nansum(off > 0)), pairs=off.size - len(ids),
                         best=round(float(np.nanmax(off)), 3)))
        print(f"\n=== {name} ===")
        print(pd.DataFrame(np.round(M, 2), index=ids, columns=[s[-2:] for s in ids]).to_string())
    print()
    print(pd.DataFrame(rows).to_string(index=False))
    json.dump(rows, open(OUT / "comparison.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "compare"])
    ap.add_argument("--variant", default="adain", choices=VARIANTS)
    ap.add_argument("--process", default="GTAW")
    a = ap.parse_args()
    if a.command == "train":
        train(a.process, a.variant)
    else:
        compare(a.process)
