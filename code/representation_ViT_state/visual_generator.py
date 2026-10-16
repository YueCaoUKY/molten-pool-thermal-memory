#!/usr/bin/env python3
"""Visual state extractor and image generator for GTAW (no dynamic model involved).

Three stages, trained one after another, each on the frozen result of the previous one:

  A  frozen DINOv2 patch tokens -> small head -> state z (6 numbers)
     -> structure decoder D(z) = mean frame + deviation.      Loss: || I - D(z) ||^2
  B  residual r = I - D(z) -> VQ-VAE.                         Loss: |r - r_hat|_1 + codebook
  C  code transition p(c_t | c_{t-1}, z_t), one shared MLP per grid position.
                                                              Loss: cross entropy

With images the state and the true detail are both available, so a frame is reproduced exactly.
Without images the detail has to be sampled from stage C, conditioned on the state.

    python visual_generator.py cache
    python visual_generator.py stage-a
    python visual_generator.py stage-b
    python visual_generator.py stage-c
    python visual_generator.py figures
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SECTION4 = HERE.parent / "Representation Learning NODE zs - All Datasets" / "paper_section4"
sys.path.insert(0, str(SECTION4))
import experiment as E  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = HERE / "cache"

DIM = 6             # state dimension, fixed
GRID = 8            # patch grid the head reads
CODES, CODE_DIM = 512, 32                 # codebook size and embedding width

# Training resolution per process: close to the native frame size, never upsampled.
# GTAW 200 x 200, GMAW 512 x 512, WAAM already cached at 128 x 128.
SIZES = {"GTAW": 192, "GMAW": 192, "WAAM": 192}
PROCESS = "GTAW"
SPEC = E.SPECS[PROCESS]
IDS = (SPEC.train_id, *SPEC.heldout_ids)
SIZE = SIZES[PROCESS]
OUT = HERE / "results_generator"


def configure(process: str) -> None:
    """Point the module at one of the three processes."""
    global PROCESS, SPEC, IDS, SIZE, OUT
    PROCESS = process
    SPEC = E.SPECS[process]
    IDS = (SPEC.train_id, *SPEC.heldout_ids)
    SIZE = SIZES[process]
    OUT = HERE / ("results_generator" if process == "GTAW" else f"results_generator_{process}")


# --------------------------------------------------------------------------- data
def _paths():
    tag = "" if PROCESS == "GTAW" else f"_{PROCESS}"
    return CACHE / f"frames{SIZE}{tag}.pt", CACHE / f"patch{GRID}{tag}.pt"


def _load_images(sid: str):
    loader = {"GTAW": E._load_gtaw, "GMAW": E._load_gmaw, "WAAM": E._load_waam}[PROCESS]
    return loader(sid, SIZE)


def build_cache() -> None:
    CACHE.mkdir(exist_ok=True)
    frames, feats = {}, {}
    for sid in IDS:
        images, _ = _load_images(sid)
        frames[sid] = images                                    # (T, 1, S, S) uint8
        blob = torch.load(E._feature_path(SPEC, sid), map_location="cpu", weights_only=False)
        p = blob["patches"].float().permute(0, 3, 1, 2)          # (T, 768, 16, 16)
        del blob
        feats[sid] = F.avg_pool2d(p, 16 // GRID).half().contiguous()
        del p
        print(f"{sid}: {len(images)} frames", flush=True)
    fp, pp = _paths()
    torch.save(frames, fp)
    torch.save(feats, pp)


def load_cache():
    fp, pp = _paths()
    return torch.load(fp, weights_only=False), torch.load(pp, weights_only=False)


def flatten(d):
    """One tensor over all records, plus the slice belonging to each record."""
    parts, spans, start = [], {}, 0
    for sid in IDS:
        x = d[sid]
        parts.append(x)
        spans[sid] = (start, start + len(x))
        start += len(x)
    return torch.cat(parts), spans


# --------------------------------------------------------------------------- stage A
class Head(nn.Module):
    """Patch grid -> state. The transformer itself is frozen and not part of this module."""

    def __init__(self, in_ch=768, width=128, dim=DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, 2, 1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(width * 4, 256), nn.SiLU(),
            nn.Linear(256, dim))

    def forward(self, x):
        return self.net(x)


def up_block(a, b):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                         nn.Conv2d(a, b, 3, 1, 1), nn.GroupNorm(8, b), nn.SiLU())


class StructureDecoder(nn.Module):
    """State -> the part of the image that the state can explain, as a deviation from the mean."""

    def __init__(self, dim=DIM, base=256, mean_frame=None, size=None):
        super().__init__()
        self.start = (size or SIZE) // 32
        self.fc = nn.Linear(dim, base * self.start * self.start)
        self.net = nn.Sequential(up_block(base, 128), up_block(128, 64), up_block(64, 32),
                                 up_block(32, 32), up_block(32, 16), nn.Conv2d(16, 1, 3, 1, 1))
        self.register_buffer("mean_frame", mean_frame)
        self.base = base

    def forward(self, z):
        h = self.fc(z).view(-1, self.base, self.start, self.start)
        return self.mean_frame + self.net(h)


def stage_a(steps=8000, batch=32, lr=2e-4):
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(0)
    frames, feats = load_cache()
    img, spans = flatten(frames)
    fea, _ = flatten(feats)
    n = len(img)
    mean_frame = (img.float() / 255.0).mean(0, keepdim=True).to(DEVICE)
    head = Head().to(DEVICE)
    dec = StructureDecoder(mean_frame=mean_frame).to(DEVICE)
    opt = torch.optim.AdamW(list(head.parameters()) + list(dec.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch,))
        x = fea[idx].to(DEVICE).float()
        y = img[idx].to(DEVICE).float() / 255.0
        loss = F.mse_loss(dec(head(x)), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"A {step:5d}  mse {loss.item():.5f}  {time.time()-t0:.0f}s", flush=True)
    torch.save({"head": head.state_dict(), "dec": dec.state_dict(),
                "mean_frame": mean_frame.cpu()}, OUT / "stage_a.pt")
    # state trajectories, for later stages and for the figures
    head.eval()
    zs = {}
    with torch.no_grad():
        for sid in IDS:
            a, b = spans[sid]
            zs[sid] = torch.cat([head(fea[a:b][i:i + 256].to(DEVICE).float()).cpu()
                                 for i in range(0, b - a, 256)])
    torch.save(zs, OUT / "states.pt")
    print({k: tuple(v.shape) for k, v in zs.items()})


# --------------------------------------------------------------------------- stage B
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1),
                                 nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 1))

    def forward(self, x):
        return x + self.net(x)


class VectorQuantiser(nn.Module):
    """Codebook with an exponential moving average, the usual formulation."""

    def __init__(self, codes=CODES, dim=CODE_DIM, decay=0.99, eps=1e-5, dead=0.5):
        super().__init__()
        self.decay, self.eps, self.dead = decay, eps, dead
        emb = torch.randn(codes, dim) * 0.1
        self.register_buffer("embedding", emb)
        self.register_buffer("cluster_size", torch.ones(codes))
        self.register_buffer("ema_w", emb.clone())
        self.register_buffer("started", torch.zeros(()))

    def _refill(self, flat):
        """Entries that no longer win any position are moved onto encoder outputs.

        Without this the codebook collapses onto a handful of entries: the ones that are far
        from the data at initialisation are never selected and never move."""
        idle = self.cluster_size < self.dead
        if not bool(idle.any()):
            return
        pick = torch.randint(0, len(flat), (int(idle.sum()),), device=flat.device)
        fresh = flat[pick] + 0.01 * torch.randn_like(flat[pick])
        self.embedding[idle] = fresh
        self.ema_w[idle] = fresh
        self.cluster_size[idle] = 1.0

    def lookup(self, indices):
        return F.embedding(indices, self.embedding)

    def forward(self, z):                                  # z: (B, D, H, W)
        b, d, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, d)
        if self.training and self.started == 0:            # start from the data, not from noise
            with torch.no_grad():
                pick = torch.randint(0, len(flat), (len(self.embedding),), device=flat.device)
                self.embedding.copy_(flat[pick])
                self.ema_w.copy_(self.embedding)
                self.started.fill_(1.0)
        dist = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embedding.T
                + self.embedding.pow(2).sum(1))
        idx = dist.argmin(1)
        q = self.lookup(idx).view(b, h, w, d).permute(0, 3, 1, 2)
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, len(self.embedding)).type(flat.dtype)
                self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                self.ema_w.mul_(self.decay).add_(onehot.T @ flat, alpha=1 - self.decay)
                nn_ = self.cluster_size.sum()
                size = (self.cluster_size + self.eps) / (nn_ + len(self.embedding) * self.eps) * nn_
                self.embedding.copy_(self.ema_w / size.unsqueeze(1))
                self._refill(flat)
        commit = F.mse_loss(z, q.detach())
        q = z + (q - z).detach()
        return q, commit, idx.view(b, h, w)


class ResidualVQ(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, ch, 4, 2, 1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 4, 2, 1), ResBlock(ch), ResBlock(ch), nn.Conv2d(ch, CODE_DIM, 1))
        self.vq = VectorQuantiser()
        self.dec = nn.Sequential(
            nn.Conv2d(CODE_DIM, ch, 3, 1, 1), ResBlock(ch), ResBlock(ch),
            up_block(ch, ch), up_block(ch, 64), up_block(64, 32), nn.Conv2d(32, 1, 3, 1, 1))

    def encode(self, r):
        return self.vq(self.enc(r))

    def decode_indices(self, idx):
        q = self.vq.lookup(idx).permute(0, 3, 1, 2)
        return self.dec(q)

    def forward(self, r):
        q, commit, idx = self.encode(r)
        return self.dec(q), commit, idx


def stage_b(steps=12000, batch=24, lr=2e-4, beta=0.25):
    torch.manual_seed(0)
    frames, feats = load_cache()
    img, spans = flatten(frames)
    ck = torch.load(OUT / "stage_a.pt", weights_only=False)
    dec_s = StructureDecoder(mean_frame=ck["mean_frame"].to(DEVICE)).to(DEVICE)
    dec_s.load_state_dict(ck["dec"])
    dec_s.eval()
    zs, _ = flatten(torch.load(OUT / "states.pt", weights_only=False))
    net = ResidualVQ().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    n, t0 = len(img), time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch,))
        y = img[idx].to(DEVICE).float() / 255.0
        with torch.no_grad():
            r = y - dec_s(zs[idx].to(DEVICE))
        rhat, commit, _ = net(r)
        loss = F.l1_loss(rhat, r) + beta * commit
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"B {step:5d}  l1 {F.l1_loss(rhat, r).item():.5f}  commit {commit.item():.5f}"
                  f"  {time.time()-t0:.0f}s", flush=True)
    torch.save({"net": net.state_dict()}, OUT / "stage_b.pt")
    # store the code field of every frame for stage C
    net.eval()
    codes = {}
    with torch.no_grad():
        for sid in IDS:
            a, b = spans[sid]
            out = []
            for i in range(a, b, 64):
                y = img[i:min(i + 64, b)].to(DEVICE).float() / 255.0
                r = y - dec_s(zs[i:min(i + 64, b)].to(DEVICE))
                out.append(net.encode(r)[2].cpu())
            codes[sid] = torch.cat(out)
    torch.save(codes, OUT / "codes.pt")
    print({k: tuple(v.shape) for k, v in codes.items()})


# --------------------------------------------------------------------------- stage C
class CodePrior(nn.Module):
    """p(c_t | c_{t-1}, z_t): a small convolution over the previous code field.

    Each position sees a neighbourhood of the previous field rather than only itself, so a
    boundary already present in the field is carried forward instead of being resampled.
    """

    def __init__(self, grid, codes=CODES, dim=DIM, width=256):
        super().__init__()
        self.emb = nn.Embedding(codes, 64)
        self.pos = nn.Parameter(torch.zeros(1, 32, grid, grid))
        nn.init.normal_(self.pos, std=0.02)
        self.net = nn.Sequential(
            nn.Conv2d(64 + 32 + dim, width, 3, 1, 1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, 1, 1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, 1, 1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, codes, 1))

    def forward(self, prev, z):                            # prev: (B, H, W), z: (B, d)
        b, h, w = prev.shape
        e = self.emb(prev).permute(0, 3, 1, 2)
        zz = z.view(b, -1, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([e, self.pos.expand(b, -1, -1, -1), zz], 1)
        return self.net(x).permute(0, 2, 3, 1)             # (B, H, W, codes)


def stage_c(steps=12000, batch=16, lr=3e-4):
    torch.manual_seed(0)
    codes = torch.load(OUT / "codes.pt", weights_only=False)
    zs = torch.load(OUT / "states.pt", weights_only=False)
    zmean = torch.cat([zs[s] for s in IDS]).mean(0)
    zstd = torch.cat([zs[s] for s in IDS]).std(0).clamp(min=1e-6)
    pairs = [(sid, t) for sid in IDS for t in range(1, len(codes[sid]))]
    grid = codes[IDS[0]].shape[-1]
    net = CodePrior(grid).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for step in range(1, steps + 1):
        sel = [pairs[i] for i in torch.randint(0, len(pairs), (batch,)).tolist()]
        prev = torch.stack([codes[s][t - 1] for s, t in sel]).to(DEVICE)
        cur = torch.stack([codes[s][t] for s, t in sel]).to(DEVICE)
        z = torch.stack([(zs[s][t] - zmean) / zstd for s, t in sel]).to(DEVICE)
        logits = net(prev, z)
        loss = F.cross_entropy(logits.reshape(-1, CODES), cur.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            acc = (logits.argmax(-1) == cur).float().mean().item()
            print(f"C {step:5d}  ce {loss.item():.4f}  copy-acc {acc:.3f}  {time.time()-t0:.0f}s",
                  flush=True)
    torch.save({"net": net.state_dict(), "zmean": zmean, "zstd": zstd, "grid": grid},
               OUT / "stage_c.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["cache", "stage-a", "stage-b", "stage-c", "all"])
    ap.add_argument("--process", default="GTAW", choices=list(SIZES))
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    configure(a.process)
    if a.command == "all":
        build_cache(); stage_a(); stage_b(); stage_c(); raise SystemExit(0)
    if a.command == "cache":
        build_cache()
    elif a.command == "stage-a":
        stage_a(**({"steps": a.steps} if a.steps else {}))
    elif a.command == "stage-b":
        stage_b(**({"steps": a.steps} if a.steps else {}))
    else:
        stage_c(**({"steps": a.steps} if a.steps else {}))
