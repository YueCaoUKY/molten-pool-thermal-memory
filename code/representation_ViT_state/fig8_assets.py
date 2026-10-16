#!/usr/bin/env python3
"""Thumbnails for Figure 8: one recorded GTAW frame through the representation, and four
frames generated from the input alone. Written as PNGs for the SVG to embed."""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import node_aligned as A           # noqa: E402
import generator_figures as G      # noqa: E402
import visual_generator as V       # noqa: E402

OUT = HERE / "figures_sec6" / "fig8"
OUT.mkdir(parents=True, exist_ok=True)
SID, FRAME = "Iden_02", 180


def png(arr, name, cmap=None):
    a = np.asarray(arr, dtype=np.float32)
    if cmap is None:
        a = np.clip(a, 0, 1)
        im = Image.fromarray((a * 255).astype(np.uint8), mode="L")
    else:
        from matplotlib import colormaps
        a = (a - a.min()) / max(float(np.ptp(a)), 1e-9)
        im = Image.fromarray((colormaps[cmap](a)[..., :3] * 255).astype(np.uint8))
    im.resize((256, 256), Image.NEAREST).save(OUT / f"{name}.png")


V.configure("GTAW")
frames, _ = V.load_cache()
codes = torch.load(V.OUT / "codes.pt", weights_only=False)
states = torch.load(V.OUT / "states.pt", weights_only=False)
dec, vq, *_ = G.models()

real = frames[SID].float().numpy()[FRAME, 0] / 255.0
z = states[SID][FRAME:FRAME + 1].float()
with torch.no_grad():
    struct = dec(z.to(V.DEVICE)).cpu().numpy()[0, 0]
    field = codes[SID][FRAME:FRAME + 1].to(V.DEVICE)
    detail = vq.decode_indices(field).cpu().numpy()[0, 0]
png(real, "frame")
png(struct, "structure")
png(codes[SID][FRAME].numpy(), "codes", cmap="turbo")
png(struct + detail, "reconstruction")

d = np.load(HERE / "results_sec6" / "3a_GMAW_exp_05.npz")
idx = np.linspace(0, len(d["gen"]) - 1, 4).round().astype(int)
for i, k in enumerate(idx):
    png(d["gen"][k], f"gen{i}")
print("thumbnails ->", OUT)
