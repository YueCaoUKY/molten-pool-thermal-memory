#!/usr/bin/env python3
"""Thumbnails for Figure 8, all from one GMAW record (exp_05).

The representation is trained per process, so visual_generator is pointed at GMAW before the
decoder and the codebook are loaded."""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generator_figures as G      # noqa: E402
import visual_generator as V       # noqa: E402

OUT = HERE / "figures_sec6" / "fig8_v3"
OUT.mkdir(parents=True, exist_ok=True)
SID, FRAME, SIDE = "exp_05", 150, 300


def png(a, name, lo=None, hi=None):
    a = np.asarray(a, dtype=np.float32)
    lo = float(a.min()) if lo is None else lo
    hi = float(a.max()) if hi is None else hi
    a = (a - lo) / max(hi - lo, 1e-9)
    Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), mode="L") \
         .resize((SIDE, SIDE), Image.LANCZOS).save(OUT / f"{name}.png")


V.configure("GMAW")
frames, _ = V.load_cache()
codes = torch.load(V.OUT / "codes.pt", weights_only=False)
states = torch.load(V.OUT / "states.pt", weights_only=False)
dec, vq, *_ = G.models()

real = frames[SID].float().numpy()[FRAME, 0] / 255.0
with torch.no_grad():
    struct = dec(states[SID][FRAME:FRAME + 1].float().to(V.DEVICE)).cpu().numpy()[0, 0]
    detail = vq.decode_indices(codes[SID][FRAME:FRAME + 1].to(V.DEVICE)).cpu().numpy()[0, 0]

png(real, "frame", 0.0, 1.0)
png(struct, "structure", 0.0, 1.0)
png(real - struct, "residual")                       # its own range; it is signed
png(codes[SID][FRAME].numpy(), "codes")              # grey, not a rainbow
png(struct + detail, "reconstruction", 0.0, 1.0)

d = np.load(HERE / "results_sec6" / "gen_GMAW_exp_05.npz")
idx = np.linspace(0, len(d["A_gen"]) - 1, 3).round().astype(int)
for i, k in enumerate(idx):
    png(d["A_gen"][k], f"gen{i}", 0.0, 1.0)
(OUT / "times.txt").write_text(" ".join(f"{d['t'][k]:.0f}" for k in idx) + "\n")
print("thumbnails ->", OUT, "times", [round(float(d["t"][k])) for k in idx])
