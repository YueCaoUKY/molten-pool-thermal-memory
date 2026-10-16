#!/usr/bin/env python3
"""Per-record reconstruction quality and state smoothness for one process."""
import json
import sys

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

import generator_figures as G
import visual_generator as V


def summarise(process: str) -> dict:
    G.configure(process)
    frames, _ = V.load_cache()
    zs = torch.load(V.OUT / "states.pt", weights_only=False)
    dec, vq, prior, zmean, zstd = G.models()
    gm = torch.load(V.OUT / "stage_a.pt", weights_only=False)["mean_frame"].cpu()
    rows = []
    with torch.no_grad():
        for sid in V.IDS:
            img = frames[sid].float() / 255.0
            z = zs[sid]
            s = torch.cat([dec(z[i:i + 128].to(V.DEVICE)).cpu() for i in range(0, len(img), 128)])
            full = []
            for i in range(0, len(img), 64):
                r = img[i:i + 64].to(V.DEVICE) - s[i:i + 64].to(V.DEVICE)
                full.append((s[i:i + 64].to(V.DEVICE) + vq(r)[0]).cpu())
            full = torch.cat(full)
            rows.append(dict(record=sid, frames=len(img),
                             struct_r2=round(float(1 - ((img - s) ** 2).sum() / ((img - gm) ** 2).sum()), 3),
                             struct_mae=round(float((img - s).abs().mean()), 4),
                             full_mae=round(float((img - full).abs().mean()), 4)))
    zc = torch.cat([zs[s] for s in V.IDS]).numpy()
    mu, sd = zc.mean(0), zc.std(0)
    win = max(2, int(round(1.0 / V.SPEC.dt)))        # one second
    ac1, slow = [], []
    for sid in V.IDS:
        zn = (zs[sid].numpy() - mu) / sd
        ac1.append([np.corrcoef(zn[:-1, k], zn[1:, k])[0, 1] for k in range(V.DIM)])
        slow.append(uniform_filter1d(zn, win, axis=0).var(0) / zn.var(0))
    ac1, slow = np.mean(ac1, 0), np.mean(slow, 0)
    codes = torch.load(V.OUT / "codes.pt", weights_only=False)
    allc = torch.cat([v.reshape(-1) for v in codes.values()])
    out = dict(process=process, per_record=rows,
               lag1_autocorr=[round(float(x), 2) for x in ac1],
               slow_share=[round(float(x), 2) for x in slow],
               codes_used=int(len(torch.unique(allc))),
               code_grid=list(codes[V.IDS[0]].shape[1:]))
    json.dump(out, open(V.OUT / "summary.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    for p in (sys.argv[1:] or ["GTAW", "GMAW", "WAAM"]):
        r = summarise(p)
        print(f"\n=== {p}  codes {r['codes_used']}/512  grid {r['code_grid']} ===")
        print(f"{'record':10s} {'frames':>7s} {'struct R2':>10s} {'struct MAE':>11s} {'full MAE':>9s}")
        for d in r["per_record"]:
            print(f"{d['record']:10s} {d['frames']:7d} {d['struct_r2']:10.3f} "
                  f"{d['struct_mae']:11.4f} {d['full_mae']:9.4f}")
        print("lag-1 autocorr :", r["lag1_autocorr"])
        print("1 s slow share :", r["slow_share"])
