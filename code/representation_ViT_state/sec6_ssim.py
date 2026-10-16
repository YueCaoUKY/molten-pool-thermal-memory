#!/usr/bin/env python3
"""Generate frames from the input alone for every record and score them against the recording.

For each of the 22 records and each variant, the identified model is rolled out from the first
state and the recorded travel speed, the pool-state trajectory is turned into images by the
structure decoder plus the retrieved detail layer, and every generated frame is compared with the
recorded frame at the same time by SSIM.

Nothing in the representation is retrained and `delta_study.retrieve` is used unchanged.

    python sec6_ssim.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sec6_experiments as X      # noqa: E402  loading, banks and decoding
import generator_figures as G    # noqa: E402
import visual_generator as V     # noqa: E402
from delta_study import retrieve  # noqa: E402

OUT = HERE / "results_sec6"
OUT.mkdir(exist_ok=True)
RECORDS = {
    "GTAW": ["Iden_01", "Iden_02", "Iden_06", "Iden_07", "Iden_08",
             "Iden_09", "Iden_10", "Iden_11", "Iden_12", "Iden_13"],
    "GMAW": ["exp_01", "exp_02", "exp_03", "exp_04", "exp_05", "exp_06", "exp_07",
             "exp_09", "exp_10"],
    "WAAM": ["video_1", "video_3", "video_5"],
}
VARIANTS = {"A": "A_m1", "B": "B_no_delta"}


def ssim_series(recorded: np.ndarray, generated: np.ndarray) -> np.ndarray:
    """SSIM of Wang et al. (2004) frame by frame, on grayscale images in [0, 1]."""
    return np.array([
        structural_similarity(r, np.clip(g, 0.0, 1.0), data_range=1.0,
                              gaussian_weights=True, sigma=1.5,
                              use_sample_covariance=False)
        for r, g in zip(recorded, generated)])


def run_record(process: str, sid: str, bz, bc, bounds, dec, vq, frames, keep: bool):
    rec, dt = X.record(process, sid)
    u = torch.tensor(rec["u"].astype(np.float32))
    real = frames[sid].float().numpy()[:, 0] / 255.0
    out, store = {}, {}
    for name, variant in VARIANTS.items():
        model, mean, std, _ = X.load_model(process, sid, variant)
        zn = torch.tensor(((rec["z"] - mean) / std).astype(np.float32))
        with torch.no_grad():
            z_hat, d_hat = model.rollout(zn[0], u, dt)
        raw = torch.tensor((z_hat.numpy() * std + mean).astype(np.float32))
        field = bc[retrieve(raw, bz, bc, bounds)].to(V.DEVICE)
        struct, full = X.decode(raw, dec, vq, field)
        n = min(len(real), len(full))
        s = ssim_series(real[:n], full[:n])
        out[name] = dict(mean=float(s.mean()), median=float(np.median(s)),
                         min=float(s.min()), max=float(s.max()), frames=int(n))
        if keep:
            store[f"{name}_gen"] = full[:n]
            store[f"{name}_struct"] = struct[:n]
            store[f"{name}_ssim"] = s
            store[f"{name}_z"] = raw.numpy()[:n]
            store[f"{name}_delta"] = d_hat.numpy()[:n]
    if keep:
        store.update(real=real[:n], z_obs=rec["z"][:n], speed=rec["speed"][:n],
                     t=np.arange(n) * dt, dt=dt)
        np.savez_compressed(OUT / f"gen_{process}_{sid}.npz", **store)
    return out


def main():
    best = {"GMAW": "exp_05", "GTAW": "Iden_02", "WAAM": "video_3"}
    rows, t0 = [], time.time()
    for process, sids in RECORDS.items():
        # the representation is trained per process, so the decoder and the codebook have to be
        # loaded after visual_generator has been pointed at that process
        bz, bc, bounds = X.bank(process, X.A.SMOOTH[process])
        V.configure(process)
        dec, vq, *_ = G.models()
        frames, _ = V.load_cache()
        for sid in sids:
            r = run_record(process, sid, bz, bc, bounds, dec, vq, frames,
                           keep=(sid == best[process]))
            rows.append(dict(process=process, record=sid,
                             ssim_A=r["A"]["mean"], ssim_B=r["B"]["mean"],
                             frames=r["A"]["frames"]))
            print(f"{process:5s} {sid:9s} SSIM  A {r['A']['mean']:.3f}  B {r['B']['mean']:.3f}",
                  flush=True)
    (OUT / "ssim_per_record.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
