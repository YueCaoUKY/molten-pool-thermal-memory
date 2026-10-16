#!/usr/bin/env python3
"""Identification and adaptation of the two-state model on the visual state (GTAW).

The state is the six numbers produced by the frozen transformer and the head trained in
visual_generator.py; nothing here changes the encoder. The model is the one settled in the
simulation section, with the state raised to six coordinates:

    dz/dt    = -gamma(z, u, delta) * (z - z_eq(u, delta))        gamma elementwise
    ddelta/dt = lambda * (phi(u, z) - delta)

Identification uses one record: free rollout from the input and the first state only.
Adaptation freezes the networks and re-estimates three positive scalars on the first 40 % of a
target record - the memory's starting value and two scales, one on what drives it and one on
its rate - and predicts the rest from the input.

    python node_visual.py fit --record Iden_07
    python node_visual.py all
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
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Representation Learning NODE zs - All Datasets" / "paper_section4"))
import experiment as E  # noqa: E402

import visual_generator as V  # noqa: E402

PROCESS = "GTAW"
OUT = HERE / "results_node"
SPEC = E.SPECS[PROCESS]
IDS = (SPEC.train_id, *SPEC.heldout_ids)
DT = SPEC.dt


def configure(process: str) -> None:
    """Point the module at one of the processes; the state files come from visual_generator."""
    global PROCESS, OUT, SPEC, IDS, DT
    PROCESS = process
    SPEC = E.SPECS[process]
    IDS = (SPEC.train_id, *SPEC.heldout_ids)
    DT = SPEC.dt
    OUT = HERE / ("results_node" if process == "GTAW" else f"results_node_{process}")
    OUT.mkdir(exist_ok=True)
GAMMA_MIN = 0.05
PREFIX = 0.40          # share of a target record used to calibrate the three scalars
OMEGA = 0.02           # weight of the pull back towards the source model
SPAN = 0.7             # the two scales stay inside exp(+-0.7)
SG_WINDOW = 21
SMOOTH = 0             # centred moving average applied to the state before fitting, in frames
TAG = ""
WIDTH = 16             # hidden width of the three networks
FIT_PREFIX = 1.0       # share of the record used for fitting; the rest is held out
EPOCHS = 250
USE_DELTA = True


# --------------------------------------------------------------------------- data
def _speed(sid: str) -> np.ndarray:
    """Travel speed for every frame the state was read from, in the recorded units."""
    if PROCESS == "GTAW":
        folder = E.DATA_ROOT / "GTAW_varying_speed" / sid
        info = pd.read_csv(folder / "info.txt", sep="\t")
        have = {int(p.stem.split("_")[1]) for p in folder.glob("images/cropped_*.jpg")}
        info = info.loc[info["index"].isin(have)].sort_values("index")
        return info["welding_speed"].to_numpy(dtype=np.float32)
    if PROCESS == "WAAM":
        import numpy as _np
        n = len(torch.load(V.OUT / "states.pt", weights_only=False)[sid])
        return _np.zeros(n, dtype=_np.float32)
    folder = E.DATA_ROOT / "GMAW_varying_speed" / sid
    info = pd.read_csv(folder / "data.csv")
    have = {int(p.stem.split("_")[-1]) for p in folder.glob("images/received_image_*.jpg")}
    info = info.loc[info["Index"].isin(have)].sort_values("Index").iloc[::2]
    return info["Speed"].to_numpy(dtype=np.float32)


def records() -> dict:
    """The recorded input and the state read from the images.

    With SMOOTH > 1 the state is passed through a centred moving average of that many frames
    before anything else. The filter is zero-phase, so it removes the frame-to-frame scatter
    without shifting the trajectory in time."""
    V.configure(PROCESS)
    z = torch.load(V.OUT / "states.pt", weights_only=False)
    out = {}
    for sid in IDS:
        v = _speed(sid)
        n = min(len(v), len(z[sid]))
        v = v[:n]
        u = (v - SPEC.u_min) / (SPEC.u_max - SPEC.u_min)
        zr = z[sid].numpy().astype(np.float32)[:n]
        zs = uniform_filter1d(zr, SMOOTH, axis=0, mode="nearest") if SMOOTH > 1 else zr
        out[sid] = dict(z=zs, z_raw=zr, u=u[:, None], speed=v)
    return out


def ceiling(z: np.ndarray) -> float:
    """Share of the variance that survives a one second moving average.

    The state is read from each frame on its own, so what is left after this filter is the part
    any model driven by the input alone could reproduce."""
    zn = (z - z.mean(0)) / np.maximum(z.std(0), 1e-6)
    slow = uniform_filter1d(zn, int(round(1.0 / DT)), axis=0)
    return float(slow.var(0).sum() / zn.var(0).sum())


def r2(obs: np.ndarray, pred: np.ndarray) -> float:
    m, s = obs.mean(0), np.maximum(obs.std(0), 1e-6)
    o, p = (obs - m) / s, (pred - m) / s
    ok = np.isfinite(p).all(1)
    return float(1 - ((o[ok] - p[ok]) ** 2).sum() / (o[ok] ** 2).sum())


# --------------------------------------------------------------------------- model
class MLP(nn.Module):
    def __init__(self, n_in, n_out, width=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, n_out))

    def forward(self, x):
        return self.net(x)


class PoolNODE(nn.Module):
    """The settled design: a relaxation towards an input dependent equilibrium, slowed or
    speeded by a scalar memory state that is itself driven by the input and the pool."""

    def __init__(self, z_dim=6, u_dim=1, width=16, use_delta=True):
        super().__init__()
        self.use_delta = use_delta
        d = 1 if use_delta else 0
        self.z_eq_net = MLP(u_dim + d, z_dim, width)
        self.gamma_net = MLP(z_dim + u_dim + d, z_dim, width)
        self.phi_net = MLP(u_dim + z_dim, 1, width) if use_delta else None
        self.raw_lam = nn.Parameter(torch.tensor(-2.5))

    @property
    def lam(self):
        return F.softplus(self.raw_lam)

    @staticmethod
    def feature(delta):
        return (delta / (1.0 + delta)).unsqueeze(-1)

    def terms(self, z, delta, u):
        if not self.use_delta:
            z_eq = self.z_eq_net(u)
            gamma = GAMMA_MIN + F.softplus(self.gamma_net(torch.cat([z, u], -1)))
            return z_eq, gamma, torch.zeros_like(delta)
        f = self.feature(delta)
        z_eq = self.z_eq_net(torch.cat([u, f], -1))
        gamma = GAMMA_MIN + F.softplus(self.gamma_net(torch.cat([z, u, f], -1)))
        phi = F.softplus(self.phi_net(torch.cat([u, z], -1)).squeeze(-1))
        return z_eq, gamma, phi

    def rates(self, z, delta, u):
        z_eq, gamma, phi = self.terms(z, delta, u)
        return -gamma * (z - z_eq), self.lam * (phi - delta)

    def rollout(self, z0, u, dt=DT, d0=None, a=1.0, b=1.0):
        z = z0
        delta = torch.zeros(()) if d0 is None else d0
        zs, ds = [], []
        for k in range(u.shape[0]):
            zs.append(z)
            ds.append(delta)
            if k == u.shape[0] - 1:
                break
            z_eq, gamma, phi = self.terms(z, delta, u[k])
            z = z_eq + (z - z_eq) * torch.exp(-gamma * dt)
            delta = delta + dt * b * self.lam * (a * phi - delta)
        return torch.stack(zs), torch.stack(ds)


# --------------------------------------------------------------------------- identification
def fit(sid: str, data: dict, epochs=None, pretrain=1000, lr=3e-3, seed=42):
    epochs = EPOCHS if epochs is None else epochs
    torch.manual_seed(seed)
    np.random.seed(seed)
    rec = data[sid]
    z_full, u_full = rec["z"], rec["u"]
    n_fit = len(z_full) if FIT_PREFIX >= 1.0 else int(FIT_PREFIX * len(z_full))
    z_raw, u_np = z_full[:n_fit], u_full[:n_fit]
    mean, std = z_raw.mean(0), np.maximum(z_raw.std(0), 1e-5)
    zn = ((z_raw - mean) / std).astype(np.float32)
    z, u = torch.tensor(zn), torch.tensor(u_np.astype(np.float32))
    model = PoolNODE(z_dim=z.shape[1], u_dim=u.shape[1], width=WIDTH, use_delta=USE_DELTA)

    # derivative pre-training, with the memory seeded by a first-order response to the heat
    # input per unit length, which falls as the speed rises
    dz = torch.tensor(np.gradient(savgol_filter(zn, SG_WINDOW, 3, axis=0), DT, axis=0).astype(np.float32))
    seed_d, acc = [], 0.0
    for k in range(len(u)):
        seed_d.append(acc)
        acc = acc + DT * 0.05 * (float(1.0 - u_np[k, 0]) - acc)
    seed_d = torch.tensor(np.asarray(seed_d, dtype=np.float32))
    pre = torch.optim.AdamW(list(model.z_eq_net.parameters()) + list(model.gamma_net.parameters()),
                            lr=lr, weight_decay=1e-5)
    for _ in range(pretrain):
        pre.zero_grad(set_to_none=True)
        dz_hat, _ = model.rates(z, seed_d, u)
        loss = F.mse_loss(dz_hat, dz)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        pre.step()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.05 * lr)
    best, best_state, hist = float("inf"), None, []
    t0 = time.time()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        z_hat, _ = model.rollout(z[0], u)
        loss = F.mse_loss(z_hat, z)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        hist.append(float(loss))
        if float(loss) < best:
            best, best_state = float(loss), {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 50 == 0:
            print(f"  {sid} epoch {epoch+1:4d}  mse {float(loss):.4f}  {time.time()-t0:.0f}s", flush=True)
    model.load_state_dict(best_state)
    with torch.no_grad():
        z_hat, d_hat = model.rollout(z[0], u)
    OUT.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "hist": hist},
               OUT / f"fit_{sid}{TAG}.pt")
    pred = z_hat.numpy() * std + mean
    held = None
    if n_fit < len(z_full):
        zn_all = torch.tensor(((z_full - mean) / std).astype(np.float32))
        u_all = torch.tensor(u_full.astype(np.float32))
        with torch.no_grad():
            z_all, _ = model.rollout(zn_all[0], u_all)
        p_all = z_all.numpy() * std + mean
        held = float(r2(z_full[n_fit:], p_all[n_fit:]))
    np.savez_compressed(OUT / f"fit_{sid}{TAG}.npz", z_true=z_raw, z_raw=rec["z_raw"], z_pred=pred,
                        delta=d_hat.numpy(), u=u_np, speed=rec["speed"])
    return model, mean, std, (float(r2(z_raw, pred)),
                              float(r2(rec["z_raw"][:n_fit], pred)), held)


# --------------------------------------------------------------------------- adaptation
def adapt(source: str, target: str, data: dict, steps=300, norm="window", scalars=3):
    """norm: how the target's state is put into the source model's coordinates.

    "source"  the source record's mean and spread, as in the simulation section, where the
              states are physical dimensions and mean the same thing in every record;
    "window"  the target's own mean and spread, estimated on the calibration window only.
              The visual state has no physical unit and its level drifts between records, so
              this is the reading of the same sensor rather than an extra model parameter.

    scalars: 3 calibrates the memory alone (d0, a, b); 4 adds c, a gain on the travel speed.
             An unrecorded change of arc power acts on the heat input per unit length in the
             same way as a change of speed, since that quantity is P / v, so one gain stands
             for whatever heat input the record actually received."""
    ck = torch.load(OUT / f"fit_{source}{TAG}.pt", weights_only=False)
    model = PoolNODE(z_dim=6, u_dim=1, width=WIDTH, use_delta=USE_DELTA)
    model.load_state_dict(ck["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    rec = data[target]
    z_raw, u_np = rec["z"], rec["u"]
    n_pre = max(20, int(PREFIX * len(z_raw)))
    if norm == "window":
        mean = z_raw[:n_pre].mean(0)
        std = np.maximum(z_raw[:n_pre].std(0), 1e-5)
    else:
        mean, std = ck["mean"], ck["std"]
    zn = torch.tensor(((z_raw - mean) / std).astype(np.float32))
    u = torch.tensor(u_np.astype(np.float32))
    speed = torch.tensor(rec["speed"].astype(np.float32))
    span = float(SPEC.u_max - SPEC.u_min)

    def input_at(c):
        return ((c * speed - SPEC.u_min) / span).unsqueeze(-1)

    raw = torch.tensor([-3.0, 0.0, 0.0, 0.0], requires_grad=True)
    opt = torch.optim.AdamW([raw], lr=0.05)
    one = torch.ones(())

    def cal(r):
        d0 = F.softplus(r[0])
        a, b = torch.exp(SPAN * torch.tanh(r[1])), torch.exp(SPAN * torch.tanh(r[2]))
        c = torch.exp(SPAN * torch.tanh(r[3])) if scalars == 4 else one
        return d0, a, b, c

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        d0, a, b, c = cal(raw)
        z_hat, _ = model.rollout(zn[0], input_at(c)[:n_pre], DT, d0, a, b)
        loss = F.mse_loss(z_hat, zn[:n_pre]) + OMEGA * (
            torch.log(a) ** 2 + torch.log(b) ** 2 + torch.log(c) ** 2 + d0 ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        d0, a, b, c = cal(raw)
        z_ad, d_ad = model.rollout(zn[0], input_at(c), DT, d0, a, b)
        z_no, d_no = model.rollout(zn[0], u, DT, torch.zeros(()), one, one)
    to_raw = lambda x: x.numpy() * std + mean
    own = np.load(OUT / f"fit_{target}{TAG}.npz")["z_pred"]
    held = slice(n_pre, len(zn))
    np.savez_compressed(OUT / f"adapt_{source}_to_{target}_{norm}{TAG}_n{scalars}.npz", n_pre=n_pre, z_true=z_raw,
                        z_source=to_raw(z_no), z_adapted=to_raw(z_ad), z_own=own,
                        delta_adapted=d_ad.numpy(), delta_source=d_no.numpy(),
                        u=u_np, speed=rec["speed"],
                        calib=np.array([float(d0), float(a), float(b), float(c)]))
    return dict(target=target, norm=norm, scalars=scalars, frames=len(zn), prefix_s=round(n_pre * DT, 1),
                d0=round(float(d0), 3), drive=round(float(a), 3), rate=round(float(b), 3),
                speed_gain=round(float(c), 3),
                r2_none=round(r2(z_raw[held], to_raw(z_no)[held]), 3),
                r2_adapted=round(r2(z_raw[held], to_raw(z_ad)[held]), 3),
                r2_own=round(r2(z_raw[held], own[held]), 3),
                r2_adapted_raw=round(r2(rec["z_raw"][held], to_raw(z_ad)[held]), 3),
                ceiling=round(ceiling(rec["z_raw"][held]), 3))


def source_record(data: dict) -> str:
    """Stated in advance: the record whose input covers the widest range; ties go to the one
    with the most changes of level."""
    def key(sid):
        v = data[sid]["speed"]
        changes = int((np.abs(np.diff(v)) > 1e-6).sum())
        return (round(float(v.max() - v.min()), 3), changes)
    return max(IDS, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fit", "all", "source", "adapt"])
    ap.add_argument("--record", default=None)
    ap.add_argument("--from", dest="src", default=None, help="source record for adapt")
    ap.add_argument("--to", nargs="+", default=None, help="target records for adapt")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--process", default="GTAW", choices=["GTAW", "GMAW", "WAAM"])
    ap.add_argument("--no-delta", action="store_true",
                    help="ablation: the same structure without the memory state")
    ap.add_argument("--fit-prefix", type=float, default=1.0,
                    help="fit on this share of the record and hold out the rest")
    ap.add_argument("--smooth", type=int, default=0,
                    help="centred moving average on the state, in frames (10 = 1 s)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    configure(a.process)
    globals()["SMOOTH"] = a.smooth
    globals()["WIDTH"], globals()["EPOCHS"] = a.width, a.epochs
    globals()["FIT_PREFIX"] = a.fit_prefix
    globals()["USE_DELTA"] = not a.no_delta
    globals()["TAG"] = (f"_s{a.smooth}" if a.smooth > 1 else "") + a.suffix
    data = records()
    if a.command == "source":
        for sid in IDS:
            v = data[sid]["speed"]
            print(f"{sid}: range {v.max()-v.min():.2f}  changes {int((np.abs(np.diff(v))>1e-6).sum())}"
                  f"  T {len(v)}  ceiling {ceiling(data[sid]['z']):.3f}")
        print("source ->", source_record(data))
        return
    if a.command == "fit":
        sid = a.record or source_record(data)
        _, _, _, score = fit(sid, data)
        extra = "" if score[2] is None else f"  held-out {score[2]:.3f}"
        print(f"{sid}: identification R2 {score[0]:.3f} (filtered) {score[1]:.3f} "
              f"(unfiltered){extra}")
        return
    if a.command == "adapt":
        rows = [adapt(a.src, t, data, norm="window", scalars=n)
                for n in (3, 4) for t in a.to]
        df = pd.DataFrame(rows)
        pd.set_option("display.width", 200)
        print(df.to_string(index=False))
        out = OUT / f"adaptation_from_{a.src}{TAG}.json"
        json.dump({"source": a.src, "smooth": SMOOTH, "rows": rows}, open(out, "w"), indent=1)
        return
    src = source_record(data)
    print(f"source record: {src}", flush=True)
    for sid in IDS:
        if not (OUT / f"fit_{sid}{TAG}.npz").exists():
            _, _, _, score = fit(sid, data)
            print(f"{sid}: identification R2 {score[0]:.3f} (filtered) "
                  f"{score[1]:.3f} (unfiltered)", flush=True)
    rows = [adapt(src, t, data, norm=n) for n in ("source", "window") for t in IDS if t != src]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    json.dump({"source": src, "smooth": SMOOTH, "rows": rows},
              open(OUT / f"adaptation{TAG}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
