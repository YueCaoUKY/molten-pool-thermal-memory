#!/usr/bin/env python3
"""Quick adaptation test (owner's request, 2026-09-15). No new simulations.

Manuscript section 6: the shared networks stay fixed for a new experiment and
only a few positive calibration scalars are re-estimated. Here, for a source
model and a target run, those scalars are

  d0   the memory state's starting value (>= 0)
  a    a scale on what drives the memory (alpha for the proposed δ, φ for the
       pool-driven state)
  b    a scale on the memory's rate (beta, or lambda)

They are fitted on the first 40 % of the target record; the rest is predicted
from the input alone. Each pair is reported against two references:

  none   the source model used unchanged (d0 = 0, a = b = 1)
  full   the target's own identification (upper bound, already computed)

The pool state is standardised with the source statistics and started from the
target's first observed frame, as in the manuscript. R² is computed on the
held-out part, in the target's own standardisation, so the three models are
comparable.

    python adapt_test.py --source NB_2_random --targets NB_1_const NB_3_random --variant pool
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F

import identify_3d as I
from delta_variants import TAG, VARIANTS, VariantNODE3D, ring_heat, window_for

PREFIX = 0.40          # share of the target record used for calibration
STEPS = 300
SPAN = 0.7          # the two scales stay inside exp(±0.7), i.e. 0.5x to 2x the source value
REG = 0.02          # weak pull back toward the source model, as in the manuscript's adaptation

# Calibration designs compared for section 4. "full" is the manuscript's current one; the others
# drop, one at a time, the loss threshold, the regulariser with its weight and reference scale,
# and each of the two scales.
MODES = {
    "full":    dict(loss="huber", reg=REG, d0="ref",      free="ab"),
    "simple":  dict(loss="mse",   reg=0.0, d0="softplus", free="ab"),
    "a_only":  dict(loss="mse",   reg=0.0, d0="softplus", free="a"),
    "b_only":  dict(loss="mse",   reg=0.0, d0="softplus", free="b"),
    "d0_only": dict(loss="mse",   reg=0.0, d0="softplus", free=""),
    "mse_reg": dict(loss="mse",   reg=REG, d0="softplus", free="ab"),
    "hub_noreg": dict(loss="huber", reg=0.0, d0="softplus", free="ab"),
}


def load_model(case, variant, tag=None):
    ck = torch.load(I.OUT / (case + (TAG[variant] if tag is None else tag)) / "checkpoint.pt",
                    weights_only=False)
    z_dim = len(ck["z_mean"])
    if variant == "base":
        model = I.DissipativeNODE3D(z_dim=z_dim)
    else:
        model = VariantNODE3D(z_dim, **ck.get("settings", VARIANTS[variant]))
    model.load_state_dict(ck["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, np.asarray(ck["z_mean"]), np.asarray(ck["z_std"])


def memory_rate(model, z, delta, u, a, b):
    """The memory equation with the two calibration scales."""
    if getattr(model, "memory", "thermal") == "pool":
        target = F.softplus(model.phi_net(torch.cat([u, z], dim=-1)).squeeze(-1))
        return b * model.lam * (a * target - delta)
    return a * model.alpha * model.p_in(u[..., 0]) - b * model.beta * model.q_wp(delta)


def rollout(model, z0, u, dt, d0, a, b):
    z, delta = z0, d0
    zs, ds = [], []
    for k in range(u.shape[0]):
        zs.append(z)
        ds.append(delta)
        if k == u.shape[0] - 1:
            break
        z_eq, gamma, _ = model._terms(z, delta, u[k])
        z = z_eq + (z - z_eq) * torch.exp(-gamma * dt)
        delta = delta + dt * memory_rate(model, z, delta, u[k], a, b)
    return torch.stack(zs), torch.stack(ds)


def r2(obs, pred, mean, std):
    o, p = (obs - mean) / std, (pred - mean) / std
    ok = np.isfinite(o).all(axis=1) & np.isfinite(p).all(axis=1)
    return float(I.metrics(o[ok], p[ok])["latent_r2_global"])


def adapt(source, target, variant, mode="full", tag=None, save=False, scalars=3):
    cfg = MODES[mode]
    tag = TAG[variant] if tag is None else tag
    model, z_mean_s, z_std_s = load_model(source, variant, tag)
    # The source's input normalisation and pool standardisation are carried over to the
    # target, as in the manuscript: the same u must mean the same power and speed in both.
    src = I.load_record(source, I.STATE_NAMES, window_for(source), False)
    p_src, v_src = src[4].max(), src[5].max()
    tgt = I.load_record(target, I.STATE_NAMES, window_for(target), False)
    z_raw, p_t, v_t = tgt[1], tgt[4], tgt[5]
    speed = torch.tensor((v_t / v_src).astype(np.float32))
    power = torch.tensor((p_t / p_src).astype(np.float32))
    u = torch.stack([power, speed], dim=-1)

    def input_at(c):
        """The recorded input with a gain on travel speed.

        Heat input per unit length is P / v, so an unrecorded change of absorbed power acts in
        the same way as a change of speed. In these simulations both are recorded exactly, so
        c is expected to stay at one; it is fitted here as a control."""
        return torch.stack([power, c * speed], dim=-1)
    zn = torch.tensor(((z_raw - z_mean_s) / z_std_s).astype(np.float32))
    n_pre = max(20, int(PREFIX * len(zn)))
    mean_t, std_t = z_raw.mean(axis=0), np.maximum(z_raw.std(axis=0), 1e-5)

    d_ref = float(np.max(np.abs(np.load(I.OUT / (source + tag) / "fit.npz")["delta_hat"]))) + 1e-6
    one = torch.ones(())
    scale = lambda r, on: torch.exp(SPAN * torch.tanh(r)) if on else one

    def cal(r):
        d0 = d_ref * torch.sigmoid(r[0]) if cfg["d0"] == "ref" else F.softplus(r[0])
        c = torch.exp(SPAN * torch.tanh(r[3])) if scalars == 4 else one
        return d0, scale(r[1], "a" in cfg["free"]), scale(r[2], "b" in cfg["free"]), c

    raw = torch.tensor([-3.0, 0.0, 0.0, 0.0], requires_grad=True)
    opt = torch.optim.AdamW([raw], lr=0.05)
    trace = []
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        d0_t, a_t, b_t, c_t = cal(raw)
        z_hat, _ = rollout(model, zn[0], input_at(c_t)[:n_pre], I.DT, d0_t, a_t, b_t)
        loss = (F.huber_loss(z_hat, zn[:n_pre], delta=0.5) if cfg["loss"] == "huber"
                else F.mse_loss(z_hat, zn[:n_pre]))
        if cfg["reg"]:
            loss = loss + cfg["reg"] * (torch.log(a_t) ** 2 + torch.log(b_t) ** 2
                                        + torch.log(c_t) ** 2 + d0_t ** 2)
        loss.backward()
        opt.step()
        trace.append([float(d0_t), float(a_t), float(b_t), float(c_t), float(loss)])
    with torch.no_grad():
        d0_t, a_t, b_t, c_t = cal(raw)
        d0, a, b, c = float(d0_t), float(a_t), float(b_t), float(c_t)
        z_ad, mem_ad = rollout(model, zn[0], input_at(c_t), I.DT, d0_t, a_t, b_t)
        z_no, mem_no = rollout(model, zn[0], u, I.DT, torch.zeros(()), torch.ones(()), torch.ones(()))
    held = slice(n_pre, len(zn))
    to_raw = lambda zh: zh.numpy() * z_std_s + z_mean_s
    full = np.load(I.OUT / (target + tag) / "fit.npz", allow_pickle=True)["z_pred"]
    t = np.load(I.OUT / (target + tag) / "fit.npz", allow_pickle=True)["t"]
    heat = ring_heat(target, t)
    m = mem_ad.numpy()
    if save:
        np.savez_compressed(I.OUT / f"adapt_{source}_to_{target}_{mode}{tag}_p{int(100 * PREFIX)}.npz",
                            t=t, n_pre=n_pre,
                            P_abs_W=tgt[4], speed_mm_s=tgt[5], z_true=z_raw, z_adapted=to_raw(z_ad),
                            z_source=to_raw(z_no), z_own=full, delta_adapted=m, ring_heat=heat,
                            delta_source=mem_no.numpy(), trace=np.asarray(trace, dtype=np.float32),
                            state_names=np.array(I.STATE_NAMES), calib=np.array([d0, a, b, c]))
    return dict(source=source, target=target, variant=variant, mode=mode, tag=tag,
                scalars=scalars, prefix_s=round(n_pre * I.DT, 1),
                d0=round(d0, 3), drive_scale=round(a, 3), rate_scale=round(b, 3),
                speed_gain=round(c, 3),
                r2_none=round(r2(z_raw[held], to_raw(z_no)[held], mean_t, std_t), 3),
                r2_adapted=round(r2(z_raw[held], to_raw(z_ad)[held], mean_t, std_t), 3),
                r2_full=round(r2(z_raw[held], full[held], mean_t, std_t), 3),
                mem_spearman=round(float(spearmanr(m[held], heat[held])[0]), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["pool", "base"])
    ap.add_argument("--modes", nargs="+", default=["full"], choices=list(MODES))
    ap.add_argument("--tag", default=None, help="model folder tag, e.g. _v_pool_simpleB")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="adaptation_scores.json", help="result file under results/")
    ap.add_argument("--save", action="store_true", help="also store the trajectories for figures")
    ap.add_argument("--scalars", type=int, default=3, choices=[3, 4],
                    help="3 calibrates the memory alone; 4 adds a gain on travel speed")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rows = [adapt(a.source, t, v, m, a.tag, a.save, a.scalars)
            for v in a.variants for m in a.modes for t in a.targets]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
    path = I.OUT / a.out
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update({f"{r['tag'] or 'base'}|{r['mode']}|{r['source']}->{r['target']}": r for r in rows})
    path.write_text(json.dumps(old, indent=1) + "\n")


if __name__ == "__main__":
    main()
