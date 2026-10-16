#!/usr/bin/env python3
"""A second stage: with the identified model frozen, fit a first-order lag on the input.

The pool responds to the thermal field around it, which follows a change of travel speed only
once the preheat region has been rebuilt. Stage one identifies the model from the input as
commanded; stage two keeps every network fixed and fits one positive rate per input channel,

    d ubar / dt = Lambda (u - ubar),

so the model is driven by the effective input ubar. Only two numbers are fitted, so they are
found by a grid search on the identification interval and no gradient is needed. A large rate
reproduces stage one exactly, so the correction switches itself off where it is not needed.

    python input_lag.py CASE [CASE ...]
"""
import sys
from pathlib import Path

import numpy as np
import torch

import identify_3d as I
from delta_variants import VariantNODE3D, window_for

OUT = Path(__file__).resolve().parent / "results"
TAG = "_v_pool_simpleD_p50"
RATES = np.concatenate([[30.0], np.geomspace(3.0, 0.01, 12)])     # 1/s; 30 is effectively no lag


def lag(u, lam, dt):
    """First-order lag on each input channel, integrated exactly over the step as the pool is."""
    decay = np.exp(-lam * dt)
    ub = np.empty_like(u)
    ub[0] = u[0]
    for k in range(1, len(u)):
        ub[k] = u[k - 1] + (ub[k - 1] - u[k - 1]) * decay
    return ub


def run(case, verbose=True):
    s, z_raw, u_np, *_ = I.load_record(case, I.STATE_NAMES, window_for(case), False)
    ck = torch.load(OUT / f"{case}{TAG}" / "checkpoint.pt", weights_only=False)
    m = VariantNODE3D(len(ck["z_mean"]), **ck["settings"])
    m.load_state_dict(ck["model"])
    m.eval()
    zn_t = torch.tensor(((z_raw - ck["z_mean"]) / ck["z_std"]).astype(np.float32))
    zn = zn_t.numpy()
    n = int(np.load(OUT / f"{case}{TAG}" / "fit.npz", allow_pickle=True)["n_train"])
    sig = np.maximum(z_raw.std(axis=0), 1e-9)

    @torch.no_grad()
    def roll(u_arr, upto=None):
        u = torch.tensor(np.ascontiguousarray(u_arr[:upto] if upto else u_arr))
        return m.rollout(zn_t[0], u, I.DT)[0].numpy()

    def train_loss(u_arr):
        zh = roll(u_arr, n)
        return float(np.mean((zh - zn[:n]) ** 2))

    def held_r2(u_arr):
        p = (roll(u_arr) * ck["z_std"] + ck["z_mean"])[n:] / sig
        t = z_raw[n:] / sig
        return float(1 - np.sum((p - t) ** 2) / np.sum((t - t.mean(axis=0)) ** 2))

    best = (RATES[0], RATES[0], train_loss(u_np))
    for lp in RATES:
        for lv in RATES:
            L = train_loss(lag(u_np, np.array([lp, lv], dtype=np.float32), I.DT))
            if L < best[2]:
                best = (lp, lv, L)
    lp, lv, _ = best
    r_base = held_r2(u_np)
    r_lag = held_r2(lag(u_np, np.array([lp, lv], dtype=np.float32), I.DT))
    if verbose:
        print(f"{case.replace('PP_', ''):>24s} {r_base:7.3f} {r_lag:9.3f} {r_lag - r_base:+7.3f} | "
              f"{lp:7.2f} {lv:7.2f} | {1 / lv:8.1f}", flush=True)
    return r_base, r_lag, lp, lv


if __name__ == "__main__":
    torch.set_num_threads(2)
    print(f"{'记录':>24s} {'原':>7s} {'加滞后':>9s} {'变化':>7s} | "
          f"{'Λ功率':>7s} {'Λ速度':>7s} | {'τ速度 s':>8s}", flush=True)
    for c in sys.argv[1:]:
        run(c)
