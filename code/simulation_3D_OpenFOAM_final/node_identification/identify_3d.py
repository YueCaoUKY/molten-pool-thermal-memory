#!/usr/bin/env python3
"""Identify the Section 5 physics-structured NODE (manuscript eq. 6-7b,
Algorithm 2) on top-view pool features from the 3D heat-conduction simulations.

States z: seven top-view pool features from extract_features.py (area,
length, width, tail length, perimeter, width at the arc, centroid lag), or a
subset (--states).
Inputs u: absorbed arc power r (the thermal drive; it enters P_in(r) and the
pool networks) and travel speed (pool networks only), each divided by its
maximum in the record.

Each case is identified on its own, as the "source experiment" of Algorithm 2:
- z standardised with the case statistics, delta_0 = 0;
- vector-field initialisation on Savitzky-Golay derivatives (31 frames,
  order 3; 1000 AdamW steps on the equilibrium and rate networks);
- 250 epochs of full no-reset rollout (exponential Euler for z, explicit Euler
  for delta), with a Huber loss (kappa 0.5) on z only;
- AdamW 3e-3 with cosine decay, gradient clip 5, weight decay 1e-5, seed 42;
  the best-loss epoch is kept.

The model classes MLP and BoundedMonotoneLoss are imported unchanged from
"Physics Structured Dynamics - Section 5/experiment.py", with the same bounds
and starting values (gamma in [0.05, 3] 1/s, beta <= 0.04 1/s, beta starting
at 0.015 1/s). The only addition is the monotone heat-input map
P_in(r) = softplus(P~(r)) of eq. (7b), because the simulations vary the arc
power (experiment.py fixes P_in = 1).

Repeated seams (--seams): the record contains several seams separated by
arc-off gaps.
- Only frames with the arc on and a pool are fitted, skipping the first 1 s
  after each ignition while the pool forms.
- The pool state z restarts from the observed pool at the first fitted frame
  of each seam.
- delta is integrated through the whole record, gaps included. No heat enters
  while the arc is off: P_in(r) - P_in(0) is used, so P_in(0) = 0. The
  no-delta ablation restarts z in the same way.

Without --seams the record must have the arc on and a pool throughout, and the
behaviour is the original single-seam one.

The true thermal state is withheld from the fit. It is the plate enthalpy
rise delta_H, expressed as a mean temperature rise (K) relative to the start
of the record. After the fit it is compared with the identified delta, scaled
by one least-squares factor (delta has no unit in the model).

Diagnostic (--beta-init, --tag): start beta from another value and write the
results to results/<case><tag>/.
Ablation (--no-delta --tag _nodelta): the same dissipative NODE without the
workpiece state, dz/dt = -gamma(z,u) [z - z_eq(u)].

    python identify_3d.py [case ...] [--window T0 T1] [--seams] [--states a,b] [--no-delta] [--tag T]
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
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
SECTION5 = BENCH.parent / "Physics Structured Dynamics - Section 5"
sys.path.insert(0, str(SECTION5))
from experiment import MLP, BoundedMonotoneLoss, inv_softplus, metrics, seed_everything  # noqa: E402

RESULTS = BENCH / "results"
FEATURES = HERE / "features"
OUT = HERE / "results"
STATE_NAMES = ["area_mm2", "length_mm", "width_mm", "tail_length_mm", "perimeter_mm", "width_at_arc_mm",
               "centroid_lag_mm"]
CASES = ["NI_A_free_const", "NI_B_cooled_const", "NI_C_cooled_highlow", "NI_D_cooled_steps", "NI_E_cooled_ramps"]
WINDOW = (1.0, 60.0)          # s: pool established -> end of the weld
DT = 0.1
BETA_INIT = 0.015             # experiment.py starting value
SEAM_SKIP = 1.0               # s after each ignition before the pool is fitted (pool forming)
SG_WINDOW = 31


class MonotonePower(nn.Module):
    """P_in(r) = softplus(P~(r)) with P~ nondecreasing in r (nonnegative weights).
    With zero_off, P_in(r) - P_in(0) is used instead, so that no heat enters while
    the arc is off; the map stays nonnegative and nondecreasing."""

    def __init__(self, units: int = 8, zero_off: bool = False):
        super().__init__()
        self.zero_off = zero_off
        self.raw_w = nn.Parameter(torch.zeros(units))
        self.b = nn.Parameter(torch.linspace(-2.0, 2.0, units))
        self.raw_a = nn.Parameter(torch.full((units,), -1.0))
        self.c = nn.Parameter(torch.tensor(-3.0))

    def _map(self, r: torch.Tensor) -> torch.Tensor:
        hidden = F.softplus(r.unsqueeze(-1) * F.softplus(self.raw_w) + self.b)
        return F.softplus((F.softplus(self.raw_a) * hidden).sum(dim=-1) + self.c)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        p = self._map(r)
        return p - self._map(torch.zeros_like(r)) if self.zero_off else p


class DissipativeNODE3D(nn.Module):
    """Eq. (6): dz/dt = -gamma(z,u,delta) [z - z_eq(u,delta)],
    d delta/dt = alpha P_in(r) - beta Q_wp(delta); bounds as in experiment.py."""

    def __init__(self, z_dim: int, u_dim: int = 2, width: int = 16,
                 gamma_min: float = 0.05, gamma_max: float = 3.0, beta_max: float = 0.04,
                 beta_init: float = BETA_INIT, use_delta: bool = True, zero_off: bool = False):
        super().__init__()
        self.gamma_min, self.gamma_max, self.beta_max = float(gamma_min), float(gamma_max), float(beta_max)
        self.use_delta = use_delta
        d = 1 if use_delta else 0          # the no-delta ablation drops delta from both networks
        self.z_eq_net = MLP(u_dim + d, z_dim, width)
        self.gamma_net = MLP(z_dim + u_dim + d, z_dim, width)   # one rate per coordinate
        self.q_wp = BoundedMonotoneLoss(units=8, q_min=0.10, q_max=1.0)
        self.p_in = MonotonePower(units=8, zero_off=zero_off)
        self.raw_alpha = nn.Parameter(torch.tensor(inv_softplus(0.035)))
        beta_ratio = beta_init / beta_max
        self.raw_beta = nn.Parameter(torch.tensor(float(np.log(beta_ratio / (1.0 - beta_ratio)))))

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha)

    @property
    def beta(self) -> torch.Tensor:
        return self.beta_max * torch.sigmoid(self.raw_beta)

    @staticmethod
    def delta_feature(delta: torch.Tensor) -> torch.Tensor:
        return delta / (1.0 + delta)

    def _terms(self, z, delta, u):
        if self.use_delta:
            dfeat = self.delta_feature(delta).unsqueeze(-1)
            z_eq = self.z_eq_net(torch.cat([u, dfeat], dim=-1))
            raw = self.gamma_net(torch.cat([z, u, dfeat], dim=-1))
            ddelta = self.alpha * self.p_in(u[..., 0]) - self.beta * self.q_wp(delta)
        else:
            z_eq = self.z_eq_net(u)
            raw = self.gamma_net(torch.cat([z, u], dim=-1))
            ddelta = torch.zeros_like(delta)
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(raw)
        return z_eq, gamma, ddelta

    def n_used_params(self) -> int:
        mods = [self.z_eq_net, self.gamma_net] + ([self.q_wp, self.p_in] if self.use_delta else [])
        return sum(p.numel() for m in mods for p in m.parameters()) + (2 if self.use_delta else 0)

    def rates(self, z, delta, u):
        z_eq, gamma, ddelta = self._terms(z, delta, u)
        return -gamma * (z - z_eq), ddelta

    def rollout(self, z0: torch.Tensor, u: torch.Tensor, dt: float, resets=None, z_obs=None):
        """Free rollout from z0 with delta_0 = 0. At the indices in `resets` the pool
        state restarts from z_obs (the first fitted frame of each seam); delta never
        restarts."""
        reset = {int(i) for i in resets} if resets is not None else set()
        z, delta = z0, torch.zeros((), dtype=z0.dtype)
        zs, deltas = [], []
        n = u.shape[0]
        for k in range(n):
            if k in reset and z_obs is not None:
                z = z_obs[k]
            zs.append(z)
            deltas.append(delta)
            if k == n - 1:
                break
            z_eq, gamma, ddelta = self._terms(z, delta, u[k])
            z = z_eq + (z - z_eq) * torch.exp(-gamma * dt)
            delta = delta + dt * ddelta
        return torch.stack(zs), torch.stack(deltas)

    @torch.no_grad()
    def thermal_seed(self, u: torch.Tensor, dt: float) -> torch.Tensor:
        delta, values = torch.zeros(()), [torch.zeros(())]
        for k in range(u.shape[0] - 1):
            delta = delta + dt * (self.alpha * self.p_in(u[k, 0]) - self.beta * self.q_wp(delta))
            values.append(delta)
        return torch.stack(values)


def _merged(case, window):
    t0, t1 = window
    s = pd.read_csv(RESULTS / case / "signals_0p1s.csv")
    f = pd.read_csv(FEATURES / f"{case}.csv")
    s["key"] = np.round(s["t_s"] * 10).astype(int)
    f["key"] = np.round(f["t_s"] * 10).astype(int)
    s = s.merge(f.drop(columns=["t_s"]), on="key")
    return s[(s["t_s"] >= t0 - 1e-9) & (s["t_s"] <= t1 + 1e-9)].reset_index(drop=True)


def load_record(case: str, states=STATE_NAMES, window=WINDOW, seams=False):
    """States, inputs, the mask of fitted frames and the seam restart indices."""
    states = list(states)
    s = _merged(case, window)
    cur, p, v = (s[c].to_numpy().copy() for c in ("I_A", "P_abs_nominal_W", "u_mm_s"))
    # The last sample is the end of the weld, where the right-continuous schedule
    # already reads zero; the rollout never uses the input of the last sample.
    for a in (cur, p, v):
        a[-1] = a[-2]
    z = s[states].to_numpy(np.float32)
    if not seams:
        if np.isnan(z).any():
            raise ValueError(f"{case}: no pool at some time in {window}")
        if (p <= 0).any():
            raise ValueError(f"{case}: arc off inside {window}")
        fit, resets = np.ones(len(s), bool), np.array([0])
    else:
        full = pd.read_csv(RESULTS / case / "signals_0p1s.csv")
        on_full = full["I_A"].to_numpy() > 0
        ign = full["t_s"].to_numpy()[np.r_[on_full[0], on_full[1:] & ~on_full[:-1]]]
        t = s["t_s"].to_numpy()
        skip = np.zeros(len(s), bool)
        for t_ig in ign:
            skip |= (t >= t_ig - 1e-9) & (t < t_ig + SEAM_SKIP - 1e-9)
        fit = (p > 0) & ~skip & ~np.isnan(z).any(axis=1)
        starts = np.flatnonzero(fit & ~np.r_[False, fit[:-1]])
        resets = starts
        if not fit.any():
            raise ValueError(f"{case}: no fitted frames")
    u = np.column_stack([p / p.max(), v / v.max()]).astype(np.float32)
    return s, z, u, cur, p, v, fit, resets


def load_case(case: str, states=STATE_NAMES, window=WINDOW):
    """Single-seam record (arc on and a pool throughout); kept for predict_crosscase.py."""
    s, z, u, cur, p, v, _, _ = load_record(case, states, window, seams=False)
    return s, z, u, cur, p, v


def _blocks(mask):
    """(start, end) of the contiguous True runs of a boolean array."""
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(idx[::2], idx[1::2]))


def fit_case(case: str, epochs: int, pretrain: int, lr: float, seed: int, beta_init: float = BETA_INIT,
             tag: str = "", use_delta: bool = True, states=STATE_NAMES, window=WINDOW, seams=False,
             train_frac: float = 1.0) -> dict:
    t_start = time.time()
    seed_everything(seed)
    states = list(states)
    window = tuple(window)
    s, z_raw, u_np, cur, p, v, fit, resets = load_record(case, states, window, seams)
    # Within-record held-out protocol: the first train_frac of the record is identified on and the
    # remainder is scored without any state correction at the boundary.
    t_rec = s["t_s"].to_numpy()
    t_split = float(t_rec[0] + train_frac * (t_rec[-1] - t_rec[0]))
    train = fit & (t_rec <= t_split + 1e-9) if train_frac < 1.0 else fit.copy()
    held = fit & ~train
    z_mean, z_std = np.nanmean(z_raw[train], axis=0), np.maximum(np.nanstd(z_raw[train], axis=0), 1e-5)
    zn = (z_raw - z_mean) / z_std
    zn0 = np.where(np.isnan(zn), 0.0, zn).astype(np.float32)          # gap frames are never fitted
    z, u = torch.tensor(zn0), torch.tensor(u_np)
    fit_t, train_t = torch.tensor(fit), torch.tensor(train)
    model = DissipativeNODE3D(z_dim=z.shape[1], beta_init=beta_init, use_delta=use_delta, zero_off=seams)

    dz_np, pre_mask = np.zeros_like(zn0), np.zeros(len(fit), bool)
    for a, b in _blocks(train):
        if b - a >= SG_WINDOW:
            dz_np[a:b] = np.gradient(savgol_filter(zn0[a:b], window_length=SG_WINDOW, polyorder=3, axis=0), DT, axis=0)
            pre_mask[a:b] = True
    dz_target, pre_t = torch.tensor(dz_np.astype(np.float32)), torch.tensor(pre_mask)
    delta_seed = model.thermal_seed(u, DT)
    pre_opt = torch.optim.AdamW(list(model.z_eq_net.parameters()) + list(model.gamma_net.parameters()),
                                lr=lr, weight_decay=1e-5)
    for _ in range(pretrain):
        pre_opt.zero_grad(set_to_none=True)
        dz_hat, _ = model.rates(z, delta_seed, u)
        loss = F.huber_loss(dz_hat[pre_t], dz_target[pre_t], delta=0.5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        pre_opt.step()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.05 * lr)
    history, best_loss, best_state = [], float("inf"), None
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        z_hat, _ = model.rollout(z[resets[0]] if not seams else z[0], u, DT, resets, z)
        loss = F.huber_loss(z_hat[train_t], z[train_t], delta=0.5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        history.append(float(loss.detach()))
        if history[-1] < best_loss:
            best_loss = history[-1]
            best_state = {k: val.detach().clone() for k, val in model.state_dict().items()}
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"{case}{tag} epoch {epoch:3d} loss {history[-1]:.5f} alpha {float(model.alpha):.4f} "
                  f"beta {float(model.beta):.4f}", flush=True)
    model.load_state_dict(best_state)
    with torch.no_grad():
        z_hat, d_hat = model.rollout(z[resets[0]] if not seams else z[0], u, DT, resets, z)
    z_hat, d_hat = z_hat.numpy(), d_hat.numpy()

    dh = s["delta_H_K"].to_numpy()
    dh = dh - dh[0]
    t = s["t_s"].to_numpy()
    m = metrics(zn0[fit], z_hat[fit])
    m_tr = metrics(zn0[train], z_hat[train])
    split = dict(train_frac=float(train_frac), t_split_s=t_split, n_train=int(train.sum()),
                 n_heldout=int(held.sum()), r2_global_train=m_tr["latent_r2_global"],
                 r2_per_state_train=m_tr["latent_r2_per_coordinate"])
    if held.any():
        m_ho = metrics(zn0[held], z_hat[held])
        split.update(r2_global_heldout=m_ho["latent_r2_global"],
                     r2_per_state_heldout=m_ho["latent_r2_per_coordinate"],
                     nrmse_std_heldout=m_ho["latent_nrmse"])
    if use_delta:
        scale = float(dh @ d_hat / (d_hat @ d_hat))
        dstats = dict(delta_scale_K=scale, delta_pearson=float(pearsonr(d_hat, dh)[0]),
                      delta_spearman=float(spearmanr(d_hat, dh)[0]),
                      delta_rel_rmse=float(np.sqrt(np.mean((scale * d_hat - dh) ** 2)) / np.max(np.abs(dh))),
                      alpha=float(model.alpha), beta=float(model.beta), beta_init=beta_init)
        if held.any():
            dstats["delta_spearman_heldout"] = float(spearmanr(d_hat[held], dh[held])[0])
    else:
        scale, dstats = 0.0, {}
    res = dict(
        case=case, tag=tag, use_delta=use_delta, seams=seams, window_s=list(window), frames=int(len(t)),
        fitted_frames=int(fit.sum()), n_seams=int(len(resets)), state_names=states,
        r2_per_state=m["latent_r2_per_coordinate"], r2_global=m["latent_r2_global"], nrmse_std=m["latent_nrmse"],
        **split, **dstats, time_ramp_pearson=float(pearsonr(t, dh)[0]), n_params=model.n_used_params(), best_loss=best_loss,
        epochs=epochs, pretrain_steps=pretrain, seed=seed, wall_s=round(time.time() - t_start, 1))
    out = OUT / (case + tag)
    out.mkdir(parents=True, exist_ok=True)
    z_pred = z_hat * z_std + z_mean
    z_pred[~fit] = np.nan                   # the model state is only meaningful on fitted frames
    np.savez_compressed(out / "fit.npz", t=t, I_A=cur, P_abs_W=p, speed_mm_s=v, z_true=z_raw, z_pred=z_pred,
                        fit=fit, train=train, t_split=t_split, resets=resets, delta_hat=d_hat, delta_true=dh, delta_scaled=scale * d_hat,
                        loss=np.asarray(history), state_names=np.array(states))
    torch.save({"model": best_state, "z_mean": z_mean, "z_std": z_std}, out / "checkpoint.pt")
    (out / "metrics.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=CASES)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--pretrain", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--beta-init", type=float, default=BETA_INIT, help="starting value of beta (diagnostic)")
    ap.add_argument("--tag", default="", help="suffix of the output folder (diagnostic runs)")
    ap.add_argument("--no-delta", action="store_true",
                    help="ablation: the same dissipative NODE without the workpiece state delta")
    ap.add_argument("--states", default=",".join(STATE_NAMES),
                    help="comma-separated subset of the features used as states (state-selection test)")
    ap.add_argument("--window", type=float, nargs=2, default=list(WINDOW),
                    help="record in s (default 1 60)")
    ap.add_argument("--train-frac", type=float, default=1.0, dest="train_frac",
                    help="identify on the first fraction of the record and score the rest as held out")
    ap.add_argument("--seams", action="store_true",
                    help="repeated seams: fit only arc-on frames with a pool, restart z at each seam, no heat in gaps")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    states = [x.strip() for x in args.states.split(",") if x.strip()]
    unknown = set(states) - set(STATE_NAMES)
    if unknown:
        raise SystemExit(f"unknown states: {sorted(unknown)}")
    for case in args.cases:
        fit_case(case, args.epochs, args.pretrain, args.lr, args.seed, beta_init=args.beta_init, tag=args.tag,
                 use_delta=not args.no_delta, states=states, window=args.window, seams=args.seams,
                 train_frac=args.train_frac)


if __name__ == "__main__":
    main()
