#!/usr/bin/env python3
"""Alternative memory states for the pool NODE (owner's request, 2026-09-15).

On every simulation the proposed δ (eq. 6-7b) became a slowly rising ramp. Its
loss slope may fall to q_min = 0.1 and its rate is capped at β ≤ 0.04 1/s
(τ ≥ 25 s), while the heat around the pool rises and falls within 10-25 s. The
variants keep the pool equation and the identification algorithm (free rollout
driven by the input only, δ0 = 0, Huber loss, SG pre-training, AdamW) and change
only the memory state:

  base  the proposed δ: the identify_3d.py results, no refit
  fast  the same δ equation, but δ may be as fast as 1/0.30 s; the pool relaxation
        is kept faster (γ ≥ 0.35 1/s), so the time-scale separation still holds
  lin   as fast, with a linear loss Q(δ) = δ and β ≥ 0.01: δ relaxes toward
        α P_in / β with a time constant of 3.3-100 s and cannot become a pure
        integrator
  pool  δ as an extra state that relaxes toward a target set by the input and the
        visible pool: dδ/dt = λ [φ(u, z) - δ], φ ≥ 0 a small network,
        λ in 0.01-0.30 1/s; γ ≥ 0.35 1/s as in fast

Each result is scored against the heat around the pool (dHK_ring_5_15mm_K, the
reference fixed before the slow-bead runs): rank and linear correlation, and how
often δ moves the same way as that heat over 5 s steps, overall and during
drops.

    python delta_variants.py run   --cases SL_1_step SL_3_slow_cycles NB_1_const --variants fast lin pool
    python delta_variants.py score --cases ... --variants base fast lin pool
    python delta_variants.py fit CASE VARIANT            (one fit; used by run)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as F

import identify_3d as I

HERE = Path(__file__).resolve().parent
WINDOWS = {"SL_1_step": (1.0, 100.0), "SL_2_blocks": (1.0, 120.0), "SL_3_slow_cycles": (4.3, 240.0),
           **{c: (1.0, 90.0) for c in ("NB_1_const", "NB_2_random", "NB_3_random", "NB_4_random", "NB_5_random")}}
VARIANTS = {
    "fast": dict(gamma_min=0.35, beta_max=0.30, beta_init=0.05),
    "lin": dict(gamma_min=0.35, beta_max=0.30, beta_min=0.01, q_min=1.0, q_max=1.0, beta_init=0.05),
    "pool": dict(gamma_min=0.35, memory="pool", lam_min=0.01, lam_max=0.30, lam_init=0.05),
    "poolv": dict(gamma_min=0.35, memory="pool", lam_min=0.01, lam_max=0.30, lam_init=0.05,
                  scale_by_speed=True),
    "poolm": dict(gamma_min=0.35, memory="pool", lam_min=0.01, lam_max=0.30, lam_init=0.05,
                  matrix_gamma=True),
    "poolc": dict(gamma_min=0.35, memory="pool", lam_min=0.01, lam_max=0.30, lam_init=0.05,
                  matrix_gamma=True, couple=(1, 3, 6)),
    "pool2": dict(gamma_min=0.35, memory="pool", lam_min=0.01, lam_max=0.30, lam_init=0.05,
                  delta_dim=2),
    "pool3": dict(gamma_min=0.35, memory="pool", lam_min=0.01, lam_max=0.30,
                  lam_init=(0.25, 0.083, 0.025), delta_dim=3),          # tau = 4, 12, 40 s
    "pool4": dict(gamma_min=0.35, memory="pool", lam_min=0.01, lam_max=0.30,
                  lam_init=(0.25, 0.111, 0.05, 0.022), delta_dim=4),    # tau = 4, 9, 20, 45 s
    # One band of thermal relaxation times per component of delta. The bands tile the interval
    # from the pool relaxation bound to 40 s and do not overlap, so the components stay distinct;
    # each rate is free to move inside its own band during identification.
    **{f"poolb{m}": dict(gamma_min=0.25, memory="pool", lam_min=0.01, lam_max=0.30,
                         delta_dim=m, lam_bands=40.0) for m in (2, 3, 4, 5)},
    "pool2c": dict(gamma_min=0.35, memory="pool", lam_min=0.01, lam_max=0.30, lam_init=0.05,
                   delta_dim=2, matrix_gamma=True, couple=(1, 3, 6)),
}
TAG = {"base": "", **{v: f"_v_{v}" for v in VARIANTS}}
STEP = 50                     # 5 s at 0.1 s, for the trend agreement


def _logit(x):
    return float(np.log(x / (1.0 - x)))


def _inv_softplus(x):
    return float(np.log(np.expm1(x)))


def window_for(case):
    """Fit window: 1 s after the pool first appears, to the end of the record."""
    if case in WINDOWS:
        return WINDOWS[case]
    path = I.OUT / "paper_windows.json"
    cached = json.loads(path.read_text()) if path.exists() else {}
    if case not in cached:
        f = pd.read_csv(I.FEATURES / f"{case}.csv")
        ok = f[f["area_mm2"].notna()]
        cached[case] = [round(float(ok["t_s"].iloc[0]) + 1.0, 2), float(f["t_s"].iloc[-1])]
        path.write_text(json.dumps(cached, indent=1) + "\n")
    return tuple(cached[case])


class VariantNODE3D(I.DissipativeNODE3D):
    """The proposed pool equation with a replaceable memory equation."""

    def __init__(self, z_dim, *, gamma_min=0.05, beta_max=0.04, beta_min=0.0, beta_init=I.BETA_INIT,
                 q_min=0.10, q_max=1.0, memory="thermal", lam_min=0.01, lam_max=0.30, lam_init=0.05,
                 bounds="both", scale_by_speed=False, matrix_gamma=False, couple=None,
                 delta_dim=1, lam_bands=None):
        super().__init__(z_dim=z_dim, gamma_min=gamma_min, beta_max=beta_max,
                         beta_init=min(beta_init, 0.999 * beta_max))
        self.beta_min, self.memory, self.bounds = float(beta_min), memory, bounds
        self.scale_by_speed = bool(scale_by_speed)
        self.matrix_gamma = bool(matrix_gamma)
        if matrix_gamma:
            # Lower-triangular factor of the rate matrix. With `couple` only those coordinates are
            # allowed to interact: a change of travel speed redistributes the pool along the track,
            # so the along-track measures move together while the transverse ones do not.
            if couple is None:
                self.tri = torch.tril_indices(z_dim, z_dim)
            else:
                g = list(couple)
                pairs = [(i, i) for i in range(z_dim)] + [(a, b) for k, a in enumerate(g) for b in g[:k]]
                self.tri = torch.tensor(sorted(pairs)).T
            self.gamma_net = I.MLP(z_dim + 2 + self.delta_dim, self.tri.shape[1], 16)
        self.q_wp = I.BoundedMonotoneLoss(units=8, q_min=q_min, q_max=q_max)
        with torch.no_grad():
            self.raw_beta.fill_(_logit((beta_init - beta_min) / (beta_max - beta_min)))
        self.delta_dim = int(delta_dim)
        if self.delta_dim > 1:
            # The surrounding material is not one lumped region: the part next to the pool responds
            # faster than the part further out. Each component of delta is one such region.
            self.z_eq_net = I.MLP(2 + self.delta_dim, z_dim, 16)
            self.gamma_net = I.MLP(z_dim + 2 + self.delta_dim, z_dim, 16)
        self.lam_bands = lam_bands
        if memory == "pool":
            self.phi_net = I.MLP(2 + z_dim, self.delta_dim, 16)
            self.lam_min, self.lam_max = float(lam_min), float(lam_max)
            if lam_bands is not None:
                lo, hi = 1.0 / gamma_min, float(lam_bands)
                edge = lo * (hi / lo) ** (np.arange(self.delta_dim + 1) / self.delta_dim)
                self.tau_lo = torch.tensor(edge[:-1], dtype=torch.float32)
                self.tau_hi = torch.tensor(edge[1:], dtype=torch.float32)
                self.raw_lam = torch.nn.Parameter(torch.zeros(self.delta_dim))
                self.lam_init = 1.0 / torch.sqrt(self.tau_lo * self.tau_hi)
            else:
                init = ([float(lam_init)] * self.delta_dim if np.isscalar(lam_init)
                        else [float(x) for x in lam_init])
                assert len(init) == self.delta_dim
                self.lam_init = torch.tensor(init)
                raw0 = [(_inv_softplus(x) if bounds == "min_only"
                         else _logit((x - lam_min) / (lam_max - lam_min))) for x in init]
                self.raw_lam = torch.nn.Parameter(torch.tensor(raw0))
        rate_max = (gamma_min if lam_bands is not None else
                    lam_max if memory == "pool" else beta_max * q_max)
        if bounds != "min_only" and rate_max >= gamma_min:
            print(f"warning: memory rate up to {rate_max} 1/s is not slower than the pool (gamma >= {gamma_min})")

    @property
    def beta(self):
        return self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.raw_beta)

    @property
    def lam(self):
        if self.lam_bands is not None:
            tau = self.tau_lo * (self.tau_hi / self.tau_lo) ** torch.sigmoid(self.raw_lam)
            return 1.0 / tau
        if self.bounds == "min_only":
            return F.softplus(self.raw_lam)
        return self.lam_min + (self.lam_max - self.lam_min) * torch.sigmoid(self.raw_lam)

    def gamma_matrix(self, raw, d):
        """Rate matrix gamma_min I + L L^T, positive definite by construction; any batch shape."""
        L = torch.zeros(*raw.shape[:-1], d, d, dtype=raw.dtype)
        L[..., self.tri[0], self.tri[1]] = raw
        eye = torch.eye(d, dtype=raw.dtype)
        return self.gamma_min * eye + L @ L.transpose(-1, -2)

    def _terms(self, z, delta, u):
        scaled = u[..., 1:2] * delta if self.scale_by_speed else delta
        dfeat = self.delta_feature(scaled)
        if self.delta_dim == 1:
            dfeat = dfeat.unsqueeze(-1)
        z_eq = self.z_eq_net(torch.cat([u, dfeat], dim=-1))
        raw = self.gamma_net(torch.cat([z, u, dfeat], dim=-1))
        if self.memory == "pool":
            target = F.softplus(self.phi_net(torch.cat([u, z], dim=-1)).squeeze(-1))
            ddelta = self.lam * (target - delta)
        else:
            ddelta = self.alpha * self.p_in(u[..., 0]) - self.beta * self.q_wp(delta)
        if self.matrix_gamma:
            return z_eq, self.gamma_matrix(raw, z.shape[-1]), ddelta
        gamma = self.gamma_min + (F.softplus(raw) if self.bounds == "min_only"
                                  else (self.gamma_max - self.gamma_min) * torch.sigmoid(raw))
        return z_eq, gamma, ddelta

    def n_used_params(self):
        pool = self.memory == "pool"
        mods = [self.z_eq_net, self.gamma_net] + ([self.phi_net] if pool else [self.q_wp, self.p_in])
        return sum(p.numel() for m in mods for p in m.parameters()) + (1 if pool else 2)

    def rates(self, z, delta, u):
        z_eq, gamma, ddelta = self._terms(z, delta, u)
        if self.matrix_gamma:
            return -(gamma @ (z - z_eq).unsqueeze(-1)).squeeze(-1), ddelta
        return -gamma * (z - z_eq), ddelta

    def rollout(self, z0, u, dt, resets=None, z_obs=None):
        if not self.matrix_gamma and self.delta_dim == 1:
            return super().rollout(z0, u, dt, resets, z_obs)
        z = z0
        delta = torch.zeros(self.delta_dim, dtype=z0.dtype) if self.delta_dim > 1 \
            else torch.zeros((), dtype=z0.dtype)
        zs, deltas = [], []
        for k in range(u.shape[0]):
            zs.append(z)
            deltas.append(delta)
            if k == u.shape[0] - 1:
                break
            z_eq, gamma, ddelta = self._terms(z, delta, u[k])
            if self.matrix_gamma:
                z = z_eq + (torch.matrix_exp(-gamma * dt) @ (z - z_eq).unsqueeze(-1)).squeeze(-1)
            else:
                z = z_eq + (z - z_eq) * torch.exp(-gamma * dt)
            delta = delta + dt * ddelta
        return torch.stack(zs), torch.stack(deltas)

    @torch.no_grad()
    def thermal_seed(self, u, dt):
        if self.memory != "pool":
            return super().thermal_seed(u, dt)
        # The temperature rise around a moving source scales with the heat entering the plate per
        # unit length, not with power, so the substitute for delta is a low-pass of P / v. With a
        # constant travel speed the two agree; when the speed varies, power alone carries nothing.
        drive = u[:, 0] / torch.clamp(u[:, 1], min=1e-3)
        drive = drive / torch.clamp(drive.max(), min=1e-9)
        rate = self.lam_init if self.delta_dim > 1 else self.lam_init.reshape(())
        d = torch.zeros_like(rate)
        out = [d]
        for k in range(u.shape[0] - 1):
            d = d + dt * rate * (drive[k] - d)
            out.append(d)
        return torch.stack(out)


def _rho(d, heat):
    """Rank correlation with the withheld reference; with several thermal components, the best one."""
    d = np.asarray(d)
    if d.ndim == 1:
        return float(spearmanr(d, heat)[0])
    return max(float(spearmanr(d[:, k], heat)[0]) for k in range(d.shape[1]))


def fit(case, variant, seed=42, epochs=250, pretrain=1000, lr=3e-3, suffix="", loss="huber",
        seed_traj="filter", bounds="both", train_frac=1.0):
    """identify_3d.fit_case for one single-seam record, with the variant's memory state.

    loss:      "huber" (kappa 0.5) or "mse"; "mse" removes the loss threshold.
    seed_traj: "filter" (first-order response to the normalised power) or "zero"; "zero" removes
               the seeded thermal trajectory from the derivative pre-training.
    bounds:    "both" (gamma in [gamma_min, gamma_max], lambda in [lam_min, lam_max]) or "min_only"
               (gamma >= gamma_min, lambda > 0; the upper bounds are dropped and checked afterwards).
    """
    t_start = time.time()
    I.seed_everything(seed)
    kw, tag, window = dict(VARIANTS[variant]), TAG[variant] + suffix, window_for(case)
    if bounds != "both":
        kw["bounds"] = bounds
    loss_fn = loss
    crit = (lambda p, q: F.huber_loss(p, q, delta=0.5)) if loss_fn == "huber" else F.mse_loss
    s, z_raw, u_np, cur, p, v, fitm, resets = I.load_record(case, I.STATE_NAMES, window, False)
    # Within-record held-out protocol: identify on the first train_frac of the record, then let the
    # rollout continue into the remainder without any state correction at the boundary.
    t_rec = s["t_s"].to_numpy()
    n_tr = len(z_raw) if train_frac >= 1.0 else int(np.searchsorted(
        t_rec, t_rec[0] + train_frac * (t_rec[-1] - t_rec[0]) + 1e-9))
    t_split = float(t_rec[min(n_tr, len(t_rec)) - 1])
    z_mean, z_std = z_raw[:n_tr].mean(axis=0), np.maximum(z_raw[:n_tr].std(axis=0), 1e-5)
    zn = ((z_raw - z_mean) / z_std).astype(np.float32)
    z, u = torch.tensor(zn), torch.tensor(u_np)
    model = VariantNODE3D(z.shape[1], **kw)

    dz_tr = torch.tensor(np.gradient(savgol_filter(zn[:n_tr], window_length=I.SG_WINDOW, polyorder=3, axis=0),
                                     I.DT, axis=0).astype(np.float32))
    seed_state = model.thermal_seed(u, I.DT) if seed_traj == "filter" else torch.zeros(len(u))
    pre = torch.optim.AdamW(list(model.z_eq_net.parameters()) + list(model.gamma_net.parameters()),
                            lr=lr, weight_decay=1e-5)
    for _ in range(pretrain):
        pre.zero_grad(set_to_none=True)
        dz_hat, _ = model.rates(z[:n_tr], seed_state[:n_tr], u[:n_tr])
        loss = crit(dz_hat, dz_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        pre.step()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.05 * lr)
    hist, best, best_state = [], float("inf"), None
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        z_hat, _ = model.rollout(z[0], u, I.DT)
        loss = crit(z_hat[:n_tr], z[:n_tr])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        hist.append(float(loss.detach()))
        if hist[-1] < best:
            best, best_state = hist[-1], {k: x.detach().clone() for k, x in model.state_dict().items()}
        if epoch % 50 == 0 or epoch == epochs - 1:
            rate = (f"lam {np.round(model.lam.detach().numpy(), 4).tolist()}"
                    if model.memory == "pool" else f"beta {float(model.beta):.4f}")
            print(f"{case}{tag} epoch {epoch:3d} loss {hist[-1]:.5f} {rate}", flush=True)
    model.load_state_dict(best_state)
    with torch.no_grad():
        z_hat, d_hat = model.rollout(z[0], u, I.DT)
        gam = torch.stack([model._terms(z_hat[k], d_hat[k], u[k])[1] for k in range(len(u))])
    z_hat, d_hat = z_hat.numpy(), d_hat.numpy()
    m = I.metrics(zn, z_hat)
    m_tr = I.metrics(zn[:n_tr], z_hat[:n_tr])
    heat = ring_heat(case, s["t_s"].to_numpy())
    split = dict(train_frac=float(train_frac), t_split_s=t_split, n_train=int(n_tr),
                 n_heldout=int(len(zn) - n_tr), r2_global_train=m_tr["latent_r2_global"],
                 r2_per_state_train=m_tr["latent_r2_per_coordinate"],
                 delta_spearman=_rho(d_hat, heat))
    if n_tr < len(zn):
        m_ho = I.metrics(zn[n_tr:], z_hat[n_tr:])
        split.update(r2_global_heldout=m_ho["latent_r2_global"],
                     r2_per_state_heldout=m_ho["latent_r2_per_coordinate"],
                     nrmse_std_heldout=m_ho["latent_nrmse"],
                     delta_spearman_heldout=_rho(d_hat[n_tr:], heat[n_tr:]))
    rates = (dict(lam=float(model.lam.reshape(-1)[0]),
                  lam_all=[round(float(x), 5) for x in model.lam.reshape(-1)])
             if model.memory == "pool" else dict(alpha=float(model.alpha), beta=float(model.beta)))
    res = dict(case=case, variant=variant, tag=tag, window_s=list(window), frames=int(len(zn)), settings=kw,
               state_names=I.STATE_NAMES, r2_per_state=m["latent_r2_per_coordinate"], r2_global=m["latent_r2_global"],
               nrmse_std=m["latent_nrmse"], **split, **rates, n_params=model.n_used_params(), best_loss=best,
               epochs=epochs, pretrain_steps=pretrain, seed=seed, loss_fn=loss_fn, seed_traj=seed_traj,
               bounds=bounds, gamma_span=[float(model.gamma_min), float(gam.min()), float(gam.max())],
               wall_s=round(time.time() - t_start, 1))
    out = I.OUT / (case + tag)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "fit.npz", t=s["t_s"].to_numpy(), I_A=cur, P_abs_W=p, speed_mm_s=v, z_true=z_raw,
                        z_pred=z_hat * z_std + z_mean, fit=fitm, n_train=n_tr, t_split=t_split,
                        resets=resets, delta_hat=d_hat, ring_heat=heat,
                        loss=np.asarray(hist), state_names=np.array(I.STATE_NAMES))
    torch.save({"model": best_state, "z_mean": z_mean, "z_std": z_std, "settings": kw}, out / "checkpoint.pt")
    (out / "metrics.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("case", "variant", "r2_global", "nrmse_std", "best_loss", "wall_s")}),
          flush=True)
    return res


def run(cases, variants, jobs, threads, seed):
    queue = sorted(((c, v) for v in variants for c in cases), key=lambda cv: -np.diff(WINDOWS[cv[0]])[0])
    running = []
    while queue or running:
        while queue and len(running) < jobs:
            c, v = queue.pop(0)
            log = open(HERE / "logs" / f"var_{c}_{v}.log", "w")
            cmd = [sys.executable, str(Path(__file__)), "fit", c, v, "--seed", str(seed), "--threads", str(threads)]
            running.append((c, v, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE), log))
        time.sleep(5)
        for item in list(running):
            if item[2].poll() is not None:
                item[3].close()
                running.remove(item)
                print(f"{item[0]} {item[1]} rc={item[2].returncode}", flush=True)


def ring_heat(case, t):
    s = pd.read_csv(I.RESULTS / case / "signals_0p1s.csv")
    s.index = np.round(s["t_s"], 3)
    return s.reindex(np.round(t, 3))["dHK_ring_5_15mm_K"].to_numpy()


def score_one(case, variant, suffix=""):
    out = I.OUT / (case + TAG[variant] + suffix)
    d = np.load(out / "fit.npz", allow_pickle=True)
    m = json.loads((out / "metrics.json").read_text())
    t = d["t"]
    fitm = d["fit"].astype(bool) if "fit" in d.files else np.ones(len(t), bool)   # older single-seam fits
    x, y = d["delta_hat"][fitm], ring_heat(case, t)[fitm]
    dx, dy = x[STEP:] - x[:-STEP], y[STEP:] - y[:-STEP]
    thr = 0.02 * (np.nanmax(y) - np.nanmin(y))
    moving, drop = np.abs(dy) > thr, dy < -thr
    nd = I.OUT / (case + "_nodelta") / "metrics.json"
    return dict(case=case, variant=variant, r2=m["r2_global"], nrmse=m["nrmse_std"],
                r2_nodelta=json.loads(nd.read_text())["r2_global"] if nd.exists() else None,
                suffix=suffix, spearman=float(spearmanr(x, y)[0]), pearson=float(pearsonr(x, y)[0]),
                trend_agree=float(np.mean(np.sign(dx[moving]) == np.sign(dy[moving]))),
                drop_agree=float(np.mean(dx[drop] < 0)) if drop.any() else float("nan"),
                heat_drop_share=float(drop.mean()), falls=bool((np.diff(x) < -1e-6).any()))


def score(cases, variants, suffix=""):
    rows = []
    for c in cases:
        for v in variants:
            try:
                rows.append(score_one(c, v, suffix))
            except FileNotFoundError:
                print(f"{c} {v}: no result")
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    cols = ["case", "variant", "r2", "r2_nodelta", "spearman", "pearson", "trend_agree", "drop_agree", "heat_drop_share"]
    print(df[cols].round(3).to_string(index=False))
    print("\nmean over cases:")
    print(df.groupby("variant")[["r2", "spearman", "pearson", "trend_agree", "drop_agree"]].mean().round(3).to_string())
    path = I.OUT / "delta_variants_scores.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update({f"{r['case']}|{r['variant']}{r['suffix']}": r for r in rows})
    path.write_text(json.dumps(old, indent=1) + "\n")
    return df


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("case")
    f.add_argument("variant", choices=list(VARIANTS))
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--threads", type=int, default=2)
    f.add_argument("--epochs", type=int, default=250)
    f.add_argument("--pretrain", type=int, default=1000)
    f.add_argument("--suffix", default="", help="appended to the output folder, e.g. _s7 for a second seed")
    f.add_argument("--loss", default="huber", choices=["huber", "mse"])
    f.add_argument("--seed-traj", default="filter", choices=["filter", "zero"], dest="seed_traj")
    f.add_argument("--bounds", default="both", choices=["both", "min_only"])
    f.add_argument("--train-frac", type=float, default=1.0, dest="train_frac",
                   help="identify on the first fraction of the record and score the rest as held out")
    r = sub.add_parser("run")
    r.add_argument("--cases", nargs="+", required=True)
    r.add_argument("--variants", nargs="+", required=True, choices=list(VARIANTS))
    r.add_argument("--jobs", type=int, default=6)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--seed", type=int, default=42)
    sc = sub.add_parser("score")
    sc.add_argument("--cases", nargs="+", required=True)
    sc.add_argument("--variants", nargs="+", required=True, choices=["base", *VARIANTS])
    sc.add_argument("--suffix", default="")
    a = ap.parse_args()
    if a.cmd == "fit":
        torch.set_num_threads(a.threads)
        fit(a.case, a.variant, a.seed, a.epochs, a.pretrain, suffix=a.suffix, loss=a.loss,
            seed_traj=a.seed_traj, bounds=a.bounds, train_frac=a.train_frac)
    elif a.cmd == "run":
        run(a.cases, a.variants, a.jobs, a.threads, a.seed)
    else:
        score(a.cases, a.variants, a.suffix)


if __name__ == "__main__":
    main()
