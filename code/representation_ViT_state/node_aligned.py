#!/usr/bin/env python3
"""Identification of the image-based two-state model with the Section 5 model settings.

The data layer is the one already used for Section 6 (`node_visual.records`): the six-number
state produced by the frozen representation, passed through a centred 2 s moving average, and
the recorded travel speed. Nothing in the representation is retrained.

The model and the training loop are taken from Section 5 (`delta_variants.VariantNODE3D`, which
builds on `identify_3d.DissipativeNODE3D` and the tanh MLP of the Section 5 `experiment.py`), so
that the structure matches Table 4.3 of the manuscript. Three things are adapted, all of them
dimension bookkeeping rather than structure:

  * the three MLPs are rebuilt for a one-dimensional input (travel speed) instead of the two
    inputs of the finite-volume study (absorbed power and travel speed);
  * the thermal state is always handled through the vector code path, so that m = 1 and m = 2
    behave identically in shape;
  * the heat input per unit length, used only to seed the preliminary fit, is 1/v, because the
    current is not recorded. Where the speed is constant it is a unit step.

    python node_aligned.py fit --process GTAW --record Iden_09 --epochs 1200
    python node_aligned.py sweep --process GTAW            # 250/600/1200/2000 on every record
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
SEC5_MODEL = HERE.parent / "simulation_3D_OpenFOAM_final" / "node_identification"
SEC5_EXP = HERE.parent / "Physics Structured Dynamics - Section 5" / "experiment.py"
SEC4_EXP = (HERE.parent / "Representation Learning NODE zs - All Datasets"
            / "paper_section4" / "experiment.py")


def _load(name: str, path: Path):
    """Two different files are called experiment.py: the Section 5 model building blocks and the
    dataset specifications of the image work. Each importer binds what it needs at import time,
    so they are loaded one after the other under the same name."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("experiment", SEC5_EXP)
sys.path.insert(0, str(SEC5_MODEL))
import identify_3d as I  # noqa: E402  the Section 5 model base
import delta_variants as DV  # noqa: E402  the Section 5 model

_load("experiment", SEC4_EXP)
sys.path.insert(0, str(HERE))
import node_visual as N  # noqa: E402  the data layer of the existing Section 6 work

SMOOTH = {"GTAW": 20, "GMAW": 24, "WAAM": 25}      # 2 s, in frames, as in the existing reports
TAU_0, TAU_M = 4.0, 40.0                            # Table 4.3
SG_WINDOW = 21
PRETRAIN = 1000
LR, WD, CLIP = 3e-3, 1e-5, 5.0


class AlignedNODE(DV.VariantNODE3D):
    """The Section 5 model with a one-dimensional input and a vector thermal state of any size."""

    def __init__(self, z_dim: int, u_dim: int, delta_dim: int = 1, use_delta: bool = True):
        super().__init__(z_dim, gamma_min=1.0 / TAU_0, memory="pool", lam_min=0.01, lam_max=0.30,
                         delta_dim=delta_dim, lam_bands=TAU_M, bounds="min_only")
        self.u_dim, self.use_delta = int(u_dim), bool(use_delta)
        m = self.delta_dim if use_delta else 0
        self.z_eq_net = I.MLP(u_dim + m, z_dim, 16)
        self.gamma_net = I.MLP(z_dim + u_dim + m, z_dim, 16)
        self.phi_net = I.MLP(u_dim + z_dim, self.delta_dim, 16)

    def _terms(self, z, delta, u):
        if not self.use_delta:
            z_eq = self.z_eq_net(u)
            gamma = self.gamma_min + F.softplus(self.gamma_net(torch.cat([z, u], dim=-1)))
            return z_eq, gamma, torch.zeros_like(delta)
        dfeat = self.delta_feature(delta)
        z_eq = self.z_eq_net(torch.cat([u, dfeat], dim=-1))
        gamma = self.gamma_min + F.softplus(self.gamma_net(torch.cat([z, u, dfeat], dim=-1)))
        target = F.softplus(self.phi_net(torch.cat([u, z], dim=-1)))
        return z_eq, gamma, self.lam * (target - delta)

    def rates(self, z, delta, u):
        z_eq, gamma, ddelta = self._terms(z, delta, u)
        return -gamma * (z - z_eq), ddelta

    def rollout(self, z0, u, dt):
        """Exact step of the pool equation with the coefficients held fixed, explicit Euler for
        the thermal state; delta starts at zero, as in Section 5."""
        z = z0
        delta = torch.zeros(self.delta_dim, dtype=z0.dtype)
        zs, ds = [], []
        for k in range(u.shape[0]):
            zs.append(z)
            ds.append(delta)
            if k == u.shape[0] - 1:
                break
            z_eq, gamma, ddelta = self._terms(z, delta, u[k])
            z = z_eq + (z - z_eq) * torch.exp(-gamma * dt)
            delta = delta + dt * ddelta
        return torch.stack(zs), torch.stack(ds)


def heat_per_length(speed: np.ndarray, dt: float, rate: float, n: int) -> np.ndarray:
    """First-order response to the heat input per unit length, used only to seed the
    preliminary fit. Only the travel speed is recorded, so the drive is 1/v with the current
    taken as constant, normalised by its maximum exactly as the Section 5 code normalises P/v.
    Where the speed does not vary the drive is a unit step."""
    v = np.asarray(speed, dtype=np.float64)[:n]
    drive = np.ones(n) if float(np.ptp(v)) < 1e-9 else 1.0 / np.clip(v, 1e-3, None)
    drive = drive / max(drive.max(), 1e-9)
    out, d = np.empty(n), 0.0
    for k in range(n):
        out[k] = d
        d = d + dt * rate * (drive[k] - d)
    return out.astype(np.float32)


def fit(process: str, sid: str, epochs: int, delta_dim: int = 1, use_delta: bool = True,
        seed: int = 42, threads: int = 2, out_root: Path | None = None) -> dict:
    torch.set_num_threads(threads)
    I.seed_everything(seed)
    N.configure(process)
    N.SMOOTH = SMOOTH[process]
    dt = N.DT
    rec = N.records()[sid]
    z_np, u_np = rec["z"], rec["u"]
    n = len(z_np)

    mean, std = z_np.mean(0), np.maximum(z_np.std(0), 1e-5)
    zn = ((z_np - mean) / std).astype(np.float32)
    z, u = torch.tensor(zn), torch.tensor(u_np.astype(np.float32))

    model = AlignedNODE(z.shape[1], u.shape[1], delta_dim=delta_dim, use_delta=use_delta)
    tau_init = (1.0 / model.lam).detach().numpy().tolist()

    # preliminary fit: z_eq and Gamma against the smoothed state derivative, with the thermal
    # state replaced by first-order heat-input responses at the initial relaxation times
    dz = torch.tensor(np.gradient(savgol_filter(zn, SG_WINDOW, 3, axis=0), dt, axis=0)
                      .astype(np.float32))
    seed_d = torch.tensor(np.stack([heat_per_length(rec["speed"], dt, 1.0 / t, n)
                                    for t in tau_init], axis=1))
    if not use_delta:
        seed_d = torch.zeros(n, model.delta_dim)
    pre = torch.optim.AdamW(list(model.z_eq_net.parameters()) + list(model.gamma_net.parameters()),
                            lr=LR, weight_decay=WD)
    for _ in range(PRETRAIN):
        pre.zero_grad(set_to_none=True)
        dz_hat, _ = model.rates(z, seed_d, u)
        loss = F.mse_loss(dz_hat, dz)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        pre.step()

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.05 * LR)
    best, best_state, hist = float("inf"), None, []
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        z_hat, _ = model.rollout(z[0], u, dt)
        loss = F.mse_loss(z_hat, z)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sched.step()
        hist.append(float(loss))
        if hist[-1] < best:
            best = hist[-1]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    wall = time.time() - t0

    with torch.no_grad():
        z_hat, d_hat = model.rollout(z[0], u, dt)
    pred = z_hat.numpy() * std + mean
    tau = (1.0 / model.lam).detach().numpy().tolist()

    variant = "B_no_delta" if not use_delta else f"A_m{delta_dim}" if delta_dim == 1 else f"C_m{delta_dim}"
    root = (out_root or HERE / f"results_node_aligned_{process}")
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{sid}_{variant}_e{epochs}"
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "hist": hist,
                "tau": tau, "tau_init": tau_init, "dt": dt, "seed": seed}, root / f"{stem}.pt")
    np.savez_compressed(root / f"{stem}.npz", z_true=z_np, z_raw=rec["z_raw"][:n], z_pred=pred,
                        delta=d_hat.numpy(), u=u_np, speed=rec["speed"][:n],
                        hist=np.asarray(hist, dtype=np.float32))

    res = dict(process=process, record=sid, variant=variant, epochs=epochs, frames=n,
               r2_smoothed=N.r2(z_np, pred), r2_as_read=N.r2(rec["z_raw"][:n], pred),
               predictable_share=N.ceiling(rec['z_raw'][:n]), tau_s=tau, tau_init_s=tau_init,
               final_loss=hist[-1], best_loss=best, wall_s=round(wall, 1), seed=seed)
    (root / f"{stem}.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fit"])
    ap.add_argument("--process", required=True, choices=["GTAW", "GMAW", "WAAM"])
    ap.add_argument("--record", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--delta-dim", type=int, default=1, dest="delta_dim")
    ap.add_argument("--no-delta", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    r = fit(a.process, a.record, a.epochs, delta_dim=a.delta_dim, use_delta=not a.no_delta,
            threads=a.threads)
    print(f"{r['process']:5s} {r['record']:8s} {r['variant']:10s} e{r['epochs']:<5d} "
          f"R2_smooth {r['r2_smoothed']:.3f}  R2_raw {r['r2_as_read']:.3f}  "
          f"tau {[round(t, 2) for t in r['tau_s']]}  {r['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
