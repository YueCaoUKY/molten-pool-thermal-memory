#!/usr/bin/env python3
"""Fit the Section 5 dissipative NODE to fixed 5-D Section 4 trajectories.

This script intentionally uses the current all-coordinate representation. It never
computes PCA or selects a coordinate for fitting. The source evidence generated here
is a full-sequence, no-reset identification fit; it is not held-out validation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SECTION4_RESULTS = (
    PROJECT
    / "Coding Project"
    / "Representation Learning NODE zs - All Datasets"
    / "paper_section4"
    / "results"
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    dt: float


SPECS = {
    "GTAW": DatasetSpec("GTAW", "Iden_06", 0.1),
    "GMAW": DatasetSpec("GMAW", "exp_01", 0.0827),
    "WAAM": DatasetSpec("WAAM", "video_1", 0.08),
}


def inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BoundedMonotoneLoss(nn.Module):
    """Q(0)=0 with q_min <= Q'(delta) <= q_max for delta >= 0."""

    def __init__(self, units: int = 8, q_min: float = 0.10, q_max: float = 1.0):
        super().__init__()
        self.units = units
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.raw_amp = nn.Parameter(torch.full((units,), -2.0))
        self.raw_slope = nn.Parameter(torch.zeros(units))
        self.bias = nn.Parameter(torch.linspace(-1.5, 1.5, units))

    def positive_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Sum(amp) <= q_max-q_min and slope <= 1, hence Q' <= q_max.
        amp = (self.q_max - self.q_min) * torch.softmax(self.raw_amp, dim=0)
        slope = torch.sigmoid(self.raw_slope)
        return amp, slope

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        amp, slope = self.positive_terms()
        d = delta.unsqueeze(-1)
        nonlinear = amp * (F.softplus(slope * d + self.bias) - F.softplus(self.bias))
        return self.q_min * delta + nonlinear.sum(dim=-1)


class DissipativeNODE(nn.Module):
    def __init__(
        self,
        z_dim: int = 5,
        u_dim: int = 1,
        width: int = 16,
        gamma_min: float = 0.05,
        gamma_max: float = 3.0,
        beta_max: float = 0.04,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.beta_max = float(beta_max)
        self.z_eq_net = MLP(u_dim + 1, z_dim, width)
        self.gamma_net = MLP(z_dim + u_dim + 1, 1, width)
        self.q_wp = BoundedMonotoneLoss(units=8, q_min=0.10, q_max=1.0)
        self.raw_alpha = nn.Parameter(torch.tensor(inv_softplus(0.035)))
        beta_ratio = 0.015 / beta_max
        self.raw_beta = nn.Parameter(torch.tensor(math.log(beta_ratio / (1.0 - beta_ratio))))

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha)

    @property
    def beta(self) -> torch.Tensor:
        return self.beta_max * torch.sigmoid(self.raw_beta)

    def delta_feature(self, delta: torch.Tensor) -> torch.Tensor:
        return delta / (1.0 + delta)

    def rates(
        self, z: torch.Tensor, delta: torch.Tensor, u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z.ndim == 1:
            u = u.reshape(1)
            dfeat = self.delta_feature(delta).reshape(1)
        else:
            u = u.reshape(-1, 1)
            dfeat = self.delta_feature(delta).reshape(-1, 1)
        z_eq = self.z_eq_net(torch.cat([u, dfeat], dim=-1))
        raw_gamma = self.gamma_net(torch.cat([z, u, dfeat], dim=-1)).squeeze(-1)
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(raw_gamma)
        gamma_for_z = gamma if z.ndim == 1 else gamma.unsqueeze(-1)
        dz = -gamma_for_z * (z - z_eq)
        # Available real datasets vary travel speed, not arc power; the normalized
        # workpiece heat drive is therefore the constant P_in=1 special case.
        ddelta = self.alpha - self.beta * self.q_wp(delta)
        return dz, ddelta

    def rollout(self, z0: torch.Tensor, u: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        z = z0
        delta = torch.zeros((), device=z.device, dtype=z.dtype)
        zs = [z]
        deltas = [delta]
        # Exponential Euler treats the bounded relaxation exactly while holding its
        # learned equilibrium and rate fixed over one recorded interval. This is
        # stable for every positive gamma and uses no observation-state resets.
        for k in range(u.shape[0] - 1):
            uk = u[k : k + 1]
            dfeat = self.delta_feature(delta).reshape(1)
            z_eq = self.z_eq_net(torch.cat([uk, dfeat], dim=-1))
            raw_gamma = self.gamma_net(torch.cat([z, uk, dfeat], dim=-1)).squeeze(-1)
            gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(raw_gamma)
            ddelta = self.alpha - self.beta * self.q_wp(delta)
            z = z_eq + (z - z_eq) * torch.exp(-gamma * dt)
            delta = delta + dt * ddelta
            zs.append(z)
            deltas.append(delta)
        return torch.stack(zs), torch.stack(deltas)

    @torch.no_grad()
    def thermal_seed(self, count: int, dt: float, device: torch.device) -> torch.Tensor:
        """Return the current monotone thermal clock for derivative pretraining."""
        delta = torch.zeros((), device=device)
        values = [delta]
        for _ in range(count - 1):
            delta = delta + dt * (self.alpha - self.beta * self.q_wp(delta))
            values.append(delta)
        return torch.stack(values)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_source(spec: DatasetSpec) -> dict[str, np.ndarray]:
    path = SECTION4_RESULTS / spec.name / "node_d5" / "evaluation_samples.npz"
    with np.load(path) as data:
        return {
            "latent": data[f"{spec.source}_latent"].astype(np.float32),
            "u": data[f"{spec.source}_u"].astype(np.float32),
            "indices": data[f"{spec.source}_indices"].astype(np.int64),
            "actual": data[f"{spec.source}_actual"].astype(np.float32),
        }


def metrics(z_true: np.ndarray, z_pred: np.ndarray) -> dict[str, object]:
    residual = z_pred - z_true
    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    sse = np.sum(residual**2, axis=0)
    sst = np.sum((z_true - z_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2_coord = 1.0 - sse / np.maximum(sst, 1e-12)
    r2_global = float(1.0 - np.sum(sse) / np.maximum(np.sum(sst), 1e-12))
    return {
        "latent_nrmse": rmse,
        "latent_nmae": mae,
        "latent_r2_global": r2_global,
        "latent_r2_per_coordinate": [float(x) for x in r2_coord],
    }


def fit_one(
    spec: DatasetSpec,
    epochs: int,
    pretrain_epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    seed_everything(seed)
    raw = load_source(spec)
    z_raw = raw["latent"]
    z_mean = z_raw.mean(axis=0)
    z_std = z_raw.std(axis=0)
    z_std = np.maximum(z_std, 1e-5)
    z_norm = (z_raw - z_mean) / z_std

    z = torch.tensor(z_norm, device=device)
    u = torch.tensor(raw["u"], device=device)
    model = DissipativeNODE().to(device)

    # Fast vector-field initialization. This uses all observed states only to
    # initialize the small networks; the reported fit remains a no-reset rollout.
    # A lightly smoothed derivative avoids asking the model to reproduce encoder
    # frame noise as physical dynamics.
    from scipy.signal import savgol_filter

    window = min(31, z_norm.shape[0] - (1 - z_norm.shape[0] % 2))
    window = max(window, 5)
    if window % 2 == 0:
        window -= 1
    z_smooth = savgol_filter(z_norm, window_length=window, polyorder=3, axis=0)
    dz_target_np = np.gradient(z_smooth, spec.dt, axis=0).astype(np.float32)
    dz_target = torch.tensor(dz_target_np, device=device)
    delta_seed = model.thermal_seed(z.shape[0], spec.dt, device)
    pre_optimizer = torch.optim.AdamW(
        list(model.z_eq_net.parameters()) + list(model.gamma_net.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )
    for epoch in range(pretrain_epochs):
        pre_optimizer.zero_grad(set_to_none=True)
        dz_hat, _ = model.rates(z, delta_seed, u)
        pre_loss = F.huber_loss(dz_hat, dz_target, delta=0.5)
        pre_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        pre_optimizer.step()
        if epoch % 250 == 0 or epoch == pretrain_epochs - 1:
            print(
                f"{spec.name} pretrain={epoch:4d} derivative_loss="
                f"{float(pre_loss.detach().cpu()):.6f}",
                flush=True,
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    history: list[float] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        z_hat, delta = model.rollout(z[0], u, spec.dt)
        loss = F.huber_loss(z_hat, z, delta=0.5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        value = float(loss.detach().cpu())
        history.append(value)
        if value < best_loss:
            best_loss = value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(
                f"{spec.name} seed={seed} epoch={epoch:4d} loss={value:.6f} "
                f"alpha={float(model.alpha.detach()):.4f} beta={float(model.beta.detach()):.4f}",
                flush=True,
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    with torch.no_grad():
        z_hat, delta = model.rollout(z[0], u, spec.dt)
    z_hat_np = z_hat.cpu().numpy()
    delta_np = delta.cpu().numpy()
    result_metrics = metrics(z_norm, z_hat_np)
    result_metrics.update(
        {
            "dataset": spec.name,
            "source_experiment": spec.source,
            "frames": int(z.shape[0]),
            "duration_s": float((z.shape[0] - 1) * spec.dt),
            "dt_s": spec.dt,
            "seed": seed,
            "epochs": epochs,
            "derivative_pretraining_epochs": pretrain_epochs,
            "integrator": "fixed-step exponential Euler; no state resets",
            "best_huber_loss": best_loss,
            "alpha": float(model.alpha.detach().cpu()),
            "beta": float(model.beta.detach().cpu()),
            "gamma_min": model.gamma_min,
            "gamma_max": model.gamma_max,
            "beta_max": model.beta_max,
            "uses_all_five_coordinates": True,
            "uses_pca": False,
            "split": "source identification fit; no held-out frames",
        }
    )

    out = HERE / "results" / spec.name / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "spec": asdict(spec),
            "z_mean": z_mean,
            "z_std": z_std,
        },
        out / "checkpoint.pt",
    )
    np.savez_compressed(
        out / "source_fit.npz",
        time=np.arange(z.shape[0], dtype=np.float32) * spec.dt,
        u=raw["u"],
        z_observed=z_norm,
        z_predicted=z_hat_np,
        delta=delta_np,
        sample_indices=raw["indices"],
        sample_images=raw["actual"],
        loss=np.asarray(history, dtype=np.float32),
        z_mean=z_mean,
        z_std=z_std,
    )
    (out / "metrics.json").write_text(json.dumps(result_metrics, indent=2) + "\n")
    print(json.dumps(result_metrics, indent=2), flush=True)
    return result_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["GTAW", "GMAW", "WAAM", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--pretrain-epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    names = list(SPECS) if args.dataset == "all" else [args.dataset]
    device = torch.device(args.device)
    print(f"device={device}", flush=True)
    for name in names:
        fit_one(SPECS[name], args.epochs, args.pretrain_epochs, args.lr, args.seed, device)


if __name__ == "__main__":
    main()
