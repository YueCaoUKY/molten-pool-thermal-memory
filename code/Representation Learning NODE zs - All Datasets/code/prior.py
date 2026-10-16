"""Control-affine Neural ODE.

    dz/dt = f(z) + g(z) · u

f(z): autonomous drift — what happens when u is constant (natural relaxation)
g(z): input coupling — how u drives state changes (controllability)

This is the standard form in nonlinear control theory. More expressive than
linear dz/dt = Az + Bu (f,g are small MLPs), but more structured than full
MLP(z,u) — the separation of drift and input effect has physical meaning
and reduces parameters.

Separate sigma_net predicts σ_prior at KL points.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlAffineNODE(nn.Module):
    def __init__(self, d_s=6, hidden=32, dt=0.1):
        super().__init__()
        self.d_s = d_s
        self.dt = dt

        # f(z): drift dynamics (1 hidden layer)
        self.f_net = nn.Sequential(
            nn.Linear(d_s, hidden), nn.SiLU(), nn.Linear(hidden, d_s))
        # g(z): input gain (1 hidden layer, outputs per-dim gain)
        self.g_net = nn.Sequential(
            nn.Linear(d_s, hidden), nn.SiLU(), nn.Linear(hidden, d_s))

        # Init both near zero → dz/dt ≈ 0 at start
        nn.init.zeros_(self.f_net[-1].weight); nn.init.zeros_(self.f_net[-1].bias)
        nn.init.zeros_(self.g_net[-1].weight); nn.init.zeros_(self.g_net[-1].bias)

        # σ_prior(z, u): learned prior uncertainty
        self.sigma_net = nn.Sequential(
            nn.Linear(d_s + 1, 32), nn.SiLU(), nn.Linear(32, d_s))
        nn.init.zeros_(self.sigma_net[-1].weight)
        nn.init.constant_(self.sigma_net[-1].bias, -2.4)   # σ ≈ 0.3

    def derivative(self, z, u):
        """dz/dt = f(z) + g(z) · u."""
        return self.f_net(z) + self.g_net(z) * u.unsqueeze(-1)

    def prior_logvar(self, z, u):
        inp = torch.cat([z, u.unsqueeze(-1)], dim=-1)
        return self.sigma_net(inp).clamp(min=-9.2, max=2.0)

    def rollout(self, z0, u_window):
        """Euler-integrate from z0. Returns smooth prior mean trajectory."""
        B, W = u_window.shape
        traj = [z0]
        z = z0
        for t in range(W - 1):
            z = z + self.dt * self.derivative(z, u_window[:, t])
            traj.append(z)
        return torch.stack(traj, dim=1)

    def get_prior_logvar(self, z_traj, u_window, indices):
        """Compute prior logvar at specified time indices."""
        return {t: self.prior_logvar(z_traj[:, t], u_window[:, t]) for t in indices}
