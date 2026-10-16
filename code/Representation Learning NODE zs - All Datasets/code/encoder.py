"""Per-frame VAE encoder: CLS_t → (μ_t, logvar_t)."""
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_dim=768, hidden=256, d_s=6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.mu_head = nn.Linear(hidden, d_s)
        self.logvar_head = nn.Linear(hidden, d_s)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -1.4)

    def forward(self, cls_window):
        h = self.net(cls_window)
        return self.mu_head(h), self.logvar_head(h)
