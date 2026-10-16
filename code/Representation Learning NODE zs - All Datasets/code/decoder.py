"""Decoder: z_s (d_s,) → logits (1, 128, 128)."""
import torch.nn as nn
import torch.nn.functional as F


class _Up(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(ic, oc, 3, padding=1)
        self.norm = nn.GroupNorm(max(1, oc // 4), oc)

    def forward(self, x):
        return F.silu(self.norm(self.conv(self.up(x))))


class Decoder(nn.Module):
    def __init__(self, d_s=6, init_ch=128, block_ch=(128, 64, 32, 16)):
        super().__init__()
        self.fc = nn.Linear(d_s, init_ch * 8 * 8)
        self.init_ch = init_ch
        blocks = []
        ic = init_ch
        for oc in block_ch:
            blocks.append(_Up(ic, oc))
            ic = oc
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(ic, 1, 3, padding=1)

    def forward(self, z_s):
        h = self.fc(z_s).view(-1, self.init_ch, 8, 8)
        return self.head(self.blocks(h))
