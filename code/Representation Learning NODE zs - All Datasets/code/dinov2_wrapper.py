"""Frozen DINOv2 ViT-B/14 wrapper.

* `build_dinov2(name)` -- load via torch.hub, freeze, eval mode.
* `Preprocess` -- grayscale-in-[0,1] -> 3-channel ImageNet-normalised 224x224.
* `extract_tokens(model, x)` -- returns CLS (B, 768) and patch tokens
  (B, 16, 16, 768) for a batch of preprocessed images.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Preprocess(nn.Module):
    """Grayscale [0,1] (B, 1, H, W) -> (B, 3, 224, 224), ImageNet-normalised."""

    def __init__(self, out_size: int = 224):
        super().__init__()
        self.out_size = out_size
        self.register_buffer("mean", _IMAGENET_MEAN)
        self.register_buffer("std", _IMAGENET_STD)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=(self.out_size, self.out_size),
                              mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std


def build_dinov2(name: str = "dinov2_vitb14"):
    """Load DINOv2 from torch.hub, freeze, eval mode."""
    model = torch.hub.load("facebookresearch/dinov2", name, trust_repo=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def extract_tokens(model, x_preproc):
    """x_preproc: (B, 3, 224, 224) already normalised.
    Returns:
        cls    : (B, 768)
        patches: (B, 16, 16, 768)
    """
    out = model.forward_features(x_preproc)
    cls = out["x_norm_clstoken"]                       # (B, D)
    patches = out["x_norm_patchtokens"]                # (B, N, D)
    B, N, D = patches.shape
    side = int(N ** 0.5)
    assert side * side == N, f"patch grid not square: N={N}"
    patches = patches.view(B, side, side, D)
    return cls, patches
