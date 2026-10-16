"""Full-trajectory z extraction — no sliding window at inference.

The encoder is genuinely per-frame (verified empirically: feeding the
full window vs feeding a single frame produces identical output for
that frame). The sliding-window dataloader is only required during
training, because the loss machinery needs μ at multiple timepoints
within a window (NODE initial condition, KL constraint points,
multi-frame recon).

This module ignores the window altogether: it loads cached CLS tokens
for an entire experiment / video, runs the encoder once over the
T-frame sequence, applies the saved PCA basis, and returns z(t) for
*every* frame t = 0 .. T-1. The previous post-processing only had
z(t ≥ window-1), wasting the first 99 frames.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from encoder import Encoder


@torch.no_grad()
def encode_sequence(encoder: Encoder, cls: torch.Tensor, device: str = "cuda",
                    batch_size: int = 1024) -> np.ndarray:
    """Encode a (T, 768) sequence of CLS tokens into (T, d_s) μ.

    Per-frame; no temporal aggregation.
    """
    encoder.eval()
    T = cls.shape[0]
    out = []
    for i in range(0, T, batch_size):
        chunk = cls[i:i+batch_size].to(device)             # (b, 768)
        mu, _ = encoder(chunk)                              # (b, d_s)
        out.append(mu.cpu().numpy())
    return np.concatenate(out, axis=0)                      # (T, d_s)


def load_encoder_from_deploy(deploy_path: Path, device: str = "cuda") -> Tuple[Encoder, np.ndarray, np.ndarray, int]:
    """Load encoder weights + PCA params from a deploy.pt."""
    blob = torch.load(deploy_path, map_location=device, weights_only=False)
    cfg_dict = blob["cfg"]
    enc = Encoder(
        in_dim=cfg_dict["dinov2_dim"],
        hidden=cfg_dict["encoder_hidden"],
        d_s=cfg_dict["d_s"],
    ).to(device)
    enc.load_state_dict(blob["enc"])
    enc.eval()
    pca_mean = blob["pca_mean"]
    pca_components = blob["pca_components"]                 # (d_eff, d_s)
    d_eff = int(blob["d_eff"])
    return enc, pca_mean, pca_components, d_eff


def extract_z_for_features(
    feature_files: Dict[str, Path],
    deploy_path: Path,
    device: str = "cuda",
    extra_arrays_keys: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """For each (id, features.pt) pair, compute full-length z(t) trajectory.

    Parameters
    ----------
    feature_files : {id_str -> Path}
        Mapping experiment id to its features cache .pt file (must
        contain 'cls' of shape (T, 768)).
    deploy_path : Path
        deploy.pt produced by training (encoder weights + PCA basis).
    extra_arrays_keys : list of str, optional
        Additional fields from the features .pt to pass through into
        the per-experiment output dict (e.g. "speed" so the caller can
        re-build u_window normalisation).

    Returns
    -------
    {id_str -> {"ts": (T,), "mu_full": (T, d_s), "z": (T, d_eff), ...}}
    """
    extra_arrays_keys = extra_arrays_keys or []
    enc, pca_mean, P, d_eff = load_encoder_from_deploy(deploy_path, device)

    results: Dict[str, dict] = {}
    for eid, fpath in feature_files.items():
        blob = torch.load(fpath, map_location="cpu", weights_only=False)
        cls = blob["cls"]                                    # (T, 768)
        T = int(cls.shape[0])
        mu_full = encode_sequence(enc, cls, device=device)   # (T, d_s)
        z = (mu_full - pca_mean) @ P.T                       # (T, d_eff)
        ts = np.arange(T, dtype=np.int64)

        entry = {"ts": ts, "mu_full": mu_full, "z": z, "T": T}
        for k in extra_arrays_keys:
            if k in blob:
                v = blob[k]
                entry[k] = v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        results[eid] = entry
    return results
