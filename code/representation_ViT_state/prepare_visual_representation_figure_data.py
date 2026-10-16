#!/usr/bin/env python3
"""Bundle the trained-model tensors used by the visual-representation schematic.

Run with the ``research`` conda environment because only this one-time preparation
step reads PyTorch checkpoints.  The publication figure itself depends only on
NumPy and Matplotlib.
"""

from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "Coding Project" / "representation_ViT_state"
OUT = ROOT / "Paper writing" / "development_notes" / "section6"
OUT.mkdir(parents=True, exist_ok=True)


def frame(cache_name: str, record: str, index: int) -> np.ndarray:
    cache = torch.load(MODEL / "cache" / cache_name, map_location="cpu", weights_only=False)
    image = cache[record][index, 0].numpy().astype(np.float32)
    if image.max() > 1.5:
        image /= 255.0
    return image


def main() -> None:
    trained = np.load(MODEL / "results_generator" / "figure_tensors.npz")
    codes = torch.load(
        MODEL / "results_generator" / "codes.pt", map_location="cpu", weights_only=False
    )["Iden_06"].numpy()
    generated = np.load(MODEL / "results_generator" / "generated_average.npy")
    node = np.load(MODEL / "results_node" / "fit_Iden_06.npz")
    state_records = torch.load(
        MODEL / "results_generator" / "states.pt", map_location="cpu", weights_only=False
    )
    state_arrays = {name: value.numpy().astype(np.float32) for name, value in state_records.items()}
    all_states = np.concatenate(list(state_arrays.values()), axis=0)
    state_mean = all_states.mean(axis=0, keepdims=True)
    state_std = np.maximum(all_states.std(axis=0, keepdims=True), 1e-6)
    standardized = (all_states - state_mean) / state_std
    _, _, vh = np.linalg.svd(standardized, full_matrices=False)
    latent_all = standardized @ vh[:3].T
    latent_track = ((state_arrays["Iden_06"] - state_mean) / state_std) @ vh[:3].T

    # Representative frames are chosen away from record boundaries.  They are
    # examples in the schematic, not observations used for a quantitative claim.
    payload = {
        "gtaw": frame("frames192.pt", "Iden_06", 260),
        "gmaw": frame("frames192_GMAW.pt", "exp_01", 205),
        "waam": frame("frames192_WAAM.pt", "video_1", 165),
        "image": trained["image"],
        "structure": trained["structure"],
        "residual": trained["residual"],
        "rhat": trained["rhat"],
        "recon": trained["recon"],
        "tokens16": trained["tokens16"],
        "tokens8": trained["tokens8"],
        "head_map": trained["head_map"],
        "codes": trained["codes"],
        "z": trained["z"],
        "code_previous": codes[259],
        "code_current": codes[260],
        "generated": generated[[0, len(generated) // 2, len(generated) - 1]],
        "node_u": node["u"][:, 0],
        "node_z": node["z_pred"],
        "node_delta": node["delta"],
        "latent_pca_all": latent_all[::5],
        "latent_pca_track": latent_track,
    }
    destination = OUT / "Figure_9_visual_representation_learning_data.npz"
    np.savez_compressed(destination, **payload)
    print(destination)


if __name__ == "__main__":
    main()
