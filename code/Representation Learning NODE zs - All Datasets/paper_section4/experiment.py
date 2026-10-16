"""Reproducible experiments for the revised paper Section 4.

The implementation deliberately preserves the original Stage-1 components but
does not apply PCA. The encoder mean itself is the fixed-dimensional state.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
METHOD_ROOT = HERE.parent
PROJECT_ROOT = METHOD_ROOT.parent.parent
DATA_ROOT = PROJECT_ROOT / "Coding Project" / "Dataset clear"
RESULTS_ROOT = HERE / "results"
SMOKE_RESULTS_ROOT = HERE / "smoke_results"
sys.path.insert(0, str(METHOD_ROOT / "code"))

from decoder import Decoder  # noqa: E402
from encoder import Encoder  # noqa: E402
from losses import multi_frame_recon, pcgrad_surgery, temporal_kl_loss  # noqa: E402


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    train_id: str
    heldout_ids: tuple[str, ...]
    window: int
    dt: float
    u_min: float
    u_max: float
    feature_folder: str


SPECS: Dict[str, ProcessSpec] = {
    "GTAW": ProcessSpec(
        "GTAW", "Iden_06",
        ("Iden_01", "Iden_02", "Iden_07", "Iden_08", "Iden_09",
         "Iden_10", "Iden_11", "Iden_12", "Iden_13"),
        100, 0.1, 2.0, 6.0, "GTAW",
    ),
    "GMAW": ProcessSpec(
        "GMAW", "exp_01",
        ("exp_02", "exp_03", "exp_04", "exp_05", "exp_06", "exp_07",
         "exp_09", "exp_10"),
        100, 0.0827, 3.0, 12.0, "GMAW",
    ),
    "WAAM": ProcessSpec(
        "WAAM", "video_1", ("video_3", "video_5"),
        100, 0.08, 0.0, 1.0, "WAAM_ViD",
    ),
}


@dataclass
class TrainConfig:
    dataset: str
    prior: str = "node"
    latent_dim: int = 5
    steps: int = 3000
    batch_size: int = 16
    lr: float = 2e-4
    prior_lr: float = 1e-3
    weight_decay: float = 1e-4
    beta: float = 1.0
    free_nats: float = 0.1
    inner_steps: int = 3
    encoder_hidden: int = 256
    prior_hidden: int = 32
    decoder_init_channels: int = 128
    grad_clip: float = 5.0
    seed: int = 42
    log_every: int = 100
    image_size: int = 128
    rollout_horizon: int = 25
    recon_indices: tuple[int, ...] = (0, 25, 50, 75, 99)
    kl_indices: tuple[int, ...] = (9, 19, 29, 39, 49, 59, 69, 79, 89, 99)


@dataclass
class Sequence:
    sequence_id: str
    cls: torch.Tensor
    images: torch.Tensor
    u: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.cls.shape[0])


def _feature_path(spec: ProcessSpec, sequence_id: str) -> Path:
    return METHOD_ROOT / spec.feature_folder / "results" / "_features" / f"{sequence_id}.pt"


def _load_grayscale(paths: Iterable[Path], image_size: int) -> torch.Tensor:
    frames = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
            frames.append(np.asarray(image, dtype=np.uint8))
    return torch.from_numpy(np.stack(frames)).unsqueeze(1)


def _load_gtaw(sequence_id: str, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    folder = DATA_ROOT / "GTAW_varying_speed" / sequence_id
    info = pd.read_csv(folder / "info.txt", sep="\t")
    indexed = {int(p.stem.split("_")[1]): p for p in folder.glob("images/cropped_*.jpg")}
    info = info.loc[info["index"].isin(indexed)].sort_values("index").reset_index(drop=True)
    paths = [indexed[int(index)] for index in info["index"]]
    return _load_grayscale(paths, image_size), torch.tensor(info["welding_speed"].to_numpy(), dtype=torch.float32)


def _load_gmaw(sequence_id: str, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    folder = DATA_ROOT / "GMAW_varying_speed" / sequence_id
    info = pd.read_csv(folder / "data.csv")
    indexed = {int(p.stem.split("_")[-1]): p for p in folder.glob("images/received_image_*.jpg")}
    info = info.loc[info["Index"].isin(indexed)].sort_values("Index").reset_index(drop=True)
    # This is the stride used to create the authoritative feature caches.
    info = info.iloc[::2].reset_index(drop=True)
    paths = [indexed[int(index)] for index in info["Index"]]
    return _load_grayscale(paths, image_size), torch.tensor(info["Speed"].to_numpy(), dtype=torch.float32)


def _load_waam(sequence_id: str, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    cache = METHOD_ROOT / "WAAM_ViD" / "results" / "_frames" / f"{sequence_id}.npy"
    frames = np.load(cache)
    if frames.shape[1:] != (image_size, image_size):
        resized = []
        for frame in frames:
            image = Image.fromarray(frame).resize((image_size, image_size), Image.Resampling.BILINEAR)
            resized.append(np.asarray(image, dtype=np.uint8))
        frames = np.stack(resized)
    return torch.from_numpy(np.asarray(frames)).unsqueeze(1), torch.zeros(len(frames), dtype=torch.float32)


def load_sequence(spec: ProcessSpec, sequence_id: str, image_size: int) -> Sequence:
    blob = torch.load(_feature_path(spec, sequence_id), map_location="cpu", weights_only=False)
    cls = blob["cls"].detach().to(torch.float32).contiguous()
    del blob
    if spec.name == "GTAW":
        images, u_raw = _load_gtaw(sequence_id, image_size)
    elif spec.name == "GMAW":
        images, u_raw = _load_gmaw(sequence_id, image_size)
    else:
        images, u_raw = _load_waam(sequence_id, image_size)
    if not (len(cls) == len(images) == len(u_raw)):
        raise ValueError(
            f"{spec.name}/{sequence_id}: cls={len(cls)}, images={len(images)}, u={len(u_raw)}"
        )
    if spec.name == "WAAM":
        u = u_raw
    else:
        u = (u_raw - spec.u_min) / (spec.u_max - spec.u_min)
    return Sequence(sequence_id, cls, images, u.to(torch.float32))


class WindowDataset(Dataset):
    def __init__(self, sequence: Sequence, window: int, recon_indices: tuple[int, ...]):
        if sequence.length < window:
            raise ValueError(f"sequence length {sequence.length} < window {window}")
        self.sequence = sequence
        self.window = window
        self.recon_indices = torch.tensor(recon_indices, dtype=torch.long)

    def __len__(self) -> int:
        return self.sequence.length - self.window + 1

    def __getitem__(self, index: int) -> dict:
        start = int(index)
        stop = start + self.window
        reconstruction_positions = start + self.recon_indices
        return {
            "cls": self.sequence.cls[start:stop],
            "u": self.sequence.u[start:stop],
            "images": self.sequence.images[reconstruction_positions].to(torch.float32) / 255.0,
        }


class GaussianPrior(nn.Module):
    def __init__(self, latent_dim: int, hidden: int):
        super().__init__()
        self.sigma_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim)
        )
        nn.init.zeros_(self.sigma_net[-1].weight)
        nn.init.constant_(self.sigma_net[-1].bias, -2.4)

    def get_prior_logvar(self, trajectory: torch.Tensor, u: torch.Tensor,
                         indices: tuple[int, ...]) -> dict[int, torch.Tensor]:
        return {
            index: self.sigma_net(torch.cat([trajectory[:, index], u[:, index, None]], dim=-1)).clamp(-9.2, 2.0)
            for index in indices
        }


class ControlAffinePrior(GaussianPrior):
    def __init__(self, latent_dim: int, hidden: int, dt: float):
        super().__init__(latent_dim, hidden)
        self.dt = dt
        self.f = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim))
        self.g = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim))
        for network in (self.f, self.g):
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)

    def rollout(self, z0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        trajectory = [z0]
        z = z0
        for step in range(u.shape[1] - 1):
            dz = self.f(z) + self.g(z) * u[:, step, None]
            z = z + self.dt * dz
            trajectory.append(z)
        return torch.stack(trajectory, dim=1)


class RecurrentPrior(GaussianPrior):
    def __init__(self, latent_dim: int, hidden: int, cell_type: str):
        super().__init__(latent_dim, hidden)
        self.cell_type = cell_type
        self.init_h = nn.Linear(latent_dim, hidden)
        self.init_c = nn.Linear(latent_dim, hidden) if cell_type == "lstm" else None
        cell = nn.LSTMCell if cell_type == "lstm" else nn.RNNCell
        self.cell = cell(latent_dim + 1, hidden)
        self.delta = nn.Linear(hidden, latent_dim)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def rollout(self, z0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        z = z0
        h = torch.tanh(self.init_h(z0))
        c = torch.tanh(self.init_c(z0)) if self.init_c is not None else None
        trajectory = [z]
        for step in range(u.shape[1] - 1):
            cell_input = torch.cat([z, u[:, step, None]], dim=-1)
            if self.cell_type == "lstm":
                h, c = self.cell(cell_input, (h, c))
            else:
                h = self.cell(cell_input, h)
            z = z + self.delta(h)
            trajectory.append(z)
        return torch.stack(trajectory, dim=1)


class TransformerPrior(GaussianPrior):
    def __init__(self, latent_dim: int, hidden: int, max_window: int):
        super().__init__(latent_dim, hidden)
        self.input_projection = nn.Linear(latent_dim + 1, hidden)
        self.position = nn.Parameter(torch.zeros(1, max_window, hidden))
        layer = nn.TransformerEncoderLayer(
            hidden, nhead=4, dim_feedforward=4 * hidden, dropout=0.0,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Linear(hidden, latent_dim)
        nn.init.normal_(self.position, std=0.02)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def rollout(self, z0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        batch, window = u.shape
        base = z0[:, None].expand(batch, window, -1)
        tokens = self.input_projection(torch.cat([base, u[:, :, None]], dim=-1))
        tokens = tokens + self.position[:, :window]
        mask = torch.triu(torch.ones(window, window, device=u.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(tokens, mask=mask)
        predicted = z0[:, None] + self.output(hidden)
        # Keep the observed initial condition without modifying an autograd
        # tensor in place.
        return torch.cat([z0[:, None], predicted[:, 1:]], dim=1)


def build_prior(kind: str, latent_dim: int, hidden: int, dt: float, window: int) -> Optional[nn.Module]:
    if kind == "none":
        return None
    if kind == "node":
        return ControlAffinePrior(latent_dim, hidden, dt)
    if kind in {"rnn", "lstm"}:
        return RecurrentPrior(latent_dim, hidden, kind)
    if kind == "transformer":
        return TransformerPrior(latent_dim, hidden, window)
    raise ValueError(kind)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _learning_rate(step: int, config: TrainConfig) -> float:
    warmup = min(300, max(config.steps // 10, 1))
    if step <= warmup:
        return config.lr * step / warmup
    progress = (step - warmup) / max(config.steps - warmup, 1)
    return 1e-5 + (config.lr - 1e-5) * 0.5 * (1 + math.cos(math.pi * progress))


def _run_dir(config: TrainConfig) -> Path:
    root = (
        SMOKE_RESULTS_ROOT
        if config.steps == 2 and config.latent_dim == 1
        else RESULTS_ROOT
    )
    return root / config.dataset / f"{config.prior}_d{config.latent_dim}"


def train_one(config: TrainConfig, force: bool = False) -> Path:
    spec = SPECS[config.dataset]
    run_dir = _run_dir(config)
    checkpoint_path = run_dir / "checkpoint.pt"
    metrics_path = run_dir / "metrics.json"
    if checkpoint_path.exists() and metrics_path.exists() and not force:
        existing_config = json.loads(metrics_path.read_text()).get("config", {})
        # Round-trip through JSON so tuple fields compare with the lists stored
        # in metrics.json.
        requested_config = json.loads(json.dumps(asdict(config)))
        if all(existing_config.get(key) == value for key, value in requested_config.items()):
            print(f"[skip] completed {run_dir}")
            return run_dir
        raise RuntimeError(
            f"{run_dir} contains results from a different configuration; "
            "use --force to replace that exact run."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    sequence = load_sequence(spec, spec.train_id, config.image_size)
    dataset = WindowDataset(sequence, spec.window, config.recon_indices)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=device.type == "cuda", generator=generator,
    )
    encoder = Encoder(768, config.encoder_hidden, config.latent_dim).to(device)
    decoder = Decoder(config.latent_dim, config.decoder_init_channels, (128, 64, 32, 16)).to(device)
    prior = build_prior(config.prior, config.latent_dim, config.prior_hidden, spec.dt, spec.window)
    if prior is not None:
        prior = prior.to(device)
    main_optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.lr, weight_decay=config.weight_decay,
    )
    prior_optimizer = None if prior is None else torch.optim.AdamW(
        prior.parameters(), lr=config.prior_lr, weight_decay=config.weight_decay,
    )
    iterator = iter(loader)
    history = []
    started = time.time()
    for step in range(1, config.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        cls = batch["cls"].to(device, non_blocking=True)
        u = batch["u"].to(device, non_blocking=True)
        images = batch["images"].to(device, non_blocking=True)
        for group in main_optimizer.param_groups:
            group["lr"] = _learning_rate(step, config)
        encoder.train(); decoder.train()
        if prior is not None:
            prior.train()
            with torch.no_grad():
                frozen_mu, frozen_logvar = encoder(cls)
            for _ in range(config.inner_steps):
                prior_trajectory = prior.rollout(frozen_mu[:, 0], u)
                prior_logvar = prior.get_prior_logvar(prior_trajectory, u, config.kl_indices)
                inner_kl, _ = temporal_kl_loss(
                    frozen_mu, frozen_logvar, prior_trajectory, prior_logvar,
                    config.kl_indices, config.free_nats,
                )
                prior_optimizer.zero_grad(set_to_none=True)
                inner_kl.backward()
                torch.nn.utils.clip_grad_norm_(prior.parameters(), config.grad_clip)
                prior_optimizer.step()
        else:
            inner_kl = torch.tensor(float("nan"), device=device)

        mu, logvar = encoder(cls)
        reconstruction_loss, reconstruction_stats = multi_frame_recon(
            images, mu, logvar, decoder, config.recon_indices,
        )
        main_optimizer.zero_grad(set_to_none=True)
        if prior is None:
            reconstruction_loss.backward()
            temporal_value = float("nan")
            conflict = False
        else:
            prior_trajectory = prior.rollout(mu[:, 0].detach(), u)
            prior_logvar = prior.get_prior_logvar(prior_trajectory, u, config.kl_indices)
            temporal_loss, temporal_stats = temporal_kl_loss(
                mu, logvar, prior_trajectory, prior_logvar,
                config.kl_indices, config.free_nats,
            )
            weighted_temporal = config.beta * temporal_loss
            conflict = bool(pcgrad_surgery(encoder, reconstruction_loss, weighted_temporal, retain_graph=True))
            encoder_gradients = {
                name: parameter.grad.clone()
                for name, parameter in encoder.named_parameters()
                if parameter.grad is not None
            }
            main_optimizer.zero_grad(set_to_none=True)
            (reconstruction_loss + weighted_temporal).backward()
            for name, parameter in encoder.named_parameters():
                if name in encoder_gradients:
                    parameter.grad = encoder_gradients[name]
            temporal_value = float(temporal_stats["kl"].item())
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), config.grad_clip
        )
        main_optimizer.step()
        if step % config.log_every == 0 or step == 1 or step == config.steps:
            record = {
                "step": step,
                "reconstruction_l1": float(reconstruction_stats["l1"].item()),
                "temporal_kl": temporal_value,
                "inner_kl": float(inner_kl.item()),
                "pcgrad_conflict": conflict,
                "elapsed_s": time.time() - started,
            }
            history.append(record)
            print(
                f"[{config.dataset} {config.prior} d={config.latent_dim}] "
                f"{step}/{config.steps} rec={record['reconstruction_l1']:.4f} "
                f"kl={record['temporal_kl']:.3f} elapsed={record['elapsed_s']:.1f}s",
                flush=True,
            )

    checkpoint = {
        "config": asdict(config), "spec": asdict(spec),
        "encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
        "prior": None if prior is None else prior.state_dict(),
        "history": history,
    }
    torch.save(checkpoint, checkpoint_path)
    metrics = evaluate(config, checkpoint_path, device=device)
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False))
    pd.DataFrame(metrics["per_sequence"]).to_csv(run_dir / "metrics_per_sequence.csv", index=False)
    print(f"[done] {run_dir}")
    return run_dir


def load_models(config: TrainConfig, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = Encoder(768, config.encoder_hidden, config.latent_dim).to(device)
    decoder = Decoder(config.latent_dim, config.decoder_init_channels, (128, 64, 32, 16)).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    prior = build_prior(config.prior, config.latent_dim, config.prior_hidden,
                        SPECS[config.dataset].dt, SPECS[config.dataset].window)
    if prior is not None:
        prior = prior.to(device)
        prior.load_state_dict(checkpoint["prior"])
        prior.eval()
    encoder.eval(); decoder.eval()
    return encoder, decoder, prior


@torch.no_grad()
def encode_decode(sequence: Sequence, encoder: nn.Module, decoder: nn.Module,
                  device: torch.device, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    latent_chunks, reconstruction_chunks = [], []
    for start in range(0, sequence.length, batch_size):
        cls = sequence.cls[start:start+batch_size].to(device)
        mu, _ = encoder(cls)
        reconstructed = torch.sigmoid(decoder(mu))
        latent_chunks.append(mu.cpu())
        reconstruction_chunks.append(reconstructed.cpu())
    latent = torch.cat(latent_chunks).numpy()
    reconstructed = torch.cat(reconstruction_chunks).numpy()[:, 0]
    return latent, reconstructed


def _image_metrics(actual: np.ndarray, reconstructed: np.ndarray) -> dict:
    actual = actual.astype(np.float32) / 255.0
    reconstructed = np.clip(reconstructed.astype(np.float32), 0.0, 1.0)
    per_frame_mae = np.mean(np.abs(actual - reconstructed), axis=(1, 2))
    per_frame_mse = np.mean((actual - reconstructed) ** 2, axis=(1, 2))
    per_frame_psnr = -10.0 * np.log10(np.maximum(per_frame_mse, 1e-12))
    per_frame_ssim = np.array([
        structural_similarity(a, b, data_range=1.0)
        for a, b in zip(actual, reconstructed)
    ])
    return {
        "mae": float(per_frame_mae.mean()),
        "psnr_db": float(per_frame_psnr.mean()),
        "ssim": float(per_frame_ssim.mean()),
    }


@torch.no_grad()
def _rollout_nrmse(prior: Optional[nn.Module], latent: np.ndarray, u: torch.Tensor,
                   train_mean: np.ndarray, train_std: np.ndarray, horizon: int,
                   device: torch.device) -> Optional[float]:
    if prior is None or len(latent) <= horizon:
        return None
    starts = np.arange(0, len(latent) - horizon, max(1, horizon // 5))
    errors = []
    for batch_start in range(0, len(starts), 128):
        batch_indices = starts[batch_start:batch_start+128]
        z0 = torch.tensor(latent[batch_indices], dtype=torch.float32, device=device)
        u_windows = torch.stack([u[index:index+horizon+1] for index in batch_indices]).to(device)
        predicted = prior.rollout(z0, u_windows).cpu().numpy()[:, 1:]
        target = np.stack([latent[index+1:index+horizon+1] for index in batch_indices])
        errors.append(((predicted - target) / train_std.reshape(1, 1, -1)) ** 2)
    return float(np.sqrt(np.concatenate(errors).mean()))


def _aggregate_sequence_metrics(rows: List[dict]) -> dict:
    output = {}
    for key in ("mae", "psnr_db", "ssim", "rollout_nrmse"):
        values = np.array([row[key] for row in rows if row.get(key) is not None], dtype=float)
        output[key] = None if len(values) == 0 else {
            "mean": float(values.mean()),
            "std_across_sequences": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n_sequences": int(len(values)),
        }
    return output


def evaluate(config: TrainConfig, checkpoint_path: Path,
             device: Optional[torch.device] = None) -> dict:
    spec = SPECS[config.dataset]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, prior = load_models(config, checkpoint_path, device)
    train_sequence = load_sequence(spec, spec.train_id, config.image_size)
    train_latent, train_reconstruction = encode_decode(train_sequence, encoder, decoder, device)
    train_mean = train_latent.mean(axis=0)
    train_std = np.maximum(train_latent.std(axis=0), 1e-4)
    rows = []
    sample_payload = {}
    for sequence_id in (spec.train_id, *spec.heldout_ids):
        sequence = train_sequence if sequence_id == spec.train_id else load_sequence(spec, sequence_id, config.image_size)
        if sequence_id == spec.train_id:
            latent, reconstructed = train_latent, train_reconstruction
        else:
            latent, reconstructed = encode_decode(sequence, encoder, decoder, device)
        metrics = _image_metrics(sequence.images.numpy()[:, 0], reconstructed)
        metrics["rollout_nrmse"] = _rollout_nrmse(
            prior, latent, sequence.u, train_mean, train_std,
            config.rollout_horizon, device,
        )
        metrics.update({"sequence_id": sequence_id, "split": "train" if sequence_id == spec.train_id else "heldout"})
        rows.append(metrics)
        sample_indices = np.linspace(0, sequence.length - 1, 4).round().astype(int)
        sample_payload[f"{sequence_id}_indices"] = sample_indices
        sample_payload[f"{sequence_id}_actual"] = sequence.images.numpy()[sample_indices, 0]
        sample_payload[f"{sequence_id}_reconstructed"] = reconstructed[sample_indices]
        sample_payload[f"{sequence_id}_latent"] = latent
        sample_payload[f"{sequence_id}_u"] = sequence.u.numpy()
    heldout = [row for row in rows if row["split"] == "heldout"]
    np.savez_compressed(_run_dir(config) / "evaluation_samples.npz", **sample_payload)
    return {
        "config": asdict(config), "spec": asdict(spec),
        "per_sequence": rows,
        "heldout_summary": _aggregate_sequence_metrics(heldout),
        "parameter_counts": {
            "encoder": sum(p.numel() for p in encoder.parameters()),
            "decoder": sum(p.numel() for p in decoder.parameters()),
            "prior": 0 if prior is None else sum(p.numel() for p in prior.parameters()),
        },
    }


def run_suite(steps: int, force: bool = False) -> None:
    for dataset in SPECS:
        for prior in ("none", "rnn", "lstm", "transformer", "node"):
            train_one(TrainConfig(dataset=dataset, prior=prior, latent_dim=5, steps=steps), force=force)
        for latent_dim in (2, 3, 7):
            train_one(TrainConfig(dataset=dataset, prior="node", latent_dim=latent_dim, steps=steps), force=force)


def smoke_test() -> None:
    for prior in ("none", "rnn", "lstm", "transformer", "node"):
        config = TrainConfig(
            # d=1 is reserved for smoke tests and cannot collide with the
            # paper's d in {2, 3, 5, 7} dimension check.
            dataset="WAAM", prior=prior, latent_dim=1, steps=2,
            batch_size=2, inner_steps=1, log_every=1,
        )
        train_one(config, force=True)
    print("All prior types completed the smoke test.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset", choices=tuple(SPECS), required=True)
    train_parser.add_argument("--prior", choices=("none", "rnn", "lstm", "transformer", "node"), default="node")
    train_parser.add_argument("--latent-dim", type=int, default=5)
    train_parser.add_argument("--steps", type=int, default=3000)
    train_parser.add_argument("--force", action="store_true")
    suite_parser = subparsers.add_parser("suite")
    suite_parser.add_argument("--steps", type=int, default=3000)
    suite_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("smoke")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "smoke":
        smoke_test()
    elif args.command == "suite":
        run_suite(args.steps, force=args.force)
    else:
        train_one(TrainConfig(
            dataset=args.dataset, prior=args.prior,
            latent_dim=args.latent_dim, steps=args.steps,
        ), force=args.force)
