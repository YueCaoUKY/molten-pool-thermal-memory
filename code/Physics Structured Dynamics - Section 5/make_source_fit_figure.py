#!/usr/bin/env python3
"""Create the Section 5 source-identification evidence figure and table.

The layout follows the successful visual grammar of the reference presentation:
source images, input history, measured-versus-fitted state, and inferred thermal
state. Unlike the older slides, this figure displays all five latent coordinates
and never computes PCA.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
FIGURE_DIR = PROJECT / "Paper writing" / "figures" / "section5"
NOTES_DIR = PROJECT / "Paper writing" / "development_notes" / "section5"
SECTION4 = PROJECT / "Coding Project" / "Representation Learning NODE zs - All Datasets"
sys.path.insert(0, str(SECTION4 / "code"))

from decoder import Decoder  # noqa: E402

SPECS = [
    ("GTAW", "Iden_06", 42),
    ("GMAW", "exp_01", 42),
    ("WAAM", "video_1", 42),
]

COLORS = {
    "observed": "#252b33",
    "fitted": "#2f6690",
    "input": "#3f7d5a",
    "thermal": "#d97706",
    "grid": "#d9dee5",
}


def load_result(dataset: str, seed: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    result_dir = HERE / "results" / dataset / f"seed_{seed}"
    with np.load(result_dir / "source_fit.npz") as data:
        arrays = {key: data[key] for key in data.files}
    metrics = json.loads((result_dir / "metrics.json").read_text())
    return arrays, metrics


def montage(images: np.ndarray, gap: int = 3) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim == 4 and images.shape[1] in (1, 3):
        images = np.moveaxis(images, 1, -1)
    if images.ndim == 4 and images.shape[-1] == 1:
        images = images[..., 0]
    h, w = images.shape[1:3]
    if images.ndim == 3:
        canvas = np.full((h, images.shape[0] * w + gap * (images.shape[0] - 1)), 255, dtype=np.uint8)
    else:
        canvas = np.full(
            (h, images.shape[0] * w + gap * (images.shape[0] - 1), images.shape[-1]),
            255,
            dtype=np.uint8,
        )
    for index, image in enumerate(images):
        start = index * (w + gap)
        canvas[:, start : start + w] = image
    return canvas


@torch.no_grad()
def decode_fitted_samples(dataset: str, data: dict[str, np.ndarray]) -> np.ndarray:
    checkpoint_path = SECTION4 / "paper_section4" / "results" / dataset / "node_d5" / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    decoder = Decoder(
        config["latent_dim"],
        config["decoder_init_channels"],
        (128, 64, 32, 16),
    )
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.eval()
    indices = data["sample_indices"]
    z_raw = data["z_predicted"][indices] * data["z_std"] + data["z_mean"]
    images = torch.sigmoid(decoder(torch.tensor(z_raw, dtype=torch.float32))).numpy()[:, 0]
    return np.clip(np.round(255.0 * images), 0, 255).astype(np.uint8)


def paired_montage(observed: np.ndarray, fitted: np.ndarray, gap: int = 3) -> np.ndarray:
    top = montage(observed, gap=gap)
    bottom = montage(fitted, gap=gap)
    separator = np.full((gap, top.shape[1]), 255, dtype=np.uint8)
    return np.concatenate([top, separator, bottom], axis=0)


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#79818c")
    ax.tick_params(length=2.2, width=0.6, colors="#4c535d", labelsize=8.0)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.45, alpha=0.65)


def write_metrics_table(rows: list[dict[str, object]]) -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = NOTES_DIR / "source_fit_metrics.csv"
    fields = [
        "process",
        "source_experiment",
        "frames",
        "duration_s",
        "latent_nrmse",
        "latent_r2_global",
        "coordinate_r2_median",
        "coordinate_r2_min",
        "coordinate_r2_max",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "| Process | Source experiment | Frames | Duration (s) | All-coordinate NRMSE | Global $R^2$ | Coordinate $R^2$, median [range] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {process} | `{source_experiment}` | {frames:d} | {duration_s:.2f} | "
            "{latent_nrmse:.3f} | {latent_r2_global:.3f} | {coordinate_r2_median:.3f} "
            "[{coordinate_r2_min:.3f}, {coordinate_r2_max:.3f}] |".format(**row)
        )
    (NOTES_DIR / "source_fit_metrics.md").write_text("\n".join(md_lines) + "\n")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.0,
            "legend.fontsize": 8.0,
            "svg.fonttype": "none",
        }
    )

    # 6.82 in = 17.32 cm, within the intended full-text paper width.
    fig = plt.figure(figsize=(6.82, 5.15), constrained_layout=False)
    outer = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        wspace=0.27,
        top=0.91,
        bottom=0.09,
        left=0.105,
        right=0.992,
    )
    letters = ["a", "b", "c"]
    table_rows: list[dict[str, object]] = []

    legend_handles = [
        Line2D([0], [0], color=COLORS["observed"], linewidth=0.8, label="encoded pool state"),
        Line2D([0], [0], color=COLORS["fitted"], linewidth=1.2, label="simulated pool state"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )

    for column_index, (dataset, source, seed) in enumerate(SPECS):
        data, metrics = load_result(dataset, seed)
        inner = gridspec.GridSpecFromSubplotSpec(
            4,
            1,
            subplot_spec=outer[column_index],
            height_ratios=[1.0, 0.42, 1.12, 0.44],
            hspace=0.13,
        )

        time = data["time"]
        u = data["u"]
        observed = data["z_observed"]
        fitted = data["z_predicted"]
        delta = data["delta"]
        sample_times = time[data["sample_indices"]]

        image_ax = fig.add_subplot(inner[0])
        decoded_samples = decode_fitted_samples(dataset, data)
        image_ax.imshow(
            paired_montage(data["sample_images"], decoded_samples),
            cmap="gray",
            vmin=0,
            vmax=255,
            aspect="equal",
        )
        image_ax.set_xticks([])
        if column_index == 0:
            image_ax.set_yticks(
                [0.25 * image_ax.images[0].get_array().shape[0], 0.75 * image_ax.images[0].get_array().shape[0]],
                ["observed", "decoded\nfit"],
                fontsize=7.2,
            )
            image_ax.tick_params(axis="y", length=0, pad=2.5)
        else:
            image_ax.set_yticks([])
        for spine in image_ax.spines.values():
            spine.set_color("#bbc2cb")
            spine.set_linewidth(0.55)
        image_ax.set_title(
            rf"$\bf{{{letters[column_index]}}}$  {dataset} — {source}",
            loc="left",
            pad=2.5,
        )
        for image_index, sample_time in enumerate(sample_times):
            image_ax.text(
                (image_index + 0.5) / len(sample_times),
                0.53,
                f"{sample_time:.1f} s",
                transform=image_ax.transAxes,
                ha="center",
                va="bottom",
                color="white",
                fontsize=7.0,
                bbox={"facecolor": "black", "alpha": 0.48, "edgecolor": "none", "pad": 1.0},
            )

        input_ax = fig.add_subplot(inner[1])
        input_ax.plot(time, u, color=COLORS["input"], linewidth=1.0)
        if column_index == 0:
            input_ax.set_ylabel("input\nu", rotation=0, labelpad=13, va="center")
        input_ax.set_xlim(time[0], time[-1])
        if np.ptp(u) < 1e-8:
            input_ax.set_ylim(-0.08, 1.0)
            input_ax.text(0.50, 0.65, "constant recorded input", transform=input_ax.transAxes,
                          ha="center", va="center", fontsize=6.8, color=COLORS["input"])
        else:
            pad = 0.08 * np.ptp(u)
            input_ax.set_ylim(np.min(u) - pad, np.max(u) + pad)
        input_ax.tick_params(labelbottom=False)
        style_axis(input_ax)

        latent_ax = fig.add_subplot(inner[2])
        offsets = np.arange(4, -1, -1, dtype=float) * 4.0
        for coordinate, offset in enumerate(offsets):
            latent_ax.plot(
                time,
                observed[:, coordinate] + offset,
                color=COLORS["observed"],
                linewidth=0.58,
                alpha=0.78,
                label="observed" if coordinate == 0 else None,
            )
            latent_ax.plot(
                time,
                fitted[:, coordinate] + offset,
                color=COLORS["fitted"],
                linewidth=0.95,
                label="free-running fit" if coordinate == 0 else None,
            )
        latent_ax.set_yticks(offsets, [f"z{index}" for index in range(1, 6)])
        if column_index == 0:
            latent_ax.set_ylabel("pool-state coordinates\n(standardized)", rotation=90, labelpad=7, va="center")
        latent_ax.set_xlim(time[0], time[-1])
        latent_ax.tick_params(labelbottom=False)
        style_axis(latent_ax)
        r2 = float(metrics["latent_r2_global"])
        nrmse = float(metrics["latent_nrmse"])
        latent_ax.set_title(
            f"$R^2$ = {r2:.3f}    NRMSE = {nrmse:.3f}",
            loc="right",
            pad=1.5,
            fontsize=7.0,
            color="#39414b",
        )
        thermal_ax = fig.add_subplot(inner[3])
        thermal_ax.plot(time, delta, color=COLORS["thermal"], linewidth=1.05)
        thermal_ax.fill_between(time, 0, delta, color=COLORS["thermal"], alpha=0.10, linewidth=0)
        if column_index == 0:
            thermal_ax.set_ylabel(
                "workpiece state δ\n(unidentified scale)",
                rotation=90,
                labelpad=8,
                va="center",
            )
        thermal_ax.set_xlabel("time (s)", labelpad=1.5)
        thermal_ax.set_xlim(time[0], time[-1])
        thermal_ax.set_ylim(bottom=0)
        thermal_ax.set_yticks([0], ["0"])
        style_axis(thermal_ax)

        r2_coord = np.asarray(metrics["latent_r2_per_coordinate"], dtype=float)
        table_rows.append(
            {
                "process": dataset,
                "source_experiment": source,
                "frames": int(metrics["frames"]),
                "duration_s": float(metrics["duration_s"]),
                "latent_nrmse": nrmse,
                "latent_r2_global": r2,
                "coordinate_r2_median": float(np.median(r2_coord)),
                "coordinate_r2_min": float(np.min(r2_coord)),
                "coordinate_r2_max": float(np.max(r2_coord)),
            }
        )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = FIGURE_DIR / "fig7_source_identification_fits.svg"
    png_path = FIGURE_DIR / "fig7_source_identification_fits.png"
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    write_metrics_table(table_rows)
    print(svg_path)
    print(png_path)


if __name__ == "__main__":
    main()
