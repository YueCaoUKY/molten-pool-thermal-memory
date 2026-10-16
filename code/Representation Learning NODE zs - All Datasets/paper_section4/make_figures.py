"""Generate the paper-facing Section 4 figures and metric tables.

All selection rules are deterministic. In particular, the reconstruction panel
uses the held-out experiment whose SSIM is closest to the process median and
then the frame whose MAE is closest to that experiment's median frame MAE.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import torch

from experiment import (
    HERE,
    RESULTS_ROOT,
    SPECS,
    TrainConfig,
    _load_gmaw,
    _load_gtaw,
    encode_decode,
    load_models,
    load_sequence,
)

PAPER_ROOT = HERE.parents[2] / "Paper writing"
FIGURE_ROOT = PAPER_ROOT / "figures" / "section4"
TABLE_ROOT = PAPER_ROOT / "development_notes" / "section4_results"
FONT = "Liberation Sans"
BLUE = "#2F6690"
GOLD = "#D99A2B"
TEAL = "#2A9D8F"
INPUT = "#3F7D5A"
INK = "#20262E"
LIGHT_BLUE = "#EAF2F8"
LIGHT_GOLD = "#FFF4DD"
GRID = "#D9DEE3"


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": FONT,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_ROOT / f"{stem}.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIGURE_ROOT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _box(ax, xy, width, height, text, face, edge=BLUE, fontsize=6.5,
         dashed=False, weight="normal"):
    box = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face, edgecolor=edge, linewidth=0.8,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, color=INK, weight=weight)
    return box


def _arrow(ax, start, end, color=INK, connectionstyle="arc3"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=7, linewidth=0.8,
        color=color, connectionstyle=connectionstyle, shrinkA=1, shrinkB=1,
    ))


def make_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y, h = 0.62, 0.20
    xs = [0.015, 0.205, 0.415, 0.615, 0.810]
    ws = [0.145, 0.155, 0.145, 0.145, 0.17]
    labels = [
        "Pool frame\n$x_t$",
        "Frozen DINOv2\n768-D",
        "Encoder\n$q_\\phi(\\mathbf{z}_t|x_t)$",
        "State\n$\\mathbf{z}_t\\in\\mathbb{R}^5$",
        "Decoder\n$\\hat x_t$",
    ]
    faces = ["#F4F5F6", LIGHT_BLUE, LIGHT_BLUE, LIGHT_GOLD, LIGHT_BLUE]
    for x, w, label, face in zip(xs, ws, labels, faces):
        _box(ax, (x, y), w, h, label, face, fontsize=5.3,
             weight="bold" if "State" in label else "normal")
    for left, right, wl, wr in zip(xs[:-1], xs[1:], ws[:-1], ws[1:]):
        _arrow(ax, (left + wl, y + h / 2), (right, y + h / 2))

    prior_x, prior_y, prior_w, prior_h = 0.285, 0.16, 0.38, 0.22
    _box(
        ax, (prior_x, prior_y), prior_w, prior_h,
        "Auxiliary control-affine prior\n"
        "$\\dot{\\bar{\\mathbf{z}}}=\\mathbf{f}(\\bar{\\mathbf{z}})+"
        "\\mathbf{g}(\\bar{\\mathbf{z}})u$",
        LIGHT_GOLD, edge=GOLD, dashed=True, fontsize=5.3,
    )
    ax.text(0.075, prior_y + prior_h / 2, "Input\n$u_{0:T-1}$",
            ha="center", va="center", color=INK, fontsize=5.5)
    _arrow(ax, (0.13, prior_y + prior_h / 2), (prior_x, prior_y + prior_h / 2), GOLD)
    _arrow(ax, (xs[3] + ws[3] / 2, y), (prior_x + prior_w * 0.68, prior_y + prior_h), GOLD)
    ax.text(0.835, prior_y + prior_h / 2,
            "$\\mathcal{L}_{dyn}$\nposterior–rollout\nalignment",
            ha="center", va="center", color=INK, fontsize=5.3)
    _arrow(ax, (prior_x + prior_w, prior_y + prior_h / 2), (0.745, prior_y + prior_h / 2), GOLD)
    ax.text(0.895, 0.50, "$\\mathcal{L}_{rec}$", ha="center", va="center", color=BLUE)
    _arrow(ax, (xs[4] + ws[4] / 2, y), (0.895, 0.535), BLUE)
    save_figure(fig, "fig2_state_extraction_pipeline")


def _montage(frames: np.ndarray) -> np.ndarray:
    gap = 3
    strips = []
    for index, frame in enumerate(frames):
        strips.append(frame)
        if index < len(frames) - 1:
            strips.append(np.full((frame.shape[0], gap), 255, dtype=np.uint8))
    return np.concatenate(strips, axis=1)


def make_dataset_overview() -> None:
    fig = plt.figure(figsize=(3.46, 3.05))
    gs = fig.add_gridspec(3, 2, width_ratios=(1.55, 1.0), hspace=0.32, wspace=0.17)
    for row, name in enumerate(("GTAW", "GMAW", "WAAM")):
        spec = SPECS[name]
        seq = load_sequence(spec, spec.train_id, 128)
        indices = np.linspace(0, seq.length - 1, 4).round().astype(int)
        frames = seq.images.numpy()[indices, 0]

        ax_image = fig.add_subplot(gs[row, 0])
        ax_image.imshow(_montage(frames), cmap="gray", vmin=0, vmax=255)
        ax_image.set_xticks([]); ax_image.set_yticks([])
        for spine in ax_image.spines.values():
            spine.set_color("#AEB5BC"); spine.set_linewidth(0.5)
        ax_image.text(0.015, 0.94, name, transform=ax_image.transAxes, ha="left", va="top",
                      color="white", fontsize=6.3, weight="bold",
                      bbox={"facecolor": INK, "edgecolor": "none", "alpha": 0.72, "pad": 1.2})
        if row == 0:
            ax_image.set_title("Complete training sequence", pad=3)
        width = frames.shape[-1]
        for col, idx in enumerate(indices):
            center = col * (width + 3) + width / 2
            ax_image.text(center, frames.shape[1] - 3, f"{idx * spec.dt:.1f} s",
                          ha="center", va="bottom", fontsize=5.2, color="white",
                          bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.45, "pad": 0.8})

        ax_input = fig.add_subplot(gs[row, 1])
        time_axis = np.arange(seq.length) * spec.dt
        if name == "GTAW":
            _, u_raw = _load_gtaw(spec.train_id, 128)
            ax_input.plot(time_axis, u_raw.numpy(), color=INPUT)
        elif name == "GMAW":
            _, u_raw = _load_gmaw(spec.train_id, 128)
            ax_input.plot(time_axis, u_raw.numpy(), color=INPUT)
        else:
            ax_input.plot(time_axis, np.full(seq.length, 2.7), color=INPUT)
            ax_input.text(0.97, 0.86, "200 A, constant", transform=ax_input.transAxes,
                          ha="right", va="top", fontsize=5.2, color=INK)
        ax_input.set_xlim(time_axis[0], time_axis[-1])
        ax_input.set_ylim(0, 12.5)
        ax_input.set_yticks([0, 6, 12])
        ax_input.set_ylabel("")
        ax_input.set_xlabel("time (s)" if row == 2 else "", labelpad=1)
        ax_input.grid(axis="y", color=GRID, linewidth=0.45)
        ax_input.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax_input.set_title(r"Travel speed $v$ (mm s$^{-1}$)", pad=3)
    fig.subplots_adjust(left=0.035, right=0.99, top=0.94, bottom=0.08)
    save_figure(fig, "fig3_dataset_overview")


def _load_metric(dataset: str, prior: str, latent_dim: int = 5) -> dict:
    path = RESULTS_ROOT / dataset / f"{prior}_d{latent_dim}" / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Run not complete: {path}")
    return json.loads(path.read_text())


def make_reconstruction_alignment() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = []
    image_rows = []
    all_errors = []
    for name in ("GTAW", "GMAW", "WAAM"):
        metrics = _load_metric(name, "node")
        heldout = [row for row in metrics["per_sequence"] if row["split"] == "heldout"]
        median_ssim = float(np.median([row["ssim"] for row in heldout]))
        selected_row = min(heldout, key=lambda row: abs(row["ssim"] - median_ssim))
        sequence_id = selected_row["sequence_id"]
        config = TrainConfig(**metrics["config"])
        checkpoint = RESULTS_ROOT / name / "node_d5" / "checkpoint.pt"
        encoder, decoder, _ = load_models(config, checkpoint, device)
        sequence = load_sequence(SPECS[name], sequence_id, config.image_size)
        _, reconstructed = encode_decode(sequence, encoder, decoder, device)
        actual = sequence.images.numpy()[:, 0].astype(np.float32) / 255.0
        errors = np.mean(np.abs(actual - reconstructed), axis=(1, 2))
        median_error = float(np.median(errors))
        frame_index = int(np.argmin(np.abs(errors - median_error)))
        error_map = np.abs(actual[frame_index] - reconstructed[frame_index])
        selected.append({
            "process": name, "sequence_id": sequence_id, "frame_index": frame_index,
            "sequence_ssim": selected_row["ssim"], "frame_mae": float(errors[frame_index]),
        })
        image_rows.append((actual[frame_index], reconstructed[frame_index], error_map))
        all_errors.append(error_map)

    vmax = max(0.08, float(np.quantile(np.concatenate([x.ravel() for x in all_errors]), 0.995)))
    fig, axes = plt.subplots(3, 3, figsize=(3.46, 3.05))
    for row, (name, images, audit) in enumerate(zip(("GTAW", "GMAW", "WAAM"), image_rows, selected)):
        display_low, display_high = np.quantile(images[0], [0.01, 0.995])
        if display_high - display_low < 0.12:
            display_high = min(1.0, display_low + 0.12)
        for col, image in enumerate(images):
            if col < 2:
                axes[row, col].imshow(image, cmap="gray", vmin=display_low, vmax=display_high)
            else:
                mappable = axes[row, col].imshow(image, cmap="magma", vmin=0, vmax=vmax)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_linewidth(0.45); spine.set_color("#AEB5BC")
        axes[row, 0].set_ylabel(
            f"{name}\nSSIM {audit['sequence_ssim']:.2f}", rotation=0,
            ha="right", va="center", labelpad=4, weight="bold", fontsize=5.7,
        )
    for ax, title in zip(axes[0], ("Original", "Reconstruction", "Absolute error")):
        ax.set_title(title, pad=3)
    fig.subplots_adjust(left=0.15, right=0.91, top=0.93, bottom=0.045, hspace=0.08, wspace=0.07)
    cax = fig.add_axes([0.925, 0.10, 0.018, 0.73])
    cbar = fig.colorbar(mappable, cax=cax, orientation="vertical")
    cbar.set_label("absolute pixel error", labelpad=2)
    cbar.ax.tick_params(length=2, pad=1)
    save_figure(fig, "fig4_reconstruction_alignment")
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_csv(TABLE_ROOT / "figure4_selection_audit.csv", index=False)


def _fmt(summary: dict, key: str, digits: int) -> str:
    value = summary[key]
    if value is None:
        return "—"
    return f"{value['mean']:.{digits}f} ± {value['std_across_sequences']:.{digits}f}"


def _to_markdown(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown table writer."""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|") for value in values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def make_tables() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    prior_rows = []
    reconstruction_rows = []
    for dataset in SPECS:
        node_metrics = _load_metric(dataset, "node")
        node_summary = node_metrics["heldout_summary"]
        train_row = next(row for row in node_metrics["per_sequence"] if row["split"] == "train")
        reconstruction_rows.append({
            "Process": dataset,
            "Held-out experiments": node_summary["ssim"]["n_sequences"],
            "Training SSIM": f"{train_row['ssim']:.3f}",
            "Held-out MAE": _fmt(node_summary, "mae", 3),
            "Held-out PSNR (dB)": _fmt(node_summary, "psnr_db", 2),
            "Held-out SSIM": _fmt(node_summary, "ssim", 3),
        })
        for prior in ("none", "rnn", "lstm", "transformer", "node"):
            prior_metrics = _load_metric(dataset, prior)
            summary = prior_metrics["heldout_summary"]
            prior_rows.append({
                "Process": dataset,
                "Prior": {"none": "None", "rnn": "RNN", "lstm": "LSTM",
                          "transformer": "Transformer", "node": "Control-affine NODE"}[prior],
                "Prior parameters": prior_metrics["parameter_counts"]["prior"],
                "SSIM": _fmt(summary, "ssim", 3),
                "Rollout NRMSE": _fmt(summary, "rollout_nrmse", 3),
            })
    dimension_rows = []
    for dataset in SPECS:
        for latent_dim in (2, 3, 5, 7):
            summary = _load_metric(dataset, "node", latent_dim)["heldout_summary"]
            dimension_rows.append({
                "Process": dataset,
                "$d_z$": latent_dim,
                "MAE": _fmt(summary, "mae", 3),
                "PSNR (dB)": _fmt(summary, "psnr_db", 2),
                "SSIM": _fmt(summary, "ssim", 3),
                "Rollout NRMSE": _fmt(summary, "rollout_nrmse", 3),
            })
    for name, rows in (
        ("reconstruction_table", reconstruction_rows),
        ("prior_comparison_table", prior_rows),
        ("dimension_sensitivity_table", dimension_rows),
    ):
        frame = pd.DataFrame(rows)
        frame.to_csv(TABLE_ROOT / f"{name}.csv", index=False)
        (TABLE_ROOT / f"{name}.md").write_text(_to_markdown(frame))


def make_prior_comparison() -> None:
    """Turn the 15-row prior table into a compact evidence figure."""
    priors = ("none", "rnn", "lstm", "transformer", "node")
    labels = {
        "none": "none",
        "rnn": "RNN",
        "lstm": "LSTM",
        "transformer": "Transformer",
        "node": "control-affine NODE",
    }
    colors = {
        "none": "#aeb6bf",
        "rnn": "#7b8794",
        "lstm": "#5f6f7f",
        "transformer": "#8d6e63",
        "node": BLUE,
    }
    markers = {"none": "x", "rnn": "o", "lstm": "s", "transformer": "^", "node": "D"}
    datasets = ("GTAW", "GMAW", "WAAM")
    y = np.arange(len(datasets))[::-1]
    offsets = dict(zip(priors, np.linspace(-0.24, 0.24, len(priors))))

    fig, (ax_roll, ax_ssim) = plt.subplots(1, 2, figsize=(3.46, 2.25), sharey=True)
    for prior in priors:
        roll_mean, roll_std, ssim_mean, ssim_std = [], [], [], []
        for dataset in datasets:
            summary = _load_metric(dataset, prior)["heldout_summary"]
            rollout = summary["rollout_nrmse"]
            roll_mean.append(np.nan if rollout is None else rollout["mean"])
            roll_std.append(0 if rollout is None else rollout["std_across_sequences"])
            ssim_mean.append(summary["ssim"]["mean"])
            ssim_std.append(summary["ssim"]["std_across_sequences"])
        yy = y + offsets[prior]
        if prior != "none":
            ax_roll.errorbar(roll_mean, yy, xerr=roll_std, fmt=markers[prior], color=colors[prior],
                             markersize=4.0 if prior == "node" else 3.3,
                             markeredgewidth=0.8, capsize=1.8, linewidth=0.8,
                             label=labels[prior])
        ax_ssim.errorbar(ssim_mean, yy, xerr=ssim_std, fmt=markers[prior], color=colors[prior],
                         markersize=4.0 if prior == "node" else 3.3,
                         markeredgewidth=0.8, capsize=1.8, linewidth=0.8,
                         label=labels[prior])

    ax_roll.set_xscale("log")
    ax_roll.set_xlim(0.45, 5.2)
    ax_roll.set_xticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
    from matplotlib.ticker import NullFormatter, NullLocator
    ax_roll.xaxis.set_minor_locator(NullLocator())
    ax_roll.xaxis.set_minor_formatter(NullFormatter())
    ax_roll.set_xlabel("25-frame rollout NRMSE\n(lower is better)")
    ax_ssim.set_xlim(0.48, 0.92)
    ax_ssim.set_xlabel("held-out SSIM\n(higher is better)")
    ax_roll.set_yticks(y, datasets)
    for ax in (ax_roll, ax_ssim):
        ax.grid(axis="x", color=GRID, linewidth=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#7f8993")
        ax.tick_params(length=2.2, width=0.6)
    handles, legend_labels = ax_ssim.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.52, 0.995),
               ncol=3, frameon=False, fontsize=5.8, handletextpad=0.35,
               columnspacing=0.75)
    fig.subplots_adjust(left=0.18, right=0.99, top=0.78, bottom=0.25, wspace=0.24)
    save_figure(fig, "fig5_prior_comparison")


def main() -> None:
    configure_style()
    make_pipeline()
    make_dataset_overview()
    make_reconstruction_alignment()
    make_prior_comparison()
    make_tables()
    print(f"Figures: {FIGURE_ROOT}")
    print(f"Tables: {TABLE_ROOT}")


if __name__ == "__main__":
    main()
