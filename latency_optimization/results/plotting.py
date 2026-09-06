"""Small plotting primitives shared by uplink and downlink figures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.transforms import blended_transform_factory


def safe_db(values):
    converted = 10.0 * np.log10(np.maximum(np.asarray(values, dtype=float), 1e-30))
    return float(converted) if np.isscalar(values) else converted


def finite_histogram_bins(*value_groups, count: int = 40) -> np.ndarray:
    """Return finite, non-degenerate shared bin edges for comparable histograms."""
    finite_groups = [
        np.asarray(values, dtype=float).reshape(-1)
        for values in value_groups
        if np.asarray(values).size > 0
    ]
    finite = np.concatenate(finite_groups) if finite_groups else np.asarray([], dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, max(int(count), 2))
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if lower == upper:
        padding = max(abs(lower) * 0.05, 1.0e-12)
        lower -= padding
        upper += padding
    return np.linspace(lower, upper, max(int(count), 2))


def load_interference_diagnostics(payload):
    if not payload:
        return None
    return {
        "blocks_per_user": [int(value) for value in payload.get("blocks_per_user", [])],
        "signal": np.asarray(payload.get("signal", []), dtype=float),
        "total_interference": np.asarray(payload.get("total_interference", []), dtype=float),
        "noise": np.asarray(payload.get("noise", []), dtype=float),
        "sinr_db": np.asarray(payload.get("sinr_db", []), dtype=float),
        "pairwise_block": np.asarray(payload.get("pairwise_block", []), dtype=float),
        "avg_pairwise_power": np.asarray(payload.get("avg_pairwise_power", []), dtype=float),
        "avg_pairwise_inr_db": np.asarray(payload.get("avg_pairwise_inr_db", []), dtype=float),
        "avg_pairwise_share": np.asarray(payload.get("avg_pairwise_share", []), dtype=float),
        "worst_block": int(payload.get("worst_block", -1)),
    }


def imshow_user_pair_matrix(
    ax,
    matrix,
    title: str,
    cbar_label: str,
    *,
    cmap: str = "viridis",
    center_zero: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    matrix = np.asarray(matrix, dtype=float)
    ax.set_title(title)
    ax.set_xlabel("Interferer user")
    ax.set_ylabel("Victim user")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    if not np.any(np.isfinite(matrix)):
        ax.text(0.5, 0.5, "No active user pair", ha="center", va="center", transform=ax.transAxes)
        return

    norm = None
    if center_zero:
        finite = matrix[np.isfinite(matrix)]
        bound = max(abs(float(np.min(finite))), abs(float(np.max(finite))), 1e-12)
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    image_kwargs = {"aspect": "auto", "interpolation": "none", "cmap": cmap, "norm": norm}
    if norm is None:
        image_kwargs.update(vmin=vmin, vmax=vmax)
    image = ax.imshow(np.ma.masked_invalid(matrix), **image_kwargs)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def add_legend_if_present(ax, *, handles=None, labels=None, **kwargs) -> None:
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if handles and any(str(label).strip() for label in labels):
        ax.legend(handles=handles, labels=labels, **kwargs)


def mask_inactive_user_pairs(matrix, bits_by_user) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    active = np.asarray(
        [sum(int(bits) for bits in user_bits) > 0 for user_bits in bits_by_user],
        dtype=bool,
    )
    if active.size != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        return matrix
    return np.where(active[:, None] & active[None, :], matrix, np.nan)


def combined_finite_limits(*matrices) -> tuple[float | None, float | None]:
    finite_parts = []
    for matrix in matrices:
        values = np.asarray(matrix, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            finite_parts.append(finite.reshape(-1))
    if not finite_parts:
        return None, None
    combined = np.concatenate(finite_parts)
    return float(np.min(combined)), float(np.max(combined))


def build_segmented_epoch_positions(
    rows: Sequence[dict[str, Any]],
    *,
    group_key: str,
    gap: float = 1.5,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    positions: list[float] = []
    segments: list[dict[str, Any]] = []
    cursor = 0.0
    index = 0
    while index < len(rows):
        group = rows[index].get(group_key)
        segment_start = cursor + 1.0
        local_epoch = 1
        while index < len(rows) and rows[index].get(group_key) == group:
            positions.append(cursor + local_epoch)
            index += 1
            local_epoch += 1
        length = local_epoch - 1
        segment_end = cursor + length
        segments.append(
            {"group": group, "start": segment_start, "end": segment_end, "length": length}
        )
        cursor = segment_end + float(gap)
    return np.asarray(positions, dtype=float), segments


def segmented_epoch_ticks(
    segments: Sequence[dict[str, Any]],
    max_labels_per_segment: int = 4,
) -> tuple[list[float], list[str]]:
    positions: list[float] = []
    labels: list[str] = []
    for segment in segments:
        length = int(segment["length"])
        if length <= 0:
            continue
        epochs = np.unique(np.linspace(1, length, min(max_labels_per_segment, length), dtype=int))
        for epoch in epochs:
            positions.append(float(segment["start"]) + int(epoch) - 1)
            labels.append(str(int(epoch)))
    return positions, labels


def draw_epoch_segment_guides(
    ax,
    segments: Sequence[dict[str, Any]],
    *,
    label_prefix: str,
    show_segment_labels: bool = False,
) -> None:
    for segment in segments:
        ax.axvline(float(segment["start"]) - 0.5, color="black", linestyle="--", linewidth=0.9, alpha=0.25)
    if not show_segment_labels or not segments:
        return

    stride = max(1, int(np.ceil(len(segments) / 16.0)))
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    for index, segment in enumerate(segments):
        if index % stride:
            continue
        ax.text(
            0.5 * (float(segment["start"]) + float(segment["end"])),
            1.02,
            f"{label_prefix} {segment['group']}",
            transform=transform,
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )


def convergence_status_color(status: str) -> str:
    return {
        "kkt_converged": "tab:green",
        "objective_stationary": "tab:green",
        "stationary_infeasible": "tab:red",
        "max_epochs_reached": "tab:orange",
        "max_epochs_feasible_best": "tab:blue",
        "max_epochs_best_primal": "tab:purple",
        "unknown": "tab:gray",
        "": "tab:gray",
    }.get(str(status), "tab:gray")


__all__ = [
    "add_legend_if_present",
    "build_segmented_epoch_positions",
    "combined_finite_limits",
    "convergence_status_color",
    "draw_epoch_segment_guides",
    "imshow_user_pair_matrix",
    "load_interference_diagnostics",
    "mask_inactive_user_pairs",
    "safe_db",
    "segmented_epoch_ticks",
]
