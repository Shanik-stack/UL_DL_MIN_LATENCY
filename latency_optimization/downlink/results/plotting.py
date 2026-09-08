from __future__ import annotations

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from latency_optimization.results.plotting import (
    add_legend_if_present as _add_legend_if_present,
    build_segmented_epoch_positions as _build_segmented_epoch_positions,
    combined_finite_limits as _combined_finite_limits,
    convergence_status_color as _kkt_status_color,
    draw_epoch_segment_guides as _draw_epoch_segment_guides,
    finite_histogram_bins as _finite_histogram_bins,
    imshow_user_pair_matrix as _imshow_with_labels,
    load_interference_diagnostics as _load_interference_diag,
    mask_inactive_user_pairs as _mask_inactive_user_pairs,
    safe_db as _safe_db,
    segmented_epoch_ticks as _segmented_epoch_ticks,
)

from ..simulation.system import DownlinkSystem


def initialize_output_dirs(base_dir: str) -> dict[str, str]:
    dirs = {
        "data": os.path.join(base_dir, "data"),
        "user_config": os.path.join(base_dir, "user_config"),
        "latency_asynchronality": os.path.join(base_dir, "latency_asynchronality"),
        "link_quality": os.path.join(base_dir, "link_quality"),
        "optimization_history": os.path.join(base_dir, "optimization_history"),
        "schedule_details": os.path.join(base_dir, "schedule_details"),
        "interference": os.path.join(base_dir, "interference"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def _build_rate_violation_matrix(result: dict[str, Any]) -> np.ndarray:
    K = len(result.get("n_kl", []))
    max_blocks = max((len(v) for v in result.get("n_kl", [])), default=0)
    mat = np.full((K, max_blocks), np.nan, dtype=float)
    for point in result.get("rate_points", []):
        user = int(point["user"])
        block = int(point["block"])
        mat[user, block] = float(point["required_rate"]) - float(point["achieved_rate"])
    return mat


def _collect_final_interference_diagnostics(system: DownlinkSystem) -> dict[str, np.ndarray | int]:
    K = int(system.K)
    max_blocks = max((len(v) for v in system.n_kl), default=0)
    signal = np.full((K, max_blocks), np.nan, dtype=float)
    total_interference = np.full((K, max_blocks), np.nan, dtype=float)
    noise = np.full((K, max_blocks), np.nan, dtype=float)
    sinr_db = np.full((K, max_blocks), np.nan, dtype=float)
    pairwise_block = np.full((max_blocks, K, K), np.nan, dtype=float)
    pairwise_sum = np.zeros((K, K), dtype=float)
    pairwise_inr_sum = np.zeros((K, K), dtype=float)
    pairwise_count = np.zeros((K, K), dtype=float)

    for k in range(K):
        for l in range(len(system.n_kl[k])):
            Hk = np.asarray(system.H[k][l], dtype=np.complex128)
            Fk = np.asarray(system.F[k][l], dtype=np.complex128)
            signal_power = float(np.linalg.norm(Hk @ Fk, ord="fro") ** 2 / max(1, int(system.Nr[k])))
            noise_power = float(system.sigma2[k])
            interference_power = 0.0

            for j in range(K):
                if j == k or l >= len(system.F[j]):
                    continue
                Fj = np.asarray(system.F[j][l], dtype=np.complex128)
                coupling = float(np.linalg.norm(Hk @ Fj, ord="fro") ** 2 / max(1, int(system.Nr[k])))
                pairwise_block[l, k, j] = coupling
                pairwise_sum[k, j] += coupling
                pairwise_inr_sum[k, j] += coupling / max(noise_power, 1e-30)
                pairwise_count[k, j] += 1.0
                interference_power += coupling

            signal[k, l] = signal_power
            total_interference[k, l] = interference_power
            noise[k, l] = noise_power
            sinr_db[k, l] = float(
                10.0 * np.log10(max(signal_power / max(interference_power + noise_power, 1e-30), 1e-30))
            )

    avg_pairwise_power = np.divide(
        pairwise_sum,
        pairwise_count,
        out=np.full_like(pairwise_sum, np.nan),
        where=pairwise_count > 0,
    )
    avg_pairwise_inr_lin = np.divide(
        pairwise_inr_sum,
        pairwise_count,
        out=np.full_like(pairwise_inr_sum, np.nan),
        where=pairwise_count > 0,
    )
    row_sum = np.nansum(avg_pairwise_power, axis=1, keepdims=True)
    share = np.divide(
        avg_pairwise_power,
        row_sum,
        out=np.full_like(avg_pairwise_power, np.nan),
        where=row_sum > 0,
    )

    block_totals = np.nansum(pairwise_block, axis=(1, 2)) if max_blocks > 0 else np.asarray([], dtype=float)
    worst_block = int(np.nanargmax(block_totals)) if block_totals.size > 0 else -1

    return {
        "signal": signal,
        "total_interference": total_interference,
        "noise": noise,
        "sinr_db": sinr_db,
        "pairwise_block": pairwise_block,
        "avg_pairwise_power": avg_pairwise_power,
        "avg_pairwise_inr_db": _safe_db(avg_pairwise_inr_lin),
        "avg_pairwise_share": share,
        "worst_block": worst_block,
    }


def _epoch_history_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return list(result.get("epoch_history", []))


def _row_epoch(row: dict[str, Any]) -> int:
    return int(row.get("epoch", 0))


def plot_kkt_residual_history(result: dict[str, Any], figs_dir: str) -> None:
    epoch_history = [
        row
        for row in _epoch_history_rows(result)
        if "kkt_primal_residual" in row
        and "kkt_complementarity_residual" in row
        and "kkt_stationarity_residual" in row
    ]
    if len(epoch_history) == 0:
        return

    epoch_positions, segments = _build_segmented_epoch_positions(epoch_history, group_key="block")
    residual_specs = [
        (
            "kkt_primal_residual",
            "r_p",
            float(result.get("sim_params", {}).get("kkt_primal_tolerance", np.nan)),
        ),
        (
            "kkt_complementarity_residual",
            "r_c",
            float(result.get("sim_params", {}).get("kkt_complementarity_tolerance", np.nan)),
        ),
        (
            "kkt_stationarity_residual",
            "r_s",
            float(result.get("sim_params", {}).get("kkt_stationarity_tolerance", np.nan)),
        ),
    ]
    status_markers = [
        (float(epoch_positions[idx]), str(row.get("solve_status", "unknown")))
        for idx, row in enumerate(epoch_history)
        if bool(row.get("solve_segment_end", False))
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (field, label, tol_value) in zip(axes, residual_specs):
        values = np.asarray([float(row.get(field, np.nan)) for row in epoch_history], dtype=float)
        finite_positive = np.where(np.isfinite(values), np.maximum(values, 1e-12), np.nan)
        ax.semilogy(epoch_positions, finite_positive, color="tab:blue", linewidth=1.5)
        if np.isfinite(tol_value) and tol_value > 0.0:
            ax.axhline(float(tol_value), color="black", linestyle="--", linewidth=1.0, alpha=0.8)
        for marker_epoch, status in status_markers:
            ax.axvline(
                marker_epoch,
                color=_kkt_status_color(status),
                linestyle=":",
                linewidth=1.0,
                alpha=0.35,
            )
        _draw_epoch_segment_guides(
            ax,
            segments,
            label_prefix="Block",
            show_segment_labels=(ax is axes[0]),
        )
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"{label} residual history")

    tick_positions, tick_labels = _segmented_epoch_ticks(segments)
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(tick_labels)
    axes[-1].set_xlabel("Epoch within block (restarts at each block)")
    status_handles = []
    present_statuses = []
    for _, status in status_markers:
        if status not in present_statuses:
            present_statuses.append(status)
    for status in present_statuses:
        status_handles.append(
            Line2D(
                [0],
                [0],
                color=_kkt_status_color(status),
                linestyle=":",
                linewidth=2.0,
                label=status,
            )
        )
    tol_handle = Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="tolerance")
    line_handle = Line2D([0], [0], color="tab:blue", linewidth=1.8, label="residual")
    legend_handles = [line_handle, tol_handle] + status_handles
    axes[0].legend(handles=legend_handles, fontsize=8, loc="upper right", ncol=min(3, max(1, len(legend_handles))))
    fig.suptitle("KKT residual convergence history", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    plt.savefig(os.path.join(figs_dir, "kkt_residual_history.png"), dpi=250)
    plt.close(fig)


def plot_user_config(system_params: dict[str, Any], figs_dir: str) -> None:
    K = int(system_params["K"])
    users = np.arange(K)

    def _as_user_array(values: Any) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            return np.full(K, float(arr), dtype=float)
        if arr.size == 1:
            return np.full(K, float(arr.reshape(-1)[0]), dtype=float)
        flat = arr.reshape(-1)
        if flat.size >= K:
            return flat[:K]
        return np.pad(flat, (0, K - flat.size), mode="edge")

    params = {
        "snr_db": _as_user_array(system_params["snr_db"]),
        "P": _as_user_array(system_params["P"]),
        "B": _as_user_array(system_params["B"]),
        "T": _as_user_array(system_params["T"]),
        "Nb": _as_user_array(system_params["Nb"]),
        "Nr": _as_user_array(system_params["Nr"]),
    }

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (name, values) in enumerate(params.items()):
        ax = axes[idx]
        ax.plot(users, values, marker="o", markersize=4)
        ax.set_title(f"{name} per user")
        ax.set_xlabel("User")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "user_config_summary.png"), dpi=250)
    plt.close(fig)


def plot_latency(result: dict[str, Any], figs_dir: str) -> None:
    initial = np.asarray(result["initial_latency"], dtype=float)
    final = np.asarray(result["final_latency"], dtype=float)
    users = np.arange(len(initial))

    plt.figure(figsize=(max(8, len(users) * 0.4), 5))
    width = 0.35
    plt.bar(users - width / 2, initial, width=width, label="Initial")
    plt.bar(users + width / 2, final, width=width, label="Final")
    plt.xlabel("User")
    plt.ylabel("Latency (s)")
    plt.title("Initial vs final latency")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "latency_comparison.png"), dpi=250)
    plt.close()


def _asynchronality_matrix(latencies: np.ndarray) -> np.ndarray:
    arr = np.asarray(latencies, dtype=float)
    return np.abs(arr[:, None] - arr[None, :])


def plot_asynchronality_comparison(result: dict[str, Any], figs_dir: str) -> None:
    initial = np.asarray(result.get("initial_latency", []), dtype=float)
    final = np.asarray(result.get("final_latency", []), dtype=float)
    if initial.size == 0 or final.size == 0:
        return

    init_mat = _asynchronality_matrix(initial)
    final_mat = _asynchronality_matrix(final)
    delta_mat = final_mat - init_mat
    K = int(len(initial))
    mask = ~np.eye(K, dtype=bool)
    init_vals = init_mat[mask] if K > 1 else np.asarray([], dtype=float)
    final_vals = final_mat[mask] if K > 1 else np.asarray([], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    def _show_async(ax: plt.Axes, mat: np.ndarray, title: str, center_zero: bool = False) -> None:
        display = np.array(mat, dtype=float, copy=True)
        np.fill_diagonal(display, np.nan)
        if center_zero:
            finite = display[np.isfinite(display)]
            if finite.size > 0:
                vmax = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), 1e-12)
                norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            else:
                norm = None
        else:
            norm = None
        masked = np.ma.masked_invalid(display)
        im = ax.imshow(masked, aspect="auto", interpolation="none", cmap="RdBu_r" if center_zero else "plasma", norm=norm)
        ax.set_title(title)
        ax.set_xlabel("User j")
        ax.set_ylabel("User i")
        ax.set_xticks(np.arange(K))
        ax.set_yticks(np.arange(K))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|Latency_i - Latency_j| (s)" if not center_zero else "After - Before (s)")

    _show_async(axes[0, 0], init_mat, "Initial asynchronality heatmap")
    _show_async(axes[0, 1], final_mat, "Final asynchronality heatmap")
    _show_async(axes[1, 0], delta_mat, "Asynchronality change heatmap", center_zero=True)

    ax = axes[1, 1]
    if init_vals.size > 0:
        bins = _finite_histogram_bins(init_vals, final_vals, count=40)
        ax.hist(init_vals, bins=bins, density=True, alpha=0.4, color="tab:red", label="Initial")
        ax.hist(final_vals, bins=bins, density=True, alpha=0.55, color="tab:blue", label="Final")
    ax.set_title("Pairwise asynchronality distribution")
    ax.set_xlabel("|Latency_i - Latency_j| (s)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "asynchronality_comparison.png"), dpi=250)
    plt.close(fig)


def plot_link_quality(result: dict[str, Any], figs_dir: str) -> None:
    users = np.arange(len(result["initial_snr_db"]))
    metrics = result.get("summary_metrics", {})
    initial_block_sinr = metrics.get("initial_sinr_db_per_user", result["initial_sinr_db"])
    final_block_sinr = metrics.get("final_sinr_db_per_user", result["final_sinr_db"])
    plt.figure(figsize=(max(8, len(users) * 0.4), 5))
    plt.plot(users, result["initial_snr_db"], label="Initial SNR", linestyle="--")
    plt.plot(users, result["final_snr_db"], label="Final SNR", linestyle=":")
    plt.plot(users, initial_block_sinr, label="Initial mean block SINR", linewidth=2)
    plt.plot(users, final_block_sinr, label="Final mean block SINR", linewidth=2)
    plt.xlabel("User")
    plt.ylabel("dB")
    plt.title("Downlink link quality")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "link_quality.png"), dpi=250)
    plt.close()


def plot_blocks(result: dict[str, Any], figs_dir: str) -> None:
    blocks_per_user = np.asarray(result["blocks_per_user"], dtype=int)
    users = np.arange(len(blocks_per_user))

    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(users) * 0.4), 8))
    axes[0].bar(users, blocks_per_user)
    axes[0].set_title("Blocks per user")
    axes[0].set_xlabel("User")
    axes[0].set_ylabel("Block count")
    axes[0].grid(True, axis="y", alpha=0.3)

    avg_n = np.asarray([np.mean(v) if len(v) > 0 else 0.0 for v in result["n_kl"]], dtype=float)
    axes[1].bar(users, avg_n)
    axes[1].set_title("Average chosen sub-block length per user")
    axes[1].set_xlabel("User")
    axes[1].set_ylabel("Average n_kl")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "block_plan_summary.png"), dpi=250)
    plt.close(fig)


def plot_rate_vs_blocklength(result: dict[str, Any], figs_dir: str) -> None:
    points = result.get("rate_points", [])
    if len(points) == 0:
        return

    users = sorted({int(p["user"]) for p in points})
    cmap = plt.get_cmap("tab10", max(len(users), 1))

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    for idx, user in enumerate(users):
        user_points = [p for p in points if int(p["user"]) == user]
        n_vals = np.asarray([p["n_kl"] for p in user_points], dtype=float)
        achieved = np.asarray([p["achieved_rate"] for p in user_points], dtype=float)
        required = np.asarray([p["required_rate"] for p in user_points], dtype=float)
        margin = np.asarray([p["rate_margin"] for p in user_points], dtype=float)
        blocks = np.asarray([p["block"] for p in user_points], dtype=int)
        color = cmap(idx)

        order = np.argsort(n_vals)
        axes[0].plot(n_vals[order], achieved[order], marker="o", color=color, label=f"User {user} achieved")
        axes[0].plot(n_vals[order], required[order], marker="x", linestyle="--", color=color, alpha=0.8, label=f"User {user} required")

        axes[1].plot(blocks, margin, marker="o", color=color, label=f"User {user}")

    axes[0].set_title("Rate vs blocklength")
    axes[0].set_xlabel("Chosen blocklength n_kl")
    axes[0].set_ylabel("Rate (bits/channel-use)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2, fontsize=8)

    axes[1].axhline(0.0, color="black", linewidth=1, linestyle=":")
    axes[1].set_title("Rate margin per scheduled block")
    axes[1].set_xlabel("Block index")
    axes[1].set_ylabel("R_fbl - B_kl / n_kl")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "rate_vs_blocklength.png"), dpi=250)
    plt.close(fig)


def plot_rate_violation_heatmap(result: dict[str, Any], figs_dir: str) -> None:
    mat = _build_rate_violation_matrix(result)
    if mat.size == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, mat.shape[1] * 0.7), max(4, mat.shape[0] * 0.8)))
    _imshow_with_labels(
        ax,
        mat,
        title="Rate violation heatmap (required - achieved)",
        cbar_label="Bits/channel-use",
        cmap="RdBu_r",
        center_zero=True,
    )
    ax.set_xlabel("Block index")
    ax.set_ylabel("User")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "rate_violation_heatmap.png"), dpi=250)
    plt.close(fig)


def plot_blocklength_feasibility_curves(system: DownlinkSystem, result: dict[str, Any], figs_dir: str) -> None:
    K = len(result.get("n_kl", []))
    if K == 0:
        return

    sim_params = result.get("sim_params", {})
    n_step = max(1, int(sim_params.get("n_kl_step", 1)))
    n_min_default = int(sim_params.get("n_kl_min", 1))
    fig, axes = plt.subplots(K, 1, figsize=(12, max(3.5 * K, 4)), squeeze=False)
    cmap = plt.get_cmap("tab20")

    for k in range(K):
        ax = axes[k, 0]
        n_blocks = len(result["n_kl"][k])
        subblock_handles: list[Line2D] = []
        for l in range(n_blocks):
            chosen_n = int(result["n_kl"][k][l])
            bits = int(result["B_kl"][k][l])
            n_min = min(n_min_default, int(system.T[k]))
            n_vals = np.arange(n_min, int(system.T[k]) + 1, n_step, dtype=int)
            if len(n_vals) == 0 or n_vals[-1] != int(system.T[k]):
                n_vals = np.unique(np.append(n_vals, int(system.T[k])))
            achieved = np.asarray([system.compute_block_rate(k, l, int(nv)) for nv in n_vals], dtype=float)
            required = bits / np.maximum(n_vals.astype(float), 1.0)
            color = cmap(l % cmap.N)
            label = f"Sub-block {l}"
            ax.plot(n_vals, achieved, color=color, linewidth=1.8, alpha=0.9)
            ax.plot(n_vals, required, color=color, linestyle="--", alpha=0.55)
            chosen_rate = float(system.compute_block_rate(k, l, chosen_n))
            ax.scatter([chosen_n], [chosen_rate], color=color, s=28, zorder=3)
            subblock_handles.append(Line2D([0], [0], color=color, linewidth=2.0, label=label))

        ax.set_title(f"User {k} blocklength feasibility curves")
        ax.set_xlabel("Blocklength n")
        ax.set_ylabel("Rate (bits/channel-use)")
        ax.grid(True, alpha=0.3)
        style_handles = [
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="-", label="Achieved R_fbl(n)"),
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="--", label="Required B_kl / n"),
            Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=6, label="Chosen n_kl"),
        ]
        if len(subblock_handles) > 0:
            legend_blocks = ax.legend(
                handles=subblock_handles,
                fontsize=8,
                ncol=min(4, max(1, len(subblock_handles))),
                loc="upper center",
                bbox_to_anchor=(0.5, -0.22),
                title="Sub-blocks",
            )
            ax.add_artist(legend_blocks)
        ax.legend(handles=style_handles, fontsize=8, loc="upper right", title="Curve meaning")

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "blocklength_feasibility_curves.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_downlink_rfbl_vs_n_axis(ax: plt.Axes, result: dict[str, Any], user_idx: int) -> None:
    user_points = [point for point in result.get("rate_points", []) if int(point.get("user", -1)) == int(user_idx)]
    if len(user_points) == 0:
        ax.set_title(f"User {user_idx}: Achieved vs required rate")
        ax.text(0.5, 0.5, "No payload rate points available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    n_vals = np.asarray([float(point.get("n_kl", np.nan)) for point in user_points], dtype=float)
    achieved = np.asarray([float(point.get("achieved_rate", np.nan)) for point in user_points], dtype=float)
    required = np.asarray([float(point.get("required_rate", np.nan)) for point in user_points], dtype=float)
    blocks = np.asarray([int(point.get("block", idx)) for idx, point in enumerate(user_points)], dtype=int)
    order = np.argsort(n_vals)

    ax.plot(n_vals[order], achieved[order], marker="o", color="tab:blue", label="Achieved R_fbl")
    ax.plot(n_vals[order], required[order], marker="x", linestyle="--", color="tab:green", label="Required R_fbl")
    for idx in order:
        ax.annotate(f"b{int(blocks[idx])}", (n_vals[idx], achieved[idx]), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_xlabel(r"$n_{k,\ell}$")
    ax.set_ylabel("Rate (bits/channel-use)")
    ax.set_title(f"User {user_idx}: Achieved vs required rate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.invert_xaxis()


def _downlink_epoch_rate_panel_data(result: dict[str, Any], user_idx: int) -> dict[str, Any] | None:
    training_history = result.get("precoder_net_training_history", {})
    if isinstance(training_history, dict):
        per_user_rate = training_history.get("per_user_rate", [])
        if user_idx < len(per_user_rate):
            user_rates = [float(v) for v in list(per_user_rate[user_idx]) if np.isfinite(float(v))]
            if len(user_rates) > 0:
                return {
                    "source_label": "Training R_fbl over epoch",
                    "x_label": "Training epoch",
                    "segments": [
                        {
                            "x": np.arange(1, len(user_rates) + 1, dtype=float),
                            "y": np.asarray(user_rates, dtype=float),
                            "label": "training",
                            "color_index": 0,
                        }
                    ],
                    "separators": [],
                }

    epoch_history = _epoch_history_rows(result)
    if len(epoch_history) == 0:
        return None

    user_block_rates: dict[int, list[tuple[int, float]]] = {}
    for row in epoch_history:
        block = int(row["block"])
        user_ids = row.get("user_ids", [])
        user_rates = row.get("user_rates", [])
        epoch = _row_epoch(row)
        for row_user_id, user_rate in zip(user_ids, user_rates):
            if int(row_user_id) != int(user_idx):
                continue
            user_block_rates.setdefault(block, []).append((epoch, float(user_rate)))

    if len(user_block_rates) == 0:
        return None

    segments = []
    separators = []
    cursor = 0.0
    for color_index, block in enumerate(sorted(user_block_rates.keys())):
        block_points = sorted(user_block_rates[block], key=lambda item: item[0])
        if len(block_points) == 0:
            continue
        x_vals = cursor + np.arange(1, len(block_points) + 1, dtype=float)
        y_vals = np.asarray([float(rate) for _, rate in block_points], dtype=float)
        segments.append(
            {
                "x": x_vals,
                "y": y_vals,
                "label": f"block {int(block)}",
                "color_index": int(color_index),
            }
        )
        cursor = float(x_vals[-1]) + 1.5
        separators.append(float(x_vals[-1]) + 0.5)

    return {
        "source_label": "Optimization R_fbl over epoch",
        "x_label": "Epoch within block",
        "segments": segments,
        "separators": separators[:-1],
    }


def _plot_downlink_epoch_rate_axis(ax: plt.Axes, result: dict[str, Any], user_idx: int) -> None:
    panel_data = _downlink_epoch_rate_panel_data(result, user_idx)
    if panel_data is None:
        ax.set_title(f"User {user_idx}: R_fbl over epoch")
        ax.text(0.5, 0.5, "No epoch-rate history available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    cmap = plt.get_cmap("tab20")
    shown_labels = set()
    for segment in panel_data["segments"]:
        label = segment["label"] if segment["label"] not in shown_labels else None
        if label is not None:
            shown_labels.add(label)
        ax.plot(
            segment["x"],
            segment["y"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            alpha=0.9,
            color=cmap(int(segment["color_index"]) % cmap.N),
            label=label,
        )
    for separator_x in panel_data.get("separators", []):
        ax.axvline(separator_x, color="black", linestyle=":", linewidth=0.8, alpha=0.3)

    ax.set_title(f"User {user_idx}: {panel_data['source_label']}")
    ax.set_xlabel(panel_data["x_label"])
    ax.set_ylabel("Achieved R_fbl")
    ax.grid(True, alpha=0.3)
    if len(shown_labels) > 0 and len(shown_labels) <= 8:
        ax.legend(fontsize=8, loc="best")


def plot_payload_rfbl_vs_n_with_epoch(result: dict[str, Any], figs_dir: str) -> None:
    K = len(result.get("n_kl", []))
    if K == 0:
        return

    save_dir = os.path.join(figs_dir, "F_vs_n")
    os.makedirs(save_dir, exist_ok=True)

    for user_idx in range(K):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), squeeze=False)
        _plot_downlink_rfbl_vs_n_axis(axes[0, 0], result, user_idx)
        _plot_downlink_epoch_rate_axis(axes[0, 1], result, user_idx)
        fig.tight_layout()
        fig.savefig(
            os.path.join(save_dir, f"user{user_idx}_Rfbl_vs_n.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    fig, axes = plt.subplots(K, 2, figsize=(14, max(4 * K, 5)), squeeze=False)
    for user_idx in range(K):
        _plot_downlink_rfbl_vs_n_axis(axes[user_idx, 0], result, user_idx)
        _plot_downlink_epoch_rate_axis(axes[user_idx, 1], result, user_idx)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "all_users_Rfbl_vs_n_with_epoch.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_optimization_history(result: dict[str, Any], figs_dir: str) -> None:
    epoch_history = _epoch_history_rows(result)
    outer_history = result.get("outer_history", [])
    if len(epoch_history) == 0 and len(outer_history) == 0:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    if len(epoch_history) > 0:
        epoch_positions, segments = _build_segmented_epoch_positions(epoch_history, group_key="block")
        sum_rate = np.asarray([row["sum_rate"] for row in epoch_history], dtype=float)
        max_delta = np.asarray([row["max_precoder_delta"] for row in epoch_history], dtype=float)

        axes[0].plot(epoch_positions, sum_rate, marker="o", markersize=3)
        _draw_epoch_segment_guides(axes[0], segments, label_prefix="Block", show_segment_labels=True)
        tick_positions, tick_labels = _segmented_epoch_ticks(segments)
        axes[0].set_xticks(tick_positions)
        axes[0].set_xticklabels(tick_labels)
        axes[0].set_title("Sum-rate during precoder optimization")
        axes[0].set_xlabel("Epoch within block (restarts at each block)")
        axes[0].set_ylabel("Sum R_fbl at n=T")
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(epoch_positions, np.maximum(max_delta, 1e-12), marker="o", markersize=3)
        _draw_epoch_segment_guides(axes[1], segments, label_prefix="Block", show_segment_labels=False)
        axes[1].set_title("Maximum relative precoder change")
        axes[1].set_xticks(tick_positions)
        axes[1].set_xticklabels(tick_labels)
        axes[1].set_xlabel("Epoch within block (restarts at each block)")
        axes[1].set_ylabel("Max delta")
        axes[1].grid(True, alpha=0.3)

    if len(outer_history) > 0:
        block_ids = np.asarray([row["block"] for row in outer_history], dtype=int)
        allocated_bits = np.asarray([row["allocated_bits"] for row in outer_history], dtype=float)
        remaining_bits = np.asarray([row.get("remaining_bits", 0.0) for row in outer_history], dtype=float)

        axes[2].bar(block_ids - 0.2, allocated_bits, width=0.4, label="Served bits")
        axes[2].plot(block_ids + 0.2, remaining_bits, marker="o", label="Remaining bits")
        axes[2].set_title("Per-block bit allocation")
        axes[2].set_xlabel("Block index")
        axes[2].set_ylabel("Bits")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "optimization_history.png"), dpi=250)
    plt.close(fig)


def plot_interference_heatmaps(system: DownlinkSystem, figs_dir: str) -> None:
    diag = _collect_final_interference_diagnostics(system)
    avg_power_db = _safe_db(diag["avg_pairwise_power"])
    avg_inr_db = np.asarray(diag["avg_pairwise_inr_db"], dtype=float)
    share_pct = 100.0 * np.asarray(diag["avg_pairwise_share"], dtype=float)
    worst_block = int(diag["worst_block"])
    worst_block_mat = None
    if worst_block >= 0:
        worst_block_mat = _safe_db(np.asarray(diag["pairwise_block"][worst_block], dtype=float))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    _imshow_with_labels(axes[0, 0], avg_power_db, "Average pairwise interference power", "dB")
    _imshow_with_labels(axes[0, 1], avg_inr_db, "Average interference-to-noise ratio", "INR (dB)")
    _imshow_with_labels(axes[1, 0], share_pct, "Average interference share", "% of victim interference")
    if worst_block_mat is not None:
        _imshow_with_labels(
            axes[1, 1],
            worst_block_mat,
            f"Worst block interference map (block {worst_block})",
            "dB",
        )
    else:
        axes[1, 1].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "interference_heatmaps.png"), dpi=250)
    plt.close(fig)


def plot_interference_before_after_heatmaps(result: dict[str, Any], figs_dir: str) -> None:
    os.makedirs(figs_dir, exist_ok=True)
    initial_diag = _load_interference_diag(result.get("initial_interference_diag"))
    final_diag = _load_interference_diag(result.get("final_interference_diag"))
    if initial_diag is None or final_diag is None:
        return

    initial_bits = result.get("initial_plan", {}).get("B_kl", [])
    final_bits = result.get("B_kl", [])
    initial_power_db = _mask_inactive_user_pairs(
        _safe_db(initial_diag["avg_pairwise_power"]), initial_bits
    )
    final_power_db = _mask_inactive_user_pairs(
        _safe_db(final_diag["avg_pairwise_power"]), final_bits
    )
    delta_power_db = final_power_db - initial_power_db
    initial_inr_db = _mask_inactive_user_pairs(
        np.asarray(initial_diag["avg_pairwise_inr_db"], dtype=float), initial_bits
    )
    final_inr_db = _mask_inactive_user_pairs(
        np.asarray(final_diag["avg_pairwise_inr_db"], dtype=float), final_bits
    )
    delta_inr_db = final_inr_db - initial_inr_db
    power_vmin, power_vmax = _combined_finite_limits(initial_power_db, final_power_db)
    inr_vmin, inr_vmax = _combined_finite_limits(initial_inr_db, final_inr_db)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    _imshow_with_labels(
        axes[0, 0],
        initial_power_db,
        "Avg interference power before optimization",
        "dB",
        vmin=power_vmin,
        vmax=power_vmax,
    )
    _imshow_with_labels(
        axes[0, 1],
        final_power_db,
        "Avg interference power after optimization",
        "dB",
        vmin=power_vmin,
        vmax=power_vmax,
    )
    _imshow_with_labels(
        axes[0, 2],
        delta_power_db,
        "Interference power change (after - before)",
        "dB",
        cmap="RdBu_r",
        center_zero=True,
    )
    _imshow_with_labels(
        axes[1, 0],
        initial_inr_db,
        "Avg INR before optimization",
        "INR (dB)",
        vmin=inr_vmin,
        vmax=inr_vmax,
    )
    _imshow_with_labels(
        axes[1, 1],
        final_inr_db,
        "Avg INR after optimization",
        "INR (dB)",
        vmin=inr_vmin,
        vmax=inr_vmax,
    )
    _imshow_with_labels(
        axes[1, 2],
        delta_inr_db,
        "INR change (after - before)",
        "dB",
        cmap="RdBu_r",
        center_zero=True,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "interference_before_after_heatmaps.png"), dpi=250)
    plt.close(fig)


def plot_per_user_interference_profiles(system: DownlinkSystem, figs_dir: str) -> None:
    diag = _collect_final_interference_diagnostics(system)
    signal_db = _safe_db(diag["signal"])
    interference_db = _safe_db(diag["total_interference"])
    noise_db = _safe_db(diag["noise"])
    sinr_db = np.asarray(diag["sinr_db"], dtype=float)
    pairwise_block = np.asarray(diag["pairwise_block"], dtype=float)
    K = int(system.K)

    fig, axes = plt.subplots(K, 2, figsize=(15, max(4 * K, 5)), squeeze=False)
    for k in range(K):
        blocks = np.arange(len(system.n_kl[k]))
        ax_left = axes[k, 0]
        ax_left.plot(blocks, signal_db[k, : len(blocks)], "o-", label="Signal", color="tab:blue")
        ax_left.plot(blocks, interference_db[k, : len(blocks)], "o-", label="Total interference", color="tab:red")
        ax_left.plot(blocks, noise_db[k, : len(blocks)], "o--", label="Noise", color="tab:gray")
        ax_left.set_title(f"User {k} signal/interference/noise profile")
        ax_left.set_xlabel("Block index")
        ax_left.set_ylabel("Power (dB)")
        ax_left.grid(True, alpha=0.3)

        ax_left_r = ax_left.twinx()
        ax_left_r.plot(blocks, sinr_db[k, : len(blocks)], "s-", color="tab:green", label="SINR")
        ax_left_r.set_ylabel("SINR (dB)")

        handles_l, labels_l = ax_left.get_legend_handles_labels()
        handles_r, labels_r = ax_left_r.get_legend_handles_labels()
        _add_legend_if_present(
            ax_left,
            handles=handles_l + handles_r,
            labels=labels_l + labels_r,
            fontsize=8,
            loc="best",
        )

        ax_right = axes[k, 1]
        contrib = pairwise_block[: len(blocks), k, :]
        bottom = np.zeros(len(blocks), dtype=float)
        for j in range(K):
            if j == k:
                continue
            vals = np.nan_to_num(contrib[:, j], nan=0.0)
            if np.allclose(vals, 0.0):
                continue
            ax_right.bar(blocks, vals, bottom=bottom, alpha=0.75, label=f"Interferer {j}")
            bottom += vals
        ax_right.set_title(f"User {k} interference contributors")
        ax_right.set_xlabel("Block index")
        ax_right.set_ylabel("Interference power")
        ax_right.grid(True, axis="y", alpha=0.3)
        if K <= 6:
            handles_right, labels_right = ax_right.get_legend_handles_labels()
            if len(handles_right) > 0 and any(str(label).strip() for label in labels_right):
                ax_right.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "per_user_interference_profiles.png"), dpi=250)
    plt.close(fig)


def plot_per_user_interference_before_after(result: dict[str, Any], figs_dir: str) -> None:
    initial_diag = _load_interference_diag(result.get("initial_interference_diag"))
    final_diag = _load_interference_diag(result.get("final_interference_diag"))
    if initial_diag is None or final_diag is None:
        return

    K = max(
        len(initial_diag.get("blocks_per_user", [])),
        len(final_diag.get("blocks_per_user", [])),
        len(result.get("n_kl", [])),
    )
    if K == 0:
        return

    initial_interf_db = _safe_db(initial_diag["total_interference"])
    final_interf_db = _safe_db(final_diag["total_interference"])
    initial_sinr_db = np.asarray(initial_diag["sinr_db"], dtype=float)
    final_sinr_db = np.asarray(final_diag["sinr_db"], dtype=float)
    initial_signal_db = _safe_db(initial_diag["signal"])
    final_signal_db = _safe_db(final_diag["signal"])

    fig, axes = plt.subplots(K, 2, figsize=(16, max(4 * K, 5)), squeeze=False)
    for k in range(K):
        init_blocks = np.arange(int(initial_diag["blocks_per_user"][k])) if k < len(initial_diag["blocks_per_user"]) else np.asarray([], dtype=int)
        final_blocks = np.arange(int(final_diag["blocks_per_user"][k])) if k < len(final_diag["blocks_per_user"]) else np.asarray([], dtype=int)

        ax_left = axes[k, 0]
        if len(init_blocks) > 0:
            ax_left.plot(
                init_blocks,
                initial_interf_db[k, : len(init_blocks)],
                "o--",
                color="tab:red",
                alpha=0.8,
                label="Initial interference",
            )
            ax_left.plot(
                init_blocks,
                initial_signal_db[k, : len(init_blocks)],
                "o--",
                color="tab:blue",
                alpha=0.5,
                label="Initial signal",
            )
        if len(final_blocks) > 0:
            ax_left.plot(
                final_blocks,
                final_interf_db[k, : len(final_blocks)],
                "o-",
                color="tab:red",
                label="Final interference",
            )
            ax_left.plot(
                final_blocks,
                final_signal_db[k, : len(final_blocks)],
                "o-",
                color="tab:blue",
                alpha=0.7,
                label="Final signal",
            )
        ax_left.set_title(f"User {k} interference before vs after")
        ax_left.set_xlabel("Block index")
        ax_left.set_ylabel("Power (dB)")
        ax_left.grid(True, alpha=0.3)
        _add_legend_if_present(ax_left, fontsize=8, loc="best")

        ax_right = axes[k, 1]
        if len(init_blocks) > 0:
            ax_right.plot(
                init_blocks,
                initial_sinr_db[k, : len(init_blocks)],
                "s--",
                color="tab:green",
                alpha=0.8,
                label="Initial SINR",
            )
        if len(final_blocks) > 0:
            ax_right.plot(
                final_blocks,
                final_sinr_db[k, : len(final_blocks)],
                "s-",
                color="tab:green",
                label="Final SINR",
            )
        ax_right.set_title(f"User {k} SINR before vs after")
        ax_right.set_xlabel("Block index")
        ax_right.set_ylabel("SINR (dB)")
        ax_right.grid(True, alpha=0.3)
        _add_legend_if_present(ax_right, fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "per_user_interference_before_after.png"), dpi=250)
    plt.close(fig)


def plot_per_user_schedule_details(result: dict[str, Any], figs_dir: str) -> None:
    K = len(result.get("n_kl", []))
    if K == 0:
        return

    initial_plan = result.get("initial_plan", {})
    initial_n = initial_plan.get("n_kl", [[] for _ in range(K)])
    initial_b = initial_plan.get("B_kl", [[] for _ in range(K)])
    final_latency = np.asarray(result.get("final_latency", [0.0] * K), dtype=float)
    initial_latency = np.asarray(result.get("initial_latency", [0.0] * K), dtype=float)
    rate_points = list(result.get("rate_points", []))
    per_user_points: dict[int, list[dict[str, Any]]] = {int(k): [] for k in range(K)}
    for point in rate_points:
        user_id = int(point.get("user", -1))
        if 0 <= user_id < K:
            per_user_points[user_id].append(dict(point))
    for user_id in range(K):
        per_user_points[user_id].sort(key=lambda point: int(point.get("block", 0)))

    fig, axes = plt.subplots(K, 2, figsize=(14, max(4 * K, 6)), squeeze=False)
    for k in range(K):
        user_points = per_user_points.get(int(k), [])
        if len(user_points) > 0:
            blocks = np.asarray([int(point.get("block", idx)) for idx, point in enumerate(user_points)], dtype=int)
            bits = np.asarray([float(point.get("B_kl", 0.0)) for point in user_points], dtype=float)
            n_vals = np.asarray([float(point.get("n_kl", 0.0)) for point in user_points], dtype=float)
            rates = np.asarray([float(point.get("achieved_rate", 0.0)) for point in user_points], dtype=float)
            required = np.asarray(
                [
                    float(point.get("required_rate", 0.0))
                    for point in user_points
                ],
                dtype=float,
            )
            margins = np.asarray(
                [
                    float(point.get("rate_margin", rates[idx] - required[idx]))
                    for idx, point in enumerate(user_points)
                ],
                dtype=float,
            )
        else:
            blocks = np.arange(len(result["n_kl"][k]))
            bits = np.asarray(result["B_kl"][k], dtype=float)
            n_vals = np.asarray(result["n_kl"][k], dtype=float)
            rates = np.asarray(result["R_fbl"][k], dtype=float)
            required = bits / np.maximum(n_vals, 1.0)
            margins = rates - required

        init_blocks = np.arange(len(initial_n[k]))
        init_bits = np.asarray(initial_b[k], dtype=float) if k < len(initial_b) else np.asarray([], dtype=float)
        init_n_vals = np.asarray(initial_n[k], dtype=float) if k < len(initial_n) else np.asarray([], dtype=float)

        ax_left = axes[k, 0]
        ax_left.set_title(
            f"User {k} schedule | init={initial_latency[k]:.4e}s final={final_latency[k]:.4e}s"
        )
        if len(init_blocks) > 0:
            ax_left.bar(init_blocks - 0.18, init_bits, width=0.36, alpha=0.35, label="Initial bits")
        if len(blocks) > 0:
            ax_left.bar(blocks + 0.18, bits, width=0.36, alpha=0.8, label="Optimized bits")
        ax_left.set_xlabel("Block index")
        ax_left.set_ylabel("Bits")
        ax_left.grid(True, axis="y", alpha=0.3)

        ax_left_r = ax_left.twinx()
        if len(init_blocks) > 0:
            ax_left_r.plot(init_blocks, init_n_vals, "o--", color="tab:orange", alpha=0.6, label="Initial n_kl")
        if len(blocks) > 0:
            ax_left_r.plot(blocks, n_vals, "o-", color="tab:red", label="Optimized n_kl")
        ax_left_r.set_ylabel("Blocklength n_kl")

        handles_l, labels_l = ax_left.get_legend_handles_labels()
        handles_r, labels_r = ax_left_r.get_legend_handles_labels()
        _add_legend_if_present(
            ax_left,
            handles=handles_l + handles_r,
            labels=labels_l + labels_r,
            fontsize=8,
            loc="upper right",
        )

        ax_right = axes[k, 1]
        if len(blocks) > 0:
            ax_right.plot(blocks, rates, "o-", label="Achieved R_fbl", color="tab:blue")
            ax_right.plot(blocks, required, "x--", label="Required B_kl / n_kl", color="tab:green")
            ax_right.bar(blocks, margins, alpha=0.25, color="tab:purple", label="Rate margin")
        ax_right.axhline(0.0, color="black", linewidth=1, linestyle=":")
        ax_right.set_title(f"User {k} per-block rate results")
        ax_right.set_xlabel("Block index")
        ax_right.set_ylabel("Rate (bits/channel-use)")
        ax_right.grid(True, alpha=0.3)
        _add_legend_if_present(ax_right, fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "per_user_schedule_details.png"), dpi=250)
    plt.close(fig)


def plot_per_user_convergence(result: dict[str, Any], figs_dir: str) -> None:
    epoch_history = _epoch_history_rows(result)
    K = len(result.get("n_kl", []))
    if len(epoch_history) == 0 or K == 0:
        return

    user_block_rates: list[dict[int, list[tuple[int, float]]]] = [dict() for _ in range(K)]
    user_block_sinr: list[dict[int, list[tuple[int, float]]]] = [dict() for _ in range(K)]
    user_block_interference: list[dict[int, list[tuple[int, float]]]] = [dict() for _ in range(K)]
    max_epochs = 0
    for row in epoch_history:
        block = int(row["block"])
        user_ids = row.get("user_ids", [])
        user_rates = row.get("user_rates", [])
        user_sinr_db = row.get("user_sinr_db", [])
        user_interference_db = row.get("user_interference_db", [])
        epoch = _row_epoch(row)
        max_epochs = max(max_epochs, epoch)
        for user_id, user_rate, user_sinr, user_interf in zip(user_ids, user_rates, user_sinr_db, user_interference_db):
            per_block = user_block_rates[int(user_id)]
            per_block.setdefault(block, []).append((epoch, float(user_rate)))
            per_block_sinr = user_block_sinr[int(user_id)]
            per_block_sinr.setdefault(block, []).append((epoch, float(user_sinr)))
            per_block_interf = user_block_interference[int(user_id)]
            per_block_interf.setdefault(block, []).append((epoch, float(user_interf)))

    fig, axes = plt.subplots(K, 3, figsize=(22, max(3.2 * K, 4)), squeeze=False)
    cmap_rate = plt.get_cmap("viridis")
    cmap_sinr = plt.get_cmap("magma")
    cmap_interf = plt.get_cmap("cividis")
    for k in range(K):
        block_map_rate = user_block_rates[k]
        block_map_sinr = user_block_sinr[k]
        block_map_interf = user_block_interference[k]
        if len(block_map_rate) == 0:
            axes[k, 0].set_visible(False)
            axes[k, 1].set_visible(False)
            axes[k, 2].set_visible(False)
            continue

        block_ids = sorted(block_map_rate.keys())
        heat_rate = np.full((max_epochs, len(block_ids)), np.nan, dtype=float)
        heat_sinr = np.full((max_epochs, len(block_ids)), np.nan, dtype=float)
        heat_interf = np.full((max_epochs, len(block_ids)), np.nan, dtype=float)
        for col, block_id in enumerate(block_ids):
            for epoch, rate in block_map_rate[block_id]:
                heat_rate[epoch - 1, col] = rate
            for epoch, sinr in block_map_sinr[block_id]:
                heat_sinr[epoch - 1, col] = sinr
            for epoch, interf in block_map_interf[block_id]:
                heat_interf[epoch - 1, col] = interf

        ax_rate = axes[k, 0]
        ax_sinr = axes[k, 1]
        ax_interf = axes[k, 2]
        masked_rate = np.ma.masked_invalid(heat_rate)
        masked_sinr = np.ma.masked_invalid(heat_sinr)
        masked_interf = np.ma.masked_invalid(heat_interf)
        im_rate = ax_rate.imshow(masked_rate, aspect="auto", origin="lower", interpolation="none", cmap=cmap_rate)
        im_sinr = ax_sinr.imshow(masked_sinr, aspect="auto", origin="lower", interpolation="none", cmap=cmap_sinr)
        im_interf = ax_interf.imshow(masked_interf, aspect="auto", origin="lower", interpolation="none", cmap=cmap_interf)

        for ax in (ax_rate, ax_sinr, ax_interf):
            ax.set_xlabel("Block index")
            ax.set_ylabel("Epoch")
            ax.set_xticks(np.arange(len(block_ids)))
            ax.set_xticklabels(block_ids)

        ax_rate.set_title(f"User {k} rate convergence by block")
        ax_sinr.set_title(f"User {k} SINR convergence by block")
        ax_interf.set_title(f"User {k} interference convergence by block")
        fig.colorbar(im_rate, ax=ax_rate, fraction=0.02, pad=0.02, label="R_fbl at n=T")
        fig.colorbar(im_sinr, ax=ax_sinr, fraction=0.02, pad=0.02, label="SINR (dB)")
        fig.colorbar(im_interf, ax=ax_interf, fraction=0.02, pad=0.02, label="Interference power (dB)")

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "per_user_convergence.png"), dpi=250)
    plt.close(fig)
