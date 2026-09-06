from __future__ import annotations

import os
from time import perf_counter
from typing import Callable

import numpy as np

from latency_optimization.core.scenarios import STREAMING_MODE
from latency_optimization.experiments.cost import build_downlink_convergence_cost, format_experiment_cost_lines
from latency_optimization.experiments.determinism import configure_determinism
from latency_optimization.results.naming import (
    format_method_tag,
    format_objective_tag,
    format_scope_tag,
    format_update_mode_tag,
    join_tag_parts,
    make_method_result_tag,
)
from latency_optimization.results.persistence import (
    current_local_timestamp,
    save_json,
    save_text,
    write_result_manifest,
)

from .config import load_config
from .convergence.allocation import (
    optimize_downlink_transmission,
)
from .objective import (
    get_convergence_objective_name,
)
from .plotting import (
    initialize_output_dirs,
    plot_asynchronality_comparison,
    plot_blocklength_feasibility_curves,
    plot_blocks,
    plot_interference_before_after_heatmaps,
    plot_interference_heatmaps,
    plot_kkt_residual_history,
    plot_latency,
    plot_link_quality,
    plot_optimization_history,
    plot_payload_rfbl_vs_n_with_epoch,
    plot_per_user_convergence,
    plot_per_user_interference_before_after,
    plot_per_user_interference_profiles,
    plot_per_user_schedule_details,
    plot_rate_violation_heatmap,
    plot_user_config,
)
from .system import DownlinkSystem


OPTIMIZERS: dict[str, Callable] = {
    "convergence_per_epoch_baseline": optimize_downlink_transmission,
}


def build_result_tag(
    method_name: str,
    cfg_stem: str,
    seed: int,
    *,
    objective_mode: str | None = None,
    model_scope: str | None = None,
    solver_mode: str | None = None,
    cfg_hash: str | None = None,
) -> str:
    method_parts = [format_method_tag(method_name)]
    if objective_mode:
        method_parts.append(format_objective_tag(objective_mode))
    solver_tag = format_update_mode_tag(solver_mode) if solver_mode else ""
    if model_scope and solver_tag != "dir":
        method_parts.append(format_scope_tag(model_scope))
    if solver_mode:
        method_parts.append(solver_tag)
    method_tag = join_tag_parts(*method_parts)
    return make_method_result_tag(method_tag, cfg_stem, seed=seed, cfg_hash=cfg_hash)


def _pairwise_latency_diffs(latencies: list[float]) -> tuple[list[list[float]], list[dict[str, float]], float]:
    arr = [float(x) for x in latencies]
    K = len(arr)
    matrix = [[abs(arr[i] - arr[j]) for j in range(K)] for i in range(K)]
    pair_details: list[dict[str, float]] = []
    async_sum = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            diff = float(matrix[i][j])
            async_sum += diff
            pair_details.append({"user_i": int(i), "user_j": int(j), "abs_latency_diff": diff})
    return matrix, pair_details, float(async_sum)


def _mean_for_served_users(values: object, served_block_counts: list[int]) -> float | None:
    user_values = [float(value) for value in values]
    valid = [
        value
        for value, count in zip(user_values, served_block_counts)
        if int(count) > 0 and np.isfinite(value)
    ]
    return float(np.mean(valid)) if valid else None


def _format_optional_db(label: str, value: object) -> str:
    if value is None:
        return f"{label}: n/a (no served blocks)"
    return f"{label}: {float(value):.4f}"


def _metric_matrix(values: object) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(arr.shape[0], 1)
    return arr


def _positive_bits_mask(bits_by_user: object, K: int, max_blocks: int) -> np.ndarray:
    mask = np.zeros((K, max_blocks), dtype=bool)
    if not isinstance(bits_by_user, list):
        return mask
    for k in range(min(K, len(bits_by_user))):
        row = bits_by_user[k]
        if not isinstance(row, list):
            continue
        for l, bits in enumerate(row[:max_blocks]):
            mask[k, l] = float(bits) > 0.0
    return mask


def _mean_valid_rows(
    values: object,
    K: int,
    *,
    include_mask: np.ndarray | None = None,
) -> tuple[list[float], float, list[int], int, bool]:
    arr = _metric_matrix(values)
    if include_mask is not None:
        mask = np.asarray(include_mask, dtype=bool)
        if mask.ndim == 1:
            mask = mask.reshape(mask.shape[0], 1)
        if mask.shape != arr.shape:
            aligned = np.zeros_like(arr, dtype=bool)
            rows = min(mask.shape[0], arr.shape[0])
            cols = min(mask.shape[1], arr.shape[1])
            aligned[:rows, :cols] = mask[:rows, :cols]
            mask = aligned
    else:
        mask = np.ones_like(arr, dtype=bool)

    per_user = [0.0 for _ in range(K)]
    per_user_counts = [0 for _ in range(K)]
    global_values: list[float] = []
    has_any = False
    for k in range(K):
        row = arr[k] if k < arr.shape[0] else np.asarray([], dtype=float)
        row_mask = mask[k] if k < mask.shape[0] else np.asarray([], dtype=bool)
        valid = row[np.isfinite(row) & row_mask]
        if valid.size > 0:
            per_user[k] = float(np.mean(valid))
            per_user_counts[k] = int(valid.size)
            global_values.extend(valid.tolist())
            has_any = True

    global_mean = float(np.mean(global_values)) if global_values else 0.0
    return per_user, global_mean, per_user_counts, int(len(global_values)), has_any


def _compute_reference_latency_metrics(
    reference_latency: list[float],
    final_latency: list[float],
    *,
    reference_completed: bool = True,
) -> dict:
    if not reference_completed or not np.all(np.isfinite(reference_latency)):
        return {
            "baseline_completed": False,
            "initial_latency": [float(v) for v in reference_latency],
            "initial_total_latency": float("nan"),
            "initial_avg_latency": float("nan"),
            "latency_reduction_per_user_percent": [float("nan") for _ in final_latency],
            "total_latency_reduction_percent": float("nan"),
            "initial_asynchronality_sum": float("nan"),
            "final_asynchronality_sum": float("nan"),
            "asynchronality_reduction_percent": float("nan"),
        }
    per_user_reduction: list[float] = []
    for init_val, final_val in zip(reference_latency, final_latency):
        if init_val > 0:
            reduction = ((init_val - final_val) / init_val) * 100.0
        else:
            reduction = 0.0
        per_user_reduction.append(float(reduction))

    initial_total_latency = float(sum(reference_latency))
    final_total_latency = float(sum(final_latency))
    if initial_total_latency > 0:
        total_latency_reduction_percent = ((initial_total_latency - final_total_latency) / initial_total_latency) * 100.0
    else:
        total_latency_reduction_percent = 0.0

    _, _, initial_async_sum = _pairwise_latency_diffs(reference_latency)
    _, _, final_async_sum = _pairwise_latency_diffs(final_latency)
    if initial_async_sum > 0:
        async_reduction_percent = ((initial_async_sum - final_async_sum) / initial_async_sum) * 100.0
    else:
        async_reduction_percent = 0.0

    return {
        "baseline_completed": True,
        "initial_latency": [float(v) for v in reference_latency],
        "initial_total_latency": initial_total_latency,
        "initial_avg_latency": float(initial_total_latency / max(len(reference_latency), 1)),
        "latency_reduction_per_user_percent": per_user_reduction,
        "total_latency_reduction_percent": float(total_latency_reduction_percent),
        "initial_asynchronality_sum": float(initial_async_sum),
        "final_asynchronality_sum": float(final_async_sum),
        "asynchronality_reduction_percent": float(async_reduction_percent),
    }


def _compute_summary_metrics(result: dict) -> dict:
    initial_latency = [float(x) for x in result["initial_latency"]]
    final_latency = [float(x) for x in result["final_latency"]]
    K = len(final_latency)
    baseline_completed = bool(result.get("initial_baseline_completed", True))
    baseline_completed = baseline_completed and bool(np.all(np.isfinite(initial_latency)))
    final_total_latency = float(sum(final_latency))
    if baseline_completed:
        latency_reduction_per_user_percent = []
        for init_val, final_val in zip(initial_latency, final_latency):
            reduction = ((init_val - final_val) / init_val) * 100.0 if init_val > 0 else 0.0
            latency_reduction_per_user_percent.append(float(reduction))
        initial_total_latency = float(sum(initial_latency))
        initial_avg_latency = float(initial_total_latency / max(K, 1))
        total_latency_reduction_percent = (
            ((initial_total_latency - final_total_latency) / initial_total_latency) * 100.0
            if initial_total_latency > 0
            else 0.0
        )
        initial_async_matrix, initial_async_pairs, initial_async_sum = _pairwise_latency_diffs(initial_latency)
        final_async_matrix, final_async_pairs, final_async_sum = _pairwise_latency_diffs(final_latency)
        async_reduction_percent = (
            ((initial_async_sum - final_async_sum) / initial_async_sum) * 100.0
            if initial_async_sum > 0
            else 0.0
        )
    else:
        initial_total_latency = float("nan")
        initial_avg_latency = float("nan")
        latency_reduction_per_user_percent = [float("nan") for _ in final_latency]
        total_latency_reduction_percent = float("nan")
        initial_async_matrix, initial_async_pairs, initial_async_sum = _pairwise_latency_diffs(
            [float("nan") for _ in final_latency]
        )
        final_async_matrix, final_async_pairs, final_async_sum = _pairwise_latency_diffs(final_latency)
        async_reduction_percent = float("nan")

    skipped_blocks_per_user = [0 for _ in range(K)]
    for point in result.get("rate_points", []):
        if bool(point.get("skipped", False)):
            skipped_blocks_per_user[int(point["user"])] += 1

    n_totals = [int(sum(v)) for v in result.get("n_kl", [[] for _ in range(K)])]
    bits_totals = [int(sum(v)) for v in result.get("B_kl", [[] for _ in range(K)])]
    blocks_per_user = [int(v) for v in result.get("blocks_per_user", [0 for _ in range(K)])]
    final_sinr_db_raw = [float(x) for x in result.get("final_sinr_db", [0.0 for _ in range(K)])]
    initial_sinr_db_raw = [float(x) for x in result.get("initial_sinr_db", [0.0 for _ in range(K)])]
    initial_sinr_matrix = _metric_matrix(result.get("initial_interference_diag", {}).get("sinr_db", []))
    final_sinr_matrix = _metric_matrix(result.get("final_interference_diag", {}).get("sinr_db", []))
    initial_block_sinr_db_all, initial_avg_block_sinr_db_all, _, _, has_initial_block_sinr_all = _mean_valid_rows(
        initial_sinr_matrix,
        K,
    )
    final_block_sinr_db_all, final_avg_block_sinr_db_all, _, _, has_final_block_sinr_all = _mean_valid_rows(
        final_sinr_matrix,
        K,
    )
    initial_served_mask = _positive_bits_mask(
        result.get("initial_plan", {}).get("B_kl", [[] for _ in range(K)]),
        K,
        int(initial_sinr_matrix.shape[1]),
    )
    final_served_mask = _positive_bits_mask(
        result.get("B_kl", [[] for _ in range(K)]),
        K,
        int(final_sinr_matrix.shape[1]),
    )
    (
        initial_block_sinr_db,
        initial_avg_block_sinr_db,
        initial_served_block_counts,
        initial_total_served_blocks,
        has_initial_block_sinr,
    ) = _mean_valid_rows(
        initial_sinr_matrix,
        K,
        include_mask=initial_served_mask,
    )
    (
        final_block_sinr_db,
        final_avg_block_sinr_db,
        final_served_block_counts,
        final_total_served_blocks,
        has_final_block_sinr,
    ) = _mean_valid_rows(
        final_sinr_matrix,
        K,
        include_mask=final_served_mask,
    )
    if not has_initial_block_sinr:
        initial_block_sinr_db = list(initial_sinr_db_raw)
        initial_avg_block_sinr_db = float(sum(initial_sinr_db_raw) / max(len(initial_sinr_db_raw), 1))
        initial_served_block_counts = [0 for _ in range(K)]
        initial_total_served_blocks = 0
    if not has_final_block_sinr:
        final_block_sinr_db = list(final_sinr_db_raw)
        final_avg_block_sinr_db = float(sum(final_sinr_db_raw) / max(len(final_sinr_db_raw), 1))
        final_served_block_counts = [0 for _ in range(K)]
        final_total_served_blocks = 0
    if not has_initial_block_sinr_all:
        initial_block_sinr_db_all = list(initial_sinr_db_raw)
        initial_avg_block_sinr_db_all = float(sum(initial_sinr_db_raw) / max(len(initial_sinr_db_raw), 1))
    if not has_final_block_sinr_all:
        final_block_sinr_db_all = list(final_sinr_db_raw)
        final_avg_block_sinr_db_all = float(sum(final_sinr_db_raw) / max(len(final_sinr_db_raw), 1))
    for k in range(K):
        if int(initial_served_block_counts[k]) <= 0:
            initial_block_sinr_db[k] = float(initial_block_sinr_db_all[k])
        if int(final_served_block_counts[k]) <= 0:
            final_block_sinr_db[k] = float(final_block_sinr_db_all[k])
    scenario_mode = str(result.get("scenario_mode", ""))
    scenario_block_targets = np.asarray(result.get("scenario_block_targets", []), dtype=int)
    target_bits_per_user = [0 for _ in range(K)]
    unserved_bits_per_user = [0 for _ in range(K)]
    partially_served_blocks_per_user = [0 for _ in range(K)]
    zero_service_blocks_per_user = [0 for _ in range(K)]

    if scenario_mode == STREAMING_MODE and scenario_block_targets.ndim == 2 and scenario_block_targets.shape[0] == K:
        target_bits_per_user = list(map(int, scenario_block_targets.sum(axis=1, dtype=int)))
        served_by_user = [list(map(int, result.get("B_kl", [[] for _ in range(K)])[k])) for k in range(K)]
        for k in range(K):
            targets = scenario_block_targets[int(k)].tolist()
            served = served_by_user[int(k)] if int(k) < len(served_by_user) else []
            unserved_bits_per_user[int(k)] = int(sum(max(int(t) - int(s), 0) for t, s in zip(targets, served)))
            partially_served_blocks_per_user[int(k)] = int(
                sum(1 for t, s in zip(targets, served) if int(t) > 0 and 0 < int(s) < int(t))
            )
            zero_service_blocks_per_user[int(k)] = int(
                sum(1 for t, s in zip(targets, served) if int(t) > 0 and int(s) <= 0)
            )

    per_user_summary = []
    for k in range(K):
        target_bits = int(target_bits_per_user[k])
        user_served_bits = int(bits_totals[k])
        missed_bits = int(unserved_bits_per_user[k])
        successful_blocks = max(
            int(blocks_per_user[k])
            - int(partially_served_blocks_per_user[k])
            - int(zero_service_blocks_per_user[k]),
            0,
        )
        per_user_summary.append(
            {
                "user": int(k),
                "initial_latency": initial_latency[k],
                "final_latency": final_latency[k],
                "latency_reduction_percent": latency_reduction_per_user_percent[k],
                "initial_sinr_db": initial_block_sinr_db[k],
                "final_sinr_db": final_block_sinr_db[k],
                "initial_sinr_db_all_blocks": initial_block_sinr_db_all[k],
                "final_sinr_db_all_blocks": final_block_sinr_db_all[k],
                "initial_sinr_db_raw": initial_sinr_db_raw[k],
                "final_sinr_db_raw": final_sinr_db_raw[k],
                "initial_served_blocks": int(initial_served_block_counts[k]),
                "final_served_blocks": int(final_served_block_counts[k]),
                "blocks": blocks_per_user[k],
                "total_n": n_totals[k],
                "target_bits": target_bits,
                "served_bits": user_served_bits,
                "unserved_bits": int(unserved_bits_per_user[k]),
                "missed_bits": missed_bits,
                "delivery_ratio": (
                    float(user_served_bits) / float(target_bits) if target_bits > 0 else 1.0
                ),
                "successful_blocks": successful_blocks,
                "deadline_miss_fraction": (
                    float(int(partially_served_blocks_per_user[k]) + int(zero_service_blocks_per_user[k]))
                    / float(blocks_per_user[k])
                    if int(blocks_per_user[k]) > 0
                    else 0.0
                ),
                "average_selected_blocklength": (
                    float(n_totals[k]) / float(blocks_per_user[k])
                    if int(blocks_per_user[k]) > 0
                    else 0.0
                ),
                "partially_served_blocks": int(partially_served_blocks_per_user[k]),
                "zero_service_blocks": int(zero_service_blocks_per_user[k]),
                "skipped_blocks": int(skipped_blocks_per_user[k]),
            }
        )

    baseline_comparison_metrics: dict[str, dict] = {}
    for baseline_name, baseline_payload in result.get("baseline_references", {}).items():
        if not isinstance(baseline_payload, dict):
            continue
        ref_latency_obj = baseline_payload.get("latency", [])
        if not isinstance(ref_latency_obj, list):
            continue
        baseline_comparison_metrics[str(baseline_name)] = _compute_reference_latency_metrics(
            [float(v) for v in ref_latency_obj],
            final_latency,
            reference_completed=bool(baseline_payload.get("completed", True)),
        )

    return {
        "initial_total_latency": initial_total_latency,
        "final_total_latency": final_total_latency,
        "initial_avg_latency": initial_avg_latency,
        "final_avg_latency": float(final_total_latency / max(K, 1)),
        "initial_max_latency": float(max(initial_latency) if initial_latency else 0.0),
        "final_max_latency": float(max(final_latency) if final_latency else 0.0),
        "initial_min_latency": float(min(initial_latency) if initial_latency else 0.0),
        "final_min_latency": float(min(final_latency) if final_latency else 0.0),
        "latency_reduction_per_user_percent": latency_reduction_per_user_percent,
        "total_latency_reduction_percent": float(total_latency_reduction_percent),
        "initial_asynchronality_matrix": initial_async_matrix,
        "final_asynchronality_matrix": final_async_matrix,
        "initial_asynchronality_pairs": initial_async_pairs,
        "final_asynchronality_pairs": final_async_pairs,
        "initial_asynchronality_sum": float(initial_async_sum),
        "final_asynchronality_sum": float(final_async_sum),
        "asynchronality_reduction_percent": float(async_reduction_percent),
        "initial_baseline_completed": bool(baseline_completed),
        "initial_avg_sinr_db": (
            float(initial_avg_block_sinr_db) if int(initial_total_served_blocks) > 0 else None
        ),
        "final_avg_sinr_db": (
            float(final_avg_block_sinr_db) if int(final_total_served_blocks) > 0 else None
        ),
        "initial_avg_sinr_db_all_blocks": float(initial_avg_block_sinr_db_all),
        "final_avg_sinr_db_all_blocks": float(final_avg_block_sinr_db_all),
        "initial_avg_sinr_db_raw": float(sum(initial_sinr_db_raw) / max(len(initial_sinr_db_raw), 1)),
        "final_avg_sinr_db_raw": float(sum(final_sinr_db_raw) / max(len(final_sinr_db_raw), 1)),
        "initial_sinr_db_per_user": initial_block_sinr_db,
        "final_sinr_db_per_user": final_block_sinr_db,
        "initial_sinr_db_per_user_all_blocks": initial_block_sinr_db_all,
        "final_sinr_db_per_user_all_blocks": final_block_sinr_db_all,
        "initial_served_block_counts": initial_served_block_counts,
        "final_served_block_counts": final_served_block_counts,
        "initial_total_served_blocks": int(initial_total_served_blocks),
        "final_total_served_blocks": int(final_total_served_blocks),
        "initial_avg_snr_db": _mean_for_served_users(
            result.get("initial_snr_db", []), initial_served_block_counts
        ),
        "final_avg_snr_db": _mean_for_served_users(
            result.get("final_snr_db", []), final_served_block_counts
        ),
        "scenario_mode": scenario_mode,
        "target_bits_per_user": target_bits_per_user,
        "unserved_bits_per_user": unserved_bits_per_user,
        "missed_bits_per_user": unserved_bits_per_user,
        "total_delivery_ratio": (
            float(sum(bits_totals)) / float(sum(target_bits_per_user))
            if sum(target_bits_per_user) > 0
            else 1.0
        ),
        "partially_served_blocks_per_user": partially_served_blocks_per_user,
        "zero_service_blocks_per_user": zero_service_blocks_per_user,
        "skipped_blocks_per_user": skipped_blocks_per_user,
        "per_user_summary": per_user_summary,
        "baseline_comparison_metrics": baseline_comparison_metrics,
    }


def _sum_per_user_summary_field(metrics: dict[str, object], field: str) -> int:
    return int(sum(int(row.get(field, 0)) for row in metrics.get("per_user_summary", [])))


def _get_baseline_comparison_metric(metrics: dict[str, object], baseline_name: str) -> dict[str, object]:
    baseline_metrics = metrics.get("baseline_comparison_metrics", {})
    if not isinstance(baseline_metrics, dict):
        return {}
    value = baseline_metrics.get(baseline_name, {})
    return value if isinstance(value, dict) else {}


def _build_downlink_final_test_section_lines(result: dict[str, object]) -> list[str]:
    metrics = result["summary_metrics"]
    assert isinstance(metrics, dict)
    random_baseline_metrics = _get_baseline_comparison_metric(metrics, "random_precoder_baseline")
    naive_full_t_metrics = _get_baseline_comparison_metric(metrics, "naive_full_T_baseline")
    lines = [
        "Final test results",
        f"Initial total latency (random baseline): {metrics['initial_total_latency']:.6f}",
        f"Final total latency: {metrics['final_total_latency']:.6f}",
        (
            "Latency reduction vs random precoder baseline (%): "
            f"{random_baseline_metrics.get('total_latency_reduction_percent', metrics['total_latency_reduction_percent']):.4f}"
        ),
        f"Initial avg latency (random baseline): {metrics['initial_avg_latency']:.6f}",
        f"Final avg latency: {metrics['final_avg_latency']:.6f}",
        f"Initial asynchronality sum (random baseline): {metrics['initial_asynchronality_sum']:.6f}",
        f"Final asynchronality sum: {metrics['final_asynchronality_sum']:.6f}",
        f"Asynchronality reduction (%): {metrics['asynchronality_reduction_percent']:.4f}",
        _format_optional_db("Initial avg served-user SNR (dB)", metrics["initial_avg_snr_db"]),
        _format_optional_db("Final avg served-user SNR (dB)", metrics["final_avg_snr_db"]),
        _format_optional_db("Initial avg served-block SINR (dB)", metrics["initial_avg_sinr_db"]),
        _format_optional_db("Final avg served-block SINR (dB)", metrics["final_avg_sinr_db"]),
        f"Initial avg all-block SINR (dB): {metrics['initial_avg_sinr_db_all_blocks']:.4f}",
        f"Final avg all-block SINR (dB): {metrics['final_avg_sinr_db_all_blocks']:.4f}",
        f"Total served bits: {_sum_per_user_summary_field(metrics, 'served_bits')}",
        f"Total skipped blocks: {_sum_per_user_summary_field(metrics, 'skipped_blocks')}",
    ]
    if naive_full_t_metrics:
        lines.extend(
            [
                f"Initial total latency (naive full-T baseline): {naive_full_t_metrics.get('initial_total_latency', 0.0):.6f}",
                (
                    "Latency reduction vs naive full-T baseline (%): "
                    f"{naive_full_t_metrics.get('total_latency_reduction_percent', 0.0):.4f}"
                ),
            ]
        )
    if metrics.get("scenario_mode", "") == STREAMING_MODE:
        lines.extend(
            [
                f"Total target bits: {_sum_per_user_summary_field(metrics, 'target_bits')}",
                f"Total unserved bits: {_sum_per_user_summary_field(metrics, 'unserved_bits')}",
                f"Overall delivery ratio: {float(metrics.get('total_delivery_ratio', 0.0)):.6f}",
                f"Total partially served blocks: {_sum_per_user_summary_field(metrics, 'partially_served_blocks')}",
                f"Total zero-service blocks: {_sum_per_user_summary_field(metrics, 'zero_service_blocks')}",
            ]
        )
    return lines


def _build_downlink_per_user_test_lines(result: dict[str, object]) -> list[str]:
    metrics = result["summary_metrics"]
    assert isinstance(metrics, dict)
    naive_full_t_metrics = _get_baseline_comparison_metric(metrics, "naive_full_T_baseline")
    naive_per_user = naive_full_t_metrics.get("latency_reduction_per_user_percent", [])
    lines = ["Per-user final test details"]
    for row in metrics["per_user_summary"]:
        parts = [
            f"User {row['user']}",
            f"init_lat={row['initial_latency']:.6f}",
            f"final_lat={row['final_latency']:.6f}",
            f"lat_red_vs_random={row['latency_reduction_percent']:.4f}%",
            (
                f"init_served_block_sinr={row['initial_sinr_db']:.4f} dB"
                if int(row["initial_served_blocks"]) > 0
                else "init_served_block_sinr=n/a"
            ),
            (
                f"final_served_block_sinr={row['final_sinr_db']:.4f} dB"
                if int(row["final_served_blocks"]) > 0
                else "final_served_block_sinr=n/a"
            ),
            f"init_all_block_sinr={row['initial_sinr_db_all_blocks']:.4f} dB",
            f"final_all_block_sinr={row['final_sinr_db_all_blocks']:.4f} dB",
            f"served_blocks={row['final_served_blocks']}",
            f"blocks={row['blocks']}",
            f"total_n={row['total_n']}",
            f"served_bits={row['served_bits']}",
            f"skipped_blocks={row['skipped_blocks']}",
        ]
        if int(row["user"]) < len(naive_per_user):
            parts.append(f"lat_red_vs_fullT={float(naive_per_user[int(row['user'])]):.4f}%")
        if metrics.get("scenario_mode", "") == STREAMING_MODE:
            parts.extend(
                [
                    f"target_bits={row.get('target_bits', 0)}",
                    f"unserved_bits={row.get('unserved_bits', 0)}",
                    f"delivery_ratio={float(row.get('delivery_ratio', 0.0)):.6f}",
                    f"successful_blocks={row.get('successful_blocks', 0)}",
                    f"deadline_miss_fraction={float(row.get('deadline_miss_fraction', 0.0)):.6f}",
                    f"avg_n_kl={float(row.get('average_selected_blocklength', 0.0)):.4f}",
                    f"partial_blocks={row.get('partially_served_blocks', 0)}",
                    f"zero_service_blocks={row.get('zero_service_blocks', 0)}",
                ]
            )
        lines.append(" | ".join(parts))
    return lines


def run_downlink_experiment(
    method_name: str,
    cfg_name: str,
    seed: int,
    verbose: bool = True,
    *,
    output_root: str | None = None,
) -> dict:
    if method_name not in OPTIMIZERS:
        known = ", ".join(sorted(OPTIMIZERS))
        raise ValueError(f"Unknown method '{method_name}'. Expected one of: {known}")

    configure_determinism(seed)
    run_started_at_local = current_local_timestamp()
    system_params, sim_params, run_meta = load_config(cfg_name)
    objective_mode_tag = (
        get_convergence_objective_name(sim_params)
        if method_name == "convergence_per_epoch_baseline"
        else None
    )
    result_tag = build_result_tag(
        method_name,
        run_meta["cfg_stem"],
        seed,
        objective_mode=objective_mode_tag,
        model_scope=sim_params.get("downlink_precoder_net_scope"),
        solver_mode=sim_params.get("convergence_precoder_update_mode"),
        cfg_hash=run_meta.get("cfg_hash"),
    )
    if output_root is None:
        output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", result_tag)
    output_dirs = initialize_output_dirs(output_root)

    core_start = perf_counter()
    system = DownlinkSystem(system_params, seed=seed)
    result = OPTIMIZERS[method_name](system, sim_params, verbose=verbose)
    core_wall_time_seconds_total = perf_counter() - core_start
    result["cfg_path"] = run_meta["cfg_path"]
    result["seed"] = int(seed)
    result["system_params"] = system_params
    result["sim_params"] = sim_params
    result["summary_metrics"] = _compute_summary_metrics(result)
    result["experiment_cost"] = build_downlink_convergence_cost(
        system_params,
        sim_params,
        result,
        core_wall_time_seconds_total=core_wall_time_seconds_total,
    )
    result["run_started_at_local"] = str(run_started_at_local)
    result["run_completed_at_local"] = current_local_timestamp()

    plot_user_config(system_params, output_dirs["user_config"])
    plot_latency(result, output_dirs["latency_asynchronality"])
    plot_asynchronality_comparison(result, output_dirs["latency_asynchronality"])
    plot_link_quality(result, output_dirs["link_quality"])
    plot_blocks(result, output_dirs["schedule_details"])
    plot_rate_violation_heatmap(result, output_dirs["optimization_history"])
    plot_optimization_history(result, output_dirs["optimization_history"])
    plot_kkt_residual_history(result, output_dirs["optimization_history"])
    plot_per_user_schedule_details(result, output_dirs["schedule_details"])
    plot_per_user_convergence(result, output_dirs["optimization_history"])
    plot_blocklength_feasibility_curves(system, result, output_dirs["optimization_history"])
    plot_payload_rfbl_vs_n_with_epoch(result, output_dirs["optimization_history"])
    plot_interference_before_after_heatmaps(result, output_dirs["interference"])
    plot_per_user_interference_before_after(result, output_dirs["interference"])
    plot_interference_heatmaps(system, output_dirs["interference"])
    plot_per_user_interference_profiles(system, output_dirs["interference"])

    save_json(result, os.path.join(output_dirs["data"], "result.json"))

    metrics = result["summary_metrics"]
    lines = [
        "Downlink optimizer summary",
        "",
        "Setup",
        f"Method: {method_name}",
        f"Config: {run_meta['cfg_path']}",
        f"Seed: {seed}",
        f"Run started at: {result.get('run_started_at_local', 'unknown')}",
        f"Run completed at: {result.get('run_completed_at_local', 'unknown')}",
        f"Scenario: {result.get('scenario_mode', result.get('allocation_mode', 'unknown'))}",
        f"Objective mode: {result.get('objective_mode', 'unknown')}",
        f"Allocation mode: {result.get('allocation_mode', 'unknown')}",
        f"Weight strategy: {result.get('weight_strategy', 'n/a')}",
        f"Convergence precoder update mode: {result.get('convergence_precoder_update_mode', 'unknown')}",
        f"Downlink precoder-net scope: {result.get('downlink_precoder_net_scope', 'unknown')}",
        f"Precoder parameterization: {result.get('precoder_parameterization', 'unknown')}",
        f"Initial schedule source: {result.get('initial_schedule_source', 'random_precoder_baseline')}",
    ]
    lines.extend([""])
    lines.extend(_build_downlink_final_test_section_lines(result))
    lines.extend([""])
    lines.extend(_build_downlink_per_user_test_lines(result))

    if metrics["initial_asynchronality_pairs"]:
        lines.extend(["", "Per-pair asynchronality"])
        for init_pair, final_pair in zip(metrics["initial_asynchronality_pairs"], metrics["final_asynchronality_pairs"]):
            lines.append(
                " | ".join(
                    [
                        f"Users {init_pair['user_i']}-{init_pair['user_j']}",
                        f"initial_diff={init_pair['abs_latency_diff']:.6f}",
                        f"final_diff={final_pair['abs_latency_diff']:.6f}",
                    ]
                )
            )

    lines.extend(format_experiment_cost_lines(result.get("experiment_cost")))
    save_text(lines, os.path.join(output_dirs["data"], "summary.txt"))
    result_root = os.path.dirname(output_root) if os.path.basename(output_root) == "testing" else output_root
    write_result_manifest(
        result_root,
        setup={
            "link": "Downlink",
            "method": method_name,
            "scenario": result.get("scenario_mode", result.get("allocation_mode", "unknown")),
            "config_path": run_meta["cfg_path"],
            "config_hash": run_meta.get("cfg_hash"),
            "seed": int(seed),
            "run_started_at_local": result.get("run_started_at_local"),
            "run_completed_at_local": result.get("run_completed_at_local"),
        },
    )
    print(f"Saved downlink results to: {output_root}")
    return result
