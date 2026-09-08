"""Calculate link-specific metrics shared by downlink experiment methods."""

from __future__ import annotations

import numpy as np

from latency_optimization.experiments.scenarios import STREAMING_MODE
from latency_optimization.results.metrics import (
    mean_for_active_users as _mean_for_served_users,
    pairwise_latency_differences as _pairwise_latency_diffs,
    reference_latency_metrics as _compute_reference_latency_metrics,
)


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


def compute_downlink_summary_metrics(result: dict) -> dict:
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


__all__ = ["compute_downlink_summary_metrics"]

