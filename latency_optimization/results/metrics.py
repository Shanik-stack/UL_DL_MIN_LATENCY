"""Link-independent latency and summary metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def pairwise_latency_differences(
    latencies: Sequence[float],
) -> tuple[list[list[float]], list[dict[str, float]], float]:
    values = [float(value) for value in latencies]
    matrix = [[abs(first - second) for second in values] for first in values]
    pairs = [
        {"user_i": first, "user_j": second, "abs_latency_diff": float(matrix[first][second])}
        for first in range(len(values))
        for second in range(first + 1, len(values))
    ]
    return matrix, pairs, float(sum(pair["abs_latency_diff"] for pair in pairs))


def mean_for_active_users(values: Sequence[float], activity: Sequence[int]) -> float | None:
    selected = [
        float(value)
        for value, count in zip(values, activity)
        if int(count) > 0 and np.isfinite(float(value))
    ]
    return float(np.mean(selected)) if selected else None


def format_optional_db(label: str, value: object) -> str:
    return f"{label}: n/a (no served blocks)" if value is None else f"{label}: {float(value):.4f}"


def build_schedule_reference(
    name: str,
    latency: Sequence[float],
    plan: dict[str, Any],
    *,
    snr_db: Sequence[float],
    sinr_db: Sequence[float],
) -> dict[str, Any]:
    """Build the common saved-result schema for a reference schedule."""
    return {
        "schedule_source": str(name),
        "completed": bool(plan.get("completed", True)),
        "failure_reason": str(plan.get("failure_reason", "")),
        "remaining_bits": [int(value) for value in plan.get("remaining_bits", [])],
        "latency": [float(value) for value in latency],
        "n_kl": [list(map(int, values)) for values in plan.get("n_kl", [])],
        "B_kl": [list(map(int, values)) for values in plan.get("B_kl", [])],
        "R_alloc": [list(map(float, values)) for values in plan.get("R_alloc", [])],
        "snr_db": [float(value) for value in snr_db],
        "sinr_db": [float(value) for value in sinr_db],
        "skipped_blocks_per_user": [
            int(value) for value in plan.get("skipped_blocks_per_user", [])
        ],
    }


def reference_latency_metrics(
    reference_latency: Sequence[float],
    final_latency: Sequence[float],
    *,
    reference_completed: bool = True,
) -> dict[str, Any]:
    initial = [float(value) for value in reference_latency]
    final = [float(value) for value in final_latency]
    if not reference_completed or not np.all(np.isfinite(initial)):
        return {
            "baseline_completed": False,
            "initial_latency": initial,
            "initial_total_latency": float("nan"),
            "initial_avg_latency": float("nan"),
            "latency_reduction_per_user_percent": [float("nan") for _ in final],
            "total_latency_reduction_percent": float("nan"),
            "initial_asynchronality_sum": float("nan"),
            "final_asynchronality_sum": float("nan"),
            "asynchronality_reduction_percent": float("nan"),
        }

    reductions = [
        100.0 * (before - after) / before if before > 0.0 else 0.0
        for before, after in zip(initial, final)
    ]
    initial_total = float(sum(initial))
    final_total = float(sum(final))
    total_reduction = 100.0 * (initial_total - final_total) / initial_total if initial_total > 0.0 else 0.0
    initial_async = pairwise_latency_differences(initial)[2]
    final_async = pairwise_latency_differences(final)[2]
    async_reduction = 100.0 * (initial_async - final_async) / initial_async if initial_async > 0.0 else 0.0
    return {
        "baseline_completed": True,
        "initial_latency": initial,
        "initial_total_latency": initial_total,
        "initial_avg_latency": initial_total / max(len(initial), 1),
        "latency_reduction_per_user_percent": [float(value) for value in reductions],
        "total_latency_reduction_percent": float(total_reduction),
        "initial_asynchronality_sum": float(initial_async),
        "final_asynchronality_sum": float(final_async),
        "asynchronality_reduction_percent": float(async_reduction),
    }


__all__ = [
    "build_schedule_reference",
    "format_optional_db",
    "mean_for_active_users",
    "pairwise_latency_differences",
    "reference_latency_metrics",
]
