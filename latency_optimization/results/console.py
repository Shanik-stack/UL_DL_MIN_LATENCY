from __future__ import annotations

import math
from typing import Any, Sequence


def _format_scalar(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        abs_value = abs(value)
        if abs_value == 0.0:
            return "0.000000"
        if abs_value < 1e-3 or abs_value >= 1e4:
            return f"{value:.6e}"
        return f"{value:.6f}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    return str(value)


def format_log_line(prefix: str, /, **fields: Any) -> str:
    parts = [str(prefix)]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={_format_scalar(value)}")
    return " | ".join(parts)


def format_latency_log_line(prefix: str, latencies: Sequence[float], /, **fields: Any) -> str:
    values = [float(v) for v in latencies]
    augmented_fields = dict(fields)
    augmented_fields["users"] = len(values)
    augmented_fields["initial_total_latency"] = float(sum(values)) if values else 0.0
    augmented_fields["initial_avg_latency"] = (
        float(sum(values) / len(values)) if values else 0.0
    )
    augmented_fields["initial_min_latency"] = min(values) if values else 0.0
    augmented_fields["initial_max_latency"] = max(values) if values else 0.0
    return format_log_line(prefix, **augmented_fields)


def format_progress_log_line(
    prefix: str,
    /,
    *,
    phase: Any = None,
    method: Any = None,
    scope: Any = None,
    user: Any = None,
    block: Any = None,
    n_kl: Any = None,
    epoch: Any = None,
    active_users: Any = None,
    updated_users: Any = None,
    rollout_queries: Any = None,
    objective: Any = None,
    sum_rate: Any = None,
    avg_user_rate: Any = None,
    rate: Any = None,
    power: Any = None,
    r_p: Any = None,
    r_c: Any = None,
    r_s: Any = None,
    status: Any = None,
    **extra_fields: Any,
) -> str:
    ordered_fields: dict[str, Any] = {}
    for key, value in (
        ("phase", phase),
        ("method", method),
        ("scope", scope),
        ("user", user),
        ("block", block),
        ("n_kl", n_kl),
        ("epoch", epoch),
        ("active_users", active_users),
        ("updated_users", updated_users),
        ("rollout_queries", rollout_queries),
        ("objective", objective),
        ("sum_rate", sum_rate),
        ("avg_user_rate", avg_user_rate),
        ("rate", rate),
        ("power", power),
        ("r_p", r_p),
        ("r_c", r_c),
        ("r_s", r_s),
        ("status", status),
    ):
        if value is not None:
            ordered_fields[key] = value
    ordered_fields.update(extra_fields)
    return format_log_line(prefix, **ordered_fields)
