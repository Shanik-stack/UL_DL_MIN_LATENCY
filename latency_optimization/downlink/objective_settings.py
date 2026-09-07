"""Names and validation for the retained downlink objective."""

from __future__ import annotations

from typing import Any

from latency_optimization.core.validation import require_choice


INVERSE_CNR_WEIGHTED_SUM_RATE = "inverse_cnr_weighted_sum_rate"
INVERSE_CNR_WEIGHT_STRATEGY = "inverse_cnr"


def validate_objective_mode(objective_mode: str) -> str:
    return require_choice(objective_mode, {INVERSE_CNR_WEIGHTED_SUM_RATE}, "convergence_block_objective_mode")


def validate_convergence_objective_mode(simulation: dict[str, Any]) -> str:
    return validate_objective_mode(simulation["convergence_block_objective_mode"])


def validate_convergence_priority_weight_strategy(simulation: dict[str, Any]) -> str:
    return require_choice(
        simulation["convergence_priority_weight_strategy"],
        {INVERSE_CNR_WEIGHT_STRATEGY},
        "convergence_priority_weight_strategy",
    )


def objective_display_name(objective_mode: str, configured_weight_strategy: str | None = None) -> str:
    validate_objective_mode(objective_mode)
    if configured_weight_strategy is not None:
        require_choice(
            configured_weight_strategy,
            {INVERSE_CNR_WEIGHT_STRATEGY},
            "convergence_priority_weight_strategy",
        )
    return INVERSE_CNR_WEIGHTED_SUM_RATE


def get_convergence_objective_name(simulation: dict[str, Any]) -> str:
    validate_convergence_objective_mode(simulation)
    validate_convergence_priority_weight_strategy(simulation)
    return INVERSE_CNR_WEIGHTED_SUM_RATE


def objective_weight_strategy_name(objective_mode: str, configured_weight_strategy: str) -> str:
    objective_display_name(objective_mode, configured_weight_strategy)
    return INVERSE_CNR_WEIGHT_STRATEGY


__all__ = [
    "INVERSE_CNR_WEIGHTED_SUM_RATE",
    "INVERSE_CNR_WEIGHT_STRATEGY",
    "get_convergence_objective_name",
    "objective_display_name",
    "objective_weight_strategy_name",
    "validate_convergence_objective_mode",
    "validate_convergence_priority_weight_strategy",
    "validate_objective_mode",
]
