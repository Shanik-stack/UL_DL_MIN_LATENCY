"""Names and validation for the retained uplink objective."""

from __future__ import annotations

from latency_optimization.experiments.config_validation import require_choice


UNWEIGHTED_SUM_RATE_OBJECTIVE = "unweighted_sum_rate"
RATE_BEAM_REWARD_MODE = "rate"


def validate_uplink_objective_mode(value: object) -> str:
    return require_choice(value, {UNWEIGHTED_SUM_RATE_OBJECTIVE}, "uplink_objective_mode")


def validate_uplink_beam_reward_mode(value: object) -> str:
    return require_choice(value, {RATE_BEAM_REWARD_MODE}, "beam_reward_mode")


__all__ = [
    "RATE_BEAM_REWARD_MODE",
    "UNWEIGHTED_SUM_RATE_OBJECTIVE",
    "validate_uplink_beam_reward_mode",
    "validate_uplink_objective_mode",
]
