"""Downlink finite-blocklength objective and inverse-CNR weighting policy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from latency_optimization.core.validation import require_choice
from latency_optimization.runtime import DEVICE

from .system import DownlinkSystem
from .user_weights import normalized_inverse_cnr_weights


INVERSE_CNR_WEIGHTED_SUM_RATE = "inverse_cnr_weighted_sum_rate"
INVERSE_CNR_WEIGHT_STRATEGY = "inverse_cnr"


def resolve_user_blocklength(
    system: DownlinkSystem,
    user: int,
    blocklengths: dict[int, int] | None = None,
) -> int:
    user = int(user)
    if blocklengths is None:
        return int(system.T[user])
    return int(blocklengths.get(user, int(system.T[user])))


def block_rate_from_precoders(
    system: DownlinkSystem,
    active_users: Sequence[int],
    block: int,
    user: int,
    blocklength: int,
    precoders: dict[int, torch.Tensor],
) -> torch.Tensor:
    user = int(user)
    block = int(block)
    channel = torch.as_tensor(
        system.H[user][block],
        dtype=torch.complex64,
        device=DEVICE,
    )
    covariance = float(system.sigma2[user]) * torch.eye(
        int(system.Nr[user]),
        dtype=torch.complex64,
        device=DEVICE,
    )
    for interferer in active_users:
        interferer = int(interferer)
        if interferer == user:
            continue
        interference = channel @ precoders[interferer]
        covariance = covariance + interference @ interference.conj().transpose(1, 0)
    return system.rate_law.mimo_torch(
        channel,
        precoders[user],
        covariance,
        int(blocklength),
        float(system.epsilon[user]),
        covariance_jitter=1e-6,
    ).rate


def block_rate_with_precoder_override(
    system: DownlinkSystem,
    user: int,
    block: int,
    blocklength: int,
    precoder_snapshot: Sequence[Sequence[np.ndarray]],
    *,
    override_user: int | None = None,
    override_precoder: torch.Tensor | None = None,
) -> torch.Tensor:
    user = int(user)
    block = int(block)
    if block >= len(system.H[user]):
        raise ValueError(f"User {user} has no channel block {block}.")
    precoders: dict[int, torch.Tensor] = {}
    for candidate_user in range(system.K):
        if override_user is not None and candidate_user == int(override_user):
            if override_precoder is None:
                raise ValueError("override_precoder is required when override_user is set.")
            precoders[candidate_user] = override_precoder
        elif block < len(precoder_snapshot[candidate_user]):
            precoders[candidate_user] = torch.as_tensor(
                precoder_snapshot[candidate_user][block],
                dtype=torch.complex64,
                device=DEVICE,
            )
    if user not in precoders:
        raise ValueError(f"User {user} has no precoder for block {block}.")
    return block_rate_from_precoders(
        system,
        list(precoders),
        block,
        user,
        blocklength,
        precoders,
    )


def validate_objective_mode(objective_mode: str) -> str:
    return require_choice(
        objective_mode,
        {INVERSE_CNR_WEIGHTED_SUM_RATE},
        "convergence_block_objective_mode",
    )


def validate_convergence_objective_mode(simulation: dict[str, Any]) -> str:
    return validate_objective_mode(simulation["convergence_block_objective_mode"])


def validate_convergence_priority_weight_strategy(simulation: dict[str, Any]) -> str:
    return require_choice(
        simulation["convergence_priority_weight_strategy"],
        {INVERSE_CNR_WEIGHT_STRATEGY},
        "convergence_priority_weight_strategy",
    )


def objective_display_name(
    objective_mode: str,
    configured_weight_strategy: str | None = None,
) -> str:
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


def objective_weight_strategy_name(
    objective_mode: str,
    configured_weight_strategy: str,
) -> str:
    objective_display_name(objective_mode, configured_weight_strategy)
    return INVERSE_CNR_WEIGHT_STRATEGY


__all__ = [
    "INVERSE_CNR_WEIGHTED_SUM_RATE",
    "INVERSE_CNR_WEIGHT_STRATEGY",
    "block_rate_from_precoders",
    "block_rate_with_precoder_override",
    "get_convergence_objective_name",
    "normalized_inverse_cnr_weights",
    "objective_display_name",
    "objective_weight_strategy_name",
    "resolve_user_blocklength",
    "validate_convergence_objective_mode",
    "validate_convergence_priority_weight_strategy",
    "validate_objective_mode",
]
