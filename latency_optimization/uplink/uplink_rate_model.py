from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from latency_optimization.core.validation import require_choice
from latency_optimization.physics.finite_blocklength import NumpyRateResult, TorchRateResult
from latency_optimization.physics.rate_law import NORMAL_APPROXIMATION_RATE_LAW, RateLaw


UPLINK_RATE_MODEL_SINR = "sinr"
UPLINK_RATE_MODEL_SNR = "snr"


def validate_uplink_rate_model(value: str) -> str:
    return require_choice(value, {UPLINK_RATE_MODEL_SINR, UPLINK_RATE_MODEL_SNR}, "uplink_rate_model")


def get_uplink_rate_model(sim_cfg: Mapping[str, Any] | None) -> str:
    if sim_cfg is None:
        return UPLINK_RATE_MODEL_SINR
    return validate_uplink_rate_model(sim_cfg.get("uplink_rate_model", UPLINK_RATE_MODEL_SINR))


def uses_uplink_interference(sim_cfg: Mapping[str, Any] | None) -> bool:
    return get_uplink_rate_model(sim_cfg) == UPLINK_RATE_MODEL_SINR


def build_uplink_rate_covariance(
    uplinksystem,
    sim_cfg: Mapping[str, Any] | None,
    user: int,
    block: int,
    *,
    F_override=None,
) -> np.ndarray | None:
    if not uses_uplink_interference(sim_cfg):
        return None
    return np.asarray(
        uplinksystem.get_interference_plus_noise_covariance(
            int(user),
            int(block),
            F_override=F_override,
        ),
        dtype=np.complex128,
    )


def evaluate_uplink_rate_numpy(
    channel: np.ndarray,
    precoder: np.ndarray,
    sigma2: float,
    epsilon: float,
    n_kl: int,
    noise_plus_interference_covariance: np.ndarray | None = None,
    *,
    covariance_jitter: float = 0.0,
    rate_law: RateLaw = NORMAL_APPROXIMATION_RATE_LAW,
) -> NumpyRateResult:
    covariance = (
        float(sigma2) * np.eye(np.asarray(channel).shape[0], dtype=np.complex128)
        if noise_plus_interference_covariance is None
        else np.asarray(noise_plus_interference_covariance, dtype=np.complex128)
    )
    return rate_law.mimo_numpy(
        channel,
        precoder,
        covariance,
        n_kl,
        epsilon,
        covariance_jitter=covariance_jitter,
    )


def evaluate_uplink_rate_torch(
    channel: torch.Tensor,
    precoder: torch.Tensor,
    sigma2: float,
    epsilon: float,
    n_kl: int,
    noise_plus_interference_covariance: torch.Tensor | None = None,
    *,
    covariance_jitter: float = 0.0,
    rate_law: RateLaw = NORMAL_APPROXIMATION_RATE_LAW,
) -> TorchRateResult:
    covariance = (
        float(sigma2)
        * torch.eye(channel.shape[0], dtype=channel.dtype, device=channel.device)
        if noise_plus_interference_covariance is None
        else noise_plus_interference_covariance.to(device=channel.device, dtype=channel.dtype)
    )
    return rate_law.mimo_torch(
        channel,
        precoder,
        covariance,
        n_kl,
        epsilon,
        covariance_jitter=covariance_jitter,
    )


__all__ = [
    "UPLINK_RATE_MODEL_SINR",
    "UPLINK_RATE_MODEL_SNR",
    "build_uplink_rate_covariance",
    "evaluate_uplink_rate_numpy",
    "evaluate_uplink_rate_torch",
    "get_uplink_rate_model",
    "uses_uplink_interference",
    "validate_uplink_rate_model",
]
