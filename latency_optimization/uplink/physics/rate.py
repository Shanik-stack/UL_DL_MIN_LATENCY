from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from latency_optimization.experiments.config_validation import require_choice
from latency_optimization.physics.finite_blocklength import (
    ScalarRateResult,
    TorchRateResult,
    scalar_rate_result,
)
from latency_optimization.physics.rate_law import NORMAL_APPROXIMATION_RATE_LAW, RateLaw
from latency_optimization.precoders.parameters import as_complex_tensor
from latency_optimization.runtime import DEVICE


UPLINK_RATE_MODEL_SINR = "sinr"
UPLINK_RATE_MODEL_SNR = "snr"


def validate_uplink_rate_model(value: str) -> str:
    """Reject unsupported uplink rate modes before an experiment starts."""
    return require_choice(value, {UPLINK_RATE_MODEL_SINR, UPLINK_RATE_MODEL_SNR}, "uplink_rate_model")


def get_uplink_rate_model(sim_cfg: Mapping[str, Any] | None) -> str:
    """Resolve whether an uplink experiment evaluates SNR or interference-aware SINR."""
    if sim_cfg is None:
        return UPLINK_RATE_MODEL_SINR
    return validate_uplink_rate_model(sim_cfg.get("uplink_rate_model", UPLINK_RATE_MODEL_SINR))


def uses_uplink_interference(sim_cfg: Mapping[str, Any] | None) -> bool:
    """Report whether other users must enter the uplink receive covariance."""
    return get_uplink_rate_model(sim_cfg) == UPLINK_RATE_MODEL_SINR


def build_uplink_rate_covariance(
    uplinksystem,
    sim_cfg: Mapping[str, Any] | None,
    user: int,
    block: int,
    *,
    F_override=None,
) -> np.ndarray | None:
    """Build the NumPy interference-plus-noise covariance for one uplink rate query.

    Returning None in SNR mode lets all callers share one evaluation interface.
    """
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


def build_uplink_rate_covariance_torch(
    uplinksystem,
    sim_cfg: Mapping[str, Any] | None,
    user: int,
    block: int,
    *,
    precoders: list[list[torch.Tensor]] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """Build differentiable uplink interference-plus-noise covariance in Torch.

    Training and convergence use it without crossing into NumPy or breaking gradients.
    """
    if not uses_uplink_interference(sim_cfg):
        return None
    k = int(user)
    target_device = device or (
        precoders[k][int(block)].device if precoders is not None else torch.device("cpu")
    )
    receive_antennas = int(uplinksystem.NR[k])
    covariance = float(uplinksystem.sigma2[k]) * torch.eye(
        receive_antennas, dtype=torch.complex64, device=target_device
    )
    for interferer in range(int(uplinksystem.K)):
        if interferer == k:
            continue
        source = uplinksystem.F if precoders is None else precoders
        if interferer >= len(source) or not source[interferer]:
            continue
        channel_block = min(int(block), len(uplinksystem.H[interferer]) - 1)
        precoder_block = min(int(block), len(source[interferer]) - 1)
        channel = torch.as_tensor(
            uplinksystem.H[interferer][channel_block],
            dtype=torch.complex64,
            device=target_device,
        )
        beam = torch.as_tensor(
            source[interferer][precoder_block],
            dtype=torch.complex64,
            device=target_device,
        )
        received = channel @ beam
        covariance = covariance + received @ received.mH
    return 0.5 * (covariance + covariance.mH)


def evaluate_uplink_rate(
    channel,
    precoder,
    sigma2: float,
    epsilon: float,
    n_kl: int,
    noise_plus_interference_covariance=None,
    *,
    covariance_jitter: float = 0.0,
    rate_law: RateLaw = NORMAL_APPROXIMATION_RATE_LAW,
) -> ScalarRateResult:
    """Evaluate and detach one uplink FBL rate for scheduling or reporting.

    Optimization-independent code uses this wrapper around the canonical Torch rate law.
    """
    channel_tensor = as_complex_tensor(channel, device=DEVICE)
    precoder_tensor = as_complex_tensor(precoder, device=DEVICE)
    covariance = (
        float(sigma2) * torch.eye(channel_tensor.shape[0], dtype=torch.complex64, device=DEVICE)
        if noise_plus_interference_covariance is None
        else as_complex_tensor(noise_plus_interference_covariance, device=DEVICE)
    )
    with torch.no_grad():
        result = rate_law.mimo(
            channel_tensor,
            precoder_tensor,
            covariance,
            n_kl,
            epsilon,
            covariance_jitter=covariance_jitter,
        )
    return scalar_rate_result(result)


def evaluate_uplink_rate_tensor(
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
    """Evaluate a differentiable uplink FBL rate for solver and neural-network losses."""
    covariance = (
        float(sigma2)
        * torch.eye(channel.shape[0], dtype=channel.dtype, device=channel.device)
        if noise_plus_interference_covariance is None
        else noise_plus_interference_covariance.to(device=channel.device, dtype=channel.dtype)
    )
    return rate_law.mimo(
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
    "build_uplink_rate_covariance_torch",
    "evaluate_uplink_rate",
    "evaluate_uplink_rate_tensor",
    "get_uplink_rate_model",
    "uses_uplink_interference",
    "validate_uplink_rate_model",
]
