"""Canonical downlink finite-blocklength rate evaluation."""

from __future__ import annotations

import numpy as np
import torch

from latency_optimization.physics.finite_blocklength import ScalarRateResult, scalar_rate_result
from latency_optimization.physics.rate_law import NORMAL_APPROXIMATION_RATE_LAW, RateLaw
from latency_optimization.precoders.parameters import as_complex_tensor
from latency_optimization.runtime import DEVICE


def evaluate_downlink_rate_tensor(
    channel: torch.Tensor,
    precoder: torch.Tensor,
    interference_plus_noise_covariance: torch.Tensor,
    blocklength: int,
    error_probability: float,
    *,
    rate_law: RateLaw = NORMAL_APPROXIMATION_RATE_LAW,
) -> torch.Tensor:
    """Return the differentiable FBL rate for one user in one downlink block.

    The caller constructs the covariance from the complete BS precoder. Keeping the
    rate-law call here ensures training and simulator evaluation use the same equation.
    """
    return rate_law.mimo(
        channel,
        precoder,
        interference_plus_noise_covariance,
        int(blocklength),
        float(error_probability),
        covariance_jitter=1.0e-6,
    ).rate


def evaluate_downlink_rate_result(
    channel: np.ndarray,
    precoder: np.ndarray,
    interference_plus_noise_covariance: np.ndarray,
    blocklength: int,
    error_probability: float,
    *,
    rate_law: RateLaw = NORMAL_APPROXIMATION_RATE_LAW,
) -> ScalarRateResult:
    """Evaluate and detach all downlink FBL metrics outside gradient optimization."""
    with torch.no_grad():
        result = rate_law.mimo(
            as_complex_tensor(channel, device=DEVICE),
            as_complex_tensor(precoder, device=DEVICE),
            as_complex_tensor(interference_plus_noise_covariance, device=DEVICE),
            int(blocklength),
            float(error_probability),
            covariance_jitter=0.0,
        )
    return scalar_rate_result(result)


def evaluate_downlink_rate(
    channel: np.ndarray,
    precoder: np.ndarray,
    interference_plus_noise_covariance: np.ndarray,
    blocklength: int,
    error_probability: float,
    *,
    rate_law: RateLaw = NORMAL_APPROXIMATION_RATE_LAW,
) -> float:
    """Return only the detached scalar rate used by allocation decisions."""
    return evaluate_downlink_rate_result(
        channel,
        precoder,
        interference_plus_noise_covariance,
        blocklength,
        error_probability,
        rate_law=rate_law,
    ).rate


__all__ = [
    "evaluate_downlink_rate",
    "evaluate_downlink_rate_result",
    "evaluate_downlink_rate_tensor",
]
