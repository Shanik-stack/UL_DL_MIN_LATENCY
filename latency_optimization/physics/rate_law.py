"""Replaceable physical-layer rate-law contract."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from .finite_blocklength import (
    NumpyRateResult,
    TorchRateResult,
    finite_blocklength_from_metric_numpy,
    finite_blocklength_from_metric_torch,
    finite_blocklength_mimo_numpy,
    finite_blocklength_mimo_torch,
)


class RateLaw(Protocol):
    """A rate law usable by simulation and gradient-based optimization."""

    name: str

    def from_metric_numpy(
        self, metric: np.ndarray, blocklength: int, error_probability: float
    ) -> NumpyRateResult: ...

    def from_metric_torch(
        self, metric: torch.Tensor, blocklength: int, error_probability: float
    ) -> TorchRateResult: ...

    def mimo_numpy(
        self,
        channel: np.ndarray,
        precoder: np.ndarray,
        noise_covariance: np.ndarray,
        blocklength: int,
        error_probability: float,
        *,
        covariance_jitter: float = 0.0,
    ) -> NumpyRateResult: ...

    def mimo_torch(
        self,
        channel: torch.Tensor,
        precoder: torch.Tensor,
        noise_covariance: torch.Tensor,
        blocklength: int,
        error_probability: float,
        *,
        covariance_jitter: float = 0.0,
    ) -> TorchRateResult: ...


class NormalApproximationRateLaw:
    name = "normal_approximation"

    def from_metric_numpy(
        self, metric: np.ndarray, blocklength: int, error_probability: float
    ) -> NumpyRateResult:
        return finite_blocklength_from_metric_numpy(metric, blocklength, error_probability)

    def from_metric_torch(
        self, metric: torch.Tensor, blocklength: int, error_probability: float
    ) -> TorchRateResult:
        return finite_blocklength_from_metric_torch(metric, blocklength, error_probability)

    def mimo_numpy(
        self,
        channel: np.ndarray,
        precoder: np.ndarray,
        noise_covariance: np.ndarray,
        blocklength: int,
        error_probability: float,
        *,
        covariance_jitter: float = 0.0,
    ) -> NumpyRateResult:
        return finite_blocklength_mimo_numpy(
            channel,
            precoder,
            noise_covariance,
            blocklength,
            error_probability,
            covariance_jitter=covariance_jitter,
        )

    def mimo_torch(
        self,
        channel: torch.Tensor,
        precoder: torch.Tensor,
        noise_covariance: torch.Tensor,
        blocklength: int,
        error_probability: float,
        *,
        covariance_jitter: float = 0.0,
    ) -> TorchRateResult:
        return finite_blocklength_mimo_torch(
            channel,
            precoder,
            noise_covariance,
            blocklength,
            error_probability,
            covariance_jitter=covariance_jitter,
        )


NORMAL_APPROXIMATION_RATE_LAW = NormalApproximationRateLaw()
RATE_LAWS: dict[str, RateLaw] = {
    NORMAL_APPROXIMATION_RATE_LAW.name: NORMAL_APPROXIMATION_RATE_LAW,
}


def resolve_rate_law(name: str) -> RateLaw:
    try:
        return RATE_LAWS[str(name)]
    except KeyError as error:
        choices = ", ".join(sorted(RATE_LAWS))
        raise ValueError(f"Unsupported finite-blocklength rate law {name!r}; choose one of: {choices}.") from error


__all__ = [
    "NORMAL_APPROXIMATION_RATE_LAW",
    "NormalApproximationRateLaw",
    "RATE_LAWS",
    "RateLaw",
    "resolve_rate_law",
]
