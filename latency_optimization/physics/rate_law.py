"""Replaceable physical-layer rate-law contract."""

from __future__ import annotations

from typing import Protocol

import torch

from .finite_blocklength import (
    TorchRateResult,
    finite_blocklength_from_metric,
    finite_blocklength_mimo,
)


class RateLaw(Protocol):
    """A rate law usable by simulation and gradient-based optimization."""

    name: str

    def from_metric(
        self, metric: torch.Tensor, blocklength: int, error_probability: float
    ) -> TorchRateResult: ...

    def mimo(
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

    def from_metric(
        self, metric: torch.Tensor, blocklength: int, error_probability: float
    ) -> TorchRateResult:
        return finite_blocklength_from_metric(metric, blocklength, error_probability)

    def mimo(
        self,
        channel: torch.Tensor,
        precoder: torch.Tensor,
        noise_covariance: torch.Tensor,
        blocklength: int,
        error_probability: float,
        *,
        covariance_jitter: float = 0.0,
    ) -> TorchRateResult:
        return finite_blocklength_mimo(
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
