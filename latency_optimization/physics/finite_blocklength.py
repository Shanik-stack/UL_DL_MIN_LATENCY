"""Canonical finite-blocklength MIMO rate calculation.

The NumPy and Torch entry points intentionally share the same decomposition:
whiten the desired channel, form its Hermitian metric, and then evaluate the
capacity and dispersion terms. Torch remains differentiable with respect to
the precoder; NumPy is used for simulation and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from scipy.stats import norm


LOG2_E_SQUARED = float(np.log2(np.e) ** 2)


@dataclass(frozen=True)
class NumpyRateResult:
    rate: float
    capacity: float
    dispersion: float
    penalty: float


@dataclass(frozen=True)
class TorchRateResult:
    rate: torch.Tensor
    capacity: torch.Tensor
    dispersion: torch.Tensor
    penalty: torch.Tensor


@lru_cache(maxsize=256)
def q_inverse(epsilon: float) -> float:
    epsilon_value = float(epsilon)
    if not 0.0 < epsilon_value < 1.0:
        raise ValueError(f"epsilon must be strictly between 0 and 1, got {epsilon_value}.")
    return float(norm.ppf(1.0 - epsilon_value))


def _validate_blocklength(n_kl: int) -> int:
    blocklength = int(n_kl)
    if blocklength <= 0:
        raise ValueError(f"n_kl must be strictly positive, got {blocklength}.")
    return blocklength


def _hermitian_numpy(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


def _hermitian_torch(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.conj().transpose(-2, -1))


def finite_blocklength_from_metric_numpy(
    metric: np.ndarray,
    n_kl: int,
    epsilon: float,
) -> NumpyRateResult:
    blocklength = _validate_blocklength(n_kl)
    effective_metric = _hermitian_numpy(np.asarray(metric, dtype=np.complex128))
    identity = np.eye(effective_metric.shape[-1], dtype=np.complex128)
    sign, log_determinant = np.linalg.slogdet(identity + effective_metric)
    if float(np.real(sign)) <= 0.0:
        raise RuntimeError("I + effective channel metric is not positive definite.")

    capacity = float(np.real(log_determinant) / np.log(2.0))
    eigenvalues = np.linalg.eigvalsh(effective_metric).real
    dispersion = float(
        np.sum(eigenvalues * (eigenvalues + 2.0) / (eigenvalues + 1.0) ** 2)
        * LOG2_E_SQUARED
    )
    penalty = float(np.sqrt(max(dispersion, 0.0) / blocklength) * q_inverse(epsilon))
    return NumpyRateResult(
        rate=float(capacity - penalty),
        capacity=capacity,
        dispersion=dispersion,
        penalty=penalty,
    )


def finite_blocklength_from_metric_torch(
    metric: torch.Tensor,
    n_kl: int,
    epsilon: float,
) -> TorchRateResult:
    blocklength = _validate_blocklength(n_kl)
    effective_metric = _hermitian_torch(metric)
    identity = torch.eye(
        effective_metric.shape[-1],
        dtype=effective_metric.dtype,
        device=effective_metric.device,
    )
    sign, log_determinant = torch.linalg.slogdet(identity + effective_metric)
    if torch.any(sign.real <= 0):
        raise RuntimeError("I + effective channel metric is not positive definite.")

    capacity = log_determinant.real / np.log(2.0)
    eigenvalues = torch.linalg.eigvalsh(effective_metric).real
    dispersion = (
        torch.sum(eigenvalues * (eigenvalues + 2.0) / (eigenvalues + 1.0) ** 2)
        * LOG2_E_SQUARED
    )
    q_value = torch.as_tensor(
        q_inverse(epsilon),
        dtype=capacity.dtype,
        device=capacity.device,
    )
    penalty = torch.sqrt(torch.clamp(dispersion, min=0.0) / float(blocklength)) * q_value
    return TorchRateResult(
        rate=(capacity - penalty).real,
        capacity=capacity.real,
        dispersion=dispersion.real,
        penalty=penalty.real,
    )


def finite_blocklength_mimo_numpy(
    channel: np.ndarray,
    precoder: np.ndarray,
    noise_covariance: np.ndarray,
    n_kl: int,
    epsilon: float,
    *,
    covariance_jitter: float = 0.0,
) -> NumpyRateResult:
    channel_matrix = np.asarray(channel, dtype=np.complex128)
    precoder_matrix = np.asarray(precoder, dtype=np.complex128)
    covariance = _hermitian_numpy(np.asarray(noise_covariance, dtype=np.complex128))
    if covariance_jitter > 0.0:
        covariance = covariance + float(covariance_jitter) * np.eye(
            covariance.shape[-1], dtype=np.complex128
        )
    whitened_channel = np.linalg.solve(
        np.linalg.cholesky(covariance),
        channel_matrix @ precoder_matrix,
    )
    metric = whitened_channel @ whitened_channel.conj().T
    return finite_blocklength_from_metric_numpy(metric, n_kl, epsilon)


def finite_blocklength_mimo_torch(
    channel: torch.Tensor,
    precoder: torch.Tensor,
    noise_covariance: torch.Tensor,
    n_kl: int,
    epsilon: float,
    *,
    covariance_jitter: float = 0.0,
) -> TorchRateResult:
    covariance = _hermitian_torch(
        noise_covariance.to(device=channel.device, dtype=channel.dtype)
    )
    if covariance_jitter > 0.0:
        covariance = covariance + float(covariance_jitter) * torch.eye(
            covariance.shape[-1], dtype=covariance.dtype, device=covariance.device
        )
    whitened_channel = torch.linalg.solve(
        torch.linalg.cholesky(covariance),
        channel @ precoder,
    )
    metric = whitened_channel @ whitened_channel.conj().transpose(-2, -1)
    return finite_blocklength_from_metric_torch(metric, n_kl, epsilon)


__all__ = [
    "LOG2_E_SQUARED",
    "NumpyRateResult",
    "TorchRateResult",
    "finite_blocklength_from_metric_numpy",
    "finite_blocklength_from_metric_torch",
    "finite_blocklength_mimo_numpy",
    "finite_blocklength_mimo_torch",
    "q_inverse",
]
