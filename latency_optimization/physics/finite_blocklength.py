"""Canonical Torch finite-blocklength MIMO rate calculation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import torch
from scipy.stats import norm


LOG2_E_SQUARED = (1.0 / math.log(2.0)) ** 2


@dataclass(frozen=True)
class TorchRateResult:
    rate: torch.Tensor
    capacity: torch.Tensor
    dispersion: torch.Tensor
    penalty: torch.Tensor


@dataclass(frozen=True)
class ScalarRateResult:
    """Detached values used only by scheduling and reporting code."""

    rate: float
    capacity: float
    dispersion: float
    penalty: float


def scalar_rate_result(result: TorchRateResult) -> ScalarRateResult:
    """Detach a differentiable rate result at the compute/reporting boundary."""
    return ScalarRateResult(
        rate=float(result.rate.detach().cpu()),
        capacity=float(result.capacity.detach().cpu()),
        dispersion=float(result.dispersion.detach().cpu()),
        penalty=float(result.penalty.detach().cpu()),
    )


@lru_cache(maxsize=256)
def q_inverse(epsilon: float) -> float:
    """Return the cached Gaussian Q-function inverse used by the FBL penalty."""
    epsilon_value = float(epsilon)
    if not 0.0 < epsilon_value < 1.0:
        raise ValueError(f"epsilon must be strictly between 0 and 1, got {epsilon_value}.")
    return float(norm.ppf(1.0 - epsilon_value))


def _validate_blocklength(n_kl: int) -> int:
    blocklength = int(n_kl)
    if blocklength <= 0:
        raise ValueError(f"n_kl must be strictly positive, got {blocklength}.")
    return blocklength


def _hermitian_torch(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.conj().transpose(-2, -1))


def finite_blocklength_from_metric(
    metric: torch.Tensor,
    n_kl: int,
    epsilon: float,
) -> TorchRateResult:
    """Evaluate the normal approximation from a Hermitian effective-channel metric.

    What: compute ``C=log2 det(I+A)``, eigenvalue-based channel dispersion ``V``,
    penalty ``Q^-1(epsilon)*sqrt(V/n)``, and ``R_fbl=C-penalty``. Why: keeping this
    equation here guarantees every link, optimizer, and benchmark uses identical FBL physics.
    """
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

    capacity = log_determinant.real / torch.log(
        torch.as_tensor(2.0, dtype=log_determinant.real.dtype, device=metric.device)
    )
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


def finite_blocklength_mimo(
    channel: torch.Tensor,
    precoder: torch.Tensor,
    noise_covariance: torch.Tensor,
    n_kl: int,
    epsilon: float,
    *,
    covariance_jitter: float = 0.0,
) -> TorchRateResult:
    """Convert ``(H,F,Sigma)`` into the metric required by the normal approximation.

    What: solve ``L G = H F`` for the Cholesky factor ``Sigma=L L^H``, form
    ``A=G G^H``, then call ``finite_blocklength_from_metric``. Why: whitening includes
    colored interference/noise without explicitly forming the less stable ``Sigma^-1``.
    """
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
    return finite_blocklength_from_metric(metric, n_kl, epsilon)


__all__ = [
    "LOG2_E_SQUARED",
    "ScalarRateResult",
    "TorchRateResult",
    "finite_blocklength_from_metric",
    "finite_blocklength_mimo",
    "q_inverse",
    "scalar_rate_result",
]
