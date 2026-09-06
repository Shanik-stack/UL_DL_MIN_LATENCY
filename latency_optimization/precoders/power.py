"""Canonical per-user and joint-BS precoder power operations."""

import math
from collections.abc import Iterable

import numpy as np
import torch


POWER_PROJECTION_SAFETY_MARGIN = 1e-6


def _identity_fallback_torch(matrix: torch.Tensor) -> torch.Tensor:
    fallback = torch.zeros_like(matrix)
    diagonal_size = min(int(matrix.shape[0]), int(matrix.shape[1]))
    fallback[:diagonal_size, :diagonal_size] = torch.eye(
        diagonal_size,
        dtype=matrix.dtype,
        device=matrix.device,
    )
    return fallback


def normalize_matrix_power_torch(
    matrix: torch.Tensor,
    power_limit: float,
    *,
    zero_fallback: bool = True,
    safety_margin: float = POWER_PROJECTION_SAFETY_MARGIN,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Scale one matrix to the requested power, even when it starts below it."""
    working = matrix
    norm = torch.linalg.norm(working, ord="fro").real
    if float(norm.detach().cpu()) <= float(eps):
        if not zero_fallback:
            return torch.zeros_like(matrix)
        working = _identity_fallback_torch(matrix)
        norm = torch.linalg.norm(working, ord="fro").real
    scale = math.sqrt(max(float(power_limit), 0.0)) / (norm + eps)
    scale = scale * (1.0 - float(safety_margin))
    return working * scale.to(working.dtype)


def cap_matrix_power_torch(
    matrix: torch.Tensor,
    power_limit: float,
    *,
    safety_margin: float = POWER_PROJECTION_SAFETY_MARGIN,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Leave a feasible matrix unchanged and scale only excess power."""
    power = (torch.linalg.norm(matrix, ord="fro") ** 2).real
    if float(power.detach().cpu()) <= float(power_limit):
        return matrix
    scale = math.sqrt(max(float(power_limit), 0.0)) / torch.sqrt(power + eps)
    scale = scale * (1.0 - float(safety_margin))
    return matrix * scale.to(matrix.dtype)


def normalize_matrix_power_numpy(
    matrix: np.ndarray,
    power_limit: float,
    *,
    zero_fallback: bool = False,
    safety_margin: float = POWER_PROJECTION_SAFETY_MARGIN,
    eps: float = 1e-12,
) -> np.ndarray:
    """NumPy counterpart of :func:`normalize_matrix_power_torch`."""
    working = np.asarray(matrix)
    norm = float(np.linalg.norm(working, ord="fro"))
    if norm <= float(eps):
        if not zero_fallback:
            return np.zeros_like(working)
        working = np.zeros_like(working)
        diagonal_size = min(int(working.shape[0]), int(working.shape[1]))
        working[:diagonal_size, :diagonal_size] = np.eye(diagonal_size, dtype=working.dtype)
        norm = float(np.linalg.norm(working, ord="fro"))
    scale = math.sqrt(max(float(power_limit), 0.0)) / (norm + eps)
    return working * (scale * (1.0 - float(safety_margin)))


def cap_matrix_power_numpy(
    matrix: np.ndarray,
    power_limit: float,
    *,
    safety_margin: float = POWER_PROJECTION_SAFETY_MARGIN,
    eps: float = 1e-12,
) -> np.ndarray:
    """NumPy counterpart of :func:`cap_matrix_power_torch`."""
    working = np.asarray(matrix)
    power = float(np.linalg.norm(working, ord="fro") ** 2)
    if power <= float(power_limit):
        return working
    scale = math.sqrt(max(float(power_limit), 0.0) / (power + eps))
    return working * (scale * (1.0 - float(safety_margin)))


def joint_power_scale_torch(
    matrices: Iterable[torch.Tensor],
    power_limit: float,
    *,
    safety_margin: float = POWER_PROJECTION_SAFETY_MARGIN,
    eps: float = 1e-12,
) -> torch.Tensor | None:
    """Return the common scale that normalizes several beams to one BS budget."""
    matrices = list(matrices)
    if not matrices:
        return None
    total_power = torch.zeros((), dtype=torch.float32, device=matrices[0].device)
    for matrix in matrices:
        total_power = total_power + (torch.linalg.norm(matrix, ord="fro") ** 2).real
    if float(total_power.detach().cpu()) <= float(eps):
        return None
    scale = torch.sqrt(float(power_limit) / (total_power + eps))
    return scale * (1.0 - float(safety_margin))


def joint_power_scale_numpy(
    matrices: Iterable[np.ndarray],
    power_limit: float,
    *,
    safety_margin: float = POWER_PROJECTION_SAFETY_MARGIN,
    eps: float = 1e-12,
) -> float | None:
    """NumPy counterpart of :func:`joint_power_scale_torch`."""
    total_power = sum(float(np.linalg.norm(matrix, ord="fro") ** 2) for matrix in matrices)
    if total_power <= float(eps):
        return None
    return math.sqrt(float(power_limit) / total_power) * (1.0 - float(safety_margin))


__all__ = [
    "POWER_PROJECTION_SAFETY_MARGIN",
    "cap_matrix_power_numpy",
    "cap_matrix_power_torch",
    "joint_power_scale_numpy",
    "joint_power_scale_torch",
    "normalize_matrix_power_numpy",
    "normalize_matrix_power_torch",
]
