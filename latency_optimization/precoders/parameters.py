"""Canonical Torch representation of complex precoder matrices."""

from __future__ import annotations

import torch

from latency_optimization.runtime import DEVICE


def as_complex_tensor(
    value,
    *,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """Convert external channel or precoder data once at a compute boundary."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.complex64)
    return torch.as_tensor(value, dtype=torch.complex64, device=device)


def complex_parameter(
    matrix,
    *,
    device: torch.device = DEVICE,
) -> torch.nn.Parameter:
    complex_matrix = as_complex_tensor(matrix, device=device)
    real_imaginary = torch.stack([complex_matrix.real, complex_matrix.imag], dim=0)
    return torch.nn.Parameter(real_imaginary)


def complex_tensor_from_parameter(parameter: torch.Tensor) -> torch.Tensor:
    if parameter.ndim < 3 or int(parameter.shape[0]) != 2:
        raise ValueError("A complex precoder parameter must have shape (2, rows, columns).")
    return (parameter[0] + 1j * parameter[1]).to(torch.complex64)


__all__ = ["as_complex_tensor", "complex_parameter", "complex_tensor_from_parameter"]
