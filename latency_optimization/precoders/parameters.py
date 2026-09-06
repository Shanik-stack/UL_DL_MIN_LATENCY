"""Shared differentiable representation of complex precoder matrices."""

from __future__ import annotations

import numpy as np
import torch

from latency_optimization.runtime import DEVICE


def complex_parameter_from_numpy(
    matrix: np.ndarray,
    *,
    device: torch.device = DEVICE,
) -> torch.nn.Parameter:
    complex_matrix = np.asarray(matrix, dtype=np.complex64)
    real_imaginary = np.stack([complex_matrix.real, complex_matrix.imag], axis=0)
    return torch.nn.Parameter(torch.as_tensor(real_imaginary, dtype=torch.float32, device=device))


def complex_tensor_from_parameter(parameter: torch.Tensor) -> torch.Tensor:
    if parameter.ndim < 3 or int(parameter.shape[0]) != 2:
        raise ValueError("A complex precoder parameter must have shape (2, rows, columns).")
    return (parameter[0] + 1j * parameter[1]).to(torch.complex64)
