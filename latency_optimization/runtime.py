"""Shared numerical runtime settings.

Compute placement is infrastructure, not part of a precoder model.  Keeping it
here prevents solvers and evaluators from importing model modules merely to
discover the active PyTorch device.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeContext:
    device: torch.device
    real_dtype: torch.dtype = torch.float32
    complex_dtype: torch.dtype = torch.complex64


DEFAULT_RUNTIME = RuntimeContext(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
DEVICE = DEFAULT_RUNTIME.device


__all__ = ["DEFAULT_RUNTIME", "DEVICE", "RuntimeContext"]
