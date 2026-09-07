"""Explicit conversion of completed Torch precoders for persistence and plotting."""

from __future__ import annotations

import numpy as np
import torch


def precoder_to_numpy(
    precoder: torch.Tensor,
    *,
    dtype=np.complex128,
) -> np.ndarray:
    """Move a completed precoder out of the Torch compute layer."""
    if not isinstance(precoder, torch.Tensor):
        raise TypeError("Only Torch precoders may cross the serialization boundary.")
    return precoder.detach().cpu().numpy().astype(dtype, copy=False)


__all__ = ["precoder_to_numpy"]
