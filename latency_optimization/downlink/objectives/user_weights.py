from __future__ import annotations

from collections.abc import Sequence

import torch

from latency_optimization.runtime import DEVICE


def normalized_inverse_cnr_weights(
    channels: Sequence[object],
    noise_powers: Sequence[float],
    active_users: Sequence[int],
) -> dict[int, float]:
    """Return inverse-CNR weights normalized to mean one over active users."""
    users = [int(k) for k in active_users]
    if not users:
        return {}

    channel_tensors = [
        torch.as_tensor(channels[k], dtype=torch.complex64, device=DEVICE)
        for k in users
    ]
    channel_gains = torch.stack(
        [channel.abs().square().sum() / max(channel.shape[0], 1) for channel in channel_tensors]
    )
    noise = torch.as_tensor(
        [float(noise_powers[k]) for k in users],
        dtype=channel_gains.dtype,
        device=channel_gains.device,
    ).clamp_min(1e-30)
    inverse_cnr = (channel_gains / noise).clamp_min(1e-30).reciprocal()
    normalized = inverse_cnr / inverse_cnr.mean().clamp_min(1e-30)
    return {
        user: float(weight)
        for user, weight in zip(users, normalized.detach().cpu().tolist())
    }
