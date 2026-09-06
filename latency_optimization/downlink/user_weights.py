from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalized_inverse_cnr_weights(
    channels: Sequence[np.ndarray],
    noise_powers: Sequence[float],
    active_users: Sequence[int],
) -> dict[int, float]:
    """Return inverse-CNR weights normalized to mean one over active users."""
    users = [int(k) for k in active_users]
    if not users:
        return {}

    inverse_cnr: dict[int, float] = {}
    for k in users:
        channel = np.asarray(channels[k], dtype=np.complex128)
        channel_gain = float(np.linalg.norm(channel, ord="fro") ** 2) / max(channel.shape[0], 1)
        cnr = channel_gain / max(float(noise_powers[k]), 1e-30)
        inverse_cnr[k] = 1.0 / max(cnr, 1e-30)

    mean_inverse_cnr = float(np.mean(list(inverse_cnr.values())))
    return {
        k: float(inverse_cnr[k] / max(mean_inverse_cnr, 1e-30))
        for k in users
    }
