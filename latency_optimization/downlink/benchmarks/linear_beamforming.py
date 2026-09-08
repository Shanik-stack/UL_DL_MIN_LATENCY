"""Standard zero-forcing beamformers for downlink benchmark evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..system import DownlinkSystem
from latency_optimization.core.validation import require_choice


def build_joint_linear_precoders(
    system: DownlinkSystem,
    block: int,
    active_users: Sequence[int],
    method: str,
) -> list[np.ndarray]:
    """Construct a standard joint ZF or RZF beamformer for one downlink block.

    What: vertically stack active users' channels, compute the zero-forcing
    pseudoinverse or its noise-regularized inverse, split the resulting columns back
    into user beam matrices, and scale the complete BS precoder to the configured
    block power budget. Inactive users receive zero matrices.

    Why: this isolates beamformer quality in the benchmark. ZF/RZF use their standard
    closed-form channel solution, while FBL rate, blocklength search, payload service,
    and latency accounting remain the same as for the proposed methods.
    """
    method_name = require_choice(method, {"zf", "rzf"}, "downlink benchmark method")

    active = [int(user) for user in active_users]
    precoders = [
        np.zeros((int(system.Nb[user]), int(system.dk[user])), dtype=np.complex128)
        for user in range(system.K)
    ]
    if not active:
        return precoders

    stacked_channel = np.concatenate([system.H[user][block] for user in active], axis=0)
    gram_matrix = stacked_channel @ stacked_channel.conj().T
    if method_name == "rzf":
        regularization = (
            float(stacked_channel.shape[0])
            * float(np.mean(system.sigma2[active]))
            / max(float(system.block_power_budget), 1e-12)
        )
        gram_matrix = gram_matrix + regularization * np.eye(
            gram_matrix.shape[0],
            dtype=np.complex128,
        )

    joint_precoder = stacked_channel.conj().T @ np.linalg.pinv(gram_matrix)
    column_start = 0
    for user in active:
        stream_count = int(system.dk[user])
        precoders[user] = np.asarray(
            joint_precoder[:, column_start : column_start + stream_count],
            dtype=np.complex128,
        )
        column_start += stream_count

    total_power = sum(np.linalg.norm(precoders[user], ord="fro") ** 2 for user in active)
    if total_power > 0.0:
        power_scale = np.sqrt(float(system.block_power_budget) / float(total_power)) * (1.0 - 1e-6)
        for user in active:
            precoders[user] *= power_scale
    return precoders


__all__ = ["build_joint_linear_precoders"]
