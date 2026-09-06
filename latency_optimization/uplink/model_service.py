"""Shared construction of uplink precoder snapshots from trained models."""

from __future__ import annotations

import numpy as np
import torch

from latency_optimization.runtime import DEVICE

from .precoder_models import infer_precoder_numpy_with_blocklength_and_sigma


def build_precoder_snapshot_from_models(system, models: list[torch.nn.Module]) -> list[list[np.ndarray]]:
    snapshot: list[list[np.ndarray]] = []
    for user in range(system.K):
        user_blocks = []
        for block, channel in enumerate(system.H[user]):
            user_blocks.append(
                infer_precoder_numpy_with_blocklength_and_sigma(
                    models[user],
                    channel,
                    int(system.T[user]),
                    float(system.sigma2[user]),
                    float(system.epsilon[user]),
                    int(system.NT[user]),
                    int(system.dk[user]),
                    float(system.P[user]),
                    device=DEVICE,
                )
            )
        snapshot.append(user_blocks)
    return snapshot
