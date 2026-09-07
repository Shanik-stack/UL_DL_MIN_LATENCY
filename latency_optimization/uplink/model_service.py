"""Shared construction of uplink precoder snapshots from trained models."""

from __future__ import annotations

import torch

from latency_optimization.precoders.serialization import precoder_to_numpy

from .precoders.inference import infer_precoder


def infer_precoder_for_simulator(
    model: torch.nn.Module,
    channel,
    n_kl: int,
    sigma2: float,
    epsilon: float,
    Nt: int,
    dk: int,
    P: float,
):
    """Run Torch inference, then cross the legacy simulator boundary once."""
    with torch.no_grad():
        precoder = infer_precoder(
            model,
            channel,
            n_kl,
            sigma2,
            epsilon,
            Nt,
            dk,
            P,
        )
    return precoder_to_numpy(precoder)


def build_precoder_snapshot_from_models(system, models: list[torch.nn.Module]):
    snapshot = []
    for user in range(system.K):
        user_blocks = []
        for block, channel in enumerate(system.H[user]):
            user_blocks.append(
                infer_precoder_for_simulator(
                    models[user],
                    channel,
                    int(system.T[user]),
                    float(system.sigma2[user]),
                    float(system.epsilon[user]),
                    int(system.NT[user]),
                    int(system.dk[user]),
                    float(system.P[user]),
                )
            )
        snapshot.append(user_blocks)
    return snapshot


__all__ = ["build_precoder_snapshot_from_models", "infer_precoder_for_simulator"]
