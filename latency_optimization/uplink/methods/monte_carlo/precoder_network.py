"""Construct and evaluate uplink precoder networks during training."""

from typing import Sequence

import torch
from latency_optimization.runtime import DEVICE

from ...precoders.inference import infer_precoder
from ...simulation.operations import (
    ensure_blocks_up_to,
)
from ...simulation.system import UplinkSystem
from ...physics.rate import (
    evaluate_uplink_rate_tensor,
)

ROLLOUT_QUERY_OBJECTIVE_TRAINING_STYLE = "rollout_query_objective"


def _compute_r_fbl_torch(
    H: torch.Tensor,
    Fmat: torch.Tensor,
    sigma2: float,
    epsilon: float,
    n_kl: int,
    noise_plus_interference_cov: torch.Tensor | None,
) -> torch.Tensor:
    if noise_plus_interference_cov is None:
        noise_cov = float(sigma2) * torch.eye(
            H.shape[0], dtype=H.dtype, device=H.device
        )
    else:
        noise_cov = noise_plus_interference_cov.to(device=H.device, dtype=torch.complex64)
    return evaluate_uplink_rate_tensor(
        H,
        Fmat,
        sigma2,
        epsilon,
        n_kl,
        noise_cov,
        covariance_jitter=1.0e-6,
    ).rate


def _uplink_training_beam_reward_torch(
    rate: torch.Tensor,
) -> torch.Tensor:
    return rate


def _zero_uplink_precoder(uplinksystem: UplinkSystem, user: int) -> torch.Tensor:
    k = int(user)
    return torch.zeros(
        (int(uplinksystem.NT[k]), int(uplinksystem.dk[k])),
        dtype=torch.complex64,
        device=DEVICE,
    )






def _build_precoder_net_snapshot(
    uplinksystem: UplinkSystem,
    user_models: Sequence[torch.nn.Module],
    block_idx: int,
) -> list[list[torch.Tensor]]:
    ensure_blocks_up_to(uplinksystem, int(block_idx))
    snapshot: list[list[torch.Tensor]] = []

    for k in range(int(uplinksystem.K)):
        user_blocks: list[torch.Tensor] = []
        for l in range(int(block_idx) + 1):
            with torch.no_grad():
                precoder = infer_precoder(
                    user_models[k],
                    uplinksystem.H[k][l],
                    blocklength=int(uplinksystem.T[k]),
                    noise_variance=float(uplinksystem.sigma2[k]),
                    error_probability=float(uplinksystem.epsilon[k]),
                    transmit_antennas=int(uplinksystem.NT[k]),
                    streams=int(uplinksystem.dk[k]),
                    power_limit=float(uplinksystem.P[k]),
                )
            user_blocks.append(precoder.detach())
        snapshot.append(user_blocks)

    return snapshot


def _build_precoder_net_snapshot_for_active_mask(
    uplinksystem: UplinkSystem,
    user_models: Sequence[torch.nn.Module],
    block_idx: int,
    active_mask: Sequence[int | float],
) -> list[list[torch.Tensor]]:
    snapshot = _build_precoder_net_snapshot(uplinksystem, user_models, block_idx)
    for k in range(int(uplinksystem.K)):
        if float(active_mask[int(k)]) > 0.5:
            continue
        snapshot[k][int(block_idx)] = _zero_uplink_precoder(uplinksystem, k)
    return snapshot
