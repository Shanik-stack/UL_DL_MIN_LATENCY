from typing import Sequence

import numpy as np
import torch
from latency_optimization.runtime import DEVICE

from ..precoder_models import (
    infer_precoder_numpy_with_blocklength_and_sigma,
)
from ..simulation import (
    ensure_blocks_up_to,
)
from ..system import UplinkSystem
from ..uplink_rate_model import (
    evaluate_uplink_rate_numpy,
    evaluate_uplink_rate_torch,
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
    return evaluate_uplink_rate_torch(
        H,
        Fmat,
        sigma2,
        epsilon,
        n_kl,
        noise_cov,
        covariance_jitter=1.0e-6,
    ).rate


def _compute_r_fbl_np(
    H: np.ndarray,
    Fmat: np.ndarray,
    sigma2: float,
    epsilon: float,
    n_kl: int,
    noise_plus_interference_cov: np.ndarray | None,
) -> float:
    return evaluate_uplink_rate_numpy(
        H,
        Fmat,
        sigma2,
        epsilon,
        n_kl,
        noise_plus_interference_cov,
    ).rate


def _uplink_training_beam_reward_torch(
    rate: torch.Tensor,
) -> torch.Tensor:
    return rate


def _zero_uplink_precoder(uplinksystem: UplinkSystem, user: int) -> np.ndarray:
    k = int(user)
    return np.zeros((int(uplinksystem.NT[k]), int(uplinksystem.dk[k])), dtype=np.complex64)






def _build_precoder_net_snapshot(
    uplinksystem: UplinkSystem,
    user_models: Sequence[torch.nn.Module],
    block_idx: int,
) -> list[list[np.ndarray]]:
    ensure_blocks_up_to(uplinksystem, int(block_idx))
    snapshot: list[list[np.ndarray]] = []

    for k in range(int(uplinksystem.K)):
        user_blocks: list[np.ndarray] = []
        for l in range(int(block_idx) + 1):
            user_blocks.append(
                infer_precoder_numpy_with_blocklength_and_sigma(
                    user_models[k],
                    np.asarray(uplinksystem.H[k][l], dtype=np.complex64),
                    n_kl=int(uplinksystem.T[k]),
                    sigma2=float(uplinksystem.sigma2[k]),
                    epsilon=float(uplinksystem.epsilon[k]),
                    Nt=int(uplinksystem.NT[k]),
                    dk=int(uplinksystem.dk[k]),
                    P=float(uplinksystem.P[k]),
                    device=DEVICE,
                )
            )
        snapshot.append(user_blocks)

    return snapshot


def _build_precoder_net_snapshot_for_active_mask(
    uplinksystem: UplinkSystem,
    user_models: Sequence[torch.nn.Module],
    block_idx: int,
    active_mask: Sequence[int | float],
) -> list[list[np.ndarray]]:
    snapshot = _build_precoder_net_snapshot(uplinksystem, user_models, block_idx)
    for k in range(int(uplinksystem.K)):
        if float(active_mask[int(k)]) > 0.5:
            continue
        snapshot[k][int(block_idx)] = _zero_uplink_precoder(uplinksystem, k)
    return snapshot
