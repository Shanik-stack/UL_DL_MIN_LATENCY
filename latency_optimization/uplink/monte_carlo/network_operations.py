import copy
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from latency_optimization.core.blocklength import build_fixed_step_n_candidates, build_n_search_config, run_n_frontier_search
from latency_optimization.core.scenarios import PAYLOAD_MODE, STREAMING_MODE, build_experiment_scenario
from latency_optimization.experiments.channels import build_training_snr_schedule, with_monte_carlo_sample_snr_by_user
from latency_optimization.results.console import format_log_line, format_progress_log_line
from latency_optimization.runtime import DEVICE

from ..config import RATE_BEAM_REWARD_MODE, UNWEIGHTED_SUM_RATE_OBJECTIVE, get_config, validate_uplink_objective_mode
from ..precoder_models import (
    build_user_precoder_net_with_blocklength_and_sigma,
    export_user_model_specs,
    export_user_model_states,
    infer_precoder_numpy_with_blocklength_and_sigma,
    infer_precoder_torch_with_blocklength_and_sigma,
)
from ..simulation import (
    apply_training_solution,
    clone_nested_arrays,
    collect_uplink_interference_diagnostics,
    ensure_blocks_up_to,
    estimate_initial_random_precoder_schedule,
    estimate_initial_random_precoder_schedule_for_scenario as shared_estimate_initial_random_precoder_schedule_for_scenario,
)
from ..system import UplinkSystem
from ..uplink_rate_model import (
    build_uplink_rate_covariance,
    evaluate_uplink_rate_numpy,
    evaluate_uplink_rate_torch,
    uses_uplink_interference,
)

ROLLOUT_QUERY_OBJECTIVE_TRAINING_STYLE = "rollout_query_objective"


def _build_monte_carlo_training_search_cfg(
    sim_cfg: dict[str, Any],
    *,
    n_min: int,
    n_max: int,
) -> dict[str, int | str]:
    return build_n_search_config(
        n_min=int(n_min),
        n_max=int(n_max),
        fine_step=int(sim_cfg["n_kl_step"]),
        direction=sim_cfg.get("n_search_direction", "descending"),
        strategy=sim_cfg.get("n_search_strategy", "fixed_step"),
        coarse_step=sim_cfg.get("n_search_coarse_step", int(sim_cfg["n_kl_step"])),
        exponential_factor=sim_cfg.get("n_search_exponential_factor", 2),
        allow_only_fixed_step=True,
    )


def _build_monte_carlo_test_search_cfg(
    sim_cfg: dict[str, Any],
    *,
    n_min: int,
    n_max: int,
) -> dict[str, int | str]:
    return build_n_search_config(
        n_min=int(n_min),
        n_max=int(n_max),
        fine_step=int(sim_cfg["n_kl_step"]),
        direction=sim_cfg.get("monte_carlo_test_n_search_direction", sim_cfg.get("n_search_direction", "descending")),
        strategy=sim_cfg.get("monte_carlo_test_n_search_strategy", sim_cfg.get("n_search_strategy", "fixed_step")),
        coarse_step=sim_cfg.get(
            "monte_carlo_test_n_search_coarse_step",
            sim_cfg.get("n_search_coarse_step", int(sim_cfg["n_kl_step"])),
        ),
        exponential_factor=sim_cfg.get(
            "monte_carlo_test_n_search_exponential_factor",
            sim_cfg.get("n_search_exponential_factor", 2),
        ),
        allow_only_fixed_step=False,
    )


def _to_complex_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.complex64, copy=False)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.complex64, copy=False)
    return np.asarray(x, dtype=np.complex64)


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
