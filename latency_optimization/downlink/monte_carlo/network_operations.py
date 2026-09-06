from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from latency_optimization.physics.finite_blocklength import finite_blocklength_mimo_torch
from latency_optimization.precoders.power import joint_power_scale_torch
from latency_optimization.results.persistence import make_serializable
from latency_optimization.runtime import DEVICE

from ..block_state import channels_for_block, make_zero_precoder
from ..config import validate_shared_bs_streaming_blocklength_input_mode
from ..model_service import describe_precoder_parameterization
from ..objective import (
    objective_display_name,
    validate_convergence_priority_weight_strategy,
)
from ..precoders.checkpoints import (
    export_user_model_specs,
    export_user_model_states,
)
from ..precoders.inference import (
    infer_raw_bs_precoders_numpy_with_blocklength,
    infer_raw_bs_precoders_torch_with_blocklength,
    infer_raw_precoder_torch_with_blocklength,
)
from ..precoders.models import (
    build_shared_bs_precoder_net_with_blocklength,
    build_user_precoder_net_with_blocklength,
    model_outputs_full_bs_precoder,
    validate_downlink_precoder_net_scope,
)
from ..system import DownlinkSystem
from ..user_weights import normalized_inverse_cnr_weights

def _build_training_user_models(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
) -> list[torch.nn.Module]:
    K = int(system_params["K"])
    max_nr = int(np.max(system_params["Nr"]))
    max_nb = int(np.max(system_params["Nb"]))
    max_dk = int(np.max(system_params["dk"]))
    model_scope = validate_downlink_precoder_net_scope(sim_params.get("downlink_precoder_net_scope", "per_user_nets"))
    if model_scope == "bs_shared_net":
        shared_model = build_shared_bs_precoder_net_with_blocklength(
            k_count=K,
            max_nr=max_nr,
            max_nb=max_nb,
            max_dk=max_dk,
            device=DEVICE,
        )
        return [shared_model for _ in range(K)]

    return [
        build_user_precoder_net_with_blocklength(
            int(system_params["Nr"][k]),
            int(system_params["Nb"][k]),
            int(system_params["dk"][k]),
            k_count=K,
            max_nr=max_nr,
            max_nb=max_nb,
            device=DEVICE,
        )
        for k in range(K)
    ]


def _unique_trainable_parameters(models: Sequence[torch.nn.Module]) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    seen_model_ids: set[int] = set()
    for model in models:
        model_id = id(model)
        if model_id in seen_model_ids:
            continue
        seen_model_ids.add(model_id)
        params.extend(list(model.parameters()))
    return params


def _models_output_full_bs_precoder(models: Sequence[torch.nn.Module]) -> bool:
    return len(models) > 0 and model_outputs_full_bs_precoder(models[0])


def _get_scenario_tensor_cache(
    scenario: dict[str, Any],
    K: int,
) -> dict[str, Any]:
    cached = scenario.get("_tensor_cache")
    if isinstance(cached, dict):
        return cached
    active_mask = np.asarray(scenario["active_mask"], dtype=np.float32)
    cache = {
        "active_mask_np": active_mask,
        "active_mask_t": torch.as_tensor(active_mask, dtype=torch.float32, device=DEVICE),
        "H_block_t": [
            torch.as_tensor(np.asarray(H_kl), dtype=torch.complex64, device=DEVICE)
            for H_kl in scenario["H_block"]
        ],
        "sigma2_t": torch.as_tensor(np.asarray(scenario["sigma2"]), dtype=torch.float32, device=DEVICE),
        "epsilon_t": torch.as_tensor(np.asarray(scenario["epsilon"]), dtype=torch.float32, device=DEVICE),
        "input_noise_covariances_t": [
            torch.as_tensor(np.asarray(cov), dtype=torch.complex64, device=DEVICE)
            for cov in scenario.get("input_noise_covariances", [])
        ],
        "n_targets_t_by_tuple": {},
        "K": int(K),
    }
    scenario["_tensor_cache"] = cache
    return cache


def _compute_r_fbl_torch(
    H: torch.Tensor,
    Fmat: torch.Tensor,
    epsilon: float,
    n_kl: int,
    noise_plus_interference_cov: torch.Tensor,
) -> torch.Tensor:
    return finite_blocklength_mimo_torch(
        H,
        Fmat,
        noise_plus_interference_cov,
        int(n_kl),
        float(epsilon),
        covariance_jitter=1.0e-6,
    ).rate


def _shared_n_targets_for_block(
    system: DownlinkSystem,
    active_mask: Sequence[int | float],
    *,
    candidate_user: int | None = None,
    candidate_n_kl: int | None = None,
) -> list[int]:
    n_targets: list[int] = []
    for k in range(system.K):
        if float(active_mask[int(k)]) <= 0.5:
            n_targets.append(0)
            continue
        if candidate_user is not None and int(k) == int(candidate_user):
            n_targets.append(int(candidate_n_kl if candidate_n_kl is not None else system.T[int(k)]))
        else:
            n_targets.append(int(system.T[int(k)]))
    return n_targets


def _shared_precoder_snapshot_for_targets(
    system: DownlinkSystem,
    model: torch.nn.Module,
    block: int,
    n_targets: Sequence[int],
    active_mask: Sequence[int | float],
    base_snapshot: list[list[np.ndarray]] | None = None,
    inference_counters: dict[str, Any] | None = None,
) -> list[list[np.ndarray]]:
    active_users = [int(k) for k, flag in enumerate(active_mask) if float(flag) > 0.5]
    if inference_counters is not None:
        inference_counters["total_forward_calls"] = int(inference_counters.get("total_forward_calls", 0)) + 1
        per_user = inference_counters.get("per_user_forward_calls")
        if isinstance(per_user, list):
            for k in active_users:
                if 0 <= int(k) < len(per_user):
                    per_user[int(k)] = int(per_user[int(k)]) + 1

    beams = infer_raw_bs_precoders_numpy_with_blocklength(
        model,
        channels_for_block(system, int(block)),
        list(n_targets),
        active_mask,
        np.asarray(system.sigma2, dtype=np.float32),
        np.asarray(system.epsilon, dtype=np.float32),
        system.Nb,
        system.dk,
        device=DEVICE,
    )
    snapshot = [list(user_blocks) for user_blocks in (base_snapshot if base_snapshot is not None else system.clone_precoders())]
    for k in active_users:
        snapshot[int(k)][int(block)] = np.asarray(beams[int(k)], dtype=np.complex128)
    system.project_block_precoders_to_power(snapshot, int(block), active_users=active_users)
    return snapshot


def _copy_snapshot_with_block_overrides(
    base_snapshot: Sequence[Sequence[np.ndarray]],
    block: int,
    block_overrides: Mapping[int, np.ndarray] | None = None,
) -> list[list[np.ndarray]]:
    snapshot = [list(user_blocks) for user_blocks in base_snapshot]
    if not isinstance(block_overrides, Mapping):
        return snapshot
    for user, precoder in block_overrides.items():
        snapshot[int(user)][int(block)] = np.asarray(precoder, dtype=np.complex128)
    return snapshot


def _masked_precoder_snapshot(
    system: DownlinkSystem,
    working_F: list[list[np.ndarray]],
    block: int,
    active_mask: Sequence[int | float],
) -> list[list[np.ndarray]]:
    zero_overrides = {
        int(k): make_zero_precoder(system, int(k))
        for k in range(system.K)
        if int(block) < len(working_F[int(k)]) and float(active_mask[int(k)]) <= 0.5
    }
    return _copy_snapshot_with_block_overrides(working_F, int(block), zero_overrides)


def _build_block_joint_scenario(
    system: DownlinkSystem,
    block: int,
    active_mask: Sequence[int | float],
    *,
    scenario_mode: str,
) -> dict[str, Any]:
    active_mask_vec = [int(float(v) > 0.5) for v in active_mask]
    return {
        "seed": int(system.seed),
        "block": int(block),
        "H_block": [np.asarray(H_kl, dtype=np.complex64) for H_kl in channels_for_block(system, int(block))],
        "active_mask": active_mask_vec,
        "max_n_targets": [
            int(system.T[int(k)]) if int(active_mask_vec[int(k)]) > 0 else 0
            for k in range(system.K)
        ],
        "block_power_budget": float(system.block_power_budget),
        "P": [float(v) for v in system.P.tolist()],
        "fs": [float(v) for v in system.fs.tolist()],
        "sigma2": [float(v) for v in system.sigma2.tolist()],
        "epsilon": [float(v) for v in system.epsilon.tolist()],
        "scenario_mode": str(scenario_mode),
    }


def _scenario_input_noise_covariances(
    system: DownlinkSystem,
    snapshot: list[list[np.ndarray]],
    block: int,
    active_mask: Sequence[int | float],
) -> list[np.ndarray]:
    covariances: list[np.ndarray] = []
    for k in range(system.K):
        if float(active_mask[int(k)]) > 0.5:
            cov = system.get_interference_plus_noise_covariance(int(k), int(block), F_override=snapshot)
        else:
            cov = np.asarray(float(system.sigma2[int(k)]) * np.eye(int(system.Nr[int(k)])), dtype=np.complex128)
        covariances.append(np.asarray(cov, dtype=np.complex128))
    return covariances


def _project_predicted_beams_to_block_power_torch(
    predicted_beams: Sequence[torch.Tensor],
    active_mask: Sequence[int | float] | np.ndarray,
    block_power_budget: float,
    eps: float = 1e-12,
) -> list[torch.Tensor]:
    active = np.asarray(active_mask, dtype=np.float32)
    active_beams = [
        beam
        for k, beam in enumerate(predicted_beams)
        if k < len(active) and float(active[k]) > 0.5
    ]
    scale = joint_power_scale_torch(
        active_beams,
        block_power_budget,
        safety_margin=0.0,
        eps=eps,
    )
    if scale is None:
        return [beam for beam in predicted_beams]
    projected: list[torch.Tensor] = []
    for k, beam in enumerate(predicted_beams):
        if k < len(active) and float(active[k]) > 0.5:
            projected.append(beam * scale.to(beam.dtype))
        else:
            projected.append(beam)
    return projected


def _joint_noise_covariance_torch(
    H_block: Sequence[torch.Tensor],
    predicted_beams: Sequence[torch.Tensor],
    sigma2: float,
    user: int,
    active_mask: Sequence[int | float],
) -> torch.Tensor:
    k = int(user)
    Hk = H_block[k]
    Nrk = int(Hk.shape[0])
    cov = float(sigma2) * torch.eye(Nrk, dtype=torch.complex64, device=Hk.device)
    for j, Fj in enumerate(predicted_beams):
        if int(j) == k or float(active_mask[int(j)]) <= 0.5:
            continue
        HFj = Hk @ Fj
        cov = cov + (HFj @ HFj.conj().transpose(1, 0))
    cov = 0.5 * (cov + cov.conj().transpose(1, 0))
    return cov + (1e-6 * torch.eye(Nrk, dtype=torch.complex64, device=Hk.device))


def _scenario_forward_pass(
    system_params: dict[str, Any],
    scenario: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
    n_targets: Sequence[int],
    anchor_bits: Sequence[int] | None = None,
    inference_counters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    K = int(system_params["K"])
    cache = _get_scenario_tensor_cache(scenario, K)
    active_mask = cache["active_mask_np"]
    n_targets_list = [int(v) for v in n_targets]
    anchor_bits_list = (
        [int(v) for v in anchor_bits]
        if anchor_bits is not None
        else [int(v) for v in scenario.get("rollout_anchor_bits", [0 for _ in range(K)])]
    )
    if len(anchor_bits_list) < K:
        anchor_bits_list = anchor_bits_list + [0 for _ in range(K - len(anchor_bits_list))]
    active_mask_t = cache["active_mask_t"]
    H_block_t = cache["H_block_t"]
    predicted_beams: list[torch.Tensor] = []
    rates: list[torch.Tensor | None] = [None for _ in range(K)]
    powers: list[torch.Tensor | None] = [None for _ in range(K)]
    required_rates = [0.0 for _ in range(K)]
    sum_rate = torch.zeros((), dtype=torch.float32, device=DEVICE)

    if _models_output_full_bs_precoder(user_models):
        if inference_counters is not None:
            inference_counters["total_forward_calls"] = int(inference_counters.get("total_forward_calls", 0)) + 1
            per_user = inference_counters.get("per_user_forward_calls")
            if isinstance(per_user, list):
                for k in range(K):
                    if float(active_mask[k]) > 0.5 and 0 <= int(k) < len(per_user):
                        per_user[int(k)] = int(per_user[int(k)]) + 1
        n_targets_cache = cache["n_targets_t_by_tuple"]
        n_targets_key = tuple(int(v) for v in n_targets_list)
        n_targets_t = n_targets_cache.get(n_targets_key)
        if n_targets_t is None:
            n_targets_t = torch.as_tensor(n_targets_list, dtype=torch.float32, device=DEVICE)
            n_targets_cache[n_targets_key] = n_targets_t
        predicted_beams = infer_raw_bs_precoders_torch_with_blocklength(
            user_models[0],
            H_block_t,
            n_targets_t,
            active_mask_t,
            cache["sigma2_t"],
            cache["epsilon_t"],
            system_params["Nb"],
            system_params["dk"],
        )
        for k in range(K):
            if float(active_mask[k]) <= 0.5 or int(n_targets_list[k]) <= 0:
                predicted_beams[k] = torch.zeros(
                    (int(system_params["Nb"][k]), int(system_params["dk"][k])),
                    dtype=torch.complex64,
                    device=DEVICE,
                )
    else:
        for k in range(K):
            if float(active_mask[k]) <= 0.5 or int(n_targets_list[k]) <= 0:
                predicted_beams.append(
                    torch.zeros(
                        (int(system_params["Nb"][k]), int(system_params["dk"][k])),
                        dtype=torch.complex64,
                        device=DEVICE,
                    )
                )
                continue

            if inference_counters is not None:
                inference_counters["total_forward_calls"] = int(inference_counters.get("total_forward_calls", 0)) + 1
                per_user = inference_counters.get("per_user_forward_calls")
                if isinstance(per_user, list) and 0 <= int(k) < len(per_user):
                    per_user[int(k)] = int(per_user[int(k)]) + 1
            if k < len(cache["input_noise_covariances_t"]):
                noise_cov_input_t = cache["input_noise_covariances_t"][k]
            else:
                noise_cov_input_t = torch.as_tensor(
                    np.asarray(scenario["input_noise_covariances"][k]),
                    dtype=torch.complex64,
                    device=DEVICE,
                )
            predicted_beams.append(
                infer_raw_precoder_torch_with_blocklength(
                    user_models[k],
                    H_block_t,
                    int(n_targets_list[k]),
                    active_mask_t,
                    noise_cov_input_t,
                    float(scenario["epsilon"][k]),
                    int(system_params["Nb"][k]),
                    int(system_params["dk"][k]),
                    user_index=int(k),
                )
            )

    predicted_beams = _project_predicted_beams_to_block_power_torch(
        predicted_beams,
        active_mask,
        float(scenario["block_power_budget"]),
    )

    block_power = torch.zeros((), dtype=torch.float32, device=DEVICE)

    for k in range(K):
        if float(active_mask[k]) <= 0.5 or int(n_targets_list[k]) <= 0:
            continue
        noise_cov_joint = _joint_noise_covariance_torch(
            H_block_t,
            predicted_beams,
            float(scenario["sigma2"][k]),
            k,
            active_mask,
        )
        rate = _compute_r_fbl_torch(
            H_block_t[k],
            predicted_beams[k],
            epsilon=float(scenario["epsilon"][k]),
            n_kl=int(n_targets_list[k]),
            noise_plus_interference_cov=noise_cov_joint,
        )
        power = (torch.linalg.norm(predicted_beams[k], ord="fro") ** 2).real
        required_bits = int(anchor_bits_list[k]) if int(k) < len(anchor_bits_list) else 0
        required_rate = (
            float(required_bits) / float(max(int(n_targets_list[k]), 1))
            if required_bits > 0
            else 0.0
        )
        rates[k] = rate
        powers[k] = power
        required_rates[k] = required_rate
        sum_rate = sum_rate + rate
        block_power = block_power + power

    block_power_gap = block_power - float(scenario["block_power_budget"])
    block_power_violation = torch.relu(block_power_gap)

    return {
        "active_mask": active_mask,
        "n_targets": n_targets_list,
        "predicted_beams": predicted_beams,
        "rates": rates,
        "powers": powers,
        "rollout_anchor_bits": [int(v) for v in anchor_bits_list],
        "required_rates": required_rates,
        "sum_rate": sum_rate,
        "block_power": block_power,
        "block_power_gap": block_power_gap,
        "block_power_violation": block_power_violation,
    }


def _resolve_downlink_rollout_anchor_bits_from_forward(forward: dict[str, Any]) -> list[int]:
    anchor_bits = [0 for _ in range(len(forward["n_targets"]))]
    for k, rate_t in enumerate(forward["rates"]):
        if rate_t is None:
            continue
        if float(forward["active_mask"][k]) <= 0.5 or int(forward["n_targets"][k]) <= 0:
            continue
        achievable_bits = int(
            np.floor(
                max(float(rate_t.detach().cpu()), 0.0)
                * float(max(int(forward["n_targets"][k]), 1))
            )
        )
        anchor_bits[int(k)] = max(1, achievable_bits)
    return anchor_bits


def _scenario_metrics_from_forward(forward: dict[str, Any]) -> dict[str, Any]:
    K = int(len(forward["rates"]))
    rate_values = [0.0 for _ in range(K)]
    required_rates = [0.0 for _ in range(K)]
    rate_margins = [0.0 for _ in range(K)]
    active_users: list[int] = []
    feasible = True

    for k in range(K):
        rate_t = forward["rates"][k]
        if rate_t is None:
            continue
        active_users.append(int(k))
        rate_val = float(rate_t.detach().cpu())
        required_rate = float(forward["required_rates"][k])
        margin = float(rate_val - required_rate)
        rate_values[k] = rate_val
        required_rates[k] = required_rate
        rate_margins[k] = margin
        if margin < 0.0:
            feasible = False
    if float(forward["block_power_gap"].detach().cpu()) > 0.0:
        feasible = False

    active_margins = [rate_margins[k] for k in active_users]
    return {
        "feasible": bool(feasible),
        "active_users": active_users,
        "rate_values": rate_values,
        "required_rates": required_rates,
        "rate_margins": rate_margins,
        "rollout_anchor_bits": [int(v) for v in forward.get("rollout_anchor_bits", [0 for _ in range(K)])],
        "min_rate_margin": float(min(active_margins)) if active_margins else 0.0,
        "sum_rate": float(forward["sum_rate"].detach().cpu()),
        "block_power_gap": float(forward["block_power_gap"].detach().cpu()),
    }


def _scenario_metrics_with_models(
    system_params: dict[str, Any],
    scenario: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
    n_targets: Sequence[int],
    anchor_bits: Sequence[int] | None = None,
    inference_counters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with torch.no_grad():
        forward = _scenario_forward_pass(
            system_params,
            scenario,
            user_models,
            n_targets,
            anchor_bits=anchor_bits,
            inference_counters=inference_counters,
        )
    return _scenario_metrics_from_forward(forward)


def _best_joint_n_target_transition(
    system_params: dict[str, Any],
    scenario: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
    current_n_targets: Sequence[int],
    anchor_bits: Sequence[int],
    *,
    n_min: int,
    n_step: int,
    direction: str = "descending",
    max_n_targets: Sequence[int] | None = None,
    eligible_users: Sequence[int] | None = None,
    inference_counters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    direction = require_choice(direction, {"descending", "ascending"}, "joint n-target transition direction")

    if max_n_targets is None:
        max_targets = [int(v) for v in scenario.get("max_n_targets", current_n_targets)]
    else:
        max_targets = [int(v) for v in max_n_targets]

    if eligible_users is None:
        base_candidate_users = [
            int(k)
            for k, active in enumerate(scenario["active_mask"])
            if int(active) > 0
        ]
    else:
        base_candidate_users = [
            int(k)
            for k in eligible_users
            if 0 <= int(k) < len(scenario["active_mask"])
            and int(scenario["active_mask"][int(k)]) > 0
        ]

    if direction == "descending":
        candidate_users = [
            int(k)
            for k in base_candidate_users
            if int(current_n_targets[int(k)]) - int(n_step) >= int(n_min)
        ]
        subset_sizes = range(len(candidate_users), 0, -1)
    else:
        candidate_users = [
            int(k)
            for k in base_candidate_users
            if int(current_n_targets[int(k)]) + int(n_step) <= int(max_targets[int(k)])
        ]
        subset_sizes = range(1, len(candidate_users) + 1)
    if len(candidate_users) == 0:
        return {"accepted": None, "rejected": None}

    best_rejected: dict[str, Any] | None = None
    best_rejected_key: tuple[float, float] | None = None

    for subset_size in subset_sizes:
        best_feasible: dict[str, Any] | None = None
        best_feasible_key: tuple[float, float] | None = None
        for subset in combinations(candidate_users, subset_size):
            candidate_n_targets = [int(v) for v in current_n_targets]
            for k in subset:
                if direction == "descending":
                    candidate_n_targets[int(k)] -= int(n_step)
                else:
                    candidate_n_targets[int(k)] += int(n_step)
            metrics = _scenario_metrics_with_models(
                system_params,
                scenario,
                user_models,
                candidate_n_targets,
                anchor_bits=anchor_bits,
                inference_counters=inference_counters,
            )
            candidate = {
                "reduced_users": [int(k) for k in subset],
                "candidate_n_targets": [int(v) for v in candidate_n_targets],
                "metrics": metrics,
            }
            rejected_key = (float(metrics["min_rate_margin"]), float(metrics["sum_rate"]))
            if best_rejected_key is None or rejected_key > best_rejected_key:
                best_rejected_key = rejected_key
                best_rejected = candidate
            if not bool(metrics["feasible"]):
                continue
            if direction == "descending":
                feasible_key = (float(metrics["sum_rate"]), float(metrics["min_rate_margin"]))
            else:
                feasible_key = (-float(sum(candidate_n_targets)), float(metrics["sum_rate"]))
            if best_feasible_key is None or feasible_key > best_feasible_key:
                best_feasible_key = feasible_key
                best_feasible = candidate
        if best_feasible is not None:
            return {"accepted": best_feasible, "rejected": best_rejected}

    return {"accepted": None, "rejected": best_rejected}


def _build_rollout_query_from_downlink_state(
    scenario: dict[str, Any],
    n_targets: Sequence[int],
    metrics: dict[str, Any],
    rollout_anchor_bits: Sequence[int],
    *,
    rollout_phase: str,
    rollout_stage: str,
    frontier_query: bool,
) -> dict[str, Any]:
    return {
        **scenario,
        "n_targets": [int(v) for v in n_targets],
        "rollout_anchor_bits": [int(v) for v in rollout_anchor_bits],
        "rollout_phase": str(rollout_phase),
        "rollout_stage": str(rollout_stage),
        "frontier_query": bool(frontier_query),
        "rollout_feasible": bool(metrics["feasible"]),
        "rollout_min_rate_margin": float(metrics["min_rate_margin"]),
        "rollout_sum_rate": float(metrics["sum_rate"]),
        "rollout_rate_values": [float(v) for v in metrics.get("rate_values", [])],
    }


def _scenario_objective_weights(
    scenario: dict[str, Any],
    sim_params: dict[str, Any],
    *,
    objective_mode: str,
) -> tuple[list[float], str]:
    active_mask = [int(v) for v in scenario.get("active_mask", [])]
    K = int(len(active_mask))
    weights = [1.0 for _ in range(K)]
    objective_display_name(objective_mode, "inverse_cnr")
    strategy = str(validate_convergence_priority_weight_strategy(sim_params))
    active_users = [int(k) for k, flag in enumerate(active_mask) if int(flag) > 0]
    if len(active_users) <= 0:
        return weights, strategy

    active_weights = normalized_inverse_cnr_weights(
        scenario["H_block"],
        scenario["sigma2"],
        active_users,
    )
    for k, weight in active_weights.items():
        weights[int(k)] = float(weight)
    return weights, strategy


def _supported_bits_from_forward(forward: dict[str, Any]) -> list[int]:
    supported_bits = [0 for _ in range(len(forward["rates"]))]
    for k, rate_t in enumerate(forward["rates"]):
        if rate_t is None:
            continue
        if float(forward["active_mask"][k]) <= 0.5 or int(forward["n_targets"][k]) <= 0:
            continue
        supported_bits[int(k)] = max(
            int(
                np.floor(
                    max(float(rate_t.detach().cpu()), 0.0)
                    * float(max(int(forward["n_targets"][k]), 1))
                )
            ),
            0,
        )
    return supported_bits




def build_precoder_net_artifact(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    train_seeds: Sequence[int],
    user_models: Sequence[torch.nn.Module],
    precoder_net_training_history: dict[str, Any],
    training_dataset_sizes: Sequence[int],
) -> dict[str, Any]:
    model_scope = validate_downlink_precoder_net_scope(sim_params.get("downlink_precoder_net_scope", "per_user_nets"))
    streaming_blocklength_input_mode = validate_shared_bs_streaming_blocklength_input_mode(
        sim_params.get(
            "shared_bs_streaming_blocklength_input_mode",
            "joint_blocklength_vector",
        )
    )
    return {
        "system_params": system_params,
        "sim_params": sim_params,
        "train_seeds": [int(v) for v in train_seeds],
        "training_dataset_sizes": [int(v) for v in training_dataset_sizes],
        "training_active_user_case_counts_per_user": [int(v) for v in training_dataset_sizes],
        "precoder_net_training_losses": [
            list(map(float, row))
            for row in precoder_net_training_history.get("per_user_objective_loss", [])
        ],
        "precoder_net_training_history": make_serializable(precoder_net_training_history),
        "user_model_specs": export_user_model_specs(
            system_params["Nr"],
            system_params["Nb"],
            system_params["dk"],
            uses_blocklength_input=True,
            context_k=int(system_params["K"]),
            context_max_nr=int(np.max(system_params["Nr"])),
            context_max_nb=int(np.max(system_params["Nb"])),
            context_max_dk=int(np.max(system_params["dk"])),
            model_scope=model_scope,
        ),
        "user_model_states": export_user_model_states(user_models),
        "precoder_parameterization": describe_precoder_parameterization(
            model_scope,
            "precoder_net",
            uses_blocklength_input=True,
        ),
        "downlink_precoder_net_scope": str(model_scope),
        "shared_bs_streaming_blocklength_input_mode": (
            str(streaming_blocklength_input_mode)
            if str(model_scope) == "bs_shared_net"
            else "not_applicable"
        ),
        "training_objective": precoder_net_training_history.get(
            "training_objective",
            "scenario_driven_rollout_finite_blocklength_rate",
        ),
    }

