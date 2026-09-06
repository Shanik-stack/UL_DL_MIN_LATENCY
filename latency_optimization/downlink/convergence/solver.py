from __future__ import annotations

import copy
from typing import Any, List, Sequence

import numpy as np
import torch

from latency_optimization.core.blocklength import build_n_search_config, run_n_frontier_search
from latency_optimization.core.scenarios import STREAMING_MODE, build_experiment_scenario
from latency_optimization.core.validation import require_choice
from latency_optimization.optimization.stopping import KktResiduals, convergence_status_from_config
from latency_optimization.precoders.power import joint_power_scale_torch
from latency_optimization.precoders.model_state import clone_model_state
from latency_optimization.precoders.parameters import (
    complex_parameter_from_numpy,
    complex_tensor_from_parameter,
)
from latency_optimization.results.console import format_log_line, format_latency_log_line, format_progress_log_line
from latency_optimization.runtime import DEVICE

from ..block_state import (
    clone_precoders,
    maximum_supported_bits,
    power_to_db,
    user_link_budget,
)
from ..model_service import (
    infer_shared_block_precoders_numpy,
    infer_shared_block_precoders_torch,
    models_output_full_bs_precoder,
)
from ..objective import (
    INVERSE_CNR_WEIGHTED_SUM_RATE as INVERSE_CNR_WEIGHTED_SUM_RATE_PUBLIC_NAME,
    INVERSE_CNR_WEIGHT_STRATEGY,
    block_rate_from_precoders,
    get_convergence_objective_name,
    objective_display_name,
    objective_weight_strategy_name,
    resolve_user_blocklength,
    validate_convergence_objective_mode,
    validate_convergence_priority_weight_strategy,
    validate_objective_mode,
)

from ..precoders.inference import (
    infer_raw_precoder_numpy,
    infer_raw_precoder_torch,
)
from ..system import DownlinkSystem
from ..user_weights import normalized_inverse_cnr_weights

CONVERGENCE_PRECODER_UPDATE_MODES = {"precoder_net", "direct_precoder"}
def validate_convergence_precoder_update_mode(sim_params: dict[str, Any]) -> str:
    return require_choice(
        sim_params.get("convergence_precoder_update_mode", "precoder_net"),
        CONVERGENCE_PRECODER_UPDATE_MODES,
        "convergence_precoder_update_mode",
    )


def _project_active_precoders_to_block_power(
    system: DownlinkSystem,
    precoders: dict[int, torch.Tensor],
    active_users: List[int],
    eps: float = 1e-12,
) -> dict[int, torch.Tensor]:
    if len(active_users) == 0:
        return precoders
    scale = joint_power_scale_torch(
        (precoders[int(k)] for k in active_users),
        system.block_power_budget,
        eps=eps,
    )
    if scale is None:
        return precoders
    return {int(k): (precoders[int(k)] * scale.to(precoders[int(k)].dtype)) for k in active_users}

def _evaluate_block_objective(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user_models: list[torch.nn.Module],
    active_users: List[int],
    update_users: List[int],
    block: int,
    requested_bits: dict[int, int],
    objective_mode: str,
    user_weights: dict[int, float],
    *,
    n_kl_overrides: dict[int, int] | None = None,
    direct_precoders: dict[int, torch.Tensor] | None = None,
) -> dict[str, Any]:
    canonical_mode = validate_objective_mode(objective_mode)
    update_user_set = {int(k) for k in update_users}

    precoders: dict[int, torch.Tensor] = {}
    if direct_precoders is not None:
        for k in active_users:
            k_int = int(k)
            if k_int in direct_precoders:
                precoders[k_int] = direct_precoders[k_int]
            else:
                precoders[k_int] = torch.tensor(
                    working_F[k_int][int(block)],
                    dtype=torch.complex64,
                    device=DEVICE,
                )
    elif models_output_full_bs_precoder(user_models) and len(update_user_set) > 0:
        precoders = infer_shared_block_precoders_torch(
            system,
            user_models[0],
            int(block),
            active_users,
        )
    else:
        for k in active_users:
            k_int = int(k)
            if k_int in update_user_set:
                H_kl = torch.tensor(system.H[k_int][int(block)], dtype=torch.complex64, device=DEVICE)
                precoders[k_int] = infer_raw_precoder_torch(
                    user_models[k_int],
                    H_kl,
                    nb=int(system.Nb[k_int]),
                    dk=int(system.dk[k_int]),
                    user_index=int(k_int),
                )
            else:
                precoders[k_int] = torch.tensor(
                    working_F[k_int][int(block)],
                    dtype=torch.complex64,
                    device=DEVICE,
                )
    precoders = _project_active_precoders_to_block_power(system, precoders, active_users)

    rates: dict[int, torch.Tensor] = {}
    powers: dict[int, torch.Tensor] = {}
    required_rates: dict[int, float] = {}
    rate_gaps: dict[int, torch.Tensor] = {}
    rate_violation_pos: dict[int, torch.Tensor] = {}

    for k in active_users:
        k_int = int(k)
        n_k = resolve_user_blocklength(system, k_int, n_kl_overrides)
        rate_k = block_rate_from_precoders(
            system,
            active_users,
            int(block),
            k_int,
            n_k,
            precoders,
        )
        power_k = (torch.linalg.norm(precoders[k_int], ord="fro") ** 2).real
        required_rate_k = float(requested_bits.get(k_int, 0)) / float(max(int(n_k), 1))
        rate_gap_k = torch.tensor(required_rate_k, dtype=torch.float32, device=DEVICE) - rate_k
        rates[k_int] = rate_k
        powers[k_int] = power_k
        required_rates[k_int] = float(required_rate_k)
        rate_gaps[k_int] = rate_gap_k
        rate_violation_pos[k_int] = torch.relu(rate_gap_k)

    block_power = (
        torch.stack([powers[int(k)] for k in active_users]).sum()
        if active_users
        else torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
    )
    block_power_gap = block_power - float(system.block_power_budget)
    block_power_violation_pos = torch.relu(block_power_gap)

    total_rate = torch.stack([rates[int(k)] for k in active_users]).sum() if active_users else torch.tensor(0.0, device=DEVICE)
    weighted_total = torch.stack(
        [
            float(user_weights.get(int(k), 1.0)) * rates[int(k)]
            for k in active_users
        ]
    ).sum() if active_users else torch.tensor(0.0, device=DEVICE)
    loss = -weighted_total

    return {
        "loss": loss,
        "rates": rates,
        "powers": powers,
        "required_rates": required_rates,
        "rate_gap": rate_gaps,
        "rate_violation_pos": rate_violation_pos,
        "block_power": block_power,
        "block_power_gap": block_power_gap,
        "block_power_violation_pos": block_power_violation_pos,
        "sum_rate": total_rate,
        INVERSE_CNR_WEIGHTED_SUM_RATE_PUBLIC_NAME: weighted_total,
        "weighted_sum_rate": weighted_total,
    }


def _build_user_weights(
    system: DownlinkSystem,
    active_users: List[int],
    block: int,
) -> dict[int, float]:
    channels = [np.asarray(system.H[k][int(block)]) for k in range(system.K)]
    return normalized_inverse_cnr_weights(channels, system.sigma2, active_users)


def _resolve_block_user_weights(
    system: DownlinkSystem,
    active_users: List[int],
    block: int,
    sim_params: dict[str, Any],
    objective_mode: str,
) -> tuple[dict[int, float], str]:
    if len(active_users) <= 0:
        return {}, INVERSE_CNR_WEIGHT_STRATEGY

    validate_objective_mode(objective_mode)
    resolved_strategy = validate_convergence_priority_weight_strategy(sim_params)
    return (
        _build_user_weights(
            system,
            active_users,
            block,
        ),
        resolved_strategy,
    )


def _optimize_user_block_precoder_for_objective(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user_models: list[torch.nn.Module],
    active_users: List[int],
    requested_bits: dict[int, int],
    user_weights: dict[int, float],
    user: int,
    block: int,
    objective_mode: str,
    precoder_model: torch.nn.Module,
    model_optimizer: torch.optim.Optimizer,
    n_kl_overrides: dict[int, int] | None = None,
) -> np.ndarray:
    k = int(user)
    l = int(block)
    model_optimizer.zero_grad()
    state = _evaluate_block_objective(
        system,
        working_F,
        user_models=user_models,
        active_users=active_users,
        update_users=[k],
        block=l,
        requested_bits=requested_bits,
        objective_mode=objective_mode,
        user_weights=user_weights,
        n_kl_overrides=n_kl_overrides,
    )
    state["loss"].backward()
    model_optimizer.step()

    beam_np = infer_raw_precoder_numpy(
        precoder_model,
        np.asarray(system.H[k][l], dtype=np.complex64),
        nb=int(system.Nb[k]),
        dk=int(system.dk[k]),
        device=DEVICE,
        user_index=int(k),
    )
    beam_snapshot = clone_precoders(working_F)
    beam_snapshot[k][l] = np.asarray(beam_np, dtype=np.complex128)
    system.project_block_precoders_to_power(beam_snapshot, l, active_users=[int(j) for j in active_users])
    return np.asarray(beam_snapshot[k][l], dtype=np.complex128)


def _optimize_active_block_precoders_direct(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    active_users: List[int],
    update_users: List[int],
    block: int,
    requested_bits: dict[int, int],
    sim_params: dict[str, Any],
    objective_mode: str,
    user_weights: dict[int, float],
    n_kl_overrides: dict[int, int] | None = None,
) -> dict[int, np.ndarray]:
    if len(update_users) == 0:
        return {}

    params = {
        int(k): complex_parameter_from_numpy(np.asarray(working_F[int(k)][int(block)], dtype=np.complex64))
        for k in update_users
    }
    optimizer = torch.optim.Adam(
        [param for param in params.values()],
        lr=float(sim_params["user_update_lr"]),
    )
    l = int(block)

    optimizer.zero_grad()
    state = _evaluate_block_objective(
        system,
        working_F,
        user_models=[],
        active_users=active_users,
        update_users=update_users,
        block=l,
        requested_bits=requested_bits,
        objective_mode=objective_mode,
        user_weights=user_weights,
        n_kl_overrides=n_kl_overrides,
        direct_precoders={int(k): complex_tensor_from_parameter(param) for k, param in params.items()},
    )
    state["loss"].backward()
    optimizer.step()

    snapshot = clone_precoders(working_F)
    for k in update_users:
        snapshot[int(k)][l] = (
            complex_tensor_from_parameter(params[int(k)]).detach().cpu().numpy().astype(np.complex128, copy=False)
        )
    system.project_block_precoders_to_power(snapshot, l, active_users=[int(j) for j in active_users])
    return {
        int(k): np.asarray(snapshot[int(k)][l], dtype=np.complex128)
        for k in update_users
    }


def _block_delta(
    before_beams: dict[int, np.ndarray],
    working_F: List[List[np.ndarray]],
    active_users: List[int],
    block: int,
) -> float:
    deltas = []
    for k in active_users:
        prev = np.asarray(before_beams[k], dtype=np.complex128)
        curr = np.asarray(working_F[k][block], dtype=np.complex128)
        denom = max(float(np.linalg.norm(prev, ord="fro")), 1e-12)
        deltas.append(float(np.linalg.norm(curr - prev, ord="fro") / denom))
    return max(deltas) if deltas else 0.0


def _copy_active_model_optimizer_states(
    user_models: list[torch.nn.Module],
    model_optimizers: list[torch.optim.Optimizer],
    active_users: List[int],
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, Any]]]:
    if len(user_models) == 0 or len(model_optimizers) == 0:
        return {}, {}
    model_states = {
        int(k): clone_model_state(user_models[int(k)])
        for k in active_users
    }
    optimizer_states = {
        int(k): copy.deepcopy(model_optimizers[int(k)].state_dict())
        for k in active_users
    }
    return model_states, optimizer_states


def _restore_active_model_optimizer_states(
    user_models: list[torch.nn.Module],
    model_optimizers: list[torch.optim.Optimizer],
    active_users: List[int],
    model_states: dict[int, dict[str, torch.Tensor]],
    optimizer_states: dict[int, dict[str, Any]],
) -> None:
    if len(user_models) == 0 or len(model_optimizers) == 0:
        return
    for k in active_users:
        user_models[int(k)].load_state_dict(model_states[int(k)])
        model_optimizers[int(k)].load_state_dict(optimizer_states[int(k)])


def _capture_active_block_solver_state(
    working_F: List[List[np.ndarray]],
    user_models: list[torch.nn.Module],
    model_optimizers: list[torch.optim.Optimizer],
    active_users: List[int],
) -> dict[str, Any]:
    model_states, optimizer_states = _copy_active_model_optimizer_states(
        user_models,
        model_optimizers,
        active_users,
    )
    return {
        "working_F": clone_precoders(working_F),
        "model_states": model_states,
        "optimizer_states": optimizer_states,
    }


def _restore_active_block_solver_state(
    working_F: List[List[np.ndarray]],
    user_models: list[torch.nn.Module],
    model_optimizers: list[torch.optim.Optimizer],
    active_users: List[int],
    state: dict[str, Any],
) -> None:
    working_F[:] = clone_precoders(state["working_F"])
    _restore_active_model_optimizer_states(
        user_models,
        model_optimizers,
        active_users,
        state["model_states"],
        state["optimizer_states"],
    )


def _all_committed_bits_feasible(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    active_users: List[int],
    block: int,
    committed_bits: dict[int, int],
    n_kl_targets: dict[int, int],
) -> tuple[bool, dict[int, float], list[int]]:
    user_rates: dict[int, float] = {}
    infeasible_users: list[int] = []
    for k in active_users:
        k_int = int(k)
        B_used = int(committed_bits.get(k_int, 0))
        n_k = resolve_user_blocklength(system, k_int, n_kl_targets)
        rate_k = float(system.compute_block_rate(k_int, int(block), n_k, F_override=working_F))
        user_rates[k_int] = rate_k
        if B_used <= 0:
            continue
        required_rate = float(B_used) / float(max(n_k, 1))
        if float(required_rate - rate_k) > 0.0:
            infeasible_users.append(int(k_int))
    return len(infeasible_users) == 0, user_rates, infeasible_users


def _resolve_reduced_n_reoptimization_users(
    active_users: List[int],
    candidate_user: int,
    infeasible_users: list[int],
    scope: str,
) -> list[int]:
    ordered_active = [int(k) for k in active_users]
    infeasible_set = {int(k) for k in infeasible_users}
    scope_key = require_choice(
        scope,
        {"all_active_users", "infeasible_users_only", "candidate_and_infeasible_users"},
        "simulation.n_kl_reduction_update_scope",
    )

    if scope_key == "all_active_users":
        return ordered_active

    if scope_key == "infeasible_users_only":
        return [k for k in ordered_active if k in infeasible_set]

    if scope_key == "candidate_and_infeasible_users":
        update_users = [int(candidate_user)]
        update_users.extend(k for k in ordered_active if k in infeasible_set and int(k) != int(candidate_user))
        return update_users

    raise ValueError(
        "simulation.n_kl_reduction_update_scope must be one of "
        "{'all_active_users', 'infeasible_users_only', 'candidate_and_infeasible_users'}."
    )


def _reduce_blocklengths_with_reoptimization(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user_models: list[torch.nn.Module],
    model_optimizers: list[torch.optim.Optimizer],
    active_users: List[int],
    block: int,
    requested_bits: dict[int, int],
    sim_params: dict[str, Any],
    *,
    objective_mode: str,
    user_weights: dict[int, float],
    verbose: bool,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, float]]]:
    n_min = int(sim_params["n_kl_min"])
    n_step = int(sim_params["n_kl_step"])
    reoptimization_scope = require_choice(
        sim_params.get("n_kl_reduction_update_scope", "all_active_users"),
        {"all_active_users", "infeasible_users_only", "candidate_and_infeasible_users"},
        "simulation.n_kl_reduction_update_scope",
    )

    current_n_targets = {
        int(k): int(system.T[int(k)])
        for k in active_users
    }
    committed_bits: dict[int, int] = {}
    current_rates: dict[int, float] = {}
    plans: dict[int, dict[str, Any]] = {}
    refinement_history: list[dict[str, float]] = []
    stopped_reduction_users: set[int] = set()
    search_direction = str(sim_params.get("n_search_direction", "descending"))
    search_strategy = str(sim_params.get("n_search_strategy", "fixed_step"))
    search_coarse_step = int(sim_params.get("n_search_coarse_step", int(n_step)))
    search_exponential_factor = int(sim_params.get("n_search_exponential_factor", 2))

    for k in active_users:
        k_int = int(k)
        T_k = int(system.T[k_int])
        R_T = float(system.compute_block_rate(k_int, int(block), T_k, F_override=working_F))
        B_max = max(maximum_supported_bits(T_k, R_T), 0)
        B_used = int(min(int(requested_bits.get(k_int, 0)), B_max))
        committed_bits[k_int] = int(B_used)
        current_rates[k_int] = float(R_T)
        plans[k_int] = {
            "B_used": int(B_used),
            "n_used": int(T_k),
            "R_used": float(R_T if B_used > 0 else 0.0),
        }

    full_service_users = []
    for k in active_users:
        k_int = int(k)
        requested_k = int(requested_bits.get(k_int, 0))
        B_used = int(committed_bits.get(k_int, 0))
        if requested_k <= 0 or B_used <= 0:
            continue
        if B_used < requested_k:
            if verbose:
                print(
                    f"  user={k_int:02d} block={block:02d} serves partial bits at n=T; "
                    "not reducing n_kl."
                )
            continue
        full_service_users.append(int(k_int))

    epoch_idx = 0
    while len(full_service_users) > 0:
        active_reduction_users = [
            int(k_int)
            for k_int in full_service_users
            if int(k_int) not in stopped_reduction_users
        ]
        if len(active_reduction_users) == 0:
            break
        progress_this_epoch = False
        start_offset = int(epoch_idx % max(len(active_reduction_users), 1))
        ordered_users = (
            active_reduction_users[start_offset:] + active_reduction_users[:start_offset]
        )
        if verbose:
            print(
                format_log_line(
                    "[DL Convergence Reduction]",
                    block=int(block),
                    epoch=int(epoch_idx + 1),
                    direction=search_direction,
                    strategy=search_strategy,
                    users=[int(k) for k in ordered_users],
                )
            )

        for k_int in ordered_users:
            current_n = int(current_n_targets.get(k_int, int(system.T[k_int])))
            if current_n <= int(n_min):
                stopped_reduction_users.add(int(k_int))
                continue

            search_cfg = build_n_search_config(
                n_min=int(n_min),
                n_max=int(current_n),
                fine_step=int(n_step),
                direction=search_direction,
                strategy=search_strategy,
                coarse_step=int(search_coarse_step),
                exponential_factor=int(search_exponential_factor),
            )

            def _evaluate_candidate_n(candidate_n: int, stage_name: str) -> dict[str, Any]:
                candidate_targets = dict(current_n_targets)
                candidate_targets[k_int] = int(candidate_n)
                feasible_without_reopt, candidate_rates, infeasible_users = _all_committed_bits_feasible(
                    system,
                    working_F,
                    active_users,
                    block,
                    committed_bits,
                    candidate_targets,
                )
                if feasible_without_reopt:
                    if verbose:
                        print(
                            f"  user={k_int:02d} block={block:02d} candidate n_kl={int(candidate_n):4d} "
                            f"accepted at search_stage={stage_name} without fresh re-optimization."
                        )
                    return {
                        "feasible": True,
                        "candidate_targets": candidate_targets,
                        "candidate_rates": candidate_rates,
                        "update_users": [],
                        "history": [],
                        "accepted_via": "no_reoptimization",
                    }

                if verbose:
                    infeasible_text = ", ".join(str(int(user_id)) for user_id in infeasible_users)
                    print(
                        f"  user={k_int:02d} block={block:02d} candidate n_kl={int(candidate_n):4d} "
                        f"breaks committed-user feasibility [{infeasible_text}] at search_stage={stage_name}; "
                        "trying fresh re-optimization."
                    )

                update_users = _resolve_reduced_n_reoptimization_users(
                    active_users,
                    k_int,
                    infeasible_users,
                    reoptimization_scope,
                )
                solver_checkpoint = _capture_active_block_solver_state(
                    working_F,
                    user_models,
                    model_optimizers,
                    active_users,
                )
                solve_result = optimize_precoders_for_block(
                    system,
                    working_F,
                    user_models,
                    model_optimizers,
                    active_users,
                    block,
                    committed_bits,
                    sim_params,
                    verbose=verbose,
                    objective_mode=objective_mode,
                    user_weights=user_weights,
                    n_kl_overrides=candidate_targets,
                    users_to_update=update_users,
                    max_epochs=int(sim_params["max_epochs"]),
                )
                candidate_history = solve_result["history"]
                feasible, candidate_rates, remaining_infeasible_users = _all_committed_bits_feasible(
                    system,
                    working_F,
                    active_users,
                    block,
                    committed_bits,
                    candidate_targets,
                )
                if not feasible:
                    _restore_active_block_solver_state(
                        working_F,
                        user_models,
                        model_optimizers,
                        active_users,
                        solver_checkpoint,
                    )
                    if verbose:
                        infeasible_text = ", ".join(str(int(user_id)) for user_id in remaining_infeasible_users)
                        updated_text = ", ".join(str(int(user_id)) for user_id in update_users)
                        print(
                            f"  user={k_int:02d} block={block:02d} candidate n_kl={int(candidate_n):4d} "
                            f"after updating users [{updated_text}] still leaves committed users infeasible "
                            f"[{infeasible_text}] at search_stage={stage_name}."
                        )
                    return {
                        "feasible": False,
                        "candidate_targets": candidate_targets,
                        "candidate_rates": candidate_rates,
                        "update_users": [int(user_id) for user_id in update_users],
                        "history": candidate_history,
                        "remaining_infeasible_users": [int(user_id) for user_id in remaining_infeasible_users],
                        "accepted_via": "reoptimization_failed",
                    }

                if verbose:
                    updated_text = ", ".join(str(int(user_id)) for user_id in update_users)
                    print(
                        f"  user={k_int:02d} block={block:02d} candidate n_kl={int(candidate_n):4d} "
                        f"accepted at search_stage={stage_name} after fresh re-optimization of users [{updated_text}]."
                    )
                return {
                    "feasible": True,
                    "candidate_targets": candidate_targets,
                    "candidate_rates": candidate_rates,
                    "update_users": [int(user_id) for user_id in update_users],
                    "history": candidate_history,
                    "accepted_via": "reoptimization",
                }

            search_result = run_n_frontier_search(search_cfg, _evaluate_candidate_n)
            accepted_events = search_result["accepted"]
            if len(accepted_events) <= 0:
                stopped_reduction_users.add(int(k_int))
                continue

            best_event = accepted_events[-1]
            best_result = best_event["result"]
            best_n = int(best_event["n_kl"])
            current_n_targets = dict(best_result["candidate_targets"])
            current_rates = dict(best_result["candidate_rates"])
            plans[k_int]["n_used"] = int(best_n)
            plans[k_int]["R_used"] = float(current_rates.get(k_int, current_rates.get(k_int, 0.0)))
            for accepted_event in accepted_events:
                candidate_history = accepted_event["result"].get("history", [])
                if len(candidate_history) > 0:
                    refinement_history.extend(candidate_history)
            progress_this_epoch = bool(best_n != current_n)
            if not progress_this_epoch:
                stopped_reduction_users.add(int(k_int))
                continue
            if verbose:
                print(
                    f"  user={k_int:02d} block={block:02d} committed best n_kl={int(best_n):4d} "
                    f"after {len(accepted_events)} accepted search states."
                )
            if best_n <= int(n_min) or search_result.get("frontier_rejected") is not None:
                stopped_reduction_users.add(int(k_int))

        if not progress_this_epoch:
            break
        epoch_idx += 1

    for k in active_users:
        k_int = int(k)
        final_n = int(current_n_targets[k_int])
        final_rate = float(system.compute_block_rate(k_int, int(block), final_n, F_override=working_F))
        plans[k_int]["n_used"] = int(final_n)
        plans[k_int]["R_used"] = float(final_rate if committed_bits.get(k_int, 0) > 0 else 0.0)

    return plans, refinement_history


def optimize_precoders_for_block(
    system: DownlinkSystem,
    working_F: List[List[np.ndarray]],
    user_models: list[torch.nn.Module],
    model_optimizers: list[torch.optim.Optimizer],
    active_users: List[int],
    block: int,
    requested_bits: dict[int, int],
    sim_params: dict[str, Any],
    *,
    verbose: bool = True,
    objective_mode: str = INVERSE_CNR_WEIGHTED_SUM_RATE_PUBLIC_NAME,
    user_weights: dict[int, float] | None = None,
    n_kl_overrides: dict[int, int] | None = None,
    users_to_update: List[int] | None = None,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    history: list[dict[str, float]] = []
    if len(active_users) == 0:
        return {
            "history": history,
            "solve_status": "empty",
        }

    weights = user_weights or {int(k): 1.0 for k in active_users}
    update_mode = validate_convergence_precoder_update_mode(sim_params)
    shared_bs_scope = (update_mode == "precoder_net") and models_output_full_bs_precoder(user_models)
    update_users = [int(k) for k in (active_users if shared_bs_scope else (users_to_update or active_users))]
    print_every = max(1, int(sim_params.get("print_every_epoch", 1)))
    max_epochs = max(1, int(max_epochs if max_epochs is not None else sim_params["max_epochs"]))
    canonical_mode = validate_objective_mode(objective_mode)
    objective_label = INVERSE_CNR_WEIGHTED_SUM_RATE_PUBLIC_NAME

    best_objective = -float("inf")
    best_feasible_found = False
    solve_status = "max_epochs_reached"
    best_objective_state = _capture_active_block_solver_state(
        working_F,
        user_models,
        model_optimizers,
        active_users,
    )

    for epoch_idx in range(max_epochs):
        before_beams = {k: np.array(working_F[k][block], copy=True) for k in update_users}

        if update_mode == "direct_precoder":
            updated_beams = _optimize_active_block_precoders_direct(
                system,
                working_F,
                active_users,
                update_users,
                block,
                requested_bits,
                sim_params,
                objective_mode,
                weights,
                n_kl_overrides,
            )
            for k in update_users:
                if int(k) in updated_beams:
                    working_F[int(k)][int(block)] = np.asarray(updated_beams[int(k)], dtype=np.complex128)
            system.project_block_precoders_to_power(working_F, int(block), active_users=[int(j) for j in active_users])
        elif shared_bs_scope:
            shared_optimizer = model_optimizers[0]
            shared_optimizer.zero_grad()
            shared_step_state = _evaluate_block_objective(
                system,
                working_F,
                user_models,
                active_users,
                active_users,
                block,
                requested_bits,
                objective_mode,
                weights,
                n_kl_overrides=n_kl_overrides,
            )
            shared_step_state["loss"].backward()
            shared_optimizer.step()
            block_precoders = infer_shared_block_precoders_numpy(
                system,
                user_models[0],
                int(block),
                active_users,
            )
            for k in active_users:
                working_F[int(k)][int(block)] = np.asarray(block_precoders[int(k)], dtype=np.complex128)
            system.project_block_precoders_to_power(working_F, int(block), active_users=[int(j) for j in active_users])
            shared_optimizer.zero_grad()
        else:
            for k in update_users:
                beam_k = _optimize_user_block_precoder_for_objective(
                    system,
                    working_F,
                    user_models,
                    active_users,
                    requested_bits,
                    weights,
                    k,
                    block,
                    objective_mode,
                    user_models[int(k)],
                    model_optimizers[int(k)],
                    n_kl_overrides,
                )
                working_F[int(k)][block] = np.array(beam_k, copy=True)
                system.project_block_precoders_to_power(working_F, int(block), active_users=[int(j) for j in active_users])

            for k in update_users:
                model_optimizers[int(k)].zero_grad()
        with torch.no_grad():
            state = _evaluate_block_objective(
                system,
                working_F,
                user_models if update_mode == "precoder_net" else [],
                active_users,
                active_users if (shared_bs_scope and update_mode == "precoder_net") else ([] if update_mode == "direct_precoder" else update_users),
                block,
                requested_bits,
                objective_mode,
                weights,
                n_kl_overrides=n_kl_overrides,
                direct_precoders={} if update_mode == "direct_precoder" else None,
            )

        rate_gaps = {int(k): float(state["rate_gap"][int(k)].detach().cpu()) for k in active_users}
        rate_violations = {int(k): float(state["rate_violation_pos"][int(k)].detach().cpu()) for k in active_users}
        block_power_gap = float(state["block_power_gap"].detach().cpu())
        block_power_violation = float(state["block_power_violation_pos"].detach().cpu())
        exact_feasible = (
            all(float(rate_gaps[int(k)]) <= 0.0 for k in active_users)
            and float(block_power_gap) <= 0.0
        )
        r_p = max(
            max(rate_violations.values(), default=0.0),
            block_power_violation,
        )
        r_c = 0.0
        user_rates = [float(state["rates"][int(k)].detach().cpu()) for k in active_users]
        user_sinr_db = []
        user_interference_db = []
        user_signal_db = []
        for k in active_users:
            signal_power, interference_power, _, sinr_db = user_link_budget(system, working_F, int(k), int(block))
            user_sinr_db.append(float(sinr_db))
            user_interference_db.append(power_to_db(interference_power))
            user_signal_db.append(power_to_db(signal_power))

        total_rate = float(state["sum_rate"].detach().cpu())
        weighted_total = float(state["weighted_sum_rate"].detach().cpu())
        block_power = float(state["block_power"].detach().cpu())
        objective_value = weighted_total
        delta = _block_delta(before_beams, working_F, update_users, block)
        r_s = float(delta)
        history.append(
            {
                "block": int(block),
                "epoch": epoch_idx + 1,
                "active_users": int(len(active_users)),
                "updated_users": int(len(update_users)),
                "user_ids": [int(k) for k in active_users],
                "updated_user_ids": [int(k) for k in update_users],
                "user_n_kl": [resolve_user_blocklength(system, int(k), n_kl_overrides) for k in active_users],
                "user_rates": user_rates,
                "user_sinr_db": user_sinr_db,
                "user_interference_db": user_interference_db,
                "user_signal_db": user_signal_db,
                "user_weights": [float(weights.get(int(k), 1.0)) for k in active_users],
                "user_rate_gaps": [float(rate_gaps[int(k)]) for k in active_users],
                "max_precoder_delta": float(delta),
                "sum_rate": total_rate,
                INVERSE_CNR_WEIGHTED_SUM_RATE_PUBLIC_NAME: weighted_total,
                "weighted_sum_rate": weighted_total,
                "objective_mode": canonical_mode,
                "block_power": block_power,
                "block_power_budget": float(system.block_power_budget),
                "block_power_gap": block_power_gap,
                "block_power_violation": block_power_violation,
                "kkt_primal_residual": float(r_p),
                "kkt_complementarity_residual": float(r_c),
                "kkt_stationarity_residual": float(r_s),
            }
        )

        if objective_value >= best_objective:
            best_objective = float(objective_value)
            best_objective_state = _capture_active_block_solver_state(
                working_F,
                user_models,
                model_optimizers,
                active_users,
            )

        best_feasible_found = best_feasible_found or exact_feasible

        epoch_status = convergence_status_from_config(
            sim_params,
            precoder_change=r_s,
            has_previous_state=epoch_idx > 0,
            residuals=KktResiduals(r_p, r_c, r_s),
        )
        if epoch_status != "running":
            solve_status = epoch_status

        if verbose and (((epoch_idx + 1) % print_every) == 0 or epoch_idx == 0 or epoch_status != "running"):
            print(
                format_progress_log_line(
                    "[DL Convergence]",
                    phase="optimize",
                    block=int(block),
                    epoch=f"{epoch_idx + 1}/{max_epochs}",
                    active_users=int(len(active_users)),
                    updated_users=int(len(update_users)),
                    objective=float(objective_value),
                    sum_rate=float(total_rate),
                    r_p=float(r_p),
                    r_c=float(r_c),
                    r_s=float(r_s),
                    status=str(solve_status if solve_status != "max_epochs_reached" else "running"),
                )
            )

        if epoch_status != "running":
            break

    restored_state = best_objective_state
    _restore_active_block_solver_state(
        working_F,
        user_models,
        model_optimizers,
        active_users,
        restored_state,
    )
    if solve_status == "max_epochs_reached":
        solve_status = "max_epochs_best_objective"
    if len(history) > 0:
        history[-1]["solve_status"] = str(solve_status)
        history[-1]["solve_segment_end"] = True

    return {
        "history": history,
        "solve_status": solve_status,
        "best_feasible_found": bool(best_feasible_found),
    }
