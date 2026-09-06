"""Downlink Monte Carlo test-time allocation and evaluation."""

from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch

from latency_optimization.runtime import DEVICE

from ..block_state import (
    channels_for_block,
    clone_precoders,
    collect_interference_diagnostics,
    ensure_precoder_block,
    evaluate_block_candidate,
    expand_precoders_for_plan,
    maximum_supported_bits,
    power_to_db,
    user_link_budget,
    zero_precoder_block,
)

from .network_operations import (
    DownlinkSystem,
    STREAMING_MODE,
    PAYLOAD_MODE,
    _best_joint_n_target_transition,
    _build_block_joint_scenario,
    _build_monte_carlo_test_search_cfg,
    _copy_snapshot_with_block_overrides,
    _downlink_monte_carlo_precoder_parameterization,
    _masked_precoder_snapshot,
    _models_output_full_bs_precoder,
    validate_shared_bs_streaming_blocklength_input_mode,
    _scenario_metrics_with_models,
    _serialize_nested_history,
    _shared_n_targets_for_block,
    _shared_precoder_snapshot_for_targets,
    _zero_downlink_precoder,
    build_experiment_scenario,
    export_user_model_specs,
    format_latency_log_line,
    format_log_line,
    infer_raw_precoder_numpy_with_blocklength,
    model_outputs_full_bs_precoder,
    normalized_inverse_cnr_weights,
    objective_display_name,
    objective_weight_strategy_name,
    validate_convergence_objective_mode,
    validate_convergence_priority_weight_strategy,
    validate_downlink_precoder_net_scope,
    run_n_frontier_search,
    shared_estimate_initial_latency_from_random_precoders_for_scenario,
)


def _predict_user_precoder_for_blocklength(
    system: DownlinkSystem,
    model: torch.nn.Module,
    user: int,
    block: int,
    n_kl: int,
    active_mask: Sequence[int | float],
    input_precoders: list[list[np.ndarray]],
    inference_counters: dict[str, Any] | None = None,
) -> np.ndarray:
    k = int(user)
    l = int(block)
    if model_outputs_full_bs_precoder(model):
        snapshot = _shared_precoder_snapshot_for_targets(
            system,
            model,
            l,
            _shared_n_targets_for_block(
                system,
                active_mask,
                candidate_user=k,
                candidate_n_kl=int(n_kl),
            ),
            active_mask,
            inference_counters=inference_counters,
        )
        return np.asarray(snapshot[k][l], dtype=np.complex128)

    if inference_counters is not None:
        inference_counters["total_forward_calls"] = int(inference_counters.get("total_forward_calls", 0)) + 1
        per_user = inference_counters.get("per_user_forward_calls")
        if isinstance(per_user, list) and 0 <= k < len(per_user):
            per_user[k] = int(per_user[k]) + 1
    H_block = channels_for_block(system, l)
    input_noise_cov = system.get_interference_plus_noise_covariance(k, l, F_override=input_precoders)
    return infer_raw_precoder_numpy_with_blocklength(
        model,
        H_block,
        int(n_kl),
        active_mask,
        np.asarray(input_noise_cov, dtype=np.complex128),
        float(system.epsilon[k]),
        nb=int(system.Nb[k]),
        dk=int(system.dk[k]),
        device=DEVICE,
        user_index=int(k),
    )


def _allocate_streaming_bits_from_precoder_snapshot(
    system: DownlinkSystem,
    frozen_F: list[list[np.ndarray]],
    user: int,
    block: int,
    target_bits: int,
    sim_params: dict[str, Any],
    *,
    allow_infeasible_zero: bool = False,
    allow_n_reduction: bool = True,
) -> tuple[int, int, float, np.ndarray]:
    k = int(user)
    l = int(block)
    T_k = int(system.T[k])
    n_min = int(sim_params["n_kl_min"])
    n_step = int(sim_params["n_kl_step"])
    zero_beam = _zero_downlink_precoder(system, k)

    if int(target_bits) <= 0:
        return 0, int(T_k), 0.0, zero_beam

    F_fixed = np.asarray(frozen_F[k][l], dtype=np.complex128)
    snapshot = _copy_snapshot_with_block_overrides(
        frozen_F,
        int(l),
        {int(k): np.array(F_fixed, copy=True)},
    )
    R_T = float(system.compute_block_rate(k, l, T_k, F_override=snapshot))
    B_max = max(maximum_supported_bits(T_k, R_T), 0)
    B_used = int(min(int(target_bits), B_max))
    if int(B_used) <= 0 and allow_infeasible_zero:
        return 0, T_k, 0.0, zero_beam

    chosen_n = int(T_k)
    chosen_R = float(R_T)
    if allow_n_reduction and int(B_used) >= int(target_bits) and int(target_bits) > 0:
        candidate = T_k - n_step
        while candidate >= n_min:
            R_candidate = float(system.compute_block_rate(k, l, int(candidate), F_override=snapshot))
            if (float(target_bits) / float(max(int(candidate), 1))) <= R_candidate:
                chosen_n = int(candidate)
                chosen_R = float(R_candidate)
                candidate -= int(n_step)
            else:
                break
    return int(B_used), int(chosen_n), float(chosen_R), np.array(F_fixed, copy=True)


def _allocate_bits_for_user_block_precoder_net(
    system: DownlinkSystem,
    frozen_F: list[list[np.ndarray]],
    model: torch.nn.Module,
    user: int,
    block: int,
    remaining_bits: int,
    sim_params: dict[str, Any],
    active_mask: Sequence[int | float],
    *,
    allow_infeasible_zero: bool = False,
    inference_counters: dict[str, Any] | None = None,
) -> tuple[int, int, float, np.ndarray]:
    k = int(user)
    l = int(block)
    T_k = int(system.T[k])
    n_min = int(sim_params["n_kl_min"])
    n_step = int(sim_params["n_kl_step"])
    F_T = _predict_user_precoder_for_blocklength(
        system,
        model,
        k,
        l,
        T_k,
        active_mask,
        frozen_F,
        inference_counters=inference_counters,
    )
    snapshot_T = _copy_snapshot_with_block_overrides(
        frozen_F,
        int(l),
        {int(k): np.array(F_T, copy=True)},
    )
    active_users = [int(user_id) for user_id, flag in enumerate(active_mask) if float(flag) > 0.5]
    system.project_block_precoders_to_power(snapshot_T, l, active_users=active_users)
    R_T = float(system.compute_block_rate(k, l, T_k, F_override=snapshot_T))
    B_max = max(maximum_supported_bits(T_k, R_T), 0)
    if B_max <= 0:
        if allow_infeasible_zero:
            return 0, T_k, R_T, np.zeros_like(F_T)
        raise RuntimeError(
            f"Precoder-net user {k} block {l} infeasible at n=T={T_k}; R_T={R_T:.6f}, B_max={B_max}."
        )

    B_used = int(min(int(remaining_bits), B_max))
    chosen_n = int(T_k)
    chosen_R = float(R_T)
    chosen_F = np.array(snapshot_T[k][l], copy=True)

    if int(remaining_bits) <= B_max:
        search_cfg = _build_monte_carlo_test_search_cfg(
            sim_params,
            n_min=int(n_min),
            n_max=int(T_k),
        )

        def _evaluate_payload_candidate(candidate_n: int, _stage_name: str) -> dict[str, Any]:
            F_candidate = _predict_user_precoder_for_blocklength(
                system,
                model,
                k,
                l,
                int(candidate_n),
                active_mask,
                frozen_F,
                inference_counters=inference_counters,
            )
            candidate_snapshot = _copy_snapshot_with_block_overrides(
                frozen_F,
                int(l),
                {int(k): np.array(F_candidate, copy=True)},
            )
            system.project_block_precoders_to_power(candidate_snapshot, l, active_users=active_users)
            R_candidate = float(system.compute_block_rate(k, l, int(candidate_n), F_override=candidate_snapshot))
            return {
                "feasible": (float(B_used) / float(max(int(candidate_n), 1))) <= float(R_candidate),
                "R_candidate": float(R_candidate),
                "F_candidate": np.array(candidate_snapshot[k][l], copy=True),
            }

        search_result = run_n_frontier_search(search_cfg, _evaluate_payload_candidate)
        for accepted in search_result["accepted"]:
            chosen_n = int(accepted["n_kl"])
            chosen_R = float(accepted["result"]["R_candidate"])
            chosen_F = np.array(accepted["result"]["F_candidate"], copy=True)

    return int(B_used), int(chosen_n), float(chosen_R), chosen_F


def _allocate_streaming_bits_with_precoder_network(
    system: DownlinkSystem,
    frozen_F: list[list[np.ndarray]],
    model: torch.nn.Module,
    user: int,
    block: int,
    target_bits: int,
    sim_params: dict[str, Any],
    active_mask: Sequence[int | float],
    *,
    allow_infeasible_zero: bool = False,
    inference_counters: dict[str, Any] | None = None,
) -> tuple[int, int, float, np.ndarray]:
    k = int(user)
    l = int(block)
    T_k = int(system.T[k])
    n_min = int(sim_params["n_kl_min"])
    n_step = int(sim_params["n_kl_step"])
    zero_beam = _zero_downlink_precoder(system, k)

    if int(target_bits) <= 0:
        return 0, int(T_k), 0.0, zero_beam

    F_T = _predict_user_precoder_for_blocklength(
        system,
        model,
        k,
        l,
        T_k,
        active_mask,
        frozen_F,
        inference_counters=inference_counters,
    )
    snapshot_T = _copy_snapshot_with_block_overrides(
        frozen_F,
        int(l),
        {int(k): np.array(F_T, copy=True)},
    )
    active_users = [int(user_id) for user_id, flag in enumerate(active_mask) if float(flag) > 0.5]
    system.project_block_precoders_to_power(snapshot_T, l, active_users=active_users)
    R_T = float(system.compute_block_rate(k, l, T_k, F_override=snapshot_T))
    B_max = max(maximum_supported_bits(T_k, R_T), 0)
    B_used = int(min(int(target_bits), B_max))
    if int(B_used) <= 0 and allow_infeasible_zero:
        return 0, T_k, 0.0, zero_beam

    chosen_n = int(T_k)
    chosen_R = float(R_T)
    chosen_F = np.array(snapshot_T[k][l], copy=True)
    if int(B_used) >= int(target_bits) and int(target_bits) > 0:
        search_cfg = _build_monte_carlo_test_search_cfg(
            sim_params,
            n_min=int(n_min),
            n_max=int(T_k),
        )

        def _evaluate_streaming_candidate(candidate_n: int, _stage_name: str) -> dict[str, Any]:
            F_candidate = _predict_user_precoder_for_blocklength(
                system,
                model,
                k,
                l,
                int(candidate_n),
                active_mask,
                frozen_F,
                inference_counters=inference_counters,
            )
            candidate_snapshot = _copy_snapshot_with_block_overrides(
                frozen_F,
                int(l),
                {int(k): np.array(F_candidate, copy=True)},
            )
            system.project_block_precoders_to_power(candidate_snapshot, l, active_users=active_users)
            R_candidate = float(system.compute_block_rate(k, l, int(candidate_n), F_override=candidate_snapshot))
            return {
                "feasible": (float(target_bits) / float(max(int(candidate_n), 1))) <= float(R_candidate),
                "R_candidate": float(R_candidate),
                "F_candidate": np.array(candidate_snapshot[k][l], copy=True),
            }

        search_result = run_n_frontier_search(search_cfg, _evaluate_streaming_candidate)
        for accepted in search_result["accepted"]:
            chosen_n = int(accepted["n_kl"])
            chosen_R = float(accepted["result"]["R_candidate"])
            chosen_F = np.array(accepted["result"]["F_candidate"], copy=True)
    return int(B_used), int(chosen_n), float(chosen_R), chosen_F


def _reconcile_streaming_plans_after_joint_precoder_commit(
    system: DownlinkSystem,
    user_models: Sequence[torch.nn.Module],
    committed_snapshot: list[list[np.ndarray]],
    corrected_plans: dict[int, dict[str, Any]],
    block_targets: np.ndarray,
    block: int,
    sim_params: dict[str, Any],
    active_mask: Sequence[int | float],
    inference_counters: dict[str, Any] | None = None,
) -> None:
    l = int(block)
    active_users = [int(k) for k in range(system.K) if float(active_mask[int(k)]) > 0.5]
    if len(active_users) == 0:
        return

    for _ in range(3):
        changed = False
        system.project_block_precoders_to_power(committed_snapshot, l, active_users=active_users)

        for k in active_users:
            plan = corrected_plans[int(k)]
            target_bits = int(plan.get("target_bits", int(block_targets[int(k), l])))
            if bool(plan.get("skipped", False)) or target_bits <= 0:
                corrected_plans[int(k)] = {
                    **plan,
                    "B_used": 0,
                    "n_used": int(system.T[int(k)]),
                    "R_used": 0.0,
                    "F_used": np.zeros((int(system.Nb[int(k)]), int(system.dk[int(k)])), dtype=np.complex128),
                    "skipped": bool(target_bits > 0),
                    "target_bits": target_bits,
                }
                committed_snapshot[int(k)][l] = np.array(corrected_plans[int(k)]["F_used"], copy=True)
                continue

            n_used = int(plan["n_used"])
            actual_rate = float(system.compute_block_rate(int(k), l, max(int(n_used), 1), F_override=committed_snapshot))
            achievable_bits = max(maximum_supported_bits(n_used, actual_rate), 0)
            if int(n_used) < int(system.T[int(k)]) and achievable_bits < target_bits:
                B_fix, n_fix, R_fix, F_fix = _allocate_streaming_bits_with_precoder_network(
                    system,
                    committed_snapshot,
                    user_models[int(k)],
                    int(k),
                    l,
                    target_bits,
                    sim_params,
                    active_mask,
                    allow_infeasible_zero=True,
                    inference_counters=inference_counters,
                )
                next_plan = {
                    "B_used": int(B_fix),
                    "n_used": int(n_fix if B_fix > 0 else int(system.T[int(k)])),
                    "R_used": float(R_fix),
                    "F_used": np.array(F_fix, copy=True),
                    "skipped": bool(B_fix <= 0),
                    "target_bits": target_bits,
                }
                if (
                    int(next_plan["n_used"]) != int(plan["n_used"])
                    or bool(next_plan["skipped"]) != bool(plan.get("skipped", False))
                    or not np.allclose(np.asarray(next_plan["F_used"]), np.asarray(plan["F_used"]))
                ):
                    changed = True
                corrected_plans[int(k)] = next_plan
                committed_snapshot[int(k)][l] = np.array(F_fix, copy=True)
        if not changed:
            break

    system.project_block_precoders_to_power(committed_snapshot, l, active_users=active_users)

    for k in active_users:
        plan = corrected_plans[int(k)]
        target_bits = int(plan.get("target_bits", int(block_targets[int(k), l])))
        if bool(plan.get("skipped", False)) or target_bits <= 0:
            corrected_plans[int(k)] = {
                **plan,
                "B_used": 0,
                "n_used": int(system.T[int(k)]),
                "R_used": 0.0,
                "F_used": np.zeros((int(system.Nb[int(k)]), int(system.dk[int(k)])), dtype=np.complex128),
                "skipped": bool(target_bits > 0),
                "target_bits": target_bits,
            }
            committed_snapshot[int(k)][l] = np.array(corrected_plans[int(k)]["F_used"], copy=True)
            continue

        n_used = int(plan["n_used"])
        actual_rate = float(system.compute_block_rate(int(k), l, max(int(n_used), 1), F_override=committed_snapshot))
        achievable_bits = max(maximum_supported_bits(n_used, actual_rate), 0)
        B_final = int(min(target_bits, achievable_bits))
        corrected_plans[int(k)] = {
            **plan,
            "B_used": int(B_final),
            "n_used": int(n_used if B_final > 0 else int(system.T[int(k)])),
            "R_used": float(actual_rate if B_final > 0 else 0.0),
            "F_used": np.array(committed_snapshot[int(k)][l], copy=True) if B_final > 0 else np.zeros(
                (int(system.Nb[int(k)]), int(system.dk[int(k)])),
                dtype=np.complex128,
            ),
            "skipped": bool(target_bits > 0 and B_final <= 0),
            "target_bits": target_bits,
        }


def _reconcile_payload_plans_after_commit(
    system: DownlinkSystem,
    user_models: Sequence[torch.nn.Module],
    committed_snapshot: list[list[np.ndarray]],
    corrected_plans: dict[int, dict[str, Any]],
    remaining_bits_by_user: np.ndarray,
    block: int,
    sim_params: dict[str, Any],
    active_mask: Sequence[int | float],
    inference_counters: dict[str, Any] | None = None,
) -> None:
    l = int(block)
    active_users = [int(k) for k in range(system.K) if float(active_mask[int(k)]) > 0.5]
    if len(active_users) == 0:
        return

    for _ in range(3):
        changed = False
        system.project_block_precoders_to_power(committed_snapshot, l, active_users=active_users)

        for k in active_users:
            plan = corrected_plans[int(k)]
            remaining_bits = int(remaining_bits_by_user[int(k)])
            if bool(plan.get("skipped", False)) or remaining_bits <= 0:
                corrected_plans[int(k)] = {
                    **plan,
                    "B_used": 0,
                    "n_used": int(system.T[int(k)]),
                    "R_used": 0.0,
                    "F_used": np.zeros((int(system.Nb[int(k)]), int(system.dk[int(k)])), dtype=np.complex128),
                    "skipped": True,
                }
                committed_snapshot[int(k)][l] = np.array(corrected_plans[int(k)]["F_used"], copy=True)
                continue

            n_used = int(plan["n_used"])
            actual_rate = float(system.compute_block_rate(int(k), l, max(int(n_used), 1), F_override=committed_snapshot))
            achievable_bits = max(maximum_supported_bits(n_used, actual_rate), 0)
            if int(n_used) < int(system.T[int(k)]) and achievable_bits < remaining_bits:
                B_fix, n_fix, R_fix, F_fix = _allocate_bits_for_user_block_precoder_net(
                    system,
                    committed_snapshot,
                    user_models[int(k)],
                    int(k),
                    l,
                    remaining_bits,
                    sim_params,
                    active_mask,
                    allow_infeasible_zero=True,
                    inference_counters=inference_counters,
                )
                next_plan = {
                    "B_used": int(B_fix),
                    "n_used": int(n_fix),
                    "R_used": float(R_fix),
                    "F_used": np.array(F_fix, copy=True),
                    "skipped": bool(B_fix <= 0),
                }
                if (
                    int(next_plan["n_used"]) != int(plan["n_used"])
                    or bool(next_plan["skipped"]) != bool(plan.get("skipped", False))
                    or not np.allclose(np.asarray(next_plan["F_used"]), np.asarray(plan["F_used"]))
                ):
                    changed = True
                corrected_plans[int(k)] = next_plan
                committed_snapshot[int(k)][l] = np.array(F_fix, copy=True)
        if not changed:
            break

    system.project_block_precoders_to_power(committed_snapshot, l, active_users=active_users)

    for k in active_users:
        plan = corrected_plans[int(k)]
        remaining_bits = int(remaining_bits_by_user[int(k)])
        if bool(plan.get("skipped", False)) or remaining_bits <= 0:
            corrected_plans[int(k)] = {
                **plan,
                "B_used": 0,
                "n_used": int(system.T[int(k)]),
                "R_used": 0.0,
                "F_used": np.zeros((int(system.Nb[int(k)]), int(system.dk[int(k)])), dtype=np.complex128),
                "skipped": True,
            }
            committed_snapshot[int(k)][l] = np.array(corrected_plans[int(k)]["F_used"], copy=True)
            continue

        n_used = int(plan["n_used"])
        actual_rate = float(system.compute_block_rate(int(k), l, max(int(n_used), 1), F_override=committed_snapshot))
        achievable_bits = max(maximum_supported_bits(n_used, actual_rate), 0)
        B_final = int(min(remaining_bits, achievable_bits))
        corrected_plans[int(k)] = {
            **plan,
            "B_used": int(B_final),
            "n_used": int(n_used if B_final > 0 else int(system.T[int(k)])),
            "R_used": float(actual_rate if B_final > 0 else 0.0),
            "F_used": np.array(committed_snapshot[int(k)][l], copy=True) if B_final > 0 else np.zeros(
                (int(system.Nb[int(k)]), int(system.dk[int(k)])),
                dtype=np.complex128,
            ),
            "skipped": bool(B_final <= 0),
        }


def _finalize_streaming_plans_from_shared_bs_precoder(
    system: DownlinkSystem,
    committed_snapshot: list[list[np.ndarray]],
    block: int,
    target_bits_by_user: Sequence[int],
    n_targets: Sequence[int],
    active_mask: Sequence[int | float],
) -> dict[int, dict[str, Any]]:
    l = int(block)
    active_users = [int(k) for k, flag in enumerate(active_mask) if float(flag) > 0.5]

    for _ in range(max(system.K, 1)):
        system.project_block_precoders_to_power(committed_snapshot, l, active_users=active_users)
        changed = False
        next_active_users: list[int] = []
        for k in range(system.K):
            target_bits = int(target_bits_by_user[int(k)])
            if int(target_bits) <= 0 or float(active_mask[int(k)]) <= 0.5:
                if int(l) < len(committed_snapshot[int(k)]):
                    committed_snapshot[int(k)][l] = _zero_downlink_precoder(system, int(k))
                continue

            n_used = int(n_targets[int(k)])
            actual_rate = float(system.compute_block_rate(int(k), l, max(int(n_used), 1), F_override=committed_snapshot))
            achievable_bits = max(maximum_supported_bits(n_used, actual_rate), 0)
            if achievable_bits <= 0:
                if np.linalg.norm(np.asarray(committed_snapshot[int(k)][l])) > 0.0:
                    changed = True
                committed_snapshot[int(k)][l] = _zero_downlink_precoder(system, int(k))
                continue
            next_active_users.append(int(k))

        if not changed and next_active_users == active_users:
            break
        active_users = next_active_users

    system.project_block_precoders_to_power(committed_snapshot, l, active_users=active_users)
    final_plans: dict[int, dict[str, Any]] = {}
    for k in range(system.K):
        target_bits = int(target_bits_by_user[int(k)])
        if int(target_bits) <= 0 or float(active_mask[int(k)]) <= 0.5:
            final_plans[int(k)] = {
                "B_used": 0,
                "n_used": int(system.T[int(k)]),
                "R_used": 0.0,
                "F_used": _zero_downlink_precoder(system, int(k)),
                "skipped": False,
                "target_bits": int(target_bits),
            }
            if int(l) < len(committed_snapshot[int(k)]):
                committed_snapshot[int(k)][l] = np.array(final_plans[int(k)]["F_used"], copy=True)
            continue

        n_used = int(n_targets[int(k)])
        actual_rate = float(system.compute_block_rate(int(k), l, max(int(n_used), 1), F_override=committed_snapshot))
        achievable_bits = max(maximum_supported_bits(n_used, actual_rate), 0)
        B_final = int(min(int(target_bits), achievable_bits))
        if B_final <= 0:
            zero_beam = _zero_downlink_precoder(system, int(k))
            committed_snapshot[int(k)][l] = np.array(zero_beam, copy=True)
            final_plans[int(k)] = {
                "B_used": 0,
                "n_used": int(system.T[int(k)]),
                "R_used": 0.0,
                "F_used": zero_beam,
                "skipped": bool(target_bits > 0),
                "target_bits": int(target_bits),
            }
            continue

        final_plans[int(k)] = {
            "B_used": int(B_final),
            "n_used": int(n_used),
            "R_used": float(actual_rate),
            "F_used": np.array(committed_snapshot[int(k)][l], copy=True),
            "skipped": False,
            "target_bits": int(target_bits),
        }
    return final_plans


def _plan_streaming_block_with_joint_blocklength_vector(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    shared_model: torch.nn.Module,
    block: int,
    target_bits_by_user: Sequence[int],
    *,
    inference_counters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_mask = [1 if int(target_bits_by_user[int(k)]) > 0 else 0 for k in range(system.K)]
    initial_n_targets = [
        int(system.T[int(k)]) if int(active_mask[int(k)]) > 0 else 0
        for k in range(system.K)
    ]
    block_scenario = _build_block_joint_scenario(
        system,
        int(block),
        active_mask,
        scenario_mode=STREAMING_MODE,
    )
    anchor_bits = [int(max(int(target_bits_by_user[int(k)]), 0)) for k in range(system.K)]
    allocation_snapshot = _shared_precoder_snapshot_for_targets(
        system,
        shared_model,
        int(block),
        initial_n_targets,
        active_mask,
        inference_counters=inference_counters,
    )
    current_n_targets = [int(v) for v in initial_n_targets]
    initial_metrics = _scenario_metrics_with_models(
        system.sc,
        block_scenario,
        [shared_model],
        current_n_targets,
        anchor_bits=anchor_bits,
        inference_counters=inference_counters,
    )

    if bool(initial_metrics["feasible"]):
        while True:
            transition = _best_joint_n_target_transition(
                system.sc,
                block_scenario,
                [shared_model],
                current_n_targets,
                anchor_bits,
                n_min=int(sim_params["n_kl_min"]),
                n_step=int(sim_params["n_kl_step"]),
                inference_counters=inference_counters,
            )
            accepted = transition.get("accepted")
            if accepted is None:
                break
            current_n_targets = [int(v) for v in accepted["candidate_n_targets"]]

    committed_snapshot = _shared_precoder_snapshot_for_targets(
        system,
        shared_model,
        int(block),
        current_n_targets,
        active_mask,
        inference_counters=inference_counters,
    )
    final_plans = _finalize_streaming_plans_from_shared_bs_precoder(
        system,
        committed_snapshot,
        int(block),
        target_bits_by_user,
        current_n_targets,
        active_mask,
    )
    return {
        "active_mask": [int(v) for v in active_mask],
        "initial_n_targets": [int(v) for v in initial_n_targets],
        "final_n_targets": [int(v) for v in current_n_targets],
        "allocation_snapshot": allocation_snapshot,
        "committed_snapshot": committed_snapshot,
        "final_plans": final_plans,
        "initial_metrics": initial_metrics,
    }


def _plan_streaming_block_one_user_change_at_a_time(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    shared_model: torch.nn.Module,
    block: int,
    target_bits_by_user: Sequence[int],
    *,
    inference_counters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_mask = [1 if int(target_bits_by_user[int(k)]) > 0 else 0 for k in range(system.K)]
    initial_n_targets = [
        int(system.T[int(k)]) if int(active_mask[int(k)]) > 0 else 0
        for k in range(system.K)
    ]
    block_scenario = _build_block_joint_scenario(
        system,
        int(block),
        active_mask,
        scenario_mode=STREAMING_MODE,
    )
    anchor_bits = [int(max(int(target_bits_by_user[int(k)]), 0)) for k in range(system.K)]
    allocation_snapshot = _shared_precoder_snapshot_for_targets(
        system,
        shared_model,
        int(block),
        initial_n_targets,
        active_mask,
        inference_counters=inference_counters,
    )
    current_n_targets = [int(v) for v in initial_n_targets]
    initial_metrics = _scenario_metrics_with_models(
        system.sc,
        block_scenario,
        [shared_model],
        current_n_targets,
        anchor_bits=anchor_bits,
        inference_counters=inference_counters,
    )

    if bool(initial_metrics["feasible"]):
        while True:
            changed = False
            for k in range(system.K):
                if int(active_mask[int(k)]) <= 0:
                    continue
                candidate_n = int(current_n_targets[int(k)]) - int(sim_params["n_kl_step"])
                if candidate_n < int(sim_params["n_kl_min"]):
                    continue
                candidate_n_targets = [int(v) for v in current_n_targets]
                candidate_n_targets[int(k)] = int(candidate_n)
                metrics = _scenario_metrics_with_models(
                    system.sc,
                    block_scenario,
                    [shared_model],
                    candidate_n_targets,
                    anchor_bits=anchor_bits,
                    inference_counters=inference_counters,
                )
                if bool(metrics["feasible"]):
                    current_n_targets = [int(v) for v in candidate_n_targets]
                    changed = True
            if not changed:
                break

    committed_snapshot = _shared_precoder_snapshot_for_targets(
        system,
        shared_model,
        int(block),
        current_n_targets,
        active_mask,
        inference_counters=inference_counters,
    )
    final_plans = _finalize_streaming_plans_from_shared_bs_precoder(
        system,
        committed_snapshot,
        int(block),
        target_bits_by_user,
        current_n_targets,
        active_mask,
    )
    return {
        "active_mask": [int(v) for v in active_mask],
        "initial_n_targets": [int(v) for v in initial_n_targets],
        "final_n_targets": [int(v) for v in current_n_targets],
        "allocation_snapshot": allocation_snapshot,
        "committed_snapshot": committed_snapshot,
        "final_plans": final_plans,
        "initial_metrics": initial_metrics,
    }


def _estimate_streaming_latency_with_random_precoders(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    scenario: dict[str, Any],
    allow_n_reduction: bool = True,
) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    baseline_system = DownlinkSystem(system.sc, seed=system.seed)
    block_targets = np.asarray(scenario["streaming_bit_targets_by_block"], dtype=int)
    num_blocks = int(scenario["number_of_blocks"])
    n_plan: list[list[int]] = [[] for _ in range(baseline_system.K)]
    B_plan: list[list[int]] = [[] for _ in range(baseline_system.K)]
    R_plan: list[list[float]] = [[] for _ in range(baseline_system.K)]
    skipped_blocks_per_user = [0 for _ in range(baseline_system.K)]
    working_F = baseline_system.clone_precoders()

    for block in range(num_blocks):
        for k in range(baseline_system.K):
            ensure_precoder_block(baseline_system, working_F, k, block, use_previous_as_template=False)
            if int(block_targets[k, block]) > 0:
                working_F[k][block] = baseline_system.sample_precoder(k, block)
        active_users = [k for k in range(baseline_system.K) if int(block_targets[k, block]) > 0]
        baseline_system.project_block_precoders_to_power(
            working_F,
            block,
            active_users=[int(k) for k in active_users],
        )

        for k in range(baseline_system.K):
            target_bits = int(block_targets[k, block])

            B_used, n_used, R_used, F_used = _allocate_streaming_bits_from_precoder_snapshot(
                baseline_system,
                working_F,
                int(k),
                int(block),
                int(target_bits),
                sim_params,
                allow_infeasible_zero=True,
                allow_n_reduction=allow_n_reduction,
            )
            working_F[int(k)][int(block)] = np.array(F_used, copy=True)
            if B_used <= 0:
                skipped_blocks_per_user[int(k)] += 1
                n_plan[k].append(int(baseline_system.T[k]))
                B_plan[k].append(0)
                R_plan[k].append(float(R_used))
                continue

            n_plan[k].append(int(n_used))
            B_plan[k].append(int(B_used))
            R_plan[k].append(float(R_used))

    initial_F = expand_precoders_for_plan(baseline_system, working_F, n_plan)
    baseline_system.apply_solution(initial_F, n_plan)
    latency = baseline_system.latency.tolist()
    initial_plan = {
        "n_kl": n_plan,
        "B_kl": B_plan,
        "R_alloc": R_plan,
        "skipped_blocks_per_user": [int(v) for v in skipped_blocks_per_user],
        "scenario_mode": STREAMING_MODE,
        "scenario_block_targets": block_targets.tolist(),
        "blocks_per_user": [int(len(v)) for v in n_plan],
    }
    return latency, initial_plan, collect_interference_diagnostics(baseline_system)


def _estimate_initial_latency_from_random_precoders_for_scenario(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    scenario: dict[str, Any],
    allow_n_reduction: bool = True,
) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    try:
        return shared_estimate_initial_latency_from_random_precoders_for_scenario(
            system,
            sim_params,
            scenario,
            allow_n_reduction=allow_n_reduction,
        )
    except RuntimeError as error:
        # A random reference can be physically unable to deliver one bit in a
        # low-SNR, strict-FBL episode. Keep the test result, but do not turn a
        # non-completing reference into a fake latency improvement.
        return (
            [float("nan") for _ in range(int(system.K))],
            {
                "completed": False,
                "failure_reason": str(error),
                "remaining_bits": [int(value) for value in system.B],
                "n_kl": [[] for _ in range(int(system.K))],
                "B_kl": [[] for _ in range(int(system.K))],
                "R_alloc": [[] for _ in range(int(system.K))],
            },
            {},
        )


def _build_initial_baseline_reference(
    baseline_name: str,
    latency: Sequence[float],
    plan: dict[str, Any],
    *,
    snr_db: Sequence[float],
    sinr_db: Sequence[float],
) -> dict[str, Any]:
    return {
        "schedule_source": str(baseline_name),
        "completed": bool(plan.get("completed", True)),
        "failure_reason": str(plan.get("failure_reason", "")),
        "remaining_bits": [int(value) for value in plan.get("remaining_bits", [])],
        "latency": [float(v) for v in latency],
        "n_kl": [list(map(int, values)) for values in plan.get("n_kl", [])],
        "B_kl": [list(map(int, values)) for values in plan.get("B_kl", [])],
        "R_alloc": [list(map(float, values)) for values in plan.get("R_alloc", [])],
        "snr_db": [float(v) for v in snr_db],
        "sinr_db": [float(v) for v in sinr_db],
        "skipped_blocks_per_user": [int(v) for v in plan.get("skipped_blocks_per_user", [])],
    }


def _evaluate_downlink_precoder_network_for_streaming(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
    *,
    verbose: bool,
    method_name: str,
    scenario: dict[str, Any],
    precoder_net_training_history: dict[str, Any] | None,
    train_seeds: Sequence[int] | None,
    training_dataset_sizes: Sequence[int] | None,
) -> dict[str, Any]:
    initial_snr_db, initial_sinr_db = system.get_snr_sinr_db()
    initial_latency, initial_plan, initial_interference_diag = _estimate_initial_latency_from_random_precoders_for_scenario(
        system,
        sim_params,
        scenario,
    )
    naive_full_t_latency, naive_full_t_plan, _ = _estimate_initial_latency_from_random_precoders_for_scenario(
        system,
        sim_params,
        scenario,
        allow_n_reduction=False,
    )
    model_scope = validate_downlink_precoder_net_scope(sim_params.get("downlink_precoder_net_scope", "per_user_nets"))
    objective_mode = validate_convergence_objective_mode(sim_params)
    objective_public_name = objective_display_name(
        objective_mode,
        validate_convergence_priority_weight_strategy(sim_params),
    )
    weight_strategy_name = objective_weight_strategy_name(
        objective_mode,
        validate_convergence_priority_weight_strategy(sim_params),
    )
    streaming_blocklength_input_mode = validate_shared_bs_streaming_blocklength_input_mode(
        sim_params.get("shared_bs_streaming_blocklength_input_mode", "joint_blocklength_vector")
    )
    if verbose:
        print(
            format_latency_log_line(
                "[DL Initial Baseline]",
                initial_latency,
                seed=int(system.seed),
                scenario="streaming",
                method="monte_carlo",
            )
        )

    block_targets = np.asarray(scenario["streaming_bit_targets_by_block"], dtype=int)
    num_blocks = int(scenario["number_of_blocks"])
    n_plan: list[list[int]] = [[] for _ in range(system.K)]
    B_plan: list[list[int]] = [[] for _ in range(system.K)]
    R_plan: list[list[float]] = [[] for _ in range(system.K)]
    working_F = system.clone_precoders()
    epoch_history: list[dict[str, Any]] = []
    outer_history: list[dict[str, Any]] = []
    rate_points: list[dict[str, Any]] = []
    skipped_blocks_per_user = [0 for _ in range(system.K)]
    evaluation_cost_counters = {
        "per_user_forward_calls": [0 for _ in range(system.K)],
        "total_forward_calls": 0,
    }
    core_evaluation_start = perf_counter()

    for block in range(num_blocks):
        target_bits_by_user = [int(block_targets[int(k), block]) for k in range(system.K)]
        active_users = [int(k) for k in range(system.K) if int(target_bits_by_user[int(k)]) > 0]
        block_users = list(range(system.K))
        for k in block_users:
            ensure_precoder_block(system, working_F, k, block)
        active_mask = [1 if int(target_bits_by_user[int(k)]) > 0 else 0 for k in range(system.K)]
        if _models_output_full_bs_precoder(user_models):
            if streaming_blocklength_input_mode == "one_user_change_at_a_time":
                shared_plan = _plan_streaming_block_one_user_change_at_a_time(
                    system,
                    sim_params,
                    user_models[0],
                    int(block),
                    target_bits_by_user,
                    inference_counters=evaluation_cost_counters,
                )
            else:
                shared_plan = _plan_streaming_block_with_joint_blocklength_vector(
                    system,
                    sim_params,
                    user_models[0],
                    int(block),
                    target_bits_by_user,
                    inference_counters=evaluation_cost_counters,
                )
            allocation_snapshot = clone_precoders(shared_plan["allocation_snapshot"])
            committed_snapshot = clone_precoders(shared_plan["committed_snapshot"])
            corrected_plans = {
                int(k): {
                    **plan,
                    "F_used": np.array(plan["F_used"], copy=True),
                }
                for k, plan in shared_plan["final_plans"].items()
            }
        else:
            input_snapshot = _masked_precoder_snapshot(system, working_F, block, active_mask)
            for k in active_users:
                working_F[k][block] = _predict_user_precoder_for_blocklength(
                    system,
                    user_models[int(k)],
                    int(k),
                    int(block),
                    int(system.T[int(k)]),
                    active_mask,
                    input_snapshot,
                    inference_counters=evaluation_cost_counters,
                )
            system.project_block_precoders_to_power(working_F, int(block), active_users=[int(k) for k in active_users])
            for k in range(system.K):
                if int(block_targets[k, block]) <= 0 and int(block) < len(working_F[k]):
                    zero_precoder_block(system, working_F, k, block)
            system.project_block_precoders_to_power(
                working_F,
                int(block),
                active_users=[int(k) for k in active_users if int(block_targets[k, block]) > 0],
            )

        if verbose:
            print(
                format_log_line(
                    "[DL Monte Carlo Eval]",
                    block=int(block),
                    active_users=int(len(active_users)),
                    target_bits=int(np.sum(block_targets[:, block])),
                    mode="streaming",
                )
            )

        if not _models_output_full_bs_precoder(user_models):
            allocation_snapshot = clone_precoders(working_F)
            block_plans: dict[int, dict[str, Any]] = {}
            for k in range(system.K):
                target_bits = int(block_targets[k, block])
                B_used, n_used, R_used, F_used = _allocate_streaming_bits_with_precoder_network(
                    system,
                    allocation_snapshot,
                    user_models[int(k)],
                    int(k),
                    int(block),
                    int(target_bits),
                    sim_params,
                    active_mask,
                    allow_infeasible_zero=True,
                    inference_counters=evaluation_cost_counters,
                )
                block_plans[int(k)] = {
                    "B_used": int(B_used),
                    "n_used": int(n_used if B_used > 0 else int(system.T[int(k)])),
                    "R_used": float(R_used),
                    "F_used": np.array(F_used, copy=True),
                    "skipped": bool(target_bits > 0 and B_used <= 0),
                    "target_bits": int(target_bits),
                }

            committed_snapshot = clone_precoders(working_F)
            for k in range(system.K):
                committed_snapshot[int(k)][block] = np.array(block_plans[int(k)]["F_used"], copy=True)
            system.project_block_precoders_to_power(committed_snapshot, int(block), active_users=[int(k) for k in active_users])

            corrected_plans = {}
            for k in range(system.K):
                plan = block_plans[int(k)]
                if bool(plan["skipped"]):
                    corrected_plans[int(k)] = plan
                    continue

                B_used = int(plan["B_used"])
                n_used = int(plan["n_used"])
                F_used = np.array(plan["F_used"], copy=True)
                required_rate = float(B_used) / float(max(n_used, 1))
                actual_rate = float(system.compute_block_rate(int(k), int(block), n_used, F_override=committed_snapshot))
                if actual_rate >= required_rate:
                    corrected_plans[int(k)] = {
                        **plan,
                        "R_used": float(actual_rate),
                        "F_used": F_used,
                        "skipped": False,
                    }
                    continue

                B_fix, n_fix, R_fix, F_fix = _allocate_streaming_bits_with_precoder_network(
                    system,
                    committed_snapshot,
                    user_models[int(k)],
                    int(k),
                    int(block),
                    int(plan["target_bits"]),
                    sim_params,
                    active_mask,
                    allow_infeasible_zero=True,
                    inference_counters=evaluation_cost_counters,
                )
                corrected_plans[int(k)] = {
                    "B_used": int(B_fix),
                    "n_used": int(n_fix if B_fix > 0 else int(system.T[int(k)])),
                    "R_used": float(R_fix),
                    "F_used": np.array(F_fix, copy=True),
                    "skipped": bool(B_fix <= 0),
                    "target_bits": int(plan["target_bits"]),
                }

            for k in range(system.K):
                final_plan = corrected_plans[int(k)]
                committed_snapshot[int(k)][block] = np.array(final_plan["F_used"], copy=True)
            _reconcile_streaming_plans_after_joint_precoder_commit(
                system,
                user_models,
                committed_snapshot,
                corrected_plans,
                block_targets,
                int(block),
                sim_params,
                active_mask,
                inference_counters=evaluation_cost_counters,
            )

        user_rates = []
        user_sinr_db = []
        user_interference_db = []
        user_signal_db = []
        for k in active_users:
            rate = float(system.compute_block_rate(int(k), int(block), int(system.T[int(k)]), F_override=allocation_snapshot))
            signal_power, interference_power, _, sinr_db = user_link_budget(
                system,
                allocation_snapshot,
                int(k),
                int(block),
            )
            user_rates.append(rate)
            user_sinr_db.append(float(sinr_db))
            user_interference_db.append(power_to_db(interference_power))
            user_signal_db.append(power_to_db(signal_power))

        block_user_weights = normalized_inverse_cnr_weights(
            [np.asarray(system.H[k][int(block)]) for k in range(system.K)],
            system.sigma2,
            active_users,
        )
        weighted_sum_rate = float(
            sum(block_user_weights[int(k)] * rate for k, rate in zip(active_users, user_rates))
        )

        epoch_history.append(
            {
                "block": int(block),
                "epoch": 1,
                "active_users": int(len(active_users)),
                "user_ids": [int(k) for k in active_users],
                "user_rates": user_rates,
                "user_sinr_db": user_sinr_db,
                "user_interference_db": user_interference_db,
                "user_signal_db": user_signal_db,
                "inverse_cnr_weights": [block_user_weights[int(k)] for k in active_users],
                "max_precoder_delta": 0.0,
                "sum_rate": float(sum(user_rates)),
                "inverse_cnr_weighted_sum_rate": weighted_sum_rate,
                "objective_mode": str(objective_public_name),
            }
        )

        block_bits = 0
        block_unserved_bits = 0
        for k in range(system.K):
            final_plan = corrected_plans[int(k)]
            working_F[int(k)][block] = np.array(final_plan["F_used"], copy=True)
            B_used = int(final_plan["B_used"])
            n_used = int(final_plan["n_used"])
            R_used = float(
                system.compute_block_rate(int(k), int(block), max(int(n_used), 1), F_override=committed_snapshot)
            ) if B_used > 0 else 0.0

            n_plan[int(k)].append(int(n_used))
            B_plan[int(k)].append(int(B_used))
            R_plan[int(k)].append(float(R_used))
            block_bits += int(B_used)
            unserved_bits = max(int(final_plan.get("target_bits", B_used)) - int(B_used), 0)
            block_unserved_bits += int(unserved_bits)
            if bool(final_plan.get("skipped", False)):
                skipped_blocks_per_user[int(k)] += 1

            required_rate = float(B_used) / float(max(n_used, 1)) if B_used > 0 else 0.0
            rate_points.append(
                {
                    "user": int(k),
                    "block": int(block),
                    "n_kl": int(n_used),
                    "B_kl": int(B_used),
                    "target_bits": int(final_plan.get("target_bits", B_used)),
                    "unserved_bits": int(unserved_bits),
                    "required_rate": required_rate,
                    "achieved_rate": float(R_used),
                    "rate_margin": float(R_used) - required_rate,
                    "inverse_cnr_weight": float(block_user_weights.get(int(k), 0.0)),
                    "skipped": bool(final_plan.get("skipped", False)),
                    "partially_served": bool(0 < int(B_used) < int(final_plan.get("target_bits", B_used))),
                }
            )
            if verbose:
                if bool(final_plan.get("skipped", False)):
                    print(
                        format_log_line(
                            "[DL Monte Carlo Allocation]",
                            user=int(k),
                            block=int(block),
                            status="skipped",
                            target_bits=int(final_plan.get("target_bits", 0)),
                        )
                    )
                else:
                    print(
                        format_log_line(
                            "[DL Monte Carlo Allocation]",
                            user=int(k),
                            block=int(block),
                            target_bits=int(final_plan.get("target_bits", 0)),
                            served_bits=int(B_used),
                            unserved_bits=int(unserved_bits),
                            n_kl=int(n_used),
                            required_rate=float(required_rate),
                            achieved_rate=float(R_used),
                        )
                    )

        outer_history.append(
            {
                "block": int(block),
                "active_users": int(len(active_users)),
                "transmitting_users": int(sum(1 for k in active_users if corrected_plans[int(k)]["B_used"] > 0)),
                "skipped_users": int(sum(1 for k in active_users if corrected_plans[int(k)]["B_used"] <= 0)),
                "allocated_bits": int(block_bits),
                "target_bits": int(np.sum(block_targets[:, block])),
                "unserved_bits": int(block_unserved_bits),
                "future_target_bits": int(max(np.sum(block_targets[:, block + 1:]), 0)) if block + 1 < num_blocks else 0,
                "remaining_bits": int(max(np.sum(block_targets[:, block + 1:]), 0)) if block + 1 < num_blocks else 0,
                "feasible_users": int(sum(1 for k in active_users if corrected_plans[int(k)]["B_used"] > 0)),
                "min_max_bits": int(min([corrected_plans[int(k)]["B_used"] for k in active_users], default=0)),
                "inverse_cnr_weights": {
                    int(k): float(block_user_weights[int(k)]) for k in active_users
                },
                "final_precoder_delta": 0.0,
            }
        )

    final_F = expand_precoders_for_plan(system, working_F, n_plan)
    system.apply_solution(final_F, n_plan)

    final_snr_db, final_sinr_db = system.get_snr_sinr_db()
    final_interference_diag = collect_interference_diagnostics(system)

    result = {
        "method_name": method_name,
        "objective_mode": str(objective_public_name),
        "allocation_mode": "streaming",
        "weight_strategy": str(weight_strategy_name),
        "precoder_parameterization": _downlink_monte_carlo_precoder_parameterization(
            model_scope,
        ),
        "downlink_precoder_net_scope": str(model_scope),
        "shared_bs_streaming_blocklength_input_mode": (
            str(streaming_blocklength_input_mode)
            if str(model_scope) == "bs_shared_net"
            else "not_applicable"
        ),
        "user_model_specs": export_user_model_specs(
            system.Nr,
            system.Nb,
            system.dk,
            uses_blocklength_input=True,
            context_k=system.K,
            context_max_nr=int(np.max(system.Nr)),
            context_max_nb=int(np.max(system.Nb)),
            context_max_dk=int(np.max(system.dk)),
            model_scope=model_scope,
        ),
        "n_kl": [list(map(int, v)) for v in n_plan],
        "B_kl": [list(map(int, v)) for v in B_plan],
        "R_fbl": [list(map(float, user_rates)) for user_rates in system.R_fbl],
        "R_alloc": [list(map(float, v)) for v in R_plan],
        "initial_latency": list(map(float, initial_latency)),
        "initial_plan": initial_plan,
        "initial_baseline_completed": bool(initial_plan.get("completed", True)),
        "initial_baseline_failure": str(initial_plan.get("failure_reason", "")),
        "initial_schedule_source": "random_precoder_baseline",
        "baseline_references": {
            "random_precoder_baseline": _build_initial_baseline_reference(
                "random_precoder_baseline",
                initial_latency,
                initial_plan,
                snr_db=initial_snr_db,
                sinr_db=initial_sinr_db,
            ),
            "naive_full_T_baseline": _build_initial_baseline_reference(
                "naive_full_T_baseline",
                naive_full_t_latency,
                naive_full_t_plan,
                snr_db=initial_snr_db,
                sinr_db=initial_sinr_db,
            ),
        },
        "initial_interference_diag": initial_interference_diag,
        "final_latency": system.latency.tolist(),
        "initial_snr_db": initial_snr_db,
        "final_snr_db": final_snr_db,
        "initial_sinr_db": initial_sinr_db,
        "final_sinr_db": final_sinr_db,
        "final_interference_diag": final_interference_diag,
        "outer_history": outer_history,
        "epoch_history": epoch_history,
        "rate_points": rate_points,
        "blocks_per_user": [len(v) for v in n_plan],
        "precoder_net_training_losses": [
            list(map(float, row))
            for row in ((precoder_net_training_history or {}).get("per_user_objective_loss", []))
        ],
        "precoder_net_training_history": _serialize_nested_history(precoder_net_training_history or {}),
        "train_seeds": [int(v) for v in (train_seeds or [])],
        "training_dataset_sizes": [int(v) for v in (training_dataset_sizes or [])],
        "training_active_user_case_counts_per_user": [int(v) for v in (training_dataset_sizes or [])],
        "skipped_blocks_per_user": [int(v) for v in skipped_blocks_per_user],
        "evaluation_cost_counters": evaluation_cost_counters,
        "core_evaluation_wall_time_seconds": float(perf_counter() - core_evaluation_start),
        "scenario_mode": STREAMING_MODE,
        "scenario_block_targets": block_targets.tolist(),
    }
    return result


def evaluate_downlink_precoder_net(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
    *,
    verbose: bool = True,
    method_name: str = "monte_carlo_precoder_net_train_test",
    precoder_net_training_history: dict[str, Any] | None = None,
    train_seeds: Sequence[int] | None = None,
    training_dataset_sizes: Sequence[int] | None = None,
) -> dict[str, Any]:
    scenario = build_experiment_scenario(system.sc, sim_params, seed=int(system.seed))
    if str(scenario["mode"]) == STREAMING_MODE:
        return _evaluate_downlink_precoder_network_for_streaming(
            system,
            sim_params,
            user_models,
            verbose=verbose,
            method_name=method_name,
            scenario=scenario,
            precoder_net_training_history=precoder_net_training_history,
            train_seeds=train_seeds,
            training_dataset_sizes=training_dataset_sizes,
        )
    initial_snr_db, initial_sinr_db = system.get_snr_sinr_db()
    initial_latency, initial_plan, initial_interference_diag = _estimate_initial_latency_from_random_precoders_for_scenario(
        system,
        sim_params,
        scenario,
    )
    naive_full_t_latency, naive_full_t_plan, _ = _estimate_initial_latency_from_random_precoders_for_scenario(
        system,
        sim_params,
        scenario,
        allow_n_reduction=False,
    )
    objective_mode = validate_convergence_objective_mode(sim_params)
    objective_public_name = objective_display_name(
        objective_mode,
        validate_convergence_priority_weight_strategy(sim_params),
    )
    weight_strategy_name = objective_weight_strategy_name(
        objective_mode,
        validate_convergence_priority_weight_strategy(sim_params),
    )
    if verbose:
        print(
            format_latency_log_line(
                "[DL Initial Baseline]",
                initial_latency,
                seed=int(system.seed),
                scenario="payload",
                method="monte_carlo",
            )
        )

    remaining = np.asarray(system.B, dtype=int).copy()
    n_plan: list[list[int]] = [[] for _ in range(system.K)]
    B_plan: list[list[int]] = [[] for _ in range(system.K)]
    R_plan: list[list[float]] = [[] for _ in range(system.K)]
    working_F = system.clone_precoders()
    epoch_history: list[dict[str, Any]] = []
    outer_history: list[dict[str, Any]] = []
    rate_points: list[dict[str, Any]] = []
    evaluation_cost_counters = {
        "per_user_forward_calls": [0 for _ in range(system.K)],
        "total_forward_calls": 0,
    }
    max_blocks = int(sim_params.get("max_total_blocks", 256))
    core_evaluation_start = perf_counter()
    block = 0
    while np.any(remaining > 0):
        if block >= max_blocks:
            raise RuntimeError(
                f"Precoder-net evaluation hit max_total_blocks={max_blocks} with remaining bits {remaining.tolist()}."
            )

        active_users = [k for k in range(system.K) if int(remaining[k]) > 0]
        # Inverse-CNR diagnostics need a channel entry for every user at this
        # block, even when only a subset still has remaining bits.
        for k in range(system.K):
            ensure_precoder_block(system, working_F, k, block)
        active_mask = [1 if k in active_users else 0 for k in range(system.K)]
        if _models_output_full_bs_precoder(user_models):
            joint_snapshot = _shared_precoder_snapshot_for_targets(
                system,
                user_models[0],
                int(block),
                _shared_n_targets_for_block(system, active_mask),
                active_mask,
                base_snapshot=working_F,
                inference_counters=evaluation_cost_counters,
            )
            for k in active_users:
                working_F[int(k)][int(block)] = np.asarray(joint_snapshot[int(k)][int(block)], dtype=np.complex128)
        else:
            input_snapshot = _masked_precoder_snapshot(system, working_F, block, active_mask)
            for k in active_users:
                working_F[k][block] = _predict_user_precoder_for_blocklength(
                    system,
                    user_models[int(k)],
                    int(k),
                    int(block),
                    int(system.T[int(k)]),
                    active_mask,
                    input_snapshot,
                    inference_counters=evaluation_cost_counters,
                )
        system.project_block_precoders_to_power(working_F, int(block), active_users=[int(k) for k in active_users])

        if verbose:
            print(
                format_log_line(
                    "[DL Monte Carlo Eval]",
                    block=int(block),
                    active_users=int(len(active_users)),
                    remaining_bits=int(np.sum(remaining)),
                    mode="payload",
                )
            )

        transmit_users = list(active_users)
        skipped_users: list[int] = []
        skipped_user_rates: dict[int, float] = {}
        while len(transmit_users) > 0:
            current_eval = evaluate_block_candidate(system, working_F, transmit_users, block)
            infeasible_users = [
                int(user_id)
                for user_id, max_bits in zip(current_eval["user_ids"], current_eval["user_max_bits"])
                if int(max_bits) <= 0
            ]
            if len(infeasible_users) == 0:
                break
            current_eval_rates = {
                int(user_id): float(rate_val)
                for user_id, rate_val in zip(current_eval["user_ids"], current_eval["user_rates"])
            }
            for k in infeasible_users:
                skipped_user_rates[int(k)] = float(current_eval_rates.get(int(k), 0.0))
                zero_precoder_block(system, working_F, k, block)
                skipped_users.append(int(k))
            transmit_users = [k for k in transmit_users if int(k) not in infeasible_users]
            if verbose:
                print(
                    format_log_line(
                        "[DL Monte Carlo Eval]",
                        block=int(block),
                        skipped_users=[int(k) for k in infeasible_users],
                    )
                )

        allocation_snapshot = clone_precoders(working_F)
        transmit_mask = [1 if k in transmit_users else 0 for k in range(system.K)]
        block_plans: dict[int, dict[str, Any]] = {}
        for k in active_users:
            if int(k) in skipped_users:
                block_plans[int(k)] = {
                    "B_used": 0,
                    "n_used": int(system.T[int(k)]),
                    "R_used": float(skipped_user_rates.get(int(k), 0.0)),
                    "F_used": np.zeros((int(system.Nb[int(k)]), int(system.dk[int(k)])), dtype=np.complex128),
                    "skipped": True,
                }
                continue

            B_used, n_used, R_used, F_used = _allocate_bits_for_user_block_precoder_net(
                system,
                allocation_snapshot,
                user_models[int(k)],
                int(k),
                int(block),
                int(remaining[int(k)]),
                sim_params,
                transmit_mask,
                allow_infeasible_zero=True,
                inference_counters=evaluation_cost_counters,
            )
            block_plans[int(k)] = {
                "B_used": int(B_used),
                "n_used": int(n_used),
                "R_used": float(R_used),
                "F_used": np.array(F_used, copy=True),
                "skipped": bool(B_used <= 0),
            }

        committed_snapshot = clone_precoders(working_F)
        for k in active_users:
            committed_snapshot[int(k)][block] = np.array(block_plans[int(k)]["F_used"], copy=True)

        corrected_plans: dict[int, dict[str, Any]] = {}
        for k in active_users:
            plan = block_plans[int(k)]
            if bool(plan["skipped"]):
                corrected_plans[int(k)] = plan
                continue

            B_used = int(plan["B_used"])
            n_used = int(plan["n_used"])
            F_used = np.array(plan["F_used"], copy=True)
            required_rate = float(B_used) / float(max(n_used, 1))
            actual_rate = float(system.compute_block_rate(int(k), int(block), n_used, F_override=committed_snapshot))
            if actual_rate >= required_rate:
                corrected_plans[int(k)] = {
                    "B_used": B_used,
                    "n_used": n_used,
                    "R_used": actual_rate,
                    "F_used": F_used,
                    "skipped": False,
                }
                continue

            B_fix, n_fix, R_fix, F_fix = _allocate_bits_for_user_block_precoder_net(
                system,
                committed_snapshot,
                user_models[int(k)],
                int(k),
                int(block),
                int(remaining[int(k)]),
                sim_params,
                transmit_mask,
                allow_infeasible_zero=True,
                inference_counters=evaluation_cost_counters,
            )
            corrected_plans[int(k)] = {
                "B_used": int(B_fix),
                "n_used": int(n_fix),
                "R_used": float(R_fix),
                "F_used": np.array(F_fix, copy=True),
                "skipped": bool(B_fix <= 0),
            }

        for k in active_users:
            final_plan = corrected_plans[int(k)]
            committed_snapshot[int(k)][block] = np.array(final_plan["F_used"], copy=True)
        _reconcile_payload_plans_after_commit(
            system,
            user_models,
            committed_snapshot,
            corrected_plans,
            remaining.copy(),
            int(block),
            sim_params,
            transmit_mask,
            inference_counters=evaluation_cost_counters,
        )

        user_rates = []
        user_sinr_db = []
        user_interference_db = []
        user_signal_db = []
        for k in active_users:
            rate = float(system.compute_block_rate(int(k), int(block), int(system.T[int(k)]), F_override=allocation_snapshot))
            signal_power, interference_power, _, sinr_db = user_link_budget(
                system, allocation_snapshot, int(k), int(block)
            )
            user_rates.append(rate)
            user_sinr_db.append(float(sinr_db))
            user_interference_db.append(power_to_db(interference_power))
            user_signal_db.append(power_to_db(signal_power))

        block_user_weights = normalized_inverse_cnr_weights(
            [np.asarray(system.H[k][int(block)]) for k in range(system.K)],
            system.sigma2,
            active_users,
        )
        weighted_sum_rate = float(
            sum(block_user_weights[int(k)] * rate for k, rate in zip(active_users, user_rates))
        )

        epoch_history.append(
            {
                "block": int(block),
                "epoch": 1,
                "active_users": int(len(active_users)),
                "user_ids": [int(k) for k in active_users],
                "user_rates": user_rates,
                "user_sinr_db": user_sinr_db,
                "user_interference_db": user_interference_db,
                "user_signal_db": user_signal_db,
                "inverse_cnr_weights": [block_user_weights[int(k)] for k in active_users],
                "max_precoder_delta": 0.0,
                "sum_rate": float(sum(user_rates)),
                "inverse_cnr_weighted_sum_rate": weighted_sum_rate,
                "objective_mode": str(objective_public_name),
            }
        )

        block_bits = 0
        for k in active_users:
            final_plan = corrected_plans[int(k)]
            working_F[int(k)][block] = np.array(final_plan["F_used"], copy=True)
            B_used = int(final_plan["B_used"])
            n_used = int(final_plan["n_used"])
            R_used = float(system.compute_block_rate(int(k), int(block), n_used, F_override=committed_snapshot))

            B_plan[int(k)].append(int(B_used))
            n_plan[int(k)].append(int(n_used))
            R_plan[int(k)].append(float(R_used))
            remaining[int(k)] -= int(B_used)
            block_bits += int(B_used)

            required_rate = float(B_used) / float(max(n_used, 1))
            rate_points.append(
                {
                    "user": int(k),
                    "block": int(block),
                    "n_kl": int(n_used),
                    "B_kl": int(B_used),
                    "required_rate": required_rate,
                    "achieved_rate": float(R_used),
                    "rate_margin": float(R_used) - required_rate,
                    "inverse_cnr_weight": float(block_user_weights[int(k)]),
                    "skipped": bool(B_used <= 0),
                }
            )
            if verbose:
                if B_used <= 0:
                    print(
                        format_log_line(
                            "[DL Monte Carlo Allocation]",
                            user=int(k),
                            block=int(block),
                            status="skipped",
                        )
                    )
                else:
                    print(
                        format_log_line(
                            "[DL Monte Carlo Allocation]",
                            user=int(k),
                            block=int(block),
                            served_bits=int(B_used),
                            n_kl=int(n_used),
                            required_rate=float(required_rate),
                            achieved_rate=float(R_used),
                        )
                    )

        outer_history.append(
            {
                "block": int(block),
                "active_users": int(len(active_users)),
                "transmitting_users": int(sum(1 for k in active_users if corrected_plans[int(k)]["B_used"] > 0)),
                "skipped_users": int(sum(1 for k in active_users if corrected_plans[int(k)]["B_used"] <= 0)),
                "allocated_bits": int(block_bits),
                "remaining_bits": int(np.sum(remaining)),
                "feasible_users": int(sum(1 for k in active_users if corrected_plans[int(k)]["B_used"] > 0)),
                "min_max_bits": int(min([corrected_plans[int(k)]["B_used"] for k in active_users], default=0)),
                "inverse_cnr_weights": {
                    int(k): float(block_user_weights[int(k)]) for k in active_users
                },
                "final_precoder_delta": 0.0,
            }
        )
        if verbose:
            print(
                format_log_line(
                    "[DL Monte Carlo Eval]",
                    block=int(block),
                    status="complete",
                    allocated_bits=int(block_bits),
                    remaining_bits=int(np.sum(remaining)),
                )
            )
        block += 1

    final_F = expand_precoders_for_plan(system, working_F, n_plan)
    system.apply_solution(final_F, n_plan)

    final_snr_db, final_sinr_db = system.get_snr_sinr_db()
    final_interference_diag = collect_interference_diagnostics(system)
    model_scope = validate_downlink_precoder_net_scope(sim_params.get("downlink_precoder_net_scope", "per_user_nets"))

    result = {
        "method_name": method_name,
        "objective_mode": str(objective_public_name),
        "allocation_mode": "greedy",
        "weight_strategy": str(weight_strategy_name),
        "precoder_parameterization": _downlink_monte_carlo_precoder_parameterization(
            model_scope,
        ),
        "downlink_precoder_net_scope": str(model_scope),
        "shared_bs_streaming_blocklength_input_mode": (
            validate_shared_bs_streaming_blocklength_input_mode(
                sim_params.get("shared_bs_streaming_blocklength_input_mode", "joint_blocklength_vector")
            )
            if str(model_scope) == "bs_shared_net"
            else "not_applicable"
        ),
        "user_model_specs": export_user_model_specs(
            system.Nr,
            system.Nb,
            system.dk,
            uses_blocklength_input=True,
            context_k=system.K,
            context_max_nr=int(np.max(system.Nr)),
            context_max_nb=int(np.max(system.Nb)),
            context_max_dk=int(np.max(system.dk)),
            model_scope=model_scope,
        ),
        "n_kl": [list(map(int, v)) for v in n_plan],
        "B_kl": [list(map(int, v)) for v in B_plan],
        "R_fbl": [list(map(float, user_rates)) for user_rates in system.R_fbl],
        "R_alloc": [list(map(float, v)) for v in R_plan],
        "initial_latency": list(map(float, initial_latency)),
        "initial_plan": initial_plan,
        "initial_baseline_completed": bool(initial_plan.get("completed", True)),
        "initial_baseline_failure": str(initial_plan.get("failure_reason", "")),
        "initial_schedule_source": "random_precoder_baseline",
        "baseline_references": {
            "random_precoder_baseline": _build_initial_baseline_reference(
                "random_precoder_baseline",
                initial_latency,
                initial_plan,
                snr_db=initial_snr_db,
                sinr_db=initial_sinr_db,
            ),
            "naive_full_T_baseline": _build_initial_baseline_reference(
                "naive_full_T_baseline",
                naive_full_t_latency,
                naive_full_t_plan,
                snr_db=initial_snr_db,
                sinr_db=initial_sinr_db,
            ),
        },
        "initial_interference_diag": initial_interference_diag,
        "final_latency": system.latency.tolist(),
        "initial_snr_db": initial_snr_db,
        "final_snr_db": final_snr_db,
        "initial_sinr_db": initial_sinr_db,
        "final_sinr_db": final_sinr_db,
        "final_interference_diag": final_interference_diag,
        "outer_history": outer_history,
        "epoch_history": epoch_history,
        "rate_points": rate_points,
        "blocks_per_user": [len(v) for v in n_plan],
        "precoder_net_training_losses": [
            list(map(float, row))
            for row in ((precoder_net_training_history or {}).get("per_user_objective_loss", []))
        ],
        "precoder_net_training_history": _serialize_nested_history(precoder_net_training_history or {}),
        "train_seeds": [int(v) for v in (train_seeds or [])],
        "training_dataset_sizes": [int(v) for v in (training_dataset_sizes or [])],
        "training_active_user_case_counts_per_user": [int(v) for v in (training_dataset_sizes or [])],
        "skipped_blocks_per_user": [0 for _ in range(system.K)],
        "evaluation_cost_counters": evaluation_cost_counters,
        "core_evaluation_wall_time_seconds": float(perf_counter() - core_evaluation_start),
        "scenario_mode": PAYLOAD_MODE,
    }
    return result


__all__ = ["evaluate_downlink_precoder_net"]
