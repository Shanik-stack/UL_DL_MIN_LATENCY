"""Uplink Monte Carlo test-time allocation and evaluation."""

from typing import Any, Sequence

import numpy as np
import torch

from latency_optimization.runtime import DEVICE

from .network_operations import (
    STREAMING_MODE,
    PAYLOAD_MODE,
    UplinkSystem,
    _build_monte_carlo_test_search_cfg,
    _build_precoder_net_snapshot_for_active_mask,
    _compute_r_fbl_np,
    _zero_uplink_precoder,
    apply_training_solution,
    build_experiment_scenario,
    build_uplink_rate_covariance,
    clone_nested_arrays,
    collect_uplink_interference_diagnostics,
    ensure_blocks_up_to,
    format_log_line,
    infer_precoder_numpy_with_blocklength_and_sigma,
    run_n_frontier_search,
    shared_estimate_initial_random_precoder_schedule_for_scenario,
    uses_uplink_interference,
)
from .rollout import (
    _count_uplink_forward_call,
    _ensure_precoder_net_snapshot_block,
    _replace_snapshot_block,
)


def _estimate_initial_random_precoder_schedule_for_streaming(
    system_params: dict[str, Any],
    sim_cfg: dict[str, Any],
    *,
    seed: int,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    baseline_system = UplinkSystem(system_params, seed=int(seed))
    K = int(baseline_system.K)
    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])
    block_targets = np.asarray(scenario["streaming_bit_targets_by_block"], dtype=int)
    num_blocks = int(scenario["number_of_blocks"])

    initial_n_kl: list[list[int]] = [[] for _ in range(K)]
    initial_B_kl: list[list[int]] = [[] for _ in range(K)]
    initial_R_fbl: list[list[float]] = [[] for _ in range(K)]
    initial_F: list[list[np.ndarray]] = [[] for _ in range(K)]
    skipped_blocks_per_user = [0 for _ in range(K)]

    for block in range(num_blocks):
        ensure_blocks_up_to(baseline_system, block)
        random_snapshot = clone_nested_arrays(baseline_system.F)

        for k in range(K):
            target_bits = int(block_targets[k, block])
            H_kl = np.asarray(baseline_system.H[k][block], dtype=np.complex64)
            F_kl = np.asarray(random_snapshot[k][block], dtype=np.complex64)
            T_ref = int(baseline_system.T[k])
            sigma2 = float(baseline_system.sigma2[k])
            epsilon = float(baseline_system.epsilon[k])
            noise_plus_interference_cov = build_uplink_rate_covariance(
                baseline_system,
                sim_cfg,
                k,
                block,
                F_override=random_snapshot,
            )
            R_T = _compute_r_fbl_np(
                H_kl,
                F_kl,
                sigma2,
                epsilon,
                T_ref,
                noise_plus_interference_cov,
            )
            B_max = max(int(np.floor(float(T_ref) * float(R_T))), 0)
            B_used = int(min(target_bits, B_max))
            best_n = int(T_ref)
            best_R = float(R_T)
            if int(B_used) >= int(target_bits) and int(target_bits) > 0:
                candidate_n = int(T_ref) - int(n_kl_step)
                while candidate_n >= int(n_kl_min):
                    R_candidate = _compute_r_fbl_np(
                        H_kl,
                        F_kl,
                        sigma2,
                        epsilon,
                        candidate_n,
                        noise_plus_interference_cov,
                    )
                    if (float(target_bits) / float(max(candidate_n, 1))) <= R_candidate:
                        best_n = int(candidate_n)
                        best_R = float(R_candidate)
                        candidate_n -= int(n_kl_step)
                    else:
                        break

            initial_n_kl[k].append(int(best_n))
            initial_B_kl[k].append(int(B_used))
            initial_R_fbl[k].append(float(best_R))
            initial_F[k].append(
                np.array(F_kl, copy=True) if int(B_used) > 0 else _zero_uplink_precoder(baseline_system, k)
            )
            if int(B_used) <= 0:
                skipped_blocks_per_user[k] += 1

    initial_n = [int(sum(int(max(v, 0)) for v in user_n)) for user_n in initial_n_kl]
    initial_latency = [
        float(initial_n[k]) / float(max(float(baseline_system.fs[k]), 1e-30))
        for k in range(K)
    ]
    initial_bits_per_symbol_by_block = []
    initial_bits_per_symbol = []
    for k in range(K):
        user_bps = [
            float(bits) / float(max(int(n_kl), 1))
            if int(n_kl) > 0 and int(bits) > 0
            else 0.0
            for bits, n_kl in zip(initial_B_kl[k], initial_n_kl[k])
        ]
        total_n = float(max(initial_n[k], 1))
        initial_bits_per_symbol_by_block.append(user_bps)
        initial_bits_per_symbol.append(float(sum(initial_B_kl[k])) / total_n if initial_n[k] > 0 else 0.0)

    apply_training_solution(baseline_system, initial_n_kl, initial_F)
    _, initial_snr_db = baseline_system.get_SNR()
    _, initial_sinr_db = baseline_system.get_SINR()
    initial_interference_diag = collect_uplink_interference_diagnostics(baseline_system)

    return {
        "initial_n_kl": initial_n_kl,
        "initial_B_kl": initial_B_kl,
        "initial_R_fbl": initial_R_fbl,
        "initial_n": initial_n,
        "initial_latency": [float(v) for v in baseline_system.latency],
        "initial_snr_db": list(map(float, initial_snr_db)),
        "initial_sinr_db": list(map(float, initial_sinr_db)),
        "initial_bits_per_symbol": initial_bits_per_symbol,
        "initial_bits_per_symbol_by_block": initial_bits_per_symbol_by_block,
        "initial_interference_diag": initial_interference_diag,
        "skipped_blocks_per_user": [int(v) for v in skipped_blocks_per_user],
        "scenario_mode": STREAMING_MODE,
        "scenario_block_targets": block_targets.tolist(),
    }


def estimate_initial_random_precoder_schedule_for_scenario(
    system_params: dict[str, Any],
    sim_cfg: dict[str, Any],
    *,
    seed: int,
    allow_n_reduction: bool = True,
) -> dict[str, Any]:
    return shared_estimate_initial_random_precoder_schedule_for_scenario(
        system_params,
        sim_cfg,
        seed=int(seed),
        allow_n_reduction=allow_n_reduction,
    )


def _evaluate_precoder_network_for_streaming(
    uplinksystem: UplinkSystem,
    user_models: Sequence[torch.nn.Module],
    sim_cfg: dict,
    *,
    method_name: str,
) -> dict:
    scenario = build_experiment_scenario(uplinksystem.sc, sim_cfg, seed=int(uplinksystem.seed))
    block_targets = np.asarray(scenario["streaming_bit_targets_by_block"], dtype=int)
    num_blocks = int(scenario["number_of_blocks"])
    K = int(uplinksystem.K)

    n_star = [[] for _ in range(K)]
    F_star = [[] for _ in range(K)]
    R_star = [[] for _ in range(K)]
    all_user_block_results = [[] for _ in range(K)]
    B_used_star = [[] for _ in range(K)]
    B_kl_star = [[] for _ in range(K)]
    target_bits_star = [[] for _ in range(K)]
    unserved_bits_star = [[] for _ in range(K)]
    skipped_blocks_per_user = [0 for _ in range(K)]

    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])

    for block in range(num_blocks):
        ensure_blocks_up_to(uplinksystem, block)
        active_mask = [1 for _ in range(K)]
        snapshot_full = _build_precoder_net_snapshot_for_active_mask(
            uplinksystem,
            user_models,
            block,
            active_mask,
        )

        for k in range(K):
            target_bits = int(block_targets[k, block])
            H_kl = np.asarray(uplinksystem.H[k][block], dtype=np.complex64)
            T_ref = int(uplinksystem.T[k])
            P = float(uplinksystem.P[k])
            sigma2 = float(uplinksystem.sigma2[k])
            epsilon = float(uplinksystem.epsilon[k])
            zero_precoder = _zero_uplink_precoder(uplinksystem, k)
            S_block = []

            F_T = infer_precoder_numpy_with_blocklength_and_sigma(
                user_models[k],
                H_kl,
                n_kl=T_ref,
                sigma2=sigma2,
                epsilon=epsilon,
                Nt=int(uplinksystem.NT[k]),
                dk=int(uplinksystem.dk[k]),
                P=P,
                device=DEVICE,
            )
            snapshot_candidate = _replace_snapshot_block(snapshot_full, int(k), int(block), F_T)
            cov_T = build_uplink_rate_covariance(
                uplinksystem,
                sim_cfg,
                k,
                block,
                F_override=snapshot_candidate,
            )
            R_T = _compute_r_fbl_np(H_kl, F_T, sigma2, epsilon, T_ref, cov_T)
            B_max = max(int(np.floor(float(T_ref) * float(R_T))), 0)
            B_used = int(min(target_bits, B_max))
            target_bits_star[k].append(int(target_bits))

            if int(B_used) < int(target_bits):
                S_block.append(
                    {
                        "n_kl": int(T_ref),
                        "n": int(T_ref),
                        "B_l": int(B_used),
                        "Bits per sub-block length B/n_kl": (
                            float(B_used) / float(max(int(T_ref), 1)) if int(B_used) > 0 else 0.0
                        ),
                        "required_R_fbl": float(target_bits) / float(max(int(T_ref), 1)),
                        "achieved_R_fbl": float(R_T),
                        "F": torch.tensor(F_T if int(B_used) > 0 else zero_precoder, dtype=torch.complex64),
                        "R_fbl": float(R_T),
                        "F_power": float(np.linalg.norm(F_T, "fro") ** 2) if int(B_used) > 0 else 0.0,
                        "lambda_rate": 0.0,
                        "lambda_power": 0.0,
                        "loss_curve": [],
                        "method": method_name,
                        "skipped": bool(int(B_used) <= 0),
                        "target_bits": int(target_bits),
                        "unserved_bits": int(max(int(target_bits) - int(B_used), 0)),
                    }
                )
                all_user_block_results[k].append(S_block)
                n_star[k].append(int(T_ref))
                F_star[k].append(np.array(F_T if int(B_used) > 0 else zero_precoder, copy=True))
                R_star[k].append(float(R_T))
                B_used_star[k].append(int(B_used))
                B_kl_star[k].append(int(B_used))
                unserved_bits_star[k].append(int(max(int(target_bits) - int(B_used), 0)))
                if int(B_used) <= 0:
                    skipped_blocks_per_user[k] += 1
                continue

            best_n = int(T_ref)
            best_R = float(R_T)
            best_F = np.array(F_T, copy=True)
            S_block.append(
                {
                    "n_kl": int(T_ref),
                    "n": int(T_ref),
                    "B_l": int(target_bits),
                    "Bits per sub-block length B/n_kl": float(target_bits) / float(max(int(T_ref), 1)),
                    "required_R_fbl": float(target_bits) / float(max(int(T_ref), 1)),
                    "achieved_R_fbl": float(R_T),
                    "F": torch.tensor(F_T, dtype=torch.complex64),
                    "R_fbl": float(R_T),
                    "F_power": float(np.linalg.norm(F_T, "fro") ** 2),
                    "lambda_rate": 0.0,
                    "lambda_power": 0.0,
                    "loss_curve": [],
                    "method": method_name,
                    "skipped": False,
                    "target_bits": int(target_bits),
                    "unserved_bits": 0,
                }
            )

            n_kl = int(T_ref) - int(n_kl_step)
            while n_kl >= int(n_kl_min):
                F_n = infer_precoder_numpy_with_blocklength_and_sigma(
                    user_models[k],
                    H_kl,
                    n_kl=n_kl,
                    sigma2=sigma2,
                    epsilon=epsilon,
                    Nt=int(uplinksystem.NT[k]),
                    dk=int(uplinksystem.dk[k]),
                    P=P,
                    device=DEVICE,
                )
                snapshot_candidate = _replace_snapshot_block(snapshot_full, int(k), int(block), F_n)
                cov_n = build_uplink_rate_covariance(
                    uplinksystem,
                    sim_cfg,
                    k,
                    block,
                    F_override=snapshot_candidate,
                )
                R_n = _compute_r_fbl_np(H_kl, F_n, sigma2, epsilon, n_kl, cov_n)
                rate_violation = (target_bits / float(max(int(n_kl), 1))) - R_n
                if rate_violation > 0.0:
                    break

                best_n = int(n_kl)
                best_R = float(R_n)
                best_F = np.array(F_n, copy=True)
                S_block.append(
                    {
                        "n_kl": int(n_kl),
                        "n": int(n_kl),
                        "B_l": int(target_bits),
                        "Bits per sub-block length B/n_kl": float(target_bits) / float(max(int(n_kl), 1)),
                        "required_R_fbl": float(target_bits) / float(max(int(n_kl), 1)),
                        "achieved_R_fbl": float(R_n),
                        "F": torch.tensor(F_n, dtype=torch.complex64),
                        "R_fbl": float(R_n),
                        "F_power": float(np.linalg.norm(F_n, "fro") ** 2),
                        "lambda_rate": 0.0,
                        "lambda_power": 0.0,
                        "loss_curve": [],
                        "method": method_name,
                        "skipped": False,
                        "target_bits": int(target_bits),
                        "unserved_bits": 0,
                    }
                )
                n_kl -= int(n_kl_step)

            all_user_block_results[k].append(S_block)
            n_star[k].append(int(best_n))
            F_star[k].append(np.array(best_F, copy=True))
            R_star[k].append(float(best_R))
            B_used_star[k].append(int(target_bits))
            B_kl_star[k].append(int(target_bits))
            unserved_bits_star[k].append(0)

    apply_training_solution(uplinksystem, n_star, F_star)

    return {
        "L_out": [int(len(v)) for v in n_star],
        "n_star": n_star,
        "F_star": F_star,
        "R_star": R_star,
        "all_user_block_results_train": all_user_block_results,
        "B_used_star": B_used_star,
        "B_kl_star": B_kl_star,
        "target_bits_star": target_bits_star,
        "unserved_bits_star": unserved_bits_star,
        "norm_stats": [(0.0 + 0.0j, 1.0) for _ in range(K)],
        "method_name": method_name,
        "skipped_blocks_per_user": [int(v) for v in skipped_blocks_per_user],
        "scenario_mode": STREAMING_MODE,
        "scenario_block_targets": block_targets.tolist(),
    }


def evaluate_blocklength_precoder_net(
    uplinksystem: UplinkSystem,
    user_models: Sequence[torch.nn.Module],
    sim_cfg: dict,
    *,
    method_name: str,
) -> dict:
    scenario = build_experiment_scenario(uplinksystem.sc, sim_cfg, seed=int(uplinksystem.seed))
    if str(scenario["mode"]) == STREAMING_MODE:
        return _evaluate_precoder_network_for_streaming(
            uplinksystem,
            user_models,
            sim_cfg,
            method_name=method_name,
        )
    K = int(uplinksystem.K)

    L_out = [1] * K
    n_star = [[] for _ in range(K)]
    F_star = [[] for _ in range(K)]
    R_star = [[] for _ in range(K)]
    all_user_block_results = [[] for _ in range(K)]
    B_used_star = [[] for _ in range(K)]
    B_kl_star = [[] for _ in range(K)]
    evaluation_cost_counters = {
        "per_user_forward_calls": [0 for _ in range(K)],
        "total_forward_calls": 0,
    }

    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])
    use_interference = uses_uplink_interference(sim_cfg)
    snapshot_cache: list[list[np.ndarray]] | None = ([[] for _ in range(K)] if use_interference else None)

    for k in range(K):
        print(
            format_log_line(
                "[UL Monte Carlo Eval]",
                phase="start",
                user=int(k),
            )
        )
        B_rem = int(uplinksystem.B[k])
        ell = 0

        while B_rem > 0:
            ensure_blocks_up_to(uplinksystem, ell)

            H_kl = np.asarray(uplinksystem.H[k][ell], dtype=np.complex64)
            T_ref = int(uplinksystem.T[k])
            P = float(uplinksystem.P[k])
            sigma2 = float(uplinksystem.sigma2[k])
            epsilon = float(uplinksystem.epsilon[k])

            snapshot_full = None
            if snapshot_cache is not None:
                snapshot_full = _ensure_precoder_net_snapshot_block(
                    uplinksystem,
                    user_models,
                    snapshot_cache,
                    int(ell),
                    evaluation_cost_counters=evaluation_cost_counters,
                )

            print(
                format_log_line(
                    "[UL Monte Carlo Eval]",
                    user=int(k),
                    block=int(ell),
                    remaining_bits=int(B_rem),
                )
            )

            S_block = []

            _count_uplink_forward_call(evaluation_cost_counters, int(k))
            F_T = infer_precoder_numpy_with_blocklength_and_sigma(
                user_models[k],
                H_kl,
                n_kl=T_ref,
                sigma2=sigma2,
                epsilon=epsilon,
                Nt=int(uplinksystem.NT[k]),
                dk=int(uplinksystem.dk[k]),
                P=P,
                device=DEVICE,
            )
            snapshot_candidate = (
                _replace_snapshot_block(snapshot_full, int(k), int(ell), F_T)
                if snapshot_full is not None
                else None
            )
            cov_T = build_uplink_rate_covariance(
                uplinksystem,
                sim_cfg,
                k,
                ell,
                F_override=snapshot_candidate,
            )
            R_T = _compute_r_fbl_np(H_kl, F_T, sigma2, epsilon, T_ref, cov_T)
            B_max = max(int(np.floor(float(T_ref) * float(R_T))), 0)
            B_used = int(min(B_rem, B_max))

            print(
                format_log_line(
                    "[UL Monte Carlo Eval]",
                    user=int(k),
                    block=int(ell),
                    n_kl=int(T_ref),
                    requested_bits=int(B_rem),
                    feasible_bits=int(B_max),
                    served_bits=int(B_used),
                    achieved_rate=float(R_T),
                )
            )

            if B_used <= 0:
                print(
                    format_log_line(
                        "[UL Monte Carlo Eval]",
                        user=int(k),
                        block=int(ell),
                        status="stop_no_feasible_T_point",
                    )
                )
                break

            S_block.append(
                {
                    "n_kl": int(T_ref),
                    "n": int(T_ref),
                    "B_l": int(B_used),
                    "Bits per sub-block length B/n_kl": float(B_used) / float(T_ref),
                    "F": torch.tensor(F_T, dtype=torch.complex64),
                    "R_fbl": float(R_T),
                    "F_power": float(np.linalg.norm(F_T, "fro") ** 2),
                    "lambda_rate": 0.0,
                    "lambda_power": 0.0,
                    "loss_curve": [],
                    "method": method_name,
                }
            )

            best_n = int(T_ref)
            best_R = float(S_block[-1]["R_fbl"])
            best_F = S_block[-1]["F"]
            if int(B_used) < int(B_rem):
                print(
                    format_log_line(
                        "[UL Monte Carlo Eval]",
                        user=int(k),
                        block=int(ell),
                        status="stop_partial_payload",
                    )
                )
            else:
                search_cfg = _build_monte_carlo_test_search_cfg(
                    sim_cfg,
                    n_min=int(n_kl_min),
                    n_max=int(T_ref),
                )

                def _evaluate_payload_eval_candidate(candidate_n: int, stage_name: str) -> dict[str, Any]:
                    _count_uplink_forward_call(evaluation_cost_counters, int(k))
                    F_n = infer_precoder_numpy_with_blocklength_and_sigma(
                        user_models[k],
                        H_kl,
                        n_kl=int(candidate_n),
                        sigma2=sigma2,
                        epsilon=epsilon,
                        Nt=int(uplinksystem.NT[k]),
                        dk=int(uplinksystem.dk[k]),
                        P=P,
                        device=DEVICE,
                    )
                    snapshot_candidate = (
                        _replace_snapshot_block(snapshot_full, int(k), int(ell), F_n)
                        if snapshot_full is not None
                        else None
                    )
                    cov_n = build_uplink_rate_covariance(
                        uplinksystem,
                        sim_cfg,
                        k,
                        ell,
                        F_override=snapshot_candidate,
                    )
                    R_n = _compute_r_fbl_np(H_kl, F_n, sigma2, epsilon, int(candidate_n), cov_n)
                    rate_violation = (float(B_used) / float(max(int(candidate_n), 1))) - float(R_n)
                    print(
                        format_log_line(
                            "[UL Monte Carlo Eval]",
                            user=int(k),
                            block=int(ell),
                            n_kl=int(candidate_n),
                            search_stage=str(stage_name),
                            achieved_rate=float(R_n),
                            rate_violation=float(rate_violation),
                        )
                    )
                    return {
                        "feasible": bool(rate_violation <= 0.0),
                        "R_fbl": float(R_n),
                        "F": np.array(F_n, copy=True),
                    }

                reduction_search = run_n_frontier_search(search_cfg, _evaluate_payload_eval_candidate)
                for accepted in reduction_search["accepted"]:
                    best_n = int(accepted["n_kl"])
                    best_R = float(accepted["result"]["R_fbl"])
                    best_F = torch.tensor(accepted["result"]["F"], dtype=torch.complex64)
                    S_block.append(
                        {
                            "n_kl": int(best_n),
                            "n": int(best_n),
                            "B_l": int(B_used),
                            "Bits per sub-block length B/n_kl": float(B_used) / float(best_n),
                            "F": torch.tensor(accepted["result"]["F"], dtype=torch.complex64),
                            "R_fbl": float(best_R),
                            "F_power": float(np.linalg.norm(accepted["result"]["F"], "fro") ** 2),
                            "lambda_rate": 0.0,
                            "lambda_power": 0.0,
                            "loss_curve": [],
                            "method": method_name,
                        }
                    )

            all_user_block_results[k].append(S_block)
            n_star[k].append(best_n)
            F_star[k].append(best_F)
            R_star[k].append(best_R)
            B_used_star[k].append(int(B_used))

            B_kl = min(B_rem, int(B_used))
            B_kl_star[k].append(int(B_kl))
            B_rem -= B_kl
            print(
                format_log_line(
                    "[UL Monte Carlo Allocation]",
                    user=int(k),
                    block=int(ell),
                    chosen_n_kl=int(best_n),
                    served_bits=int(B_used),
                    committed_bits=int(B_kl),
                    remaining_bits=int(B_rem),
                )
            )

            if B_rem > 0:
                ell += 1
                L_out[k] = ell + 1

    apply_training_solution(uplinksystem, n_star, F_star)

    return {
        "L_out": L_out,
        "n_star": n_star,
        "F_star": F_star,
        "R_star": R_star,
        "all_user_block_results_train": all_user_block_results,
        "B_used_star": B_used_star,
        "B_kl_star": B_kl_star,
        "norm_stats": [(0.0 + 0.0j, 1.0) for _ in range(K)],
        "method_name": method_name,
        "skipped_blocks_per_user": [0 for _ in range(K)],
        "evaluation_cost_counters": evaluation_cost_counters,
        "scenario_mode": PAYLOAD_MODE,
    }


__all__ = [
    "estimate_initial_random_precoder_schedule_for_scenario",
    "evaluate_blocklength_precoder_net",
]
