"""Uplink Monte Carlo test-time allocation and evaluation."""

from typing import Any, Sequence

import numpy as np
import torch

from latency_optimization.core.blocklength import build_monte_carlo_n_search_config, run_n_frontier_search
from latency_optimization.core.scenarios import PAYLOAD_MODE, STREAMING_MODE, build_experiment_scenario
from latency_optimization.results.console import format_log_line
from latency_optimization.runtime import DEVICE
from latency_optimization.precoders.serialization import precoder_to_numpy

from ..precoders.inference import infer_precoder
from ..simulation import (
    apply_training_solution,
    clone_nested_arrays,
    collect_uplink_interference_diagnostics,
    ensure_blocks_up_to,
)
from ..system import UplinkSystem
from ..uplink_rate_model import build_uplink_rate_covariance_torch, uses_uplink_interference

from .network_operations import (
    _build_precoder_net_snapshot_for_active_mask,
    _compute_r_fbl_torch,
    _zero_uplink_precoder,
)
from .rollout import (
    _count_uplink_forward_call,
    _ensure_precoder_net_snapshot_block,
    _replace_snapshot_block,
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
            H_kl = torch.as_tensor(
                uplinksystem.H[k][block], dtype=torch.complex64, device=DEVICE
            )
            T_ref = int(uplinksystem.T[k])
            P = float(uplinksystem.P[k])
            sigma2 = float(uplinksystem.sigma2[k])
            epsilon = float(uplinksystem.epsilon[k])
            zero_precoder = _zero_uplink_precoder(uplinksystem, k)
            S_block = []

            with torch.no_grad():
                F_T = infer_precoder(
                    user_models[k], H_kl, T_ref, sigma2, epsilon,
                    transmit_antennas=int(uplinksystem.NT[k]),
                    streams=int(uplinksystem.dk[k]), power_limit=P,
                ).detach()
            snapshot_candidate = _replace_snapshot_block(snapshot_full, int(k), int(block), F_T)
            cov_T = build_uplink_rate_covariance_torch(
                uplinksystem,
                sim_cfg,
                k,
                block,
                precoders=snapshot_candidate,
                device=DEVICE,
            )
            R_T = float(
                _compute_r_fbl_torch(H_kl, F_T, sigma2, epsilon, T_ref, cov_T).detach().cpu()
            )
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
                        "F": (F_T if int(B_used) > 0 else zero_precoder).detach(),
                        "R_fbl": float(R_T),
                        "F_power": float(torch.linalg.norm(F_T, ord="fro").square().real.cpu()) if int(B_used) > 0 else 0.0,
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
                F_star[k].append((F_T if int(B_used) > 0 else zero_precoder).detach().clone())
                R_star[k].append(float(R_T))
                B_used_star[k].append(int(B_used))
                B_kl_star[k].append(int(B_used))
                unserved_bits_star[k].append(int(max(int(target_bits) - int(B_used), 0)))
                if int(B_used) <= 0:
                    skipped_blocks_per_user[k] += 1
                continue

            best_n = int(T_ref)
            best_R = float(R_T)
            best_F = F_T.detach().clone()
            S_block.append(
                {
                    "n_kl": int(T_ref),
                    "n": int(T_ref),
                    "B_l": int(target_bits),
                    "Bits per sub-block length B/n_kl": float(target_bits) / float(max(int(T_ref), 1)),
                    "required_R_fbl": float(target_bits) / float(max(int(T_ref), 1)),
                    "achieved_R_fbl": float(R_T),
                    "F": F_T.detach(),
                    "R_fbl": float(R_T),
                    "F_power": float(torch.linalg.norm(F_T, ord="fro").square().real.cpu()),
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
                with torch.no_grad():
                    F_n = infer_precoder(
                        user_models[k], H_kl, n_kl, sigma2, epsilon,
                        transmit_antennas=int(uplinksystem.NT[k]),
                        streams=int(uplinksystem.dk[k]), power_limit=P,
                    ).detach()
                snapshot_candidate = _replace_snapshot_block(snapshot_full, int(k), int(block), F_n)
                cov_n = build_uplink_rate_covariance_torch(
                    uplinksystem,
                    sim_cfg,
                    k,
                    block,
                    precoders=snapshot_candidate,
                    device=DEVICE,
                )
                R_n = float(
                    _compute_r_fbl_torch(H_kl, F_n, sigma2, epsilon, n_kl, cov_n).detach().cpu()
                )
                rate_violation = (target_bits / float(max(int(n_kl), 1))) - R_n
                if rate_violation > 0.0:
                    break

                best_n = int(n_kl)
                best_R = float(R_n)
                best_F = F_n.detach().clone()
                S_block.append(
                    {
                        "n_kl": int(n_kl),
                        "n": int(n_kl),
                        "B_l": int(target_bits),
                        "Bits per sub-block length B/n_kl": float(target_bits) / float(max(int(n_kl), 1)),
                        "required_R_fbl": float(target_bits) / float(max(int(n_kl), 1)),
                        "achieved_R_fbl": float(R_n),
                        "F": F_n.detach(),
                        "R_fbl": float(R_n),
                        "F_power": float(torch.linalg.norm(F_n, ord="fro").square().real.cpu()),
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
            F_star[k].append(best_F.detach().clone())
            R_star[k].append(float(best_R))
            B_used_star[k].append(int(target_bits))
            B_kl_star[k].append(int(target_bits))
            unserved_bits_star[k].append(0)

    serialized_precoders = [
        [precoder_to_numpy(precoder) for precoder in user_blocks]
        for user_blocks in F_star
    ]
    apply_training_solution(uplinksystem, n_star, serialized_precoders)

    return {
        "L_out": [int(len(v)) for v in n_star],
        "n_star": n_star,
        "F_star": serialized_precoders,
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
    snapshot_cache: list[list[torch.Tensor]] | None = ([[] for _ in range(K)] if use_interference else None)

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

            H_kl = torch.as_tensor(
                uplinksystem.H[k][ell], dtype=torch.complex64, device=DEVICE
            )
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
            with torch.no_grad():
                F_T = infer_precoder(
                    user_models[k], H_kl, T_ref, sigma2, epsilon,
                    transmit_antennas=int(uplinksystem.NT[k]),
                    streams=int(uplinksystem.dk[k]), power_limit=P,
                ).detach()
            snapshot_candidate = (
                _replace_snapshot_block(snapshot_full, int(k), int(ell), F_T)
                if snapshot_full is not None
                else None
            )
            cov_T = build_uplink_rate_covariance_torch(
                uplinksystem,
                sim_cfg,
                k,
                ell,
                precoders=snapshot_candidate,
                device=DEVICE,
            )
            R_T = float(
                _compute_r_fbl_torch(H_kl, F_T, sigma2, epsilon, T_ref, cov_T).detach().cpu()
            )
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
                    "F": F_T.detach(),
                    "R_fbl": float(R_T),
                    "F_power": float(torch.linalg.norm(F_T, ord="fro").square().real.cpu()),
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
                search_cfg = build_monte_carlo_n_search_config(
                    sim_cfg,
                    n_min=int(n_kl_min),
                    n_max=int(T_ref),
                    phase="testing",
                )

                def _evaluate_payload_eval_candidate(candidate_n: int, stage_name: str) -> dict[str, Any]:
                    _count_uplink_forward_call(evaluation_cost_counters, int(k))
                    with torch.no_grad():
                        F_n = infer_precoder(
                            user_models[k], H_kl, int(candidate_n), sigma2, epsilon,
                            transmit_antennas=int(uplinksystem.NT[k]),
                            streams=int(uplinksystem.dk[k]), power_limit=P,
                        ).detach()
                    snapshot_candidate = (
                        _replace_snapshot_block(snapshot_full, int(k), int(ell), F_n)
                        if snapshot_full is not None
                        else None
                    )
                    cov_n = build_uplink_rate_covariance_torch(
                        uplinksystem,
                        sim_cfg,
                        k,
                        ell,
                        precoders=snapshot_candidate,
                        device=DEVICE,
                    )
                    R_n = float(
                        _compute_r_fbl_torch(
                            H_kl, F_n, sigma2, epsilon, int(candidate_n), cov_n
                        ).detach().cpu()
                    )
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
                        "F": F_n.detach().clone(),
                    }

                reduction_search = run_n_frontier_search(search_cfg, _evaluate_payload_eval_candidate)
                for accepted in reduction_search["accepted"]:
                    best_n = int(accepted["n_kl"])
                    best_R = float(accepted["result"]["R_fbl"])
                    best_F = accepted["result"]["F"].detach().clone()
                    S_block.append(
                        {
                            "n_kl": int(best_n),
                            "n": int(best_n),
                            "B_l": int(B_used),
                            "Bits per sub-block length B/n_kl": float(B_used) / float(best_n),
                            "F": accepted["result"]["F"].detach(),
                            "R_fbl": float(best_R),
                            "F_power": float(torch.linalg.norm(accepted["result"]["F"], ord="fro").square().real.cpu()),
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

    serialized_precoders = [
        [precoder_to_numpy(precoder) for precoder in user_blocks]
        for user_blocks in F_star
    ]
    apply_training_solution(uplinksystem, n_star, serialized_precoders)

    return {
        "L_out": L_out,
        "n_star": n_star,
        "F_star": serialized_precoders,
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
    "evaluate_blocklength_precoder_net",
]
