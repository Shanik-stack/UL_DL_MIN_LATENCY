"""Optimize a finite downlink payload across generated channel blocks."""

import copy
from typing import Any, List

import numpy as np

from latency_optimization.experiments.scenarios import STREAMING_MODE, build_experiment_scenario
from latency_optimization.results.console import format_latency_log_line, format_log_line
from latency_optimization.results.metrics import build_schedule_reference as _build_initial_baseline_reference

from ...simulation.baselines import (
    estimate_random_precoder_payload_latency,
    estimate_random_precoder_streaming_latency,
)
from ...simulation.block_state import (
    collect_interference_diagnostics,
    ensure_precoder_block,
    evaluate_block_candidate,
    expand_precoders_for_plan,
    zero_precoder_block,
)
from ...precoders.service import (
    build_model_optimizers,
    build_precoder_models,
    describe_precoder_parameterization,
    refresh_block_precoders,
)
from ...precoders.checkpoints import export_user_model_specs
from ...precoders.models import validate_downlink_precoder_net_scope
from ...simulation.system import DownlinkSystem
from ...objectives.settings import (
    objective_display_name,
    objective_weight_strategy_name,
    validate_convergence_objective_mode,
    validate_convergence_priority_weight_strategy,
    validate_objective_mode,
)

from .optimize_precoder import (
    _reduce_blocklengths_with_reoptimization,
    _resolve_block_user_weights,
    optimize_precoders_for_block,
    validate_convergence_precoder_update_mode,
)

def optimize_payload_transmission(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    verbose: bool,
    method_name: str,
    objective_mode: str,
) -> dict[str, Any]:
    """Allocate a finite payload across generated channel blocks.

    Each block first optimizes service at full blocklength, commits feasible
    bits, then searches smaller blocklengths only after the request is served.
    """
    objective_mode = validate_objective_mode(objective_mode)
    model_scope = validate_downlink_precoder_net_scope(sim_params.get("downlink_precoder_net_scope", "per_user_nets"))
    update_mode = validate_convergence_precoder_update_mode(sim_params)
    configured_weight_strategy = validate_convergence_priority_weight_strategy(sim_params)
    initial_snr_db, initial_sinr_db = system.get_snr_sinr_db()
    initial_latency, initial_plan, initial_interference_diag = estimate_random_precoder_payload_latency(
        system,
        sim_params,
    )
    naive_full_t_latency, naive_full_t_plan, _ = estimate_random_precoder_payload_latency(
        system,
        sim_params,
        allow_n_reduction=False,
    )
    if verbose:
        print(
            format_latency_log_line(
                "[DL Initial Baseline]",
                initial_latency,
                seed=int(system.seed),
                scenario="payload",
                method="convergence",
            )
        )
    if update_mode == "precoder_net":
        user_models = build_precoder_models(
            system,
            model_scope=model_scope,
        )
        model_optimizers = build_model_optimizers(
            user_models,
            learning_rate=float(sim_params["user_update_lr"]),
        )
    else:
        user_models = []
        model_optimizers = []

    remaining = np.asarray(system.B, dtype=int).copy()
    n_plan: List[List[int]] = [[] for _ in range(system.K)]
    B_plan: List[List[int]] = [[] for _ in range(system.K)]
    R_plan: List[List[float]] = [[] for _ in range(system.K)]
    working_F: List[List[np.ndarray]] = system.clone_precoders()
    epoch_history: list[dict[str, float]] = []
    outer_history: list[dict[str, float]] = []
    rate_points: list[dict[str, float]] = []
    max_blocks = int(sim_params.get("max_total_blocks", 256))

    block = 0
    while np.any(remaining > 0):
        if block >= max_blocks:
            raise RuntimeError(
                f"Reached max_total_blocks={max_blocks} with remaining bits {remaining.tolist()}."
            )

        active_users = [k for k in range(system.K) if int(remaining[k]) > 0]
        # Weight computation uses every user's channel at this block, including
        # users that already finished their payload.
        for k in range(system.K):
            ensure_precoder_block(system, working_F, k, block)
        if update_mode == "precoder_net":
            refresh_block_precoders(
                system,
                working_F,
                user_models,
                active_users,
                block,
            )
        inverse_cnr_weights, active_weight_strategy = _resolve_block_user_weights(
            system,
            active_users,
            block,
            sim_params,
            objective_mode,
        )
        if verbose:
            print(
                format_log_line(
                    "[DL Convergence Block]",
                    block=int(block),
                    active_users=int(len(active_users)),
                    remaining_bits=int(np.sum(remaining)),
                    objective=str(objective_mode),
                )
            )
            weights_text = ", ".join(f"u{k}={inverse_cnr_weights[k]:.3f}" for k in active_users)
            print(f"    weight_strategy={active_weight_strategy} | user_weights: {weights_text}")

        transmit_users = list(active_users)
        skipped_users: list[int] = []
        skipped_user_rates: dict[int, float] = {}
        block_history: list[dict[str, float]] = []
        block_eval: dict[str, Any] = {
            "feasible_count": 0,
            "min_max_bits": 0,
            "user_ids": [],
            "user_max_bits": [],
            "user_rates": [],
            "sum_rate": 0.0,
            "min_rate": 0.0,
        }

        while len(transmit_users) > 0:
            transmit_weights = {int(k): float(inverse_cnr_weights.get(int(k), 1.0)) for k in transmit_users}
            requested_bits_block = {int(k): int(remaining[int(k)]) for k in transmit_users}
            solve_result = optimize_precoders_for_block(
                system,
                working_F,
                user_models,
                model_optimizers,
                transmit_users,
                block,
                requested_bits_block,
                sim_params,
                verbose=verbose,
                objective_mode=objective_mode,
                user_weights=transmit_weights,
                max_epochs=int(sim_params["max_epochs"]),
            )
            current_history = solve_result["history"]
            current_eval = evaluate_block_candidate(system, working_F, transmit_users, block)
            block_history.extend(current_history)
            block_eval = current_eval
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
                        "[DL Convergence Block]",
                        block=int(block),
                        skipped_users=[int(k) for k in infeasible_users],
                        action="reoptimize_remaining",
                    )
                )
        final_plans: dict[int, dict[str, Any]] = {}
        if len(transmit_users) > 0:
            transmit_weights = {int(k): float(inverse_cnr_weights.get(int(k), 1.0)) for k in transmit_users}
            requested_bits_block = {int(k): int(remaining[int(k)]) for k in transmit_users}
            final_plans, refinement_history = _reduce_blocklengths_with_reoptimization(
                system,
                working_F,
                user_models,
                model_optimizers,
                transmit_users,
                block,
                requested_bits_block,
                sim_params,
                objective_mode=objective_mode,
                user_weights=transmit_weights,
                verbose=verbose,
            )
            block_history.extend(refinement_history)
        epoch_history.extend(block_history)

        block_bits = 0
        for k in active_users:
            inverse_cnr_weight = float(inverse_cnr_weights.get(int(k), 1.0))
            if int(k) in skipped_users:
                zero_precoder_block(system, working_F, k, block)
                skipped_rate = float(skipped_user_rates.get(int(k), 0.0))
                B_plan[k].append(0)
                n_plan[k].append(int(system.T[k]))
                R_plan[k].append(skipped_rate)
                rate_points.append(
                    {
                        "user": int(k),
                        "block": int(block),
                        "n_kl": int(system.T[k]),
                        "B_kl": 0,
                        "required_rate": 0.0,
                        "achieved_rate": skipped_rate,
                        "rate_margin": skipped_rate,
                        "inverse_cnr_weight": inverse_cnr_weight,
                        "skipped": True,
                    }
                )
                if verbose:
                    print(
                        format_log_line(
                            "[DL Convergence Allocation]",
                            user=int(k),
                            block=int(block),
                            status="skipped",
                            n_kl=int(system.T[k]),
                            achieved_rate=float(skipped_rate),
                        )
                    )
                continue

            plan = final_plans.get(int(k), {"B_used": 0, "n_used": int(system.T[k]), "R_used": 0.0})
            B_used = int(plan["B_used"])
            n_used = int(plan["n_used"])
            R_used = float(plan["R_used"])
            B_plan[k].append(int(B_used))
            n_plan[k].append(int(n_used))
            R_plan[k].append(float(R_used))
            remaining[k] -= int(B_used)
            block_bits += int(B_used)
            required_rate = float(B_used) / float(max(int(n_used), 1))
            rate_margin = float(R_used) - required_rate
            rate_points.append(
                {
                    "user": int(k),
                    "block": int(block),
                    "n_kl": int(n_used),
                    "B_kl": int(B_used),
                    "required_rate": required_rate,
                    "achieved_rate": float(R_used),
                    "rate_margin": rate_margin,
                    "inverse_cnr_weight": inverse_cnr_weight,
                    "skipped": False,
                }
            )
            if verbose:
                print(
                    format_log_line(
                        "[DL Convergence Allocation]",
                        user=int(k),
                        block=int(block),
                        served_bits=int(B_used),
                        n_kl=int(n_used),
                        required_rate=float(required_rate),
                        achieved_rate=float(R_used),
                        rate_margin=float(rate_margin),
                    )
                )

        outer_history.append(
            {
                "block": int(block),
                "active_users": int(len(active_users)),
                "transmitting_users": int(len(transmit_users)),
                "skipped_users": int(len(skipped_users)),
                "allocated_bits": int(block_bits),
                "remaining_bits": int(np.sum(remaining)),
                "feasible_users": int(block_eval["feasible_count"]),
                "min_max_bits": int(block_eval["min_max_bits"]),
                "inverse_cnr_weights": {int(k): float(v) for k, v in inverse_cnr_weights.items()},
                "final_precoder_delta": float(block_history[-1]["max_precoder_delta"]) if block_history else 0.0,
            }
        )
        if verbose:
            print(
                format_log_line(
                    "[DL Convergence Block]",
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
    return {
        "method_name": method_name,
        "objective_mode": objective_display_name(objective_mode, configured_weight_strategy),
        "allocation_mode": "payload",
        "weight_strategy": objective_weight_strategy_name(objective_mode, configured_weight_strategy),
        "convergence_precoder_update_mode": str(update_mode),
        "precoder_parameterization": describe_precoder_parameterization(
            model_scope,
            update_mode,
        ),
        "downlink_precoder_net_scope": str(model_scope),
        "user_model_specs": (
            export_user_model_specs(
                system.Nr,
                system.Nb,
                system.dk,
                model_scope=model_scope,
                context_k=int(system.K),
                context_max_nr=int(np.max(system.Nr)),
                context_max_nb=int(np.max(system.Nb)),
                context_max_dk=int(np.max(system.dk)),
            )
            if update_mode == "precoder_net"
            else []
        ),
        "n_kl": copy.deepcopy(n_plan),
        "B_kl": copy.deepcopy(B_plan),
        "R_fbl": [list(map(float, user_rates)) for user_rates in system.R_fbl],
        "R_alloc": copy.deepcopy(R_plan),
        "initial_latency": initial_latency,
        "initial_plan": initial_plan,
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
    }

__all__ = ["optimize_payload_transmission"]
