"""Optimize one uplink user on one channel block and search its blocklength."""

import copy

import numpy as np
import torch
from torch import nn

from latency_optimization.optimization.blocklength_search import build_n_search_config, run_n_frontier_search
from latency_optimization.experiments.scenarios import STREAMING_MODE, build_experiment_scenario
from latency_optimization.precoders.parameters import complex_parameter
from latency_optimization.results.console import format_log_line
from latency_optimization.runtime import DEVICE

from ...precoders.checkpoints import (
    export_user_model_specs,
    export_user_model_states,
)
from ...precoders.models import build_user_precoder_net
from ...physics.rate import build_uplink_rate_covariance
from ...objectives.precoder import UplinkPrecoderObjective
from .optimize_precoder import (
    optimize_precoder_for_nl,
    validate_convergence_precoder_update_mode,
)

def optimize_user_blocklength_and_precoder(
    uplinksystem,
    user: int,
    block: int,
    B_rem: int,
    sim_cfg: dict,
    precoder_net: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    precoder_param: torch.nn.Parameter | None = None,
    interference_F_snapshot=None,
    bit_budget_role: str = "payload",
):
    """Optimize one user's beam and blocklength for one channel block.

    The beam is first solved at n=T. Partial service is committed at T; only a
    fully served request proceeds to smaller n values with fresh optimization.
    """
    update_mode = validate_convergence_precoder_update_mode(
        sim_cfg.get("convergence_precoder_update_mode", "precoder_net")
    )
    is_streaming_block = bit_budget_role == "streaming"
    bit_budget_label = "bits_requested_this_block" if is_streaming_block else "remaining_payload_bits"

    print(
        format_log_line(
            "[UL Convergence Block]",
            user=int(user),
            block=int(block),
            budget_mode=str(bit_budget_label),
            requested_bits=int(B_rem),
        )
    )

    P = float(uplinksystem.P[user])
    Nt = int(uplinksystem.NT[user])
    dk = int(uplinksystem.dk[user])
    sigma2 = float(uplinksystem.sigma2[user])
    epsilon = float(uplinksystem.epsilon[user])

    H_kl = torch.tensor(uplinksystem.H[user][block], dtype=torch.complex64, device=DEVICE)
    noise_plus_interference_cov_np = build_uplink_rate_covariance(
        uplinksystem,
        sim_cfg,
        user,
        block,
        F_override=interference_F_snapshot,
    )
    noise_plus_interference_cov = (
        None
        if noise_plus_interference_cov_np is None
        else torch.tensor(
            noise_plus_interference_cov_np,
            dtype=torch.complex64,
            device=DEVICE,
        )
    )

    n_kl_max = int(uplinksystem.T[user])
    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])
    max_epochs = max(1, int(sim_cfg["max_epochs"]))
    loss_fn = UplinkPrecoderObjective(
        channel=H_kl,
        noise_variance=sigma2,
        epsilon=epsilon,
        payload_bits=float(B_rem),
        power_limit=float(P),
        blocklength=n_kl_max,
        noise_plus_interference_covariance=noise_plus_interference_cov,
        rate_law=uplinksystem.rate_law,
    ).to(DEVICE)

    results = []

    # ------------------------------------------------------------
    # STEP A: Optimize once at n = T, then clip the served bits to the
    # maximum payload that the optimized beam can support.
    # ------------------------------------------------------------
    print(
        format_log_line(
            "[UL Convergence Solve]",
            user=int(user),
            block=int(block),
            stage="n=T",
            n_kl=int(n_kl_max),
            requested_bits=int(B_rem),
        )
    )
    loss_fn.set_blocklength(n_kl_max)
    loss_fn.set_payload(float(B_rem))
    B_initial = int(B_rem)
    out = optimize_precoder_for_nl(
        precoder_net=precoder_net,
        loss_fn=loss_fn,
        Nt=Nt, dk=dk,
        max_epochs=max_epochs,
        optimizer=optimizer,
        stopping_config=sim_cfg,
        print_every_epoch=int(sim_cfg.get("print_every_epoch", 1)),
        verbose=True,
        log_context={"user": int(user), "block": int(block), "n_kl": int(n_kl_max)},
        precoder_param=precoder_param,
        update_mode=update_mode,
    )

    step_a_diagnostics = {
        "achieved_R_fbl": float(out["R_fbl"]),
        "F": out["F"],
        "F_power": float(out["F_power"]),
        "rate_gap": float(out["rate_gap"]),
        "power_gap": float(out["power_gap"]),
        "loss_curve": out["loss_curve"],
        "kkt_history": copy.deepcopy(out.get("kkt_history", [])),
        "solve_status": out.get("solve_status", "unknown"),
        "final_primal_residual": float(out.get("final_primal_residual", 0.0)),
    }

    print(
        format_log_line(
            "[UL Convergence Result]",
            user=int(user),
            block=int(block),
            n_kl=int(n_kl_max),
            achieved_rate=float(out["R_fbl"]),
            power=float(out["F_power"]),
            rate_gap=float(out["rate_gap"]),
            power_gap=float(out["power_gap"]),
            solve_status=str(out.get("solve_status", "unknown")),
        )
    )

    if out["power_gap"] > 0.0:
        print(format_log_line("[UL Convergence Result]", user=int(user), block=int(block), n_kl=int(n_kl_max), status="power_infeasible"))
        return [], 0, step_a_diagnostics

    B_max_T = max(int(np.floor(float(n_kl_max) * float(out["R_fbl"]))), 0)
    B_used = int(min(B_initial, B_max_T))
    fully_feasible_at_T = out["rate_gap"] <= 0.0

    if B_used <= 0:
        if is_streaming_block:
            print(format_log_line("[UL Convergence Result]", user=int(user), block=int(block), n_kl=int(n_kl_max), status="zero_service_streaming_block"))
        else:
            print(format_log_line("[UL Convergence Result]", user=int(user), block=int(block), n_kl=int(n_kl_max), status="zero_service_payload"))
        return [], 0, step_a_diagnostics

    print(
        format_log_line(
            "[UL Convergence Allocation]",
            user=int(user),
            block=int(block),
            n_kl=int(n_kl_max),
            requested_bits=int(B_initial),
            served_bits=int(B_used),
            full_service=bool(fully_feasible_at_T),
        )
    )

    results.append({
        "n_kl": int(n_kl_max),
        "n": int(n_kl_max),
        "B_l": int(B_used),
        "Bits per sub-block length B/n_kl": float(B_used) / float(n_kl_max),
        "required_R_fbl": float(B_initial) / float(max(int(n_kl_max), 1)),
        "achieved_R_fbl": float(out["R_fbl"]),
        "F": out["F"],
        "R_fbl": float(out["R_fbl"]),
        "F_power": float(out["F_power"]),
        "loss_curve": out["loss_curve"],
        "kkt_history": copy.deepcopy(out.get("kkt_history", [])),
        "solve_status": out.get("solve_status", "unknown"),
        "final_primal_residual": float(out.get("final_primal_residual", 0.0)),
    })

    bits_left_after_current_request = int(B_initial - B_used)
    if is_streaming_block:
        print(
            format_log_line(
                "[UL Convergence Block]",
                user=int(user),
                block=int(block),
                stage="post_T",
                served_bits=int(B_used),
                unserved_bits=int(bits_left_after_current_request),
            )
        )
    else:
        print(
            format_log_line(
                "[UL Convergence Block]",
                user=int(user),
                block=int(block),
                stage="post_T",
                served_bits=int(B_used),
                remaining_bits=int(bits_left_after_current_request),
            )
        )

    # ------------------------------------------------------------
    # STEP B: Reduce n while keeping B fixed. Each new n triggers a fresh
    # inner optimization warm-started from the last feasible solution.
    # ------------------------------------------------------------
    print(
        format_log_line(
            "[UL Convergence Reduction]",
            user=int(user),
            block=int(block),
            served_bits=int(B_used),
            direction=str(sim_cfg.get("n_search_direction", "descending")),
            strategy=str(sim_cfg.get("n_search_strategy", "fixed_step")),
            n_min=int(n_kl_min),
        )
    )
    if bits_left_after_current_request != 0:
        print(
            format_log_line(
                "[UL Convergence Reduction]",
                user=int(user),
                block=int(block),
                status="stop_partial_streaming_block" if is_streaming_block else "stop_partial_payload",
            )
        )
    else:
        search_cfg = build_n_search_config(
            n_min=int(n_kl_min),
            n_max=int(n_kl_max),
            fine_step=int(n_kl_step),
            direction=sim_cfg.get("n_search_direction", "descending"),
            strategy=sim_cfg.get("n_search_strategy", "fixed_step"),
            coarse_step=sim_cfg.get("n_search_coarse_step", int(n_kl_step)),
            exponential_factor=sim_cfg.get("n_search_exponential_factor", 2),
        )

        def _evaluate_reduced_n(candidate_n: int, stage_name: str) -> dict:
            print(
                format_log_line(
                    "[UL Convergence Solve]",
                    user=int(user),
                    block=int(block),
                    stage=f"reduced_n_{stage_name}",
                    n_kl=int(candidate_n),
                    served_bits=int(B_used),
                )
            )
            if update_mode == "direct_precoder":
                model_checkpoint = precoder_param.detach().cpu().clone()
            else:
                model_checkpoint = {
                    key: value.detach().cpu().clone()
                    for key, value in precoder_net.state_dict().items()
                }
            optimizer_checkpoint = copy.deepcopy(optimizer.state_dict())
            loss_fn.set_blocklength(int(candidate_n))
            loss_fn.set_payload(float(B_used))
            out = optimize_precoder_for_nl(
                precoder_net=precoder_net,
                loss_fn=loss_fn,
                Nt=Nt,
                dk=dk,
                max_epochs=max_epochs,
                optimizer=optimizer,
                stopping_config=sim_cfg,
                print_every_epoch=int(sim_cfg.get("print_every_epoch", 1)),
                verbose=True,
                log_context={"user": int(user), "block": int(block), "n_kl": int(candidate_n)},
                precoder_param=precoder_param,
                update_mode=update_mode,
            )
            feasible = (out["rate_gap"] <= 0.0) and (out["power_gap"] <= 0.0)
            print(
                format_log_line(
                    "[UL Convergence Candidate]",
                    user=int(user),
                    block=int(block),
                    n_kl=int(candidate_n),
                    search_stage=str(stage_name),
                    achieved_rate=float(out["R_fbl"]),
                    rate_gap=float(out["rate_gap"]),
                    power_gap=float(out["power_gap"]),
                    status="accepted" if feasible else "rejected",
                )
            )
            if not feasible:
                if update_mode == "direct_precoder":
                    with torch.no_grad():
                        precoder_param.copy_(model_checkpoint.to(device=DEVICE, dtype=precoder_param.dtype))
                else:
                    precoder_net.load_state_dict(model_checkpoint)
                optimizer.load_state_dict(optimizer_checkpoint)
            return {
                "feasible": bool(feasible),
                "solver_output": out,
            }

        reduction_search = run_n_frontier_search(search_cfg, _evaluate_reduced_n)
        for accepted in reduction_search["accepted"]:
            n_candidate = int(accepted["n_kl"])
            out = accepted["result"]["solver_output"]
            results.append({
                "n_kl": int(n_candidate),
                "n": int(n_candidate),
                "B_l": int(B_used),
                "Bits per sub-block length B/n_kl": float(B_used) / float(n_candidate),
                "required_R_fbl": float(B_used) / float(max(int(n_candidate), 1)),
                "achieved_R_fbl": float(out["R_fbl"]),
                "F": out["F"],
                "R_fbl": float(out["R_fbl"]),
                "F_power": float(out["F_power"]),
                "loss_curve": out["loss_curve"],
                "kkt_history": copy.deepcopy(out.get("kkt_history", [])),
                "solve_status": out.get("solve_status", "unknown"),
                "final_primal_residual": float(out.get("final_primal_residual", 0.0)),
            })
        frontier_rejected = reduction_search.get("frontier_rejected")
        if frontier_rejected is not None:
            print(
                format_log_line(
                    "[UL Convergence Reduction]",
                    user=int(user),
                    block=int(block),
                    n_kl=int(frontier_rejected["n_kl"]),
                    status="stop_infeasible",
                )
            )

    if len(results) > 0:
        best_result = results[-1]
        print(
            format_log_line(
                "[UL Convergence Final]",
                user=int(user),
                block=int(block),
                feasible_points=int(len(results)),
                chosen_n_kl=int(best_result["n_kl"]),
                served_bits=int(B_used),
                achieved_rate=float(best_result["R_fbl"]),
                power=float(best_result["F_power"]),
                solve_status=str(best_result.get("solve_status", "unknown")),
            )
        )

    return results, int(B_used), step_a_diagnostics

__all__ = ["optimize_user_blocklength_and_precoder"]
