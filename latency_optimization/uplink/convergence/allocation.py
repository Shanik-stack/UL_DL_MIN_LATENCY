"""Uplink payload scheduling and blocklength allocation."""

import copy

import numpy as np
import torch
from torch import nn

from latency_optimization.core.blocklength import build_n_search_config, run_n_frontier_search
from latency_optimization.core.scenarios import STREAMING_MODE, build_experiment_scenario
from latency_optimization.precoders.parameters import complex_parameter_from_numpy
from latency_optimization.precoders.power import cap_matrix_power_numpy
from latency_optimization.results.console import format_log_line
from latency_optimization.runtime import DEVICE

from ..model_service import build_precoder_snapshot_from_models
from ..precoder_models import (
    build_user_precoder_net_with_blocklength_and_sigma,
    export_user_model_specs,
    export_user_model_states,
    load_user_precoder_models,
)
from ..uplink_rate_model import build_uplink_rate_covariance, evaluate_uplink_rate_numpy
from ..objective import FiniteBlocklengthRateObjective
from .solver import (
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
    remaining_bits_by_user: list[int] | np.ndarray | None = None,
    block_index_by_user: list[int] | np.ndarray | None = None,
):
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
    Nr = int(uplinksystem.NR[user])
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

    T = int(uplinksystem.T[user])
    n_kl_max = int(uplinksystem.T[user])
    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])
    max_epochs = max(1, int(sim_cfg["max_epochs"]))
    reduced_n_kl_log_interval = max(1, int(sim_cfg.get("reduced_n_kl_log_interval", 1)))
    
    lr_net = float(sim_cfg["lr_net"])

    loss_fn = FiniteBlocklengthRateObjective(
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

def optimize_payload_with_precoder_training(
    uplinksystem,
    sim_cfg: dict,
    interference_F_snapshot=None,
    commit_live_precoders: bool = True,
):
    """
    Training version that is consistent with the TEST logic style:
      - For each user, iteratively allocate blocks until B_rem == 0
      - For each block:
          Step A: optimize once at n=T and clip the served bits to the
                  feasible payload supported by that beam
          Step B: if the whole request was served, decrease n_kl and
                  re-optimize the precoder at each new n_kl
      - Saves per-block trajectories S for plotting.

    Returns:
      post_training_data_dict with:
        - L_out, n_star, F_star, R_star, norm_stats
        - all_user_block_results: list[user][block] = S (trajectory over n_kl)
        - B_used_star: list[user][block] = B_used at n=T (after reduction)
        - B_kl_star:   list[user][block] = bits actually transmitted in that block
    """
    K = int(uplinksystem.K)
    update_mode = validate_convergence_precoder_update_mode(
        sim_cfg.get("convergence_precoder_update_mode", "precoder_net")
    )

    L_out = [1] * K
    n_star = [[] for _ in range(K)]
    F_star = [[] for _ in range(K)]
    R_star = [[] for _ in range(K)]

    # extra bookkeeping (useful for plots/repro)
    B_used_star = [[] for _ in range(K)]
    B_kl_star = [[] for _ in range(K)]

    norm_stats = []
    all_user_block_results = [[] for _ in range(K)]

    user_precoder_models: list[nn.Module] = []
    remaining_bits_by_user = [int(v) for v in uplinksystem.B]
    block_index_by_user = [0 for _ in range(K)]

    for k in range(K):
        print(
            format_log_line(
                "[UL Convergence Train]",
                phase="start",
                user=int(k),
                mode="payload",
            )
        )

        norm_stats.append((0.0 + 0.0j, 1.0))

        # ---- remaining payload ----
        B_rem = int(uplinksystem.B[k])
        ell = 0

        # ensure at least one block exists
        while len(uplinksystem.H[k]) < 1:
            uplinksystem.add_block(k)

        if update_mode == "precoder_net":
            user_model = build_user_precoder_net_with_blocklength_and_sigma(
                Nr=int(uplinksystem.NR[k]),
                Nt=int(uplinksystem.NT[k]),
                dk=int(uplinksystem.dk[k]),
                device=DEVICE,
            )
            user_precoder_models.append(user_model)
        else:
            user_model = None

        while B_rem > 0:
            # ensure block exists
            if ell >= len(uplinksystem.H[k]):
                uplinksystem.add_block(k)
            block_index_by_user[k] = int(ell)

            print(
                format_log_line(
                    "[UL Convergence Train]",
                    user=int(k),
                    block=int(ell),
                    remaining_bits=int(B_rem),
                )
            )

            if update_mode == "direct_precoder":
                initial_precoder = np.asarray(uplinksystem.F[k][ell], dtype=np.complex64)
                precoder_param = complex_parameter_from_numpy(initial_precoder, device=DEVICE)
                user_optimizer = torch.optim.Adam([precoder_param], lr=float(sim_cfg["lr_net"]))
            else:
                precoder_param = None
                user_optimizer = torch.optim.Adam(user_model.parameters(), lr=float(sim_cfg["lr_net"]))

            # This function already performs:
            #  Step A: optimize once at n=T and clip to feasible served bits
            #  Step B: if this is the tail block, decrease n_kl and re-optimize
            S, B_used, _ = optimize_user_blocklength_and_precoder(
                uplinksystem=uplinksystem,
                user=k,
                block=ell,
                B_rem=B_rem,
                sim_cfg=sim_cfg,
                precoder_net=user_model,
                optimizer=user_optimizer,
                precoder_param=precoder_param,
                interference_F_snapshot=interference_F_snapshot,
                remaining_bits_by_user=remaining_bits_by_user,
                block_index_by_user=block_index_by_user,
            )

            if len(S) == 0 or B_used <= 0:
                print(
                    format_log_line(
                        "[UL Convergence Train]",
                        user=int(k),
                        block=int(ell),
                        status="stop_no_feasible_service",
                    )
                )
                break

            # Save trajectory for plotting (your plot_optimization_result_train expects this)
            all_user_block_results[k].append(S)

            # Pick smallest feasible n_kl in this block
            best = S[-1]
            n_opt = int(best["n_kl"])
            F_opt = best["F"]
            R_opt = float(best["R_fbl"])

            # Save chosen decisions
            n_star[k].append(n_opt)
            F_star[k].append(F_opt)
            R_star[k].append(R_opt)
            if commit_live_precoders:
                uplinksystem.F[k][ell] = F_opt.detach().cpu().numpy()


            # Save payload bookkeeping
            B_used_star[k].append(int(B_used))
            B_kl = min(B_rem, int(B_used))
            B_kl_star[k].append(int(B_kl))

            B_rem -= B_kl
            remaining_bits_by_user[k] = int(B_rem)
            print(
                format_log_line(
                    "[UL Convergence Allocation]",
                    user=int(k),
                    block=int(ell),
                    chosen_n_kl=int(n_opt),
                    served_bits=int(B_used),
                    committed_bits=int(B_kl),
                    remaining_bits=int(B_rem),
                )
            )

            # advance block if bits remain
            if B_rem > 0:
                ell += 1
                L_out[k] = ell + 1
                block_index_by_user[k] = int(ell)

        # write back to system bookkeeping (optional)
        uplinksystem.L[k] = int(L_out[k])
        if len(n_star[k]) > 0:
            uplinksystem.n_kl[k] = list(n_star[k])
    post_training_data_dict = {
        "L_out": L_out,
        "n_star": n_star,
        "F_star": F_star,
        "R_star": R_star,
        "norm_stats": norm_stats,
        "all_user_block_results_train": all_user_block_results,
        "B_used_star": B_used_star,
        "B_kl_star": B_kl_star,
        "user_model_specs": (
            export_user_model_specs(
                uplinksystem.NR,
                uplinksystem.NT,
                uplinksystem.dk,
                uses_blocklength_input=True,
                input_mode="channel_sigma_epsilon_n",
            )
            if update_mode == "precoder_net"
            else []
        ),
        "user_model_states": export_user_model_states(user_precoder_models) if update_mode == "precoder_net" else [],
        "convergence_precoder_update_mode": str(update_mode),
        "precoder_parameterization": (
            "shared_user_channel_n_sigma_epsilon_to_precoder_mlp_online_convergence"
            if update_mode == "precoder_net"
            else "direct_complex_precoder_per_user_block_online_convergence"
        ),
    }
    return post_training_data_dict

def optimize_streaming_blocks_with_precoder_training(
    uplinksystem,
    sim_cfg: dict,
    interference_F_snapshot=None,
    commit_live_precoders: bool = True,
):
    scenario = build_experiment_scenario(uplinksystem.sc, sim_cfg, seed=int(uplinksystem.seed))
    if str(scenario["mode"]) != STREAMING_MODE:
        raise ValueError("optimize_streaming_blocks_with_precoder_training requires streaming mode.")

    block_targets = np.asarray(scenario["streaming_bit_targets_by_block"], dtype=int)
    num_blocks = int(scenario["number_of_blocks"])
    K = int(uplinksystem.K)
    update_mode = validate_convergence_precoder_update_mode(
        sim_cfg.get("convergence_precoder_update_mode", "precoder_net")
    )

    L_out = [int(num_blocks)] * K
    n_star = [[] for _ in range(K)]
    F_star = [[] for _ in range(K)]
    R_star = [[] for _ in range(K)]
    B_used_star = [[] for _ in range(K)]
    B_kl_star = [[] for _ in range(K)]
    target_bits_star = [[] for _ in range(K)]
    unserved_bits_star = [[] for _ in range(K)]
    skipped_blocks_per_user = [0 for _ in range(K)]
    norm_stats = []
    all_user_block_results = [[] for _ in range(K)]

    user_precoder_models: list[nn.Module] = []
    block_index_by_user = [0 for _ in range(K)]

    for k in range(K):
        print(
            format_log_line(
                "[UL Convergence Train]",
                phase="start",
                user=int(k),
                mode="streaming",
            )
        )

        while len(uplinksystem.H[k]) < num_blocks:
            uplinksystem.add_block(k)

        norm_stats.append((0.0 + 0.0j, 1.0))

        if update_mode == "precoder_net":
            user_model = build_user_precoder_net_with_blocklength_and_sigma(
                Nr=int(uplinksystem.NR[k]),
                Nt=int(uplinksystem.NT[k]),
                dk=int(uplinksystem.dk[k]),
                device=DEVICE,
            )
            user_precoder_models.append(user_model)
        else:
            user_model = None

        for ell in range(num_blocks):
            while len(uplinksystem.H[k]) <= ell:
                uplinksystem.add_block(k)
            block_index_by_user = [int(ell) for _ in range(K)]

            target_bits = int(block_targets[k, ell])
            print(
                format_log_line(
                    "[UL Convergence Train]",
                    user=int(k),
                    block=int(ell),
                    target_bits=int(target_bits),
                    mode="streaming",
                )
            )
            if update_mode == "direct_precoder":
                initial_precoder = np.asarray(uplinksystem.F[k][ell], dtype=np.complex64)
                precoder_param = complex_parameter_from_numpy(initial_precoder, device=DEVICE)
                user_optimizer = torch.optim.Adam([precoder_param], lr=float(sim_cfg["lr_net"]))
            else:
                precoder_param = None
                user_optimizer = torch.optim.Adam(user_model.parameters(), lr=float(sim_cfg["lr_net"]))
            S, B_used, step_a_diagnostics = optimize_user_blocklength_and_precoder(
                uplinksystem=uplinksystem,
                user=k,
                block=ell,
                B_rem=int(target_bits),
                sim_cfg=sim_cfg,
                precoder_net=user_model,
                optimizer=user_optimizer,
                precoder_param=precoder_param,
                interference_F_snapshot=interference_F_snapshot,
                bit_budget_role="streaming",
                remaining_bits_by_user=block_targets[:, ell].astype(int).tolist(),
                block_index_by_user=block_index_by_user,
            )

            target_bits_star[k].append(int(target_bits))

            if len(S) == 0:
                H_kl = np.asarray(uplinksystem.H[k][ell], dtype=np.complex64)
                F_zero = np.zeros((int(uplinksystem.NT[k]), int(uplinksystem.dk[k])), dtype=np.complex64)
                zero_result = {
                    "n_kl": int(uplinksystem.T[k]),
                    "n": int(uplinksystem.T[k]),
                    "B_l": 0,
                    "Bits per sub-block length B/n_kl": 0.0,
                    "required_R_fbl": float(target_bits) / float(max(int(uplinksystem.T[k]), 1)),
                    "achieved_R_fbl": float(step_a_diagnostics.get("achieved_R_fbl", 0.0)),
                    "F": torch.tensor(F_zero, dtype=torch.complex64),
                    "R_fbl": float(step_a_diagnostics.get("achieved_R_fbl", 0.0)),
                    "F_power": 0.0,
                    "loss_curve": list(step_a_diagnostics.get("loss_curve", [])),
                    "kkt_history": list(step_a_diagnostics.get("kkt_history", [])),
                    "solve_status": step_a_diagnostics.get("solve_status", "unknown"),
                    "final_primal_residual": float(step_a_diagnostics.get("final_primal_residual", 0.0)),
                    "target_bits": int(target_bits),
                    "unserved_bits": int(target_bits),
                    "skipped": True,
                }
                S = [zero_result]
                B_used = 0
                skipped_blocks_per_user[k] += 1

            all_user_block_results[k].append(S)
            best = S[-1]
            n_opt = int(best["n_kl"])
            F_opt = best["F"]
            R_opt = float(best["R_fbl"])
            unserved_bits = max(0, int(target_bits) - int(B_used))

            n_star[k].append(int(n_opt))
            F_star[k].append(F_opt)
            R_star[k].append(float(R_opt))
            B_used_star[k].append(int(B_used))
            B_kl_star[k].append(int(B_used))
            unserved_bits_star[k].append(int(unserved_bits))
            if int(B_used) <= 0 and len(S) > 0 and not bool(S[-1].get("skipped", False)):
                skipped_blocks_per_user[k] += 1

            if commit_live_precoders:
                uplinksystem.F[k][ell] = F_opt.detach().cpu().numpy()

            print(
                format_log_line(
                    "[UL Convergence Allocation]",
                    user=int(k),
                    block=int(ell),
                    chosen_n_kl=int(n_opt),
                    served_bits=int(B_used),
                    unserved_bits=int(unserved_bits),
                    mode="streaming",
                )
            )

        uplinksystem.L[k] = int(num_blocks)
        uplinksystem.n_kl[k] = list(map(int, n_star[k]))

    post_training_data_dict = {
        "L_out": L_out,
        "n_star": n_star,
        "F_star": F_star,
        "R_star": R_star,
        "norm_stats": norm_stats,
        "all_user_block_results_train": all_user_block_results,
        "B_used_star": B_used_star,
        "B_kl_star": B_kl_star,
        "target_bits_star": target_bits_star,
        "unserved_bits_star": unserved_bits_star,
        "skipped_blocks_per_user": [int(v) for v in skipped_blocks_per_user],
        "scenario_mode": STREAMING_MODE,
        "scenario_block_targets": block_targets.tolist(),
        "user_model_specs": (
            export_user_model_specs(
                uplinksystem.NR,
                uplinksystem.NT,
                uplinksystem.dk,
                uses_blocklength_input=True,
                input_mode="channel_sigma_epsilon_n",
            )
            if update_mode == "precoder_net"
            else []
        ),
        "user_model_states": export_user_model_states(user_precoder_models) if update_mode == "precoder_net" else [],
        "convergence_precoder_update_mode": str(update_mode),
        "precoder_parameterization": (
            "shared_user_channel_n_sigma_epsilon_to_precoder_mlp_online_convergence"
            if update_mode == "precoder_net"
            else "direct_complex_precoder_per_user_block_online_convergence"
        ),
    }
    return post_training_data_dict

def evaluate_payload_with_trained_precoder_network(
    uplinksystem,
    post_training_data_dict: dict,
    sim_cfg: dict,
):
    """
    Testing version consistent with training_v2 block allocation, but WITHOUT precoder optimization:
      - Uses fixed trained precoders F_star (per user per block; fallback to last if needed)
      - Step A: at n=T, serve the feasible payload supported by the fixed F
      - Step B: if this is the tail block, reduce n_kl while keeping B_used fixed
      - Create additional blocks while B_rem > 0
      - Saves per-block trajectory S_test for plotting (same structure as training)

    Returns:
      test_data_dict with:
        - L_out_test, n_star_test, F_star_test, R_star_test
        - all_user_block_results_test: list[user][block] = S_test (trajectory over n_kl)
        - B_used_star_test, B_kl_star_test
        - norm_stats_used (the normalization stats applied)
    """
    K = int(uplinksystem.K)

    # from training
    F_star_train = post_training_data_dict["F_star"]
    norm_stats_train = post_training_data_dict["norm_stats"]
    user_model_specs = post_training_data_dict.get("user_model_specs")
    user_model_states = post_training_data_dict.get("user_model_states")
    user_models = None
    if user_model_specs is not None and user_model_states is not None:
        user_models = load_user_precoder_models(user_model_specs, user_model_states, device=DEVICE)

    # outputs
    L_out = [1] * K
    n_star = [[] for _ in range(K)]
    F_star = [[] for _ in range(K)]
    R_star = [[] for _ in range(K)]
    all_user_block_results = [[] for _ in range(K)]
    B_used_star = [[] for _ in range(K)]
    B_kl_star = [[] for _ in range(K)]

    for k in range(K):
        print(f"\n================ TEST USER {k} ================")

        P = float(uplinksystem.P[k])
        sigma2 = float(uplinksystem.sigma2[k])
        epsilon = float(uplinksystem.epsilon[k])
        T = int(uplinksystem.T[k])

        n_kl_min = int(sim_cfg["n_kl_min"])
        n_kl_step = int(sim_cfg["n_kl_step"])

        B_rem = int(uplinksystem.B[k])
        ell = 0

        while len(uplinksystem.H[k]) < 1:
            uplinksystem.add_block(k)

        while B_rem > 0:
            if ell >= len(uplinksystem.H[k]):
                uplinksystem.add_block(k)

            H_kl = np.array(uplinksystem.H[k][ell], dtype=np.complex64)
            if user_models is not None:
                shared_snapshot = build_precoder_snapshot_from_models(uplinksystem, user_models)
                F_fix = np.asarray(shared_snapshot[k][ell], dtype=np.complex64)
                F_fix_t = torch.tensor(F_fix, dtype=torch.complex64)
                noise_plus_interference_cov = build_uplink_rate_covariance(
                    uplinksystem,
                    sim_cfg,
                    k,
                    ell,
                    F_override=shared_snapshot,
                )
            else:
                # ---- fallback for older saved training artifacts ----
                if k < len(F_star_train) and ell < len(F_star_train[k]):
                    F_fix_t = F_star_train[k][ell]
                elif k < len(F_star_train) and len(F_star_train[k]) > 0:
                    print("Number of sub-blocks has exceeded expected value, no fixed precoder from training found, resorting to precoder from last block")
                    F_fix_t = F_star_train[k][-1]
                elif len(F_star_train[k]) == 0:
                    print(f">>> STOP test user {k}: no trained precoder available.")
                    break

                F_fix = F_fix_t.detach().cpu().numpy().astype(np.complex64)
                F_fix = cap_matrix_power_numpy(F_fix, P)
                noise_plus_interference_cov = build_uplink_rate_covariance(
                    uplinksystem,
                    sim_cfg,
                    k,
                    ell,
                    F_override=F_star_train,
                )

            print(f"\n--- TEST User {k}, Block {ell}, B_rem={B_rem} ---")

            # ==========================================================
            # STEP A: evaluate the fixed beam at n=T and serve the
            # feasible payload directly, without retrying.
            # ==========================================================
            R_T = evaluate_uplink_rate_numpy(
                H_kl, F_fix, sigma2, epsilon, n_kl=T,
                noise_plus_interference_covariance=noise_plus_interference_cov
            ).rate
            B_max = max(int(np.floor(float(T) * float(R_T))), 0)
            B_used = int(min(B_rem, B_max))

            print(
                f"n=T={T}, requested_bits={B_rem}, feasible_bits={B_max}, "
                f"served_bits={B_used}, R_fbl={R_T}"
            )

            if B_used <= 0:
                print(f">>> STOP test user {k} at block {ell}: cannot make n=T feasible.")
                break

            # Build S trajectory like training: include n=T point + feasible decreasing n_kl points
            S_block = []

            # n=T point
            R_T = evaluate_uplink_rate_numpy(
                H_kl, F_fix, sigma2, epsilon, n_kl=T,
                noise_plus_interference_covariance=noise_plus_interference_cov
            ).rate
            S_block.append({
                "n_kl": int(T),
                "n": int(T),
                "B_l": int(B_used),
                "Bits per sub-block length B/n_kl": float(B_used) / float(T),
                "F": F_fix_t,  # keep torch tensor for consistency with training plots/SNR plots
                "R_fbl": float(R_T),
                "F_power": float(np.linalg.norm(F_fix, "fro") ** 2),
                "loss_curve": [],
            })

            # ==========================================================
            # STEP B: decrease n_kl with B_used fixed (fixed F)
            # ==========================================================
            best_n = int(T)
            best_R = float(R_T)
            if int(B_used) < int(B_rem):
                print("Not reducing n_kl since this block only served a partial payload.")
            else:
                search_cfg = build_n_search_config(
                    n_min=int(n_kl_min),
                    n_max=int(T),
                    fine_step=int(n_kl_step),
                    direction=sim_cfg.get("n_search_direction", "descending"),
                    strategy=sim_cfg.get("n_search_strategy", "fixed_step"),
                    coarse_step=sim_cfg.get("n_search_coarse_step", int(n_kl_step)),
                    exponential_factor=sim_cfg.get("n_search_exponential_factor", 2),
                )
                reduction_search = run_n_frontier_search(
                    search_cfg,
                    lambda candidate_n, stage_name: {
                        "feasible": (
                            float(B_used) / float(max(int(candidate_n), 1))
                        ) <= evaluate_uplink_rate_numpy(
                            H_kl,
                            F_fix,
                            sigma2,
                            epsilon,
                            n_kl=int(candidate_n),
                            noise_plus_interference_covariance=noise_plus_interference_cov,
                        ).rate,
                        "R_candidate": evaluate_uplink_rate_numpy(
                            H_kl,
                            F_fix,
                            sigma2,
                            epsilon,
                            n_kl=int(candidate_n),
                            noise_plus_interference_covariance=noise_plus_interference_cov,
                        ).rate,
                        "search_stage": str(stage_name),
                    },
                )
                for accepted in reduction_search["accepted"]:
                    best_n = int(accepted["n_kl"])
                    best_R = float(accepted["result"]["R_candidate"])
                    print(
                        f"Test n_kl={best_n}: R_fbl={best_R}, "
                        f"search_stage={accepted['result']['search_stage']}, rate_violation=0.0"
                    )
                    S_block.append({
                        "n_kl": int(best_n),
                        "n": int(best_n),
                        "B_l": int(B_used),
                        "Bits per sub-block length B/n_kl": float(B_used) / float(best_n),
                        "F": F_fix_t,
                        "R_fbl": float(best_R),
                        "F_power": float(np.linalg.norm(F_fix, "fro") ** 2),
                        "loss_curve": [],
                    })
                rejected = reduction_search.get("frontier_rejected")
                if rejected is not None:
                    rejected_n = int(rejected["n_kl"])
                    rejected_R = float(rejected["result"]["R_candidate"])
                    rejected_gap = (float(B_used) / float(max(rejected_n, 1))) - rejected_R
                    print(
                        f"Test n_kl={rejected_n}: R_fbl={rejected_R}, "
                        f"search_stage={rejected['result']['search_stage']}, rate_violation={rejected_gap}"
                    )

            print(f">>> Chosen block {ell}: n_kl={best_n}, B_used={B_used}, R_fbl={best_R}")

            # save trajectory for plotting
            all_user_block_results[k].append(S_block)

            # save chosen decisions
            n_star[k].append(best_n)
            F_star[k].append(F_fix_t)
            R_star[k].append(best_R)
            B_used_star[k].append(int(B_used))

            B_kl = min(B_rem, int(B_used))
            B_kl_star[k].append(int(B_kl))
            B_rem -= B_kl
            print(f">>> Transmitted B_kl={B_kl}, remaining B_rem={B_rem}")

            if B_rem > 0:
                ell += 1
                L_out[k] = ell + 1
            else:
                ell += 1
                L_out[k] = ell
                
        
        uplinksystem.L[k] = int(L_out[k])
        if len(n_star[k]) > 0:
            uplinksystem.n_kl[k] = list(n_star[k])

    # push F into uplinksystem (numpy)
    for k in range(K):
        if len(F_star[k]) > 0:
            uplinksystem.F[k] = np.array([F.detach().cpu().numpy() for F in F_star[k]])

    try:
        uplinksystem.update_system()
    except Exception:
        pass

    test_data_dict = {
        "L_out_test": L_out,
        "n_star_test": n_star,
        "F_star_test": F_star,
        "R_star_test": R_star,
        "all_user_block_results_test": all_user_block_results,
        "B_used_star_test": B_used_star,
        "B_kl_star_test": B_kl_star,
        "norm_stats_used": norm_stats_train,
        "user_model_specs": user_model_specs,
        "user_model_states": user_model_states,
        "precoder_parameterization": post_training_data_dict.get(
            "precoder_parameterization",
            "per_block_precoders" if user_models is None else "shared_user_channel_n_sigma_epsilon_to_precoder_mlp",
        ),
    }
    return test_data_dict

def run_convergence_baseline(
    uplinksystem,
    sim_cfg: dict,
    interference_F_snapshot=None,
    commit_live_precoders: bool = True,
):
    """Run the configured online convergence allocation for one uplink system."""
    local_sim_cfg = dict(sim_cfg)
    effective_epochs = max(1, int(local_sim_cfg["max_epochs"]))
    local_sim_cfg["max_epochs"] = effective_epochs
    local_sim_cfg["max_precoder_epochs"] = effective_epochs

    runner = (
        optimize_streaming_blocks_with_precoder_training
        if str(local_sim_cfg.get("experiment_scenario_mode", "")) == STREAMING_MODE
        else optimize_payload_with_precoder_training
    )
    convergence_data = runner(
        uplinksystem=uplinksystem,
        sim_cfg=local_sim_cfg,
        interference_F_snapshot=interference_F_snapshot,
        commit_live_precoders=commit_live_precoders,
    )
    update_mode = validate_convergence_precoder_update_mode(
        local_sim_cfg["convergence_precoder_update_mode"]
    )
    convergence_data["convergence_precoder_update_mode"] = update_mode
    convergence_data["precoder_parameterization"] = (
        "shared_user_channel_n_sigma_epsilon_to_precoder_mlp_online_convergence"
        if update_mode == "precoder_net"
        else "direct_complex_precoder_per_user_block_online_convergence"
    )
    convergence_data["method_name"] = "convergence_per_epoch_baseline"
    convergence_data["configured_max_epochs"] = effective_epochs
    return convergence_data


__all__ = [
    "evaluate_payload_with_trained_precoder_network",
    "optimize_payload_with_precoder_training",
    "optimize_streaming_blocks_with_precoder_training",
    "optimize_user_blocklength_and_precoder",
    "run_convergence_baseline",
]
