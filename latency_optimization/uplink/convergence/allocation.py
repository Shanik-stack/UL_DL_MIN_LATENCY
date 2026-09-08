"""Uplink payload scheduling and blocklength allocation."""

import copy

import numpy as np
import torch
from torch import nn

from latency_optimization.core.blocklength import build_n_search_config, run_n_frontier_search
from latency_optimization.core.scenarios import STREAMING_MODE, build_experiment_scenario
from latency_optimization.precoders.parameters import complex_parameter
from latency_optimization.results.console import format_log_line
from latency_optimization.runtime import DEVICE

from ..precoders.checkpoints import (
    export_user_model_specs,
    export_user_model_states,
)
from ..precoders.models import build_user_precoder_net
from ..uplink_rate_model import build_uplink_rate_covariance
from ..objective import UplinkPrecoderObjective
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
            user_model = build_user_precoder_net(
                receive_antennas=int(uplinksystem.NR[k]),
                transmit_antennas=int(uplinksystem.NT[k]),
                streams=int(uplinksystem.dk[k]),
                device=DEVICE,
            )
            user_precoder_models.append(user_model)
        else:
            user_model = None

        while B_rem > 0:
            # ensure block exists
            if ell >= len(uplinksystem.H[k]):
                uplinksystem.add_block(k)
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
                precoder_param = complex_parameter(initial_precoder, device=DEVICE)
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
    """Solve every independent uplink streaming request in its assigned block.

    Each user has its own optimizer and beam; missed bits are measured but do
    not carry over because the next block starts a new streaming request.
    """
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
            user_model = build_user_precoder_net(
                receive_antennas=int(uplinksystem.NR[k]),
                transmit_antennas=int(uplinksystem.NT[k]),
                streams=int(uplinksystem.dk[k]),
                device=DEVICE,
            )
            user_precoder_models.append(user_model)
        else:
            user_model = None

        for ell in range(num_blocks):
            while len(uplinksystem.H[k]) <= ell:
                uplinksystem.add_block(k)
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
                precoder_param = complex_parameter(initial_precoder, device=DEVICE)
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
            )

            target_bits_star[k].append(int(target_bits))

            if len(S) == 0:
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


def run_convergence_baseline(
    uplinksystem,
    sim_cfg: dict,
    interference_F_snapshot=None,
    commit_live_precoders: bool = True,
):
    """Dispatch the uplink training-only solver using the configured traffic semantics.

    What: map the public convergence settings to the payload-completion or streaming
    allocator, run online beam optimization for every tested user/blocklength, and
    annotate the result with the direct-precoder or precoder-network parameterization.

    Why: callers need one stable entry point while payload and streaming intentionally
    use different bit-carry rules. ``interference_F_snapshot`` optionally freezes the
    other users' beams for interference evaluation, and ``commit_live_precoders``
    controls whether accepted beams mutate the supplied system.

    Returns: the common post-optimization schedule and convergence result schema.
    """
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
    "optimize_payload_with_precoder_training",
    "optimize_streaming_blocks_with_precoder_training",
    "optimize_user_blocklength_and_precoder",
    "run_convergence_baseline",
]
