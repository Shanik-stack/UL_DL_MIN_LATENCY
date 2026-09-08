"""Optimize independent uplink bit targets over a fixed streaming horizon."""

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

from .optimize_user_transmission import optimize_user_blocklength_and_precoder

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

__all__ = ["optimize_streaming_blocks_with_precoder_training"]
