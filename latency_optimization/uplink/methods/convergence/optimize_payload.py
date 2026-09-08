"""Optimize finite uplink payloads across generated channel blocks."""

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

__all__ = ["optimize_payload_with_precoder_training"]
