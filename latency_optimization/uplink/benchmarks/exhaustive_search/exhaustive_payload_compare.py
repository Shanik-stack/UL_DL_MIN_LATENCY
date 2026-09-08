import argparse
import contextlib
import copy
import csv
import io
import itertools
import math
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


from latency_optimization.experiments.scenarios import PAYLOAD_MODE
from latency_optimization.experiments.config_validation import require_choice
from latency_optimization.experiments.configuration import load_config_document
from latency_optimization.project import BENCHMARK_CONFIG_ROOT
from latency_optimization.precoders.parameters import complex_parameter
from latency_optimization.results.naming import (
    format_method_tag,
    format_update_mode_tag,
    join_tag_parts,
    make_method_result_tag,
)
from latency_optimization.results.metrics import (
    pairwise_latency_differences as _pairwise_latency_diffs,
)
from latency_optimization.results.paths import build_uplink_convergence_result_dirs
from latency_optimization.results.persistence import (
    current_local_timestamp,
    save_json,
    save_text,
    write_result_manifest,
)
from latency_optimization.runtime import DEVICE

from ...configuration.loader import load_config
from ...objectives.settings import (
    RATE_BEAM_REWARD_MODE,
    UNWEIGHTED_SUM_RATE_OBJECTIVE,
    validate_uplink_beam_reward_mode,
    validate_uplink_objective_mode,
)
from ...objectives.precoder import UplinkPrecoderObjective
from ...methods.convergence.optimize_precoder import (
    optimize_precoder_for_nl,
    validate_convergence_precoder_update_mode,
)
from ...methods.convergence.optimize_user_transmission import optimize_user_blocklength_and_precoder
from ...simulation.operations import ensure_blocks_up_to
from ...simulation.system import UplinkSystem
from ...physics.rate import UPLINK_RATE_MODEL_SNR, build_uplink_rate_covariance


METHOD_NAME = "small_exhaustive_payload_compare"
METHOD_LABEL = "Small Exhaustive validation"
CATALOG_MODE_NONE = "none"
CATALOG_MODE_OUTCOMES_ONLY = "outcomes_only"
CATALOG_MODE_FULL = "full"


def _load_raw_config(cfg_name: str) -> tuple[dict, str]:
    document = load_config_document(cfg_name)
    return document.data, str(document.path)


def _validate_experiment_inputs(system_params: dict, sim_cfg: dict) -> None:
    if int(system_params["K"]) != 2:
        raise ValueError("This validation experiment is intentionally limited to K=2 users.")
    if sim_cfg.get("experiment_scenario_mode", "") != PAYLOAD_MODE:
        raise ValueError("This validation experiment supports the payload scenario only.")
    if validate_convergence_precoder_update_mode(
        sim_cfg.get("convergence_precoder_update_mode", "precoder_net")
    ) != "direct_precoder":
        raise ValueError(
            "Use simulation.convergence_precoder_update_mode: direct_precoder for this validation. "
            "The exhaustive search is meant to validate the outer (B,n) rule, not a learned net state path."
        )
    if sim_cfg.get("uplink_rate_model", "") != UPLINK_RATE_MODEL_SNR:
        raise ValueError("Use simulation.uplink_rate_model: snr for this validation.")
    if validate_uplink_objective_mode(
        sim_cfg.get("uplink_objective_mode", UNWEIGHTED_SUM_RATE_OBJECTIVE)
    ) != UNWEIGHTED_SUM_RATE_OBJECTIVE:
        raise ValueError("Use simulation.uplink_objective_mode: unweighted_sum_rate for this validation.")
    if validate_uplink_beam_reward_mode(
        sim_cfg.get("beam_reward_mode", RATE_BEAM_REWARD_MODE)
    ) != RATE_BEAM_REWARD_MODE:
        raise ValueError("Use simulation.beam_reward_mode: rate for this validation.")


def validate_catalog_detail_mode(value: str) -> str:
    return require_choice(
        value,
        {CATALOG_MODE_NONE, CATALOG_MODE_OUTCOMES_ONLY, CATALOG_MODE_FULL},
        "exhaustive_compare.catalog_detail_mode",
    )


def _channel_uses_to_seconds(channel_uses: float, fs_hz: float) -> float:
    return float(channel_uses) / max(float(fs_hz), 1.0e-12)


def _enrich_strategy_summary_with_seconds(strategy_summary: dict, fs_per_user: list[float]) -> dict:
    per_user_latency_seconds = [
        _channel_uses_to_seconds(strategy_summary["per_user_latency"][k], fs_per_user[k])
        for k in range(len(strategy_summary.get("per_user_latency", [])))
    ]
    async_matrix, async_pairs, async_sum = _pairwise_latency_diffs(per_user_latency_seconds)
    strategy_summary["fs_per_user"] = [float(v) for v in fs_per_user]
    strategy_summary["per_user_latency_seconds"] = [float(v) for v in per_user_latency_seconds]
    strategy_summary["global_latency_sum_seconds"] = float(sum(per_user_latency_seconds))
    strategy_summary["asynchronality_matrix_seconds"] = async_matrix
    strategy_summary["asynchronality_pairs_seconds"] = async_pairs
    strategy_summary["asynchronality_sum_seconds"] = float(async_sum)
    return strategy_summary


def _build_strategy_action_rows(
    *,
    strategy_name: str,
    per_user_actions: list[list[dict]],
    total_payload_bits: list[int],
    fs_per_user: list[float],
) -> list[dict]:
    rows: list[dict] = []
    for user_idx, actions in enumerate(per_user_actions):
        remaining = int(total_payload_bits[user_idx])
        cumulative_channel_uses = 0
        cumulative_latency_seconds = 0.0
        for action in actions:
            latency_cost = int(action.get("latency_cost", 0) or 0)
            latency_seconds = _channel_uses_to_seconds(latency_cost, fs_per_user[user_idx])
            served_bits = int(action.get("served_bits", 0) or 0)
            action_type = str(action.get("action_type", "serve"))
            if action_type in {"serve", "skip"}:
                cumulative_channel_uses += latency_cost
                cumulative_latency_seconds += latency_seconds
                remaining = max(0, remaining - served_bits)
            rows.append(
                {
                    "strategy": str(strategy_name),
                    "user": int(user_idx),
                    "block": int(action.get("block", 0) or 0),
                    "action_type": action_type,
                    "requested_bits": int(action.get("requested_bits", remaining) or 0),
                    "served_bits": int(served_bits),
                    "n_kl": (None if action.get("n_kl", None) is None else int(action.get("n_kl", 0))),
                    "latency_channel_uses": int(latency_cost),
                    "latency_seconds": float(latency_seconds),
                    "cumulative_latency_channel_uses": int(cumulative_channel_uses),
                    "cumulative_latency_seconds": float(cumulative_latency_seconds),
                    "remaining_bits_after_action": int(remaining),
                    "achieved_rate": float(action.get("achieved_rate", float("nan"))),
                    "solve_status": str(action.get("solve_status", "")),
                }
            )
    return rows


def _build_strategy_overview_lines(
    strategy_name: str,
    strategy: dict,
    action_rows: list[dict],
) -> list[str]:
    lines = [
        str(strategy_name),
        f"  - Per-user final latency (s): {[round(float(v), 9) for v in strategy.get('per_user_latency_seconds', [])]}",
        f"  - Total latency sum (s): {float(strategy.get('global_latency_sum_seconds', 0.0)):.9f}",
        f"  - Final asynchronality sum (s): {float(strategy.get('asynchronality_sum_seconds', 0.0)):.9f}",
        f"  - Per-user served bits: {strategy.get('per_user_served_bits', [])}",
        f"  - Per-user remaining bits: {strategy.get('per_user_remaining_bits', [])}",
    ]
    for user_idx in sorted({int(row['user']) for row in action_rows}):
        lines.append(f"  - User {int(user_idx)} block plan:")
        user_rows = [row for row in action_rows if int(row["user"]) == int(user_idx)]
        if len(user_rows) <= 0:
            lines.append("      no actions")
            continue
        for row in user_rows:
            lines.append(
                "      "
                f"block {int(row['block'])}: action={row['action_type']}, bits={int(row['served_bits'])}, "
                f"n_kl={row['n_kl']}, latency_s={float(row['latency_seconds']):.9f}, "
                f"remaining_bits={int(row['remaining_bits_after_action'])}"
            )
    return lines


def _enrich_uplink_exhaustive_catalogs(exhaustive_summary: dict, fs_per_user: list[float]) -> dict:
    for action in exhaustive_summary.get("exact_pair_action_catalog", []):
        action["latency_seconds"] = _channel_uses_to_seconds(action.get("latency_cost", 0), fs_per_user[int(action.get("user", 0))])
    for user_idx, catalog in enumerate(exhaustive_summary.get("per_user_schedule_catalogs", [])):
        for row in catalog:
            row["total_latency_seconds"] = _channel_uses_to_seconds(row.get("total_latency", 0), fs_per_user[int(user_idx)])
    for row in exhaustive_summary.get("global_schedule_catalog", []):
        per_user_latency_seconds = [
            _channel_uses_to_seconds(row["per_user_latency"][k], fs_per_user[k])
            for k in range(len(row.get("per_user_latency", [])))
        ]
        _, async_pairs, async_sum = _pairwise_latency_diffs(per_user_latency_seconds)
        row["per_user_latency_seconds"] = [float(v) for v in per_user_latency_seconds]
        row["total_latency_sum_seconds"] = float(sum(per_user_latency_seconds))
        row["asynchronality_pairs_seconds"] = async_pairs
        row["asynchronality_sum_seconds"] = float(async_sum)
    best_schedule = exhaustive_summary.get("best_global_schedule")
    if isinstance(best_schedule, dict):
        per_user_latency_seconds = [
            _channel_uses_to_seconds(best_schedule["per_user_latency"][k], fs_per_user[k])
            for k in range(len(best_schedule.get("per_user_latency", [])))
        ]
        _, async_pairs, async_sum = _pairwise_latency_diffs(per_user_latency_seconds)
        best_schedule["per_user_latency_seconds"] = [float(v) for v in per_user_latency_seconds]
        best_schedule["total_latency_sum_seconds"] = float(sum(per_user_latency_seconds))
        best_schedule["asynchronality_pairs_seconds"] = async_pairs
        best_schedule["asynchronality_sum_seconds"] = float(async_sum)
    return exhaustive_summary


def _prepare_frozen_episode(
    system_params: dict,
    *,
    seed: int,
    max_blocks: int,
) -> UplinkSystem:
    base_system = UplinkSystem(system_params, seed=int(seed))
    ensure_blocks_up_to(base_system, int(max_blocks) - 1)
    return base_system


def _serialize_precoder(Fmat) -> list[list[list[float]]]:
    arr = np.asarray(Fmat, dtype=np.complex128)
    return [[[float(value.real), float(value.imag)] for value in row] for row in arr]


def _build_exact_action_solver(
    frozen_system: UplinkSystem,
    sim_cfg: dict,
    *,
    suppress_inner_logs: bool,
):
    lr_net = float(sim_cfg["lr_net"])
    max_epochs = max(1, int(sim_cfg["max_epochs"]))
    cache: dict[tuple[int, int, int, int], dict] = {}
    frontier_cache: dict[tuple[int, int, int], dict[int, dict]] = {}

    def _make_action_dict(
        *,
        user: int,
        block: int,
        bits: int,
        n_kl: int,
        out: dict,
        feasible: bool,
    ) -> dict:
        return {
            "user": int(user),
            "block": int(block),
            "requested_bits": int(bits),
            "served_bits": int(bits) if feasible else 0,
            "n_kl": int(n_kl),
            "latency_cost": int(n_kl),
            "feasible": feasible,
            "achieved_rate": float(out["R_fbl"]),
            "beam_reward": float(out["beam_reward"]),
            "power": float(out["F_power"]),
            "rate_gap": float(out["rate_gap"]),
            "power_gap": float(out["power_gap"]),
            "solve_status": str(out["solve_status"]),
            "final_primal_residual": float(out.get("final_primal_residual", 0.0)),
            "final_complementarity_residual": float(out.get("final_complementarity_residual", 0.0)),
            "F": np.asarray(out["F"].detach().cpu().numpy(), dtype=np.complex64),
        }

    def _run_inner_solve(
        *,
        user: int,
        block: int,
        bits: int,
        n_kl: int,
        loss_fn: UplinkPrecoderObjective,
        Nt: int,
        dk: int,
        optimizer: torch.optim.Optimizer,
        precoder_param: torch.nn.Parameter,
    ) -> dict:
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer) if suppress_inner_logs else contextlib.nullcontext():
            return optimize_precoder_for_nl(
                precoder_net=None,
                loss_fn=loss_fn,
                Nt=Nt,
                dk=dk,
                max_epochs=max_epochs,
                optimizer=optimizer,
                stopping_config=sim_cfg,
                print_every_epoch=max(1, int(sim_cfg.get("print_every_epoch", 1))),
                verbose=not suppress_inner_logs,
                log_context={
                    "user": int(user),
                    "block": int(block),
                    "bits": int(bits),
                    "n_kl": int(n_kl),
                    "mode": "exhaustive_frontier",
                },
                precoder_param=precoder_param,
                update_mode="direct_precoder",
            )

    def _build_frontier_for_bits(user: int, block: int, bits: int) -> dict[int, dict]:
        frontier_key = (int(user), int(block), int(bits))
        cached_frontier = frontier_cache.get(frontier_key)
        if cached_frontier is not None:
            return cached_frontier

        P = float(frozen_system.P[user])
        Nt = int(frozen_system.NT[user])
        dk = int(frozen_system.dk[user])
        sigma2 = float(frozen_system.sigma2[user])
        epsilon = float(frozen_system.epsilon[user])
        T_user = int(frozen_system.T[user])
        H_kl = torch.tensor(
            np.asarray(frozen_system.H[user][block], dtype=np.complex64),
            dtype=torch.complex64,
            device=DEVICE,
        )
        noise_cov_np = build_uplink_rate_covariance(
            frozen_system,
            sim_cfg,
            int(user),
            int(block),
        )
        noise_cov = (
            None
            if noise_cov_np is None
            else torch.tensor(noise_cov_np, dtype=torch.complex64, device=DEVICE)
        )
        loss_fn = UplinkPrecoderObjective(
            channel=H_kl,
            noise_variance=sigma2,
            epsilon=epsilon,
            payload_bits=float(bits),
            power_limit=P,
            blocklength=int(T_user),
            noise_plus_interference_covariance=noise_cov,
            rate_law=frozen_system.rate_law,
        ).to(DEVICE)
        initial_precoder = np.asarray(frozen_system.F[user][block], dtype=np.complex64)
        precoder_param = complex_parameter(initial_precoder, device=DEVICE)
        optimizer = torch.optim.Adam([precoder_param], lr=lr_net)

        frontier: dict[int, dict] = {}
        out_t = _run_inner_solve(
            user=int(user),
            block=int(block),
            bits=int(bits),
            n_kl=int(T_user),
            loss_fn=loss_fn,
            Nt=Nt,
            dk=dk,
            optimizer=optimizer,
            precoder_param=precoder_param,
        )
        feasible_t = bool(float(out_t["rate_gap"]) <= 0.0 and float(out_t["power_gap"]) <= 0.0)
        frontier[int(T_user)] = _make_action_dict(
            user=int(user),
            block=int(block),
            bits=int(bits),
            n_kl=int(T_user),
            out=out_t,
            feasible=feasible_t,
        )

        if feasible_t:
            for candidate_n in range(int(T_user) - 1, 0, -1):
                model_checkpoint = precoder_param.detach().cpu().clone()
                optimizer_checkpoint = copy.deepcopy(optimizer.state_dict())

                loss_fn.set_blocklength(int(candidate_n))
                loss_fn.set_payload(float(bits))
                out_candidate = _run_inner_solve(
                    user=int(user),
                    block=int(block),
                    bits=int(bits),
                    n_kl=int(candidate_n),
                    loss_fn=loss_fn,
                    Nt=Nt,
                    dk=dk,
                    optimizer=optimizer,
                    precoder_param=precoder_param,
                )
                feasible_candidate = bool(
                    float(out_candidate["rate_gap"]) <= 0.0 and float(out_candidate["power_gap"]) <= 0.0
                )
                frontier[int(candidate_n)] = _make_action_dict(
                    user=int(user),
                    block=int(block),
                    bits=int(bits),
                    n_kl=int(candidate_n),
                    out=out_candidate,
                    feasible=feasible_candidate,
                )
                if not feasible_candidate:
                    with torch.no_grad():
                        precoder_param.copy_(model_checkpoint.to(device=DEVICE, dtype=precoder_param.dtype))
                    optimizer.load_state_dict(optimizer_checkpoint)
                    break

        frontier_cache[frontier_key] = frontier
        for n_value, action in frontier.items():
            cache[(int(user), int(block), int(bits), int(n_value))] = action
        return frontier

    def solve_exact_pair(user: int, block: int, bits: int, n_kl: int) -> dict:
        key = (int(user), int(block), int(bits), int(n_kl))
        cached = cache.get(key)
        if cached is not None:
            return cached

        frontier = _build_frontier_for_bits(int(user), int(block), int(bits))
        frontier_action = frontier.get(int(n_kl))
        if frontier_action is not None:
            return frontier_action

        infeasible_action = {
            "user": int(user),
            "block": int(block),
            "requested_bits": int(bits),
            "served_bits": 0,
            "n_kl": int(n_kl),
            "latency_cost": int(n_kl),
            "feasible": False,
            "achieved_rate": float("-inf"),
            "beam_reward": float("-inf"),
            "power": float("inf"),
            "rate_gap": float("inf"),
            "power_gap": float("inf"),
            "solve_status": "frontier_not_reached",
            "final_primal_residual": float("inf"),
            "final_complementarity_residual": float("inf"),
            "F": np.asarray(frozen_system.F[user][block], dtype=np.complex64),
        }
        cache[key] = infeasible_action
        return infeasible_action

    solve_exact_pair.cache = cache  # type: ignore[attr-defined]
    solve_exact_pair.frontier_cache = frontier_cache  # type: ignore[attr-defined]
    return solve_exact_pair


def _run_current_online_strategy(
    frozen_system: UplinkSystem,
    sim_cfg: dict,
    *,
    max_blocks: int,
    suppress_inner_logs: bool,
) -> dict:
    system = copy.deepcopy(frozen_system)
    K = int(system.K)
    per_user_actions: list[list[dict]] = [[] for _ in range(K)]

    for k in range(K):
        B_rem = int(system.B[k])
        ell = 0
        while B_rem > 0 and ell < int(max_blocks):
            initial_precoder = np.asarray(system.F[k][ell], dtype=np.complex64)
            precoder_param = complex_parameter(initial_precoder, device=DEVICE)
            user_optimizer = torch.optim.Adam([precoder_param], lr=float(sim_cfg["lr_net"]))

            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer) if suppress_inner_logs else contextlib.nullcontext():
                trajectory, B_used, step_a = optimize_user_blocklength_and_precoder(
                    uplinksystem=system,
                    user=int(k),
                    block=int(ell),
                    B_rem=int(B_rem),
                    sim_cfg=sim_cfg,
                    precoder_net=None,
                    optimizer=user_optimizer,
                    precoder_param=precoder_param,
                    interference_F_snapshot=None,
                )

            if len(trajectory) == 0 or int(B_used) <= 0:
                per_user_actions[k].append(
                    {
                        "user": int(k),
                        "block": int(ell),
                        "action_type": "stop_no_service",
                        "requested_bits": int(B_rem),
                        "served_bits": 0,
                        "n_kl": None,
                        "latency_cost": 0,
                        "achieved_rate_at_T": float(step_a.get("achieved_R_fbl", 0.0)),
                        "solve_status_at_T": str(step_a.get("solve_status", "unknown")),
                    }
                )
                break

            best = trajectory[-1]
            chosen_n = int(best["n_kl"])
            chosen_bits = int(B_used)
            chosen_precoder = np.asarray(best["F"].detach().cpu().numpy(), dtype=np.complex64)
            system.n_kl[k][ell] = int(chosen_n)
            system.F[k][ell] = chosen_precoder
            per_user_actions[k].append(
                {
                    "user": int(k),
                    "block": int(ell),
                    "action_type": "serve",
                    "requested_bits": int(B_rem),
                    "served_bits": int(chosen_bits),
                    "n_kl": int(chosen_n),
                    "latency_cost": int(chosen_n),
                    "achieved_rate": float(best["R_fbl"]),
                    "power": float(best["F_power"]),
                    "rate_gap": float(max(float(chosen_bits) / max(float(chosen_n), 1.0) - float(best["R_fbl"]), 0.0)),
                    "power_gap": float(best.get("F_power", 0.0) - float(system.P[k])),
                    "solve_status": str(best.get("solve_status", "unknown")),
                    "F": chosen_precoder.copy(),
                }
            )
            B_rem = max(0, int(B_rem) - int(chosen_bits))
            ell += 1

        if B_rem > 0 and ell >= int(max_blocks):
            per_user_actions[k].append(
                {
                    "user": int(k),
                    "block": int(max_blocks),
                    "action_type": "horizon_exhausted",
                    "requested_bits": int(B_rem),
                    "served_bits": 0,
                    "n_kl": None,
                    "latency_cost": 0,
                }
            )

    return _summarize_strategy(
        per_user_actions=per_user_actions,
        total_payload_bits=[int(v) for v in system.B],
        max_blocks=int(max_blocks),
    )


def _summarize_strategy(
    *,
    per_user_actions: list[list[dict]],
    total_payload_bits: list[int],
    max_blocks: int,
) -> dict:
    per_user_latency = []
    per_user_served_bits = []
    per_user_completed = []
    per_user_block_count = []
    remaining_after_each_block = []

    for user_idx, actions in enumerate(per_user_actions):
        served_total = 0
        latency_total = 0
        block_trace = []
        remaining = int(total_payload_bits[user_idx])
        used_blocks = 0
        for action in actions:
            action_type = str(action.get("action_type", "serve"))
            if action_type in {"serve", "skip"}:
                used_blocks += 1
                served_total += int(action.get("served_bits", 0))
                latency_total += int(action.get("latency_cost", 0))
                remaining = max(0, remaining - int(action.get("served_bits", 0)))
                block_trace.append(int(remaining))
        per_user_latency.append(int(latency_total))
        per_user_served_bits.append(int(served_total))
        per_user_completed.append(bool(served_total >= int(total_payload_bits[user_idx])))
        per_user_block_count.append(int(used_blocks))
        remaining_after_each_block.append(block_trace)

    return {
        "per_user_actions": per_user_actions,
        "per_user_latency": per_user_latency,
        "per_user_served_bits": per_user_served_bits,
        "per_user_remaining_bits": [
            max(0, int(total_payload_bits[k]) - int(per_user_served_bits[k]))
            for k in range(len(total_payload_bits))
        ],
        "per_user_completed": per_user_completed,
        "per_user_block_count": per_user_block_count,
        "remaining_after_each_block": remaining_after_each_block,
        "global_latency_sum": int(sum(per_user_latency)),
        "global_served_bits": int(sum(per_user_served_bits)),
        "all_completed": bool(all(per_user_completed)),
        "max_blocks": int(max_blocks),
    }


def _run_exhaustive_strategy(
    frozen_system: UplinkSystem,
    sim_cfg: dict,
    *,
    max_blocks: int,
    allow_block_skip: bool,
    suppress_inner_logs: bool,
    catalog_detail_mode: str,
) -> dict:
    K = int(frozen_system.K)
    total_payload_bits = [int(v) for v in frozen_system.B]
    build_catalogs = catalog_detail_mode in {CATALOG_MODE_OUTCOMES_ONLY, CATALOG_MODE_FULL}
    store_action_details = catalog_detail_mode == CATALOG_MODE_FULL
    solve_exact_pair = _build_exact_action_solver(
        frozen_system,
        sim_cfg,
        suppress_inner_logs=bool(suppress_inner_logs),
    )
    per_user_actions: list[list[dict]] = [[] for _ in range(K)]
    per_user_schedule_catalogs: list[list[dict]] = []

    for user_idx in range(K):
        T_user = int(frozen_system.T[user_idx])
        payload_bits = int(total_payload_bits[user_idx])

        @lru_cache(maxsize=None)
        def dp(block: int, remaining_bits: int):
            if remaining_bits <= 0:
                return 0, []
            if block >= int(max_blocks):
                return math.inf, None

            best_cost = math.inf
            best_plan = None

            if allow_block_skip:
                suffix_cost, suffix_plan = dp(block + 1, int(remaining_bits))
                if suffix_plan is not None:
                    best_cost = int(T_user) + float(suffix_cost)
                    best_plan = [
                        {
                            "user": int(user_idx),
                            "block": int(block),
                            "action_type": "skip",
                            "requested_bits": int(remaining_bits),
                            "served_bits": 0,
                            "n_kl": int(T_user),
                            "latency_cost": int(T_user),
                        }
                    ] + list(suffix_plan)

            for bits in range(int(remaining_bits), 0, -1):
                for n_kl in range(1, int(T_user) + 1):
                    action = solve_exact_pair(int(user_idx), int(block), int(bits), int(n_kl))
                    if not bool(action["feasible"]):
                        continue
                    suffix_cost, suffix_plan = dp(block + 1, int(remaining_bits) - int(bits))
                    if suffix_plan is None:
                        continue
                    total_cost = int(n_kl) + float(suffix_cost)
                    if total_cost < best_cost - 1e-9:
                        best_cost = total_cost
                        best_plan = [
                            {
                                **action,
                                "action_type": "serve",
                            }
                        ] + list(suffix_plan)

            return best_cost, best_plan

        _, best_plan = dp(0, int(payload_bits))
        per_user_actions[user_idx] = [] if best_plan is None else list(best_plan)
        if build_catalogs:
            per_user_schedule_catalogs.append(
                _enumerate_uplink_user_schedule_catalog(
                    user_idx=int(user_idx),
                    payload_bits=int(payload_bits),
                    T_user=int(T_user),
                    max_blocks=int(max_blocks),
                    allow_block_skip=bool(allow_block_skip),
                    solve_exact_pair=solve_exact_pair,
                    store_action_details=bool(store_action_details),
                )
            )

    strategy_summary = _summarize_strategy(
        per_user_actions=per_user_actions,
        total_payload_bits=total_payload_bits,
        max_blocks=int(max_blocks),
    )
    if build_catalogs:
        global_schedule_catalog, best_global_schedule = _build_uplink_global_schedule_catalog(
            per_user_schedule_catalogs
        )
    else:
        global_schedule_catalog = []
        best_global_schedule = {
            "global_schedule_id": 1,
            "all_completed": bool(strategy_summary["all_completed"]),
            "total_latency_sum": int(strategy_summary["global_latency_sum"]),
            "total_remaining_bits": int(sum(int(v) for v in strategy_summary["per_user_remaining_bits"])),
            "per_user_schedule_ids": [1 for _ in range(K)],
            "per_user_latency": [int(v) for v in strategy_summary["per_user_latency"]],
            "per_user_remaining_bits": [int(v) for v in strategy_summary["per_user_remaining_bits"]],
            "per_user_action_signatures": [
                _action_signature(user_actions) for user_actions in per_user_actions
            ],
        }
    strategy_summary["unique_exact_pair_solves"] = int(len(getattr(solve_exact_pair, "cache", {})))
    strategy_summary["exact_pair_action_catalog"] = (
        _flatten_uplink_exact_pair_catalog(solve_exact_pair) if store_action_details else []
    )
    strategy_summary["per_user_schedule_catalogs"] = per_user_schedule_catalogs
    strategy_summary["global_schedule_catalog"] = global_schedule_catalog
    strategy_summary["best_global_schedule"] = best_global_schedule
    strategy_summary["catalog_detail_mode"] = catalog_detail_mode
    strategy_summary["catalog_counts"] = {
        "exact_pair_actions": int(len(strategy_summary["exact_pair_action_catalog"])) if store_action_details else None,
        "per_user_schedule_counts": [int(len(catalog)) for catalog in per_user_schedule_catalogs] if build_catalogs else None,
        "global_schedule_count": int(len(global_schedule_catalog)) if build_catalogs else None,
    }
    return strategy_summary


def _strategy_actions_to_serializable(per_user_actions: list[list[dict]]) -> list[list[dict]]:
    serializable: list[list[dict]] = []
    for user_actions in per_user_actions:
        user_serialized: list[dict] = []
        for action in user_actions:
            action_copy = dict(action)
            if "F" in action_copy and action_copy["F"] is not None:
                action_copy["F"] = _serialize_precoder(action_copy["F"])
            user_serialized.append(action_copy)
        serializable.append(user_serialized)
    return serializable


def _strip_action_for_catalog(action: dict) -> dict:
    action_copy = dict(action)
    if "F" in action_copy:
        action_copy.pop("F", None)
    return action_copy


def _action_signature(actions: list[dict]) -> str:
    tokens: list[str] = []
    for action in actions:
        action_type = str(action.get("action_type", "serve"))
        block = int(action.get("block", 0))
        if action_type == "serve":
            tokens.append(
                f"b{block}:serve(B={int(action.get('served_bits', 0))},n={int(action.get('n_kl', 0))})"
            )
        elif action_type == "skip":
            tokens.append(f"b{block}:skip(n={int(action.get('n_kl', 0))})")
        else:
            tokens.append(f"b{block}:{action_type}")
    return " | ".join(tokens)


def _uplink_bn_plan_from_actions(actions: list[dict]) -> tuple[list[int], list[int], list[dict]]:
    b_plan: list[int] = []
    n_plan: list[int] = []
    block_rows: list[dict] = []
    for action in actions:
        action_type = str(action.get("action_type", "serve"))
        if action_type not in {"serve", "skip"}:
            continue
        served_bits = int(action.get("served_bits", 0))
        n_kl = int(action.get("n_kl", 0) or 0)
        b_plan.append(int(served_bits))
        n_plan.append(int(n_kl))
        block_rows.append(
            {
                "block": int(action.get("block", 0)),
                "served_bits": int(served_bits),
                "n_kl": int(n_kl),
                "action_type": action_type,
            }
        )
    return b_plan, n_plan, block_rows


def _enumerate_uplink_user_schedule_catalog(
    *,
    user_idx: int,
    payload_bits: int,
    T_user: int,
    max_blocks: int,
    allow_block_skip: bool,
    solve_exact_pair,
    store_action_details: bool,
) -> list[dict]:
    catalog: list[dict] = []
    schedule_counter = 0

    def finalize(actions: list[dict], remaining_bits: int, reason: str) -> None:
        nonlocal schedule_counter
        schedule_counter += 1
        served_bits = int(payload_bits) - int(max(remaining_bits, 0))
        total_latency = int(
            sum(int(action.get("latency_cost", 0)) for action in actions if str(action.get("action_type", "")) in {"serve", "skip"})
        )
        row = {
            "schedule_id": int(schedule_counter),
            "user": int(user_idx),
            "completed": bool(remaining_bits <= 0),
            "terminal_reason": str(reason),
            "served_bits_total": int(served_bits),
            "remaining_bits_final": int(max(remaining_bits, 0)),
            "total_latency": int(total_latency),
            "blocks_used": int(
                sum(1 for action in actions if str(action.get("action_type", "")) in {"serve", "skip"})
            ),
            "action_signature": _action_signature(actions),
        }
        b_plan, n_plan, block_rows = _uplink_bn_plan_from_actions(actions)
        row["B_kl_plan"] = [int(v) for v in b_plan]
        row["n_kl_plan"] = [int(v) for v in n_plan]
        row["block_plan"] = block_rows
        if bool(store_action_details):
            row["actions"] = [_strip_action_for_catalog(action) for action in actions]
        catalog.append(row)

    def dfs(block: int, remaining_bits: int, actions: list[dict]) -> None:
        if remaining_bits <= 0:
            finalize(actions, int(remaining_bits), "completed")
            return
        if block >= int(max_blocks):
            finalize(actions, int(remaining_bits), "horizon_exhausted")
            return

        if allow_block_skip:
            dfs(
                block + 1,
                int(remaining_bits),
                actions
                + [
                    {
                        "user": int(user_idx),
                        "block": int(block),
                        "action_type": "skip",
                        "requested_bits": int(remaining_bits),
                        "served_bits": 0,
                        "n_kl": int(T_user),
                        "latency_cost": int(T_user),
                    }
                ],
            )

        for bits in range(int(remaining_bits), 0, -1):
            for n_kl in range(1, int(T_user) + 1):
                action = solve_exact_pair(int(user_idx), int(block), int(bits), int(n_kl))
                if not bool(action.get("feasible", False)):
                    continue
                dfs(
                    block + 1,
                    int(remaining_bits) - int(bits),
                    actions + [{**_strip_action_for_catalog(action), "action_type": "serve"}],
                )

    dfs(0, int(payload_bits), [])
    catalog.sort(
        key=lambda row: (
            not bool(row["completed"]),
            int(row["total_latency"]),
            int(row["remaining_bits_final"]),
            int(row["schedule_id"]),
        )
    )
    return catalog


def _build_uplink_global_schedule_catalog(
    per_user_schedule_catalogs: list[list[dict]],
) -> tuple[list[dict], dict | None]:
    if any(len(catalog) == 0 for catalog in per_user_schedule_catalogs):
        return [], None

    global_rows: list[dict] = []
    best_row: dict | None = None
    best_key: tuple | None = None

    for combo_id, combo in enumerate(itertools.product(*per_user_schedule_catalogs), start=1):
        total_latency = int(sum(int(entry["total_latency"]) for entry in combo))
        total_remaining = int(sum(int(entry["remaining_bits_final"]) for entry in combo))
        all_completed = bool(all(bool(entry["completed"]) for entry in combo))
        row = {
            "global_schedule_id": int(combo_id),
            "all_completed": bool(all_completed),
            "total_latency_sum": int(total_latency),
            "total_remaining_bits": int(total_remaining),
            "per_user_schedule_ids": [int(entry["schedule_id"]) for entry in combo],
            "per_user_latency": [int(entry["total_latency"]) for entry in combo],
            "per_user_remaining_bits": [int(entry["remaining_bits_final"]) for entry in combo],
            "per_user_action_signatures": [str(entry["action_signature"]) for entry in combo],
            "per_user_B_kl_plans": [entry.get("B_kl_plan", []) for entry in combo],
            "per_user_n_kl_plans": [entry.get("n_kl_plan", []) for entry in combo],
            "per_user_block_plans": [entry.get("block_plan", []) for entry in combo],
        }
        global_rows.append(row)
        current_key = (
            not bool(all_completed),
            int(total_latency),
            int(total_remaining),
            tuple(int(entry["schedule_id"]) for entry in combo),
        )
        if best_key is None or current_key < best_key:
            best_key = current_key
            best_row = dict(row)

    global_rows.sort(
        key=lambda row: (
            not bool(row["all_completed"]),
            int(row["total_latency_sum"]),
            int(row["total_remaining_bits"]),
            tuple(int(v) for v in row["per_user_schedule_ids"]),
        )
    )
    return global_rows, best_row


def _flatten_uplink_exact_pair_catalog(solve_exact_pair) -> list[dict]:
    rows: list[dict] = []
    for key in sorted(getattr(solve_exact_pair, "cache", {}).keys()):
        action = getattr(solve_exact_pair, "cache", {})[key]
        rows.append(_strip_action_for_catalog(action))
    return rows


def _write_csv_rows(rows: list[dict], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) <= 0:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {
                key: (
                    ",".join(map(str, value))
                    if isinstance(value, (list, tuple))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(normalized)


def _build_exhaustive_outcome_rows(global_schedule_catalog: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in global_schedule_catalog:
        rows.append(
            {
                "global_schedule_id": int(row.get("global_schedule_id", 0)),
                "completed": bool(row.get("all_completed", False)),
                "total_latency_sum_seconds": float(row.get("total_latency_sum_seconds", 0.0)),
                "asynchronality_sum_seconds": float(row.get("asynchronality_sum_seconds", 0.0)),
                "total_remaining_bits": int(row.get("total_remaining_bits", 0)),
                "per_user_latency_seconds": row.get("per_user_latency_seconds", []),
                "per_user_remaining_bits": row.get("per_user_remaining_bits", []),
                "per_user_schedule_ids": row.get("per_user_schedule_ids", []),
                "per_user_B_kl_plans": row.get("per_user_B_kl_plans", []),
                "per_user_n_kl_plans": row.get("per_user_n_kl_plans", []),
                "per_user_block_plans": row.get("per_user_block_plans", []),
                "per_user_action_signatures": row.get("per_user_action_signatures", []),
            }
        )
    rows.sort(
        key=lambda entry: (
            float(entry.get("total_latency_sum_seconds", 0.0)),
            float(entry.get("asynchronality_sum_seconds", 0.0)),
            int(entry.get("total_remaining_bits", 0)),
            int(entry.get("global_schedule_id", 0)),
        ),
        reverse=True,
    )
    return rows


def _sort_exhaustive_outcome_rows_best_first(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda entry: (
            float(entry.get("total_latency_sum_seconds", 0.0)),
            float(entry.get("asynchronality_sum_seconds", 0.0)),
            int(entry.get("total_remaining_bits", 0)),
            int(entry.get("global_schedule_id", 0)),
        ),
    )


def _plot_latency_comparison(result: dict, save_dir: str) -> None:
    online = result["online_strategy"]["per_user_latency_seconds"]
    exhaustive = result["exhaustive_strategy"]["per_user_latency_seconds"]
    indices = np.arange(len(online) + 1)
    width = 0.35
    online_values = list(online) + [float(result["online_strategy"]["global_latency_sum_seconds"])]
    exhaustive_values = list(exhaustive) + [float(result["exhaustive_strategy"]["global_latency_sum_seconds"])]
    labels = [f"user {idx}" for idx in range(len(online))] + ["total"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(indices - width / 2, online_values, width=width, label="current online")
    ax.bar(indices + width / 2, exhaustive_values, width=width, label="exhaustive")
    ax.set_xticks(indices)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Current online rule vs exhaustive search")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(save_dir) / "latency_comparison.png", dpi=200)
    plt.close(fig)


def _plot_remaining_bits_trajectories(result: dict, save_dir: str) -> None:
    online = result["online_strategy"]["remaining_after_each_block"]
    exhaustive = result["exhaustive_strategy"]["remaining_after_each_block"]
    total_bits = result["payload_bits"]
    num_users = len(total_bits)
    fig, axes = plt.subplots(num_users, 1, figsize=(8, max(3.2, 2.8 * num_users)), sharex=False)
    if num_users == 1:
        axes = [axes]

    for user_idx, ax in enumerate(axes):
        online_trace = [int(total_bits[user_idx])] + list(map(int, online[user_idx]))
        exhaustive_trace = [int(total_bits[user_idx])] + list(map(int, exhaustive[user_idx]))
        ax.step(range(len(online_trace)), online_trace, where="post", label="current online")
        ax.step(range(len(exhaustive_trace)), exhaustive_trace, where="post", label="exhaustive")
        ax.set_ylabel(f"user {user_idx} bits")
        ax.set_title(f"user {user_idx} remaining payload")
        ax.grid(alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("Block index")
    fig.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(save_dir) / "remaining_bits_trajectories.png", dpi=200)
    plt.close(fig)


def _build_summary_lines(result: dict) -> list[str]:
    online = result["online_strategy"]
    exhaustive = result["exhaustive_strategy"]
    catalog_counts = exhaustive.get("catalog_counts", {})
    best_global_schedule = exhaustive.get("best_global_schedule")
    latency_gap_seconds = float(online["global_latency_sum_seconds"]) - float(exhaustive["global_latency_sum_seconds"])
    latency_gap_pct = (
        100.0 * float(latency_gap_seconds) / max(float(online["global_latency_sum_seconds"]), 1.0e-12)
        if float(online["global_latency_sum_seconds"]) > 0.0
        else 0.0
    )
    async_gap_seconds = float(online.get("asynchronality_sum_seconds", 0.0)) - float(exhaustive.get("asynchronality_sum_seconds", 0.0))
    lines = [
        "Uplink exhaustive payload-completion validation",
        f"Run started (local): {result['run_started_at_local']}",
        f"Run completed (local): {result['run_completed_at_local']}",
        f"Config: {result['cfg_path']}",
        f"Seed: {result['seed']}",
        f"Payload bits per user: {result['payload_bits']}",
        f"T per user: {result['T_per_user']}",
        f"fs per user (channel uses/s): {result['fs_per_user']}",
        f"Max compared blocks per user: {result['max_compared_blocks_per_user']}",
        f"Allow exhaustive skip actions: {result['allow_block_skip']}",
        f"Suppressed detailed inner logs: {result['suppress_inner_logs']}",
        f"Catalog detail mode: {result['catalog_detail_mode']}",
        "",
        "Assumptions",
        "  - Uplink SNR mode only.",
        "  - Direct-precoder inner solve only.",
        "  - Unweighted rate reward inside the beam optimizer.",
        "  - Exact same frozen channel episode is used for both strategies.",
        "",
        "Current online rule",
        f"  - Completed all payloads: {online['all_completed']}",
        f"  - Per-user final latency (s): {[round(float(v), 9) for v in online['per_user_latency_seconds']]}",
        f"  - Total latency sum (s): {float(online['global_latency_sum_seconds']):.9f}",
        f"  - Final asynchronality sum (s): {float(online['asynchronality_sum_seconds']):.9f}",
        f"  - Per-user served bits: {online['per_user_served_bits']}",
        f"  - Per-user remaining bits: {online['per_user_remaining_bits']}",
        "",
        "Exhaustive search",
        f"  - Completed all payloads: {exhaustive['all_completed']}",
        f"  - Per-user final latency (s): {[round(float(v), 9) for v in exhaustive['per_user_latency_seconds']]}",
        f"  - Total latency sum (s): {float(exhaustive['global_latency_sum_seconds']):.9f}",
        f"  - Final asynchronality sum (s): {float(exhaustive['asynchronality_sum_seconds']):.9f}",
        f"  - Per-user served bits: {exhaustive['per_user_served_bits']}",
        f"  - Per-user remaining bits: {exhaustive['per_user_remaining_bits']}",
        f"  - Unique exact (B,n) solves: {exhaustive['unique_exact_pair_solves']}",
        f"  - Exact pair catalog size: {catalog_counts.get('exact_pair_actions', 0)}",
        f"  - Per-user schedule counts: {catalog_counts.get('per_user_schedule_counts', [])}",
        f"  - Global schedule count: {catalog_counts.get('global_schedule_count', 0)}",
        "",
        "Comparison",
        f"  - Exhaustive latency gain over current online rule: {latency_gap_seconds:.9f} s",
        f"  - Exhaustive latency gain percentage: {latency_gap_pct:.4f}%",
        f"  - Exhaustive asynchronality gain over current online rule: {async_gap_seconds:.9f} s",
        "",
        "Best exhaustive allocation combination",
    ]
    if best_global_schedule is not None:
        lines.extend(
            [
                f"  - Global schedule id: {best_global_schedule.get('global_schedule_id')}",
                f"  - Completed all users: {best_global_schedule.get('all_completed')}",
                f"  - Total latency sum (s): {float(best_global_schedule.get('total_latency_sum_seconds', 0.0)):.9f}",
                f"  - Asynchronality sum (s): {float(best_global_schedule.get('asynchronality_sum_seconds', 0.0)):.9f}",
                f"  - Total remaining bits: {best_global_schedule.get('total_remaining_bits')}",
                f"  - Per-user schedule ids: {best_global_schedule.get('per_user_schedule_ids')}",
                f"  - Per-user latency (s): {[round(float(v), 9) for v in best_global_schedule.get('per_user_latency_seconds', [])]}",
                f"  - Per-user action signatures: {best_global_schedule.get('per_user_action_signatures')}",
                "",
            ]
        )
    else:
        lines.extend(["  - No exhaustive schedule was generated.", ""])

    lines.extend(
        [
        "Interpretation",
        "  - If exhaustive wins, then the current max-feasible-at-n=T then tail-reduce rule is not globally optimal on this frozen episode.",
        "  - If both match, this episode does not provide a counterexample against the current rule.",
        ]
    )
    return lines


def run_exhaustive_payload_compare(
    cfg_name: str,
    seed: int,
    *,
    catalog_detail_mode: str | None = None,
) -> dict:
    raw_cfg, cfg_path = _load_raw_config(cfg_name)
    system_params, sim_cfg, _ = load_config(cfg_name)
    _validate_experiment_inputs(system_params, sim_cfg)

    exhaustive_cfg = raw_cfg.get("simulation", {}).get("exhaustive_compare", {})
    max_blocks = min(
        int(sim_cfg.get("max_total_blocks", 256)),
        max(1, int(exhaustive_cfg.get("max_blocks", sim_cfg.get("max_total_blocks", 256)))),
    )
    allow_block_skip = bool(exhaustive_cfg.get("allow_block_skip", True))
    suppress_inner_logs = bool(exhaustive_cfg.get("suppress_inner_logs", True))
    resolved_catalog_mode = validate_catalog_detail_mode(
        catalog_detail_mode
        if catalog_detail_mode is not None
        else exhaustive_cfg.get("catalog_detail_mode", CATALOG_MODE_FULL),
    )

    run_started_at_local = current_local_timestamp()
    overall_start = perf_counter()
    frozen_system = _prepare_frozen_episode(
        system_params,
        seed=int(seed),
        max_blocks=int(max_blocks),
    )

    online_start = perf_counter()
    online_result = _run_current_online_strategy(
        frozen_system,
        sim_cfg,
        max_blocks=int(max_blocks),
        suppress_inner_logs=bool(suppress_inner_logs),
    )
    online_wall_time = perf_counter() - online_start

    exhaustive_start = perf_counter()
    exhaustive_result = _run_exhaustive_strategy(
        frozen_system,
        sim_cfg,
        max_blocks=int(max_blocks),
        allow_block_skip=bool(allow_block_skip),
        suppress_inner_logs=bool(suppress_inner_logs),
        catalog_detail_mode=resolved_catalog_mode,
    )
    exhaustive_wall_time = perf_counter() - exhaustive_start
    fs_per_user = [float(v) for v in system_params["fs"]]
    online_result = _enrich_strategy_summary_with_seconds(online_result, fs_per_user)
    exhaustive_result = _enrich_strategy_summary_with_seconds(exhaustive_result, fs_per_user)
    exhaustive_result = _enrich_uplink_exhaustive_catalogs(exhaustive_result, fs_per_user)

    result = {
        "method_name": METHOD_NAME,
        "cfg_name": str(cfg_name),
        "cfg_path": str(cfg_path),
        "seed": int(seed),
        "payload_bits": [int(v) for v in system_params["B"]],
        "T_per_user": [int(v) for v in system_params["T"]],
        "snr_db_per_user": [float(v) for v in system_params["snr_db"]],
        "fs_per_user": fs_per_user,
        "max_compared_blocks_per_user": int(max_blocks),
        "allow_block_skip": bool(allow_block_skip),
        "suppress_inner_logs": bool(suppress_inner_logs),
        "catalog_detail_mode": resolved_catalog_mode,
        "online_strategy": {
            **online_result,
            "per_user_actions": _strategy_actions_to_serializable(online_result["per_user_actions"]),
            "wall_time_seconds": float(online_wall_time),
        },
        "exhaustive_strategy": {
            **exhaustive_result,
            "per_user_actions": _strategy_actions_to_serializable(exhaustive_result["per_user_actions"]),
            "wall_time_seconds": float(exhaustive_wall_time),
        },
        "core_wall_time_seconds_total": float(perf_counter() - overall_start),
        "run_started_at_local": str(run_started_at_local),
        "run_completed_at_local": current_local_timestamp(),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the current uplink payload-completion online rule against exhaustive (B,n) search."
    )
    parser.add_argument(
        "--cfg_name",
        type=str,
        default=str(BENCHMARK_CONFIG_ROOT / "uplink_exhaustive_dispersion_heavy.yaml"),
        help="Configuration file name or path",
    )
    parser.add_argument("--seed", type=int, default=3, help="Deterministic random seed")
    parser.add_argument(
        "--catalog_mode",
        type=str,
        default=None,
        choices=[CATALOG_MODE_NONE, CATALOG_MODE_OUTCOMES_ONLY, CATALOG_MODE_FULL],
        help="Override exhaustive catalog mode: none, outcomes_only, or full",
    )
    args = parser.parse_args()

    _, sim_cfg, run_meta = load_config(args.cfg_name)
    update_mode = validate_convergence_precoder_update_mode(
        sim_cfg.get("convergence_precoder_update_mode", "direct_precoder")
    )
    result_tag = make_method_result_tag(
        join_tag_parts(
            format_method_tag(METHOD_NAME),
            format_update_mode_tag(update_mode),
        ),
        run_meta["cfg_stem"],
        seed=int(args.seed),
        cfg_hash=run_meta.get("cfg_hash"),
    )
    result_dirs = build_uplink_convergence_result_dirs(METHOD_LABEL, result_tag)
    result = run_exhaustive_payload_compare(
        cfg_name=args.cfg_name,
        seed=int(args.seed),
        catalog_detail_mode=args.catalog_mode,
    )

    summary_lines = _build_summary_lines(result)
    save_json(result, str(Path(result_dirs["data"]) / "exhaustive_compare.json"))
    save_text(summary_lines, str(Path(result_dirs["data"]) / "exhaustive_compare_summary.txt"))
    detail_dir = Path(result_dirs["data"]) / "small_exhaustive_details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    fs_per_user = [float(v) for v in result["fs_per_user"]]
    payload_bits = [int(v) for v in result["payload_bits"]]
    online_rows = _build_strategy_action_rows(
        strategy_name="online",
        per_user_actions=result["online_strategy"]["per_user_actions"],
        total_payload_bits=payload_bits,
        fs_per_user=fs_per_user,
    )
    exhaustive_rows = _build_strategy_action_rows(
        strategy_name="exhaustive",
        per_user_actions=result["exhaustive_strategy"]["per_user_actions"],
        total_payload_bits=payload_bits,
        fs_per_user=fs_per_user,
    )
    strategy_overview_lines = [
        "Uplink online vs exhaustive strategy comparison",
        f"Config: {result['cfg_path']}",
        f"Seed: {result['seed']}",
        "",
        *_build_strategy_overview_lines("Online strategy", result["online_strategy"], online_rows),
        "",
        *_build_strategy_overview_lines("Exhaustive strategy", result["exhaustive_strategy"], exhaustive_rows),
    ]
    strategy_overview = {
        "online": result["online_strategy"],
        "exhaustive": result["exhaustive_strategy"],
        "online_action_rows": online_rows,
        "exhaustive_action_rows": exhaustive_rows,
    }
    exhaustive_outcome_rows = _build_exhaustive_outcome_rows(result["exhaustive_strategy"].get("global_schedule_catalog", []))
    save_text(strategy_overview_lines, str(detail_dir / "strategy_comparison_overview.txt"))
    save_json(strategy_overview, str(detail_dir / "strategy_comparison_overview.json"))
    save_json({"rows": online_rows + exhaustive_rows}, str(detail_dir / "strategy_actions.json"))
    _write_csv_rows(online_rows + exhaustive_rows, str(detail_dir / "strategy_actions.csv"))
    save_json({"rows": exhaustive_outcome_rows}, str(detail_dir / "exhaustive_schedule_outcomes.json"))
    _write_csv_rows(exhaustive_outcome_rows, str(detail_dir / "exhaustive_schedule_outcomes.csv"))
    save_json({"rows": exhaustive_outcome_rows}, str(detail_dir / "exhaustive_combinations_descending_latency.json"))
    _write_csv_rows(exhaustive_outcome_rows, str(detail_dir / "exhaustive_combinations_descending_latency.csv"))
    exhaustive_outcome_rows_best_first = _sort_exhaustive_outcome_rows_best_first(exhaustive_outcome_rows)
    save_json(
        {"rows": exhaustive_outcome_rows_best_first},
        str(detail_dir / "exhaustive_combinations_ascending_latency.json"),
    )
    _write_csv_rows(
        exhaustive_outcome_rows_best_first,
        str(detail_dir / "exhaustive_combinations_ascending_latency.csv"),
    )
    exhaustive = result["exhaustive_strategy"]
    save_json(
        {"rows": exhaustive.get("exact_pair_action_catalog", [])},
        str(detail_dir / "exact_pair_action_catalog.json"),
    )
    _write_csv_rows(
        exhaustive.get("exact_pair_action_catalog", []),
        str(detail_dir / "exact_pair_action_catalog.csv"),
    )
    for user_idx, catalog in enumerate(exhaustive.get("per_user_schedule_catalogs", [])):
        save_json(
            {"rows": catalog},
            str(detail_dir / f"user_{int(user_idx)}_schedule_catalog.json"),
        )
        _write_csv_rows(
            catalog,
            str(detail_dir / f"user_{int(user_idx)}_schedule_catalog.csv"),
        )
    save_json(
        {"rows": exhaustive.get("global_schedule_catalog", [])},
        str(detail_dir / "global_schedule_catalog.json"),
    )
    _write_csv_rows(
        exhaustive.get("global_schedule_catalog", []),
        str(detail_dir / "global_schedule_catalog.csv"),
    )
    save_json(
        {"best_global_schedule": exhaustive.get("best_global_schedule")},
        str(detail_dir / "best_global_schedule.json"),
    )
    best_lines = [
        "Best uplink exhaustive schedule",
        f"Config: {result['cfg_path']}",
        f"Seed: {result['seed']}",
        f"Best schedule: {exhaustive.get('best_global_schedule')}",
    ]
    save_text(best_lines, str(detail_dir / "best_global_schedule.txt"))
    _plot_latency_comparison(result, result_dirs["optimization_history"])
    _plot_remaining_bits_trajectories(result, result_dirs["optimization_history"])

    write_result_manifest(
        result_dirs["experiment_root"],
        setup={
            "link": "Uplink",
            "method": METHOD_LABEL,
            "scenario": PAYLOAD_MODE,
            "config_path": result["cfg_path"],
            "config_hash": run_meta.get("cfg_hash"),
            "seed": int(args.seed),
            "catalog_detail_mode": args.catalog_mode,
        },
    )

    print("\n".join(summary_lines))
    print(f"\nSaved exhaustive validation results to: {result_dirs['experiment_root']}")
    print(f"Saved exhaustive validation results to: {result_dirs['experiment_root']}")


if __name__ == "__main__":
    main()
