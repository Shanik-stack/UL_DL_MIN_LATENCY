"""Uplink Monte Carlo channel episodes and rollout-query generation."""

import copy
from typing import Any, Sequence

import numpy as np
import torch

from latency_optimization.core.blocklength import build_monte_carlo_n_search_config, run_n_frontier_search
from latency_optimization.core.scenarios import (
    PAYLOAD_MODE,
    STREAMING_MODE,
    build_monte_carlo_sample_scenarios_for_seeds,
)
from latency_optimization.experiments.channels import (
    build_training_snr_schedule,
    with_monte_carlo_sample_snr_by_user,
)
from latency_optimization.results.console import format_log_line
from latency_optimization.runtime import DEVICE

from ..config import (
    RATE_BEAM_REWARD_MODE,
    UNWEIGHTED_SUM_RATE_OBJECTIVE,
    get_config,
)
from ..precoder_models import infer_precoder_numpy_with_blocklength_and_sigma
from ..simulation import ensure_blocks_up_to
from ..system import UplinkSystem
from ..uplink_rate_model import build_uplink_rate_covariance

from .network_operations import (
    ROLLOUT_QUERY_OBJECTIVE_TRAINING_STYLE,
    _build_precoder_net_snapshot_for_active_mask,
    _compute_r_fbl_np,
)


def build_training_dataset(
    cfg_name: str,
    train_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    system_params, sim_cfg = get_config(cfg_name)
    snr_db_by_user_by_seed = build_training_snr_schedule(
        train_seeds,
        sim_cfg["monte_carlo_training_snr_db_ranges"],
        num_users=int(system_params["K"]),
    )
    episodes: list[dict[str, Any]] = []

    for seed in train_seeds:
        print(
            format_log_line(
                "[UL Monte Carlo Dataset]",
                phase="collect",
                seed=int(seed),
                base_dataset=(
                    "payload_channel_episode"
                    if str(sim_cfg["experiment_scenario_mode"]) == PAYLOAD_MODE
                    else "streaming_block"
                ),
                snr_db_by_user=snr_db_by_user_by_seed[int(seed)],
            )
        )
        scenario = build_monte_carlo_sample_scenarios_for_seeds(
            system_params,
            sim_cfg,
            [int(seed)],
        )[0]
        episodes.append(
            {
                "seed": int(seed),
                "snr_db_by_user": list(snr_db_by_user_by_seed[int(seed)]),
                "num_users": int(system_params["K"]),
                "scenario_mode": str(scenario.get("mode", PAYLOAD_MODE)),
                "scenario": scenario,
            }
        )

    return episodes


def summarize_training_dataset(training_episodes: Sequence[dict[str, Any]]) -> dict:
    if len(training_episodes) == 0:
        return {
            "total_training_samples": 0,
            "training_sample_kind": "unknown",
            "training_sample_unit": "unknown",
            "training_samples_by_seed": {},
            "training_samples_per_user": [],
            "num_users": 0,
            "base_dataset_kind": "unknown",
            "scenario_modes": [],
            "training_samples_by_user_snr_db": [],
            "training_snr_db_ranges": [],
        }
    num_users = int(training_episodes[0].get("num_users", 0))
    episodes_by_user_snr_db: list[dict[str, int]] = [{} for _ in range(num_users)]
    for episode in training_episodes:
        for user, snr_db in enumerate(episode["snr_db_by_user"]):
            snr_key = f"{float(snr_db):.6g}"
            user_counts = episodes_by_user_snr_db[int(user)]
            user_counts[snr_key] = user_counts.get(snr_key, 0) + 1
    scenario_modes = sorted({str(episode.get("scenario_mode", PAYLOAD_MODE)) for episode in training_episodes})
    sample_kind = (
        "payload_channel_episode"
        if scenario_modes == [PAYLOAD_MODE]
        else "streaming_block"
        if scenario_modes == [STREAMING_MODE]
        else "mixed"
    )
    return {
        "total_training_samples": int(len(training_episodes)),
        "training_sample_kind": sample_kind,
        "training_sample_unit": "channel_episode" if sample_kind == "payload_channel_episode" else "block",
        "training_samples_by_seed": {str(int(episode["seed"])): 1 for episode in training_episodes},
        "training_samples_per_user": [int(len(training_episodes)) for _ in range(num_users)],
        "num_users": int(num_users),
        "base_dataset_kind": sample_kind,
        "scenario_modes": scenario_modes,
        "training_samples_by_user_snr_db": episodes_by_user_snr_db,
        "training_snr_db_ranges": [
            [
                float(min(float(episode["snr_db_by_user"][user]) for episode in training_episodes)),
                float(max(float(episode["snr_db_by_user"][user]) for episode in training_episodes)),
            ]
            for user in range(num_users)
        ],
    }


def _summarize_selected_n_kl(n_star: Sequence[Sequence[int]]) -> dict[str, object]:
    global_counts: dict[int, int] = {}
    per_user = []
    for user_idx, user_n in enumerate(n_star):
        user_counts: dict[int, int] = {}
        for n_kl in user_n:
            n_val = int(n_kl)
            user_counts[n_val] = user_counts.get(n_val, 0) + 1
            global_counts[n_val] = global_counts.get(n_val, 0) + 1
        per_user.append(
            {
                "user": int(user_idx),
                "selected_examples_by_n_kl": {str(int(k)): int(v) for k, v in sorted(user_counts.items())},
            }
        )
    return {
        "global_selected_examples_by_n_kl": {str(int(k)): int(v) for k, v in sorted(global_counts.items())},
        "per_user": per_user,
    }


def _aggregate_epoch_means(per_user_histories: Sequence[Sequence[float]]) -> list[float]:
    max_len = max((len(history) for history in per_user_histories), default=0)
    aggregated: list[float] = []
    for epoch_idx in range(max_len):
        values = [float(history[epoch_idx]) for history in per_user_histories if epoch_idx < len(history)]
        aggregated.append(float(np.mean(values)) if values else 0.0)
    return aggregated


def _serialize_count_dict(counts: dict[int, int]) -> dict[str, int]:
    return {str(int(k)): int(v) for k, v in sorted(counts.items())}


def _resolve_rollout_anchor_bits(rate: float, n_kl: int) -> int:
    achievable_bits = int(np.floor(max(float(rate), 0.0) * float(max(int(n_kl), 1))))
    return max(1, achievable_bits)


def _evaluate_uplink_rollout_query_numpy(
    model: torch.nn.Module,
    episode: dict[str, Any],
    n_kl: int,
) -> dict[str, Any]:
    H = np.asarray(episode["H"], dtype=np.complex64)
    noise_cov = episode.get("noise_plus_interference_cov")
    if noise_cov is not None:
        noise_cov = np.asarray(noise_cov, dtype=np.complex128)
    F_pred = infer_precoder_numpy_with_blocklength_and_sigma(
        model,
        H,
        int(n_kl),
        float(episode["sigma2"]),
        float(episode["epsilon"]),
        Nt=int(H.shape[1]),
        dk=int(episode.get("dk", H.shape[1] if H.ndim > 1 else 1)),
        P=float(episode["P"]),
        device=DEVICE,
    )
    power = float(np.linalg.norm(F_pred, ord="fro") ** 2)
    rate = float(
        _compute_r_fbl_np(
            H,
            F_pred,
            float(episode["sigma2"]),
            float(episode["epsilon"]),
            int(n_kl),
            noise_cov,
        )
    )
    power_margin = float(episode["P"]) - float(power)
    return {
        "rate": rate,
        "power": power,
        "power_margin": power_margin,
    }


def _replace_snapshot_block(
    snapshot: Sequence[Sequence[np.ndarray]],
    user: int,
    block: int,
    precoder: np.ndarray,
) -> list[list[np.ndarray]]:
    replaced = [list(user_blocks) for user_blocks in snapshot]
    replaced[int(user)][int(block)] = precoder
    return replaced


def _count_uplink_forward_call(
    evaluation_cost_counters: dict[str, Any] | None,
    user: int,
) -> None:
    if evaluation_cost_counters is None:
        return
    evaluation_cost_counters["total_forward_calls"] = int(
        evaluation_cost_counters.get("total_forward_calls", 0)
    ) + 1
    per_user = evaluation_cost_counters.get("per_user_forward_calls")
    if isinstance(per_user, list) and 0 <= int(user) < len(per_user):
        per_user[int(user)] = int(per_user[int(user)]) + 1


def _ensure_precoder_net_snapshot_block(
    uplinksystem: UplinkSystem,
    user_models: Sequence[torch.nn.Module],
    snapshot_cache: list[list[np.ndarray]],
    block_idx: int,
    *,
    evaluation_cost_counters: dict[str, Any] | None = None,
) -> list[list[np.ndarray]]:
    ensure_blocks_up_to(uplinksystem, int(block_idx))
    for k in range(int(uplinksystem.K)):
        while len(snapshot_cache[int(k)]) <= int(block_idx):
            l = len(snapshot_cache[int(k)])
            H_kl = np.asarray(uplinksystem.H[int(k)][int(l)], dtype=np.complex64)
            _count_uplink_forward_call(evaluation_cost_counters, int(k))
            snapshot_cache[int(k)].append(
                infer_precoder_numpy_with_blocklength_and_sigma(
                    user_models[int(k)],
                    H_kl,
                    n_kl=int(uplinksystem.T[int(k)]),
                    sigma2=float(uplinksystem.sigma2[int(k)]),
                    epsilon=float(uplinksystem.epsilon[int(k)]),
                    Nt=int(uplinksystem.NT[int(k)]),
                    dk=int(uplinksystem.dk[int(k)]),
                    P=float(uplinksystem.P[int(k)]),
                    device=DEVICE,
                )
            )
    return snapshot_cache


def _build_uplink_rollout_query(
    *,
    seed: int,
    user: int,
    block: int,
    H: np.ndarray,
    T_ref: int,
    P: float,
    dk: int,
    sigma2: float,
    epsilon: float,
    noise_cov: np.ndarray | None,
    n_kl: int,
    required_bits: int,
    metrics: dict[str, Any],
    scenario_mode: str,
    rollout_phase: str,
    rollout_stage: str,
    frontier_query: bool,
) -> dict[str, Any]:
    required_rate = (
        float(required_bits) / float(max(int(n_kl), 1))
        if int(required_bits) > 0
        else 0.0
    )
    rate_margin = float(metrics["rate"] - required_rate)
    return {
        "seed": int(seed),
        "user": int(user),
        "block": int(block),
        "H": np.asarray(H, dtype=np.complex64),
        "T_ref": int(T_ref),
        "P": float(P),
        "dk": int(dk),
        "sigma2": float(sigma2),
        "epsilon": float(epsilon),
        "noise_plus_interference_cov": (
            None if noise_cov is None else np.asarray(noise_cov, dtype=np.complex128)
        ),
        "scenario_mode": str(scenario_mode),
        "n_kl": int(n_kl),
        "rollout_anchor_bits": int(required_bits),
        "required_rate": float(required_rate),
        "rate": float(metrics["rate"]),
        "power": float(metrics["power"]),
        "rate_margin": float(rate_margin),
        "power_margin": float(metrics["power_margin"]),
        "feasible": bool(rate_margin >= 0.0 and float(metrics["power_margin"]) >= 0.0),
        "frontier_query": bool(frontier_query),
        "rollout_phase": str(rollout_phase),
        "rollout_stage": str(rollout_stage),
    }


def _collect_uplink_payload_rollout_queries_for_episode(
    system_params: dict[str, Any],
    sim_cfg: dict[str, Any],
    training_episode: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
) -> list[list[dict[str, Any]]]:
    seed = int(training_episode["seed"])
    scenario = dict(training_episode["scenario"])
    K = int(system_params["K"])
    episode_system_params = with_monte_carlo_sample_snr_by_user(
        system_params,
        training_episode["snr_db_by_user"],
    )
    system = UplinkSystem(episode_system_params, seed=int(seed))
    remaining = np.asarray(scenario["payload_bits_per_user"], dtype=int).copy()
    max_blocks = int(sim_cfg.get("max_total_blocks", 256))
    queries_by_user: list[list[dict[str, Any]]] = [[] for _ in range(K)]
    block = 0

    while np.any(remaining > 0):
        if block >= max_blocks:
            raise RuntimeError(
                f"Uplink Monte Carlo training rollout hit max_total_blocks={max_blocks} for seed={seed} with remaining bits {remaining.tolist()}."
            )
        ensure_blocks_up_to(system, int(block))
        active_mask = [1 if int(remaining[int(k)]) > 0 else 0 for k in range(K)]
        snapshot_full = _build_precoder_net_snapshot_for_active_mask(system, user_models, int(block), active_mask)

        for k in range(K):
            if int(active_mask[int(k)]) <= 0:
                continue
            remaining_before_block = int(remaining[int(k)])
            H_kl = np.asarray(system.H[int(k)][int(block)], dtype=np.complex64)
            T_ref = int(system.T[int(k)])
            P = float(system.P[int(k)])
            sigma2 = float(system.sigma2[int(k)])
            epsilon = float(system.epsilon[int(k)])
            dk = int(system.dk[int(k)])
            F_T = infer_precoder_numpy_with_blocklength_and_sigma(
                user_models[int(k)],
                H_kl,
                n_kl=T_ref,
                sigma2=sigma2,
                epsilon=epsilon,
                Nt=int(system.NT[int(k)]),
                dk=dk,
                P=P,
                device=DEVICE,
            )
            snapshot_candidate = _replace_snapshot_block(snapshot_full, int(k), int(block), F_T)
            cov_T = build_uplink_rate_covariance(
                system,
                sim_cfg,
                int(k),
                int(block),
                F_override=snapshot_candidate,
            )
            base_episode = {
                "H": H_kl,
                "sigma2": sigma2,
                "epsilon": epsilon,
                "P": P,
                "dk": dk,
                "noise_plus_interference_cov": cov_T,
            }
            full_metrics = _evaluate_uplink_rollout_query_numpy(user_models[int(k)], base_episode, T_ref)
            queries_by_user[int(k)].append(
                _build_uplink_rollout_query(
                    seed=seed,
                    user=int(k),
                    block=int(block),
                    H=H_kl,
                    T_ref=T_ref,
                    P=P,
                    dk=dk,
                    sigma2=sigma2,
                    epsilon=epsilon,
                    noise_cov=cov_T,
                    n_kl=T_ref,
                    required_bits=0,
                    metrics=full_metrics,
                    scenario_mode=PAYLOAD_MODE,
                    rollout_phase="full_block",
                    rollout_stage="full_block",
                    frontier_query=False,
                )
            )

            supported_bits = max(int(np.floor(float(full_metrics["rate"]) * float(max(T_ref, 1)))), 0)
            committed_bits = min(int(remaining_before_block), int(supported_bits))
            if int(committed_bits) <= 0:
                continue
            if int(committed_bits) < int(remaining_before_block):
                remaining[int(k)] = max(int(remaining[int(k)]) - int(committed_bits), 0)
                continue

            queries_by_user[int(k)].append(
                _build_uplink_rollout_query(
                    seed=seed,
                    user=int(k),
                    block=int(block),
                    H=H_kl,
                    T_ref=T_ref,
                    P=P,
                    dk=dk,
                    sigma2=sigma2,
                    epsilon=epsilon,
                    noise_cov=cov_T,
                    n_kl=T_ref,
                    required_bits=int(committed_bits),
                    metrics=full_metrics,
                    scenario_mode=PAYLOAD_MODE,
                    rollout_phase="tail_feasible",
                    rollout_stage="full_block_commit",
                    frontier_query=False,
                )
            )

            search_cfg = build_monte_carlo_n_search_config(
                sim_cfg,
                n_min=int(sim_cfg["n_kl_min"]),
                n_max=int(T_ref),
                phase="training",
            )
            def _evaluate_payload_rollout_candidate(candidate: int, stage_name: str) -> dict[str, Any]:
                metrics = _evaluate_uplink_rollout_query_numpy(
                    user_models[int(k)],
                    base_episode,
                    int(candidate),
                )
                query = _build_uplink_rollout_query(
                    seed=seed,
                    user=int(k),
                    block=int(block),
                    H=H_kl,
                    T_ref=T_ref,
                    P=P,
                    dk=dk,
                    sigma2=sigma2,
                    epsilon=epsilon,
                    noise_cov=cov_T,
                    n_kl=int(candidate),
                    required_bits=int(committed_bits),
                    metrics=metrics,
                    scenario_mode=PAYLOAD_MODE,
                    rollout_phase="tail_feasible",
                    rollout_stage=str(stage_name),
                    frontier_query=False,
                )
                return {
                    "query": query,
                    "feasible": bool(query["feasible"]),
                }

            search_result = run_n_frontier_search(search_cfg, _evaluate_payload_rollout_candidate)
            for visited in search_result["visited"]:
                query = dict(visited["result"]["query"])
                query["feasible"] = bool(query["feasible"])
                if bool(visited["feasible"]):
                    query["rollout_phase"] = "tail_feasible"
                    query["frontier_query"] = False
                else:
                    query["rollout_phase"] = "tail_frontier"
                    query["frontier_query"] = True
                queries_by_user[int(k)].append(query)
            remaining[int(k)] = max(int(remaining[int(k)]) - int(committed_bits), 0)
        block += 1

    return queries_by_user


def _collect_uplink_streaming_rollout_queries_for_channel_episode(
    system_params: dict[str, Any],
    sim_cfg: dict[str, Any],
    training_episode: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
) -> list[list[dict[str, Any]]]:
    seed = int(training_episode["seed"])
    scenario = dict(training_episode["scenario"])
    block_targets = np.asarray(scenario["streaming_bit_targets_by_block"], dtype=int)
    num_blocks = int(scenario["number_of_blocks"])
    K = int(system_params["K"])
    episode_system_params = with_monte_carlo_sample_snr_by_user(
        system_params,
        training_episode["snr_db_by_user"],
    )
    system = UplinkSystem(episode_system_params, seed=int(seed))
    queries_by_user: list[list[dict[str, Any]]] = [[] for _ in range(K)]

    for block in range(num_blocks):
        ensure_blocks_up_to(system, int(block))
        active_mask = [1 if int(block_targets[int(k), int(block)]) > 0 else 0 for k in range(K)]
        snapshot_full = _build_precoder_net_snapshot_for_active_mask(system, user_models, int(block), active_mask)
        for k in range(K):
            target_bits = int(block_targets[int(k), int(block)])
            if target_bits <= 0:
                continue
            H_kl = np.asarray(system.H[int(k)][int(block)], dtype=np.complex64)
            T_ref = int(system.T[int(k)])
            P = float(system.P[int(k)])
            sigma2 = float(system.sigma2[int(k)])
            epsilon = float(system.epsilon[int(k)])
            dk = int(system.dk[int(k)])
            F_T = infer_precoder_numpy_with_blocklength_and_sigma(
                user_models[int(k)],
                H_kl,
                n_kl=T_ref,
                sigma2=sigma2,
                epsilon=epsilon,
                Nt=int(system.NT[int(k)]),
                dk=dk,
                P=P,
                device=DEVICE,
            )
            snapshot_candidate = _replace_snapshot_block(snapshot_full, int(k), int(block), F_T)
            cov_T = build_uplink_rate_covariance(
                system,
                sim_cfg,
                int(k),
                int(block),
                F_override=snapshot_candidate,
            )
            base_episode = {
                "H": H_kl,
                "sigma2": sigma2,
                "epsilon": epsilon,
                "P": P,
                "dk": dk,
                "noise_plus_interference_cov": cov_T,
            }
            full_metrics = _evaluate_uplink_rollout_query_numpy(user_models[int(k)], base_episode, T_ref)
            queries_by_user[int(k)].append(
                _build_uplink_rollout_query(
                    seed=seed,
                    user=int(k),
                    block=int(block),
                    H=H_kl,
                    T_ref=T_ref,
                    P=P,
                    dk=dk,
                    sigma2=sigma2,
                    epsilon=epsilon,
                    noise_cov=cov_T,
                    n_kl=T_ref,
                    required_bits=0,
                    metrics=full_metrics,
                    scenario_mode=STREAMING_MODE,
                    rollout_phase="full_block",
                    rollout_stage="full_block",
                    frontier_query=False,
                )
            )

            supported_bits = max(int(np.floor(float(full_metrics["rate"]) * float(max(T_ref, 1)))), 0)
            if int(supported_bits) < int(target_bits):
                continue

            queries_by_user[int(k)].append(
                _build_uplink_rollout_query(
                    seed=seed,
                    user=int(k),
                    block=int(block),
                    H=H_kl,
                    T_ref=T_ref,
                    P=P,
                    dk=dk,
                    sigma2=sigma2,
                    epsilon=epsilon,
                    noise_cov=cov_T,
                    n_kl=T_ref,
                    required_bits=int(target_bits),
                    metrics=full_metrics,
                    scenario_mode=STREAMING_MODE,
                    rollout_phase="tail_feasible",
                    rollout_stage="full_block_commit",
                    frontier_query=False,
                )
            )

            n_min = int(sim_cfg["n_kl_min"])
            fine_step = max(1, int(sim_cfg["n_kl_step"]))
            coarse_step = fine_step
            last_feasible_n = int(T_ref)
            first_infeasible_query: dict[str, Any] | None = None

            candidate = int(T_ref) - int(coarse_step)
            while candidate >= int(n_min):
                metrics = _evaluate_uplink_rollout_query_numpy(user_models[int(k)], base_episode, int(candidate))
                query = _build_uplink_rollout_query(
                    seed=seed,
                    user=int(k),
                    block=int(block),
                    H=H_kl,
                    T_ref=T_ref,
                    P=P,
                    dk=dk,
                    sigma2=sigma2,
                    epsilon=epsilon,
                    noise_cov=cov_T,
                    n_kl=int(candidate),
                    required_bits=int(target_bits),
                    metrics=metrics,
                    scenario_mode=STREAMING_MODE,
                    rollout_phase="tail_feasible",
                    rollout_stage="coarse",
                    frontier_query=False,
                )
                if bool(query["feasible"]):
                    queries_by_user[int(k)].append(query)
                    last_feasible_n = int(candidate)
                    candidate -= int(coarse_step)
                    continue
                query["rollout_phase"] = "tail_frontier"
                query["frontier_query"] = True
                first_infeasible_query = query
                break

            if (
                first_infeasible_query is not None
                and int(fine_step) < int(coarse_step)
                and int(last_feasible_n) - int(fine_step) > int(first_infeasible_query["n_kl"])
            ):
                candidate = int(last_feasible_n) - int(fine_step)
                first_infeasible_n = int(first_infeasible_query["n_kl"])
                while candidate > int(first_infeasible_n):
                    metrics = _evaluate_uplink_rollout_query_numpy(user_models[int(k)], base_episode, int(candidate))
                    query = _build_uplink_rollout_query(
                        seed=seed,
                        user=int(k),
                        block=int(block),
                        H=H_kl,
                        T_ref=T_ref,
                        P=P,
                        dk=dk,
                        sigma2=sigma2,
                        epsilon=epsilon,
                        noise_cov=cov_T,
                        n_kl=int(candidate),
                        required_bits=int(target_bits),
                        metrics=metrics,
                        scenario_mode=STREAMING_MODE,
                        rollout_phase="tail_feasible",
                        rollout_stage="fine",
                        frontier_query=False,
                    )
                    if bool(query["feasible"]):
                        queries_by_user[int(k)].append(query)
                        last_feasible_n = int(candidate)
                        candidate -= int(fine_step)
                        continue
                    query["rollout_phase"] = "tail_frontier"
                    query["frontier_query"] = True
                    first_infeasible_query = query
                    break

            if first_infeasible_query is not None:
                queries_by_user[int(k)].append(first_infeasible_query)

    return queries_by_user


def _generate_rollout_queries_for_training_episodes(
    system_params: dict[str, Any],
    sim_cfg: dict[str, Any],
    training_episodes: Sequence[dict[str, Any]],
    user_models: Sequence[torch.nn.Module],
) -> list[list[dict[str, Any]]]:
    K = int(system_params["K"])
    queries_by_user: list[list[dict[str, Any]]] = [[] for _ in range(K)]
    for training_episode in training_episodes:
        scenario_mode = str(training_episode.get("scenario", {}).get("mode", PAYLOAD_MODE))
        collect_episode_queries = (
            _collect_uplink_streaming_rollout_queries_for_channel_episode
            if scenario_mode == STREAMING_MODE
            else _collect_uplink_payload_rollout_queries_for_episode
        )
        episode_queries = collect_episode_queries(
            system_params,
            sim_cfg,
            training_episode,
            user_models,
        )
        for k in range(K):
            queries_by_user[int(k)].extend(episode_queries[int(k)])
    return queries_by_user


def _summarize_rollout_queries_by_user(queries_by_user: Sequence[Sequence[dict[str, Any]]]) -> dict[str, Any]:
    global_queries_by_n_kl: dict[int, int] = {}
    global_frontier_queries_by_n_kl: dict[int, int] = {}
    global_queries_by_feasibility = {"feasible": 0, "infeasible": 0, "frontier": 0}
    per_user = []

    for user_idx, queries in enumerate(queries_by_user):
        user_queries_by_n_kl: dict[int, int] = {}
        user_frontier_queries_by_n_kl: dict[int, int] = {}
        feasible_count = 0
        infeasible_count = 0
        frontier_count = 0
        for query in queries:
            n_val = int(query["n_kl"])
            user_queries_by_n_kl[n_val] = user_queries_by_n_kl.get(n_val, 0) + 1
            global_queries_by_n_kl[n_val] = global_queries_by_n_kl.get(n_val, 0) + 1
            if bool(query.get("frontier_query", False)):
                user_frontier_queries_by_n_kl[n_val] = user_frontier_queries_by_n_kl.get(n_val, 0) + 1
                global_frontier_queries_by_n_kl[n_val] = global_frontier_queries_by_n_kl.get(n_val, 0) + 1
                frontier_count += 1
                global_queries_by_feasibility["frontier"] += 1
            if bool(query.get("feasible", False)):
                feasible_count += 1
                global_queries_by_feasibility["feasible"] += 1
            else:
                infeasible_count += 1
                global_queries_by_feasibility["infeasible"] += 1

        per_user.append(
            {
                "user": int(user_idx),
                "total_rollout_queries": int(len(queries)),
                "rollout_queries_by_n_kl": _serialize_count_dict(user_queries_by_n_kl),
                "frontier_rollout_queries_by_n_kl": _serialize_count_dict(user_frontier_queries_by_n_kl),
                "feasible_rollout_queries": int(feasible_count),
                "infeasible_rollout_queries": int(infeasible_count),
                "frontier_rollout_queries": int(frontier_count),
            }
        )

    return {
        "total_rollout_queries": int(sum(len(queries) for queries in queries_by_user)),
        "global_rollout_queries_by_n_kl": _serialize_count_dict(global_queries_by_n_kl),
        "global_frontier_rollout_queries_by_n_kl": _serialize_count_dict(global_frontier_queries_by_n_kl),
        "global_rollout_queries_by_feasibility": {
            key: int(value) for key, value in global_queries_by_feasibility.items()
        },
        "per_user": per_user,
    }


def _build_post_training_summary(
    train_eval_system: UplinkSystem,
    train_eval_post: dict,
    training_history: dict,
    *,
    train_eval_seed: int,
    train_eval_snr_db_by_user: Sequence[float],
    epochs: int,
    dataset_summary: dict[str, Any],
    initial_baseline: dict | None = None,
) -> dict:
    per_user_objective_loss = training_history.get("per_user_objective_loss", [])
    per_user_rate = training_history.get("per_user_rate", [])
    per_user_rate_violation = training_history.get("avg_rate_violation", [])
    per_user_power_violation = training_history.get("avg_power_violation", [])
    avg_objective_loss = training_history.get("avg_objective_loss", [])
    avg_user_rate = training_history.get("avg_user_rate", [])
    avg_rate_violation = training_history.get("avg_rate_violation_over_users", [])
    avg_power_violation = training_history.get("avg_power_violation_over_users", [])
    initial_latency = (
        [float(v) for v in initial_baseline.get("initial_latency", [])]
        if isinstance(initial_baseline, dict)
        else [float(v) for v in train_eval_system.initial_latency]
    )
    initial_n = (
        [int(v) for v in initial_baseline.get("initial_n", [])]
        if isinstance(initial_baseline, dict)
        else [int(v) for v in train_eval_system.n]
    )
    initial_n_kl = (
        [[int(x) for x in user_blocks] for user_blocks in initial_baseline.get("initial_n_kl", [])]
        if isinstance(initial_baseline, dict)
        else [list(map(int, user_blocks)) for user_blocks in train_eval_system.n_kl]
    )
    initial_B_kl = (
        [[int(x) for x in user_bits] for user_bits in initial_baseline.get("initial_B_kl", [])]
        if isinstance(initial_baseline, dict)
        else [[int(train_eval_system.B[k])] for k in range(int(train_eval_system.K))]
    )
    final_latency = [float(v) for v in train_eval_system.latency]
    initial_total_latency = float(sum(initial_latency))
    final_total_latency = float(sum(final_latency))
    total_latency_reduction_percent = (
        float(((initial_total_latency - final_total_latency) / initial_total_latency) * 100.0)
        if initial_total_latency > 0.0
        else 0.0
    )
    initial_selected_n_summary = _summarize_selected_n_kl(initial_n_kl)
    selected_n_summary = _summarize_selected_n_kl(train_eval_post.get("n_star", []))

    return {
        "train_eval_seed": int(train_eval_seed),
        "train_eval_snr_db_by_user": [float(value) for value in train_eval_snr_db_by_user],
        "epochs_requested": int(epochs),
        "configured_max_epochs": int(training_history.get("configured_max_epochs", epochs)),
        "per_user_epochs_completed": [
            int(v) for v in training_history.get("per_user_epochs_completed", [len(history) for history in per_user_objective_loss])
        ],
        "per_user_training_solve_status": [
            str(v) for v in training_history.get("per_user_training_solve_status", ["unknown" for _ in per_user_objective_loss])
        ],
        "per_user_restored_solution_source": [
            str(v) for v in training_history.get("per_user_restored_solution_source", ["unknown" for _ in per_user_objective_loss])
        ],
        "base_dataset_kind": dataset_summary.get("base_dataset_kind", "unknown"),
        "total_training_samples": int(dataset_summary.get("total_training_samples", 0)),
        "training_sample_kind": dataset_summary.get("training_sample_kind", "unknown"),
        "training_sample_unit": dataset_summary.get("training_sample_unit", "unknown"),
        "rollout_anchor_bits_mode": "derived_online_from_current_episode_served_bits",
        "monte_carlo_training_style": str(
            training_history.get("monte_carlo_training_style", ROLLOUT_QUERY_OBJECTIVE_TRAINING_STYLE)
        ),
        "uplink_objective_mode": str(
            training_history.get("uplink_objective_mode", UNWEIGHTED_SUM_RATE_OBJECTIVE)
        ),
        "beam_reward_mode": str(
            training_history.get("beam_reward_mode", RATE_BEAM_REWARD_MODE)
        ),
        "cumulative_rollout_queries_by_n_kl": training_history.get("cumulative_rollout_queries_by_n_kl", {}),
        "cumulative_frontier_rollout_queries_by_n_kl": training_history.get(
            "cumulative_frontier_rollout_queries_by_n_kl",
            {},
        ),
        "final_epoch_rollout_query_summary": training_history.get("final_epoch_rollout_query_summary", {}),
        "per_user_num_epochs": [int(len(history)) for history in per_user_objective_loss],
        "per_user_final_objective_loss": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in per_user_objective_loss
        ],
        "per_user_best_objective_loss": [
            float(min(history)) if len(history) > 0 else 0.0 for history in per_user_objective_loss
        ],
        "per_user_final_rate": [float(history[-1]) if len(history) > 0 else 0.0 for history in per_user_rate],
        "per_user_last_epoch_avg_rate_over_rollout_queries": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in per_user_rate
        ],
        "per_user_last_epoch_avg_objective_loss_over_rollout_queries": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in per_user_objective_loss
        ],
        "per_user_final_rate_violation": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in per_user_rate_violation
        ],
        "per_user_final_power_violation": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in per_user_power_violation
        ],
        "per_user_final_kkt_primal_residual": [
            float(history[-1]) if len(history) > 0 else 0.0
            for history in training_history.get("per_user_kkt_primal_residual", [])
        ],
        "per_user_final_kkt_complementarity_residual": [
            float(history[-1]) if len(history) > 0 else 0.0
            for history in training_history.get("per_user_kkt_complementarity_residual", [])
        ],
        "per_user_final_kkt_stationarity_residual": [
            float(history[-1]) if len(history) > 0 else 0.0
            for history in training_history.get("per_user_kkt_stationarity_residual", [])
        ],
        "final_avg_objective_loss": float(avg_objective_loss[-1]) if len(avg_objective_loss) > 0 else 0.0,
        "best_avg_objective_loss": float(min(avg_objective_loss)) if len(avg_objective_loss) > 0 else 0.0,
        "last_epoch_mean_user_rollout_objective_loss": float(avg_objective_loss[-1]) if len(avg_objective_loss) > 0 else 0.0,
        "best_epoch_mean_user_rollout_objective_loss": float(min(avg_objective_loss)) if len(avg_objective_loss) > 0 else 0.0,
        "final_avg_user_rate": float(avg_user_rate[-1]) if len(avg_user_rate) > 0 else 0.0,
        "best_avg_user_rate": float(max(avg_user_rate)) if len(avg_user_rate) > 0 else 0.0,
        "last_epoch_mean_user_rollout_rate": float(avg_user_rate[-1]) if len(avg_user_rate) > 0 else 0.0,
        "best_epoch_mean_user_rollout_rate": float(max(avg_user_rate)) if len(avg_user_rate) > 0 else 0.0,
        "final_avg_rate_violation": float(avg_rate_violation[-1]) if len(avg_rate_violation) > 0 else 0.0,
        "best_avg_rate_violation": float(min(avg_rate_violation)) if len(avg_rate_violation) > 0 else 0.0,
        "final_avg_power_violation": float(avg_power_violation[-1]) if len(avg_power_violation) > 0 else 0.0,
        "best_avg_power_violation": float(min(avg_power_violation)) if len(avg_power_violation) > 0 else 0.0,
        "per_user_final_loss": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in per_user_objective_loss
        ],
        "per_user_best_loss": [
            float(min(history)) if len(history) > 0 else 0.0 for history in per_user_objective_loss
        ],
        "last_epoch_total_rollout_queries": int(
            training_history.get("final_epoch_rollout_query_summary", {}).get("total_rollout_queries", 0)
        ),
        "last_epoch_feasible_rollout_queries": int(
            training_history.get("final_epoch_rollout_query_summary", {})
            .get("global_rollout_queries_by_feasibility", {})
            .get("feasible", 0)
        ),
        "last_epoch_infeasible_rollout_queries": int(
            training_history.get("final_epoch_rollout_query_summary", {})
            .get("global_rollout_queries_by_feasibility", {})
            .get("infeasible", 0)
        ),
        "train_eval_initial_latency": initial_latency,
        "train_eval_final_latency": final_latency,
        "train_eval_initial_total_latency": float(initial_total_latency),
        "train_eval_final_total_latency": float(final_total_latency),
        "train_eval_total_latency_reduction_percent": float(total_latency_reduction_percent),
        "train_eval_initial_blocks_per_user": [int(len(v)) for v in initial_n_kl],
        "train_eval_initial_total_n_per_user": initial_n,
        "train_eval_initial_served_bits_per_user": [int(sum(bits)) for bits in initial_B_kl],
        "train_eval_initial_selected_n_kl_summary": initial_selected_n_summary,
        "train_eval_blocks_per_user": [int(v) for v in train_eval_post.get("L_out", [])],
        "train_eval_total_n_per_user": [int(v) for v in train_eval_system.n],
        "train_eval_served_bits_per_user": [
            int(sum(block_bits))
            for block_bits in train_eval_post.get("B_kl_star", [[] for _ in range(int(train_eval_system.K))])
        ],
        "train_eval_initial_skipped_blocks_per_user": [
            int(v) for v in (initial_baseline.get("skipped_blocks_per_user", []) if isinstance(initial_baseline, dict) else [])
        ],
        "train_eval_skipped_blocks_per_user": [
            int(v) for v in train_eval_post.get("skipped_blocks_per_user", [0 for _ in range(int(train_eval_system.K))])
        ],
        "train_eval_selected_n_kl_summary": selected_n_summary,
    }




__all__ = ["build_training_dataset", "summarize_training_dataset"]
