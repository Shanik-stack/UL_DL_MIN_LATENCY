"""Downlink Monte Carlo channel episodes and rollout-query generation."""

from typing import Any, Sequence

import numpy as np
import torch

from latency_optimization.optimization.blocklength_search import build_monte_carlo_n_search_config
from latency_optimization.experiments.scenarios import (
    PAYLOAD_MODE,
    STREAMING_MODE,
    build_monte_carlo_sample_scenarios_for_seeds,
)
from latency_optimization.experiments.channels import (
    build_training_snr_schedule,
    with_monte_carlo_sample_snr_by_user,
)
from latency_optimization.experiments.determinism import configure_determinism
from latency_optimization.precoders.serialization import precoder_to_numpy
from latency_optimization.results.console import format_log_line

from ...simulation.block_state import ensure_precoder_block, make_zero_precoder
from ...simulation.system import DownlinkSystem

from .precoder_network import (
    _best_joint_n_target_transition,
    _build_block_joint_scenario,
    _build_rollout_query_from_downlink_state,
    _masked_precoder_snapshot,
    _scenario_forward_pass,
    _scenario_input_noise_covariances,
    _scenario_metrics_from_forward,
    _scenario_metrics_with_models,
    _supported_bits_from_forward,
)


def _build_training_block_scenario(
    system: DownlinkSystem,
    working_F: list[list[np.ndarray]],
    block: int,
    active_mask: Sequence[int | float],
    *,
    scenario_mode: str,
) -> dict[str, Any]:
    """Create the immutable network input for one visited downlink block.

    What: ensure active users have block state, mask inactive beams, and store channels,
    activity, scenario type, and input interference covariances. Why: an epoch needs a
    reproducible description of the exact joint state from which its beams were predicted.
    """
    for k, flag in enumerate(active_mask):
        if float(flag) > 0.5:
            ensure_precoder_block(system, working_F, int(k), int(block))
    input_snapshot = _masked_precoder_snapshot(system, working_F, int(block), active_mask)
    scenario = _build_block_joint_scenario(
        system,
        int(block),
        active_mask,
        scenario_mode=str(scenario_mode),
    )
    scenario["input_noise_covariances"] = _scenario_input_noise_covariances(
        system,
        input_snapshot,
        int(block),
        active_mask,
    )
    return scenario


def _apply_forward_to_working_precoders(
    system: DownlinkSystem,
    working_F: list[list[np.ndarray]],
    block: int,
    forward: dict[str, Any] | None,
) -> None:
    """Advance the rollout's beam state using one completed network forward pass.

    Active users receive their predicted beams; inactive users receive zero beams.
    Later rollout blocks and n candidates therefore observe interference produced by
    the current networks rather than the initial random precoders.
    """
    if forward is None:
        return
    active_mask = list(forward.get("active_mask", []))
    n_targets = list(forward.get("n_targets", []))
    for k in range(system.K):
        ensure_precoder_block(system, working_F, int(k), int(block))
        is_active = int(k) < len(active_mask) and float(active_mask[int(k)]) > 0.5
        has_n = int(k) < len(n_targets) and int(n_targets[int(k)]) > 0
        if is_active and has_n:
            working_F[int(k)][int(block)] = np.asarray(
                precoder_to_numpy(forward["predicted_beams"][int(k)]),
                dtype=np.complex128,
            )
        else:
            working_F[int(k)][int(block)] = make_zero_precoder(system, int(k))


def _collect_downlink_tail_rollout_queries(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    system: DownlinkSystem,
    working_F: list[list[np.ndarray]],
    user_models: Sequence[torch.nn.Module],
    *,
    block: int,
    scenario_mode: str,
    committed_bits: Sequence[int],
    service_mask: Sequence[int | float],
    reducible_users: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Explore smaller blocklengths for bits supportable at the full blocklength.

    What: hold committed bits fixed, reduce active users' n targets, evaluate the current
    networks, and retain visited feasible states plus the first infeasible frontier.
    Why: these are the n-aware examples that teach one network how beam choice changes
    near the minimum feasible blocklength instead of training only at ``n=T``.
    """
    tail_scenario = _build_training_block_scenario(
        system,
        working_F,
        int(block),
        service_mask,
        scenario_mode=str(scenario_mode),
    )
    current_n_targets = [
        int(system.T[int(k)]) if float(service_mask[int(k)]) > 0.5 else 0
        for k in range(system.K)
    ]
    tail_queries: list[dict[str, Any]] = []
    committed_metrics = _scenario_metrics_with_models(
        system_params,
        tail_scenario,
        user_models,
        current_n_targets,
        anchor_bits=committed_bits,
    )
    tail_queries.append(
        _build_rollout_query_from_downlink_state(
            tail_scenario,
            current_n_targets,
            committed_metrics,
            committed_bits,
            rollout_phase="tail_feasible",
            rollout_stage="full_block_commit",
            frontier_query=False,
        )
    )

    search_cfg = build_monte_carlo_n_search_config(
        sim_params,
        n_min=int(sim_params["n_kl_min"]),
        n_max=max(int(v) for v in current_n_targets if int(v) > 0) if any(int(v) > 0 for v in current_n_targets) else 1,
        phase="training",
    )
    n_min = int(search_cfg["n_min"])
    fixed_step = int(search_cfg["fine_step"])

    if str(search_cfg["direction"]) == "ascending":
        current_n_targets = [
            (
                int(n_min)
                if float(service_mask[int(k)]) > 0.5 and int(k) in set(int(v) for v in reducible_users)
                else int(current_n_targets[int(k)])
            )
            for k in range(system.K)
        ]
        current_metrics = _scenario_metrics_with_models(
            system_params,
            tail_scenario,
            user_models,
            current_n_targets,
            anchor_bits=committed_bits,
        )
        tail_queries = [
            _build_rollout_query_from_downlink_state(
                tail_scenario,
                current_n_targets,
                current_metrics,
                committed_bits,
                rollout_phase="tail_feasible" if bool(current_metrics["feasible"]) else "tail_frontier",
                rollout_stage="fixed_step",
                frontier_query=not bool(current_metrics["feasible"]),
            )
        ]
        visited_states = {tuple(int(v) for v in current_n_targets)}
        while not bool(current_metrics["feasible"]):
            transition = _best_joint_n_target_transition(
                system_params,
                tail_scenario,
                user_models,
                current_n_targets,
                committed_bits,
                n_min=n_min,
                n_step=int(fixed_step),
                direction="ascending",
                max_n_targets=tail_scenario.get("max_n_targets", current_n_targets),
                eligible_users=[int(k) for k in reducible_users],
            )
            accepted = transition.get("accepted")
            if accepted is not None:
                current_n_targets = [int(v) for v in accepted["candidate_n_targets"]]
                current_metrics = accepted["metrics"]
                state_key = tuple(int(v) for v in current_n_targets)
                if state_key not in visited_states:
                    visited_states.add(state_key)
                    tail_queries.append(
                        _build_rollout_query_from_downlink_state(
                            tail_scenario,
                            current_n_targets,
                            current_metrics,
                            committed_bits,
                            rollout_phase="tail_feasible",
                            rollout_stage="fixed_step",
                            frontier_query=False,
                        )
                    )
                break
            rejected = transition.get("rejected")
            if rejected is None:
                break
            current_n_targets = [int(v) for v in rejected["candidate_n_targets"]]
            current_metrics = rejected["metrics"]
            state_key = tuple(int(v) for v in current_n_targets)
            if state_key in visited_states:
                break
            visited_states.add(state_key)
            tail_queries.append(
                _build_rollout_query_from_downlink_state(
                    tail_scenario,
                    current_n_targets,
                    current_metrics,
                    committed_bits,
                    rollout_phase="tail_frontier",
                    rollout_stage="fixed_step",
                    frontier_query=True,
                )
            )
    else:
        visited_states = {tuple(int(v) for v in current_n_targets)}
        while True:
            transition = _best_joint_n_target_transition(
                system_params,
                tail_scenario,
                user_models,
                current_n_targets,
                committed_bits,
                n_min=n_min,
                n_step=int(fixed_step),
                direction="descending",
                eligible_users=[int(k) for k in reducible_users],
            )
            accepted = transition.get("accepted")
            if accepted is None:
                rejected = transition.get("rejected")
                if rejected is not None:
                    rejected_key = tuple(int(v) for v in rejected["candidate_n_targets"])
                    if rejected_key not in visited_states:
                        visited_states.add(rejected_key)
                        tail_queries.append(
                            _build_rollout_query_from_downlink_state(
                                tail_scenario,
                                rejected["candidate_n_targets"],
                                rejected["metrics"],
                                committed_bits,
                                rollout_phase="tail_frontier",
                                rollout_stage="fixed_step",
                                frontier_query=True,
                            )
                        )
                break

            current_n_targets = [int(v) for v in accepted["candidate_n_targets"]]
            state_key = tuple(int(v) for v in current_n_targets)
            if state_key in visited_states:
                break
            visited_states.add(state_key)
            tail_queries.append(
                _build_rollout_query_from_downlink_state(
                    tail_scenario,
                    current_n_targets,
                    accepted["metrics"],
                    committed_bits,
                    rollout_phase="tail_feasible",
                    rollout_stage="fixed_step",
                    frontier_query=False,
                )
            )

    final_forward = _scenario_forward_pass(
        system_params,
        tail_scenario,
        user_models,
        current_n_targets,
        anchor_bits=committed_bits,
    )
    return tail_queries, final_forward


def _collect_downlink_episode_rollout_queries(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    training_episode: dict[str, Any],
    user_models: Sequence[torch.nn.Module],
) -> list[dict[str, Any]]:
    """Roll one channel episode forward and collect its visited joint states.

    Payload episodes add blocks until completion; streaming episodes contribute
    independent block requests. No static channel/n grid is constructed.
    """
    seed = int(training_episode["seed"])
    scenario = dict(training_episode["scenario"])
    scenario_mode = str(scenario.get("mode", PAYLOAD_MODE))
    episode_system_params = with_monte_carlo_sample_snr_by_user(
        system_params,
        training_episode["snr_db_by_user"],
    )
    system = DownlinkSystem(episode_system_params, seed=int(seed))
    working_F = system.clone_precoders()
    episode_queries: list[dict[str, Any]] = []
    max_blocks = int(sim_params.get("max_total_blocks", 256))

    if scenario_mode == STREAMING_MODE:
        block_targets = np.asarray(scenario["streaming_bit_targets_by_block"], dtype=int)
        num_blocks = int(scenario["number_of_blocks"])
        for block in range(num_blocks):
            target_bits = [int(block_targets[int(k), int(block)]) for k in range(system.K)]
            active_mask = [1 if int(bits) > 0 else 0 for bits in target_bits]
            if not any(active_mask):
                continue

            full_scenario = _build_training_block_scenario(
                system,
                working_F,
                int(block),
                active_mask,
                scenario_mode=str(scenario_mode),
            )
            full_n_targets = [
                int(system.T[int(k)]) if int(active_mask[int(k)]) > 0 else 0
                for k in range(system.K)
            ]
            full_forward = _scenario_forward_pass(
                episode_system_params,
                full_scenario,
                user_models,
                full_n_targets,
                anchor_bits=[0 for _ in range(system.K)],
            )
            full_metrics = _scenario_metrics_from_forward(full_forward)
            episode_queries.append(
                _build_rollout_query_from_downlink_state(
                    full_scenario,
                    full_n_targets,
                    full_metrics,
                    [0 for _ in range(system.K)],
                    rollout_phase="full_block",
                    rollout_stage="full_block",
                    frontier_query=False,
                )
            )

            initial_supported_bits = _supported_bits_from_forward(full_forward)
            service_mask = [1 if int(initial_supported_bits[int(k)]) > 0 else 0 for k in range(system.K)]
            if not any(service_mask):
                for k in range(system.K):
                    ensure_precoder_block(system, working_F, int(k), int(block))
                    working_F[int(k)][int(block)] = make_zero_precoder(system, int(k))
                continue

            service_scenario = _build_training_block_scenario(
                system,
                working_F,
                int(block),
                service_mask,
                scenario_mode=str(scenario_mode),
            )
            service_n_targets = [
                int(system.T[int(k)]) if int(service_mask[int(k)]) > 0 else 0
                for k in range(system.K)
            ]
            service_forward = _scenario_forward_pass(
                episode_system_params,
                service_scenario,
                user_models,
                service_n_targets,
                anchor_bits=[0 for _ in range(system.K)],
            )
            service_metrics = _scenario_metrics_from_forward(service_forward)
            if service_mask != active_mask:
                episode_queries.append(
                    _build_rollout_query_from_downlink_state(
                        service_scenario,
                        service_n_targets,
                        service_metrics,
                        [0 for _ in range(system.K)],
                        rollout_phase="full_block",
                        rollout_stage="service_mask",
                        frontier_query=False,
                    )
                )

            supported_bits = _supported_bits_from_forward(service_forward)
            committed_bits = [
                min(int(target_bits[int(k)]), int(supported_bits[int(k)]))
                if int(service_mask[int(k)]) > 0
                else 0
                for k in range(system.K)
            ]
            reducible_users = [
                int(k)
                for k in range(system.K)
                if int(service_mask[int(k)]) > 0 and int(committed_bits[int(k)]) >= int(target_bits[int(k)]) > 0
            ]
            final_forward = service_forward
            if len(reducible_users) > 0:
                tail_queries, final_forward = _collect_downlink_tail_rollout_queries(
                    episode_system_params,
                    sim_params,
                    system,
                    working_F,
                    user_models,
                    block=int(block),
                    scenario_mode=str(scenario_mode),
                    committed_bits=committed_bits,
                    service_mask=service_mask,
                    reducible_users=reducible_users,
                )
                episode_queries.extend(tail_queries)

            _apply_forward_to_working_precoders(system, working_F, int(block), final_forward)
        return episode_queries

    remaining = np.asarray(scenario["payload_bits_per_user"], dtype=int).copy()
    block = 0
    payload_stop_reason: str | None = None
    payload_blocks_visited = 0
    while np.any(remaining > 0) and block < max_blocks:
        payload_blocks_visited = int(block) + 1
        active_mask = [1 if int(remaining[int(k)]) > 0 else 0 for k in range(system.K)]
        full_scenario = _build_training_block_scenario(
            system,
            working_F,
            int(block),
            active_mask,
            scenario_mode=str(scenario_mode),
        )
        full_n_targets = [
            int(system.T[int(k)]) if int(active_mask[int(k)]) > 0 else 0
            for k in range(system.K)
        ]
        full_forward = _scenario_forward_pass(
            episode_system_params,
            full_scenario,
            user_models,
            full_n_targets,
            anchor_bits=[0 for _ in range(system.K)],
        )
        full_metrics = _scenario_metrics_from_forward(full_forward)
        episode_queries.append(
            _build_rollout_query_from_downlink_state(
                full_scenario,
                full_n_targets,
                full_metrics,
                [0 for _ in range(system.K)],
                rollout_phase="full_block",
                rollout_stage="full_block",
                frontier_query=False,
            )
        )

        initial_supported_bits = _supported_bits_from_forward(full_forward)
        service_mask = [1 if int(initial_supported_bits[int(k)]) > 0 else 0 for k in range(system.K)]
        if not any(service_mask):
            # The full-block state above is still a valid rate-training query. Repeating
            # zero-service blocks cannot change the rollout state; stop here and let the
            # next training epoch revisit this episode with the updated network.
            episode_queries[-1]["rollout_stage"] = "zero_service_frontier"
            payload_stop_reason = "zero_service_frontier"
            break

        service_scenario = _build_training_block_scenario(
            system,
            working_F,
            int(block),
            service_mask,
            scenario_mode=str(scenario_mode),
        )
        service_n_targets = [
            int(system.T[int(k)]) if int(service_mask[int(k)]) > 0 else 0
            for k in range(system.K)
        ]
        service_forward = _scenario_forward_pass(
            episode_system_params,
            service_scenario,
            user_models,
            service_n_targets,
            anchor_bits=[0 for _ in range(system.K)],
        )
        service_metrics = _scenario_metrics_from_forward(service_forward)
        if service_mask != active_mask:
            episode_queries.append(
                _build_rollout_query_from_downlink_state(
                    service_scenario,
                    service_n_targets,
                    service_metrics,
                    [0 for _ in range(system.K)],
                    rollout_phase="full_block",
                    rollout_stage="service_mask",
                    frontier_query=False,
                )
            )

        supported_bits = _supported_bits_from_forward(service_forward)
        committed_bits = [
            min(int(remaining[int(k)]), int(supported_bits[int(k)]))
            if int(service_mask[int(k)]) > 0
            else 0
            for k in range(system.K)
        ]
        reducible_users = [
            int(k)
            for k in range(system.K)
            if int(service_mask[int(k)]) > 0 and int(committed_bits[int(k)]) >= int(remaining[int(k)]) > 0
        ]
        final_forward = service_forward
        if len(reducible_users) > 0:
            tail_queries, final_forward = _collect_downlink_tail_rollout_queries(
                episode_system_params,
                sim_params,
                system,
                working_F,
                user_models,
                block=int(block),
                scenario_mode=str(scenario_mode),
                committed_bits=committed_bits,
                service_mask=service_mask,
                reducible_users=reducible_users,
            )
            episode_queries.extend(tail_queries)

        _apply_forward_to_working_precoders(system, working_F, int(block), final_forward)
        remaining = np.maximum(remaining - np.asarray(committed_bits, dtype=int), 0)
        block += 1

    if episode_queries:
        episode_completed = not bool(np.any(remaining > 0))
        episode_queries[-1]["episode_completed"] = bool(episode_completed)
        episode_queries[-1]["episode_stop_reason"] = (
            "payload_completed"
            if episode_completed
            else payload_stop_reason or "max_total_blocks_reached"
        )
        episode_queries[-1]["episode_blocks_visited"] = int(payload_blocks_visited)
        episode_queries[-1]["episode_remaining_bits_at_stop"] = [int(value) for value in remaining]

    return episode_queries


def _generate_rollout_queries_for_downlink(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    training_episodes: Sequence[dict[str, Any]],
    user_models: Sequence[torch.nn.Module],
) -> list[dict[str, Any]]:
    """Regenerate all training queries from the current model parameters.

    What: execute a complete payload or streaming rollout for each base sample and merge
    every visited joint state. Why: training data follows states induced by the improving
    networks rather than remaining a fixed hand-built grid throughout training.
    """
    rollout_queries: list[dict[str, Any]] = []
    for training_episode in training_episodes:
        rollout_queries.extend(
            _collect_downlink_episode_rollout_queries(
                system_params,
                sim_params,
                training_episode,
                user_models,
            )
        )

    return rollout_queries


def _summarize_downlink_rollout_queries(rollout_queries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = _summarize_training_cases_with_n_kl(
        rollout_queries,
        n_key="n_targets",
        global_n_key="global_active_user_rollout_queries_by_n_kl",
        per_user_n_key="per_user_active_user_rollout_queries_by_n_kl",
    )
    frontier_queries = [query for query in rollout_queries if bool(query.get("frontier_query", False))]
    frontier_summary = _summarize_training_cases_with_n_kl(
        frontier_queries,
        n_key="n_targets",
        global_n_key="global_active_user_frontier_rollout_queries_by_n_kl",
        per_user_n_key="per_user_active_user_frontier_rollout_queries_by_n_kl",
    )
    feasible_queries = int(sum(1 for query in rollout_queries if bool(query.get("rollout_feasible", False))))
    infeasible_queries = int(len(rollout_queries) - feasible_queries)
    payload_episode_outcomes = {
        int(query["seed"]): {
            "completed": bool(query["episode_completed"]),
            "stop_reason": str(query["episode_stop_reason"]),
            "blocks_visited": int(query["episode_blocks_visited"]),
            "remaining_bits_at_stop": [
                int(value) for value in query["episode_remaining_bits_at_stop"]
            ],
        }
        for query in rollout_queries
        if "episode_completed" in query
    }
    return {
        **summary,
        **frontier_summary,
        "total_rollout_queries": int(len(rollout_queries)),
        "feasible_rollout_queries": int(feasible_queries),
        "infeasible_rollout_queries": int(infeasible_queries),
        "frontier_rollout_queries": int(len(frontier_queries)),
        "payload_episode_outcomes": {
            str(seed): outcome for seed, outcome in sorted(payload_episode_outcomes.items())
        },
        "completed_payload_episodes": int(
            sum(bool(outcome["completed"]) for outcome in payload_episode_outcomes.values())
        ),
        "truncated_payload_episodes": int(
            sum(not bool(outcome["completed"]) for outcome in payload_episode_outcomes.values())
        ),
    }


def _empty_case_count_summary(global_n_key: str, per_user_n_key: str) -> dict[str, Any]:
    return {
        "total_training_cases": 0,
        "training_cases_by_seed": {},
        "training_cases_by_active_user_count": {},
        "training_cases_by_active_mask": {},
        "active_user_cases_per_user": [],
        global_n_key: {},
        per_user_n_key: [],
    }


def _serialize_n_kl_case_counts(
    global_counts: dict[int, int],
    per_user_counts: Sequence[dict[int, int]],
    *,
    global_key: str,
    per_user_key: str,
) -> dict[str, Any]:
    return {
        global_key: {str(int(k)): int(v) for k, v in sorted(global_counts.items())},
        per_user_key: [
            {str(int(k)): int(v) for k, v in sorted(user_counts.items())}
            for user_counts in per_user_counts
        ],
    }


def _summarize_training_case_structure(training_cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(training_cases) == 0:
        return {
            "total_training_cases": 0,
            "training_cases_by_seed": {},
            "training_cases_by_block": {},
            "training_cases_by_active_user_count": {},
            "training_cases_by_active_mask": {},
            "active_training_cases_per_user": [],
        }

    K = len(training_cases[0]["active_mask"])
    cases_by_seed: dict[int, int] = {}
    cases_by_block: dict[int, int] = {}
    cases_by_active_user_count: dict[int, int] = {}
    cases_by_active_mask: dict[str, int] = {}
    active_cases_per_user = [0 for _ in range(K)]

    for case in training_cases:
        seed = int(case["seed"])
        block = int(case.get("block", 0))
        active_mask = [int(v) for v in case["active_mask"]]
        active_users = [int(k) for k, is_active in enumerate(active_mask) if int(is_active) > 0]
        active_count = len(active_users)
        mask_key = "".join(str(int(v)) for v in active_mask)

        cases_by_seed[seed] = cases_by_seed.get(seed, 0) + 1
        cases_by_block[block] = cases_by_block.get(block, 0) + 1
        cases_by_active_user_count[active_count] = (
            cases_by_active_user_count.get(active_count, 0) + 1
        )
        cases_by_active_mask[mask_key] = cases_by_active_mask.get(mask_key, 0) + 1
        for k in active_users:
            active_cases_per_user[int(k)] += 1

    return {
        "total_training_cases": int(len(training_cases)),
        "training_cases_by_seed": {str(int(k)): int(v) for k, v in sorted(cases_by_seed.items())},
        "training_cases_by_block": {str(int(k)): int(v) for k, v in sorted(cases_by_block.items())},
        "training_cases_by_active_user_count": {
            str(int(k)): int(v) for k, v in sorted(cases_by_active_user_count.items())
        },
        "training_cases_by_active_mask": {
            str(k): int(v) for k, v in sorted(cases_by_active_mask.items())
        },
        "active_training_cases_per_user": [int(v) for v in active_cases_per_user],
    }


def _count_active_user_cases_by_n_kl(
    training_cases: Sequence[dict[str, Any]],
    *,
    n_key: str,
) -> tuple[dict[int, int], list[dict[int, int]]]:
    if len(training_cases) == 0:
        return {}, []

    K = len(training_cases[0]["active_mask"])
    global_counts: dict[int, int] = {}
    per_user_counts: list[dict[int, int]] = [{} for _ in range(K)]

    for training_case in training_cases:
        active_mask = [int(v) for v in training_case["active_mask"]]
        n_targets = training_case[n_key]
        for k, is_active in enumerate(active_mask):
            if int(is_active) <= 0:
                continue
            n_val = int(n_targets[int(k)])
            per_user_counts[int(k)][n_val] = per_user_counts[int(k)].get(n_val, 0) + 1
            global_counts[n_val] = global_counts.get(n_val, 0) + 1

    return global_counts, per_user_counts


def _summarize_training_cases_with_n_kl(
    training_cases: Sequence[dict[str, Any]],
    *,
    n_key: str,
    global_n_key: str,
    per_user_n_key: str,
) -> dict[str, Any]:
    if len(training_cases) == 0:
        return _empty_case_count_summary(global_n_key, per_user_n_key)

    summary = _summarize_training_case_structure(training_cases)
    global_counts, per_user_counts = _count_active_user_cases_by_n_kl(training_cases, n_key=n_key)
    summary.update(
        _serialize_n_kl_case_counts(
            global_counts,
            per_user_counts,
            global_key=global_n_key,
            per_user_key=per_user_n_key,
        )
    )
    return summary


def summarize_training_dataset(training_episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Describe the immutable base samples supplied before rollout generation.

    It reports seed count, sample unit, scenario, per-user SNR coverage, and users.
    Rollout-query counts are reported separately because one channel can visit many n states.
    """
    if len(training_episodes) == 0:
        return {
            "total_training_samples": 0,
            "training_sample_kind": "unknown",
            "training_sample_unit": "unknown",
            "training_samples_by_seed": {},
            "training_samples_per_user": [],
            "base_dataset_kind": "unknown",
            "scenario_modes": [],
        }
    K = int(training_episodes[0].get("num_users", 0))
    episodes_by_seed = {str(int(case["seed"])): 1 for case in training_episodes}
    episodes_by_user_snr_db: list[dict[str, int]] = [{} for _ in range(K)]
    for case in training_episodes:
        for user, snr_db in enumerate(case["snr_db_by_user"]):
            snr_key = f"{float(snr_db):.6g}"
            user_counts = episodes_by_user_snr_db[int(user)]
            user_counts[snr_key] = user_counts.get(snr_key, 0) + 1
    scenario_modes = sorted({str(case.get("scenario_mode", PAYLOAD_MODE)) for case in training_episodes})
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
        "training_samples_by_seed": episodes_by_seed,
        "training_samples_per_user": [int(len(training_episodes)) for _ in range(K)],
        "training_samples_by_user_snr_db": episodes_by_user_snr_db,
        "training_snr_db_ranges": [
            [
                float(min(float(case["snr_db_by_user"][user]) for case in training_episodes)),
                float(max(float(case["snr_db_by_user"][user]) for case in training_episodes)),
            ]
            for user in range(K)
        ],
        "base_dataset_kind": sample_kind,
        "scenario_modes": scenario_modes,
    }


def _accumulate_active_user_case_uses_by_n_kl(
    training_cases: Sequence[dict[str, Any]],
    *,
    n_key: str,
    global_counts: dict[int, int],
    per_user_counts: list[dict[int, int]],
) -> None:
    if len(training_cases) == 0:
        return
    K = len(training_cases[0]["active_mask"])
    if len(per_user_counts) == 0:
        per_user_counts.extend({} for _ in range(K))

    current_global_counts, current_per_user_counts = _count_active_user_cases_by_n_kl(training_cases, n_key=n_key)
    for n_val, count in current_global_counts.items():
        global_counts[int(n_val)] = global_counts.get(int(n_val), 0) + int(count)
    for k in range(K):
        for n_val, count in current_per_user_counts[k].items():
            per_user_counts[k][int(n_val)] = per_user_counts[k].get(int(n_val), 0) + int(count)


def build_training_dataset(
    train_seeds: Sequence[int],
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    *,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Create the immutable base dataset from requested channel seeds and SNR ranges.

    Payload: one sample is a channel episode whose later blocks are generated during its
    rollout. Streaming: one sample is one independent channel block. This function does
    not enumerate n values; the current-network rollout creates those training queries.
    """
    snr_db_by_user_by_seed = build_training_snr_schedule(
        train_seeds,
        sim_params["monte_carlo_training_snr_db_ranges"],
        num_users=int(system_params["K"]),
    )
    episodes: list[dict[str, Any]] = []

    for seed in train_seeds:
        if verbose:
            print(
                format_log_line(
                    "[DL Monte Carlo Dataset]",
                    phase="collect",
                    seed=int(seed),
                    base_dataset=(
                        "payload_channel_episode"
                        if str(sim_params["experiment_scenario_mode"]) == PAYLOAD_MODE
                        else "streaming_block"
                    ),
                    snr_db_by_user=snr_db_by_user_by_seed[int(seed)],
                )
            )
        configure_determinism(int(seed))
        scenario = build_monte_carlo_sample_scenarios_for_seeds(
            system_params,
            sim_params,
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






__all__ = ["build_training_dataset", "summarize_training_dataset"]
