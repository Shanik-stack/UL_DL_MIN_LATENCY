from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from latency_optimization.core.validation import require_choice


FORWARD_BACKWARD_FLOP_FACTOR = 3.0
UPLINK_INPUT_MODES = {
    "channel_only",
    "channel_sigma_epsilon",
    "channel_noise_epsilon",
    "channel_noise_epsilon_n",
    "channel_sigma_epsilon_n",
}
DOWNLINK_INPUT_MODES = {
    "channel_only",
    "block_context_noise_epsilon",
    "block_context_noise_epsilon_n",
}
CONVERGENCE_UPDATE_MODES = {"precoder_net", "direct_precoder", "closed_form_benchmark"}
DOWNLINK_NET_SCOPES = {"per_user_nets", "bs_shared_net"}


def _layer_flops(in_dim: int, out_dim: int) -> int:
    return int((2 * int(in_dim) * int(out_dim)) + int(out_dim))


def _relu_flops(width: int) -> int:
    return int(width)


def _hidden_dims(out_dim: int) -> tuple[int, int, int]:
    out_dim = int(out_dim)
    return (
        max(256, 8 * out_dim),
        max(128, 4 * out_dim),
        max(64, 2 * out_dim),
    )


def _mlp_forward_flops(in_dim: int, out_dim: int) -> int:
    h1, h2, h3 = _hidden_dims(int(out_dim))
    total = 0
    total += _layer_flops(in_dim, h1) + _relu_flops(h1)
    total += _layer_flops(h1, h2) + _relu_flops(h2)
    total += _layer_flops(h2, h3) + _relu_flops(h3)
    total += _layer_flops(h3, int(out_dim))
    return int(total)


def _sum_numeric_mapping(values: Mapping[str, Any] | None) -> int:
    if not isinstance(values, Mapping):
        return 0
    total = 0
    for value in values.values():
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
    return int(total)


def _sum_numeric_sequence_mappings(values: Sequence[Mapping[str, Any]] | None) -> list[int]:
    if not isinstance(values, Sequence):
        return []
    return [_sum_numeric_mapping(value) for value in values if isinstance(value, Mapping)]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _summed_nested_state_count(all_user_block_results: Sequence[Sequence[Sequence[dict[str, Any]]]]) -> list[int]:
    per_user_counts: list[int] = []
    for user_blocks in all_user_block_results:
        count = 0
        for block_states in user_blocks:
            count += len(block_states)
        per_user_counts.append(int(count))
    return per_user_counts


def _is_nonstring_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _uplink_state_epoch_count(state: Mapping[str, Any]) -> int:
    kkt_history = state.get("kkt_history")
    if _is_nonstring_sequence(kkt_history):
        return int(len(kkt_history))

    loss_curve = state.get("loss_curve")
    if _is_nonstring_sequence(loss_curve):
        return int(len(loss_curve))

    return 0


def _summed_uplink_solver_work(
    all_user_block_results: Sequence[Sequence[Sequence[dict[str, Any]]]],
) -> tuple[list[int], list[int], list[int], int]:
    per_user_epochs: list[int] = []
    per_user_optimizer_updates: list[int] = []
    per_user_solver_calls: list[int] = []
    total_solver_calls = 0
    for user_blocks in all_user_block_results:
        user_epochs = 0
        user_optimizer_updates = 0
        user_solver_calls = 0
        for block_states in user_blocks:
            for state in block_states:
                if not isinstance(state, Mapping):
                    continue
                epochs = _uplink_state_epoch_count(state)
                if epochs <= 0:
                    continue
                total_solver_calls += 1
                user_solver_calls += 1
                user_epochs += int(epochs)
                user_optimizer_updates += max(int(epochs) - 1, 0)
        per_user_epochs.append(int(user_epochs))
        per_user_optimizer_updates.append(int(user_optimizer_updates))
        per_user_solver_calls.append(int(user_solver_calls))
    return (
        per_user_epochs,
        per_user_optimizer_updates,
        per_user_solver_calls,
        int(total_solver_calls),
    )


def _uplink_model_forward_flops_from_spec(spec: Mapping[str, Any]) -> int:
    nr = int(spec["Nr"])
    nt = int(spec["Nt"])
    dk = int(spec.get("dk", 0))
    input_mode = require_choice(spec.get("input_mode", "channel_only"), UPLINK_INPUT_MODES, "uplink model input_mode")
    out_dim = 2 * nt * dk

    if input_mode == "channel_sigma_epsilon":
        in_dim = (2 * nr * nt) + 2
    elif input_mode == "channel_noise_epsilon":
        in_dim = (2 * nr * nt) + (2 * nr * nr) + 1
    elif input_mode == "channel_noise_epsilon_n":
        in_dim = (2 * nr * nt) + (2 * nr * nr) + 2
    elif input_mode == "channel_sigma_epsilon_n":
        in_dim = (2 * nr * nt) + 3
    else:
        in_dim = 2 * nr * nt
    return _mlp_forward_flops(in_dim, out_dim)


def _downlink_model_forward_flops_from_spec(spec: Mapping[str, Any]) -> int:
    nr = int(spec["nr"])
    nb = int(spec["nb"])
    dk = int(spec.get("dk", 0))
    input_mode = require_choice(spec.get("input_mode", "channel_only"), DOWNLINK_INPUT_MODES, "downlink model input_mode")
    out_dim = 2 * nb * dk

    if input_mode == "block_context_noise_epsilon":
        k_count = int(spec.get("context_k", 1))
        max_nr = int(spec.get("context_max_nr", nr))
        max_nb = int(spec.get("context_max_nb", nb))
        in_dim = (2 * k_count * max_nr * max_nb) + k_count + (2 * max_nr * max_nr) + 1
    elif input_mode == "block_context_noise_epsilon_n":
        k_count = int(spec.get("context_k", 1))
        max_nr = int(spec.get("context_max_nr", nr))
        max_nb = int(spec.get("context_max_nb", nb))
        in_dim = (2 * k_count * max_nr * max_nb) + k_count + (2 * max_nr * max_nr) + 2
    else:
        in_dim = 2 * nr * nb
    return _mlp_forward_flops(in_dim, out_dim)


def build_uplink_sigma_context_specs(system_params: Mapping[str, Any]) -> list[dict[str, Any]]:
    nr = list(system_params.get("NR", []))
    nt = list(system_params.get("NT", []))
    dk = list(system_params.get("dk", []))
    return [
        {
            "Nr": int(nr[k]),
            "Nt": int(nt[k]),
            "dk": int(dk[k]),
            "input_mode": "channel_sigma_epsilon",
        }
        for k in range(len(nr))
    ]


def build_downlink_channel_only_specs(system_params: Mapping[str, Any]) -> list[dict[str, Any]]:
    nr = list(system_params.get("Nr", []))
    nb = list(system_params.get("Nb", []))
    dk = list(system_params.get("dk", []))
    return [
        {
            "nr": int(nr[k]),
            "nb": int(nb[k]),
            "dk": int(dk[k]),
            "input_mode": "channel_only",
        }
        for k in range(len(nr))
    ]


def build_forward_flops_per_user(
    model_specs: Sequence[Mapping[str, Any]],
    *,
    link: str,
) -> list[int]:
    link_name = require_choice(link, {"uplink", "downlink"}, "link")
    if link_name == "uplink":
        return [_uplink_model_forward_flops_from_spec(spec) for spec in model_specs]
    return [_downlink_model_forward_flops_from_spec(spec) for spec in model_specs]


def _format_large_number(value: float) -> str:
    return f"{float(value):.3e}"


def format_experiment_cost_lines(experiment_cost: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(experiment_cost, Mapping) or len(experiment_cost) == 0:
        return []

    lines = [
        "",
        "Experiment cost",
        f"Core wall time total (s): {_safe_float(experiment_cost.get('core_wall_time_seconds_total')):.6f}",
        f"Core wall time training (s): {_safe_float(experiment_cost.get('core_wall_time_seconds_training')):.6f}",
        f"Testing-phase wall time (s): {_safe_float(experiment_cost.get('core_wall_time_seconds_testing')):.6f}",
        f"Forward+backward NN FLOPs: {_format_large_number(_safe_float(experiment_cost.get('estimated_nn_training_flops')))}",
        f"Forward-only NN FLOPs: {_format_large_number(_safe_float(experiment_cost.get('estimated_nn_inference_flops')))}",
        f"Total NN FLOPs: {_format_large_number(_safe_float(experiment_cost.get('estimated_nn_total_flops')))}",
        (
            "Forward+backward NN evaluations: "
            f"{_safe_int(experiment_cost.get('training_forward_backward_sample_equivalents'))}"
        ),
    ]

    if "forward_only_beam_evaluations" in experiment_cost:
        lines.append(
            f"Forward-only beam evaluations: {_safe_int(experiment_cost.get('forward_only_beam_evaluations'))}"
        )
    else:
        lines.append(f"Forward-only NN evaluations: {_safe_int(experiment_cost.get('inference_forward_calls'))}")

    if "actual_optimizer_updates" in experiment_cost:
        lines.append(
            f"Actual optimizer updates: {_safe_int(experiment_cost.get('actual_optimizer_updates'))}"
        )
    else:
        lines.append(f"Optimizer steps: {_safe_int(experiment_cost.get('optimizer_steps'))}")

    workload = experiment_cost.get("workload_counters", {})
    if isinstance(workload, Mapping) and len(workload) > 0:
        lines.append("Workload counters:")
        lines.append(str(workload))

    notes = experiment_cost.get("notes", [])
    if isinstance(notes, Sequence) and not isinstance(notes, (str, bytes)) and len(notes) > 0:
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")

    return lines


def build_uplink_convergence_cost(
    system_params: Mapping[str, Any],
    convergence_data: Mapping[str, Any],
    *,
    core_wall_time_seconds_total: float,
) -> dict[str, Any]:
    update_mode = require_choice(
        convergence_data.get("convergence_precoder_update_mode", "precoder_net"),
        CONVERGENCE_UPDATE_MODES,
        "convergence_precoder_update_mode",
    )
    model_specs = build_uplink_sigma_context_specs(system_params)
    per_user_forward_flops = build_forward_flops_per_user(model_specs, link="uplink")
    block_results = convergence_data.get("all_user_block_results_train", [])
    (
        solver_epochs_per_user,
        optimizer_updates_per_user,
        solver_calls_per_user,
        solver_calls,
    ) = _summed_uplink_solver_work(block_results)
    visited_states_per_user = _summed_nested_state_count(block_results)

    training_forward_backward_equivalents = int(sum(solver_epochs_per_user))
    actual_optimizer_updates = int(sum(optimizer_updates_per_user))
    inference_forward_calls = int(sum(visited_states_per_user))
    if update_mode == "direct_precoder":
        estimated_training_flops = 0.0
        estimated_inference_flops = 0.0
        per_user_forward_flops = []
        notes = [
            "This convergence run used direct complex-precoder optimization, so NN FLOP counters are reported as 0.0.",
            "Testing-phase wall time is 0.0 for convergence because there is no separate train/test split.",
        ]
    else:
        estimated_training_flops = float(
            sum(
                FORWARD_BACKWARD_FLOP_FACTOR * per_user_forward_flops[k] * solver_epochs_per_user[k]
                for k in range(min(len(per_user_forward_flops), len(solver_epochs_per_user)))
            )
        )
        estimated_inference_flops = float(
            sum(
                per_user_forward_flops[k] * visited_states_per_user[k]
                for k in range(min(len(per_user_forward_flops), len(visited_states_per_user)))
            )
        )
        notes = [
            "Forward+backward NN FLOPs count the online solver epochs that optimize the uplink precoder net.",
            "Forward-only NN FLOPs count forward-only beam evaluations inside the same online convergence routine, not a separate testing phase.",
            "Testing-phase wall time is 0.0 for convergence because there is no separate train/test split.",
        ]

    return {
        "core_wall_time_seconds_total": float(core_wall_time_seconds_total),
        "core_wall_time_seconds_training": float(core_wall_time_seconds_total),
        "core_wall_time_seconds_testing": 0.0,
        "estimated_nn_training_flops": float(estimated_training_flops),
        "estimated_nn_inference_flops": float(estimated_inference_flops),
        "estimated_nn_total_flops": float(estimated_training_flops + estimated_inference_flops),
        "training_forward_backward_sample_equivalents": int(training_forward_backward_equivalents),
        "inference_forward_calls": int(inference_forward_calls),
        "optimizer_steps": int(actual_optimizer_updates),
        "actual_optimizer_updates": int(actual_optimizer_updates),
        "forward_only_beam_evaluations": int(inference_forward_calls),
        "per_user_forward_flops": [int(v) for v in per_user_forward_flops],
        "workload_counters": {
            "solver_calls": int(solver_calls),
            "solver_calls_per_user": [int(v) for v in solver_calls_per_user],
            "solver_epochs_per_user": [int(v) for v in solver_epochs_per_user],
            "optimizer_updates_per_user": [int(v) for v in optimizer_updates_per_user],
            "visited_candidate_n_states_per_user": [int(v) for v in visited_states_per_user],
        },
        "notes": notes,
    }


def build_downlink_convergence_cost(
    system_params: Mapping[str, Any],
    sim_params: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    core_wall_time_seconds_total: float,
) -> dict[str, Any]:
    update_mode = require_choice(
        result.get("convergence_precoder_update_mode", sim_params["convergence_precoder_update_mode"]),
        CONVERGENCE_UPDATE_MODES,
        "convergence_precoder_update_mode",
    )
    model_scope = require_choice(
        result.get("downlink_precoder_net_scope", sim_params["downlink_precoder_net_scope"]),
        DOWNLINK_NET_SCOPES,
        "downlink_precoder_net_scope",
    )
    model_specs = list(result.get("user_model_specs", []))
    if not model_specs:
        model_specs = build_downlink_channel_only_specs(system_params)
    model_forward_flops = build_forward_flops_per_user(model_specs, link="downlink")
    epoch_history = [entry for entry in result.get("epoch_history", []) if isinstance(entry, Mapping)]

    optimizer_updates = 0
    forward_only_calls = 0
    estimated_training_flops = 0.0
    estimated_inference_flops = 0.0
    for entry in epoch_history:
        updated_users = [int(value) for value in entry.get("updated_user_ids", [])]
        model_indices = [0] if model_scope == "bs_shared_net" and updated_users else updated_users
        optimizer_updates += len(model_indices)
        forward_only_calls += len(model_indices)
        if update_mode == "direct_precoder":
            continue
        for model_index in model_indices:
            if 0 <= model_index < len(model_forward_flops):
                forward_flops = float(model_forward_flops[model_index])
                estimated_training_flops += FORWARD_BACKWARD_FLOP_FACTOR * forward_flops
                estimated_inference_flops += forward_flops

    if update_mode == "direct_precoder":
        model_forward_flops = []
        optimizer_updates = 0
        forward_only_calls = 0
        notes = [
            "This run optimized complex precoders directly, so neural-network FLOP counters are 0.0.",
            "Testing-phase wall time is 0.0 because convergence has no separate testing phase.",
        ]
    else:
        notes = [
            "Forward+backward NN FLOPs are the network work used to optimize the precoder.",
            "Forward-only NN FLOPs are beam extraction and evaluation work in the same optimization routine.",
            "Testing-phase wall time is 0.0 because convergence has no separate testing phase.",
        ]

    return {
        "core_wall_time_seconds_total": float(core_wall_time_seconds_total),
        "core_wall_time_seconds_training": float(core_wall_time_seconds_total),
        "core_wall_time_seconds_testing": 0.0,
        "estimated_nn_training_flops": float(estimated_training_flops),
        "estimated_nn_inference_flops": float(estimated_inference_flops),
        "estimated_nn_total_flops": float(estimated_training_flops + estimated_inference_flops),
        "training_forward_backward_sample_equivalents": int(optimizer_updates),
        "inference_forward_calls": int(forward_only_calls),
        "optimizer_steps": int(optimizer_updates),
        "actual_optimizer_updates": int(optimizer_updates),
        "forward_only_beam_evaluations": int(forward_only_calls),
        "per_user_forward_flops": [int(value) for value in model_forward_flops],
        "workload_counters": {
            "block_optimization_epochs": int(len(epoch_history)),
            "updated_model_calls": int(optimizer_updates),
        },
        "notes": notes,
    }


def build_uplink_monte_carlo_training_cost(
    train_artifact: Mapping[str, Any],
    *,
    batch_size: int,
    core_wall_time_seconds_training: float,
) -> dict[str, Any]:
    model_specs = list(train_artifact.get("user_model_specs", []))
    per_user_forward_flops = build_forward_flops_per_user(model_specs, link="uplink")
    training_history = train_artifact.get("precoder_net_training_history", {})
    rollout_counts_per_user = _sum_numeric_sequence_mappings(
        training_history.get("cumulative_rollout_queries_by_n_kl", {}).get(
            "per_user_rollout_queries_by_n_kl_over_all_epochs",
            [],
        )
    )
    rollout_summaries_per_user = training_history.get("rollout_query_summaries_per_user", [])
    optimizer_steps = 0
    for user_summaries in rollout_summaries_per_user:
        if not isinstance(user_summaries, Sequence):
            continue
        for epoch_summary in user_summaries:
            if not isinstance(epoch_summary, Mapping):
                continue
            total_queries = int(epoch_summary.get("total_rollout_queries", 0))
            optimizer_steps += int(math.ceil(total_queries / max(int(batch_size), 1)))

    train_eval_states_per_user = _summed_nested_state_count(train_artifact.get("all_user_block_results_train", []))
    estimated_training_flops = float(
        sum(
            FORWARD_BACKWARD_FLOP_FACTOR * per_user_forward_flops[k] * rollout_counts_per_user[k]
            for k in range(min(len(per_user_forward_flops), len(rollout_counts_per_user)))
        )
    )
    estimated_inference_flops = float(
        sum(
            per_user_forward_flops[k] * train_eval_states_per_user[k]
            for k in range(min(len(per_user_forward_flops), len(train_eval_states_per_user)))
        )
    )

    return {
        "core_wall_time_seconds_total": float(core_wall_time_seconds_training),
        "core_wall_time_seconds_training": float(core_wall_time_seconds_training),
        "core_wall_time_seconds_testing": 0.0,
        "estimated_nn_training_flops": float(estimated_training_flops),
        "estimated_nn_inference_flops": float(estimated_inference_flops),
        "estimated_nn_total_flops": float(estimated_training_flops + estimated_inference_flops),
        "training_forward_backward_sample_equivalents": int(sum(rollout_counts_per_user)),
        "inference_forward_calls": int(sum(train_eval_states_per_user)),
        "optimizer_steps": int(optimizer_steps),
        "per_user_forward_flops": [int(v) for v in per_user_forward_flops],
        "workload_counters": {
            "training_rollout_queries_per_user": [int(v) for v in rollout_counts_per_user],
            "train_eval_candidate_n_states_per_user": [int(v) for v in train_eval_states_per_user],
            "batch_size": int(batch_size),
        },
        "notes": [
            "Forward+backward NN FLOPs count the rollout-query training passes that optimize the uplink precoder net.",
            "Forward-only NN FLOPs count the forward-only beam evaluations used in the post-training train-eval pass.",
        ],
    }


def build_uplink_monte_carlo_total_cost(
    train_artifact: Mapping[str, Any],
    evaluation_cost_counters: Mapping[str, Any] | Sequence[int],
    *,
    batch_size: int,
    core_wall_time_seconds_training: float,
    core_wall_time_seconds_testing: float,
) -> dict[str, Any]:
    training_cost = build_uplink_monte_carlo_training_cost(
        train_artifact,
        batch_size=batch_size,
        core_wall_time_seconds_training=core_wall_time_seconds_training,
    )
    per_user_forward_flops = [int(v) for v in training_cost.get("per_user_forward_flops", [])]
    if isinstance(evaluation_cost_counters, Mapping):
        test_states_per_user = [
            int(v)
            for v in evaluation_cost_counters.get("per_user_candidate_n_states", [])
        ]
        per_user_eval_calls = [
            int(v)
            for v in evaluation_cost_counters.get(
                "per_user_forward_calls",
                test_states_per_user,
            )
        ]
        total_eval_calls = int(
            evaluation_cost_counters.get("total_forward_calls", sum(per_user_eval_calls))
        )
    else:
        test_states_per_user = [int(v) for v in evaluation_cost_counters]
        per_user_eval_calls = list(test_states_per_user)
        total_eval_calls = int(sum(per_user_eval_calls))
    testing_inference_flops = float(
        sum(
            per_user_forward_flops[k] * per_user_eval_calls[k]
            for k in range(min(len(per_user_forward_flops), len(per_user_eval_calls)))
        )
    )

    combined_workload = dict(training_cost.get("workload_counters", {}))
    combined_workload["test_candidate_n_states_per_user"] = [int(v) for v in test_states_per_user]
    combined_workload["test_forward_calls_per_user"] = [int(v) for v in per_user_eval_calls]
    combined_workload["test_total_forward_calls"] = int(total_eval_calls)

    return {
        **training_cost,
        "core_wall_time_seconds_total": float(core_wall_time_seconds_training + core_wall_time_seconds_testing),
        "core_wall_time_seconds_testing": float(core_wall_time_seconds_testing),
        "estimated_nn_inference_flops": float(training_cost.get("estimated_nn_inference_flops", 0.0) + testing_inference_flops),
        "estimated_nn_total_flops": float(
            training_cost.get("estimated_nn_training_flops", 0.0)
            + training_cost.get("estimated_nn_inference_flops", 0.0)
            + testing_inference_flops
        ),
        "inference_forward_calls": int(training_cost.get("inference_forward_calls", 0) + total_eval_calls),
        "workload_counters": combined_workload,
        "notes": [
            "Training FLOPs are estimated from rollout-query counts and precoder MLP shapes.",
            "Testing-phase wall time measures only the held-out uplink precoder-net evaluation; plotting, file export, and baseline reconstruction are excluded.",
            "Inference forward calls include both the post-training train-eval pass and the held-out test pass.",
        ],
    }


def build_downlink_monte_carlo_training_cost(
    artifact: Mapping[str, Any],
    *,
    batch_size: int,
    core_wall_time_seconds_training: float,
) -> dict[str, Any]:
    model_specs = list(artifact.get("user_model_specs", []))
    per_user_forward_flops = build_forward_flops_per_user(model_specs, link="downlink")
    training_history = artifact.get("precoder_net_training_history", {})
    post_training_summary = training_history.get("post_training_summary", {})
    cumulative_counts = training_history.get(
        "cumulative_rollout_queries_by_n_kl",
        post_training_summary.get("cumulative_rollout_queries_by_n_kl", {}),
    )
    rollout_counts_per_user = _sum_numeric_sequence_mappings(
        cumulative_counts.get(
            "per_user_active_user_rollout_queries_by_n_kl_over_all_epochs",
            [],
        )
    )
    epoch_rollout_summaries = training_history.get("epoch_rollout_query_summaries", [])
    optimizer_steps = 0
    joint_rollout_queries = 0
    for epoch_summary in epoch_rollout_summaries:
        if not isinstance(epoch_summary, Mapping):
            continue
        total_queries = int(epoch_summary.get("total_rollout_queries", 0))
        joint_rollout_queries += int(total_queries)
        optimizer_steps += int(math.ceil(total_queries / max(int(batch_size), 1)))

    model_scope = require_choice(
        artifact.get("downlink_precoder_net_scope", "per_user_nets"),
        DOWNLINK_NET_SCOPES,
        "downlink_precoder_net_scope",
    )
    if model_scope == "bs_shared_net" and model_specs:
        first_state = next(iter(artifact.get("user_model_states", [])), {})
        parameter_count = int(
            sum(int(value.numel()) for value in first_state.values() if hasattr(value, "numel"))
        )
        shared_forward_flops = max(2 * parameter_count, 1)
        estimated_training_flops = float(
            FORWARD_BACKWARD_FLOP_FACTOR
            * shared_forward_flops
            * joint_rollout_queries
        )
        per_user_forward_flops = [int(shared_forward_flops)]
        training_evaluations = int(joint_rollout_queries)
    else:
        estimated_training_flops = float(
            sum(
                FORWARD_BACKWARD_FLOP_FACTOR * per_user_forward_flops[k] * rollout_counts_per_user[k]
                for k in range(min(len(per_user_forward_flops), len(rollout_counts_per_user)))
            )
        )
        training_evaluations = int(sum(rollout_counts_per_user))

    return {
        "core_wall_time_seconds_total": float(core_wall_time_seconds_training),
        "core_wall_time_seconds_training": float(core_wall_time_seconds_training),
        "core_wall_time_seconds_testing": 0.0,
        "estimated_nn_training_flops": float(estimated_training_flops),
        "estimated_nn_inference_flops": 0.0,
        "estimated_nn_total_flops": float(estimated_training_flops),
        "training_forward_backward_sample_equivalents": int(training_evaluations),
        "inference_forward_calls": 0,
        "optimizer_steps": int(optimizer_steps),
        "per_user_forward_flops": [int(v) for v in per_user_forward_flops],
        "workload_counters": {
            "training_active_user_rollout_queries_per_user": [int(v) for v in rollout_counts_per_user],
            "training_joint_rollout_queries": int(joint_rollout_queries),
            "batch_size": int(batch_size),
        },
        "notes": [
            "Forward+backward NN FLOPs count the active-user rollout-query training passes that optimize the downlink precoder nets.",
            "Forward-only NN FLOPs are 0.0 here because the downlink training phase does not include a separate train-eval pass.",
        ],
    }


def build_downlink_monte_carlo_total_cost(
    artifact: Mapping[str, Any],
    evaluation_cost_counters: Mapping[str, Any] | None,
    *,
    batch_size: int,
    core_wall_time_seconds_training: float,
    core_wall_time_seconds_testing: float,
) -> dict[str, Any]:
    training_cost = build_downlink_monte_carlo_training_cost(
        artifact,
        batch_size=batch_size,
        core_wall_time_seconds_training=core_wall_time_seconds_training,
    )
    per_user_forward_flops = [int(v) for v in training_cost.get("per_user_forward_flops", [])]
    per_user_eval_calls = [
        int(v)
        for v in (evaluation_cost_counters or {}).get(
            "per_user_forward_calls",
            [],
        )
    ]
    model_scope = require_choice(
        artifact.get("downlink_precoder_net_scope", "per_user_nets"),
        DOWNLINK_NET_SCOPES,
        "downlink_precoder_net_scope",
    )
    if model_scope == "bs_shared_net":
        eval_inference_flops = float(
            (per_user_forward_flops[0] if per_user_forward_flops else 0)
            * int((evaluation_cost_counters or {}).get("total_forward_calls", 0))
        )
    else:
        eval_inference_flops = float(
            sum(
                per_user_forward_flops[k] * per_user_eval_calls[k]
                for k in range(min(len(per_user_forward_flops), len(per_user_eval_calls)))
            )
        )

    combined_workload = dict(training_cost.get("workload_counters", {}))
    combined_workload["test_forward_calls_per_user"] = [int(v) for v in per_user_eval_calls]
    combined_workload["test_total_forward_calls"] = int((evaluation_cost_counters or {}).get("total_forward_calls", 0))

    return {
        **training_cost,
        "core_wall_time_seconds_total": float(core_wall_time_seconds_training + core_wall_time_seconds_testing),
        "core_wall_time_seconds_testing": float(core_wall_time_seconds_testing),
        "estimated_nn_inference_flops": float(eval_inference_flops),
        "estimated_nn_total_flops": float(training_cost.get("estimated_nn_training_flops", 0.0) + eval_inference_flops),
        "inference_forward_calls": int((evaluation_cost_counters or {}).get("total_forward_calls", 0)),
        "workload_counters": combined_workload,
        "notes": [
            "Forward+backward NN FLOPs count the active-user rollout-query training passes that optimize the downlink precoder nets.",
            "Testing-phase wall time measures only the held-out downlink precoder-net evaluation; plotting, file export, and baseline reconstruction are excluded.",
            "Forward-only NN FLOPs count the forward-only beam evaluations in the held-out downlink test pass.",
        ],
    }
