"""Downlink Monte Carlo precoder-network training and checkpoint export."""

import copy
from typing import Any, Sequence

import numpy as np
import torch

from latency_optimization.runtime import DEVICE
from latency_optimization.optimization.stopping import KktResiduals, convergence_status_from_config
from latency_optimization.precoders.model_state import (
    clone_model_states,
    relative_models_state_change,
    restore_model_states,
)
from latency_optimization.results.console import format_log_line, format_progress_log_line

from ..objective_settings import (
    objective_display_name,
    objective_weight_strategy_name,
    validate_convergence_objective_mode,
    validate_convergence_priority_weight_strategy,
)
from ..config import validate_shared_bs_streaming_blocklength_input_mode
from ..precoders.models import validate_downlink_precoder_net_scope

from .network_operations import (
    _build_training_user_models,
    _scenario_forward_pass,
    _scenario_objective_weights,
    _unique_trainable_parameters,
)
from .rollout import (
    _generate_rollout_queries_for_downlink,
    _serialize_n_kl_case_counts,
    _summarize_downlink_rollout_queries,
    summarize_training_dataset,
)


def train_blocklength_aware_precoder_net(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    training_episodes: Sequence[dict[str, Any]],
    *,
    epochs: int | None = None,
    batch_size: int = 32,
    lr: float = 1e-3,
    verbose: bool = True,
) -> tuple[list[torch.nn.Module], dict[str, Any], list[int]]:
    """Train reusable downlink beamformers from seeded channel episodes.

    What: build either per-user networks or one shared-BS network, roll the current
    models through each channel episode, and collect the ``n_kl`` states reached by
    payload or streaming allocation. A batch performs one joint optimizer update
    using the configured weighted FBL-rate objective and the shared BS power/rate
    constraints; no expert beam targets or precoder MSE labels are used.

    Why: the base dataset should contain channels, not a manually enumerated grid of
    blocklength labels. Regenerating rollout queries makes training follow the states
    the current policy will actually encounter, while one optimizer preserves the
    coupling among all beams produced for a downlink block.

    Returns: the trained model list, complete training/KKT/dataset diagnostics, and
    the number of rollout queries attributed to each user. The same model objects are
    later saved and reused without weight updates during Monte Carlo testing.
    """
    K = int(system_params["K"])
    model_scope = validate_downlink_precoder_net_scope(sim_params.get("downlink_precoder_net_scope", "per_user_nets"))
    streaming_blocklength_input_mode = validate_shared_bs_streaming_blocklength_input_mode(
        sim_params.get("shared_bs_streaming_blocklength_input_mode", "joint_blocklength_vector")
    )
    objective_mode = validate_convergence_objective_mode(sim_params)
    objective_public_name = objective_display_name(
        objective_mode,
        validate_convergence_priority_weight_strategy(sim_params),
    )
    dataset_summary = summarize_training_dataset(training_episodes)
    models = _build_training_user_models(system_params, sim_params)
    optimizer = torch.optim.Adam(
        _unique_trainable_parameters(models),
        lr=float(lr),
    )
    training_history = {
        "per_user_objective_loss": [[] for _ in range(K)],
        "per_user_rate": [[] for _ in range(K)],
        "sum_rate": [],
        "avg_user_rate": [],
        "avg_rate_violation": [[] for _ in range(K)],
        "avg_block_power_violation": [],
        "avg_objective_loss": [],
        "avg_rate_violation_over_users": [],
        "kkt_primal_residual": [],
        "kkt_complementarity_residual": [],
        "kkt_stationarity_residual": [],
        "training_epoch_status": [],
        "weighted_rate_objective": [],
        "dataset_summary": dataset_summary,
        "epoch_rollout_query_summaries": [],
        "downlink_precoder_net_scope": str(model_scope),
        "objective_mode": str(objective_public_name),
        "weight_strategy": objective_weight_strategy_name(
            objective_mode,
            validate_convergence_priority_weight_strategy(sim_params),
        ),
        "shared_bs_streaming_blocklength_input_mode": (
            str(streaming_blocklength_input_mode)
            if str(model_scope) == "bs_shared_net"
            else "not_applicable"
        ),
        "training_objective": (
            "scenario_driven_rollout_"
            f"{objective_public_name}_finite_blocklength_rate"
        ),
    }
    dataset_sizes = [
        int(dataset_summary.get("training_samples_per_user", [0 for _ in range(K)])[k])
        for k in range(K)
    ]
    max_epochs = max(1, int(epochs if epochs is not None else sim_params["monte_carlo_training_max_epochs"]))
    print_every = max(1, int(sim_params["print_every_epoch"]))
    if verbose:
        print(
            format_log_line(
                "[DL Monte Carlo Train]",
                phase="start",
                scope="joint",
                training_sample_unit=dataset_summary.get("training_sample_unit", "unknown"),
                training_samples=int(len(training_episodes)),
                training_samples_per_user=[int(v) for v in dataset_sizes],
                epochs=int(max_epochs),
                batch_size=int(batch_size),
                precoder_net_scope=str(model_scope),
            )
        )

    if len(training_episodes) == 0:
        return [model.eval() for model in models], training_history, dataset_sizes

    rng = np.random.default_rng(1000)
    cumulative_rollout_query_global_counts: dict[int, int] = {}
    cumulative_rollout_query_per_user_counts: list[dict[int, int]] = [{} for _ in range(K)]
    cumulative_frontier_query_global_counts: dict[int, int] = {}
    cumulative_frontier_query_per_user_counts: list[dict[int, int]] = [{} for _ in range(K)]
    final_epoch_rollout_summary: dict[str, Any] = {}
    previous_epoch_model_states: list[dict[str, torch.Tensor]] | None = None
    best_training_objective = -float("inf")
    best_training_epoch = 0
    best_training_model_states = clone_model_states(models)
    best_training_optimizer_state = copy.deepcopy(optimizer.state_dict())
    solve_status = "max_epochs_reached"
    epochs_completed = 0

    for epoch in range(int(max_epochs)):
        epochs_completed = int(epoch + 1)
        for model in models:
            model.eval()
        rollout_queries = _generate_rollout_queries_for_downlink(
            system_params,
            sim_params,
            training_episodes,
            models,
        )
        final_epoch_rollout_summary = _summarize_downlink_rollout_queries(rollout_queries)
        final_epoch_rollout_summary["epoch"] = int(epoch + 1)
        training_history["epoch_rollout_query_summaries"].append(final_epoch_rollout_summary)

        for n_key, count in final_epoch_rollout_summary.get("global_active_user_rollout_queries_by_n_kl", {}).items():
            n_val = int(n_key)
            cumulative_rollout_query_global_counts[n_val] = (
                cumulative_rollout_query_global_counts.get(n_val, 0) + int(count)
            )
        for k, user_counts in enumerate(
            final_epoch_rollout_summary.get("per_user_active_user_rollout_queries_by_n_kl", [])
        ):
            for n_key, count in user_counts.items():
                n_val = int(n_key)
                cumulative_rollout_query_per_user_counts[k][n_val] = (
                    cumulative_rollout_query_per_user_counts[k].get(n_val, 0) + int(count)
                )
        for n_key, count in final_epoch_rollout_summary.get(
            "global_active_user_frontier_rollout_queries_by_n_kl",
            {},
        ).items():
            n_val = int(n_key)
            cumulative_frontier_query_global_counts[n_val] = (
                cumulative_frontier_query_global_counts.get(n_val, 0) + int(count)
            )
        for k, user_counts in enumerate(
            final_epoch_rollout_summary.get("per_user_active_user_frontier_rollout_queries_by_n_kl", [])
        ):
            for n_key, count in user_counts.items():
                n_val = int(n_key)
                cumulative_frontier_query_per_user_counts[k][n_val] = (
                    cumulative_frontier_query_per_user_counts[k].get(n_val, 0) + int(count)
                )

        for model in models:
            model.train()
        indices = np.arange(len(rollout_queries))
        rng.shuffle(indices)
        epoch_term_sums = np.zeros(K, dtype=float)
        epoch_term_counts = np.zeros(K, dtype=float)
        epoch_rate_sums = np.zeros(K, dtype=float)
        epoch_sum_rate_sums = 0.0
        epoch_sum_rate_weight = 0.0
        epoch_rate_violation_sums = np.zeros(K, dtype=float)
        epoch_block_power_violation_sum = 0.0
        epoch_weighted_rate_sum = 0.0
        epoch_active_weight_sum = 0.0

        if len(rollout_queries) == 0:
            for k in range(K):
                training_history["per_user_objective_loss"][k].append(0.0)
                training_history["per_user_rate"][k].append(0.0)
                training_history["avg_rate_violation"][k].append(0.0)
            training_history["sum_rate"].append(0.0)
            training_history["avg_user_rate"].append(0.0)
            training_history["avg_objective_loss"].append(0.0)
            training_history["avg_rate_violation_over_users"].append(0.0)
            training_history["avg_block_power_violation"].append(0.0)
            training_history["kkt_primal_residual"].append(0.0)
            training_history["kkt_complementarity_residual"].append(0.0)
            training_history["kkt_stationarity_residual"].append(0.0)
            training_history["training_epoch_status"].append("no_rollout_queries")
            training_history["weighted_rate_objective"].append(0.0)
            if verbose:
                print(
                    format_progress_log_line(
                        "[DL Monte Carlo]",
                        phase="train",
                        method="monte_carlo",
                        scope="joint",
                        epoch=f"{epoch + 1}/{int(max_epochs)}",
                        objective=0.0,
                        sum_rate=0.0,
                        avg_user_rate=0.0,
                        r_p=0.0,
                        r_c=0.0,
                        r_s=0.0,
                        status="no_rollout_queries",
                    )
                )
            continue

        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = indices[start : start + max(int(batch_size), 1)]
            optimizer.zero_grad()

            loss = torch.zeros((), dtype=torch.float32, device=DEVICE)
            total_active_weight = 0.0

            for idx in batch_idx:
                scenario = rollout_queries[int(idx)]
                scenario_user_weights, _ = _scenario_objective_weights(
                    scenario,
                    sim_params,
                    objective_mode=objective_mode,
                )
                forward = _scenario_forward_pass(
                    system_params,
                    scenario,
                    models,
                    scenario["n_targets"],
                )
                active_mask = forward["active_mask"]
                scenario_term = torch.zeros((), dtype=torch.float32, device=DEVICE)
                scenario_has_active_user = False
                for k in range(K):
                    rate = forward["rates"][k]
                    power = forward["powers"][k]
                    if rate is None or power is None:
                        continue

                    required_rate = float(forward["required_rates"][k])
                    rate_violation = rate.new_tensor(required_rate) - rate
                    rate_violation_pos = torch.relu(rate_violation)
                    term = -(float(scenario_user_weights[int(k)]) * rate)
                    scenario_term = scenario_term + term
                    rate_value = float(rate.detach().cpu())
                    total_active_weight += 1.0
                    epoch_term_sums[k] += float(term.detach().cpu())
                    epoch_term_counts[k] += 1.0
                    epoch_rate_sums[k] += rate_value
                    epoch_rate_violation_sums[k] += float(rate_violation_pos.detach().cpu())
                    epoch_weighted_rate_sum += float(scenario_user_weights[int(k)]) * rate_value
                    epoch_active_weight_sum += 1.0
                    scenario_has_active_user = True
                block_power_violation = forward["block_power_violation"]
                loss = loss + scenario_term
                if scenario_has_active_user:
                    epoch_block_power_violation_sum += float(block_power_violation.detach().cpu())
                if float(np.sum(active_mask)) > 0.0:
                    epoch_sum_rate_sums += float(forward["sum_rate"].detach().cpu())
                    epoch_sum_rate_weight += 1.0

            if total_active_weight <= 0.0:
                continue

            loss = loss / float(total_active_weight)
            loss.backward()
            for model in models:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        for model in models:
            model.eval()
        epoch_objective_losses = []
        epoch_rates = []
        epoch_rate_violations = []
        avg_block_power_violation = float(epoch_block_power_violation_sum / max(epoch_sum_rate_weight, 1.0))
        for k in range(K):
            avg_term = float(epoch_term_sums[k] / max(epoch_term_counts[k], 1.0))
            avg_rate = float(epoch_rate_sums[k] / max(epoch_term_counts[k], 1.0))
            avg_rate_violation = float(epoch_rate_violation_sums[k] / max(epoch_term_counts[k], 1.0))
            training_history["per_user_objective_loss"][k].append(avg_term)
            training_history["per_user_rate"][k].append(avg_rate)
            training_history["avg_rate_violation"][k].append(avg_rate_violation)
            epoch_objective_losses.append(avg_term)
            epoch_rates.append(avg_rate)
            epoch_rate_violations.append(avg_rate_violation)
        avg_sum_rate = float(epoch_sum_rate_sums / max(epoch_sum_rate_weight, 1.0))
        training_history["sum_rate"].append(avg_sum_rate)
        training_history["avg_user_rate"].append(float(np.mean(epoch_rates)) if epoch_rates else 0.0)
        training_history["avg_objective_loss"].append(
            float(np.mean(epoch_objective_losses)) if epoch_objective_losses else 0.0
        )
        training_history["avg_rate_violation_over_users"].append(
            float(np.mean(epoch_rate_violations)) if epoch_rate_violations else 0.0
        )
        training_history["avg_block_power_violation"].append(avg_block_power_violation)
        epoch_weighted_rate_objective = float(
            epoch_weighted_rate_sum / max(epoch_active_weight_sum, 1.0)
        )
        training_history["weighted_rate_objective"].append(epoch_weighted_rate_objective)
        epoch_r_p = float(max(max(epoch_rate_violations, default=0.0), max(avg_block_power_violation, 0.0)))
        epoch_r_c = 0.0
        epoch_r_s = relative_models_state_change(models, previous_epoch_model_states)
        previous_epoch_model_states = clone_model_states(models)
        if epoch_weighted_rate_objective >= best_training_objective:
            best_training_objective = float(epoch_weighted_rate_objective)
            best_training_epoch = int(epoch + 1)
            best_training_model_states = clone_model_states(models)
            best_training_optimizer_state = copy.deepcopy(optimizer.state_dict())

        epoch_status = convergence_status_from_config(
            sim_params,
            precoder_change=epoch_r_s,
            has_previous_state=epoch > 0,
            residuals=KktResiduals(epoch_r_p, epoch_r_c, epoch_r_s),
        )
        if epoch_status != "running":
            solve_status = epoch_status
        training_history["kkt_primal_residual"].append(float(epoch_r_p))
        training_history["kkt_complementarity_residual"].append(float(epoch_r_c))
        training_history["kkt_stationarity_residual"].append(float(epoch_r_s))
        training_history["training_epoch_status"].append(str(epoch_status))
        if verbose:
            if (
                ((epoch + 1) % print_every) == 0
                or epoch == 0
                or epoch_status != "running"
            ):
                print(
                    format_progress_log_line(
                        "[DL Monte Carlo]",
                        phase="train",
                        method="monte_carlo",
                        scope="joint",
                        epoch=f"{epoch + 1}/{int(max_epochs)}",
                        objective=float(training_history["avg_objective_loss"][-1]),
                        sum_rate=float(avg_sum_rate),
                        avg_user_rate=float(training_history["avg_user_rate"][-1]),
                        r_p=epoch_r_p,
                        r_c=epoch_r_c,
                        r_s=epoch_r_s,
                        status=epoch_status,
                    )
                )
        if epoch_status != "running":
            break

    restore_model_states(models, best_training_model_states)
    optimizer.load_state_dict(best_training_optimizer_state)
    restored_solution_source = "highest_training_weighted_rate"
    if solve_status == "max_epochs_reached":
        solve_status = "max_epochs_best_training_objective"
    if training_history["training_epoch_status"]:
        training_history["training_epoch_status"][-1] = str(solve_status)
    training_history["configured_max_epochs"] = int(max_epochs)
    training_history["epochs_completed"] = int(epochs_completed)
    training_history["training_solve_status"] = str(solve_status)
    training_history["restored_solution_source"] = str(restored_solution_source)

    training_history["post_training_summary"] = {
        "epochs_requested": int(max_epochs),
        "configured_max_epochs": int(max_epochs),
        "epochs_completed": int(epochs_completed),
        "training_solve_status": str(solve_status),
        "restored_solution_source": str(restored_solution_source),
        "selected_checkpoint_epoch": int(best_training_epoch),
        "selected_checkpoint_weighted_rate_objective": float(best_training_objective),
        "downlink_precoder_net_scope": str(model_scope),
        "objective_mode": str(training_history.get("objective_mode", objective_public_name)),
        "weight_strategy": str(training_history.get("weight_strategy", "none")),
        "shared_bs_streaming_blocklength_input_mode": (
            str(streaming_blocklength_input_mode)
            if str(model_scope) == "bs_shared_net"
            else "not_applicable"
        ),
        "base_dataset_kind": dataset_summary.get("base_dataset_kind", "unknown"),
        "total_training_samples": int(dataset_summary.get("total_training_samples", 0)),
        "training_sample_kind": dataset_summary.get("training_sample_kind", "unknown"),
        "training_sample_unit": dataset_summary.get("training_sample_unit", "unknown"),
        "rollout_anchor_bits_mode": "derived_online_from_current_episode_served_bits",
        "per_user_final_objective_loss": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in training_history["per_user_objective_loss"]
        ],
        "per_user_best_objective_loss": [
            float(min(history)) if len(history) > 0 else 0.0 for history in training_history["per_user_objective_loss"]
        ],
        "per_user_final_rate": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in training_history["per_user_rate"]
        ],
        "per_user_last_epoch_avg_objective_loss_over_active_rollout_queries": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in training_history["per_user_objective_loss"]
        ],
        "per_user_last_epoch_avg_rate_over_active_rollout_queries": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in training_history["per_user_rate"]
        ],
        "per_user_final_rate_violation": [
            float(history[-1]) if len(history) > 0 else 0.0 for history in training_history["avg_rate_violation"]
        ],
        "final_avg_sum_rate": float(training_history["sum_rate"][-1]) if training_history["sum_rate"] else 0.0,
        "best_avg_sum_rate": float(max(training_history["sum_rate"])) if training_history["sum_rate"] else 0.0,
        "last_epoch_rollout_weighted_avg_sum_rate": (
            float(training_history["sum_rate"][-1]) if training_history["sum_rate"] else 0.0
        ),
        "best_epoch_rollout_weighted_avg_sum_rate": (
            float(max(training_history["sum_rate"])) if training_history["sum_rate"] else 0.0
        ),
        "last_epoch_weighted_rate_objective": (
            float(training_history["weighted_rate_objective"][-1])
            if training_history["weighted_rate_objective"]
            else 0.0
        ),
        "best_epoch_weighted_rate_objective": (
            float(max(training_history["weighted_rate_objective"]))
            if training_history["weighted_rate_objective"]
            else 0.0
        ),
        "final_avg_user_rate": (
            float(training_history["avg_user_rate"][-1]) if training_history["avg_user_rate"] else 0.0
        ),
        "best_avg_user_rate": float(max(training_history["avg_user_rate"])) if training_history["avg_user_rate"] else 0.0,
        "last_epoch_mean_user_rollout_rate": (
            float(training_history["avg_user_rate"][-1]) if training_history["avg_user_rate"] else 0.0
        ),
        "best_epoch_mean_user_rollout_rate": (
            float(max(training_history["avg_user_rate"])) if training_history["avg_user_rate"] else 0.0
        ),
        "final_avg_objective_loss": (
            float(training_history["avg_objective_loss"][-1]) if training_history["avg_objective_loss"] else 0.0
        ),
        "best_avg_objective_loss": float(min(training_history["avg_objective_loss"])) if training_history["avg_objective_loss"] else 0.0,
        "last_epoch_mean_user_objective_loss": (
            float(training_history["avg_objective_loss"][-1]) if training_history["avg_objective_loss"] else 0.0
        ),
        "best_epoch_mean_user_objective_loss": (
            float(min(training_history["avg_objective_loss"])) if training_history["avg_objective_loss"] else 0.0
        ),
        "final_avg_rate_violation": (
            float(training_history["avg_rate_violation_over_users"][-1])
            if training_history["avg_rate_violation_over_users"]
            else 0.0
        ),
        "best_avg_rate_violation": (
            float(min(training_history["avg_rate_violation_over_users"]))
            if training_history["avg_rate_violation_over_users"]
            else 0.0
        ),
        "final_avg_block_power_violation": (
            float(training_history["avg_block_power_violation"][-1])
            if training_history["avg_block_power_violation"]
            else 0.0
        ),
        "best_avg_block_power_violation": (
            float(min(training_history["avg_block_power_violation"]))
            if training_history["avg_block_power_violation"]
            else 0.0
        ),
        "final_kkt_primal_residual": (
            float(training_history["kkt_primal_residual"][-1]) if training_history["kkt_primal_residual"] else 0.0
        ),
        "best_kkt_primal_residual": (
            float(min(training_history["kkt_primal_residual"])) if training_history["kkt_primal_residual"] else 0.0
        ),
        "final_kkt_complementarity_residual": (
            float(training_history["kkt_complementarity_residual"][-1])
            if training_history["kkt_complementarity_residual"]
            else 0.0
        ),
        "best_kkt_complementarity_residual": (
            float(min(training_history["kkt_complementarity_residual"]))
            if training_history["kkt_complementarity_residual"]
            else 0.0
        ),
        "final_kkt_stationarity_residual": (
            float(training_history["kkt_stationarity_residual"][-1])
            if training_history["kkt_stationarity_residual"]
            else 0.0
        ),
        "best_kkt_stationarity_residual": (
            float(min(training_history["kkt_stationarity_residual"]))
            if training_history["kkt_stationarity_residual"]
            else 0.0
        ),
        "final_feasible_rollout_query_fraction": (
            float(final_epoch_rollout_summary.get("feasible_rollout_queries", 0))
            / float(max(int(final_epoch_rollout_summary.get("total_rollout_queries", 0)), 1))
        ),
        "last_epoch_total_rollout_queries": int(final_epoch_rollout_summary.get("total_rollout_queries", 0)),
        "last_epoch_feasible_rollout_queries": int(final_epoch_rollout_summary.get("feasible_rollout_queries", 0)),
        "last_epoch_infeasible_rollout_queries": int(final_epoch_rollout_summary.get("infeasible_rollout_queries", 0)),
        "last_epoch_frontier_rollout_queries": int(final_epoch_rollout_summary.get("frontier_rollout_queries", 0)),
        "cumulative_rollout_queries_by_n_kl": _serialize_n_kl_case_counts(
            cumulative_rollout_query_global_counts,
            cumulative_rollout_query_per_user_counts,
            global_key="global_active_user_rollout_queries_by_n_kl_over_all_epochs",
            per_user_key="per_user_active_user_rollout_queries_by_n_kl_over_all_epochs",
        ),
        "cumulative_frontier_rollout_queries_by_n_kl": _serialize_n_kl_case_counts(
            cumulative_frontier_query_global_counts,
            cumulative_frontier_query_per_user_counts,
            global_key="global_active_user_frontier_rollout_queries_by_n_kl_over_all_epochs",
            per_user_key="per_user_active_user_frontier_rollout_queries_by_n_kl_over_all_epochs",
        ),
        "final_epoch_rollout_query_summary": final_epoch_rollout_summary,
    }

    return [model.eval() for model in models], training_history, dataset_sizes


__all__ = [
    "train_blocklength_aware_precoder_net",
]
