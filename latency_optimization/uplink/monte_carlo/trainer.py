"""Uplink Monte Carlo precoder-network training."""

import copy
from typing import Any, Sequence

import numpy as np
import torch

from latency_optimization.runtime import DEVICE
from latency_optimization.experiments.channels import with_monte_carlo_sample_snr_by_user
from latency_optimization.optimization.stopping import KktResiduals, convergence_status_from_config
from latency_optimization.precoders.model_state import clone_model_state, relative_model_state_change
from latency_optimization.results.console import format_log_line, format_progress_log_line
from latency_optimization.core.scenarios import STREAMING_MODE

from ..config import load_config
from ..objective_settings import (
    RATE_BEAM_REWARD_MODE,
    UNWEIGHTED_SUM_RATE_OBJECTIVE,
    validate_uplink_objective_mode,
)
from ..precoders.checkpoints import (
    export_user_model_specs,
    export_user_model_states,
)
from ..precoders.inference import infer_precoder
from ..precoders.models import build_user_precoder_net
from ..simulation import (
    estimate_initial_random_precoder_payload_schedule,
    estimate_initial_random_precoder_streaming_schedule,
)
from ..system import UplinkSystem

from .network_operations import (
    ROLLOUT_QUERY_OBJECTIVE_TRAINING_STYLE,
    _compute_r_fbl_torch,
    _uplink_training_beam_reward_torch,
)
from .evaluator import evaluate_blocklength_precoder_net
from .rollout import (
    _build_post_training_summary,
    _generate_rollout_queries_for_training_episodes,
    _serialize_count_dict,
    _summarize_rollout_queries_by_user,
    build_training_dataset,
    summarize_training_dataset,
)


def train_blocklength_aware_precoder_net(
    cfg_name: str,
    train_seeds: Sequence[int],
    *,
    epochs: int | None = None,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> dict:
    """Train one reusable, blocklength-aware uplink beamformer per user.

    What: create channel-only episodes from the requested seeds, run the current
    user networks through payload or streaming allocation, and train on the visited
    ``(H_kl, n_kl, P_k, sigma_k^2, epsilon_k)`` states. Each user's optimizer updates
    only that user's network by differentiating the configured FBL-rate Lagrangian;
    there are no expert precoders and no beam-entry MSE loss.

    Why: uplink users do not share a transmitter or precoder, so their trainable
    parameters remain decoupled even though system evaluation may include receiver
    interference. Rollout regeneration supplies the n values and later channel
    blocks that the trained network will encounter instead of hand-building them.

    Returns: the trained per-user networks together with dataset, rollout, objective,
    KKT, and stopping summaries used by checkpoint saving and result reporting.
    """
    system_params, sim_cfg, _ = load_config(cfg_name)
    K = int(system_params["K"])
    objective_mode = validate_uplink_objective_mode(
        sim_cfg.get("uplink_objective_mode", UNWEIGHTED_SUM_RATE_OBJECTIVE)
    )
    max_epochs = max(
        1,
        int(epochs if epochs is not None else sim_cfg.get("monte_carlo_training_max_epochs", sim_cfg.get("max_epochs", 20))),
    )
    print_every_epoch = max(1, int(sim_cfg.get("print_every_epoch", 1)))
    training_episodes = build_training_dataset(cfg_name, train_seeds)
    dataset_summary = summarize_training_dataset(training_episodes)
    beam_reward_mode = RATE_BEAM_REWARD_MODE
    training_history = {
        "per_user_objective_loss": [[] for _ in range(K)],
        "per_user_rate": [[] for _ in range(K)],
        "avg_rate_violation": [[] for _ in range(K)],
        "avg_power_violation": [[] for _ in range(K)],
        "per_user_kkt_primal_residual": [[] for _ in range(K)],
        "per_user_kkt_complementarity_residual": [[] for _ in range(K)],
        "per_user_kkt_stationarity_residual": [[] for _ in range(K)],
        "per_user_training_epoch_status": [[] for _ in range(K)],
        "avg_objective_loss": [],
        "avg_user_rate": [],
        "avg_rate_violation_over_users": [],
        "avg_power_violation_over_users": [],
        "dataset_summary": dataset_summary,
        "rollout_query_summaries_per_user": [[] for _ in range(K)],
        "monte_carlo_training_style": ROLLOUT_QUERY_OBJECTIVE_TRAINING_STYLE,
        "uplink_objective_mode": str(objective_mode),
        "beam_reward_mode": str(beam_reward_mode),
        "configured_max_epochs": int(max_epochs),
        "training_objective": (
            "scenario_driven_episode_rollout_objective_"
            f"{objective_mode}_user_{beam_reward_mode}"
        ),
    }
    user_models = [
        build_user_precoder_net(
            receive_antennas=int(system_params["NR"][k]),
            transmit_antennas=int(system_params["NT"][k]),
            streams=int(system_params["dk"][k]),
            device=DEVICE,
        )
        for k in range(K)
    ]
    optimizers = [
        torch.optim.Adam(user_models[k].parameters(), lr=float(lr))
        for k in range(K)
    ]
    rngs = [
        np.random.default_rng(int(train_seeds[0]) + 17 * (k + 1) if len(train_seeds) > 0 else (17 * (k + 1)))
        for k in range(K)
    ]
    cumulative_rollout_query_global_counts: dict[int, int] = {}
    cumulative_rollout_query_per_user_counts: list[dict[int, int]] = [{} for _ in range(K)]
    cumulative_frontier_query_global_counts: dict[int, int] = {}
    cumulative_frontier_query_per_user_counts: list[dict[int, int]] = [{} for _ in range(K)]
    last_epoch_queries_by_user: list[list[dict[str, Any]]] = [[] for _ in range(K)]
    previous_epoch_model_states: list[dict[str, torch.Tensor] | None] = [None for _ in range(K)]
    best_training_rate = [-float("inf") for _ in range(K)]
    best_model_states = [clone_model_state(model) for model in user_models]
    best_optimizer_states = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
    per_user_solve_status = ["max_epochs_reached" for _ in range(K)]
    epochs_completed = 0

    print(
        format_log_line(
            "[UL Monte Carlo Train]",
            phase="start",
            training_sample_unit=dataset_summary.get("training_sample_unit", "unknown"),
            training_samples=int(len(training_episodes)),
            epochs=int(max_epochs),
            batch_size=int(batch_size),
        )
    )

    for epoch in range(int(max_epochs)):
        epochs_completed = int(epoch + 1)
        rollout_queries_by_user = _generate_rollout_queries_for_training_episodes(
            system_params,
            sim_cfg,
            training_episodes,
            user_models,
        )
        last_epoch_queries_by_user = [
            [dict(query) for query in user_queries]
            for user_queries in rollout_queries_by_user
        ]
        rollout_summary = _summarize_rollout_queries_by_user(rollout_queries_by_user)

        for n_key, count in rollout_summary.get("global_rollout_queries_by_n_kl", {}).items():
            n_val = int(n_key)
            cumulative_rollout_query_global_counts[n_val] = (
                cumulative_rollout_query_global_counts.get(n_val, 0) + int(count)
            )
        for user_idx, user_summary in enumerate(rollout_summary.get("per_user", [])):
            epoch_rollout_summary = dict(user_summary)
            epoch_rollout_summary["epoch"] = int(epoch + 1)
            training_history["rollout_query_summaries_per_user"][int(user_idx)].append(epoch_rollout_summary)
            for n_key, count in epoch_rollout_summary.get("rollout_queries_by_n_kl", {}).items():
                n_val = int(n_key)
                cumulative_rollout_query_per_user_counts[int(user_idx)][n_val] = (
                    cumulative_rollout_query_per_user_counts[int(user_idx)].get(n_val, 0) + int(count)
                )
            for n_key, count in epoch_rollout_summary.get("frontier_rollout_queries_by_n_kl", {}).items():
                n_val = int(n_key)
                cumulative_frontier_query_per_user_counts[int(user_idx)][n_val] = (
                    cumulative_frontier_query_per_user_counts[int(user_idx)].get(n_val, 0) + int(count)
                )
        for n_key, count in rollout_summary.get("global_frontier_rollout_queries_by_n_kl", {}).items():
            n_val = int(n_key)
            cumulative_frontier_query_global_counts[n_val] = (
                cumulative_frontier_query_global_counts.get(n_val, 0) + int(count)
            )

        epoch_objective_losses: list[float] = []
        epoch_rates: list[float] = []
        epoch_rate_violations: list[float] = []
        epoch_power_violations: list[float] = []
        epoch_statuses: list[str] = []

        for k in range(K):
            model = user_models[int(k)]
            optimizer = optimizers[int(k)]
            rollout_queries = rollout_queries_by_user[int(k)]
            Nt = int(system_params["NT"][k])
            dk = int(system_params["dk"][k])
            if len(rollout_queries) == 0:
                training_history["per_user_objective_loss"][k].append(0.0)
                training_history["per_user_rate"][k].append(0.0)
                training_history["avg_rate_violation"][k].append(0.0)
                training_history["avg_power_violation"][k].append(0.0)
                training_history["per_user_kkt_primal_residual"][k].append(0.0)
                training_history["per_user_kkt_complementarity_residual"][k].append(0.0)
                training_history["per_user_kkt_stationarity_residual"][k].append(0.0)
                training_history["per_user_training_epoch_status"][k].append("no_rollout_queries")
                epoch_objective_losses.append(0.0)
                epoch_rates.append(0.0)
                epoch_rate_violations.append(0.0)
                epoch_power_violations.append(0.0)
                epoch_statuses.append("no_rollout_queries")
                continue

            model.train()
            indices = np.arange(len(rollout_queries))
            rngs[int(k)].shuffle(indices)
            epoch_term_sum = 0.0
            epoch_rate_sum = 0.0
            epoch_rate_violation_sum = 0.0
            epoch_power_violation_sum = 0.0
            epoch_query_count = 0

            for start in range(0, len(indices), max(int(batch_size), 1)):
                batch_idx = indices[start : start + max(int(batch_size), 1)]
                optimizer.zero_grad()
                loss = torch.zeros((), dtype=torch.float32, device=DEVICE)
                batch_query_count = 0

                for idx in batch_idx:
                    query = rollout_queries[int(idx)]
                    H_t = torch.as_tensor(query["H"], dtype=torch.complex64, device=DEVICE)
                    noise_cov_t = (
                        None
                        if query.get("noise_plus_interference_cov") is None
                        else torch.as_tensor(
                            query["noise_plus_interference_cov"],
                            dtype=torch.complex64,
                            device=DEVICE,
                        )
                    )
                    pred_t = infer_precoder(
                        model,
                        H_t,
                        int(query["n_kl"]),
                        float(query["sigma2"]),
                        float(query["epsilon"]),
                        transmit_antennas=Nt,
                        streams=dk,
                        power_limit=query["P"],
                    )
                    rate = _compute_r_fbl_torch(
                        H_t,
                        pred_t,
                        sigma2=float(query["sigma2"]),
                        epsilon=float(query["epsilon"]),
                        n_kl=int(query["n_kl"]),
                        noise_plus_interference_cov=noise_cov_t,
                    )
                    power = (torch.linalg.norm(pred_t, ord="fro") ** 2).real
                    required_rate = float(query.get("required_rate", 0.0))
                    rate_violation = rate.new_tensor(required_rate) - rate
                    power_violation = power - float(query["P"])
                    rate_violation_pos = torch.relu(rate_violation)
                    power_violation_pos = torch.relu(power_violation)
                    reward = _uplink_training_beam_reward_torch(rate)
                    term = -reward
                    loss = loss + term
                    batch_query_count += 1
                    epoch_term_sum += float(term.detach().cpu())
                    epoch_rate_sum += float(rate.detach().cpu())
                    epoch_rate_violation_sum += float(rate_violation_pos.detach().cpu())
                    epoch_power_violation_sum += float(power_violation_pos.detach().cpu())
                    epoch_query_count += 1

                if batch_query_count <= 0:
                    continue
                loss = loss / float(batch_query_count)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            avg_objective_loss = float(epoch_term_sum / max(epoch_query_count, 1))
            avg_rate = float(epoch_rate_sum / max(epoch_query_count, 1))
            avg_rate_violation = float(epoch_rate_violation_sum / max(epoch_query_count, 1))
            avg_power_violation = float(epoch_power_violation_sum / max(epoch_query_count, 1))
            epoch_r_p = float(max(max(avg_rate_violation, 0.0), max(avg_power_violation, 0.0)))
            epoch_r_c = 0.0
            epoch_r_s = relative_model_state_change(model, previous_epoch_model_states[int(k)])
            previous_epoch_model_states[int(k)] = clone_model_state(model)
            if avg_rate >= best_training_rate[int(k)]:
                best_training_rate[int(k)] = float(avg_rate)
                best_model_states[int(k)] = clone_model_state(model)
                best_optimizer_states[int(k)] = copy.deepcopy(optimizer.state_dict())

            epoch_status = convergence_status_from_config(
                sim_cfg,
                precoder_change=epoch_r_s,
                has_previous_state=epoch > 0,
                residuals=KktResiduals(epoch_r_p, epoch_r_c, epoch_r_s),
            )
            if epoch_status != "running":
                per_user_solve_status[int(k)] = epoch_status

            training_history["per_user_objective_loss"][k].append(avg_objective_loss)
            training_history["per_user_rate"][k].append(avg_rate)
            training_history["avg_rate_violation"][k].append(avg_rate_violation)
            training_history["avg_power_violation"][k].append(avg_power_violation)
            training_history["per_user_kkt_primal_residual"][k].append(float(epoch_r_p))
            training_history["per_user_kkt_complementarity_residual"][k].append(float(epoch_r_c))
            training_history["per_user_kkt_stationarity_residual"][k].append(float(epoch_r_s))
            training_history["per_user_training_epoch_status"][k].append(str(epoch_status))
            epoch_objective_losses.append(avg_objective_loss)
            epoch_rates.append(avg_rate)
            epoch_rate_violations.append(avg_rate_violation)
            epoch_power_violations.append(avg_power_violation)
            epoch_statuses.append(epoch_status)

            if (
                ((epoch + 1) % print_every_epoch) == 0
                or epoch == 0
                or epoch_status != "running"
            ):
                print(
                    format_progress_log_line(
                        "[UL Monte Carlo]",
                        phase="train",
                        method="monte_carlo",
                        user=int(k),
                        epoch=f"{epoch + 1}/{int(max_epochs)}",
                        objective=avg_objective_loss,
                        rate=avg_rate,
                        r_p=epoch_r_p,
                        r_c=epoch_r_c,
                        r_s=epoch_r_s,
                        status=epoch_status,
                    )
                )
            model.eval()

        training_history["avg_objective_loss"].append(
            float(np.mean(epoch_objective_losses)) if epoch_objective_losses else 0.0
        )
        training_history["avg_user_rate"].append(float(np.mean(epoch_rates)) if epoch_rates else 0.0)
        training_history["avg_rate_violation_over_users"].append(
            float(np.mean(epoch_rate_violations)) if epoch_rate_violations else 0.0
        )
        training_history["avg_power_violation_over_users"].append(
            float(np.mean(epoch_power_violations)) if epoch_power_violations else 0.0
        )

        terminal_statuses = {
            "objective_stationary",
            "kkt_converged",
            "stationary_infeasible",
            "no_rollout_queries",
        }
        if epoch_statuses and all(status in terminal_statuses for status in epoch_statuses):
            break

    training_history.setdefault("per_user_epochs_completed", [0 for _ in range(K)])
    training_history.setdefault("per_user_training_solve_status", ["not_started" for _ in range(K)])
    training_history.setdefault("per_user_restored_solution_source", ["not_started" for _ in range(K)])

    for k in range(K):
        user_models[int(k)].load_state_dict(best_model_states[int(k)])
        optimizers[int(k)].load_state_dict(best_optimizer_states[int(k)])
        restored_solution_source = "highest_training_rate"
        if per_user_solve_status[int(k)] == "max_epochs_reached":
            per_user_solve_status[int(k)] = "max_epochs_best_training_rate"
        if training_history["per_user_training_epoch_status"][int(k)]:
            training_history["per_user_training_epoch_status"][int(k)][-1] = str(per_user_solve_status[int(k)])
        training_history["per_user_epochs_completed"][int(k)] = int(epochs_completed)
        training_history["per_user_training_solve_status"][int(k)] = str(per_user_solve_status[int(k)])
        training_history["per_user_restored_solution_source"][int(k)] = str(restored_solution_source)
        user_models[int(k)].eval()

    training_history["cumulative_rollout_queries_by_n_kl"] = {
        "global_rollout_queries_by_n_kl_over_all_epochs": _serialize_count_dict(cumulative_rollout_query_global_counts),
        "per_user_rollout_queries_by_n_kl_over_all_epochs": [
            _serialize_count_dict(user_counts) for user_counts in cumulative_rollout_query_per_user_counts
        ],
    }
    training_history["cumulative_frontier_rollout_queries_by_n_kl"] = {
        "global_frontier_rollout_queries_by_n_kl_over_all_epochs": _serialize_count_dict(
            cumulative_frontier_query_global_counts
        ),
        "per_user_frontier_rollout_queries_by_n_kl_over_all_epochs": [
            _serialize_count_dict(user_counts) for user_counts in cumulative_frontier_query_per_user_counts
        ],
    }
    training_history["final_epoch_rollout_query_summary"] = _summarize_rollout_queries_by_user(last_epoch_queries_by_user)

    train_eval_seed = int(train_seeds[0]) if len(train_seeds) > 0 else 0
    train_eval_snr_db_by_user = (
        [float(value) for value in training_episodes[0]["snr_db_by_user"]]
        if len(training_episodes) > 0
        else [float(value) for value in system_params["snr_db"]]
    )
    train_eval_system_params = with_monte_carlo_sample_snr_by_user(
        system_params,
        train_eval_snr_db_by_user,
    )
    train_eval_sim_cfg = copy.deepcopy(sim_cfg)
    if str(train_eval_sim_cfg["experiment_scenario_mode"]) == "streaming":
        train_eval_sim_cfg["experiment_scenario"]["number_of_blocks"] = 1
    baseline_builder = (
        estimate_initial_random_precoder_streaming_schedule
        if str(sim_cfg["experiment_scenario_mode"]) == STREAMING_MODE
        else estimate_initial_random_precoder_payload_schedule
    )
    train_eval_initial_baseline = baseline_builder(
        train_eval_system_params,
        train_eval_sim_cfg,
        seed=train_eval_seed,
    )
    train_eval_system = UplinkSystem(train_eval_system_params, seed=train_eval_seed)
    train_eval_post = evaluate_blocklength_precoder_net(
        uplinksystem=train_eval_system,
        user_models=user_models,
        sim_cfg=train_eval_sim_cfg,
        method_name="monte_carlo_precoder_net_train_eval",
    )
    post_training_summary = _build_post_training_summary(
        train_eval_system,
        train_eval_post,
        training_history,
        train_eval_seed=train_eval_seed,
        train_eval_snr_db_by_user=train_eval_snr_db_by_user,
        epochs=int(max_epochs),
        dataset_summary=dataset_summary,
        initial_baseline=train_eval_initial_baseline,
    )

    train_eval_post.update(
        {
            "train_seeds": [int(s) for s in train_seeds],
            "training_dataset_sizes": [int(len(training_episodes)) for _ in range(K)],
            "training_sample_counts_per_user": [int(len(training_episodes)) for _ in range(K)],
            "training_dataset_summary": dataset_summary,
            "post_training_summary": post_training_summary,
            "precoder_net_training_losses": [
                list(map(float, history)) for history in training_history["per_user_objective_loss"]
            ],
            "precoder_net_training_history": training_history,
            "user_model_specs": export_user_model_specs(
                system_params["NR"],
                system_params["NT"],
                system_params["dk"],
            ),
            "user_model_states": export_user_model_states(user_models),
            "precoder_parameterization": "shared_user_channel_n_sigma_epsilon_to_precoder_mlp",
            "training_objective": training_history["training_objective"],
            "monte_carlo_training_style": str(
                training_history.get(
                    "monte_carlo_training_style",
                    ROLLOUT_QUERY_OBJECTIVE_TRAINING_STYLE,
                )
            ),
            "uplink_objective_mode": str(training_history.get("uplink_objective_mode", objective_mode)),
            "beam_reward_mode": str(training_history.get("beam_reward_mode", beam_reward_mode)),
            "initial_skipped_blocks_per_user": [
                int(v) for v in train_eval_initial_baseline.get("skipped_blocks_per_user", [0 for _ in range(K)])
            ],
        }
    )
    return train_eval_post


__all__ = ["train_blocklength_aware_precoder_net"]
