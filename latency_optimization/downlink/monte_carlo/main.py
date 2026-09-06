from __future__ import annotations

import argparse
import os
from time import perf_counter

import torch


from latency_optimization.core.scenarios import (
    STREAMING_MODE,
    build_experiment_scenario_summary,
    build_experiment_scenario_summary_lines,
    build_monte_carlo_sample_scenarios_for_seeds,
)
from latency_optimization.experiments.channels import (
    build_test_snr_schedule,
    build_training_snr_schedule,
    save_monte_carlo_base_samples,
    with_monte_carlo_sample_snr_by_user,
)
from latency_optimization.experiments.cost import (
    build_downlink_monte_carlo_total_cost,
    build_downlink_monte_carlo_training_cost,
    format_experiment_cost_lines,
)
from latency_optimization.experiments.determinism import configure_determinism
from latency_optimization.experiments.monte_carlo_testing import (
    build_seeded_scenario_collection_lines as _build_seeded_scenario_collection_lines,
    build_test_dataset_summary as _build_test_dataset_summary,
    build_test_sample_dirs,
    build_test_search_overrides as _build_test_search_overrides,
    build_test_search_tag as _build_test_search_tag,
)
from latency_optimization.experiments.seeds import (
    build_test_seeds_from_num_test_samples,
    resolve_monte_carlo_train_and_test_seeds,
)
from latency_optimization.results.naming import (
    format_method_tag,
    format_objective_tag,
    format_scope_tag,
    join_tag_parts,
    make_method_result_tag,
)
from latency_optimization.results.paths import build_downlink_result_dirs
from latency_optimization.results.persistence import (
    current_local_timestamp,
    save_json,
    save_text,
    write_result_manifest,
)

from ..config import load_config
from ..objective import get_convergence_objective_name
from ..plotting import (
    plot_asynchronality_comparison,
    plot_blocklength_feasibility_curves,
    plot_blocks,
    plot_interference_before_after_heatmaps,
    plot_interference_heatmaps,
    plot_latency,
    plot_link_quality,
    plot_optimization_history,
    plot_payload_rfbl_vs_n_with_epoch,
    plot_per_user_convergence,
    plot_per_user_interference_before_after,
    plot_per_user_interference_profiles,
    plot_per_user_schedule_details,
    plot_rate_violation_heatmap,
    plot_user_config,
)
from ..precoders.checkpoints import load_user_precoder_models
from ..runner import _compute_summary_metrics, _format_optional_db
from ..system import DownlinkSystem
from .evaluator import evaluate_downlink_precoder_net
from .rollout import build_training_dataset
from .trainer import (
    build_precoder_net_artifact,
    train_blocklength_aware_precoder_net,
)


def _build_dataset_summary_lines(dataset_summary: dict[str, object]) -> list[str]:
    return [
        "Downlink training dataset summary",
        f"Training sample unit: {dataset_summary.get('training_sample_unit', 'unknown')}",
        f"Training sample kind: {dataset_summary.get('training_sample_kind', 'unknown')}",
        f"Total training samples: {int(dataset_summary.get('total_training_samples', 0))}",
        f"Base dataset kind: {dataset_summary.get('base_dataset_kind', 'unknown')}",
        f"Training scenario modes: {dataset_summary.get('scenario_modes', [])}",
        f"Training SNR ranges by user (dB): {dataset_summary.get('training_snr_db_ranges', [])}",
        f"Training samples by seed: {dataset_summary.get('training_samples_by_seed', {})}",
        f"Training samples by user SNR (dB): {dataset_summary.get('training_samples_by_user_snr_db', [])}",
        f"Training samples per user: {dataset_summary.get('training_samples_per_user', [])}",
        "",
        "Terminology",
        "- payload channel episode: one seed-based payload rollout anchor; later blocks are generated online",
        "- streaming block sample: one independent seed-based block with a fresh per-block bit target",
    ]


def _build_post_training_summary_lines(post_training_summary: dict[str, object]) -> list[str]:
    lines = [
        "Downlink post-training summary",
        f"Run started at: {post_training_summary.get('run_started_at_local', 'unknown')}",
        f"Training started at: {post_training_summary.get('training_started_at_local', 'unknown')}",
        f"Training completed at: {post_training_summary.get('training_completed_at_local', 'unknown')}",
        f"Epochs requested: {int(post_training_summary.get('epochs_requested', 0))}",
        f"Epochs completed: {int(post_training_summary.get('epochs_completed', 0))}",
        f"Training solve status: {post_training_summary.get('training_solve_status', 'unknown')}",
        f"Restored solution source: {post_training_summary.get('restored_solution_source', 'unknown')}",
        f"Downlink precoder-net scope: {post_training_summary.get('downlink_precoder_net_scope', 'unknown')}",
        f"Selected checkpoint epoch: {int(post_training_summary.get('selected_checkpoint_epoch', 0))}",
        f"Selected checkpoint weighted-rate objective: {float(post_training_summary.get('selected_checkpoint_weighted_rate_objective', 0.0)):.6f}",
        f"Training sample unit: {post_training_summary.get('training_sample_unit', 'unknown')}",
        f"Base training dataset: {post_training_summary.get('base_dataset_kind', 'unknown')}",
        f"Training samples: {int(post_training_summary.get('total_training_samples', 0))}",
        f"Rollout anchor-bits mode: {post_training_summary.get('rollout_anchor_bits_mode', 'unknown')}",
        f"Final KKT primal residual: {float(post_training_summary.get('final_kkt_primal_residual', 0.0)):.6e}",
        f"Final KKT complementarity residual: {float(post_training_summary.get('final_kkt_complementarity_residual', 0.0)):.6e}",
        f"Final KKT stationarity residual: {float(post_training_summary.get('final_kkt_stationarity_residual', 0.0)):.6e}",
        f"Last epoch rollout-weighted avg sum rate: {float(post_training_summary.get('last_epoch_rollout_weighted_avg_sum_rate', post_training_summary.get('final_avg_sum_rate', 0.0))):.6f}",
        f"Best epoch rollout-weighted avg sum rate: {float(post_training_summary.get('best_epoch_rollout_weighted_avg_sum_rate', post_training_summary.get('best_avg_sum_rate', 0.0))):.6f}",
        f"Last epoch mean per-user rollout rate: {float(post_training_summary.get('last_epoch_mean_user_rollout_rate', post_training_summary.get('final_avg_user_rate', 0.0))):.6f}",
        f"Best epoch mean per-user rollout rate: {float(post_training_summary.get('best_epoch_mean_user_rollout_rate', post_training_summary.get('best_avg_user_rate', 0.0))):.6f}",
        f"Last epoch mean per-user objective loss: {float(post_training_summary.get('last_epoch_mean_user_objective_loss', 0.0)):.6f}",
        f"Best epoch mean per-user objective loss: {float(post_training_summary.get('best_epoch_mean_user_objective_loss', 0.0)):.6f}",
        (
            "Last epoch feasible rollout queries: "
            f"{int(post_training_summary.get('last_epoch_feasible_rollout_queries', 0))} / "
            f"{max(int(post_training_summary.get('last_epoch_total_rollout_queries', 0)), 0)} "
            f"({float(post_training_summary.get('final_feasible_rollout_query_fraction', 0.0)):.6f})"
        ),
        (
            "Per-user last epoch avg objective loss over active rollout queries: "
            f"{post_training_summary.get('per_user_last_epoch_avg_objective_loss_over_active_rollout_queries', [])}"
        ),
        f"Per-user best objective loss: {post_training_summary.get('per_user_best_objective_loss', [])}",
        (
            "Per-user last epoch avg rate over active rollout queries: "
            f"{post_training_summary.get('per_user_last_epoch_avg_rate_over_active_rollout_queries', post_training_summary.get('per_user_final_rate', []))}"
        ),
        "Global active-user rollout queries by n_kl over all epochs:",
        f"{post_training_summary.get('cumulative_rollout_queries_by_n_kl', {}).get('global_active_user_rollout_queries_by_n_kl_over_all_epochs', {})}",
        "Per-user active-user rollout queries by n_kl over all epochs:",
        f"{post_training_summary.get('cumulative_rollout_queries_by_n_kl', {}).get('per_user_active_user_rollout_queries_by_n_kl_over_all_epochs', [])}",
        "Global active-user frontier rollout queries by n_kl over all epochs:",
        f"{post_training_summary.get('cumulative_frontier_rollout_queries_by_n_kl', {}).get('global_active_user_frontier_rollout_queries_by_n_kl_over_all_epochs', {})}",
        "Per-user active-user frontier rollout queries by n_kl over all epochs:",
        f"{post_training_summary.get('cumulative_frontier_rollout_queries_by_n_kl', {}).get('per_user_active_user_frontier_rollout_queries_by_n_kl_over_all_epochs', [])}",
        "Last epoch active-user rollout queries by n_kl:",
        f"{post_training_summary.get('final_epoch_rollout_query_summary', {}).get('global_active_user_rollout_queries_by_n_kl', {})}",
        "Last epoch active-user frontier rollout queries by n_kl:",
        f"{post_training_summary.get('final_epoch_rollout_query_summary', {}).get('global_active_user_frontier_rollout_queries_by_n_kl', {})}",
    ]
    lines.extend(format_experiment_cost_lines(post_training_summary.get("experiment_cost")))
    lines.extend(
        [
            "",
            "Terminology",
            "- payload channel episode: one seed-based channel anchor whose later blocks are generated online",
            "- streaming block sample: one independent seed-based training or test block",
            "- rollout query: one visited joint (episode, n_targets) state generated online from the current precoder nets",
            "- last epoch rollout-weighted avg sum rate: average sum rate over the visited rollout queries in the last epoch, using the trainer's rollout weights",
            "- last epoch mean per-user rollout rate: first average each user's rate over that user's active rollout queries in the last epoch, then average over users",
        ]
    )
    return lines


def _sum_per_user_summary_field(metrics: dict[str, object], field: str) -> int:
    return int(sum(int(row.get(field, 0)) for row in metrics.get("per_user_summary", [])))


def _get_baseline_comparison_metric(metrics: dict[str, object], baseline_name: str) -> dict[str, object]:
    baseline_metrics = metrics.get("baseline_comparison_metrics", {})
    if not isinstance(baseline_metrics, dict):
        return {}
    value = baseline_metrics.get(baseline_name, {})
    return value if isinstance(value, dict) else {}


def _build_final_test_summary_lines(result: dict[str, object]) -> list[str]:
    metrics = result["summary_metrics"]
    assert isinstance(metrics, dict)
    random_baseline_metrics = _get_baseline_comparison_metric(metrics, "random_precoder_baseline")
    naive_full_t_metrics = _get_baseline_comparison_metric(metrics, "naive_full_T_baseline")
    lines = [
        "Final test results",
        f"Initial total latency (random baseline): {metrics['initial_total_latency']:.6f}",
        f"Final total latency: {metrics['final_total_latency']:.6f}",
        (
            "Latency reduction vs random precoder baseline (%): "
            f"{random_baseline_metrics.get('total_latency_reduction_percent', metrics['total_latency_reduction_percent']):.4f}"
        ),
        f"Initial avg latency (random baseline): {metrics['initial_avg_latency']:.6f}",
        f"Final avg latency: {metrics['final_avg_latency']:.6f}",
        f"Initial asynchronality sum (random baseline): {metrics['initial_asynchronality_sum']:.6f}",
        f"Final asynchronality sum: {metrics['final_asynchronality_sum']:.6f}",
        f"Asynchronality reduction (%): {metrics['asynchronality_reduction_percent']:.4f}",
        _format_optional_db("Initial avg served-user SNR (dB)", metrics["initial_avg_snr_db"]),
        _format_optional_db("Final avg served-user SNR (dB)", metrics["final_avg_snr_db"]),
        _format_optional_db("Initial avg served-block SINR (dB)", metrics["initial_avg_sinr_db"]),
        _format_optional_db("Final avg served-block SINR (dB)", metrics["final_avg_sinr_db"]),
        f"Total served bits: {_sum_per_user_summary_field(metrics, 'served_bits')}",
        f"Total skipped blocks: {_sum_per_user_summary_field(metrics, 'skipped_blocks')}",
    ]
    if naive_full_t_metrics:
        lines.extend(
            [
                f"Initial total latency (naive full-T baseline): {naive_full_t_metrics.get('initial_total_latency', 0.0):.6f}",
                (
                    "Latency reduction vs naive full-T baseline (%): "
                    f"{naive_full_t_metrics.get('total_latency_reduction_percent', 0.0):.4f}"
                ),
            ]
        )
    if metrics.get("scenario_mode", "") == STREAMING_MODE:
        lines.extend(
            [
                f"Total target bits: {_sum_per_user_summary_field(metrics, 'target_bits')}",
                f"Total unserved bits: {_sum_per_user_summary_field(metrics, 'unserved_bits')}",
                f"Total partially served blocks: {_sum_per_user_summary_field(metrics, 'partially_served_blocks')}",
                f"Total zero-service blocks: {_sum_per_user_summary_field(metrics, 'zero_service_blocks')}",
            ]
        )
    return lines


def _build_per_user_test_lines(result: dict[str, object]) -> list[str]:
    metrics = result["summary_metrics"]
    assert isinstance(metrics, dict)
    naive_full_t_metrics = _get_baseline_comparison_metric(metrics, "naive_full_T_baseline")
    naive_per_user = naive_full_t_metrics.get("latency_reduction_per_user_percent", [])
    lines = ["Per-user final test details"]
    for row in metrics["per_user_summary"]:
        parts = [
            f"User {row['user']}",
            f"init_lat={row['initial_latency']:.6f}",
            f"final_lat={row['final_latency']:.6f}",
            f"lat_red_vs_random={row['latency_reduction_percent']:.4f}%",
            (
                f"init_served_block_sinr={row['initial_sinr_db']:.4f} dB"
                if int(row["initial_served_blocks"]) > 0
                else "init_served_block_sinr=n/a"
            ),
            (
                f"final_served_block_sinr={row['final_sinr_db']:.4f} dB"
                if int(row["final_served_blocks"]) > 0
                else "final_served_block_sinr=n/a"
            ),
            f"blocks={row['blocks']}",
            f"total_n={row['total_n']}",
            f"served_bits={row['served_bits']}",
            f"skipped_blocks={row['skipped_blocks']}",
        ]
        if int(row["user"]) < len(naive_per_user):
            parts.append(f"lat_red_vs_fullT={float(naive_per_user[int(row['user'])]):.4f}%")
        if metrics.get("scenario_mode", "") == STREAMING_MODE:
            parts.extend(
                [
                    f"target_bits={row.get('target_bits', 0)}",
                    f"unserved_bits={row.get('unserved_bits', 0)}",
                    f"partial_blocks={row.get('partially_served_blocks', 0)}",
                    f"zero_service_blocks={row.get('zero_service_blocks', 0)}",
                ]
            )
        lines.append(" | ".join(parts))
    return lines


def _build_summary_lines(result: dict[str, object], cfg_path: str, test_seed: int) -> list[str]:
    metrics = result["summary_metrics"]
    assert isinstance(metrics, dict)
    dataset_summary = result.get("training_dataset_summary", {})
    post_training_summary = result.get("post_training_summary", {})

    lines = [
        "Downlink optimizer summary",
        "",
        "Setup",
        f"Method: {result.get('method_name', 'unknown')}",
        f"Config: {cfg_path}",
        f"Test seed: {int(test_seed)}",
        f"Train seeds: {result.get('train_seeds', [])}",
        f"Run started at: {result.get('run_started_at_local', 'unknown')}",
        f"Run completed at: {result.get('run_completed_at_local', 'unknown')}",
        f"Scenario: {result.get('experiment_scenario_mode', 'unknown')}",
        f"Objective mode: {result.get('objective_mode', 'unknown')}",
        f"Allocation mode: {result.get('allocation_mode', 'unknown')}",
        f"Weight strategy: {result.get('weight_strategy', 'n/a')}",
        f"Downlink precoder-net scope: {result.get('downlink_precoder_net_scope', 'unknown')}",
        f"Precoder parameterization: {result.get('precoder_parameterization', 'unknown')}",
        f"Training objective: {result.get('training_objective', 'unknown')}",
        f"Test n search strategy: {result.get('test_n_search_strategy', 'config_default')}",
        f"Test n search direction: {result.get('test_n_search_direction', 'config_default')}",
        f"Initial schedule source: {result.get('initial_schedule_source', 'random_precoder_baseline')}",
        f"Training dataset sample kind: {dataset_summary.get('training_sample_kind', 'unknown') if isinstance(dataset_summary, dict) else 'unknown'}",
        f"Training dataset sample unit: {dataset_summary.get('training_sample_unit', 'unknown') if isinstance(dataset_summary, dict) else 'unknown'}",
        f"Training dataset total samples: {int(dataset_summary.get('total_training_samples', 0)) if isinstance(dataset_summary, dict) else 0}",
        f"Training SNR ranges by user (dB): {dataset_summary.get('training_snr_db_ranges', []) if isinstance(dataset_summary, dict) else []}",
        f"Training sample counts per user: {result.get('training_dataset_sizes', [])}",
    ]
    if result.get("reused_training_artifact"):
        lines.append(f"Reused training artifact: {result.get('reused_training_artifact')}")
    lines.extend([""])
    lines.extend(_build_final_test_summary_lines(result))
    training_history = result.get("precoder_net_training_history", {})
    if isinstance(training_history, dict) and training_history.get("sum_rate"):
        sum_rate_hist = training_history.get("sum_rate", [])
        avg_user_rate_hist = training_history.get("avg_user_rate", [])
        lines.extend(
            [
                "",
                "Training results",
                f"Epochs requested: {int(post_training_summary.get('epochs_requested', 0))}" if isinstance(post_training_summary, dict) else "Epochs requested: 0",
                (
                    f"Epochs completed: {int(post_training_summary.get('epochs_completed', 0))}"
                    if isinstance(post_training_summary, dict)
                    else "Epochs completed: 0"
                ),
                (
                    f"Training solve status: {post_training_summary.get('training_solve_status', 'unknown')}"
                    if isinstance(post_training_summary, dict)
                    else "Training solve status: unknown"
                ),
                (
                    f"Base training dataset: {post_training_summary.get('base_dataset_kind', 'unknown')}"
                    if isinstance(post_training_summary, dict)
                    else "Base training dataset: unknown"
                ),
                (
                    f"Rollout anchor-bits mode: {post_training_summary.get('rollout_anchor_bits_mode', 'unknown')}"
                    if isinstance(post_training_summary, dict)
                    else "Rollout anchor-bits mode: unknown"
                ),
                (
                    f"Last epoch rollout-weighted avg sum rate: {float(post_training_summary.get('last_epoch_rollout_weighted_avg_sum_rate', post_training_summary.get('final_avg_sum_rate', float(sum_rate_hist[-1])))):.6f}"
                    if isinstance(post_training_summary, dict)
                    else f"Last epoch rollout-weighted avg sum rate: {float(sum_rate_hist[-1]):.6f}"
                ),
                (
                    f"Best epoch rollout-weighted avg sum rate: {float(post_training_summary.get('best_epoch_rollout_weighted_avg_sum_rate', post_training_summary.get('best_avg_sum_rate', float(sum_rate_hist[-1])))):.6f}"
                    if isinstance(post_training_summary, dict)
                    else f"Best epoch rollout-weighted avg sum rate: {float(sum_rate_hist[-1]):.6f}"
                ),
                (
                    f"Last epoch mean per-user rollout rate: {float(post_training_summary.get('last_epoch_mean_user_rollout_rate', post_training_summary.get('final_avg_user_rate', float(avg_user_rate_hist[-1]) if avg_user_rate_hist else 0.0))):.6f}"
                    if isinstance(post_training_summary, dict)
                    else (f"Last epoch mean per-user rollout rate: {float(avg_user_rate_hist[-1]):.6f}" if avg_user_rate_hist else "Last epoch mean per-user rollout rate: n/a")
                ),
                (
                    f"Best epoch mean per-user rollout rate: {float(post_training_summary.get('best_epoch_mean_user_rollout_rate', post_training_summary.get('best_avg_user_rate', float(avg_user_rate_hist[-1]) if avg_user_rate_hist else 0.0))):.6f}"
                    if isinstance(post_training_summary, dict)
                    else (f"Best epoch mean per-user rollout rate: {float(avg_user_rate_hist[-1]):.6f}" if avg_user_rate_hist else "Best epoch mean per-user rollout rate: n/a")
                ),
                (
                    f"Last epoch mean per-user objective loss: {float(post_training_summary.get('last_epoch_mean_user_objective_loss', float(training_history.get('avg_objective_loss', [])[-1]) if training_history.get('avg_objective_loss') else 0.0)):.6f}"
                    if isinstance(post_training_summary, dict)
                    else (f"Last epoch mean per-user objective loss: {float(training_history.get('avg_objective_loss', [])[-1]):.6f}" if training_history.get("avg_objective_loss") else "Last epoch mean per-user objective loss: n/a")
                ),
                (
                    f"Best epoch mean per-user objective loss: {float(post_training_summary.get('best_epoch_mean_user_objective_loss', float(training_history.get('avg_objective_loss', [])[-1]) if training_history.get('avg_objective_loss') else 0.0)):.6f}"
                    if isinstance(post_training_summary, dict)
                    else (f"Best epoch mean per-user objective loss: {float(training_history.get('avg_objective_loss', [])[-1]):.6f}" if training_history.get("avg_objective_loss") else "Best epoch mean per-user objective loss: n/a")
                ),
                (
                    "Per-user last epoch avg rate over active rollout queries: "
                    f"{training_history.get('per_user_rate', []) and [float(row[-1]) if len(row) > 0 else 0.0 for row in training_history.get('per_user_rate', [])]}"
                ),
                (
                    "Per-user last epoch avg objective loss over active rollout queries: "
                    f"{training_history.get('per_user_objective_loss', []) and [float(row[-1]) if len(row) > 0 else 0.0 for row in training_history.get('per_user_objective_loss', [])]}"
                ),
                (
                    "Last epoch feasible rollout queries: "
                    f"{int(post_training_summary.get('last_epoch_feasible_rollout_queries', 0))} / "
                    f"{max(int(post_training_summary.get('last_epoch_total_rollout_queries', 0)), 0)} "
                    f"({float(post_training_summary.get('final_feasible_rollout_query_fraction', 0.0)):.6f})"
                    if isinstance(post_training_summary, dict)
                    else "Last epoch feasible rollout queries: n/a"
                ),
            ]
        )
    else:
        lines.extend(["", "Training results", "Training metrics: n/a"])

    lines.extend([""])
    lines.extend(_build_per_user_test_lines(result))

    if metrics["initial_asynchronality_pairs"]:
        lines.extend(["", "Per-pair asynchronality"])
        for init_pair, final_pair in zip(metrics["initial_asynchronality_pairs"], metrics["final_asynchronality_pairs"]):
            lines.append(
                " | ".join(
                    [
                        f"Users {init_pair['user_i']}-{init_pair['user_j']}",
                        f"initial_diff={init_pair['abs_latency_diff']:.6f}",
                        f"final_diff={final_pair['abs_latency_diff']:.6f}",
                    ]
                )
            )
    if isinstance(post_training_summary, dict) and len(post_training_summary) > 0:
        lines.extend(
            [
                "",
                "Additional training details",
                f"Global active-user rollout queries by n_kl over all epochs: {post_training_summary.get('cumulative_rollout_queries_by_n_kl', {}).get('global_active_user_rollout_queries_by_n_kl_over_all_epochs', {})}",
                f"Per-user active-user rollout queries by n_kl over all epochs: {post_training_summary.get('cumulative_rollout_queries_by_n_kl', {}).get('per_user_active_user_rollout_queries_by_n_kl_over_all_epochs', [])}",
                f"Per-user active-user frontier rollout queries by n_kl over all epochs: {post_training_summary.get('cumulative_frontier_rollout_queries_by_n_kl', {}).get('per_user_active_user_frontier_rollout_queries_by_n_kl_over_all_epochs', [])}",
            ]
        )

    lines.extend(format_experiment_cost_lines(result.get("experiment_cost")))
    lines.extend(
        [
            "",
            "Terminology",
            "- payload channel episode: one seed-based channel anchor whose later blocks are generated online",
            "- streaming block sample: one independent seed-based training or test block",
            "- rollout query: one visited joint (episode, n_targets) state generated online from the current precoder nets",
        ]
    )

    return lines


def evaluate_trained_precoder_network_on_test_channel(
    train_artifact: dict[str, object],
    cfg_name: str,
    test_seed: int,
    *,
    output_dirs: dict[str, str],
    train_seeds: list[int],
    verbose: bool,
    do_plots: bool,
    test_snr_db_by_user: list[float],
    test_search_overrides: dict[str, object] | None = None,
    reused_training_artifact: str | None = None,
    training_scenario_summaries: list[dict[str, object]] | None = None,
    training_wall_time_seconds: float,
    run_started_at_local: str,
    training_started_at_local: str,
    training_completed_at_local: str,
    precoder_net_batch_size: int,
) -> dict[str, object]:
    configure_determinism(int(test_seed))
    system_params, sim_params, run_meta = load_config(cfg_name)
    system_params = with_monte_carlo_sample_snr_by_user(system_params, test_snr_db_by_user)
    sim_params = dict(sim_params)
    if str(sim_params["experiment_scenario_mode"]) == "streaming":
        sim_params["experiment_scenario"] = {
            **sim_params["experiment_scenario"],
            "number_of_blocks": 1,
        }
    if test_search_overrides:
        sim_params.update(test_search_overrides)

    test_system = DownlinkSystem(system_params, seed=int(test_seed))
    test_scenario_summary = build_experiment_scenario_summary(
        build_monte_carlo_sample_scenarios_for_seeds(system_params, sim_params, [int(test_seed)])[0]
    )
    user_models = load_user_precoder_models(
        train_artifact["user_model_specs"],
        train_artifact["user_model_states"],
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    )
    testing_started_at_local = current_local_timestamp()
    core_evaluation_start = perf_counter()
    result = evaluate_downlink_precoder_net(
        test_system,
        sim_params,
        user_models,
        verbose=verbose,
        precoder_net_training_history=train_artifact.get("precoder_net_training_history", {}),
        train_seeds=train_seeds,
        training_dataset_sizes=train_artifact.get("training_dataset_sizes", []),
    )
    testing_wall_time_seconds = float(
        result.get(
            "core_evaluation_wall_time_seconds",
            perf_counter() - core_evaluation_start,
        )
    )
    testing_completed_at_local = current_local_timestamp()
    post_training_summary = train_artifact.get("post_training_summary", {})
    if not isinstance(post_training_summary, dict):
        post_training_summary = {}
    result["cfg_path"] = run_meta["cfg_path"]
    result["seed"] = int(test_seed)
    result["test_snr_db_by_user"] = [float(value) for value in test_snr_db_by_user]
    result["system_params"] = system_params
    result["sim_params"] = sim_params
    result["training_dataset_summary"] = train_artifact.get("training_dataset_summary", {})
    result["post_training_summary"] = post_training_summary
    result["experiment_scenario_mode"] = sim_params.get("experiment_scenario_mode", "payload")
    result["experiment_scenario"] = test_scenario_summary
    result["training_experiment_scenarios"] = (
        training_scenario_summaries
        if training_scenario_summaries is not None
        else train_artifact.get("training_experiment_scenarios", [])
    )
    result["training_objective"] = train_artifact.get(
        "training_objective",
        result.get("training_objective", "inverse_cnr_weighted_finite_blocklength_rate"),
    )
    result["test_n_search_strategy"] = str(
        sim_params.get("monte_carlo_test_n_search_strategy", sim_params.get("n_search_strategy", "unknown"))
    )
    result["test_n_search_direction"] = str(
        sim_params.get("monte_carlo_test_n_search_direction", sim_params.get("n_search_direction", "unknown"))
    )
    if reused_training_artifact:
        result["reused_training_artifact"] = str(reused_training_artifact)
    result["experiment_cost"] = build_downlink_monte_carlo_total_cost(
        train_artifact,
        result.get("evaluation_cost_counters", {}),
        batch_size=precoder_net_batch_size,
        core_wall_time_seconds_training=training_wall_time_seconds,
        core_wall_time_seconds_testing=testing_wall_time_seconds,
    )
    result["run_started_at_local"] = str(run_started_at_local)
    result["run_completed_at_local"] = str(testing_completed_at_local)
    result["training_started_at_local"] = str(training_started_at_local)
    result["training_completed_at_local"] = str(training_completed_at_local)
    result["testing_started_at_local"] = str(testing_started_at_local)
    result["testing_completed_at_local"] = str(testing_completed_at_local)
    result["core_evaluation_wall_time_seconds"] = float(testing_wall_time_seconds)
    result["summary_metrics"] = _compute_summary_metrics(result)

    if do_plots:
        plot_user_config(system_params, output_dirs["user_config"])
        plot_latency(result, output_dirs["latency_asynchronality"])
        plot_asynchronality_comparison(result, output_dirs["latency_asynchronality"])
        plot_link_quality(result, output_dirs["link_quality"])
        plot_blocks(result, output_dirs["schedule_details"])
        plot_rate_violation_heatmap(result, output_dirs["optimization_history"])
        plot_optimization_history(result, output_dirs["optimization_history"])
        plot_per_user_schedule_details(result, output_dirs["schedule_details"])
        plot_per_user_convergence(result, output_dirs["optimization_history"])
        plot_blocklength_feasibility_curves(test_system, result, output_dirs["optimization_history"])
        plot_payload_rfbl_vs_n_with_epoch(result, output_dirs["optimization_history"])
        plot_interference_before_after_heatmaps(result, output_dirs["interference"])
        plot_per_user_interference_before_after(result, output_dirs["interference"])
        plot_interference_heatmaps(test_system, output_dirs["interference"])
        plot_per_user_interference_profiles(test_system, output_dirs["interference"])

    save_json(result, os.path.join(output_dirs["test_data"], "result.json"))
    save_text(
        _build_summary_lines(result, run_meta["cfg_path"], int(test_seed)),
        os.path.join(output_dirs["test_data"], "summary.txt"),
    )
    save_json(test_scenario_summary, os.path.join(output_dirs["test_data"], "experiment_scenario.json"))
    save_text(
        build_experiment_scenario_summary_lines(test_scenario_summary),
        os.path.join(output_dirs["test_data"], "experiment_scenario.txt"),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline downlink Monte Carlo precoder-net train/test")
    parser.add_argument("--cfg_name", type=str, default="downlink_dispersion_heavy.yaml", help="Configuration file name or path")
    parser.add_argument("--train_seeds", type=str, default=None, help="Explicit comma-separated training seeds")
    parser.add_argument("--num_train_channels", type=int, default=None, help="Payload: number of training channel episodes")
    parser.add_argument("--num_train_blocks", type=int, default=None, help="Streaming: number of independent training blocks")
    parser.add_argument("--test_seed", type=int, default=None, help="Deterministic Monte Carlo test seed")
    parser.add_argument("--num_test_channels", type=int, default=None, help="Payload: number of held-out channel episodes")
    parser.add_argument("--num_test_blocks", type=int, default=None, help="Streaming: number of held-out blocks")
    parser.add_argument("--precoder_net_epochs", type=int, default=None)
    parser.add_argument("--precoder_net_batch_size", type=int, default=32)
    parser.add_argument("--precoder_net_lr", type=float, default=1e-3)
    parser.add_argument("--reuse_train_artifact", type=str, default=None, help="Reuse a saved train_artifact.pt and rerun only the test phase")
    parser.add_argument("--test_n_search_strategy", type=str, default=None, help="Override Monte Carlo test-only n-search strategy")
    parser.add_argument("--test_n_search_direction", type=str, default=None, help="Override Monte Carlo test-only n-search direction")
    parser.add_argument("--test_n_search_coarse_step", type=int, default=None, help="Override Monte Carlo test-only n-search coarse step")
    parser.add_argument("--test_n_search_exponential_factor", type=int, default=None, help="Override Monte Carlo test-only n-search exponential factor")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
    args = parser.parse_args()

    verbose = not args.quiet
    system_params, sim_params, run_meta = load_config(args.cfg_name)
    scenario_mode = str(sim_params["experiment_scenario_mode"])
    if scenario_mode == "payload":
        if args.num_train_blocks is not None or args.num_test_blocks is not None:
            raise ValueError("Payload Monte Carlo uses channel-count options, not block-count options.")
        cli_training_samples = args.num_train_channels
        config_training_samples = sim_params.get("monte_carlo_num_training_channels")
        cli_test_samples = args.num_test_channels
        config_test_samples = sim_params.get("monte_carlo_num_test_channels")
    else:
        if args.num_train_channels is not None or args.num_test_channels is not None:
            raise ValueError("Streaming Monte Carlo uses block-count options, not channel-count options.")
        cli_training_samples = args.num_train_blocks
        config_training_samples = sim_params.get("monte_carlo_num_training_blocks")
        cli_test_samples = args.num_test_blocks
        config_test_samples = sim_params.get("monte_carlo_num_test_blocks")
    if cli_training_samples is None and config_training_samples is None:
        raise ValueError(f"Monte Carlo training sample count is missing for {scenario_mode}.")
    if cli_test_samples is None and config_test_samples is None:
        raise ValueError(f"Monte Carlo test sample count is missing for {scenario_mode}.")
    run_started_at_local = current_local_timestamp()
    test_search_overrides = _build_test_search_overrides(args)
    train_epochs = int(
        args.precoder_net_epochs
        if args.precoder_net_epochs is not None
        else sim_params.get("monte_carlo_training_max_epochs", sim_params.get("max_epochs", 100))
    )
    train_seeds, test_seed = resolve_monte_carlo_train_and_test_seeds(
        cli_train_seeds=args.train_seeds,
        cli_num_training_samples=cli_training_samples,
        cli_test_seed=args.test_seed,
        config_train_seeds=sim_params.get("monte_carlo_train_seeds"),
        config_num_training_samples=config_training_samples,
        config_test_seed=sim_params.get("monte_carlo_test_seed"),
    )
    test_seeds = build_test_seeds_from_num_test_samples(
        int(cli_test_samples if cli_test_samples is not None else config_test_samples),
        first_test_seed=int(test_seed),
        excluded_seeds=train_seeds,
    )
    test_snr_db_by_user_by_seed = build_test_snr_schedule(
        test_seeds,
        sim_params["monte_carlo_test_snr_db_ranges"],
        num_users=int(system_params["K"]),
    )
    artifact: dict[str, object] | None = None
    reused_training_artifact: str | None = None
    if args.reuse_train_artifact:
        reused_training_artifact = os.path.abspath(args.reuse_train_artifact)
        artifact = torch.load(reused_training_artifact, map_location="cpu", weights_only=False)
        if not isinstance(artifact, dict):
            raise TypeError("Expected a dictionary train artifact when reusing Monte Carlo training.")
        artifact_train_seeds = [int(v) for v in artifact.get("train_seeds", [])]
        if artifact_train_seeds:
            train_seeds = artifact_train_seeds
    configure_determinism(train_seeds[0] if train_seeds else 0)
    training_snr_db_by_user_by_seed = build_training_snr_schedule(
        train_seeds,
        sim_params["monte_carlo_training_snr_db_ranges"],
        num_users=int(system_params["K"]),
    )
    print(f"Resolved Monte Carlo train seeds: {train_seeds}")
    print(f"Monte Carlo training sample unit: {sim_params['monte_carlo_sample_unit']}")
    print(f"Resolved Monte Carlo test seed: {int(test_seed)}")
    print(f"Resolved Monte Carlo test dataset seeds: {test_seeds}")
    training_scenario_summaries = []
    for seed, scenario in zip(
        train_seeds,
        build_monte_carlo_sample_scenarios_for_seeds(system_params, sim_params, train_seeds),
    ):
        summary = build_experiment_scenario_summary(scenario)
        summary["training_snr_db_by_user"] = list(training_snr_db_by_user_by_seed[int(seed)])
        training_scenario_summaries.append(summary)
    if isinstance(artifact, dict) and artifact.get("training_experiment_scenarios"):
        training_scenario_summaries = list(artifact["training_experiment_scenarios"])
    scope_name = sim_params.get("downlink_precoder_net_scope", "per_user_nets")
    objective_mode = get_convergence_objective_name(sim_params)
    scope_tag = format_scope_tag(scope_name)
    result_tag = make_method_result_tag(
        join_tag_parts(
            format_method_tag("monte_carlo_precoder_net_train_test"),
            format_objective_tag(objective_mode),
            scope_tag,
            f"testset{len(test_seeds)}",
            _build_test_search_tag(test_search_overrides),
        ),
        run_meta["cfg_stem"],
        seed=int(test_seed),
        cfg_hash=run_meta.get("cfg_hash"),
    )
    output_dirs = build_downlink_result_dirs(
        "Monte Carlo",
        result_tag,
        scenario_mode=str(sim_params["experiment_scenario_mode"]),
    )
    output_root = output_dirs["experiment_root"]
    training_sample_manifest = save_monte_carlo_base_samples(
        output_dir=output_dirs["train_samples"],
        link_name="Downlink",
        system_params=system_params,
        sample_seeds=train_seeds,
        system_factory=DownlinkSystem,
        sample_snr_db_by_user_by_seed=training_snr_db_by_user_by_seed,
        dataset_role="training",
        sample_kind="payload_channel_episode" if str(sim_params["experiment_scenario_mode"]) == "payload" else "streaming_block",
    )
    test_sample_manifest = save_monte_carlo_base_samples(
        output_dir=output_dirs["test_samples"],
        link_name="Downlink test",
        system_params=system_params,
        sample_seeds=test_seeds,
        system_factory=DownlinkSystem,
        sample_snr_db_by_user_by_seed=test_snr_db_by_user_by_seed,
        dataset_role="test",
        sample_kind="payload_channel_episode" if str(sim_params["experiment_scenario_mode"]) == "payload" else "streaming_block",
    )

    if artifact is None:
        training_started_at_local = current_local_timestamp()
        training_start = perf_counter()
        training_scenarios = build_training_dataset(
            train_seeds,
            system_params,
            sim_params,
            verbose=verbose,
        )
        user_models, precoder_net_training_history, training_dataset_sizes = train_blocklength_aware_precoder_net(
            system_params,
            sim_params,
            training_scenarios,
            epochs=train_epochs,
            batch_size=args.precoder_net_batch_size,
            lr=args.precoder_net_lr,
            verbose=verbose,
        )
        training_wall_time_seconds = perf_counter() - training_start
        training_completed_at_local = current_local_timestamp()
        dataset_summary = precoder_net_training_history.get("dataset_summary", {})
        post_training_summary = precoder_net_training_history.get("post_training_summary", {})
        artifact = build_precoder_net_artifact(
            system_params,
            sim_params,
            train_seeds,
            user_models,
            precoder_net_training_history,
            training_dataset_sizes,
        )
        training_cost = build_downlink_monte_carlo_training_cost(
            artifact,
            batch_size=args.precoder_net_batch_size,
            core_wall_time_seconds_training=training_wall_time_seconds,
        )
        post_training_summary["experiment_cost"] = training_cost
        post_training_summary["run_started_at_local"] = str(run_started_at_local)
        post_training_summary["training_started_at_local"] = str(training_started_at_local)
        post_training_summary["training_completed_at_local"] = str(training_completed_at_local)

        artifact["training_dataset_summary"] = dataset_summary
        artifact["post_training_summary"] = post_training_summary
        artifact["experiment_scenario_mode"] = sim_params.get("experiment_scenario_mode", "payload")
        artifact["training_experiment_scenarios"] = training_scenario_summaries
        artifact["training_snr_db_by_user_by_seed"] = training_snr_db_by_user_by_seed
        artifact["training_sample_manifest"] = training_sample_manifest
        artifact["experiment_cost"] = training_cost
        torch.save(artifact, os.path.join(output_dirs["train_data"], "train_artifact.pt"))
        save_json(dataset_summary, os.path.join(output_dirs["train_data"], "training_dataset_summary.json"))
        save_text(
            _build_dataset_summary_lines(dataset_summary),
            os.path.join(output_dirs["train_data"], "training_dataset_summary.txt"),
        )
        save_json(post_training_summary, os.path.join(output_dirs["train_data"], "post_training_summary.json"))
        save_text(
            _build_post_training_summary_lines(post_training_summary),
            os.path.join(output_dirs["train_data"], "post_training_summary.txt"),
        )
        save_json(
            {"seed_scenarios": training_scenario_summaries},
            os.path.join(output_dirs["train_data"], "experiment_scenarios.json"),
        )
        save_text(
            _build_seeded_scenario_collection_lines(
                training_scenario_summaries,
                title="Training experiment scenarios by seed",
            ),
            os.path.join(output_dirs["train_data"], "experiment_scenarios.txt"),
        )
    else:
        print(f"Reusing Monte Carlo training artifact: {reused_training_artifact}")
        prior_cost = artifact.get("experiment_cost", {})
        if not isinstance(prior_cost, dict):
            prior_cost = {}
        training_wall_time_seconds = float(prior_cost.get("core_wall_time_seconds_training", 0.0))
        prior_training_summary = artifact.get("post_training_summary", {})
        if not isinstance(prior_training_summary, dict):
            prior_training_summary = {}
        training_started_at_local = str(
            prior_training_summary.get("training_started_at_local", "reused_training_artifact")
        )
        training_completed_at_local = str(
            prior_training_summary.get("training_completed_at_local", "reused_training_artifact")
        )
        save_json(
            {
                "reused_training_artifact": reused_training_artifact,
                "train_seeds": train_seeds,
                "test_seed": int(test_seed),
                "test_search_overrides": test_search_overrides,
            },
            os.path.join(output_dirs["train_data"], "reused_training_artifact.json"),
        )
        save_text(
            [
                "This Monte Carlo run reused an existing training artifact and reran only the test phase.",
                f"Source artifact: {reused_training_artifact}",
                f"Train seeds: {train_seeds}",
                f"Test seed: {int(test_seed)}",
                f"Test n-search overrides: {test_search_overrides}",
            ],
            os.path.join(output_dirs["train_data"], "reused_training_artifact.txt"),
        )

    test_results: list[dict[str, object]] = []
    for test_index, episode_seed in enumerate(test_seeds):
        sample_dirs = build_test_sample_dirs(output_dirs, int(episode_seed), link="downlink")
        test_results.append(
            evaluate_trained_precoder_network_on_test_channel(
                artifact,
                args.cfg_name,
                int(episode_seed),
                output_dirs=sample_dirs,
                train_seeds=train_seeds,
                verbose=verbose and test_index == 0,
                do_plots=True,
                test_snr_db_by_user=test_snr_db_by_user_by_seed[int(episode_seed)],
                test_search_overrides=test_search_overrides,
                reused_training_artifact=reused_training_artifact,
                training_scenario_summaries=training_scenario_summaries,
                training_wall_time_seconds=training_wall_time_seconds,
                run_started_at_local=str(run_started_at_local),
                training_started_at_local=str(training_started_at_local),
                training_completed_at_local=str(training_completed_at_local),
                precoder_net_batch_size=args.precoder_net_batch_size,
            )
        )
    test_dataset_summary = _build_test_dataset_summary(
        test_results,
        test_snr_db_by_user_by_seed,
    )
    test_dataset_summary["sample_manifest"] = test_sample_manifest
    save_json(test_dataset_summary, os.path.join(output_dirs["test_data"], "test_dataset_summary.json"))
    save_text(
        [
            "Downlink Monte Carlo test dataset summary",
            f"Held-out sample kind: {test_dataset_summary.get('test_sample_kind', 'unknown')}",
            f"Held-out sample unit: {test_dataset_summary.get('test_sample_unit', 'unknown')}",
            f"Total held-out samples: {test_dataset_summary['total_test_samples']}",
            f"Mean final total latency (s): {test_dataset_summary['final_total_latency_seconds']['mean']:.6f}",
            f"Std final total latency (s): {test_dataset_summary['final_total_latency_seconds']['std']:.6f}",
            f"Mean latency reduction vs random (%): {test_dataset_summary['latency_reduction_vs_random_percent']['mean']:.6f}",
            f"Std latency reduction vs random (%): {test_dataset_summary['latency_reduction_vs_random_percent']['std']:.6f}",
            f"Mean final asynchronality sum (s): {test_dataset_summary['final_asynchronality_sum_seconds']['mean']:.6f}",
            f"Std final asynchronality sum (s): {test_dataset_summary['final_asynchronality_sum_seconds']['std']:.6f}",
            "",
            "Each sample is saved under testing/samples/seed_<seed>/ with its own per-user SNR vector.",
        ],
        os.path.join(output_dirs["test_data"], "test_dataset_summary.txt"),
    )
    write_result_manifest(
        output_root,
        setup={
            "link": "Downlink",
            "method": "Monte Carlo",
            "scenario": sim_params["experiment_scenario_mode"],
            "config_path": run_meta["cfg_path"],
            "config_hash": run_meta.get("cfg_hash"),
            "train_seeds": train_seeds,
            "test_seed": int(test_seed),
            "test_seeds": test_seeds,
            "training_artifact_reused": reused_training_artifact,
            "run_started_at_local": run_started_at_local,
            "run_completed_at_local": current_local_timestamp(),
        },
    )
    print(f"Saved downlink Monte Carlo results to: {output_root}")


if __name__ == "__main__":
    main()
