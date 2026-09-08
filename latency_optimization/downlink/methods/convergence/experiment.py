"""Run one complete downlink convergence experiment.

Start at :func:`run_convergence_experiment` for configuration, optimization,
metrics, plots, and persistence. The mathematical procedure begins in
``optimize_transmission.py``.
"""

from __future__ import annotations

import argparse
import os
from time import perf_counter

from latency_optimization.experiments.scenarios import STREAMING_MODE
from latency_optimization.experiments.cost import build_downlink_convergence_cost, format_experiment_cost_lines
from latency_optimization.experiments.determinism import configure_determinism
from latency_optimization.results.metrics import (
    format_optional_db as _format_optional_db,
)
from latency_optimization.results.naming import (
    format_method_tag,
    format_objective_tag,
    format_scope_tag,
    format_update_mode_tag,
    join_tag_parts,
    make_method_result_tag,
)
from latency_optimization.results.persistence import (
    current_local_timestamp,
    save_json,
    save_text,
    write_result_manifest,
)
from latency_optimization.results.paths import build_downlink_convergence_result_dirs

from ...configuration.loader import load_config
from ...results.metrics import compute_downlink_summary_metrics
from .optimize_transmission import (
    optimize_downlink_transmission,
)
from ...objectives.settings import (
    get_convergence_objective_name,
)
from ...results.plotting import (
    initialize_output_dirs,
    plot_asynchronality_comparison,
    plot_blocklength_feasibility_curves,
    plot_blocks,
    plot_interference_before_after_heatmaps,
    plot_interference_heatmaps,
    plot_kkt_residual_history,
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
from ...simulation.system import DownlinkSystem


METHOD_NAME = "convergence_per_epoch_baseline"
METHOD_LABEL = "Convergence per epoch"


def build_result_tag(
    method_name: str,
    cfg_stem: str,
    seed: int,
    *,
    objective_mode: str | None = None,
    model_scope: str | None = None,
    solver_mode: str | None = None,
    cfg_hash: str | None = None,
) -> str:
    """Build the compact method/config/seed identifier used for result directories."""
    method_parts = [format_method_tag(method_name)]
    if objective_mode:
        method_parts.append(format_objective_tag(objective_mode))
    solver_tag = format_update_mode_tag(solver_mode) if solver_mode else ""
    if model_scope and solver_tag != "dir":
        method_parts.append(format_scope_tag(model_scope))
    if solver_mode:
        method_parts.append(solver_tag)
    method_tag = join_tag_parts(*method_parts)
    return make_method_result_tag(method_tag, cfg_stem, seed=seed, cfg_hash=cfg_hash)





def _sum_per_user_summary_field(metrics: dict[str, object], field: str) -> int:
    return int(sum(int(row.get(field, 0)) for row in metrics.get("per_user_summary", [])))


def _get_baseline_comparison_metric(metrics: dict[str, object], baseline_name: str) -> dict[str, object]:
    baseline_metrics = metrics.get("baseline_comparison_metrics", {})
    if not isinstance(baseline_metrics, dict):
        return {}
    value = baseline_metrics.get(baseline_name, {})
    return value if isinstance(value, dict) else {}


def _build_downlink_final_test_section_lines(result: dict[str, object]) -> list[str]:
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
        f"Initial avg all-block SINR (dB): {metrics['initial_avg_sinr_db_all_blocks']:.4f}",
        f"Final avg all-block SINR (dB): {metrics['final_avg_sinr_db_all_blocks']:.4f}",
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
                f"Overall delivery ratio: {float(metrics.get('total_delivery_ratio', 0.0)):.6f}",
                f"Total partially served blocks: {_sum_per_user_summary_field(metrics, 'partially_served_blocks')}",
                f"Total zero-service blocks: {_sum_per_user_summary_field(metrics, 'zero_service_blocks')}",
            ]
        )
    return lines


def _build_downlink_per_user_test_lines(result: dict[str, object]) -> list[str]:
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
            f"init_all_block_sinr={row['initial_sinr_db_all_blocks']:.4f} dB",
            f"final_all_block_sinr={row['final_sinr_db_all_blocks']:.4f} dB",
            f"served_blocks={row['final_served_blocks']}",
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
                    f"delivery_ratio={float(row.get('delivery_ratio', 0.0)):.6f}",
                    f"successful_blocks={row.get('successful_blocks', 0)}",
                    f"deadline_miss_fraction={float(row.get('deadline_miss_fraction', 0.0)):.6f}",
                    f"avg_n_kl={float(row.get('average_selected_blocklength', 0.0)):.4f}",
                    f"partial_blocks={row.get('partially_served_blocks', 0)}",
                    f"zero_service_blocks={row.get('zero_service_blocks', 0)}",
                ]
            )
        lines.append(" | ".join(parts))
    return lines


def run_convergence_experiment(
    cfg_name: str,
    seed: int,
    verbose: bool = True,
    *,
    output_root: str | None = None,
) -> dict:
    """Execute and persist one complete downlink convergence experiment.

    What: validate the config, seed every random source, construct the system and
    scenario, measure the common random-precoder baseline, dispatch the selected
    optimizer, and compute the final latency, service, link-quality, convergence,
    and cost summaries. It then writes the manifest, text/JSON data, and plots to the
    content-hashed result directory.

    Why: method implementations should solve communication problems, not duplicate
    experiment setup or reporting. This orchestration boundary guarantees that two
    methods run with the same channel realization and accounting conventions.

    Returns: the in-memory result record exactly corresponding to the persisted run.
    """
    # ALGORITHM 1: Load one deterministic, validated experiment definition.
    configure_determinism(seed)
    run_started_at_local = current_local_timestamp()
    system_params, sim_params, run_meta = load_config(cfg_name)
    # SETUP: Build a content-addressed result identity; this does not affect optimization.
    objective_mode_tag = get_convergence_objective_name(sim_params)
    result_tag = build_result_tag(
        METHOD_NAME,
        run_meta["cfg_stem"],
        seed,
        objective_mode=objective_mode_tag,
        model_scope=sim_params.get("downlink_precoder_net_scope"),
        solver_mode=sim_params.get("convergence_precoder_update_mode"),
        cfg_hash=run_meta.get("cfg_hash"),
    )
    if output_root is None:
        output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", result_tag)
    output_dirs = initialize_output_dirs(output_root)

    # ALGORITHM 2: Construct the seeded channel system and run the complete allocator.
    # The delegated call measures baselines, dispatches payload/streaming, optimizes F, and searches n_kl.
    core_start = perf_counter()
    system = DownlinkSystem(system_params, seed=seed)
    result = optimize_downlink_transmission(system, sim_params, verbose=verbose)
    core_wall_time_seconds_total = perf_counter() - core_start
    # RESULTS: Attach reproducibility metadata, final metrics, and computational cost.
    result["cfg_path"] = run_meta["cfg_path"]
    result["seed"] = int(seed)
    result["system_params"] = system_params
    result["sim_params"] = sim_params
    result["summary_metrics"] = compute_downlink_summary_metrics(result)
    result["experiment_cost"] = build_downlink_convergence_cost(
        system_params,
        sim_params,
        result,
        core_wall_time_seconds_total=core_wall_time_seconds_total,
    )
    result["run_started_at_local"] = str(run_started_at_local)
    result["run_completed_at_local"] = current_local_timestamp()

    # OUTPUT 1: Generate figures from the completed result; plotting cannot change the solution.
    plot_user_config(system_params, output_dirs["user_config"])
    plot_latency(result, output_dirs["latency_asynchronality"])
    plot_asynchronality_comparison(result, output_dirs["latency_asynchronality"])
    plot_link_quality(result, output_dirs["link_quality"])
    plot_blocks(result, output_dirs["schedule_details"])
    plot_rate_violation_heatmap(result, output_dirs["optimization_history"])
    plot_optimization_history(result, output_dirs["optimization_history"])
    plot_kkt_residual_history(result, output_dirs["optimization_history"])
    plot_per_user_schedule_details(result, output_dirs["schedule_details"])
    plot_per_user_convergence(result, output_dirs["optimization_history"])
    plot_blocklength_feasibility_curves(system, result, output_dirs["optimization_history"])
    plot_payload_rfbl_vs_n_with_epoch(result, output_dirs["optimization_history"])
    plot_interference_before_after_heatmaps(result, output_dirs["interference"])
    plot_per_user_interference_before_after(result, output_dirs["interference"])
    plot_interference_heatmaps(system, output_dirs["interference"])
    plot_per_user_interference_profiles(system, output_dirs["interference"])

    # OUTPUT 2: Save the canonical machine-readable result before deriving its text summary.
    save_json(result, os.path.join(output_dirs["data"], "result.json"))

    metrics = result["summary_metrics"]
    # OUTPUT 3: Build the human-readable summary from the same canonical result record.
    lines = [
        "Downlink optimizer summary",
        "",
        "Setup",
        f"Method: {METHOD_NAME}",
        f"Config: {run_meta['cfg_path']}",
        f"Seed: {seed}",
        f"Run started at: {result.get('run_started_at_local', 'unknown')}",
        f"Run completed at: {result.get('run_completed_at_local', 'unknown')}",
        f"Scenario: {result.get('scenario_mode', result.get('allocation_mode', 'unknown'))}",
        f"Objective mode: {result.get('objective_mode', 'unknown')}",
        f"Allocation mode: {result.get('allocation_mode', 'unknown')}",
        f"Weight strategy: {result.get('weight_strategy', 'n/a')}",
        f"Convergence precoder update mode: {result.get('convergence_precoder_update_mode', 'unknown')}",
        f"Downlink precoder-net scope: {result.get('downlink_precoder_net_scope', 'unknown')}",
        f"Precoder parameterization: {result.get('precoder_parameterization', 'unknown')}",
        f"Initial schedule source: {result.get('initial_schedule_source', 'random_precoder_baseline')}",
    ]
    lines.extend([""])
    lines.extend(_build_downlink_final_test_section_lines(result))
    lines.extend([""])
    lines.extend(_build_downlink_per_user_test_lines(result))

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

    lines.extend(format_experiment_cost_lines(result.get("experiment_cost")))
    save_text(lines, os.path.join(output_dirs["data"], "summary.txt"))
    result_root = os.path.dirname(output_root) if os.path.basename(output_root) == "testing" else output_root
    # OUTPUT 4: Write the compact run manifest used to discover and compare experiments.
    write_result_manifest(
        result_root,
        setup={
            "link": "Downlink",
            "method": METHOD_NAME,
            "scenario": result.get("scenario_mode", result.get("allocation_mode", "unknown")),
            "config_path": run_meta["cfg_path"],
            "config_hash": run_meta.get("cfg_hash"),
            "seed": int(seed),
            "run_started_at_local": result.get("run_started_at_local"),
            "run_completed_at_local": result.get("run_completed_at_local"),
        },
    )
    print(f"Saved downlink results to: {output_root}")
    return result


def main() -> None:
    """Parse CLI arguments and run one complete downlink convergence experiment."""
    # ENTRY 1: Parse command-line options; no optimization is performed in this section.
    parser = argparse.ArgumentParser(description="Downlink online convergence baseline")
    parser.add_argument("--cfg_name", default="downlink_dispersion_heavy.yaml", help="Configuration file name or path")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic random seed")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
    args = parser.parse_args()

    # ENTRY 2: Resolve the content-addressed output path from the chosen configuration.
    _, sim_params, run_meta = load_config(args.cfg_name)
    objective_mode = get_convergence_objective_name(sim_params)
    result_tag = build_result_tag(
        METHOD_NAME,
        run_meta["cfg_stem"],
        int(args.seed),
        objective_mode=objective_mode,
        model_scope=sim_params.get("downlink_precoder_net_scope"),
        solver_mode=sim_params.get("convergence_precoder_update_mode"),
        cfg_hash=run_meta.get("cfg_hash"),
    )
    output_dirs = build_downlink_convergence_result_dirs(
        METHOD_LABEL,
        result_tag,
        scenario_mode=str(sim_params["experiment_scenario_mode"]),
    )
    # ENTRY 3: Hand control to the complete convergence experiment orchestrator above.
    run_convergence_experiment(
        args.cfg_name,
        int(args.seed),
        verbose=not args.quiet,
        output_root=output_dirs["testing_root"],
    )
    print(f"Saved downlink convergence results to: {output_dirs['experiment_root']}")


if __name__ == "__main__":
    main()
