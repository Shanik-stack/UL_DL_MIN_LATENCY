import argparse
import contextlib
import os
from time import perf_counter

import numpy as np


from latency_optimization.experiments.cost import build_uplink_convergence_cost
from latency_optimization.experiments.configuration import load_config_document
from latency_optimization.core.scenarios import STREAMING_MODE
from latency_optimization.results.console import format_latency_log_line
from latency_optimization.results.naming import (
    format_method_tag,
    format_objective_tag,
    format_update_mode_tag,
    join_tag_parts,
    make_method_result_tag,
)
from latency_optimization.results.paths import build_uplink_convergence_result_dirs
from latency_optimization.results.persistence import (
    current_local_timestamp,
    save_json,
    save_text,
    write_result_manifest,
)

from ..config import load_config
from ..objective_settings import validate_uplink_objective_mode
from .solver import validate_convergence_precoder_update_mode
from ..reporting import (
    build_convergence_result,
    build_convergence_summary_lines,
    build_dispersion_diagnostic_lines,
)
from ..result_writer import save_test_results_to_txt
from ..simulation import (
    apply_training_solution,
    estimate_initial_random_precoder_payload_schedule,
    estimate_initial_random_precoder_streaming_schedule,
)
from ..system import UplinkSystem
from ..plotting import (
    plot_F_vs_n_for_all_subblocks,
    plot_interference_before_after_heatmaps,
    plot_interference_heatmaps,
    plot_kkt_residual_history,
    plot_latency_and_asynchronality_from_json,
    plot_link_quality_from_json,
    plot_optimization_result,
    plot_optimization_result_summary_dict,
    plot_per_user_schedule_details,
    plot_per_user_interference_before_after,
    plot_per_user_interference_profiles,
    plot_user_config,
)
from .allocation import run_convergence_baseline

METHOD_NAME = "convergence_per_epoch_baseline"
METHOD_LABEL = "Convergence per epoch"


def run_convergence_experiment(
    cfg_name: str,
    seed: int,
    *,
    do_plots: bool = True,
    plot_output_dirs: dict[str, str] | None = None,
) -> dict:
    """Run one complete uplink convergence experiment from baseline through reporting.

    This orchestration boundary loads config, preserves the common initial state, optimizes, and writes plots.
    """
    run_started_at_local = current_local_timestamp()
    system_params, sim_cfg, _ = load_config(cfg_name)
    sim_cfg = dict(sim_cfg)
    core_start = perf_counter()

    baseline_builder = (
        estimate_initial_random_precoder_streaming_schedule
        if str(sim_cfg["experiment_scenario_mode"]) == STREAMING_MODE
        else estimate_initial_random_precoder_payload_schedule
    )
    initial_baseline = baseline_builder(
        system_params,
        sim_cfg,
        seed=int(seed),
    )
    naive_full_t_baseline = baseline_builder(
        system_params,
        sim_cfg,
        seed=int(seed),
        allow_n_reduction=False,
    )
    print(
        format_latency_log_line(
            "[UL Initial Baseline]",
            initial_baseline["initial_latency"],
            seed=int(seed),
            scenario=str(sim_cfg.get("experiment_scenario_mode", "payload")),
            method="convergence",
        )
    )

    report_system = UplinkSystem(system_params, seed=int(seed))
    initial_snr_db = list(initial_baseline["initial_snr_db"])
    initial_sinr_db = list(initial_baseline["initial_sinr_db"])

    if do_plots:
        if plot_output_dirs is None:
            raise ValueError("plot_output_dirs is required when do_plots is True.")
        plot_params = dict(system_params)
        plot_params["initial_bits_per_symbol"] = np.asarray(initial_baseline["initial_bits_per_symbol"], dtype=float)
        plot_user_config(
            plot_params,
            plot_output_dirs["user_config"],
            extra_params={
                "measured_snr_db_k": np.asarray(initial_snr_db),
                "measured_sinr_db_k": np.asarray(initial_sinr_db),
            },
        )

    convergence_system = UplinkSystem(system_params, seed=int(seed))
    convergence_data = run_convergence_baseline(
        uplinksystem=convergence_system,
        sim_cfg=sim_cfg,
    )

    apply_training_solution(report_system, convergence_data["n_star"], convergence_data["F_star"])

    result = build_convergence_result(
        report_system,
        convergence_data,
        method_name=METHOD_NAME,
        cfg_path=str(load_config_document(cfg_name).path),
        seed=int(seed),
        initial_R_fbl=[np.array(v, copy=True) for v in initial_baseline["initial_R_fbl"]],
        initial_n_kl=[list(values) for values in initial_baseline["initial_n_kl"]],
        initial_n=list(initial_baseline["initial_n"]),
        initial_latency=list(initial_baseline["initial_latency"]),
        initial_snr_db=initial_snr_db,
        initial_sinr_db=initial_sinr_db,
        initial_bits_per_symbol=list(initial_baseline["initial_bits_per_symbol"]),
        initial_B_kl=[list(values) for values in initial_baseline["initial_B_kl"]],
        initial_bits_per_symbol_by_block=[
            list(values) for values in initial_baseline["initial_bits_per_symbol_by_block"]
        ],
        initial_interference_diag=initial_baseline.get("initial_interference_diag"),
        sim_cfg=sim_cfg,
        naive_full_t_baseline=naive_full_t_baseline,
    )
    result["scenario_mode"] = convergence_data.get(
        "scenario_mode",
        sim_cfg.get("experiment_scenario_mode", "payload"),
    )
    result["experiment_cost"] = build_uplink_convergence_cost(
        system_params,
        convergence_data,
        core_wall_time_seconds_total=perf_counter() - core_start,
    )
    result["run_started_at_local"] = str(run_started_at_local)
    result["run_completed_at_local"] = current_local_timestamp()
    return {
        "result": result,
        "report_system": report_system,
        "convergence_data": convergence_data,
        "initial_baseline": initial_baseline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Uplink online convergence baseline")
    parser.add_argument("--cfg_name", type=str, default="uplink_dispersion_heavy.yaml", help="Configuration file name or path")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic random seed")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
    args = parser.parse_args()

    run_seed = int(args.seed)
    _, sim_cfg, run_meta = load_config(args.cfg_name)
    update_mode = validate_convergence_precoder_update_mode(
        sim_cfg.get("convergence_precoder_update_mode", "precoder_net")
    )
    objective_mode = validate_uplink_objective_mode(
        sim_cfg.get("uplink_objective_mode", "unweighted_sum_rate")
    )
    result_tag = make_method_result_tag(
        join_tag_parts(
            format_method_tag(METHOD_NAME),
            format_objective_tag(objective_mode),
            format_update_mode_tag(update_mode),
        ),
        run_meta["cfg_stem"],
        seed=run_seed,
        cfg_hash=run_meta.get("cfg_hash"),
    )
    result_dirs = build_uplink_convergence_result_dirs(
        METHOD_LABEL,
        result_tag,
        scenario_mode=str(sim_cfg["experiment_scenario_mode"]),
    )
    with contextlib.ExitStack() as stack:
        if args.quiet:
            output_sink = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
            stack.enter_context(contextlib.redirect_stdout(output_sink))
        experiment = run_convergence_experiment(
            cfg_name=args.cfg_name,
            seed=run_seed,
            do_plots=True,
            plot_output_dirs=result_dirs,
        )
    result = experiment["result"]
    result["cfg_hash"] = run_meta.get("cfg_hash")
    report_system = experiment["report_system"]
    convergence_data = experiment["convergence_data"]
    initial_baseline = experiment["initial_baseline"]

    convergence_plot_dict = {
        "n_star": convergence_data["n_star"],
        "R_star": convergence_data["R_star"],
        "all_user_block_results": convergence_data["all_user_block_results_train"],
    }
    convergence_raw_dict = {
        "B_kl_star_test": convergence_data["B_kl_star"],
        "n_star_test": convergence_data["n_star"],
        "R_star_test": convergence_data["R_star"],
        "all_user_block_results_test": convergence_data["all_user_block_results_train"],
        "scenario_mode": convergence_data.get("scenario_mode", ""),
        "scenario_block_targets": convergence_data.get("scenario_block_targets", []),
    }

    save_test_results_to_txt(
        test_uplinksystem=report_system,
        test_data_dict=convergence_raw_dict,
        initial_Rfbl=[np.array(v, copy=True) for v in initial_baseline["initial_R_fbl"]],
        initial_n_kl=[list(values) for values in initial_baseline["initial_n_kl"]],
        initial_n=list(initial_baseline["initial_n"]),
        initial_latency=list(initial_baseline["initial_latency"]),
        initial_snr_db=list(result["initial_snr_db"]),
        initial_sinr_db=list(result["initial_sinr_db"]),
        initial_bits_per_symbol=list(initial_baseline["initial_bits_per_symbol"]),
        save_dir=result_dirs["data"],
        filename="convergence_results.txt",
        initial_B_kl=[list(values) for values in initial_baseline["initial_B_kl"]],
        initial_bits_per_symbol_by_block=[
            list(values) for values in initial_baseline["initial_bits_per_symbol_by_block"]
        ],
    )

    plot_optimization_result(
        convergence_data["all_user_block_results_train"],
        train=False,
        save_dir=result_dirs["optimization_history"],
        phase_label="Convergence",
        filename_prefix="convergence",
    )
    plot_optimization_result_summary_dict(
        convergence_plot_dict,
        train=False,
        save_dir=result_dirs["optimization_history"],
        phase_label="Convergence",
        filename_prefix="convergence",
    )
    plot_F_vs_n_for_all_subblocks(
        convergence_data,
        save_dir="F_vs_n",
        base_dir=result_dirs["optimization_history"],
    )
    plot_kkt_residual_history(
        result,
        train=False,
        save_dir=result_dirs["optimization_history"],
        phase_label="Convergence",
        filename_prefix="convergence",
    )
    plot_latency_and_asynchronality_from_json(
        json_path=os.path.join(result_dirs["data"], "convergence_results.json"),
        save_dir=result_dirs["latency_asynchronality"],
        prefix="convergence",
    )
    plot_link_quality_from_json(
        json_path=os.path.join(result_dirs["data"], "convergence_results.json"),
        save_dir=result_dirs["link_quality"],
        prefix="convergence",
    )

    save_json(result, os.path.join(result_dirs["data"], "result.json"))
    plot_per_user_schedule_details(result, result_dirs["schedule_details"])
    plot_interference_before_after_heatmaps(result, result_dirs["interference"])
    plot_per_user_interference_before_after(result, result_dirs["interference"])
    plot_interference_heatmaps(report_system, result_dirs["interference"])
    plot_per_user_interference_profiles(report_system, result_dirs["interference"])
    summary_lines = build_convergence_summary_lines(result)
    summary_lines.insert(5, f"Config content hash: {result.get('cfg_hash', 'unknown')}")
    save_text(summary_lines, os.path.join(result_dirs["data"], "summary.txt"))
    dispersion_lines = build_dispersion_diagnostic_lines(result)
    if dispersion_lines:
        save_text(dispersion_lines, os.path.join(result_dirs["data"], "dispersion_diagnostics.txt"))
    write_result_manifest(
        result_dirs["experiment_root"],
        setup={
            "link": "Uplink",
            "method": METHOD_LABEL,
            "scenario": sim_cfg["experiment_scenario_mode"],
            "config_path": run_meta["cfg_path"],
            "config_hash": run_meta.get("cfg_hash"),
            "seed": int(args.seed),
            "run_started_at_local": result.get("run_started_at_local"),
            "run_completed_at_local": result.get("run_completed_at_local"),
        },
    )
    print(f"Saved uplink convergence results to: {result_dirs['experiment_root']}")


if __name__ == "__main__":
    main()
