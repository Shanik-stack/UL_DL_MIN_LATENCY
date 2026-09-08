"""Run uplink Monte Carlo training and held-out evaluation.

Start at :func:`main` to see dataset construction, network training, checkpoint
handling, test-channel evaluation, and result persistence in execution order.
The individual stages live in the explicitly named sibling modules.
"""

import argparse
import os
from time import perf_counter

import numpy as np
import torch


from latency_optimization.experiments.scenarios import (
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
from latency_optimization.experiments.configuration import load_config_document
from latency_optimization.experiments.cost import (
    build_uplink_monte_carlo_total_cost,
    build_uplink_monte_carlo_training_cost,
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
from latency_optimization.results.console import format_latency_log_line
from latency_optimization.results.naming import (
    format_method_tag,
    format_objective_tag,
    format_training_style_tag,
    join_tag_parts,
    make_method_result_tag,
)
from latency_optimization.results.paths import build_uplink_result_dirs
from latency_optimization.results.persistence import (
    current_local_timestamp,
    save_json,
    save_text,
    write_result_manifest,
)

from ...configuration.loader import load_config
from ...objectives.settings import validate_uplink_objective_mode
from ...results.plotting import (
    plot_F_vs_n_for_all_subblocks,
    plot_interference_before_after_heatmaps,
    plot_interference_heatmaps,
    plot_optimization_result,
    plot_optimization_result_summary_dict,
    plot_latency_and_asynchronality_from_json,
    plot_link_quality_from_json,
    plot_per_user_schedule_details,
    plot_per_user_interference_before_after,
    plot_per_user_interference_profiles,
    plot_user_config,
)
from ...precoders.checkpoints import load_user_precoder_models
from ...results.reporting import (
    build_dispersion_diagnostic_lines,
    build_post_training_summary_lines,
    build_precoder_net_result,
    build_summary_lines,
    build_training_dataset_summary_lines,
)
from ...results.persistence import save_test_results_to_txt
from ...simulation.operations import (
    estimate_initial_random_precoder_payload_schedule,
    estimate_initial_random_precoder_streaming_schedule,
)
from ...simulation.system import UplinkSystem
from .evaluate_precoder_network import (
    evaluate_blocklength_precoder_net,
)
from .train_precoder_network import (
    train_blocklength_aware_precoder_net,
)


def evaluate_trained_precoder_network_on_test_channel(
    train_artifact: dict,
    cfg_name: str,
    test_seed: int,
    *,
    do_plots: bool,
    result_dirs: dict[str, str],
    train_seeds: list[int],
    test_snr_db_by_user: list[float],
    test_search_overrides: dict[str, object] | None = None,
    reused_training_artifact: str | None = None,
):
    """Measure trained uplink networks on one deterministic held-out channel.

    What: rebuild the per-user test channels at the requested seed/SNR values, load
    the training artifact, and perform inference-only payload or streaming allocation
    with the configured blocklength search. The resulting schedule is compared with
    common random and full-T references and passed through standard reporting.

    Why: this is the train/test boundary: model parameters are fixed, the held-out
    seed was absent from training, and test search overrides never update weights.

    Returns: the complete persisted test record, including latency, service,
    link-quality, convergence, timing, and FLOP diagnostics.
    """
    # TEST 1: Recreate one held-out channel and apply only test-time search overrides.
    # No model parameter is updated anywhere in this function.
    configure_determinism(int(test_seed))
    system_params, sim_cfg, _ = load_config(cfg_name)
    system_params = with_monte_carlo_sample_snr_by_user(system_params, test_snr_db_by_user)
    sim_cfg = dict(sim_cfg)
    if str(sim_cfg["experiment_scenario_mode"]) == "streaming":
        sim_cfg["experiment_scenario"] = {
            **sim_cfg["experiment_scenario"],
            "number_of_blocks": 1,
        }
    if test_search_overrides:
        sim_cfg.update(test_search_overrides)
    test_scenario = build_monte_carlo_sample_scenarios_for_seeds(
        system_params,
        sim_cfg,
        [int(test_seed)],
    )[0]
    test_scenario_summary = build_experiment_scenario_summary(test_scenario)
    # TEST 2: Build random adaptive-n and random full-T references on this exact channel.
    baseline_builder = (
        estimate_initial_random_precoder_streaming_schedule
        if str(sim_cfg["experiment_scenario_mode"]) == STREAMING_MODE
        else estimate_initial_random_precoder_payload_schedule
    )
    initial_baseline = baseline_builder(
        system_params,
        sim_cfg,
        seed=int(test_seed),
    )
    naive_full_t_baseline = baseline_builder(
        system_params,
        sim_cfg,
        seed=int(test_seed),
        allow_n_reduction=False,
    )
    print(
        format_latency_log_line(
            "[UL Initial Baseline]",
            initial_baseline["initial_latency"],
            seed=int(test_seed),
            scenario=str(test_scenario.get("mode", sim_cfg.get("experiment_scenario_mode", "unknown"))),
            method="monte_carlo",
        )
    )
    # TEST 3: Construct the simulator that will receive the network-generated schedule.
    test_uplinksystem = UplinkSystem(system_params, seed=int(test_seed))
    initial_snr_db = list(initial_baseline["initial_snr_db"])
    initial_sinr_db = list(initial_baseline["initial_sinr_db"])

    if do_plots:
        plot_params = dict(system_params)
        plot_params["initial_bits_per_symbol"] = np.asarray(initial_baseline["initial_bits_per_symbol"], dtype=float)
        plot_user_config(
            plot_params,
            result_dirs["user_config"],
            extra_params={
                "measured_snr_db_k": np.asarray(initial_snr_db),
                "measured_sinr_db_k": np.asarray(initial_sinr_db),
            },
        )

    initial_Rfbl = [np.array(v, copy=True) for v in initial_baseline["initial_R_fbl"]]
    initial_latency = list(initial_baseline["initial_latency"])
    initial_bits_per_symbol = list(initial_baseline["initial_bits_per_symbol"])
    initial_bits_per_symbol_by_block = [
        list(values) for values in initial_baseline["initial_bits_per_symbol_by_block"]
    ]
    initial_n = list(initial_baseline["initial_n"])
    initial_n_kl = [list(values) for values in initial_baseline["initial_n_kl"]]
    initial_B_kl = [list(values) for values in initial_baseline["initial_B_kl"]]

    # TEST 4: Restore the fixed trained weights, then run inference and n_kl allocation.
    user_models = load_user_precoder_models(
        train_artifact["user_model_specs"],
        train_artifact["user_model_states"],
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    )
    testing_started_at_local = current_local_timestamp()
    core_evaluation_start = perf_counter()
    post_test = evaluate_blocklength_precoder_net(
        uplinksystem=test_uplinksystem,
        user_models=user_models,
        sim_cfg=sim_cfg,
        method_name="monte_carlo_precoder_net_test",
    )
    core_evaluation_wall_time_seconds = perf_counter() - core_evaluation_start
    testing_completed_at_local = current_local_timestamp()

    # TEST 5: Convert inference output into the common schedule schema used by reporting.
    test_data_dict = {
        "L_out_test": post_test["L_out"],
        "n_star_test": post_test["n_star"],
        "F_star_test": post_test["F_star"],
        "R_star_test": post_test["R_star"],
        "all_user_block_results_test": post_test["all_user_block_results_train"],
        "B_used_star_test": post_test["B_used_star"],
        "B_kl_star_test": post_test["B_kl_star"],
        "skipped_blocks_per_user": post_test.get("skipped_blocks_per_user", []),
        "scenario_mode": post_test.get("scenario_mode", ""),
        "scenario_block_targets": post_test.get("scenario_block_targets", []),
        "precoder_parameterization": train_artifact["precoder_parameterization"],
        "user_model_specs": train_artifact["user_model_specs"],
        "user_model_states": train_artifact["user_model_states"],
    }

    # TEST OUTPUT: Persist detailed schedule values and optional diagnostic figures.
    save_test_results_to_txt(
        test_uplinksystem=test_uplinksystem,
        test_data_dict=test_data_dict,
        initial_Rfbl=initial_Rfbl,
        initial_n_kl=initial_n_kl,
        initial_n=initial_n,
        initial_latency=initial_latency,
        initial_snr_db=initial_snr_db,
        initial_sinr_db=initial_sinr_db,
        save_dir=result_dirs["test_data"],
        filename="test_results.txt",
        initial_bits_per_symbol=initial_bits_per_symbol,
        initial_B_kl=initial_B_kl,
        initial_bits_per_symbol_by_block=initial_bits_per_symbol_by_block,
    )

    if do_plots:
        plot_optimization_result(
            test_data_dict["all_user_block_results_test"],
            train=False,
            save_dir=result_dirs["optimization_history"],
            phase_label="Testing",
            filename_prefix="testing",
        )
        plot_optimization_result_summary_dict(
            {
                "n_star": test_data_dict["n_star_test"],
                "R_star": test_data_dict["R_star_test"],
                "all_user_block_results": test_data_dict["all_user_block_results_test"],
            },
            train=False,
            save_dir=result_dirs["optimization_history"],
            phase_label="Testing",
            filename_prefix="testing",
        )
        test_plot_data = dict(test_data_dict)
        test_plot_data["precoder_net_training_history"] = train_artifact.get("precoder_net_training_history", {})
        plot_F_vs_n_for_all_subblocks(
            test_plot_data,
            save_dir="F_vs_n",
            base_dir=result_dirs["optimization_history"],
        )
        if int(test_uplinksystem.K) > 1:
            plot_latency_and_asynchronality_from_json(
                json_path=os.path.join(result_dirs["test_data"], "test_results.json"),
                save_dir=result_dirs["latency_asynchronality"],
                prefix="test",
            )
        plot_link_quality_from_json(
            json_path=os.path.join(result_dirs["test_data"], "test_results.json"),
            save_dir=result_dirs["link_quality"],
            prefix="test",
        )

    # TEST 6: Build the canonical result with baseline comparisons and evaluation cost.
    result = build_precoder_net_result(
        test_uplinksystem,
        test_data_dict,
        method_name="monte_carlo_precoder_net_train_test",
        cfg_path=str(load_config_document(cfg_name).path),
        test_seed=int(test_seed),
        train_seeds=train_seeds,
        train_artifact=train_artifact,
        initial_R_fbl=initial_Rfbl,
        initial_n_kl=initial_n_kl,
        initial_n=initial_n,
        initial_latency=initial_latency,
        initial_snr_db=initial_snr_db,
        initial_sinr_db=initial_sinr_db,
        initial_bits_per_symbol=initial_bits_per_symbol,
        initial_B_kl=initial_B_kl,
        initial_bits_per_symbol_by_block=initial_bits_per_symbol_by_block,
        initial_interference_diag=initial_baseline.get("initial_interference_diag"),
        uplink_rate_model=sim_cfg.get("uplink_rate_model", "unknown"),
        naive_full_t_baseline=naive_full_t_baseline,
    )
    result["experiment_scenario_mode"] = sim_cfg.get("experiment_scenario_mode", "payload")
    result["test_snr_db_by_user"] = [float(value) for value in test_snr_db_by_user]
    result["experiment_scenario"] = test_scenario_summary
    result["test_n_search_strategy"] = str(
        sim_cfg.get("monte_carlo_test_n_search_strategy", sim_cfg.get("n_search_strategy", "unknown"))
    )
    result["test_n_search_direction"] = str(
        sim_cfg.get("monte_carlo_test_n_search_direction", sim_cfg.get("n_search_direction", "unknown"))
    )
    if reused_training_artifact:
        result["reused_training_artifact"] = str(reused_training_artifact)
    test_candidate_n_states_per_user = [
        int(sum(len(block_states) for block_states in user_blocks))
        for user_blocks in test_data_dict["all_user_block_results_test"]
    ]
    raw_evaluation_counters = post_test.get("evaluation_cost_counters", {})
    if not isinstance(raw_evaluation_counters, dict):
        raw_evaluation_counters = {}
    result["evaluation_cost_counters"] = {
        "per_user_forward_calls": [
            int(v)
            for v in raw_evaluation_counters.get(
                "per_user_forward_calls",
                [0 for _ in range(int(test_uplinksystem.K))],
            )
        ],
        "total_forward_calls": int(raw_evaluation_counters.get("total_forward_calls", 0)),
        "per_user_candidate_n_states": test_candidate_n_states_per_user,
        "total_candidate_n_states": int(sum(test_candidate_n_states_per_user)),
    }
    result["core_evaluation_wall_time_seconds"] = float(core_evaluation_wall_time_seconds)
    result["testing_started_at_local"] = str(testing_started_at_local)
    result["testing_completed_at_local"] = str(testing_completed_at_local)
    # TEST OUTPUT: Save the canonical test result, summary, and scenario description.
    save_json(result, os.path.join(result_dirs["test_data"], "result.json"))
    if do_plots:
        plot_per_user_schedule_details(result, result_dirs["schedule_details"])
        plot_interference_before_after_heatmaps(result, result_dirs["interference"])
        plot_per_user_interference_before_after(result, result_dirs["interference"])
        plot_interference_heatmaps(test_uplinksystem, result_dirs["interference"])
        plot_per_user_interference_profiles(test_uplinksystem, result_dirs["interference"])
    save_text(build_summary_lines(result), os.path.join(result_dirs["test_data"], "summary.txt"))
    dispersion_lines = build_dispersion_diagnostic_lines(result)
    if dispersion_lines:
        save_text(dispersion_lines, os.path.join(result_dirs["test_data"], "dispersion_diagnostics.txt"))
    save_json(test_scenario_summary, os.path.join(result_dirs["test_data"], "experiment_scenario.json"))
    save_text(
        build_experiment_scenario_summary_lines(test_scenario_summary),
        os.path.join(result_dirs["test_data"], "experiment_scenario.txt"),
    )
    return result


def main():
    """Run Monte Carlo dataset setup, training, held-out testing, and persistence."""
    # ENTRY 1: Parse experiment-size, optimizer, checkpoint, and test-search options.
    parser = argparse.ArgumentParser(description="Offline Monte Carlo precoder-net train/test")
    parser.add_argument("--cfg_name", type=str, default="uplink_dispersion_heavy.yaml", help="Configuration file name or path")
    parser.add_argument("--train_seeds", type=str, default=None, help="Explicit comma-separated training seeds")
    parser.add_argument("--num_train_channels", type=int, default=None, help="Payload: number of training channel episodes")
    parser.add_argument("--num_train_blocks", type=int, default=None, help="Streaming: number of independent training blocks")
    parser.add_argument("--test_seed", type=int, default=None, help="Monte Carlo test seed")
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
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()

    # ENTRY 2: Validate whether sample counts represent payload channels or streaming blocks.
    system_params, sim_cfg, run_meta = load_config(args.cfg_name)
    scenario_mode = str(sim_cfg["experiment_scenario_mode"])
    if scenario_mode == "payload":
        if args.num_train_blocks is not None or args.num_test_blocks is not None:
            raise ValueError("Payload Monte Carlo uses channel-count options, not block-count options.")
        cli_training_samples = args.num_train_channels
        config_training_samples = sim_cfg.get("monte_carlo_num_training_channels")
        cli_test_samples = args.num_test_channels
        config_test_samples = sim_cfg.get("monte_carlo_num_test_channels")
    else:
        if args.num_train_channels is not None or args.num_test_channels is not None:
            raise ValueError("Streaming Monte Carlo uses block-count options, not channel-count options.")
        cli_training_samples = args.num_train_blocks
        config_training_samples = sim_cfg.get("monte_carlo_num_training_blocks")
        cli_test_samples = args.num_test_blocks
        config_test_samples = sim_cfg.get("monte_carlo_num_test_blocks")
    if cli_training_samples is None and config_training_samples is None:
        raise ValueError(f"Monte Carlo training sample count is missing for {scenario_mode}.")
    if cli_test_samples is None and config_test_samples is None:
        raise ValueError(f"Monte Carlo test sample count is missing for {scenario_mode}.")
    run_started_at_local = current_local_timestamp()
    test_search_overrides = _build_test_search_overrides(args)
    train_epochs = int(
        args.precoder_net_epochs
        if args.precoder_net_epochs is not None
        else sim_cfg.get("monte_carlo_training_max_epochs", sim_cfg.get("max_epochs", 100))
    )
    # DATA 1: Resolve disjoint training/test seeds and deterministic per-user SNR samples.
    train_seeds, test_seed = resolve_monte_carlo_train_and_test_seeds(
        cli_train_seeds=args.train_seeds,
        cli_num_training_samples=cli_training_samples,
        cli_test_seed=args.test_seed,
        config_train_seeds=sim_cfg.get("monte_carlo_train_seeds"),
        config_num_training_samples=config_training_samples,
        config_test_seed=sim_cfg.get("monte_carlo_test_seed"),
    )
    test_seeds = build_test_seeds_from_num_test_samples(
        int(cli_test_samples if cli_test_samples is not None else config_test_samples),
        first_test_seed=int(test_seed),
        excluded_seeds=train_seeds,
    )
    test_snr_db_by_user_by_seed = build_test_snr_schedule(
        test_seeds,
        sim_cfg["monte_carlo_test_snr_db_ranges"],
        num_users=int(system_params["K"]),
    )
    # DATA 2: Optionally restore a completed training artifact for inference-only reruns.
    train_artifact: dict[str, object] | None = None
    reused_training_artifact: str | None = None
    if args.reuse_train_artifact:
        reused_training_artifact = os.path.abspath(args.reuse_train_artifact)
        train_artifact = torch.load(reused_training_artifact, map_location="cpu", weights_only=False)
        if not isinstance(train_artifact, dict):
            raise TypeError("Expected a dictionary train artifact when reusing Monte Carlo training.")
        artifact_train_seeds = [int(v) for v in train_artifact.get("train_seeds", [])]
        if artifact_train_seeds:
            train_seeds = artifact_train_seeds
    configure_determinism(train_seeds[0] if train_seeds else 0)
    training_snr_db_by_user_by_seed = build_training_snr_schedule(
        train_seeds,
        sim_cfg["monte_carlo_training_snr_db_ranges"],
        num_users=int(system_params["K"]),
    )
    print(f"Resolved Monte Carlo train seeds: {train_seeds}")
    print(f"Monte Carlo training sample unit: {sim_cfg['monte_carlo_sample_unit']}")
    print(f"Resolved Monte Carlo test seed: {int(test_seed)}")
    print(f"Resolved Monte Carlo test dataset seeds: {test_seeds}")
    # DATA 3: Materialize the exact scenario description associated with every training seed.
    training_scenario_summaries = []
    for seed, scenario in zip(
        train_seeds,
        build_monte_carlo_sample_scenarios_for_seeds(system_params, sim_cfg, train_seeds),
    ):
        summary = build_experiment_scenario_summary(scenario)
        summary["training_snr_db_by_user"] = list(training_snr_db_by_user_by_seed[int(seed)])
        training_scenario_summaries.append(summary)
    if isinstance(train_artifact, dict) and train_artifact.get("training_experiment_scenarios"):
        training_scenario_summaries = list(train_artifact["training_experiment_scenarios"])
    # SETUP: Build the content-addressed experiment name and result directory tree.
    objective_mode = validate_uplink_objective_mode(
        sim_cfg.get("uplink_objective_mode", "unweighted_sum_rate")
    )
    training_style_name = str(
        train_artifact.get("monte_carlo_training_style", "rollout_query_objective")
        if isinstance(train_artifact, dict)
        else "rollout_query_objective"
    )
    result_tag = make_method_result_tag(
        join_tag_parts(
            format_method_tag("monte_carlo_precoder_net_train_test"),
            format_objective_tag(objective_mode),
            format_training_style_tag(training_style_name),
            f"testset{len(test_seeds)}",
            _build_test_search_tag(test_search_overrides),
        ),
        run_meta["cfg_stem"],
        seed=int(test_seed),
        cfg_hash=run_meta.get("cfg_hash"),
    )
    result_dirs = build_uplink_result_dirs(
        "Monte Carlo",
        result_tag,
        scenario_mode=str(sim_cfg["experiment_scenario_mode"]),
    )
    # DATA 4: Save reproducible training and held-out channel/block manifests.
    training_sample_manifest = save_monte_carlo_base_samples(
        output_dir=result_dirs["train_samples"],
        link_name="Uplink",
        system_params=system_params,
        sample_seeds=train_seeds,
        system_factory=UplinkSystem,
        sample_snr_db_by_user_by_seed=training_snr_db_by_user_by_seed,
        dataset_role="training",
        sample_kind="payload_channel_episode" if str(sim_cfg["experiment_scenario_mode"]) == "payload" else "streaming_block",
    )
    test_sample_manifest = save_monte_carlo_base_samples(
        output_dir=result_dirs["test_samples"],
        link_name="Uplink test",
        system_params=system_params,
        sample_seeds=test_seeds,
        system_factory=UplinkSystem,
        sample_snr_db_by_user_by_seed=test_snr_db_by_user_by_seed,
        dataset_role="test",
        sample_kind="payload_channel_episode" if str(sim_cfg["experiment_scenario_mode"]) == "payload" else "streaming_block",
    )

    # TRAINING 1: Optimize one persistent precoder network per uplink user.
    # This delegated function builds rollout states, evaluates their FBL objective,
    # backpropagates through each user's network, and returns trained weights plus history.
    if train_artifact is None:
        training_started_at_local = current_local_timestamp()
        training_start = perf_counter()
        train_artifact = train_blocklength_aware_precoder_net(
            cfg_name=args.cfg_name,
            train_seeds=train_seeds,
            epochs=train_epochs,
            batch_size=args.precoder_net_batch_size,
            lr=args.precoder_net_lr,
        )
        training_wall_time_seconds = perf_counter() - training_start
        training_completed_at_local = current_local_timestamp()
        train_artifact["cfg_path"] = str(load_config_document(args.cfg_name).path)
        train_artifact["cfg_hash"] = run_meta.get("cfg_hash")
        train_artifact["method_name"] = "monte_carlo_precoder_net_train_test"
        train_artifact["test_seed"] = int(test_seed)
        train_artifact["experiment_scenario_mode"] = sim_cfg.get("experiment_scenario_mode", "payload")
        train_artifact["training_experiment_scenarios"] = training_scenario_summaries
        train_artifact["training_snr_db_by_user_by_seed"] = training_snr_db_by_user_by_seed
        train_artifact["training_sample_manifest"] = training_sample_manifest
        training_cost = build_uplink_monte_carlo_training_cost(
            train_artifact,
            batch_size=args.precoder_net_batch_size,
            core_wall_time_seconds_training=training_wall_time_seconds,
        )
        train_artifact["experiment_cost"] = training_cost
        if isinstance(train_artifact.get("post_training_summary"), dict):
            train_artifact["post_training_summary"]["run_started_at_local"] = str(run_started_at_local)
            train_artifact["post_training_summary"]["training_started_at_local"] = str(training_started_at_local)
            train_artifact["post_training_summary"]["training_completed_at_local"] = str(training_completed_at_local)
            train_artifact["post_training_summary"]["experiment_cost"] = training_cost

        # TRAINING OUTPUT: Save learning diagnostics and the reusable model artifact.
        plot_optimization_result(
            train_artifact["all_user_block_results_train"],
            train=True,
            save_dir=result_dirs["train_evaluation"],
            phase_label="Training-sample evaluation",
            filename_prefix="training_evaluation",
        )
        plot_optimization_result_summary_dict(
            train_artifact,
            train=True,
            save_dir=result_dirs["train_optimization_history"],
            phase_label="Training",
            filename_prefix="training",
        )
        plot_F_vs_n_for_all_subblocks(
            train_artifact,
            base_dir=result_dirs["train_evaluation"],
        )

        torch.save(train_artifact, os.path.join(result_dirs["train_data"], "train_artifact.pt"))
        save_json(
            train_artifact.get("training_dataset_summary", {}),
            os.path.join(result_dirs["train_data"], "training_dataset_summary.json"),
        )
        save_text(
            build_training_dataset_summary_lines(train_artifact.get("training_dataset_summary", {})),
            os.path.join(result_dirs["train_data"], "training_dataset_summary.txt"),
        )
        save_json(
            train_artifact.get("post_training_summary", {}),
            os.path.join(result_dirs["train_data"], "post_training_summary.json"),
        )
        save_text(
            build_post_training_summary_lines(train_artifact.get("post_training_summary", {})),
            os.path.join(result_dirs["train_data"], "post_training_summary.txt"),
        )
        save_json(
            {"seed_scenarios": training_scenario_summaries},
            os.path.join(result_dirs["train_data"], "experiment_scenarios.json"),
        )
        save_text(
            _build_seeded_scenario_collection_lines(
                training_scenario_summaries,
                title="Training experiment scenarios by seed",
            ),
            os.path.join(result_dirs["train_data"], "experiment_scenarios.txt"),
        )
    else:
        # TRAINING REUSE: Preserve original training cost/timestamps and skip backpropagation.
        print(f"Reusing Monte Carlo training artifact: {reused_training_artifact}")
        prior_cost = train_artifact.get("experiment_cost", {})
        if not isinstance(prior_cost, dict):
            prior_cost = {}
        training_wall_time_seconds = float(prior_cost.get("core_wall_time_seconds_training", 0.0))
        prior_training_summary = train_artifact.get("post_training_summary", {})
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
            os.path.join(result_dirs["train_data"], "reused_training_artifact.json"),
        )
        save_text(
            [
                "This Monte Carlo run reused an existing training artifact and reran only the test phase.",
                f"Source artifact: {reused_training_artifact}",
                f"Train seeds: {train_seeds}",
                f"Test seed: {int(test_seed)}",
                f"Test n-search overrides: {test_search_overrides}",
            ],
            os.path.join(result_dirs["train_data"], "reused_training_artifact.txt"),
        )

    # TESTING: Evaluate the same fixed networks independently on every held-out seed.
    # The helper above performs inference, blocklength search, metrics, plots, and persistence.
    if not args.skip_test:
        test_results: list[dict] = []
        for episode_seed in test_seeds:
            sample_dirs = build_test_sample_dirs(result_dirs, int(episode_seed), link="uplink")
            test_result = evaluate_trained_precoder_network_on_test_channel(
                train_artifact=train_artifact,
                cfg_name=args.cfg_name,
                test_seed=int(episode_seed),
                # Every held-out channel gets its own diagnostics.  Keeping only
                # the first episode made the remaining episode folders appear empty.
                do_plots=True,
                result_dirs=sample_dirs,
                train_seeds=train_seeds,
                test_snr_db_by_user=test_snr_db_by_user_by_seed[int(episode_seed)],
                test_search_overrides=test_search_overrides,
                reused_training_artifact=reused_training_artifact,
            )
            testing_wall_time_seconds = float(test_result.get("core_evaluation_wall_time_seconds", 0.0))
            testing_completed_at_local = current_local_timestamp()
            test_result["experiment_cost"] = build_uplink_monte_carlo_total_cost(
                train_artifact,
                test_result.get("evaluation_cost_counters", {}),
                batch_size=args.precoder_net_batch_size,
                core_wall_time_seconds_training=training_wall_time_seconds,
                core_wall_time_seconds_testing=testing_wall_time_seconds,
            )
            test_result["cfg_hash"] = run_meta.get("cfg_hash")
            test_result["run_started_at_local"] = str(run_started_at_local)
            test_result["run_completed_at_local"] = str(testing_completed_at_local)
            test_result["training_started_at_local"] = str(training_started_at_local)
            test_result["training_completed_at_local"] = str(training_completed_at_local)
            save_json(test_result, os.path.join(sample_dirs["test_data"], "result.json"))
            save_text(build_summary_lines(test_result), os.path.join(sample_dirs["test_data"], "summary.txt"))
            test_results.append(test_result)
        # TEST OUTPUT: Aggregate per-seed results without replacing their detailed records.
        test_dataset_summary = _build_test_dataset_summary(
            test_results,
            test_snr_db_by_user_by_seed,
        )
        test_dataset_summary["sample_manifest"] = test_sample_manifest
        save_json(test_dataset_summary, os.path.join(result_dirs["test_data"], "test_dataset_summary.json"))
        save_text(
            [
                "Uplink Monte Carlo test dataset summary",
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
            os.path.join(result_dirs["test_data"], "test_dataset_summary.txt"),
        )
    # FINAL OUTPUT: Write the experiment-level manifest used for discovery and comparison.
    write_result_manifest(
        result_dirs["experiment_root"],
        setup={
            "link": "Uplink",
            "method": "Monte Carlo",
            "scenario": sim_cfg["experiment_scenario_mode"],
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
    print(f"Saved uplink Monte Carlo results to: {result_dirs['experiment_root']}")


if __name__ == "__main__":
    main()
