from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from latency_optimization.experiments.scenarios import build_experiment_scenario_summary_lines
from latency_optimization.optimization.blocklength_search import validate_n_search_direction, validate_n_search_strategy
from latency_optimization.experiments.config_validation import require_choice
from latency_optimization.results.naming import join_tag_parts


def build_test_search_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.test_n_search_strategy:
        overrides["monte_carlo_test_n_search_strategy"] = str(args.test_n_search_strategy)
    if args.test_n_search_direction:
        overrides["monte_carlo_test_n_search_direction"] = str(args.test_n_search_direction)
    if args.test_n_search_coarse_step is not None:
        overrides["monte_carlo_test_n_search_coarse_step"] = int(args.test_n_search_coarse_step)
    if args.test_n_search_exponential_factor is not None:
        overrides["monte_carlo_test_n_search_exponential_factor"] = int(
            args.test_n_search_exponential_factor
        )
    return overrides


def build_test_search_tag(overrides: dict[str, object]) -> str | None:
    if not overrides:
        return None
    strategy = overrides.get("monte_carlo_test_n_search_strategy", "")
    direction = overrides.get("monte_carlo_test_n_search_direction", "")
    if strategy:
        strategy = validate_n_search_strategy(strategy)
    if direction:
        direction = validate_n_search_direction(direction)
    if strategy == "binary":
        return "ntest_bin"
    if strategy == "fixed_step":
        if direction == "ascending":
            return "ntest_asc"
        if direction == "descending":
            return "ntest_desc"
    return join_tag_parts("ntest", strategy or None, direction or None)


def build_test_sample_dirs(
    result_dirs: dict[str, str],
    test_seed: int,
    *,
    link: str,
) -> dict[str, str]:
    root = Path(result_dirs["testing_root"]) / "samples" / f"seed_{int(test_seed):04d}"
    require_choice(link, {"uplink", "downlink"}, "link")
    dirs = {
        "testing_root": root,
        "test_data": root / "data",
        "user_config": root / "user_config",
        "latency_asynchronality": root / "latency_asynchronality",
        "link_quality": root / "link_quality",
        "optimization_history": root / "optimization_history",
        "schedule_details": root / "schedule_details",
        "interference": root / "interference",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return {key: str(value) for key, value in dirs.items()}


def build_test_dataset_summary(
    test_results: list[dict[str, Any]],
    test_snr_db_by_user_by_seed: dict[int, list[float]],
) -> dict[str, Any]:
    def mean_std(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0}
        mean = float(sum(values) / len(values))
        variance = float(sum((value - mean) ** 2 for value in values) / len(values))
        return {"mean": mean, "std": variance ** 0.5}

    samples: list[dict[str, Any]] = []
    final_latency: list[float] = []
    latency_reduction: list[float] = []
    asynchronality: list[float] = []
    for result in test_results:
        metrics = result.get("summary_metrics", {})
        if not isinstance(metrics, dict):
            continue
        seed = int(result["seed"])
        final_latency.append(float(metrics.get("final_total_latency", 0.0)))
        latency_reduction.append(float(metrics.get("total_latency_reduction_percent", 0.0)))
        asynchronality.append(float(metrics.get("final_asynchronality_sum", 0.0)))
        samples.append(
            {
                "seed": seed,
                "snr_db_by_user": test_snr_db_by_user_by_seed[seed],
                "final_total_latency_seconds": final_latency[-1],
                "latency_reduction_vs_random_percent": latency_reduction[-1],
                "final_asynchronality_sum_seconds": asynchronality[-1],
            }
        )
    sample_kind = (
        "streaming_block"
        if any(str(result.get("experiment_scenario_mode", "")) == "streaming" for result in test_results)
        else "payload_channel_episode"
    )
    return {
        "total_test_samples": len(samples),
        "test_sample_unit": "block" if sample_kind == "streaming_block" else "channel_episode",
        "test_sample_kind": sample_kind,
        "test_snr_db_by_user_by_seed": {
            str(seed): values for seed, values in test_snr_db_by_user_by_seed.items()
        },
        "final_total_latency_seconds": mean_std(final_latency),
        "latency_reduction_vs_random_percent": mean_std(latency_reduction),
        "final_asynchronality_sum_seconds": mean_std(asynchronality),
        "samples": samples,
    }


def build_seeded_scenario_collection_lines(
    summaries: list[dict[str, Any]],
    *,
    title: str,
) -> list[str]:
    lines = [title]
    for index, summary in enumerate(summaries):
        if index > 0:
            lines.append("")
        lines.extend(build_experiment_scenario_summary_lines(summary))
    return lines
