from __future__ import annotations

import re
from pathlib import Path

from latency_optimization.experiments.scenarios import PAYLOAD_MODE, STREAMING_MODE
from latency_optimization.experiments.config_validation import require_choice
from latency_optimization.project import RESULTS_ROOT



def _sanitize(value: str) -> str:
    text = re.sub(r"\s+", "_", str(value).strip())
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


_SCENARIO_RESULT_NAMES = {
    PAYLOAD_MODE: "payload_completion",
    STREAMING_MODE: "streaming",
}
_METHOD_RESULT_NAMES = {
    "Convergence per epoch": "convergence_per_epoch",
    "Monte Carlo": "monte_carlo",
    "Benchmark ZF": "benchmark_zf",
    "Benchmark RZF": "benchmark_rzf",
    "Small Exhaustive validation": "benchmark_exhaustive",
}


def _create_and_stringify(directories: dict[str, Path]) -> dict[str, str]:
    for directory in set(directories.values()):
        directory.mkdir(parents=True, exist_ok=True)
    return {key: str(value) for key, value in directories.items()}


def _clean_scenario_result_name(scenario_mode: str) -> str:
    mode = require_choice(scenario_mode, {PAYLOAD_MODE, STREAMING_MODE}, "scenario mode")
    return _SCENARIO_RESULT_NAMES[mode]


def _clean_method_result_name(method_name: str) -> str:
    clean_name = str(method_name).strip()
    return _METHOD_RESULT_NAMES.get(clean_name, _sanitize(clean_name).lower())


def build_experiment_root(
    link_name: str,
    method_name: str,
    experiment_name: str,
    *,
    scenario_mode: str = PAYLOAD_MODE,
) -> Path:
    """Return the one canonical result root for a configured experiment."""
    return (
        RESULTS_ROOT
        / str(link_name).strip()
        / _clean_scenario_result_name(scenario_mode)
        / _clean_method_result_name(method_name)
        / _sanitize(experiment_name)
    )


def build_uplink_result_dirs(
    method_name: str,
    experiment_name: str,
    *,
    scenario_mode: str = PAYLOAD_MODE,
) -> dict[str, str]:
    """Create the standard uplink Monte Carlo training/testing artifact tree."""
    root = build_experiment_root(
        "Uplink", method_name, experiment_name, scenario_mode=scenario_mode
    )
    training_root = root / "training"
    testing_root = root / "testing"
    sample_directory = "channels" if scenario_mode == PAYLOAD_MODE else "blocks"
    dirs = {
        "experiment_root": root,
        "training_root": training_root,
        "testing_root": testing_root,
        "train_data": training_root / "data",
        "train_samples": training_root / sample_directory,
        "train_optimization_history": training_root / "optimization_history",
        "train_evaluation": training_root / "evaluation",
        "test_samples": testing_root / sample_directory,
        "test_data": testing_root / "data",
        "optimization_history": testing_root / "optimization_history",
        "user_config": testing_root / "user_config",
        "latency_asynchronality": testing_root / "latency_asynchronality",
        "link_quality": testing_root / "link_quality",
        "interference": testing_root / "interference",
        "schedule_details": testing_root / "schedule_details",
    }
    return _create_and_stringify(dirs)


def build_uplink_convergence_result_dirs(
    method_name: str,
    experiment_name: str,
    *,
    scenario_mode: str = PAYLOAD_MODE,
) -> dict[str, str]:
    """Create the testing-only uplink convergence artifact tree."""
    root = build_experiment_root(
        "Uplink", method_name, experiment_name, scenario_mode=scenario_mode
    )
    testing_root = root / "testing"
    data_root = testing_root / "data"
    optimization_root = testing_root / "optimization_history"
    user_config_root = testing_root / "user_config"
    latency_root = testing_root / "latency_asynchronality"
    link_root = testing_root / "link_quality"
    interference_root = testing_root / "interference"
    schedule_root = testing_root / "schedule_details"
    dirs = {
        "experiment_root": root,
        "testing_root": testing_root,
        "data": data_root,
        "optimization_history": optimization_root,
        "user_config": user_config_root,
        "latency_asynchronality": latency_root,
        "link_quality": link_root,
        "interference": interference_root,
        "schedule_details": schedule_root,
    }
    return _create_and_stringify(dirs)


def build_downlink_result_dirs(
    method_name: str,
    experiment_name: str,
    *,
    scenario_mode: str = PAYLOAD_MODE,
) -> dict[str, str]:
    """Create the standard downlink Monte Carlo training/testing artifact tree."""
    root = build_experiment_root(
        "Downlink", method_name, experiment_name, scenario_mode=scenario_mode
    )
    training_root = root / "training"
    testing_root = root / "testing"
    sample_directory = "channels" if scenario_mode == PAYLOAD_MODE else "blocks"
    dirs = {
        "experiment_root": root,
        "training_root": training_root,
        "testing_root": testing_root,
        "train_data": training_root / "data",
        "train_samples": training_root / sample_directory,
        "train_optimization_history": training_root / "optimization_history",
        "test_samples": testing_root / sample_directory,
        "test_data": testing_root / "data",
        "user_config": testing_root / "user_config",
        "latency_asynchronality": testing_root / "latency_asynchronality",
        "link_quality": testing_root / "link_quality",
        "optimization_history": testing_root / "optimization_history",
        "schedule_details": testing_root / "schedule_details",
        "interference": testing_root / "interference",
    }
    return _create_and_stringify(dirs)


def build_downlink_convergence_result_dirs(
    method_name: str,
    experiment_name: str,
    *,
    scenario_mode: str = PAYLOAD_MODE,
) -> dict[str, str]:
    """Build the testing-only layout used by the convergence method."""
    root = build_experiment_root(
        "Downlink", method_name, experiment_name, scenario_mode=scenario_mode
    )
    testing_root = root / "testing"
    dirs = {
        "experiment_root": root,
        "testing_root": testing_root,
    }
    return _create_and_stringify(dirs)
