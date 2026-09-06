"""Canonical repository paths used by configuration and result code."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_ROOT = PROJECT_ROOT / "configs"
EXPERIMENT_CONFIG_ROOT = CONFIG_ROOT / "experiments"
BENCHMARK_CONFIG_ROOT = CONFIG_ROOT / "benchmarks"
BATCH_RUN_CONFIG_ROOT = CONFIG_ROOT / "batch_runs"
RESULTS_ROOT = PROJECT_ROOT / "Results"


__all__ = [
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "CONFIG_ROOT",
    "EXPERIMENT_CONFIG_ROOT",
    "BENCHMARK_CONFIG_ROOT",
    "BATCH_RUN_CONFIG_ROOT",
    "RESULTS_ROOT",
]
