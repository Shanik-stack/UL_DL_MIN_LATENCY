"""Shared loading and validation for experiment configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from latency_optimization.core.scenarios import PAYLOAD_MODE, STREAMING_MODE
from latency_optimization.project import BENCHMARK_CONFIG_ROOT, EXPERIMENT_CONFIG_ROOT, PROJECT_ROOT
from latency_optimization.results.naming import build_config_content_hash


@dataclass(frozen=True)
class ConfigDocument:
    data: dict[str, Any]
    path: Path

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "cfg_path": str(self.path),
            "cfg_stem": self.path.stem,
            "cfg_hash": build_config_content_hash(self.data),
        }


def load_config_document(config_name: str) -> ConfigDocument:
    """Resolve a YAML name/path and return its data plus canonical path metadata."""
    if not str(config_name).endswith(".yaml"):
        raise ValueError(f"Configuration name must end with '.yaml'; got {config_name!r}.")

    requested = Path(config_name)
    candidates = (
        [requested]
        if requested.is_absolute()
        else [
            Path(EXPERIMENT_CONFIG_ROOT) / requested,
            Path(BENCHMARK_CONFIG_ROOT) / requested,
            Path(PROJECT_ROOT) / requested,
            Path.cwd() / requested,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as stream:
                data = yaml.safe_load(stream)
            if not isinstance(data, dict):
                raise ValueError(f"Configuration root must be a mapping: {candidate}")
            return ConfigDocument(data=data, path=candidate.resolve())
    raise FileNotFoundError(f"Could not find config file: {config_name}")


def optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _snr_ranges(raw_ranges: Any, nominal_snr_db: Sequence[float], key: str) -> list[list[float]]:
    ranges = (
        [[float(value), float(value)] for value in nominal_snr_db]
        if raw_ranges is None
        else [[float(bound) for bound in user_range] for user_range in raw_ranges]
    )
    if len(ranges) != len(nominal_snr_db) or any(len(user_range) != 2 for user_range in ranges):
        raise ValueError(f"{key} must contain one [minimum, maximum] pair per user.")
    for minimum, maximum in ranges:
        if minimum > maximum:
            raise ValueError(f"{key} minimum cannot exceed its maximum.")
    return ranges


def build_monte_carlo_sampling_settings(
    simulation: Mapping[str, Any],
    *,
    scenario_mode: str,
    nominal_snr_db: Sequence[float],
) -> dict[str, Any]:
    """Validate scenario-specific sample counts and per-user train/test SNR ranges."""
    training_channels = optional_int(simulation.get("monte_carlo_num_training_channels"))
    training_blocks = optional_int(simulation.get("monte_carlo_num_training_blocks"))
    test_channels = optional_int(simulation.get("monte_carlo_num_test_channels"))
    test_blocks = optional_int(simulation.get("monte_carlo_num_test_blocks"))

    if scenario_mode == PAYLOAD_MODE:
        if training_blocks is not None or test_blocks is not None:
            raise ValueError("Payload Monte Carlo uses channel counts, not block counts.")
        sample_unit = "channel_episode"
        training_samples = training_channels
        test_samples = test_channels
    elif scenario_mode == STREAMING_MODE:
        if training_channels is not None or test_channels is not None:
            raise ValueError("Streaming Monte Carlo uses block counts, not channel counts.")
        sample_unit = "block"
        training_samples = training_blocks
        test_samples = test_blocks
    else:
        raise ValueError(f"Unsupported experiment scenario mode: {scenario_mode!r}.")

    training_ranges = _snr_ranges(
        simulation.get("monte_carlo_training_snr_db_ranges"),
        nominal_snr_db,
        "monte_carlo_training_snr_db_ranges",
    )
    test_ranges = _snr_ranges(
        simulation.get("monte_carlo_test_snr_db_ranges", training_ranges),
        nominal_snr_db,
        "monte_carlo_test_snr_db_ranges",
    )
    return {
        "monte_carlo_train_seeds": simulation.get("monte_carlo_train_seeds"),
        "monte_carlo_num_training_channels": training_channels,
        "monte_carlo_num_training_blocks": training_blocks,
        "monte_carlo_sample_unit": sample_unit,
        "monte_carlo_num_training_samples": training_samples,
        "monte_carlo_training_snr_db_ranges": training_ranges,
        "monte_carlo_num_test_channels": test_channels,
        "monte_carlo_num_test_blocks": test_blocks,
        "monte_carlo_num_test_samples": test_samples,
        "monte_carlo_test_snr_db_ranges": test_ranges,
        "monte_carlo_test_seed": optional_int(simulation.get("monte_carlo_test_seed")),
    }
