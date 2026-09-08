"""Communication-scenario validation and deterministic state construction."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .config_validation import require_choice


PAYLOAD_MODE = "payload"
STREAMING_MODE = "streaming"
SCENARIO_MODES = {PAYLOAD_MODE, STREAMING_MODE}
PAYLOAD_SOURCES = {"system_B", "explicit"}


def require_scenario_mode(value: Any) -> str:
    """Validate the exact public scenario name: payload or streaming."""
    return require_choice(value, SCENARIO_MODES, "experiment_scenario.mode")


def _require_integer_vector(values: Any, number_of_users: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=int)
    if vector.ndim == 0:
        raise ValueError(f"{name} must provide one value per user; scalar broadcasting is not supported.")
    if vector.shape != (number_of_users,):
        raise ValueError(f"{name} must have shape ({number_of_users},), got {vector.shape}.")
    if np.any(vector < 0):
        raise ValueError(f"{name} cannot contain negative values.")
    return vector


def _reject_unknown_options(config: dict[str, Any], mode: str) -> None:
    options_by_mode = {
        PAYLOAD_MODE: {"mode", "payload_bits_source", "payload_bits_values"},
        STREAMING_MODE: {"mode", "number_of_blocks"},
    }
    unknown = sorted(set(config) - options_by_mode[mode])
    if unknown:
        raise ValueError(
            f"Unsupported experiment_scenario option(s) for {mode}: {', '.join(unknown)}."
        )


def validate_experiment_scenario_config(
    scenario_config: Any,
    *,
    system_params: dict[str, Any],
    max_total_blocks: int | None = None,
) -> dict[str, Any]:
    """Convert scenario configuration into one strict, canonical specification.

    What: reject unknown or missing fields, validate per-user integer bit vectors,
    and derive either one payload target per user or a fixed target repeated over a
    streaming horizon. Why: allocation code should consume one unambiguous schema;
    payload carries remaining bits between blocks, whereas streaming explicitly does
    not. The returned mapping is safe to place in ``sim_params`` and persist in the
    experiment manifest.
    """
    if not isinstance(scenario_config, dict):
        raise ValueError("simulation.experiment_scenario must be a mapping.")
    if "mode" not in scenario_config:
        raise ValueError("simulation.experiment_scenario.mode is required.")

    number_of_users = int(system_params["K"])
    system_bits = _require_integer_vector(system_params["B"], number_of_users, "system B")
    mode = require_scenario_mode(scenario_config["mode"])
    _reject_unknown_options(scenario_config, mode)

    if mode == PAYLOAD_MODE:
        source = require_choice(
            scenario_config.get("payload_bits_source", "system_B"),
            PAYLOAD_SOURCES,
            "experiment_scenario.payload_bits_source",
        )
        if source == "system_B":
            payload_bits = system_bits
        else:
            if "payload_bits_values" not in scenario_config:
                raise ValueError(
                    "experiment_scenario.payload_bits_values is required when "
                    "payload_bits_source is 'explicit'."
                )
            payload_bits = _require_integer_vector(
                scenario_config["payload_bits_values"],
                number_of_users,
                "experiment_scenario.payload_bits_values",
            )
        return {
            "mode": PAYLOAD_MODE,
            "payload_bits_source": source,
            "payload_bits": payload_bits.tolist(),
        }

    if "number_of_blocks" not in scenario_config:
        raise ValueError("The streaming scenario requires number_of_blocks.")
    number_of_blocks = int(scenario_config["number_of_blocks"])
    if number_of_blocks <= 0:
        raise ValueError("experiment_scenario.number_of_blocks must be positive.")
    if np.any(system_bits <= 0):
        raise ValueError("Streaming requires every system B[k] value to be positive.")
    if max_total_blocks is not None and number_of_blocks > int(max_total_blocks):
        raise ValueError(
            "experiment_scenario.number_of_blocks cannot exceed simulation.max_total_blocks."
        )
    return {
        "mode": STREAMING_MODE,
        "number_of_blocks": number_of_blocks,
        "bits_per_block_per_user": system_bits.tolist(),
    }


def build_experiment_scenario(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Instantiate the scheduling state associated with one channel seed.

    What: attach the seed and expand the canonical scenario into runtime targets,
    totals, stopping rules, and Monte Carlo sample metadata. Why: convergence,
    Monte Carlo, and benchmark methods must receive identical payload/streaming
    semantics when they use the same config and seed. This function creates no
    channels or beams; it describes only the traffic instance they must serve.
    """
    scenario_config = sim_params["experiment_scenario"]
    if not isinstance(scenario_config, dict):
        raise ValueError("simulation.experiment_scenario must be a mapping.")
    if "payload_bits" not in scenario_config and "bits_per_block_per_user" not in scenario_config:
        scenario_config = validate_experiment_scenario_config(
            scenario_config,
            system_params=system_params,
            max_total_blocks=sim_params.get("max_total_blocks"),
        )
    mode = require_scenario_mode(scenario_config["mode"])
    shared_state = {"mode": mode, "seed": int(seed)}

    if mode == PAYLOAD_MODE:
        payload_bits = _require_integer_vector(
            scenario_config["payload_bits"],
            int(system_params["K"]),
            "payload_bits",
        )
        return {
            **shared_state,
            "payload_bits_per_user": payload_bits.tolist(),
            "per_user_total_target_bits": payload_bits.tolist(),
            "total_target_bits": int(np.sum(payload_bits)),
            "termination_rule": "until_payload_drained",
        }

    bits_per_block = _require_integer_vector(
        scenario_config["bits_per_block_per_user"],
        int(system_params["K"]),
        "bits_per_block_per_user",
    )
    number_of_blocks = int(scenario_config["number_of_blocks"])
    targets_by_block = np.repeat(bits_per_block[:, np.newaxis], number_of_blocks, axis=1)
    per_user_total_bits = bits_per_block * number_of_blocks
    return {
        **shared_state,
        "number_of_blocks": number_of_blocks,
        "bits_per_block_per_user": bits_per_block.tolist(),
        "streaming_bit_targets_by_block": targets_by_block.tolist(),
        "per_user_total_target_bits": per_user_total_bits.tolist(),
        "total_target_bits": int(np.sum(per_user_total_bits)),
        "carry_unserved_bits_to_next_block": False,
        "termination_rule": "after_number_of_blocks",
    }


def build_experiment_scenarios_for_seeds(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Create one deterministic runtime scenario for each channel seed."""
    return [
        build_experiment_scenario(system_params, sim_params, seed=int(seed))
        for seed in seeds
    ]


def build_monte_carlo_sample_scenarios_for_seeds(
    system_params: dict[str, Any],
    sim_params: dict[str, Any],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Build the base samples used by the Monte Carlo trainer and evaluator.

    Payload samples are complete channel episodes: the rollout may create and
    optimize later blocks while the payload is being drained. Streaming has no
    carried state, so each requested training/test sample is one independent
    streaming block with a fresh seed.
    """
    scenarios: list[dict[str, Any]] = []
    for seed in seeds:
        scenario = build_experiment_scenario(system_params, sim_params, seed=int(seed))
        if scenario["mode"] == STREAMING_MODE:
            bits = np.asarray(scenario["bits_per_block_per_user"], dtype=int)
            scenario = {
                **scenario,
                "number_of_blocks": 1,
                "streaming_bit_targets_by_block": bits[:, np.newaxis].tolist(),
                "per_user_total_target_bits": bits.tolist(),
                "total_target_bits": int(np.sum(bits)),
                "monte_carlo_sample_unit": "block",
                "monte_carlo_sample_kind": "streaming_block",
            }
        else:
            scenario = {
                **scenario,
                "monte_carlo_sample_unit": "channel_episode",
                "monte_carlo_sample_kind": "payload_channel_episode",
            }
        scenarios.append(scenario)
    return scenarios


def build_experiment_scenario_summary(scenario: dict[str, Any]) -> dict[str, Any]:
    """Aggregate target-bit and sample-unit metadata for saved dataset summaries."""
    mode = require_scenario_mode(scenario.get("mode"))
    summary = {
        "mode": mode,
        "seed": int(scenario.get("seed", 0)),
        "monte_carlo_sample_unit": str(scenario.get("monte_carlo_sample_unit", "")),
        "monte_carlo_sample_kind": str(scenario.get("monte_carlo_sample_kind", "")),
        "termination_rule": str(scenario.get("termination_rule", "")),
        "per_user_total_target_bits": [
            int(value) for value in scenario.get("per_user_total_target_bits", [])
        ],
        "total_target_bits": int(scenario.get("total_target_bits", 0)),
    }
    if mode == PAYLOAD_MODE:
        summary["payload_bits_per_user"] = [
            int(value) for value in scenario.get("payload_bits_per_user", [])
        ]
    else:
        summary.update(
            number_of_blocks=int(scenario.get("number_of_blocks", 0)),
            bits_per_block_per_user=[
                int(value) for value in scenario.get("bits_per_block_per_user", [])
            ],
            streaming_bit_targets_by_block=scenario.get("streaming_bit_targets_by_block", []),
            carry_unserved_bits_to_next_block=False,
        )
    return summary


def build_experiment_scenario_summary_lines(summary: dict[str, Any]) -> list[str]:
    """Render scenario metadata as the human-readable result-summary preamble."""
    mode = require_scenario_mode(summary.get("mode"))
    lines = [
        "Experiment scenario summary",
        f"Mode: {mode}",
        f"Seed: {int(summary.get('seed', 0))}",
        f"Termination rule: {summary.get('termination_rule', '')}",
        f"Per-user total target bits: {summary.get('per_user_total_target_bits', [])}",
        f"Total target bits: {int(summary.get('total_target_bits', 0))}",
    ]
    if summary.get("monte_carlo_sample_kind"):
        lines.append(f"Monte Carlo sample unit: {summary.get('monte_carlo_sample_unit', '')}")
        lines.append(f"Monte Carlo sample kind: {summary['monte_carlo_sample_kind']}")
    if mode == PAYLOAD_MODE:
        lines.append(f"Payload bits per user: {summary.get('payload_bits_per_user', [])}")
    else:
        lines.extend(
            [
                f"Number of blocks: {int(summary.get('number_of_blocks', 0))}",
                f"Bits requested per block per user: {summary.get('bits_per_block_per_user', [])}",
                "Unserved bits carried to next block: False",
            ]
        )
    return lines


__all__ = [
    "PAYLOAD_MODE",
    "STREAMING_MODE",
    "build_experiment_scenario",
    "build_experiment_scenario_summary",
    "build_experiment_scenario_summary_lines",
    "build_experiment_scenarios_for_seeds",
    "build_monte_carlo_sample_scenarios_for_seeds",
    "require_scenario_mode",
    "validate_experiment_scenario_config",
]
