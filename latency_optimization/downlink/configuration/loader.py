from __future__ import annotations

from typing import Any

import numpy as np
from latency_optimization.optimization.blocklength_search import validate_n_search_direction, validate_n_search_strategy
from latency_optimization.experiments.scenarios import validate_experiment_scenario_config
from latency_optimization.experiments.config_validation import require_choice
from latency_optimization.experiments.configuration import (
    build_monte_carlo_sampling_settings,
    load_config_document,
)
from latency_optimization.optimization.convergence_criteria import CONVERGENCE_STOPPING_RULES
from latency_optimization.physics.rate_law import resolve_rate_law

from ..precoders.models import validate_downlink_precoder_net_scope


SHARED_BS_STREAMING_BLOCKLENGTH_INPUT_MODES = {
    "joint_blocklength_vector",
    "one_user_change_at_a_time",
}


def validate_shared_bs_streaming_blocklength_input_mode(mode: str) -> str:
    return require_choice(
        mode,
        SHARED_BS_STREAMING_BLOCKLENGTH_INPUT_MODES,
        "shared_bs_streaming_blocklength_input_mode",
    )


ALLOWED_SIMULATION_KEYS = {
    "finite_blocklength_rate_law",
    "convergence_precoder_update_mode",
    "downlink_precoder_net_scope",
    "shared_bs_streaming_blocklength_input_mode",
    "max_epochs",
    "user_update_lr",
    "print_every_epoch",
    "precoder_change_tolerance",
    "convergence_stopping_rule",
    "kkt_primal_tolerance",
    "kkt_complementarity_tolerance",
    "kkt_stationarity_tolerance",
    "monte_carlo_train_seeds",
    "monte_carlo_num_training_channels",
    "monte_carlo_num_training_blocks",
    "monte_carlo_training_snr_db_ranges",
    "monte_carlo_num_test_channels",
    "monte_carlo_num_test_blocks",
    "monte_carlo_test_snr_db_ranges",
    "monte_carlo_test_seed",
    "monte_carlo_training_max_epochs",
    "max_total_blocks",
    "experiment_scenario",
    "n_kl_range",
    "n_search_direction",
    "n_search_strategy",
    "n_search_coarse_step",
    "n_search_exponential_factor",
    "monte_carlo_test_n_search_direction",
    "monte_carlo_test_n_search_strategy",
    "monte_carlo_test_n_search_coarse_step",
    "monte_carlo_test_n_search_exponential_factor",
}


def _validate_simulation_keys(sim_cfg: dict) -> None:
    unsupported = sorted(set(sim_cfg) - ALLOWED_SIMULATION_KEYS)
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(
            "Unsupported simulation option(s): "
            f"{joined}. The cleaned suite uses objective-only inverse-CNR "
            "optimization and the documented payload/streaming and search options only."
        )


def _as_array(values: Any, K: int, name: str, dtype) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.shape != (K,):
        raise ValueError(f"{name} must have shape ({K},), got {arr.shape}")
    return arr


def _resolve_downlink_block_power_budget(power_values: np.ndarray) -> float:
    if power_values.size <= 0:
        raise ValueError("Downlink P must contain at least one value.")
    budget = float(power_values[0])
    if not np.allclose(power_values, budget, rtol=1e-6, atol=1e-9):
        raise ValueError(
            "Downlink uses one BS block power budget for the full precoder F_b, "
            "so test.P must contain the same value for every user."
        )
    return budget


def load_config(cfg_name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load and strictly validate one downlink experiment configuration."""
    document = load_config_document(cfg_name)
    cfg = document.data

    test_cfg = cfg["test"]
    K = int(test_cfg["K"])

    if "Nb" not in test_cfg:
        raise ValueError("Downlink test configuration requires Nb.")
    Nb = _as_array(test_cfg["Nb"], K, "Nb", int)
    Nr = _as_array(test_cfg["Nr"], K, "Nr", int)

    system_params = {
        "K": K,
        "Nb": Nb,
        "Nr": Nr,
        "dk": np.minimum(Nb, Nr),
        "B": _as_array(test_cfg["B"], K, "B", int),
        "P": _as_array(test_cfg["P"], K, "P", float),
        "fs": _as_array(test_cfg["fs"], K, "fs", float),
        "snr_db": _as_array(test_cfg["snr_db"], K, "snr_db", float),
        "epsilon": _as_array(test_cfg["epsilon"], K, "epsilon", float),
        "T": _as_array(test_cfg["T"], K, "T", int),
    }
    system_params["block_power_budget"] = _resolve_downlink_block_power_budget(system_params["P"])

    sim_cfg_raw = cfg.get("simulation", {})
    _validate_simulation_keys(sim_cfg_raw)
    rate_law_name = str(sim_cfg_raw["finite_blocklength_rate_law"])
    resolve_rate_law(rate_law_name)
    n_range = sim_cfg_raw.get("n_kl_range", {})
    scenario_cfg = validate_experiment_scenario_config(
        sim_cfg_raw.get("experiment_scenario", {}),
        system_params=system_params,
        max_total_blocks=int(sim_cfg_raw.get("max_total_blocks", 256)),
    )
    scenario_mode = str(scenario_cfg["mode"])
    monte_carlo_settings = build_monte_carlo_sampling_settings(
        sim_cfg_raw,
        scenario_mode=scenario_mode,
        nominal_snr_db=system_params["snr_db"],
    )
    max_epochs = int(sim_cfg_raw.get("max_epochs", 500))
    model_scope = validate_downlink_precoder_net_scope(
        sim_cfg_raw.get("downlink_precoder_net_scope", "per_user_nets")
    )
    sim_params = {
        "finite_blocklength_rate_law": rate_law_name,
        "max_epochs": max(1, max_epochs),
        "max_precoder_epochs": max(1, max_epochs),
        "print_every_epoch": int(sim_cfg_raw.get("print_every_epoch", 1)),
        "user_update_lr": float(sim_cfg_raw.get("user_update_lr", 5e-3)),
        "convergence_precoder_update_mode": require_choice(
            sim_cfg_raw.get("convergence_precoder_update_mode", "precoder_net"),
            {"precoder_net", "direct_precoder"},
            "convergence_precoder_update_mode",
        ),
        # The retained methods optimize the inverse-CNR rate objective only.
        # Rate and power feasibility remain allocation checks, not loss terms.
        "precoder_change_tolerance": float(sim_cfg_raw["precoder_change_tolerance"]),
        "convergence_stopping_rule": require_choice(
            sim_cfg_raw["convergence_stopping_rule"],
            CONVERGENCE_STOPPING_RULES,
            "convergence_stopping_rule",
        ),
        "kkt_primal_tolerance": float(sim_cfg_raw["kkt_primal_tolerance"]),
        "kkt_complementarity_tolerance": float(sim_cfg_raw["kkt_complementarity_tolerance"]),
        "kkt_stationarity_tolerance": float(sim_cfg_raw["kkt_stationarity_tolerance"]),
        "max_total_blocks": int(sim_cfg_raw.get("max_total_blocks", 256)),
        "n_kl_min": int(n_range.get("min", 5)),
        "n_kl_step": int(n_range.get("step", 1)),
        "n_search_direction": validate_n_search_direction(
            sim_cfg_raw.get("n_search_direction", "descending")
        ),
        "n_search_strategy": validate_n_search_strategy(
            sim_cfg_raw.get("n_search_strategy", "fixed_step")
        ),
        "n_search_coarse_step": int(sim_cfg_raw.get("n_search_coarse_step", int(n_range.get("step", 1)))),
        "n_search_exponential_factor": int(sim_cfg_raw.get("n_search_exponential_factor", 2)),
        "monte_carlo_test_n_search_direction": validate_n_search_direction(
            sim_cfg_raw.get("monte_carlo_test_n_search_direction", sim_cfg_raw.get("n_search_direction", "descending"))
        ),
        "monte_carlo_test_n_search_strategy": validate_n_search_strategy(
            sim_cfg_raw.get("monte_carlo_test_n_search_strategy", sim_cfg_raw.get("n_search_strategy", "fixed_step"))
        ),
        "monte_carlo_test_n_search_coarse_step": int(
            sim_cfg_raw.get("monte_carlo_test_n_search_coarse_step", sim_cfg_raw.get("n_search_coarse_step", int(n_range.get("step", 1))))
        ),
        "monte_carlo_test_n_search_exponential_factor": int(
            sim_cfg_raw.get("monte_carlo_test_n_search_exponential_factor", sim_cfg_raw.get("n_search_exponential_factor", 2))
        ),
        "n_kl_reduction_update_scope": "all_active_users",
        "monte_carlo_training_max_epochs": int(
            sim_cfg_raw.get("monte_carlo_training_max_epochs", max_epochs)
        ),
        **monte_carlo_settings,
        "convergence_block_objective_mode": "inverse_cnr_weighted_sum_rate",
        "convergence_priority_weight_strategy": "inverse_cnr",
        "downlink_precoder_net_scope": str(model_scope),
        "shared_bs_streaming_blocklength_input_mode": require_choice(
            sim_cfg_raw.get(
                "shared_bs_streaming_blocklength_input_mode",
                "joint_blocklength_vector",
            ),
            {"joint_blocklength_vector", "one_user_change_at_a_time"},
            "shared_bs_streaming_blocklength_input_mode",
        ),
        "experiment_scenario": scenario_cfg,
        "experiment_scenario_mode": str(scenario_cfg["mode"]),
    }
    system_params["finite_blocklength_rate_law"] = rate_law_name

    return system_params, sim_params, document.metadata
