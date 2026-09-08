from __future__ import annotations

from latency_optimization.optimization.blocklength_search import validate_n_search_direction, validate_n_search_strategy
from latency_optimization.experiments.scenarios import validate_experiment_scenario_config
from latency_optimization.experiments.config_validation import require_choice
from latency_optimization.experiments.configuration import (
    build_monte_carlo_sampling_settings,
    load_config_document,
)
from latency_optimization.optimization.convergence_criteria import CONVERGENCE_STOPPING_RULES
from latency_optimization.physics.rate_law import resolve_rate_law

from ..objectives.settings import RATE_BEAM_REWARD_MODE, UNWEIGHTED_SUM_RATE_OBJECTIVE
from ..physics.rate import validate_uplink_rate_model
from ..simulation.parameters import initialize_system_params


ALLOWED_SIMULATION_KEYS = {
    "finite_blocklength_rate_law",
    "convergence_precoder_update_mode",
    "max_epochs",
    "lr_net",
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
    "uplink_rate_model",
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
    # Used only by the deliberately small exhaustive-search benchmark.
    "exhaustive_compare",
}

def _validate_simulation_keys(sim_cfg: dict) -> None:
    unsupported = sorted(set(sim_cfg) - ALLOWED_SIMULATION_KEYS)
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(
            "Unsupported simulation option(s): "
            f"{joined}. The cleaned suite uses objective-only optimization, "
            "the documented payload/streaming and precoder/search options only."
        )


def _build_config(cfg: dict) -> tuple[dict, dict]:
    test_cfg = cfg["test"]
    test_k = test_cfg["K"]
    test_Nr = test_cfg["Nr"]
    test_Nt = test_cfg["Nt"]
    initial_bits_per_symbol = test_cfg.get("initial_bits_per_symbol")

    if "T" in test_cfg:
        system_test_params = initialize_system_params(
            B=test_cfg["B"],
            P=test_cfg["P"],
            fs=test_cfg["fs"],
            snr_db=test_cfg["snr_db"],
            desired_CNR=None,
            Nt=test_Nt,
            Nr=test_Nr,
            K=test_k,
            epsilon=test_cfg["epsilon"],
            initial_bits_per_symbol=initial_bits_per_symbol,
            T=test_cfg["T"],
        )
    else:
        system_test_params = initialize_system_params(
            B=test_cfg["B"],
            P=test_cfg["P"],
            fs=test_cfg["fs"],
            snr_db=test_cfg["snr_db"],
            desired_CNR=None,
            Nt=test_Nt,
            Nr=test_Nr,
            K=test_k,
            f_carrier=test_cfg["f_carrier"],
            v=test_cfg["v"],
            epsilon=test_cfg["epsilon"],
            initial_bits_per_symbol=initial_bits_per_symbol,
        )

    sim_cfg = cfg["simulation"]
    _validate_simulation_keys(sim_cfg)
    rate_law_name = str(sim_cfg["finite_blocklength_rate_law"])
    resolve_rate_law(rate_law_name)
    uplink_rate_model = validate_uplink_rate_model(sim_cfg["uplink_rate_model"])
    scenario_cfg = validate_experiment_scenario_config(
        sim_cfg.get("experiment_scenario", {}),
        system_params=system_test_params,
        max_total_blocks=int(sim_cfg.get("max_total_blocks", 256)),
    )
    scenario_mode = str(scenario_cfg["mode"])
    monte_carlo_settings = build_monte_carlo_sampling_settings(
        sim_cfg,
        scenario_mode=scenario_mode,
        nominal_snr_db=system_test_params["snr_db"],
    )
    n_kl_max = [system_test_params["T"][user] for user in range(system_test_params["K"])]
    max_epochs = int(sim_cfg.get("max_epochs", 500))
    # The cleaned experiment suite uses a direct rate objective. Feasibility is
    # checked when allocating bits, rather than added as a loss penalty.
    uplink_objective_mode = UNWEIGHTED_SUM_RATE_OBJECTIVE
    uplink_beam_reward_mode = RATE_BEAM_REWARD_MODE
    simulation_test_params = {
        "max_epochs": max(1, max_epochs),
        "lr_net": float(sim_cfg.get("lr_net", 1e-2)),
        "convergence_precoder_update_mode": sim_cfg.get(
            "convergence_precoder_update_mode", "precoder_net"
        ),
        "n_kl_min": sim_cfg["n_kl_range"]["min"],
        "n_kl_max": n_kl_max,
        "n_kl_step": sim_cfg["n_kl_range"]["step"],
        "n_search_direction": validate_n_search_direction(
            sim_cfg.get("n_search_direction", "descending")
        ),
        "n_search_strategy": validate_n_search_strategy(
            sim_cfg.get("n_search_strategy", "fixed_step")
        ),
        "n_search_coarse_step": int(sim_cfg.get("n_search_coarse_step", sim_cfg["n_kl_range"]["step"])),
        "n_search_exponential_factor": int(sim_cfg.get("n_search_exponential_factor", 2)),
        "monte_carlo_test_n_search_direction": validate_n_search_direction(
            sim_cfg.get("monte_carlo_test_n_search_direction", sim_cfg.get("n_search_direction", "descending"))
        ),
        "monte_carlo_test_n_search_strategy": validate_n_search_strategy(
            sim_cfg.get("monte_carlo_test_n_search_strategy", sim_cfg.get("n_search_strategy", "fixed_step"))
        ),
        "monte_carlo_test_n_search_coarse_step": int(
            sim_cfg.get("monte_carlo_test_n_search_coarse_step", sim_cfg.get("n_search_coarse_step", sim_cfg["n_kl_range"]["step"]))
        ),
        "monte_carlo_test_n_search_exponential_factor": int(
            sim_cfg.get("monte_carlo_test_n_search_exponential_factor", sim_cfg.get("n_search_exponential_factor", 2))
        ),
        "max_total_blocks": int(sim_cfg.get("max_total_blocks", 256)),
        "max_precoder_epochs": max(1, max_epochs),
        "print_every_epoch": int(sim_cfg.get("print_every_epoch", 1)),
        "monte_carlo_training_max_epochs": int(
            sim_cfg.get("monte_carlo_training_max_epochs", max_epochs)
        ),
        **monte_carlo_settings,
        "precoder_change_tolerance": float(sim_cfg["precoder_change_tolerance"]),
        "convergence_stopping_rule": require_choice(
            sim_cfg["convergence_stopping_rule"],
            CONVERGENCE_STOPPING_RULES,
            "convergence_stopping_rule",
        ),
        "kkt_primal_tolerance": float(sim_cfg["kkt_primal_tolerance"]),
        "kkt_complementarity_tolerance": float(sim_cfg["kkt_complementarity_tolerance"]),
        "kkt_stationarity_tolerance": float(sim_cfg["kkt_stationarity_tolerance"]),
        "reduced_n_kl_log_interval": 1,
        "uplink_rate_model": uplink_rate_model,
        "finite_blocklength_rate_law": rate_law_name,
        "uplink_objective_mode": uplink_objective_mode,
        "beam_reward_mode": uplink_beam_reward_mode,
        "experiment_scenario": scenario_cfg,
        "experiment_scenario_mode": str(scenario_cfg["mode"]),
    }
    system_test_params["uplink_rate_model"] = uplink_rate_model
    system_test_params["finite_blocklength_rate_law"] = rate_law_name
    system_test_params["uplink_objective_mode"] = uplink_objective_mode
    system_test_params["beam_reward_mode"] = uplink_beam_reward_mode
    return system_test_params, simulation_test_params


def load_config(cfg_name: str) -> tuple[dict, dict, dict]:
    """Load and strictly validate one uplink experiment configuration."""
    document = load_config_document(cfg_name)
    system_test_params, simulation_test_params = _build_config(document.data)
    return system_test_params, simulation_test_params, document.metadata
