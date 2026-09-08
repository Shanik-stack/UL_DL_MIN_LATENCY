"""Select and run the complete uplink convergence procedure.

This is the first algorithm file to read after ``experiment.py``. It selects
payload or streaming behavior; each scenario module delegates one user's
fixed-blocklength solve to ``optimize_precoder.py`` through
``optimize_user_transmission.py``.
"""

from latency_optimization.experiments.scenarios import STREAMING_MODE

from .optimize_payload import optimize_payload_with_precoder_training
from .optimize_precoder import validate_convergence_precoder_update_mode
from .optimize_streaming import optimize_streaming_blocks_with_precoder_training


def run_convergence_baseline(
    uplinksystem,
    sim_cfg: dict,
    interference_F_snapshot=None,
    commit_live_precoders: bool = True,
):
    """Run payload or streaming convergence optimization for one uplink system."""
    local_sim_cfg = dict(sim_cfg)
    effective_epochs = max(1, int(local_sim_cfg["max_epochs"]))
    local_sim_cfg["max_epochs"] = effective_epochs
    local_sim_cfg["max_precoder_epochs"] = effective_epochs

    optimize_scenario = (
        optimize_streaming_blocks_with_precoder_training
        if str(local_sim_cfg["experiment_scenario_mode"]) == STREAMING_MODE
        else optimize_payload_with_precoder_training
    )
    convergence_data = optimize_scenario(
        uplinksystem=uplinksystem,
        sim_cfg=local_sim_cfg,
        interference_F_snapshot=interference_F_snapshot,
        commit_live_precoders=commit_live_precoders,
    )
    update_mode = validate_convergence_precoder_update_mode(
        local_sim_cfg["convergence_precoder_update_mode"]
    )
    convergence_data["convergence_precoder_update_mode"] = update_mode
    convergence_data["precoder_parameterization"] = (
        "shared_user_channel_n_sigma_epsilon_to_precoder_mlp_online_convergence"
        if update_mode == "precoder_net"
        else "direct_complex_precoder_per_user_block_online_convergence"
    )
    convergence_data["method_name"] = "convergence_per_epoch_baseline"
    convergence_data["configured_max_epochs"] = effective_epochs
    return convergence_data


__all__ = ["run_convergence_baseline"]
