"""Select and run the complete downlink convergence procedure.

This is the first algorithm file to read after ``experiment.py``. It validates
the objective and scenario once, then delegates to the payload or streaming
implementation. Both implementations use ``optimize_precoder.py`` for each
continuous fixed-blocklength solve.
"""

from typing import Any

from latency_optimization.experiments.scenarios import STREAMING_MODE, build_experiment_scenario

from ...objectives.settings import validate_convergence_objective_mode
from ...simulation.system import DownlinkSystem
from .optimize_payload import optimize_payload_transmission
from .optimize_streaming import optimize_streaming_transmission


def optimize_downlink_transmission(
    system: DownlinkSystem,
    sim_params: dict[str, Any],
    verbose: bool = True,
) -> dict[str, Any]:
    """Run payload or streaming optimization for one configured downlink system."""
    objective_mode = validate_convergence_objective_mode(sim_params)
    scenario = build_experiment_scenario(system.sc, sim_params, seed=int(system.seed))
    if str(scenario["mode"]) == STREAMING_MODE:
        return optimize_streaming_transmission(
            system,
            sim_params,
            verbose=verbose,
            method_name="convergence_per_epoch_baseline",
            objective_mode=objective_mode,
        )
    return optimize_payload_transmission(
        system,
        sim_params,
        verbose=verbose,
        method_name="convergence_per_epoch_baseline",
        objective_mode=objective_mode,
    )


__all__ = ["optimize_downlink_transmission"]
