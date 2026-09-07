from __future__ import annotations

import argparse

from latency_optimization.results.paths import build_downlink_convergence_result_dirs

from ..config import load_config
from ..runner import build_result_tag, run_downlink_experiment
from ..objective_settings import validate_convergence_objective_mode

METHOD_NAME = "convergence_per_epoch_baseline"
METHOD_LABEL = "Convergence per epoch"


def main() -> None:
    parser = argparse.ArgumentParser(description="Downlink online convergence baseline")
    parser.add_argument("--cfg_name", type=str, default="downlink_dispersion_heavy.yaml", help="Configuration file name or path")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic random seed")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
    args = parser.parse_args()

    _, sim_params, run_meta = load_config(args.cfg_name)
    objective_mode = validate_convergence_objective_mode(sim_params)
    result_tag = build_result_tag(
        METHOD_NAME,
        run_meta["cfg_stem"],
        int(args.seed),
        objective_mode=objective_mode,
        model_scope=sim_params.get("downlink_precoder_net_scope"),
        solver_mode=sim_params.get("convergence_precoder_update_mode"),
        cfg_hash=run_meta.get("cfg_hash"),
    )
    output_dirs = build_downlink_convergence_result_dirs(
        METHOD_LABEL,
        result_tag,
        scenario_mode=str(sim_params["experiment_scenario_mode"]),
    )
    run_downlink_experiment(
        METHOD_NAME,
        args.cfg_name,
        args.seed,
        verbose=not args.quiet,
        output_root=output_dirs["testing_root"],
    )
    print(f"Saved downlink convergence results to: {output_dirs['experiment_root']}")


if __name__ == "__main__":
    main()
