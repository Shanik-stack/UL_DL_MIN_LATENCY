"""Evaluate standard stacked-channel downlink ZF/RZF on a saved test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from latency_optimization.experiments.channels import with_monte_carlo_sample_snr_by_user
from latency_optimization.experiments.scenarios import STREAMING_MODE, build_experiment_scenario
from latency_optimization.results.paths import build_experiment_root
from latency_optimization.results.persistence import (
    current_local_timestamp,
    save_json,
    save_text,
    write_result_manifest,
)
from latency_optimization.results.naming import make_method_result_tag

from ..configuration.loader import load_config
from ..simulation.baselines import estimate_random_precoder_payload_latency, estimate_random_precoder_streaming_latency
from ..simulation.system import DownlinkSystem
from .linear_beamforming import build_joint_linear_precoders


def _evaluate_episode(system_params: dict, sim_params: dict, seed: int, snr_db_by_user: list[float], method: str) -> dict:
    params = with_monte_carlo_sample_snr_by_user(system_params, snr_db_by_user)
    system = DownlinkSystem(params, seed=int(seed))
    scenario = build_experiment_scenario(params, sim_params, seed=int(seed))
    scenario_mode = str(scenario["mode"])
    streaming_targets = np.asarray(scenario.get("streaming_bit_targets_by_block", []), dtype=int)
    random_baseline_failure = ""
    try:
        if scenario_mode == STREAMING_MODE:
            initial_latency, _, _ = estimate_random_precoder_streaming_latency(
                system, sim_params, scenario
            )
        else:
            initial_latency, _, _ = estimate_random_precoder_payload_latency(system, sim_params)
        random_baseline_completed = True
    except RuntimeError as error:
        initial_latency = []
        random_baseline_completed = False
        random_baseline_failure = str(error)
    remaining = np.asarray(system.B, dtype=int).copy()
    n_plan = [[] for _ in range(system.K)]
    bit_plan = [[] for _ in range(system.K)]
    max_blocks = int(sim_params["max_total_blocks"])

    blocks_to_run = int(scenario["number_of_blocks"]) if scenario_mode == STREAMING_MODE else max_blocks
    for block in range(blocks_to_run):
        if scenario_mode != STREAMING_MODE and not np.any(remaining > 0):
            break
        requested = (
            streaming_targets[:, int(block)]
            if scenario_mode == STREAMING_MODE
            else remaining
        )
        active = [k for k in range(system.K) if int(requested[k]) > 0]
        for k in range(system.K):
            system.ensure_block(k, block)
        beams = build_joint_linear_precoders(system, block, active, method)
        snapshot = system.clone_precoders()
        for k in range(system.K):
            snapshot[k][block] = beams[k]
        for k in active:
            T_k = int(system.T[k])
            rate_T = float(system.compute_block_rate(k, block, T_k, F_override=snapshot))
            served = max(0, min(int(requested[k]), int(np.floor(T_k * rate_T))))
            n_used = T_k
            if 0 < served >= int(requested[k]):
                for candidate in range(int(sim_params["n_kl_min"]), T_k + 1):
                    rate = float(system.compute_block_rate(k, block, candidate, F_override=snapshot))
                    if float(served) / float(candidate) <= rate:
                        n_used = candidate
                        break
            n_plan[k].append(int(n_used))
            bit_plan[k].append(int(served))
            if scenario_mode != STREAMING_MODE:
                remaining[k] -= int(served)
    if scenario_mode != STREAMING_MODE and np.any(remaining > 0):
        raise RuntimeError(f"{method.upper()} hit max_total_blocks with remaining bits {remaining.tolist()}.")

    final_latency = [float(sum(n_plan[k]) / system.fs[k]) for k in range(system.K)]
    initial_total = float(sum(initial_latency))
    final_total = float(sum(final_latency))
    pairs = [abs(final_latency[i] - final_latency[j]) for i in range(system.K) for j in range(i + 1, system.K)]
    return {
        "seed": int(seed), "snr_db_by_user": [float(v) for v in snr_db_by_user],
        "final_total_latency_seconds": final_total,
        "latency_reduction_vs_random_percent": (
            100.0 * (initial_total - final_total) / initial_total
            if random_baseline_completed and initial_total > 0.0
            else float("nan")
        ),
        "random_baseline_completed": bool(random_baseline_completed),
        "random_baseline_failure": random_baseline_failure,
        "final_asynchronality_seconds": float(sum(pairs)),
        "final_latency_per_user_seconds": final_latency,
        "scenario_mode": scenario_mode,
        "n_kl_per_user": n_plan, "bits_per_user": bit_plan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate downlink ZF/RZF on a saved held-out test dataset.")
    parser.add_argument("--method", choices=("zf", "rzf"), required=True)
    parser.add_argument("--cfg_name", default="downlink_dispersion_heavy.yaml")
    parser.add_argument("--test_manifest", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.test_manifest).read_text(encoding="utf-8"))
    system_params, sim_params, run_meta = load_config(args.cfg_name)
    episodes = [_evaluate_episode(system_params, sim_params, int(seed), manifest["snr_db_by_user_by_seed"][str(seed)], args.method) for seed in manifest["seeds"]]
    aggregate = {"method": args.method, "test_manifest": str(Path(args.test_manifest).resolve()), "episodes": episodes, "completed_at_local": current_local_timestamp()}
    for metric in ("final_total_latency_seconds", "latency_reduction_vs_random_percent", "final_asynchronality_seconds"):
        values = np.asarray([row[metric] for row in episodes], dtype=float)
        finite_values = values[np.isfinite(values)]
        aggregate[metric] = {
            "mean": float(np.mean(finite_values)) if finite_values.size else float("nan"),
            "std": float(np.std(finite_values)) if finite_values.size else float("nan"),
            "valid_episode_count": int(finite_values.size),
        }
    result_tag = make_method_result_tag(
        f"benchmark_{args.method}_testset{len(episodes)}",
        run_meta["cfg_stem"],
        cfg_hash=run_meta.get("cfg_hash"),
    )
    result_root = build_experiment_root(
        "Downlink",
        f"Benchmark {args.method.upper()}",
        result_tag,
        scenario_mode=str(sim_params["experiment_scenario_mode"]),
    )
    output_dir = result_root / "testing" / "data"
    save_json(aggregate, str(output_dir / "test_dataset_summary.json"))
    save_text([f"Downlink {args.method.upper()} held-out test dataset summary", f"Total held-out channel episodes: {len(episodes)}"] + [f"{key}: {value}" for key, value in aggregate.items() if key in ("final_total_latency_seconds", "latency_reduction_vs_random_percent", "final_asynchronality_seconds")], str(output_dir / "test_dataset_summary.txt"))
    write_result_manifest(
        result_root,
        setup={
            "link": "Downlink",
            "method": f"Benchmark {args.method.upper()}",
            "scenario": sim_params["experiment_scenario_mode"],
            "config_path": run_meta["cfg_path"],
            "config_hash": run_meta.get("cfg_hash"),
            "test_seeds": [int(seed) for seed in manifest["seeds"]],
        },
    )
    print(f"Saved held-out {args.method.upper()} summary to: {result_root}")


if __name__ == "__main__":
    main()
