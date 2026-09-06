"""Evaluate the standard uplink ZF/RZF beamformers on a saved held-out set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from latency_optimization.experiments.channels import with_monte_carlo_sample_snr_by_user
from latency_optimization.results.paths import build_experiment_root
from latency_optimization.results.persistence import (
    current_local_timestamp,
    save_json,
    save_text,
    write_result_manifest,
)
from latency_optimization.results.naming import make_method_result_tag

from .linear_beamforming import run_uplink_closed_form_benchmark
from ..config import load_config


def _aggregate(episodes: list[dict]) -> dict:
    metric_names = {
        "final_total_latency_seconds": "final_total_latency",
        "latency_reduction_vs_random_percent": "total_latency_reduction_percent",
        "final_asynchronality_seconds": "final_asynchronality_sum",
    }
    summary: dict[str, object] = {"episodes": episodes}
    for output_name, source_name in metric_names.items():
        values = np.asarray([row["summary_metrics"][source_name] for row in episodes], dtype=float)
        summary[f"mean_{output_name}"] = float(np.mean(values))
        summary[f"std_{output_name}"] = float(np.std(values))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Uplink ZF or RZF on a saved held-out test dataset.")
    parser.add_argument("--method", choices=("zf", "rzf"), required=True)
    parser.add_argument("--cfg_name", default="uplink_dispersion_heavy.yaml")
    parser.add_argument("--test_manifest", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.test_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _, _, run_meta = load_config(args.cfg_name)
    tag = make_method_result_tag(
        f"benchmark_{args.method}_testset{len(manifest['seeds'])}",
        run_meta["cfg_stem"],
        cfg_hash=run_meta.get("cfg_hash"),
    )
    result_root = build_experiment_root(
        "Uplink",
        f"Benchmark {args.method.upper()}",
        tag,
        scenario_mode="payload",
    )
    output_dir = result_root / "testing" / "data"

    episodes: list[dict] = []
    for seed in manifest["seeds"]:
        snr_db_by_user = manifest["snr_db_by_user_by_seed"][str(seed)]
        system_params, _, _ = load_config(args.cfg_name)
        experiment = run_uplink_closed_form_benchmark(
            method_key=args.method,
            cfg_name=args.cfg_name,
            seed=int(seed),
            verbose=False,
            system_params_override=with_monte_carlo_sample_snr_by_user(system_params, snr_db_by_user),
        )
        result = experiment["result"]
        result["test_snr_db_by_user"] = [float(value) for value in snr_db_by_user]
        episodes.append({
            "seed": int(seed),
            "snr_db_by_user": result["test_snr_db_by_user"],
            "summary_metrics": result["summary_metrics"],
        })

    aggregate = _aggregate(episodes)
    aggregate.update({
        "method": args.method,
        "test_manifest": str(manifest_path),
        "completed_at_local": current_local_timestamp(),
    })
    save_json(aggregate, str(output_dir / "test_dataset_summary.json"))
    lines = [
        f"Uplink {args.method.upper()} held-out test dataset summary",
        f"Total held-out channel episodes: {len(episodes)}",
    ]
    for key, value in aggregate.items():
        if key.startswith(("mean_", "std_")):
            lines.append(f"{key}: {float(value):.6f}")
    save_text(lines, str(output_dir / "test_dataset_summary.txt"))
    write_result_manifest(
        result_root,
        setup={
            "link": "Uplink",
            "method": f"Benchmark {args.method.upper()}",
            "scenario": "payload",
            "config_path": run_meta["cfg_path"],
            "config_hash": run_meta.get("cfg_hash"),
            "test_seeds": [int(seed) for seed in manifest["seeds"]],
        },
    )
    print(f"Saved held-out {args.method.upper()} summary to: {result_root}")


if __name__ == "__main__":
    main()
