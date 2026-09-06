from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from latency_optimization.project import BENCHMARK_CONFIG_ROOT, PROJECT_ROOT

from .exhaustive_payload_compare import (
    CATALOG_MODE_FULL,
    CATALOG_MODE_NONE,
    CATALOG_MODE_OUTCOMES_ONLY,
    run_exhaustive_payload_compare,
)


RESULTS_ROOT = PROJECT_ROOT / "Results" / "Uplink" / "Benchmark Methods" / "Exhaustive Search"
DEFAULT_CONFIG = str(BENCHMARK_CONFIG_ROOT / "uplink_exhaustive_dispersion_heavy.yaml")


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d__%H%M%S")


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return sorted({int(value) for value in args.seeds.split(",") if value.strip()})
    if args.seed_start is None or args.seed_end is None:
        raise ValueError("Provide --seeds or both --seed_start and --seed_end.")
    if args.seed_end < args.seed_start:
        raise ValueError("--seed_end must be greater than or equal to --seed_start.")
    return list(range(args.seed_start, args.seed_end + 1))


def _load_validator():
    return run_exhaustive_payload_compare


def _score(strategy: dict[str, Any]) -> tuple[bool, int, int]:
    remaining = sum(max(0, int(bits)) for bits in strategy.get("per_user_remaining_bits", []))
    return (
        bool(strategy.get("all_completed", False)),
        -int(strategy.get("global_latency_sum", 10**12)),
        -remaining,
    )


def _evaluate(seed: int, cfg_name: str, catalog_mode: str) -> dict[str, Any]:
    result = _load_validator()(
        cfg_name=cfg_name,
        seed=seed,
        catalog_detail_mode=catalog_mode,
    )
    online = result.get("online_strategy", {})
    exhaustive = result.get("exhaustive_strategy", {})
    online_latency = int(online.get("global_latency_sum", 0))
    exhaustive_latency = int(exhaustive.get("global_latency_sum", 0))
    return {
        "seed": seed,
        "online_latency_symbols": online_latency,
        "exhaustive_latency_symbols": exhaustive_latency,
        "latency_gap_symbols": online_latency - exhaustive_latency,
        "online_completed": bool(online.get("all_completed", False)),
        "exhaustive_completed": bool(exhaustive.get("all_completed", False)),
        "online_remaining_bits": sum(max(0, int(bits)) for bits in online.get("per_user_remaining_bits", [])),
        "exhaustive_remaining_bits": sum(
            max(0, int(bits)) for bits in exhaustive.get("per_user_remaining_bits", [])
        ),
        "exhaustive_is_better": _score(exhaustive) > _score(online),
        "result": result,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [key for key in rows[0] if key != "result"] if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Uplink online allocation with exhaustive allocation over seeds.")
    parser.add_argument("--cfg_name", default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", default="", help="Comma-separated seeds, for example 0,1,2.")
    parser.add_argument("--seed_start", type=int)
    parser.add_argument("--seed_end", type=int)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument(
        "--catalog_mode",
        choices=[CATALOG_MODE_NONE, CATALOG_MODE_OUTCOMES_ONLY, CATALOG_MODE_FULL],
        default=CATALOG_MODE_FULL,
        help="Detail stored for exhaustive outcomes: none, outcomes_only, or full.",
    )
    args = parser.parse_args()

    seeds = _parse_seeds(args)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    output_dir = RESULTS_ROOT / f"seed_comparison__{_timestamp()}"
    per_seed_dir = output_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    print(f"Comparing seeds {seeds}. Results: {output_dir}", flush=True)

    rows: list[dict[str, Any]] = []
    workers = max(1, args.max_workers)
    if workers == 1:
        for seed in seeds:
            rows.append(_evaluate(seed, args.cfg_name, args.catalog_mode))
            print(f"Completed seed {seed}.", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            jobs = {
                executor.submit(_evaluate, seed, args.cfg_name, args.catalog_mode): seed
                for seed in seeds
            }
            for job in as_completed(jobs):
                row = job.result()
                rows.append(row)
                print(f"Completed seed {row['seed']}.", flush=True)

    rows.sort(key=lambda row: row["seed"])
    for row in rows:
        (per_seed_dir / f"seed_{row['seed']:04d}.json").write_text(
            json.dumps(row["result"], indent=2), encoding="utf-8"
        )

    summary = {
        "config": args.cfg_name,
        "seeds": seeds,
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rows": [{key: value for key, value in row.items() if key != "result"} for row in rows],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(rows, output_dir / "summary.csv")
    print(f"Saved comparison summary to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
