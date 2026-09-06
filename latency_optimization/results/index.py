"""Create a compact navigation index for completed result runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latency_optimization.project import RESULTS_ROOT
from latency_optimization.results.persistence import save_json, save_text


def build_results_index(results_root: str | Path = RESULTS_ROOT) -> dict[str, object]:
    root = Path(results_root).resolve()
    runs: list[dict[str, object]] = []
    for manifest_path in sorted(root.glob("*/**/run_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        relative_root = manifest_path.parent.relative_to(root)
        setup = manifest.get("setup", {})
        if not isinstance(setup, dict):
            setup = {}
        runs.append(
            {
                "path": str(relative_root),
                "status": manifest.get("status", "unknown"),
                "link": setup.get("link", "unknown"),
                "scenario": setup.get("scenario", "unknown"),
                "method": setup.get("method", "unknown"),
                "config_hash": setup.get("config_hash", "unknown"),
                "seed": setup.get("seed", setup.get("test_seed", "unknown")),
                "file_count": manifest.get("file_count", 0),
                "completed_at_local": setup.get(
                    "run_completed_at_local", manifest.get("written_at_local", "unknown")
                ),
            }
        )
    return {
        "results_root": str(root),
        "run_count": len(runs),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Index completed experiment results.")
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    args = parser.parse_args()
    index = build_results_index(args.results_root)
    root = Path(args.results_root).resolve()
    save_json(index, str(root / "index.json"))
    lines = [
        "Experiment result index",
        f"Results root: {root}",
        f"Completed runs indexed: {index['run_count']}",
        "",
        "Runs:",
    ]
    for run in index["runs"]:
        lines.append(
            " | ".join(
                [
                    str(run["path"]),
                    f"method={run['method']}",
                    f"scenario={run['scenario']}",
                    f"hash={run['config_hash']}",
                    f"seed={run['seed']}",
                    f"files={run['file_count']}",
                ]
            )
        )
    save_text(lines, str(root / "index.txt"))
    print(f"Indexed {index['run_count']} completed runs under {root}")


if __name__ == "__main__":
    main()
