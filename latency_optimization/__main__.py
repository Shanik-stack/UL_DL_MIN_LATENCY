from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence


RUN_MODULES = {
    ("uplink", "convergence"): "latency_optimization.uplink.methods.convergence.experiment",
    ("uplink", "monte_carlo"): "latency_optimization.uplink.methods.monte_carlo.experiment",
    ("downlink", "convergence"): "latency_optimization.downlink.methods.convergence.experiment",
    ("downlink", "monte_carlo"): "latency_optimization.downlink.methods.monte_carlo.experiment",
}

UPLINK_BENCHMARK_MODULES = {
    "zf": "latency_optimization.uplink.benchmarks.zero_forcing.main",
    "rzf": "latency_optimization.uplink.benchmarks.regularized_zero_forcing.main",
    "exhaustive": "latency_optimization.uplink.benchmarks.exhaustive_search.exhaustive_payload_compare",
}


def _run_module_main(module_name: str, arguments: Sequence[str]) -> int:
    module = importlib.import_module(module_name)
    previous_argv = sys.argv
    try:
        sys.argv = [module_name, *arguments]
        result = module.main()
    finally:
        sys.argv = previous_argv
    return int(result or 0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m latency_optimization",
        description="Run finite-blocklength uplink and downlink experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        add_help=False,
        help="Run one convergence or Monte Carlo experiment.",
    )
    run_parser.add_argument("-h", "--help", action="store_true", dest="show_help")
    run_parser.add_argument("--link", choices=("uplink", "downlink"))
    run_parser.add_argument("--method", choices=("convergence", "monte_carlo"))
    run_parser.set_defaults(command_parser=run_parser)

    subparsers.add_parser("batch", add_help=False, help="Run experiments from a batch-run YAML file.")

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        add_help=False,
        help="Run one benchmark method.",
    )
    benchmark_parser.add_argument("-h", "--help", action="store_true", dest="show_help")
    benchmark_parser.add_argument("--link", choices=("uplink", "downlink"))
    benchmark_parser.add_argument("--name", choices=("zf", "rzf", "exhaustive"))
    benchmark_parser.set_defaults(command_parser=benchmark_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    parsed, remaining = parser.parse_known_args(arguments)

    if parsed.command == "batch":
        from .cli.batch import main as run_batch

        return int(run_batch(remaining))

    if parsed.command == "run":
        if parsed.link is None or parsed.method is None:
            if parsed.show_help:
                parsed.command_parser.print_help()
                return 0
            parser.error("run requires --link and --method")
        key = (str(parsed.link), str(parsed.method))
        method_arguments = [*remaining, *(["--help"] if parsed.show_help else [])]
        return _run_module_main(RUN_MODULES[key], method_arguments)

    if parsed.link is None or parsed.name is None:
        if parsed.show_help:
            parsed.command_parser.print_help()
            return 0
        parser.error("benchmark requires --link and --name")
    link = str(parsed.link)
    benchmark_name = str(parsed.name)
    benchmark_arguments = [*remaining, *(["--help"] if parsed.show_help else [])]
    if benchmark_name == "exhaustive":
        if link != "uplink":
            parser.error("The exhaustive benchmark is currently available for uplink only.")
        return _run_module_main(UPLINK_BENCHMARK_MODULES[benchmark_name], benchmark_arguments)

    use_test_dataset = link == "downlink" or "--test_manifest" in benchmark_arguments
    if use_test_dataset:
        module_name = f"latency_optimization.{link}.benchmarks.evaluate_test_dataset"
        return _run_module_main(module_name, ["--method", benchmark_name, *benchmark_arguments])
    return _run_module_main(UPLINK_BENCHMARK_MODULES[benchmark_name], benchmark_arguments)


if __name__ == "__main__":
    raise SystemExit(main())
