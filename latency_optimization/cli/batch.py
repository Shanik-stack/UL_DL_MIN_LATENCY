from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from latency_optimization.project import BATCH_RUN_CONFIG_ROOT, EXPERIMENT_CONFIG_ROOT, PROJECT_ROOT
from latency_optimization.core.validation import require_bool, require_choice

METHOD_SPECS: dict[tuple[str, str], dict[str, Any]] = {
    (
        "uplink",
        "convergence",
    ): {
        "display_name": "Uplink | Convergence per epoch",
        "supports_quiet": True,
    },
    (
        "uplink",
        "monte_carlo",
    ): {
        "display_name": "Uplink | Monte Carlo",
        "supports_quiet": False,
    },
    (
        "downlink",
        "convergence",
    ): {
        "display_name": "Downlink | Convergence per epoch",
        "supports_quiet": True,
    },
    (
        "downlink",
        "monte_carlo",
    ): {
        "display_name": "Downlink | Monte Carlo",
        "supports_quiet": True,
    },
}


def _resolve_batch_run_path(batch_run_name: str) -> Path:
    raw = str(batch_run_name)
    candidate = Path(raw)
    if candidate.suffix != ".yaml":
        raise ValueError(f"Batch-run name must end with '.yaml'; got {batch_run_name!r}.")

    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                PROJECT_ROOT / candidate,
                BATCH_RUN_CONFIG_ROOT / candidate,
                Path.cwd() / candidate,
            ]
        )

    for path in candidates:
        if path.exists():
            return path.resolve()

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find batch-run file '{batch_run_name}'. Searched:\n{searched}")


def _validate_link(link_name: str) -> str:
    return require_choice(link_name, {"uplink", "downlink"}, "link")


def _validate_method(method_name: str) -> str:
    return require_choice(method_name, {"convergence", "monte_carlo"}, "method")


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return []
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _as_int_list(value: Any) -> list[int]:
    return [int(item) for item in _as_string_list(value)]


def _as_extra_arg_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _discover_config_names_for_link(link_name: str) -> list[str]:
    prefix = "uplink_" if link_name == "uplink" else "downlink_"
    config_names = sorted(
        path.name
        for path in EXPERIMENT_CONFIG_ROOT.glob(f"{prefix}*.yaml")
        if path.is_file()
    )
    if len(config_names) == 0:
        raise FileNotFoundError(f"No config files found for link '{link_name}' under {EXPERIMENT_CONFIG_ROOT}.")
    return config_names


def _expand_cfg_names(raw_cfg_names: Any, *, link_name: str) -> list[str]:
    cfg_names = _as_string_list(raw_cfg_names)
    if len(cfg_names) == 0:
        return _discover_config_names_for_link(link_name)
    return cfg_names


def _validate_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    return require_bool(value, "batch boolean option")


def _validate_launch_mode(value: Any) -> str:
    if value is None:
        return "parallel_terminals"
    return require_choice(value, {"parallel_terminals", "sequential"}, "launch_mode")


def _build_command(
    *,
    python_executable: str,
    link_name: str,
    method_name: str,
    cfg_name: str,
    seed: int | None,
    train_seeds: list[int],
    num_train_channels: int | None,
    num_train_blocks: int | None,
    test_seed: int | None,
    num_test_channels: int | None,
    num_test_blocks: int | None,
    quiet: bool,
    skip_test: bool,
    extra_args: list[str],
) -> list[str]:
    spec = METHOD_SPECS[(link_name, method_name)]
    cmd = [
        python_executable,
        "-m",
        "latency_optimization",
        "run",
        "--link",
        link_name,
        "--method",
        method_name,
        "--cfg_name",
        str(cfg_name),
    ]
    if method_name == "convergence":
        if seed is None:
            raise ValueError(f"{link_name} convergence requires 'seed'.")
        cmd.extend(["--seed", str(int(seed))])
    else:
        if num_train_channels is not None and num_train_blocks is not None:
            raise ValueError("Set either num_train_channels or num_train_blocks, not both.")
        if num_test_channels is not None and num_test_blocks is not None:
            raise ValueError("Set either num_test_channels or num_test_blocks, not both.")
        if len(train_seeds) > 0:
            cmd.extend(["--train_seeds", ",".join(str(int(v)) for v in train_seeds)])
        elif num_train_channels is not None:
            cmd.extend(["--num_train_channels", str(int(num_train_channels))])
        elif num_train_blocks is not None:
            cmd.extend(["--num_train_blocks", str(int(num_train_blocks))])
        if test_seed is not None:
            cmd.extend(["--test_seed", str(int(test_seed))])
        if num_test_channels is not None:
            cmd.extend(["--num_test_channels", str(int(num_test_channels))])
        elif num_test_blocks is not None:
            cmd.extend(["--num_test_blocks", str(int(num_test_blocks))])
        if skip_test:
            cmd.append("--skip_test")
    if quiet and bool(spec.get("supports_quiet", False)):
        cmd.append("--quiet")
    cmd.extend(extra_args)
    return cmd


def _load_batch_run(batch_run_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with batch_run_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Batch-run file must be a mapping with 'defaults' and 'runs'.")
    defaults = payload.get("defaults", {})
    runs = payload.get("runs", [])
    if not isinstance(defaults, dict):
        raise ValueError("Batch-run 'defaults' must be a mapping.")
    if not isinstance(runs, list) or len(runs) == 0:
        raise ValueError("Batch-run file must contain a non-empty 'runs' list.")
    normalized_runs: list[dict[str, Any]] = []
    for idx, raw_run in enumerate(runs):
        if not isinstance(raw_run, dict):
            raise ValueError(f"Batch-run entry at index {idx} must be a mapping.")
        normalized_runs.append(dict(raw_run))
    return defaults, normalized_runs


def _build_invocations(
    *,
    defaults: dict[str, Any],
    runs: list[dict[str, Any]],
    python_override: str | None,
) -> list[dict[str, Any]]:
    default_python = str(defaults.get("python_executable", sys.executable))
    invocations: list[dict[str, Any]] = []
    for raw_run in runs:
        link_name = _validate_link(raw_run.get("link"))
        method_name = _validate_method(raw_run.get("method"))
        if not _validate_bool(raw_run.get("enabled", True), default=True):
            continue

        cfg_names = _expand_cfg_names(raw_run.get("cfg_names"), link_name=link_name)
        seed = raw_run.get("seed", defaults.get("seed"))
        test_seed = raw_run.get("test_seed", defaults.get("test_seed"))
        train_seeds = _as_int_list(raw_run.get("train_seeds", defaults.get("train_seeds", [])))
        raw_num_train_channels = raw_run.get("num_train_channels", defaults.get("num_train_channels"))
        num_train_channels = int(raw_num_train_channels) if raw_num_train_channels is not None else None
        raw_num_train_blocks = raw_run.get("num_train_blocks", defaults.get("num_train_blocks"))
        num_train_blocks = int(raw_num_train_blocks) if raw_num_train_blocks is not None else None
        raw_num_test_channels = raw_run.get("num_test_channels", defaults.get("num_test_channels"))
        num_test_channels = int(raw_num_test_channels) if raw_num_test_channels is not None else None
        raw_num_test_blocks = raw_run.get("num_test_blocks", defaults.get("num_test_blocks"))
        num_test_blocks = int(raw_num_test_blocks) if raw_num_test_blocks is not None else None
        quiet = _validate_bool(raw_run.get("quiet", defaults.get("quiet", False)), default=False)
        skip_test = _validate_bool(raw_run.get("skip_test", defaults.get("skip_test", False)), default=False)
        extra_args = _as_extra_arg_list(raw_run.get("extra_args", []))
        python_executable = str(python_override or raw_run.get("python_executable", default_python))

        for cfg_name in cfg_names:
            command = _build_command(
                python_executable=python_executable,
                link_name=link_name,
                method_name=method_name,
                cfg_name=cfg_name,
                seed=int(seed) if seed is not None else None,
                train_seeds=train_seeds,
                num_train_channels=num_train_channels,
                num_train_blocks=num_train_blocks,
                test_seed=int(test_seed) if test_seed is not None else None,
                num_test_channels=num_test_channels,
                num_test_blocks=num_test_blocks,
                quiet=quiet,
                skip_test=skip_test,
                extra_args=extra_args,
            )
            invocations.append(
                {
                    "link": link_name,
                    "method": method_name,
                    "cfg_name": str(cfg_name),
                    "display_name": str(METHOD_SPECS[(link_name, method_name)]["display_name"]),
                    "command": command,
                }
            )
    return invocations


def _validate_filter_set(raw_value: str | None, *, kind: str) -> set[str]:
    values = _as_string_list(raw_value)
    if len(values) == 0:
        return set()
    if kind == "link":
        return {_validate_link(value) for value in values}
    if kind == "method":
        return {_validate_method(value) for value in values}
    return {str(value).strip() for value in values}


def _filter_invocations(
    invocations: list[dict[str, Any]],
    *,
    link_filter: set[str],
    method_filter: set[str],
    cfg_filter: set[str],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for invocation in invocations:
        cfg_name = str(invocation["cfg_name"])
        cfg_stem = Path(cfg_name).stem
        if len(link_filter) > 0 and str(invocation["link"]) not in link_filter:
            continue
        if len(method_filter) > 0 and str(invocation["method"]) not in method_filter:
            continue
        if len(cfg_filter) > 0 and cfg_name not in cfg_filter and cfg_stem not in cfg_filter:
            continue
        filtered.append(invocation)
    return filtered


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _print_plan(invocations: list[dict[str, Any]], *, launch_mode: str) -> None:
    print("")
    print("Batch experiment plan")
    print(f"Launch mode: {launch_mode}")
    print(f"Total runs: {len(invocations)}")
    for idx, invocation in enumerate(invocations, start=1):
        print(
            f"{idx:02d}. {invocation['display_name']} | cfg={invocation['cfg_name']} | "
            f"cmd={_format_command(invocation['command'])}"
        )


def _quote_powershell_literal(text: str) -> str:
    return str(text).replace("'", "''")


def _format_powershell_invocation(command: list[str]) -> str:
    return "& " + " ".join(f"'{_quote_powershell_literal(part)}'" for part in command)


def _build_terminal_script(invocation: dict[str, Any], *, index: int, total: int) -> str:
    label = f"[{index}/{total}] {invocation['display_name']} | cfg={invocation['cfg_name']}"
    label_q = _quote_powershell_literal(label)
    cwd_q = _quote_powershell_literal(str(PROJECT_ROOT))
    run_expr = _format_powershell_invocation(invocation["command"])
    return (
        f"$runLabel = '{label_q}'; "
        f"$Host.UI.RawUI.WindowTitle = $runLabel; "
        f"Set-Location '{cwd_q}'; "
        f"Write-Host $runLabel -ForegroundColor Cyan; "
        f"{run_expr}; "
        f"$exitCode = $LASTEXITCODE; "
        f"Write-Host ''; "
        f"if ($exitCode -eq 0) {{ "
        f"Write-Host 'Experiment finished successfully.' -ForegroundColor Green "
        f"}} else {{ "
        f"Write-Host ('Experiment failed with exit code ' + $exitCode) -ForegroundColor Red "
        f"}}; "
        f"Read-Host 'Press Enter to close'; "
        f"exit $exitCode"
    )


def _launch_in_new_terminal(invocation: dict[str, Any], *, index: int, total: int) -> subprocess.Popen[Any]:
    if os.name != "nt":
        raise RuntimeError("parallel_terminals launch mode currently supports Windows only.")
    terminal_script = _build_terminal_script(invocation, index=index, total=total)
    powershell_cmd = [
        "powershell.exe",
        "-NoLogo",
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        terminal_script,
    ]
    return subprocess.Popen(
        powershell_cmd,
        cwd=str(PROJECT_ROOT),
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run uplink/downlink experiments from one batch-run file.")
    parser.add_argument(
        "--batch_run",
        dest="batch_run",
        type=str,
        default="run_all.yaml",
        help="Batch-run YAML name or path. Defaults to configs/batch_runs/run_all.yaml",
    )
    parser.add_argument("--python_exe", type=str, default=None, help="Override Python executable used for child runs.")
    parser.add_argument("--links", type=str, default=None, help="Optional filter, e.g. uplink or uplink,downlink")
    parser.add_argument("--methods", type=str, default=None, help="Optional filter, e.g. convergence or convergence,monte_carlo")
    parser.add_argument("--configs", type=str, default=None, help="Optional filter by cfg filename or stem, comma-separated")
    parser.add_argument(
        "--launch_mode",
        type=str,
        default=None,
        help="Launch mode: parallel_terminals or sequential. Defaults to the batch-run setting.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--continue_on_error", action="store_true", help="Keep running remaining experiments after a failure.")
    args = parser.parse_args(argv)

    batch_run_path = _resolve_batch_run_path(args.batch_run)
    defaults, runs = _load_batch_run(batch_run_path)
    launch_mode = _validate_launch_mode(args.launch_mode or defaults.get("launch_mode", "parallel_terminals"))
    invocations = _build_invocations(
        defaults=defaults,
        runs=runs,
        python_override=args.python_exe,
    )
    invocations = _filter_invocations(
        invocations,
        link_filter=_validate_filter_set(args.links, kind="link"),
        method_filter=_validate_filter_set(args.methods, kind="method"),
        cfg_filter=_validate_filter_set(args.configs, kind="cfg"),
    )
    if len(invocations) == 0:
        print("No experiment runs matched the batch-run file and filters.")
        return 1

    print(f"Loaded batch-run file: {batch_run_path}")
    _print_plan(invocations, launch_mode=launch_mode)
    if args.dry_run:
        return 0

    if launch_mode == "parallel_terminals":
        launched: list[dict[str, Any]] = []
        for idx, invocation in enumerate(invocations, start=1):
            print("")
            print("=" * 100)
            print(f"Launching [{idx}/{len(invocations)}] {invocation['display_name']} | cfg={invocation['cfg_name']}")
            print(_format_command(invocation["command"]))
            print("=" * 100)
            process = _launch_in_new_terminal(invocation, index=idx, total=len(invocations))
            launched.append(
                {
                    "index": idx,
                    "display_name": invocation["display_name"],
                    "cfg_name": invocation["cfg_name"],
                    "pid": int(process.pid),
                }
            )

        print("")
        print("Batch experiment launch summary")
        print(f"Requested runs: {len(invocations)}")
        print(f"Launched terminal windows: {len(launched)}")
        print("Each experiment is now running in its own PowerShell window.")
        print("Those windows stay open after completion so you can inspect the logs.")
        for item in launched:
            print(
                f"- run {item['index']}: {item['display_name']} | "
                f"cfg={item['cfg_name']} | terminal_pid={item['pid']}"
            )
        return 0

    failures: list[dict[str, Any]] = []
    for idx, invocation in enumerate(invocations, start=1):
        print("")
        print("=" * 100)
        print(f"[{idx}/{len(invocations)}] {invocation['display_name']} | cfg={invocation['cfg_name']}")
        print(_format_command(invocation["command"]))
        print("=" * 100)
        completed = subprocess.run(
            invocation["command"],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        if completed.returncode != 0:
            failures.append(
                {
                    "index": idx,
                    "display_name": invocation["display_name"],
                    "cfg_name": invocation["cfg_name"],
                    "returncode": int(completed.returncode),
                }
            )
            print(
                f"FAILED [{idx}/{len(invocations)}] {invocation['display_name']} | "
                f"cfg={invocation['cfg_name']} | returncode={completed.returncode}"
            )
            if not args.continue_on_error:
                break

    print("")
    print("Batch experiment summary")
    print(f"Requested runs: {len(invocations)}")
    print(f"Failed runs: {len(failures)}")
    if len(failures) == 0:
        print("All runs completed successfully.")
        return 0

    for failure in failures:
        print(
            f"- run {failure['index']}: {failure['display_name']} | "
            f"cfg={failure['cfg_name']} | returncode={failure['returncode']}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
