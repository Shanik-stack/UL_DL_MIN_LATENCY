from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(data: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(make_serializable(data), output_file, indent=4)


def save_text(lines: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines))


def remove_empty_directories(root: str | Path) -> list[str]:
    """Remove empty result folders and return their relative paths."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    removed: list[str] = []
    for directory in sorted(
        (path for path in root_path.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            continue
        removed.append(str(directory.relative_to(root_path)))
    return removed


def write_result_manifest(
    result_root: str | Path,
    *,
    setup: dict[str, Any],
    status: str = "complete",
) -> dict[str, Any]:
    """Write a setup-first index and validate the generated result artifacts."""
    root = Path(result_root)
    root.mkdir(parents=True, exist_ok=True)
    removed_empty_directories = remove_empty_directories(root)
    artifact_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"run_manifest.json", "run_manifest.txt"}
    )
    artifact_files = [str(path.relative_to(root)) for path in artifact_paths]
    files = sorted(artifact_files + ["run_manifest.json", "run_manifest.txt"])
    zero_byte_files = [
        str(path.relative_to(root)) for path in artifact_paths if path.stat().st_size == 0
    ]
    file_counts_by_section: dict[str, int] = {}
    for relative_path in artifact_files:
        section = Path(relative_path).parts[0] if Path(relative_path).parts else "root"
        file_counts_by_section[section] = file_counts_by_section.get(section, 0) + 1
    plot_count = sum(Path(path).suffix.lower() == ".png" for path in artifact_files)
    if not artifact_files:
        artifact_health = "no_result_artifacts"
    elif zero_byte_files:
        artifact_health = "zero_byte_files_detected"
    else:
        artifact_health = "complete"
    manifest = {
        "status": str(status),
        "result_root": str(root.resolve()),
        "setup": make_serializable(setup),
        "file_count": int(len(files)),
        "result_artifact_count": int(len(artifact_files)),
        "plot_count": int(plot_count),
        "file_counts_by_section": file_counts_by_section,
        "artifact_health": artifact_health,
        "zero_byte_files": zero_byte_files,
        "files": files,
        "removed_empty_directories": removed_empty_directories,
        "written_at_local": current_local_timestamp(),
    }
    setup_lines = [
        f"{key}: {json.dumps(make_serializable(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else value}"
        for key, value in setup.items()
    ]
    section_lines = [
        f"- {section}: {count}"
        for section, count in sorted(file_counts_by_section.items())
    ] or ["- none"]
    grouped_file_lines: list[str] = []
    grouped_files: dict[str, list[str]] = {}
    for relative_path in files:
        parts = Path(relative_path).parts
        section = parts[0] if len(parts) > 1 else "root"
        grouped_files.setdefault(section, []).append(relative_path)
    for section, section_files in sorted(grouped_files.items()):
        if grouped_file_lines:
            grouped_file_lines.append("")
        grouped_file_lines.append(f"[{section}]")
        grouped_file_lines.extend(f"- {path}" for path in section_files)
    save_text(
        [
            "Run manifest",
            "",
            "Setup",
            *setup_lines,
            "",
            "Artifact summary",
            f"Status: {manifest['status']}",
            f"Artifact health: {manifest['artifact_health']}",
            f"Result root: {manifest['result_root']}",
            f"Files recorded: {manifest['file_count']}",
            f"Result artifacts: {manifest['result_artifact_count']}",
            f"Plots: {manifest['plot_count']}",
            f"Empty directories removed: {len(removed_empty_directories)}",
            f"Zero-byte files: {len(zero_byte_files)}",
            "",
            "Files by section",
            *section_lines,
            "",
            "Files",
            *grouped_file_lines,
        ],
        str(root / "run_manifest.txt"),
    )
    save_json(manifest, str(root / "run_manifest.json"))
    return manifest


def current_local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
