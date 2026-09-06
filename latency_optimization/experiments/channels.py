from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def build_training_snr_schedule(
    train_seeds: Sequence[int],
    snr_db_ranges: Sequence[Sequence[float]],
    *,
    num_users: int,
) -> dict[int, list[float]]:
    """Assign every user a stratified SNR schedule with a deterministic offset."""
    if len(snr_db_ranges) != int(num_users):
        raise ValueError(
            "monte_carlo_training_snr_db_ranges must provide one [min_db, max_db] range per user."
        )

    ranges: list[tuple[float, float]] = []
    for user, snr_db_range in enumerate(snr_db_ranges):
        if len(snr_db_range) != 2:
            raise ValueError(
                f"monte_carlo_training_snr_db_ranges[{user}] must contain [min_db, max_db]."
            )
        minimum, maximum = (float(value) for value in snr_db_range)
        if maximum < minimum:
            raise ValueError(
                f"monte_carlo_training_snr_db_ranges[{user}] maximum must be at least its minimum."
            )
        ranges.append((minimum, maximum))

    num_samples = len(train_seeds)
    if num_samples == 0:
        return {}
    schedule: dict[int, list[float]] = {}
    for sample_index, seed in enumerate(train_seeds):
        snr_by_user: list[float] = []
        for user, (minimum, maximum) in enumerate(ranges):
            # Offsets decorrelate user SNRs while keeping each user's coverage uniform.
            grid_index = (sample_index + (user * num_samples) // int(num_users)) % num_samples
            fraction = 0.0 if num_samples == 1 else grid_index / (num_samples - 1)
            snr_by_user.append(float(minimum + (maximum - minimum) * fraction))
        schedule[int(seed)] = snr_by_user
    return schedule


def build_test_snr_schedule(
    test_seeds: Sequence[int],
    snr_db_ranges: Sequence[Sequence[float]],
    *,
    num_users: int,
) -> dict[int, list[float]]:
    """Build held-out, midpoint-stratified SNR vectors for every test sample."""
    if len(snr_db_ranges) != int(num_users):
        raise ValueError(
            "monte_carlo_test_snr_db_ranges must provide one [min_db, max_db] range per user."
        )
    ranges = [[float(value) for value in snr_db_range] for snr_db_range in snr_db_ranges]
    for user, snr_db_range in enumerate(ranges):
        if len(snr_db_range) != 2 or snr_db_range[1] < snr_db_range[0]:
            raise ValueError(
                f"monte_carlo_test_snr_db_ranges[{user}] must contain an ordered [min_db, max_db] range."
            )

    num_samples = len(test_seeds)
    if num_samples == 0:
        return {}
    schedule: dict[int, list[float]] = {}
    for sample_index, seed in enumerate(test_seeds):
        values: list[float] = []
        for user, (minimum, maximum) in enumerate(ranges):
            grid_index = (sample_index + (user * num_samples) // int(num_users)) % num_samples
            fraction = (grid_index + 0.5) / num_samples
            values.append(float(minimum + (maximum - minimum) * fraction))
        schedule[int(seed)] = values
    return schedule


def with_monte_carlo_sample_snr_by_user(
    system_params: dict[str, Any],
    snr_db_by_user: Sequence[float],
) -> dict[str, Any]:
    """Copy system parameters and apply one SNR value to each user."""
    params = copy.copy(system_params)
    original = system_params["snr_db"]
    user_count = int(system_params.get("K", len(original) if hasattr(original, "__len__") else 1))
    if len(snr_db_by_user) != user_count:
        raise ValueError(f"Expected {user_count} episode SNR values, received {len(snr_db_by_user)}.")
    if isinstance(original, np.ndarray):
        params["snr_db"] = np.asarray(snr_db_by_user, dtype=float)
    else:
        params["snr_db"] = [float(value) for value in snr_db_by_user]
    return params


def save_monte_carlo_base_samples(
    *,
    output_dir: str | Path,
    link_name: str,
    system_params: dict[str, Any],
    sample_seeds: Sequence[int],
    system_factory: Callable[..., Any],
    sample_snr_db_by_user_by_seed: Mapping[int, Sequence[float]],
    dataset_role: str = "training",
    sample_kind: str = "payload_channel_episode",
) -> dict[str, Any]:
    """Save one deterministic Monte Carlo base sample for every dataset seed."""
    sample_units = {
        "payload_channel_episode": "channel_episode",
        "streaming_block": "block",
    }
    if sample_kind not in sample_units:
        raise ValueError(f"Unsupported Monte Carlo sample kind: {sample_kind}.")
    sample_unit = sample_units[sample_kind]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []

    for seed in sample_seeds:
        snr_db_by_user = [float(value) for value in sample_snr_db_by_user_by_seed[int(seed)]]
        system = system_factory(
            with_monte_carlo_sample_snr_by_user(system_params, snr_db_by_user),
            seed=int(seed),
        )
        channels = {
            f"H_user_{user}_block_0": np.asarray(system.H[user][0], dtype=np.complex64)
            for user in range(int(system.K))
        }
        file_name = f"seed_{int(seed):04d}.npz"
        np.savez_compressed(
            root / file_name,
            **channels,
            noise_variance=np.asarray(system.sigma2, dtype=np.float64),
            power_budget=np.asarray(system.P, dtype=np.float64),
            max_blocklength=np.asarray(system.T, dtype=np.int64),
            error_target=np.asarray(system.epsilon, dtype=np.float64),
            episode_snr_db_by_user=np.asarray(snr_db_by_user, dtype=np.float64),
        )
        files.append(
            {
                "seed": int(seed),
                "snr_db_by_user": snr_db_by_user,
                "file": file_name,
                "channel_shapes": {name: list(value.shape) for name, value in channels.items()},
            }
        )

    manifest = {
        "link": str(link_name),
        "dataset_role": str(dataset_role),
        "sample_unit": sample_unit,
        "sample_kind": str(sample_kind),
        "sample_count": int(len(files)),
        "sample_scope": (
            "payload_episode_anchor_block_0"
            if sample_kind == "payload_channel_episode"
            else "independent_streaming_block"
        ),
        "description": (
            "Payload files are channel-episode anchors; later blocks are generated during "
            "the payload rollout. Streaming files are independent one-block training/test "
            "samples. Later blocks are not part of a streaming base sample."
        ),
        "seeds": [int(seed) for seed in sample_seeds],
        "snr_db_by_user_by_seed": {
            str(int(seed)): [float(value) for value in sample_snr_db_by_user_by_seed[int(seed)]]
            for seed in sample_seeds
        },
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
