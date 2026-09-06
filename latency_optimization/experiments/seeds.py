from __future__ import annotations

from typing import Any


def parse_seed_list(seed_text: str) -> list[int]:
    return [int(part.strip()) for part in str(seed_text).split(",") if part.strip()]


def parse_optional_seed_values(seed_values: Any) -> list[int]:
    if seed_values is None:
        return []
    if isinstance(seed_values, str):
        text = seed_values.strip()
        return [] if text == "" else parse_seed_list(text)
    if isinstance(seed_values, (list, tuple)):
        return [int(value) for value in seed_values]
    return [int(seed_values)]


def build_train_seeds_from_num_training_samples(
    num_training_samples: int,
    test_seed: int,
) -> list[int]:
    count = int(num_training_samples)
    if count < 1:
        raise ValueError("The Monte Carlo training sample count must be at least 1.")
    seeds: list[int] = []
    candidate = 1
    while len(seeds) < count:
        if candidate != int(test_seed):
            seeds.append(candidate)
        candidate += 1
    return seeds


def build_test_seeds_from_num_test_samples(
    num_test_samples: int,
    *,
    first_test_seed: int,
    excluded_seeds: list[int],
) -> list[int]:
    count = int(num_test_samples)
    if count < 1:
        raise ValueError("The Monte Carlo test sample count must be at least 1.")
    excluded = {int(seed) for seed in excluded_seeds}
    seeds: list[int] = []
    candidate = int(first_test_seed)
    while len(seeds) < count:
        if candidate not in excluded:
            seeds.append(candidate)
        candidate += 1
    return seeds


def resolve_monte_carlo_train_and_test_seeds(
    *,
    cli_train_seeds: Any = None,
    cli_num_training_samples: int | None = None,
    cli_test_seed: int | None = None,
    config_train_seeds: Any = None,
    config_num_training_samples: int | None = None,
    config_test_seed: int | None = None,
    fallback_train_seeds: Any = "0,1,2",
    fallback_test_seed: int = 3,
) -> tuple[list[int], int]:
    test_seed = int(
        cli_test_seed
        if cli_test_seed is not None
        else config_test_seed
        if config_test_seed is not None
        else fallback_test_seed
    )
    explicit_cli_seeds = parse_optional_seed_values(cli_train_seeds)
    if explicit_cli_seeds:
        return explicit_cli_seeds, test_seed
    if cli_num_training_samples is not None:
        return build_train_seeds_from_num_training_samples(cli_num_training_samples, test_seed), test_seed
    configured_seeds = parse_optional_seed_values(config_train_seeds)
    if configured_seeds:
        return configured_seeds, test_seed
    if config_num_training_samples is not None:
        return build_train_seeds_from_num_training_samples(config_num_training_samples, test_seed), test_seed
    fallback_seeds = parse_optional_seed_values(fallback_train_seeds)
    if fallback_seeds:
        return fallback_seeds, test_seed
    raise ValueError("Could not resolve Monte Carlo training seeds.")
