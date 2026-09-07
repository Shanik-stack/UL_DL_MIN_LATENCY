"""Downlink block state, feasibility, and interference diagnostics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .system import DownlinkSystem


def clone_precoders(precoders: Sequence[Sequence[np.ndarray]]) -> list[list[np.ndarray]]:
    return [[np.array(block, copy=True) for block in user_blocks] for user_blocks in precoders]


def expand_precoders_for_plan(
    system: DownlinkSystem,
    base_precoders: list[list[np.ndarray]],
    blocklength_plan: Sequence[Sequence[int]],
) -> list[list[np.ndarray]]:
    expanded = clone_precoders(base_precoders)
    for user in range(system.K):
        for block in range(len(blocklength_plan[user])):
            if block >= len(expanded[user]):
                system.ensure_block(user, block)
                expanded[user].append(np.array(system.F[user][block], copy=True))
    return expanded


def maximum_supported_bits(blocklength: int, rate: float) -> int:
    return int(np.floor(float(blocklength) * float(rate)))


def resolve_blocklength(
    system: DownlinkSystem,
    user: int,
    blocklengths: dict[int, int] | None,
) -> int:
    user = int(user)
    if blocklengths is None:
        return int(system.T[user])
    return int(blocklengths.get(user, int(system.T[user])))


def make_zero_precoder(system: DownlinkSystem, user: int) -> np.ndarray:
    user = int(user)
    return np.zeros((int(system.Nb[user]), int(system.dk[user])), dtype=np.complex128)


def ensure_precoder_block(
    system: DownlinkSystem,
    precoders: list[list[np.ndarray]],
    user: int,
    block: int,
    *,
    use_previous_as_template: bool = True,
) -> None:
    user = int(user)
    block = int(block)
    if block < len(precoders[user]):
        return
    template = None
    if use_previous_as_template and precoders[user]:
        previous = np.asarray(precoders[user][-1], dtype=np.complex128)
        if float(np.linalg.norm(previous, ord="fro")) > 1e-12:
            template = previous
    system.ensure_block(user, block, template_precoder=template)
    while len(precoders[user]) <= block:
        next_block = len(precoders[user])
        precoders[user].append(np.array(system.F[user][next_block], copy=True))


def zero_precoder_block(
    system: DownlinkSystem,
    precoders: list[list[np.ndarray]],
    user: int,
    block: int,
) -> None:
    user = int(user)
    precoders[user][int(block)] = make_zero_precoder(system, user)


def channels_for_block(system: DownlinkSystem, block: int) -> list[np.ndarray]:
    block = int(block)
    return [
        np.asarray(system.H[user][block], dtype=np.complex64)
        if block < len(system.H[user])
        else np.zeros((int(system.Nr[user]), int(system.Nb[user])), dtype=np.complex64)
        for user in range(system.K)
    ]


def user_link_budget(
    system: DownlinkSystem,
    precoders: list[list[np.ndarray]],
    user: int,
    block: int,
) -> tuple[float, float, float, float]:
    user = int(user)
    block = int(block)
    channel = np.asarray(system.H[user][block], dtype=np.complex128)
    beam = np.asarray(precoders[user][block], dtype=np.complex128)
    signal_power = float(
        np.linalg.norm(channel @ beam, ord="fro") ** 2 / max(1, int(system.Nr[user]))
    )
    interference_power = 0.0
    for interferer in range(system.K):
        if interferer == user or block >= len(precoders[interferer]):
            continue
        interfering_beam = np.asarray(precoders[interferer][block], dtype=np.complex128)
        interference_power += float(
            np.linalg.norm(channel @ interfering_beam, ord="fro") ** 2
            / max(1, int(system.Nr[user]))
        )
    noise_power = float(system.sigma2[user])
    sinr_db = power_to_db(signal_power / max(interference_power + noise_power, 1e-30))
    return signal_power, interference_power, noise_power, sinr_db


def power_to_db(power: float) -> float:
    return float(10.0 * np.log10(max(float(power), 1e-30)))


def collect_interference_diagnostics(system: DownlinkSystem) -> dict[str, Any]:
    user_count = int(system.K)
    max_blocks = max((len(values) for values in system.n_kl), default=0)
    signal = np.full((user_count, max_blocks), np.nan, dtype=float)
    total_interference = np.full((user_count, max_blocks), np.nan, dtype=float)
    noise = np.full((user_count, max_blocks), np.nan, dtype=float)
    sinr_db = np.full((user_count, max_blocks), np.nan, dtype=float)
    pairwise_block = np.full((max_blocks, user_count, user_count), np.nan, dtype=float)
    pairwise_sum = np.zeros((user_count, user_count), dtype=float)
    pairwise_inr_sum = np.zeros((user_count, user_count), dtype=float)
    pairwise_count = np.zeros((user_count, user_count), dtype=float)

    for user in range(user_count):
        for block in range(len(system.n_kl[user])):
            channel = np.asarray(system.H[user][block], dtype=np.complex128)
            beam = np.asarray(system.F[user][block], dtype=np.complex128)
            signal_power = float(
                np.linalg.norm(channel @ beam, ord="fro") ** 2 / max(1, int(system.Nr[user]))
            )
            noise_power = float(system.sigma2[user])
            interference_power = 0.0
            for interferer in range(user_count):
                if interferer == user or block >= len(system.F[interferer]):
                    continue
                interfering_beam = np.asarray(system.F[interferer][block], dtype=np.complex128)
                coupling = float(
                    np.linalg.norm(channel @ interfering_beam, ord="fro") ** 2
                    / max(1, int(system.Nr[user]))
                )
                pairwise_block[block, user, interferer] = coupling
                pairwise_sum[user, interferer] += coupling
                pairwise_inr_sum[user, interferer] += coupling / max(noise_power, 1e-30)
                pairwise_count[user, interferer] += 1.0
                interference_power += coupling
            signal[user, block] = signal_power
            total_interference[user, block] = interference_power
            noise[user, block] = noise_power
            sinr_db[user, block] = power_to_db(
                signal_power / max(interference_power + noise_power, 1e-30)
            )

    average_pairwise_power = np.divide(
        pairwise_sum,
        pairwise_count,
        out=np.full_like(pairwise_sum, np.nan),
        where=pairwise_count > 0,
    )
    average_pairwise_inr_db = 10.0 * np.log10(
        np.maximum(
            np.divide(
                pairwise_inr_sum,
                pairwise_count,
                out=np.full_like(pairwise_inr_sum, np.nan),
                where=pairwise_count > 0,
            ),
            1e-30,
        )
    )
    row_sum = np.nansum(average_pairwise_power, axis=1, keepdims=True)
    average_pairwise_share = np.divide(
        average_pairwise_power,
        row_sum,
        out=np.full_like(average_pairwise_power, np.nan),
        where=row_sum > 0,
    )
    block_totals = (
        np.nansum(pairwise_block, axis=(1, 2))
        if max_blocks > 0
        else np.asarray([], dtype=float)
    )
    worst_block = int(np.nanargmax(block_totals)) if block_totals.size > 0 else -1
    return {
        "blocks_per_user": [len(values) for values in system.n_kl],
        "signal": signal.tolist(),
        "total_interference": total_interference.tolist(),
        "noise": noise.tolist(),
        "sinr_db": sinr_db.tolist(),
        "pairwise_block": pairwise_block.tolist(),
        "avg_pairwise_power": average_pairwise_power.tolist(),
        "avg_pairwise_inr_db": average_pairwise_inr_db.tolist(),
        "avg_pairwise_share": average_pairwise_share.tolist(),
        "worst_block": worst_block,
    }


def evaluate_block_candidate(
    system: DownlinkSystem,
    precoders: list[list[np.ndarray]],
    active_users: Sequence[int],
    block: int,
    blocklengths: dict[int, int] | None = None,
) -> dict[str, Any]:
    rates: list[float] = []
    supported_bits: list[int] = []
    resolved_blocklengths: list[int] = []
    for user in active_users:
        user = int(user)
        blocklength = int(
            system.T[user] if blocklengths is None else blocklengths.get(user, system.T[user])
        )
        rate = float(system.compute_block_rate(user, int(block), blocklength, F_override=precoders))
        bits = max(maximum_supported_bits(blocklength, rate), 0)
        rates.append(rate)
        supported_bits.append(bits)
        resolved_blocklengths.append(blocklength)
    return {
        "user_ids": [int(user) for user in active_users],
        "user_n_kl": resolved_blocklengths,
        "user_rates": rates,
        "user_max_bits": supported_bits,
        "feasible_count": int(sum(bits > 0 for bits in supported_bits)),
        "min_max_bits": int(min(supported_bits)) if supported_bits else 0,
        "min_rate": float(min(rates)) if rates else 0.0,
        "sum_rate": float(sum(rates)),
    }


__all__ = [
    "channels_for_block",
    "clone_precoders",
    "collect_interference_diagnostics",
    "ensure_precoder_block",
    "evaluate_block_candidate",
    "expand_precoders_for_plan",
    "maximum_supported_bits",
    "power_to_db",
    "user_link_budget",
    "zero_precoder_block",
]
