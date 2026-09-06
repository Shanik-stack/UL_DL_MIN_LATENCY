"""Deterministic random-precoder baselines shared by every downlink method."""

from __future__ import annotations

from typing import Any

import numpy as np

from latency_optimization.core.blocklength import build_n_search_config, run_n_frontier_search
from latency_optimization.core.scenarios import STREAMING_MODE

from .block_state import (
    collect_interference_diagnostics,
    ensure_precoder_block,
    expand_precoders_for_plan,
    maximum_supported_bits,
    zero_precoder_block,
)
from .system import DownlinkSystem


def _allocate_random_precoder_bits(
    system: DownlinkSystem,
    precoders: list[list[np.ndarray]],
    simulation: dict[str, Any],
    *,
    user: int,
    block: int,
    requested_bits: int,
    allow_n_reduction: bool,
) -> tuple[int, int, float]:
    maximum_blocklength = int(system.T[user])
    full_rate = float(system.compute_block_rate(user, block, maximum_blocklength, F_override=precoders))
    bits_sent = min(max(int(requested_bits), 0), max(maximum_supported_bits(maximum_blocklength, full_rate), 0))
    if bits_sent <= 0 or not allow_n_reduction:
        return bits_sent, maximum_blocklength, full_rate

    search = build_n_search_config(
        n_min=int(simulation["n_kl_min"]),
        n_max=maximum_blocklength,
        fine_step=int(simulation["n_kl_step"]),
        direction=str(simulation["n_search_direction"]),
        strategy=str(simulation["n_search_strategy"]),
        coarse_step=int(simulation["n_search_coarse_step"]),
        exponential_factor=int(simulation["n_search_exponential_factor"]),
    )

    def evaluate(candidate: int, _stage: str) -> dict[str, float | bool]:
        rate = float(system.compute_block_rate(user, block, candidate, F_override=precoders))
        return {"feasible": bits_sent / candidate <= rate, "rate": rate}

    accepted = run_n_frontier_search(search, evaluate)["accepted"]
    if not accepted:
        return bits_sent, maximum_blocklength, full_rate
    best = accepted[-1]
    return bits_sent, int(best["n_kl"]), float(best["result"]["rate"])


def estimate_initial_latency_from_random_precoders_for_scenario(
    system: DownlinkSystem,
    simulation: dict[str, Any],
    scenario: dict[str, Any],
    allow_n_reduction: bool = True,
) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    baseline = DownlinkSystem(system.sc, seed=system.seed, rate_law=system.rate_law)
    streaming = str(scenario["mode"]) == STREAMING_MODE
    streaming_targets = (
        np.asarray(scenario["streaming_bit_targets_by_block"], dtype=int)
        if streaming
        else None
    )
    remaining = np.asarray(baseline.B, dtype=int).copy()
    precoders = baseline.clone_precoders()
    n_plan = [[] for _ in range(baseline.K)]
    bit_plan = [[] for _ in range(baseline.K)]
    rate_plan = [[] for _ in range(baseline.K)]
    skipped_blocks = [0 for _ in range(baseline.K)]
    block = 0

    while (block < int(scenario["number_of_blocks"])) if streaming else bool(np.any(remaining > 0)):
        if block >= int(simulation["max_total_blocks"]):
            raise RuntimeError(
                f"Random-precoder baseline reached max_total_blocks={simulation['max_total_blocks']} "
                f"with remaining bits {remaining.tolist()}."
            )
        requested = streaming_targets[:, block] if streaming else remaining
        active_users = [user for user in range(baseline.K) if int(requested[user]) > 0]

        for user in active_users:
            ensure_precoder_block(baseline, precoders, user, block, use_previous_as_template=False)
            precoders[user][block] = baseline.sample_precoder(user, block)
        baseline.project_block_precoders_to_power(precoders, block, active_users=active_users)

        for user in active_users:
            bits, blocklength, rate = _allocate_random_precoder_bits(
                baseline,
                precoders,
                simulation,
                user=user,
                block=block,
                requested_bits=int(requested[user]),
                allow_n_reduction=allow_n_reduction,
            )
            if bits <= 0:
                zero_precoder_block(baseline, precoders, user, block)
                skipped_blocks[user] += 1
            n_plan[user].append(blocklength)
            bit_plan[user].append(bits)
            rate_plan[user].append(rate)
            if not streaming:
                remaining[user] -= bits
        block += 1

    baseline.apply_solution(expand_precoders_for_plan(baseline, precoders, n_plan), n_plan)
    plan: dict[str, Any] = {"n_kl": n_plan, "B_kl": bit_plan, "R_alloc": rate_plan}
    if streaming:
        plan.update(
            {
                "skipped_blocks_per_user": skipped_blocks,
                "scenario_mode": STREAMING_MODE,
                "scenario_block_targets": streaming_targets.tolist(),
            }
        )
    return baseline.latency.tolist(), plan, collect_interference_diagnostics(baseline)


def estimate_initial_latency_from_random_precoders(
    system: DownlinkSystem,
    simulation: dict[str, Any],
    allow_n_reduction: bool = True,
) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    scenario = {"mode": "payload_completion", "number_of_blocks": 0}
    return estimate_initial_latency_from_random_precoders_for_scenario(
        system,
        simulation,
        scenario,
        allow_n_reduction=allow_n_reduction,
    )
