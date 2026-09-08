"""Deterministic random-precoder baselines shared by every downlink method."""

from __future__ import annotations

from typing import Any

import numpy as np

from latency_optimization.optimization.blocklength_search import build_n_search_config, run_n_frontier_search
from latency_optimization.experiments.scenarios import PAYLOAD_MODE, STREAMING_MODE

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


def _estimate_random_precoder_schedule(
    system: DownlinkSystem,
    simulation: dict[str, Any],
    streaming_targets: np.ndarray | None,
    allow_n_reduction: bool = True,
) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    """Evaluate the deterministic random-beam reference under common scheduling rules.

    What: reproduce the seeded downlink system, sample and jointly power-project one
    random beam per active user/block, then apply the same FBL-rate and blocklength
    search used to account for service. Payload bits carry forward until completion;
    streaming targets expire at the end of each block.

    Why: optimized and closed-form methods need a method-independent initial point.
    Reusing the same seed, rate law, traffic semantics, and latency accounting means
    reported improvement reflects the beam/allocation method rather than a different
    baseline realization.

    Returns: per-user latency, the complete schedule/diagnostic record, and the final
    system metrics generated from that random-precoder plan.
    """
    baseline = DownlinkSystem(system.sc, seed=system.seed, rate_law=system.rate_law)
    streaming = streaming_targets is not None
    remaining = np.asarray(baseline.B, dtype=int).copy()
    precoders = baseline.clone_precoders()
    n_plan = [[] for _ in range(baseline.K)]
    bit_plan = [[] for _ in range(baseline.K)]
    rate_plan = [[] for _ in range(baseline.K)]
    skipped_blocks = [0 for _ in range(baseline.K)]
    block = 0

    while (block < int(streaming_targets.shape[1])) if streaming else bool(np.any(remaining > 0)):
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
    plan: dict[str, Any] = {
        "n_kl": n_plan,
        "B_kl": bit_plan,
        "R_alloc": rate_plan,
        "scenario_mode": STREAMING_MODE if streaming else PAYLOAD_MODE,
    }
    if streaming:
        plan.update(
            {
                "skipped_blocks_per_user": skipped_blocks,
                "scenario_block_targets": streaming_targets.tolist(),
            }
        )
    return baseline.latency.tolist(), plan, collect_interference_diagnostics(baseline)


def estimate_random_precoder_payload_latency(
    system: DownlinkSystem,
    simulation: dict[str, Any],
    allow_n_reduction: bool = True,
) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    """Evaluate the shared random-beam reference for payload completion."""
    return _estimate_random_precoder_schedule(
        system,
        simulation,
        None,
        allow_n_reduction=allow_n_reduction,
    )


def estimate_random_precoder_streaming_latency(
    system: DownlinkSystem,
    simulation: dict[str, Any],
    scenario: dict[str, Any],
    allow_n_reduction: bool = True,
) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    """Evaluate the shared random-beam reference for fixed streaming blocks."""
    if str(scenario["mode"]) != STREAMING_MODE:
        raise ValueError("Streaming baseline requires a streaming scenario.")
    targets = np.asarray(scenario["streaming_bit_targets_by_block"], dtype=int)
    return _estimate_random_precoder_schedule(
        system,
        simulation,
        targets,
        allow_n_reduction=allow_n_reduction,
    )
