from typing import List, Sequence, Tuple

import numpy as np
import torch

from latency_optimization.optimization.blocklength_search import build_n_search_config, run_n_frontier_search
from latency_optimization.experiments.scenarios import PAYLOAD_MODE, STREAMING_MODE, build_experiment_scenario
from latency_optimization.precoders.power import cap_matrix_power_torch
from latency_optimization.runtime import DEVICE
from latency_optimization.physics.rate_law import NORMAL_APPROXIMATION_RATE_LAW, RateLaw

from .system import UplinkSystem
from ..physics.rate import (
    build_uplink_rate_covariance,
    evaluate_uplink_rate,
    evaluate_uplink_rate_tensor,
)

def clone_nested_arrays(nested: Sequence[Sequence[np.ndarray]]) -> List[List[np.ndarray]]:
    return [[np.array(block, copy=True) for block in user_blocks] for user_blocks in nested]


def _to_complex_numpy(F) -> np.ndarray:
    if isinstance(F, np.ndarray):
        return F.astype(np.complex128, copy=False)
    if hasattr(F, "detach"):
        return F.detach().cpu().numpy().astype(np.complex128, copy=False)
    return np.asarray(F, dtype=np.complex128)


def collect_uplink_interference_diagnostics(uplinksystem: UplinkSystem) -> dict:
    """Measure signal, interference, noise, and pairwise coupling over an uplink schedule.

    Reporting uses this record to explain user rate and latency outcomes.
    """
    K = int(uplinksystem.K)
    max_blocks = max((len(v) for v in uplinksystem.n_kl), default=0)
    signal = np.full((K, max_blocks), np.nan, dtype=float)
    total_interference = np.full((K, max_blocks), np.nan, dtype=float)
    noise = np.full((K, max_blocks), np.nan, dtype=float)
    sinr_db = np.full((K, max_blocks), np.nan, dtype=float)
    pairwise_block = np.full((max_blocks, K, K), np.nan, dtype=float)
    pairwise_sum = np.zeros((K, K), dtype=float)
    pairwise_inr_sum = np.zeros((K, K), dtype=float)
    pairwise_count = np.zeros((K, K), dtype=float)

    for k in range(K):
        for l in range(len(uplinksystem.n_kl[k])):
            H_k = np.asarray(uplinksystem.H[k][l], dtype=np.complex128)
            F_k = np.asarray(uplinksystem.F[k][l], dtype=np.complex128)
            X_k = np.asarray(uplinksystem.X[k][l], dtype=np.complex128)
            desired = H_k @ (F_k @ X_k)

            p_sig = float(np.mean(np.abs(desired) ** 2))
            if l < len(uplinksystem.N[k]):
                p_noise = float(np.mean(np.abs(uplinksystem.N[k][l]) ** 2))
            else:
                p_noise = float(uplinksystem.sigma2[k])
            p_interf = 0.0

            for j in range(K):
                if j == k:
                    continue
                if len(uplinksystem.H[j]) == 0 or len(uplinksystem.F[j]) == 0 or len(uplinksystem.X[j]) == 0:
                    continue

                lj = uplinksystem._resolve_metric_block_index(j, l, require_x=True)
                H_j = np.asarray(uplinksystem.H[j][lj], dtype=np.complex128)
                if H_j.shape[0] != H_k.shape[0]:
                    raise ValueError(
                        "Uplink interference diagnostics require a common BS receive dimension NR across users. "
                        f"Victim user {k} block {l} has NR={H_k.shape[0]}, "
                        f"interferer user {j} block {lj} has NR={H_j.shape[0]}."
                    )
                F_j = np.asarray(uplinksystem.F[j][lj], dtype=np.complex128)
                X_j = np.asarray(uplinksystem.X[j][lj], dtype=np.complex128)
                interference = H_j @ (F_j @ X_j)
                coupling = float(np.mean(np.abs(interference) ** 2))
                pairwise_block[l, k, j] = coupling
                pairwise_sum[k, j] += coupling
                pairwise_inr_sum[k, j] += coupling / max(p_noise, 1e-30)
                pairwise_count[k, j] += 1.0
                p_interf += coupling

            signal[k, l] = p_sig
            total_interference[k, l] = p_interf
            noise[k, l] = p_noise
            sinr_db[k, l] = float(
                10.0 * np.log10(max(p_sig / max(p_interf + p_noise, 1e-30), 1e-30))
            )

    avg_pairwise_power = np.divide(
        pairwise_sum,
        pairwise_count,
        out=np.full_like(pairwise_sum, np.nan),
        where=pairwise_count > 0,
    )
    avg_pairwise_inr_lin = np.divide(
        pairwise_inr_sum,
        pairwise_count,
        out=np.full_like(pairwise_inr_sum, np.nan),
        where=pairwise_count > 0,
    )
    avg_pairwise_inr_db = 10.0 * np.log10(np.maximum(avg_pairwise_inr_lin, 1e-30))
    row_sum = np.nansum(avg_pairwise_power, axis=1, keepdims=True)
    avg_pairwise_share = np.divide(
        avg_pairwise_power,
        row_sum,
        out=np.full_like(avg_pairwise_power, np.nan),
        where=row_sum > 0,
    )
    block_totals = np.nansum(pairwise_block, axis=(1, 2)) if max_blocks > 0 else np.asarray([], dtype=float)
    worst_block = int(np.nanargmax(block_totals)) if block_totals.size > 0 else -1

    return {
        "blocks_per_user": [len(v) for v in uplinksystem.n_kl],
        "signal": signal.tolist(),
        "total_interference": total_interference.tolist(),
        "noise": noise.tolist(),
        "sinr_db": sinr_db.tolist(),
        "pairwise_block": pairwise_block.tolist(),
        "avg_pairwise_power": avg_pairwise_power.tolist(),
        "avg_pairwise_inr_db": avg_pairwise_inr_db.tolist(),
        "avg_pairwise_share": avg_pairwise_share.tolist(),
        "worst_block": int(worst_block),
    }


def apply_training_solution(
    uplink_system: UplinkSystem,
    n_star: Sequence[Sequence[int]],
    F_star: Sequence[Sequence],
) -> None:
    """Commit optimized uplink beams and blocklengths to simulator state."""
    K = int(uplink_system.K)
    n_kl_new: List[List[int]] = []
    F_new: List[List[np.ndarray]] = []

    for k in range(K):
        nk = list(map(int, n_star[k]))
        if any(int(n_kl) <= 0 for n_kl in nk):
            raise ValueError(f"Uplink blocklengths must be strictly positive for user {k}, got {nk}.")
        n_kl_new.append(nk)

        Lk = len(nk)
        if len(F_star[k]) == 0:
            Fk = list(uplink_system.F[k])[:Lk]
        else:
            Fk = [_to_complex_numpy(F) for F in F_star[k]][:Lk]
        if len(Fk) < Lk and Fk:
            Fk.extend(np.array(Fk[-1], copy=True) for _ in range(Lk - len(Fk)))
        F_new.append(Fk)

    uplink_system.update_system(F=F_new, n_kl=n_kl_new, regenerate_noise_on_nl_change=True)


def ensure_blocks_up_to(uplinksystem: UplinkSystem, block_idx: int) -> None:
    for k in range(int(uplinksystem.K)):
        while len(uplinksystem.H[k]) <= int(block_idx):
            uplinksystem.add_block(k)


def _evaluate_fixed_precoder_blocklength(
    channel: np.ndarray,
    precoder: np.ndarray,
    sigma2: float,
    epsilon: float,
    noise_plus_interference_covariance: np.ndarray | None,
    bits: int,
    candidate_n: int,
    rate_law: RateLaw = NORMAL_APPROXIMATION_RATE_LAW,
) -> dict[str, float | bool]:
    """Evaluate supported bits and FBL feasibility for one fixed beam and candidate n_kl."""
    rate = evaluate_uplink_rate(
        channel,
        precoder,
        sigma2,
        epsilon,
        candidate_n,
        noise_plus_interference_covariance,
        rate_law=rate_law,
    ).rate
    return {
        "feasible": float(bits) / float(candidate_n) <= rate,
        "R_candidate": rate,
    }


def estimate_initial_random_precoder_payload_schedule(
    system_params: dict,
    sim_cfg: dict,
    *,
    seed: int,
    allow_n_reduction: bool = True,
) -> dict:
    """Build the deterministic random-beam payload reference schedule.

    Proposed and benchmark methods compare against this channel-seeded baseline.
    """
    baseline_system = UplinkSystem(system_params, seed=int(seed))
    K = int(baseline_system.K)
    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])
    max_total_blocks = int(sim_cfg.get("max_total_blocks", 256))

    remaining_bits = [int(v) for v in baseline_system.B]
    initial_n_kl: List[List[int]] = [[] for _ in range(K)]
    initial_B_kl: List[List[int]] = [[] for _ in range(K)]
    initial_R_fbl: List[List[float]] = [[] for _ in range(K)]
    initial_F: List[List[np.ndarray]] = [[] for _ in range(K)]

    block = 0
    while any(bits > 0 for bits in remaining_bits):
        if block >= max_total_blocks:
            raise RuntimeError(
                f"Initial random-precoder uplink schedule hit max_total_blocks={max_total_blocks} "
                f"with remaining bits {remaining_bits}."
            )

        ensure_blocks_up_to(baseline_system, block)
        random_snapshot = clone_nested_arrays(baseline_system.F)

        for k in range(K):
            if remaining_bits[k] <= 0:
                continue

            H_kl = np.asarray(baseline_system.H[k][block], dtype=np.complex64)
            F_kl = np.asarray(random_snapshot[k][block], dtype=np.complex64)
            T_ref = int(baseline_system.T[k])
            sigma2 = float(baseline_system.sigma2[k])
            epsilon = float(baseline_system.epsilon[k])
            noise_plus_interference_cov = build_uplink_rate_covariance(
                baseline_system,
                sim_cfg,
                k,
                block,
                F_override=random_snapshot,
            )

            R_T = evaluate_uplink_rate(
                H_kl,
                F_kl,
                sigma2,
                epsilon,
                T_ref,
                noise_plus_interference_cov,
                rate_law=baseline_system.rate_law,
            ).rate
            B_max = max(int(np.floor(float(T_ref) * float(R_T))), 0)
            B_used = int(min(int(remaining_bits[k]), B_max))

            best_n = int(T_ref)
            best_R = float(R_T)
            if allow_n_reduction and B_used > 0:
                search_cfg = build_n_search_config(
                    n_min=int(n_kl_min),
                    n_max=int(T_ref),
                    fine_step=int(n_kl_step),
                    direction=sim_cfg.get("n_search_direction", "descending"),
                    strategy=sim_cfg.get("n_search_strategy", "fixed_step"),
                    coarse_step=sim_cfg.get("n_search_coarse_step", int(n_kl_step)),
                    exponential_factor=sim_cfg.get("n_search_exponential_factor", 2),
                )
                search_result = run_n_frontier_search(
                    search_cfg,
                    lambda candidate_n, _stage: _evaluate_fixed_precoder_blocklength(
                        H_kl,
                        F_kl,
                        sigma2,
                        epsilon,
                        noise_plus_interference_cov,
                        B_used,
                        int(candidate_n),
                        rate_law=baseline_system.rate_law,
                    ),
                )
                for accepted in search_result["accepted"]:
                    best_n = int(accepted["n_kl"])
                    best_R = float(accepted["result"]["R_candidate"])

            initial_n_kl[k].append(int(best_n))
            initial_B_kl[k].append(int(B_used))
            initial_R_fbl[k].append(float(best_R))
            initial_F[k].append(np.array(F_kl, copy=True))
            remaining_bits[k] = max(0, int(remaining_bits[k]) - int(B_used))

        block += 1

    apply_training_solution(baseline_system, initial_n_kl, initial_F)
    _, initial_snr_db = baseline_system.get_SNR()
    _, initial_sinr_db = baseline_system.get_SINR()
    initial_interference_diag = collect_uplink_interference_diagnostics(baseline_system)

    initial_n = [int(sum(user_n)) for user_n in initial_n_kl]
    initial_latency = [float(v) for v in baseline_system.latency]
    initial_bits_per_symbol_by_block = []
    initial_bits_per_symbol = []
    for k in range(K):
        user_bps = [
            float(bits) / float(max(n_kl, 1))
            for bits, n_kl in zip(initial_B_kl[k], initial_n_kl[k])
        ]
        total_n = float(max(initial_n[k], 1))
        initial_bits_per_symbol_by_block.append(user_bps)
        initial_bits_per_symbol.append(float(sum(initial_B_kl[k])) / total_n)

    return {
        "initial_n_kl": initial_n_kl,
        "initial_B_kl": initial_B_kl,
        "initial_R_fbl": initial_R_fbl,
        "initial_n": initial_n,
        "initial_latency": initial_latency,
        "initial_snr_db": list(map(float, initial_snr_db)),
        "initial_sinr_db": list(map(float, initial_sinr_db)),
        "initial_bits_per_symbol": initial_bits_per_symbol,
        "initial_bits_per_symbol_by_block": initial_bits_per_symbol_by_block,
        "initial_interference_diag": initial_interference_diag,
        "initial_schedule_source": (
            "random_precoder_baseline"
            if allow_n_reduction
            else "naive_full_T_baseline"
        ),
        "skipped_blocks_per_user": [0 for _ in range(K)],
        "scenario_mode": PAYLOAD_MODE,
    }


def estimate_initial_random_precoder_streaming_schedule(
    system_params: dict,
    sim_cfg: dict,
    *,
    seed: int,
    allow_n_reduction: bool = True,
) -> dict:
    """Build the deterministic random-beam reference over fixed streaming blocks."""
    scenario = build_experiment_scenario(system_params, sim_cfg, seed=int(seed))
    if str(scenario["mode"]) != STREAMING_MODE:
        raise ValueError("Streaming baseline requires experiment_scenario.mode='streaming'.")
    baseline_system = UplinkSystem(system_params, seed=int(seed))
    K = int(baseline_system.K)
    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])
    block_targets = np.asarray(scenario["streaming_bit_targets_by_block"], dtype=int)
    num_blocks = int(scenario["number_of_blocks"])

    initial_n_kl: List[List[int]] = [[] for _ in range(K)]
    initial_B_kl: List[List[int]] = [[] for _ in range(K)]
    initial_R_fbl: List[List[float]] = [[] for _ in range(K)]
    initial_F: List[List[np.ndarray]] = [[] for _ in range(K)]
    skipped_blocks_per_user = [0 for _ in range(K)]

    for block in range(num_blocks):
        ensure_blocks_up_to(baseline_system, block)
        random_snapshot = clone_nested_arrays(baseline_system.F)

        for k in range(K):
            target_bits = int(block_targets[k, block])
            H_kl = np.asarray(baseline_system.H[k][block], dtype=np.complex64)
            F_kl = np.asarray(random_snapshot[k][block], dtype=np.complex64)
            T_ref = int(baseline_system.T[k])
            sigma2 = float(baseline_system.sigma2[k])
            epsilon = float(baseline_system.epsilon[k])
            noise_plus_interference_cov = build_uplink_rate_covariance(
                baseline_system,
                sim_cfg,
                k,
                block,
                F_override=random_snapshot,
            )
            R_T = evaluate_uplink_rate(
                H_kl,
                F_kl,
                sigma2,
                epsilon,
                T_ref,
                noise_plus_interference_cov,
                rate_law=baseline_system.rate_law,
            ).rate
            B_max = max(int(np.floor(float(T_ref) * float(R_T))), 0)
            B_used = int(min(target_bits, B_max))
            best_n = int(T_ref)
            best_R = float(R_T)

            if allow_n_reduction and int(B_used) >= int(target_bits) and int(target_bits) > 0:
                search_cfg = build_n_search_config(
                    n_min=int(n_kl_min),
                    n_max=int(T_ref),
                    fine_step=int(n_kl_step),
                    direction=sim_cfg.get("n_search_direction", "descending"),
                    strategy=sim_cfg.get("n_search_strategy", "fixed_step"),
                    coarse_step=sim_cfg.get("n_search_coarse_step", int(n_kl_step)),
                    exponential_factor=sim_cfg.get("n_search_exponential_factor", 2),
                )
                search_result = run_n_frontier_search(
                    search_cfg,
                    lambda candidate_n, _stage: _evaluate_fixed_precoder_blocklength(
                        H_kl,
                        F_kl,
                        sigma2,
                        epsilon,
                        noise_plus_interference_cov,
                        target_bits,
                        int(candidate_n),
                        rate_law=baseline_system.rate_law,
                    ),
                )
                for accepted in search_result["accepted"]:
                    best_n = int(accepted["n_kl"])
                    best_R = float(accepted["result"]["R_candidate"])

            initial_n_kl[k].append(int(best_n))
            initial_B_kl[k].append(int(B_used))
            initial_R_fbl[k].append(float(best_R))
            initial_F[k].append(
                np.array(F_kl, copy=True) if int(B_used) > 0 else np.zeros_like(F_kl, dtype=np.complex64)
            )
            if int(B_used) <= 0:
                skipped_blocks_per_user[k] += 1

    apply_training_solution(baseline_system, initial_n_kl, initial_F)
    _, initial_snr_db = baseline_system.get_SNR()
    _, initial_sinr_db = baseline_system.get_SINR()
    initial_interference_diag = collect_uplink_interference_diagnostics(baseline_system)

    initial_n = [int(sum(int(max(v, 0)) for v in user_n)) for user_n in initial_n_kl]
    initial_latency = [float(v) for v in baseline_system.latency]
    initial_bits_per_symbol_by_block = []
    initial_bits_per_symbol = []
    for k in range(K):
        user_bps = [
            float(bits) / float(max(int(n_kl), 1))
            if int(n_kl) > 0 and int(bits) > 0
            else 0.0
            for bits, n_kl in zip(initial_B_kl[k], initial_n_kl[k])
        ]
        total_n = float(max(initial_n[k], 1))
        initial_bits_per_symbol_by_block.append(user_bps)
        initial_bits_per_symbol.append(float(sum(initial_B_kl[k])) / total_n if initial_n[k] > 0 else 0.0)

    return {
        "initial_n_kl": initial_n_kl,
        "initial_B_kl": initial_B_kl,
        "initial_R_fbl": initial_R_fbl,
        "initial_n": initial_n,
        "initial_latency": initial_latency,
        "initial_snr_db": list(map(float, initial_snr_db)),
        "initial_sinr_db": list(map(float, initial_sinr_db)),
        "initial_bits_per_symbol": initial_bits_per_symbol,
        "initial_bits_per_symbol_by_block": initial_bits_per_symbol_by_block,
        "initial_interference_diag": initial_interference_diag,
        "skipped_blocks_per_user": [int(v) for v in skipped_blocks_per_user],
        "scenario_mode": STREAMING_MODE,
        "scenario_block_targets": block_targets.tolist(),
        "initial_schedule_source": (
            "random_precoder_baseline"
            if allow_n_reduction
            else "naive_full_T_baseline"
        ),
    }


def max_precoder_delta(F_old: Sequence[Sequence], F_new: Sequence[Sequence]) -> float:
    """Return the largest relative beam change across a nested uplink schedule."""
    max_delta = 0.0
    K = max(len(F_old), len(F_new))
    for k in range(K):
        old_blocks = F_old[k] if k < len(F_old) else []
        new_blocks = F_new[k] if k < len(F_new) else []
        Lk = max(len(old_blocks), len(new_blocks))
        for l in range(Lk):
            if l >= len(old_blocks) or l >= len(new_blocks):
                max_delta = max(max_delta, 1.0)
                continue
            A = _to_complex_numpy(old_blocks[l])
            B = _to_complex_numpy(new_blocks[l])
            denom = max(float(np.linalg.norm(A, ord="fro")), 1e-12)
            delta = float(np.linalg.norm(A - B, ord="fro") / denom)
            max_delta = max(max_delta, delta)
    return max_delta


def complex_to_ri_parameter(F: np.ndarray, device=DEVICE) -> torch.nn.Parameter:
    arr = np.asarray(F, dtype=np.complex64)
    stacked = np.stack([arr.real, arr.imag], axis=0)
    return torch.nn.Parameter(torch.tensor(stacked, dtype=torch.float32, device=device))


def ri_to_complex_tensor(param: torch.Tensor) -> torch.Tensor:
    return (param[0] + 1j * param[1]).to(torch.complex64)


def project_power_complex_torch(F: torch.Tensor, P: float, eps: float = 1e-12) -> torch.Tensor:
    return cap_matrix_power_torch(F, P, eps=eps)


def compute_joint_rates_torch(
    H_list: Sequence[torch.Tensor],
    F_list: Sequence[torch.Tensor],
    sigma2_list: Sequence[float],
    epsilon_list: Sequence[float],
    n_values: Sequence[int],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Evaluate every user's differentiable rate under one joint uplink beam snapshot.

    Monte Carlo training uses this when interference couples otherwise independent user nets.
    """
    rates: List[torch.Tensor] = []
    capacities: List[torch.Tensor] = []
    dispersions: List[torch.Tensor] = []

    K = len(H_list)
    for k in range(K):
        Hk = H_list[k]
        Fk = F_list[k]
        Nr = Hk.shape[0]
        I = torch.eye(Nr, dtype=torch.complex64, device=Hk.device)
        noise_cov = float(sigma2_list[k]) * I

        for j in range(K):
            if j == k:
                continue
            Hj = H_list[j]
            if Hj.shape[0] != Nr:
                raise ValueError(
                    "Joint SINR optimization assumes a common BS receive dimension NR across users. "
                    f"User {k} has NR={Nr}, user {j} has NR={Hj.shape[0]}."
                )
            HFj = Hj @ F_list[j]
            noise_cov = noise_cov + HFj @ HFj.conj().transpose(1, 0)

        rate_result = evaluate_uplink_rate_tensor(
            Hk,
            Fk,
            float(sigma2_list[k]),
            float(epsilon_list[k]),
            int(n_values[k]),
            noise_cov,
            covariance_jitter=1.0e-6,
        )
        capacities.append(rate_result.capacity)
        dispersions.append(rate_result.dispersion)
        rates.append(rate_result.rate)

    return rates, capacities, dispersions


def build_single_block_post_training_dict(
    uplinksystem: UplinkSystem,
    norm_stats,
    n_values: Sequence[int],
    F_tensors: Sequence[torch.Tensor],
    rates: Sequence[torch.Tensor | float],
    *,
    loss_history: Sequence[float],
    method_name: str,
    metadata: dict | None = None,
):
    """Convert one trained block into the common result schema used by writers and plots."""
    K = int(uplinksystem.K)
    n_star = [[int(n_values[k])] for k in range(K)]
    F_star = [[F_tensors[k].detach().cpu()] for k in range(K)]
    R_star = [[float(rates[k].detach().cpu() if torch.is_tensor(rates[k]) else rates[k])] for k in range(K)]
    L_out = [1] * K

    all_user_block_results = []
    B_used_star = []
    B_kl_star = []

    for k in range(K):
        B_used = max(0, min(int(uplinksystem.B[k]), int(np.floor(int(n_values[k]) * R_star[k][0]))))
        B_used_star.append([B_used])
        B_kl_star.append([B_used])
        Fk = F_tensors[k].detach().cpu()
        all_user_block_results.append([[
            {
                "n_kl": int(n_values[k]),
                "n": int(n_values[k]),
                "B_l": int(B_used),
                "Bits per sub-block length B/n_kl": float(B_used) / float(max(int(n_values[k]), 1)),
                "F": Fk,
                "R_fbl": float(R_star[k][0]),
                "F_power": float((torch.linalg.norm(Fk, ord="fro") ** 2).real.cpu()),
                "lambda_rate": 0.0,
                "lambda_power": 0.0,
                "loss_curve": list(loss_history),
                "method": method_name,
            }
        ]])

    post = {
        "L_out": L_out,
        "n_star": n_star,
        "F_star": F_star,
        "R_star": R_star,
        "norm_stats": norm_stats,
        "all_user_block_results_train": all_user_block_results,
        "B_used_star": B_used_star,
        "B_kl_star": B_kl_star,
        "method_name": method_name,
    }
    if metadata:
        post.update(metadata)
    return post


def nested_int_lists_equal(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> bool:
    return list(map(list, a)) == list(map(list, b))
