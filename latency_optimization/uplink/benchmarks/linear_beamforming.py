from __future__ import annotations

import math
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np


from latency_optimization.core.blocklength import build_n_search_config, run_n_frontier_search
from latency_optimization.core.scenarios import PAYLOAD_MODE
from latency_optimization.core.validation import require_choice
from latency_optimization.experiments.cost import format_experiment_cost_lines
from latency_optimization.experiments.configuration import load_config_document
from latency_optimization.experiments.determinism import configure_determinism
from latency_optimization.results.console import format_latency_log_line, format_log_line
from latency_optimization.results.naming import make_method_result_tag
from latency_optimization.results.paths import build_uplink_convergence_result_dirs
from latency_optimization.results.persistence import current_local_timestamp, save_json, save_text

from ..config import get_config, load_config
from ..plotting import (
    plot_interference_before_after_heatmaps,
    plot_interference_heatmaps,
    plot_kkt_residual_history,
    plot_latency_and_asynchronality_from_json,
    plot_link_quality_from_json,
    plot_per_user_interference_before_after,
    plot_per_user_interference_profiles,
    plot_per_user_schedule_details,
    plot_user_config,
)
from ..reporting import (
    _build_uplink_final_test_section_lines,
    _build_uplink_per_user_test_lines,
    build_convergence_result,
)
from ..result_writer import save_test_results_to_txt
from ..simulation import (
    apply_training_solution,
    clone_nested_arrays,
    estimate_initial_random_precoder_schedule_for_scenario,
)
from ..system import UplinkSystem
from ..uplink_rate_model import build_uplink_rate_covariance, evaluate_uplink_rate_numpy


ArrayC = np.ndarray


@dataclass(frozen=True)
class BenchmarkMethodSpec:
    method_key: str
    public_name: str
    result_method_name: str
    result_tag_name: str
    precoder_parameterization: str
    builder: Callable[[np.ndarray, int, float, float], tuple[np.ndarray, dict[str, Any]]]


def _pad_columns(matrix: np.ndarray, target_columns: int) -> np.ndarray:
    current_columns = int(matrix.shape[1])
    if current_columns >= int(target_columns):
        return np.asarray(matrix[:, : int(target_columns)], dtype=np.complex128)
    padded = np.zeros((int(matrix.shape[0]), int(target_columns)), dtype=np.complex128)
    if current_columns > 0:
        padded[:, :current_columns] = matrix
    return padded


def _project_user_power(F_raw: np.ndarray, power_budget: float) -> np.ndarray:
    F = np.asarray(F_raw, dtype=np.complex128)
    fro = float(np.linalg.norm(F, ord="fro"))
    if not np.isfinite(fro) or fro <= 0.0:
        return np.zeros_like(F, dtype=np.complex128)
    return F * math.sqrt(max(float(power_budget), 0.0)) / fro


def build_zero_forcing_precoder(
    H_kl: np.ndarray,
    dk: int,
    sigma2: float,
    power_budget: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    H = np.asarray(H_kl, dtype=np.complex128)
    _, singular_values, vh = np.linalg.svd(H, full_matrices=False)
    d_eff = max(0, min(int(dk), int(len(singular_values))))
    if d_eff == 0:
        return np.zeros((H.shape[1], int(dk)), dtype=np.complex128), {
            "effective_rank": 0,
            "min_singular_value": 0.0,
            "max_singular_value": 0.0,
        }
    V = vh.conj().T[:, :d_eff]
    inv_sigma = 1.0 / np.maximum(singular_values[:d_eff], 1e-9)
    F_raw = V @ np.diag(inv_sigma)
    F_raw = _pad_columns(F_raw, int(dk))
    return _project_user_power(F_raw, power_budget), {
        "effective_rank": int(np.linalg.matrix_rank(H)),
        "min_singular_value": float(np.min(singular_values[:d_eff])),
        "max_singular_value": float(np.max(singular_values[:d_eff])),
    }


def build_regularized_zero_forcing_precoder(
    H_kl: np.ndarray,
    dk: int,
    sigma2: float,
    power_budget: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    H = np.asarray(H_kl, dtype=np.complex128)
    _, singular_values, vh = np.linalg.svd(H, full_matrices=False)
    d_eff = max(0, min(int(dk), int(len(singular_values))))
    if d_eff == 0:
        return np.zeros((H.shape[1], int(dk)), dtype=np.complex128), {
            "effective_rank": 0,
            "rzf_alpha": 0.0,
            "min_singular_value": 0.0,
            "max_singular_value": 0.0,
        }
    alpha = float(int(dk) * float(sigma2) / max(float(power_budget), 1e-30))
    V = vh.conj().T[:, :d_eff]
    sigma = singular_values[:d_eff]
    regularized_weights = sigma / (sigma * sigma + float(alpha))
    F_raw = V @ np.diag(regularized_weights)
    F_raw = _pad_columns(F_raw, int(dk))
    return _project_user_power(F_raw, power_budget), {
        "effective_rank": int(np.linalg.matrix_rank(H)),
        "rzf_alpha": float(alpha),
        "min_singular_value": float(np.min(sigma)),
        "max_singular_value": float(np.max(sigma)),
    }


BENCHMARK_METHODS: dict[str, BenchmarkMethodSpec] = {
    "zf": BenchmarkMethodSpec(
        method_key="zf",
        public_name="Zero Forcing",
        result_method_name="Benchmark ZF",
        result_tag_name="bm_zf",
        precoder_parameterization="self_channel_zero_forcing_closed_form",
        builder=build_zero_forcing_precoder,
    ),
    "rzf": BenchmarkMethodSpec(
        method_key="rzf",
        public_name="Regularized Zero Forcing",
        result_method_name="Benchmark RZF",
        result_tag_name="bm_rzf",
        precoder_parameterization="self_channel_regularized_zero_forcing_closed_form",
        builder=build_regularized_zero_forcing_precoder,
    ),
}


def _build_closed_form_experiment_cost(
    *,
    core_wall_time_seconds_total: float,
    beamformer_build_calls: int,
) -> dict[str, Any]:
    return {
        "core_wall_time_seconds_total": float(core_wall_time_seconds_total),
        "core_wall_time_seconds_training": 0.0,
        "core_wall_time_seconds_testing": float(core_wall_time_seconds_total),
        "estimated_nn_training_flops": 0.0,
        "estimated_nn_inference_flops": 0.0,
        "estimated_nn_total_flops": 0.0,
        "training_forward_backward_sample_equivalents": 0,
        "inference_forward_calls": 0,
        "optimizer_steps": 0,
        "actual_optimizer_updates": 0,
        "extra_gradient_evaluations": 0,
        "forward_only_beam_evaluations": int(beamformer_build_calls),
        "workload_counters": {
            "closed_form_beamformer_build_calls": int(beamformer_build_calls),
        },
        "notes": [
            "This benchmark uses a closed-form analytical uplink precoder, so NN FLOP counters are reported as 0.0.",
            "Core wall time is reported under testing because there is no separate training phase.",
        ],
    }


def _build_summary_lines(result: dict[str, Any], spec: BenchmarkMethodSpec) -> list[str]:
    lines = [
        "Uplink optimizer summary",
        "",
        "Setup",
        f"Method: {result.get('method_name', spec.result_method_name)}",
        f"Benchmark beamformer: {spec.public_name}",
        f"Config: {result.get('cfg_path', 'unknown')}",
        f"Config content hash: {result.get('cfg_hash', 'unknown')}",
        f"Seed: {int(result.get('seed', 0))}",
        f"Run started at: {result.get('run_started_at_local', 'unknown')}",
        f"Run completed at: {result.get('run_completed_at_local', 'unknown')}",
        f"Scenario: {result.get('scenario_mode', 'unknown')}",
        f"Uplink rate model: {result.get('uplink_rate_model', 'unknown')}",
        f"Uplink objective mode: {result.get('uplink_objective_mode', 'unknown')}",
        f"Beam reward mode: {result.get('beam_reward_mode', 'unknown')}",
        f"Convergence precoder update mode: {result.get('convergence_precoder_update_mode', 'closed_form_benchmark')}",
        f"Precoder parameterization: {result.get('precoder_parameterization', spec.precoder_parameterization)}",
        f"Initial schedule source: {result.get('initial_schedule_source', 'unknown')}",
    ]
    benchmark_details = result.get("benchmark_method_details", {})
    if isinstance(benchmark_details, dict):
        if benchmark_details.get("rzf_alpha_mean") is not None:
            lines.append(
                f"Average RZF alpha across block builds: {float(benchmark_details['rzf_alpha_mean']):.6e}"
            )
        lines.append(
            f"Closed-form beamformer builds: {int(benchmark_details.get('beamformer_build_calls', 0))}"
        )
    lines.extend([""])
    lines.extend(_build_uplink_final_test_section_lines(result))
    lines.extend([""])
    lines.extend(_build_uplink_per_user_test_lines(result))
    lines.extend(format_experiment_cost_lines(result.get("experiment_cost")))
    return lines


def run_uplink_closed_form_benchmark(
    *,
    method_key: str,
    cfg_name: str,
    seed: int,
    verbose: bool = True,
    system_params_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    method_name = require_choice(method_key, set(BENCHMARK_METHODS), "uplink benchmark method")
    spec = BENCHMARK_METHODS[method_name]
    configure_determinism(int(seed))
    run_started_at_local = current_local_timestamp()
    system_params, sim_cfg, run_meta = load_config(cfg_name)
    if system_params_override is not None:
        # A held-out test episode supplies its own deterministic per-user SNR vector.
        system_params = dict(system_params_override)
    scenario_mode = require_choice(
        sim_cfg.get("experiment_scenario_mode", PAYLOAD_MODE),
        {PAYLOAD_MODE},
        "experiment_scenario_mode",
    )
    if scenario_mode != PAYLOAD_MODE:
        raise ValueError(
            "Uplink benchmark beamformer methods currently support only payload-completion scenarios."
        )

    result_tag = make_method_result_tag(
        spec.result_tag_name,
        run_meta["cfg_stem"],
        seed=int(seed),
        cfg_hash=run_meta.get("cfg_hash"),
    )
    result_dirs = build_uplink_convergence_result_dirs(
        spec.result_method_name,
        result_tag,
        scenario_mode=scenario_mode,
    )
    core_start = perf_counter()

    initial_baseline = estimate_initial_random_precoder_schedule_for_scenario(
        system_params,
        sim_cfg,
        seed=int(seed),
    )
    naive_full_t_baseline = estimate_initial_random_precoder_schedule_for_scenario(
        system_params,
        sim_cfg,
        seed=int(seed),
        allow_n_reduction=False,
    )
    if verbose:
        print(
            format_latency_log_line(
                "[UL Benchmark Initial Baseline]",
                initial_baseline["initial_latency"],
                seed=int(seed),
                scenario="payload",
                method=spec.method_key,
            )
        )

    system = UplinkSystem(system_params, seed=int(seed))
    K = int(system.K)
    remaining_bits = [int(v) for v in system.B]
    max_total_blocks = int(sim_cfg.get("max_total_blocks", 256))
    n_kl_min = int(sim_cfg["n_kl_min"])
    n_kl_step = int(sim_cfg["n_kl_step"])
    working_F = clone_nested_arrays(system.F)

    n_star: list[list[int]] = [[] for _ in range(K)]
    F_star: list[list[np.ndarray]] = [[] for _ in range(K)]
    R_star: list[list[float]] = [[] for _ in range(K)]
    B_used_star: list[list[int]] = [[] for _ in range(K)]
    B_kl_star: list[list[int]] = [[] for _ in range(K)]
    all_user_block_results_train: list[list[list[dict[str, Any]]]] = [[] for _ in range(K)]
    beamformer_metadata: list[dict[str, Any]] = []
    beamformer_build_calls = 0
    block = 0

    while any(bits > 0 for bits in remaining_bits):
        if block >= max_total_blocks:
            raise RuntimeError(
                f"Uplink benchmark {spec.public_name} hit max_total_blocks={max_total_blocks} "
                f"with remaining bits {remaining_bits}."
            )

        for k in range(K):
            while len(system.H[k]) <= int(block):
                system.add_block(k)
            while len(working_F[k]) <= int(block):
                working_F[k].append(np.array(system.F[k][-1], copy=True))

        active_users = [int(k) for k in range(K) if int(remaining_bits[k]) > 0]
        if verbose:
            print(
                format_log_line(
                    "[UL Benchmark Block]",
                    block=int(block),
                    active_users=int(len(active_users)),
                    remaining_bits=int(sum(remaining_bits)),
                    beamformer=spec.method_key,
                )
            )

        for user in active_users:
            F_bm, metadata = spec.builder(
                np.asarray(system.H[user][block], dtype=np.complex128),
                int(system.dk[user]),
                float(system.sigma2[user]),
                float(system.P[user]),
            )
            working_F[user][block] = np.array(F_bm, copy=True)
            beamformer_build_calls += 1
            beamformer_metadata.append(
                {
                    "block": int(block),
                    "user": int(user),
                    **metadata,
                }
            )

        for user in active_users:
            H_kl = np.asarray(system.H[user][block], dtype=np.complex128)
            F_kl = np.asarray(working_F[user][block], dtype=np.complex128)
            T_ref = int(system.T[user])
            sigma2 = float(system.sigma2[user])
            epsilon = float(system.epsilon[user])
            noise_plus_interference_cov = build_uplink_rate_covariance(
                system,
                sim_cfg,
                int(user),
                int(block),
                F_override=working_F,
            )

            R_T = evaluate_uplink_rate_numpy(
                H_kl,
                F_kl,
                sigma2,
                epsilon,
                T_ref,
                noise_plus_interference_cov,
            ).rate
            B_max = max(int(np.floor(float(T_ref) * float(R_T))), 0)
            B_used = int(min(int(remaining_bits[user]), B_max))

            best_n = int(T_ref)
            best_R = float(R_T)
            if B_used > 0:
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
                    lambda candidate_n, _stage: {
                        "feasible": (
                            float(B_used) / float(max(int(candidate_n), 1))
                        ) <= evaluate_uplink_rate_numpy(
                            H_kl,
                            F_kl,
                            sigma2,
                            epsilon,
                            int(candidate_n),
                            noise_plus_interference_cov,
                        ).rate,
                        "R_candidate": evaluate_uplink_rate_numpy(
                            H_kl,
                            F_kl,
                            sigma2,
                            epsilon,
                            int(candidate_n),
                            noise_plus_interference_cov,
                        ).rate,
                    },
                )
                for accepted in search_result["accepted"]:
                    best_n = int(accepted["n_kl"])
                    best_R = float(accepted["result"]["R_candidate"])

            result_row = {
                "n_kl": int(best_n),
                "n": int(best_n),
                "B_l": int(B_used),
                "Bits per sub-block length B/n_kl": float(B_used) / float(max(int(best_n), 1)),
                "required_R_fbl": float(B_used) / float(max(int(best_n), 1)),
                "achieved_R_fbl": float(best_R),
                "F": np.array(F_kl, copy=True),
                "R_fbl": float(best_R),
                "F_power": float(np.linalg.norm(F_kl, ord="fro") ** 2),
                "lambda_rate": 0.0,
                "lambda_power": 0.0,
                "loss_curve": [],
                "kkt_history": [],
                "solve_status": "closed_form_benchmark",
                "final_primal_residual": 0.0,
                "final_complementarity_residual": 0.0,
            }
            all_user_block_results_train[user].append([result_row])
            n_star[user].append(int(best_n))
            F_star[user].append(np.array(F_kl, copy=True))
            R_star[user].append(float(best_R))
            B_used_star[user].append(int(B_used))
            B_kl_star[user].append(int(B_used))
            remaining_bits[user] = max(0, int(remaining_bits[user]) - int(B_used))

            if verbose:
                print(
                    format_log_line(
                        "[UL Benchmark Allocation]",
                        user=int(user),
                        block=int(block),
                        chosen_n_kl=int(best_n),
                        served_bits=int(B_used),
                        remaining_bits=int(remaining_bits[user]),
                        achieved_rate=float(best_R),
                    )
                )

        block += 1

    benchmark_data = {
        "L_out": [len(v) for v in n_star],
        "n_star": n_star,
        "F_star": F_star,
        "R_star": R_star,
        "B_used_star": B_used_star,
        "B_kl_star": B_kl_star,
        "all_user_block_results_train": all_user_block_results_train,
        "scenario_mode": PAYLOAD_MODE,
        "scenario_block_targets": [],
        "convergence_precoder_update_mode": "closed_form_benchmark",
        "precoder_parameterization": spec.precoder_parameterization,
        "benchmark_beamformer": spec.public_name,
        "benchmark_method_details": {
            "beamformer_build_calls": int(beamformer_build_calls),
            "rzf_alpha_mean": (
                float(np.mean([row["rzf_alpha"] for row in beamformer_metadata if "rzf_alpha" in row]))
                if any("rzf_alpha" in row for row in beamformer_metadata)
                else None
            ),
        },
    }

    report_system = UplinkSystem(system_params, seed=int(seed))
    apply_training_solution(report_system, benchmark_data["n_star"], benchmark_data["F_star"])
    result = build_convergence_result(
        report_system,
        benchmark_data,
        method_name=spec.result_method_name,
        cfg_path=str(load_config_document(cfg_name).path),
        seed=int(seed),
        initial_R_fbl=[np.array(v, copy=True) for v in initial_baseline["initial_R_fbl"]],
        initial_n_kl=[list(values) for values in initial_baseline["initial_n_kl"]],
        initial_n=list(initial_baseline["initial_n"]),
        initial_latency=list(initial_baseline["initial_latency"]),
        initial_snr_db=list(initial_baseline["initial_snr_db"]),
        initial_sinr_db=list(initial_baseline["initial_sinr_db"]),
        initial_bits_per_symbol=list(initial_baseline["initial_bits_per_symbol"]),
        initial_B_kl=[list(values) for values in initial_baseline["initial_B_kl"]],
        initial_bits_per_symbol_by_block=[
            list(values) for values in initial_baseline["initial_bits_per_symbol_by_block"]
        ],
        initial_interference_diag=initial_baseline.get("initial_interference_diag"),
        sim_cfg=sim_cfg,
        naive_full_t_baseline=naive_full_t_baseline,
    )
    result["cfg_hash"] = run_meta.get("cfg_hash")
    result["convergence_precoder_update_mode"] = "closed_form_benchmark"
    result["precoder_parameterization"] = spec.precoder_parameterization
    result["benchmark_beamformer"] = spec.public_name
    result["benchmark_method_details"] = benchmark_data["benchmark_method_details"]
    result["scenario_mode"] = PAYLOAD_MODE
    result["experiment_cost"] = _build_closed_form_experiment_cost(
        core_wall_time_seconds_total=perf_counter() - core_start,
        beamformer_build_calls=beamformer_build_calls,
    )
    result["run_started_at_local"] = str(run_started_at_local)
    result["run_completed_at_local"] = current_local_timestamp()

    return {
        "result": result,
        "report_system": report_system,
        "benchmark_data": benchmark_data,
        "initial_baseline": initial_baseline,
        "result_dirs": result_dirs,
        "summary_lines": _build_summary_lines(result, spec),
    }
