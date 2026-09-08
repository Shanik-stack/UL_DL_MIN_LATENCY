from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from latency_optimization.core.scenarios import STREAMING_MODE
from latency_optimization.experiments.cost import format_experiment_cost_lines
from latency_optimization.results.metrics import (
    format_optional_db as _format_optional_db,
    mean_for_active_users as _mean_for_served_users,
    pairwise_latency_differences as _pairwise_latency_diffs,
    reference_latency_metrics as _compute_reference_latency_metrics,
)

from .simulation import collect_uplink_interference_diagnostics


def _compute_dispersion_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    c_values = result.get("C", [])
    v_values = result.get("V", [])
    r_values = result.get("R_fbl", [])
    n_values = result.get("n_kl", [])
    b_values = result.get("B_kl", [])

    num_users = max(
        len(c_values) if isinstance(c_values, list) else 0,
        len(v_values) if isinstance(v_values, list) else 0,
        len(r_values) if isinstance(r_values, list) else 0,
        len(n_values) if isinstance(n_values, list) else 0,
        len(b_values) if isinstance(b_values, list) else 0,
    )

    accepted_records: list[dict[str, Any]] = []
    capacity_values: list[float] = []
    dispersion_values: list[float] = []
    penalty_values: list[float] = []
    ratio_values: list[float] = []

    def _value_at(source: Any, user: int, block: int, default: float = np.nan) -> float:
        if not isinstance(source, list) or user >= len(source):
            return float(default)
        row = source[user]
        if not isinstance(row, list) or block >= len(row):
            return float(default)
        try:
            return float(row[block])
        except (TypeError, ValueError):
            return float(default)

    for user in range(num_users):
        max_blocks = max(
            len(c_values[user]) if isinstance(c_values, list) and user < len(c_values) and isinstance(c_values[user], list) else 0,
            len(v_values[user]) if isinstance(v_values, list) and user < len(v_values) and isinstance(v_values[user], list) else 0,
            len(r_values[user]) if isinstance(r_values, list) and user < len(r_values) and isinstance(r_values[user], list) else 0,
            len(n_values[user]) if isinstance(n_values, list) and user < len(n_values) and isinstance(n_values[user], list) else 0,
            len(b_values[user]) if isinstance(b_values, list) and user < len(b_values) and isinstance(b_values[user], list) else 0,
        )
        for block in range(max_blocks):
            served_bits = int(round(_value_at(b_values, user, block, default=0.0)))
            if served_bits <= 0:
                continue
            n_kl = int(round(_value_at(n_values, user, block, default=0.0)))
            capacity = _value_at(c_values, user, block)
            dispersion = _value_at(v_values, user, block)
            achieved_rate = _value_at(r_values, user, block)
            penalty = max(capacity - achieved_rate, 0.0) if np.isfinite(capacity) and np.isfinite(achieved_rate) else np.nan
            ratio = (
                float(penalty / capacity)
                if np.isfinite(penalty) and np.isfinite(capacity) and abs(capacity) > 1e-12
                else np.nan
            )
            required_rate = float(served_bits / max(n_kl, 1))
            rate_margin = float(achieved_rate - required_rate) if np.isfinite(achieved_rate) else np.nan
            accepted_records.append(
                {
                    "user": int(user),
                    "block": int(block),
                    "served_bits": int(served_bits),
                    "n_kl": int(n_kl),
                    "capacity": float(capacity),
                    "dispersion": float(dispersion),
                    "fbl_penalty": float(penalty) if np.isfinite(penalty) else np.nan,
                    "penalty_over_capacity": float(ratio) if np.isfinite(ratio) else np.nan,
                    "achieved_rate": float(achieved_rate),
                    "required_rate": float(required_rate),
                    "rate_margin": float(rate_margin) if np.isfinite(rate_margin) else np.nan,
                }
            )
            if np.isfinite(capacity):
                capacity_values.append(float(capacity))
            if np.isfinite(dispersion):
                dispersion_values.append(float(dispersion))
            if np.isfinite(penalty):
                penalty_values.append(float(penalty))
            if np.isfinite(ratio):
                ratio_values.append(float(ratio))

    return {
        "accepted_block_count": int(len(accepted_records)),
        "accepted_block_records": accepted_records,
        "avg_capacity": float(np.mean(capacity_values)) if capacity_values else 0.0,
        "avg_dispersion": float(np.mean(dispersion_values)) if dispersion_values else 0.0,
        "avg_fbl_penalty": float(np.mean(penalty_values)) if penalty_values else 0.0,
        "max_fbl_penalty": float(np.max(penalty_values)) if penalty_values else 0.0,
        "avg_penalty_over_capacity": float(np.mean(ratio_values)) if ratio_values else 0.0,
        "max_penalty_over_capacity": float(np.max(ratio_values)) if ratio_values else 0.0,
    }


def _flatten_uplink_epoch_history(all_user_block_results: Sequence[Sequence[Sequence[dict[str, Any]]]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    segment_id = 0

    for user_idx, user_blocks in enumerate(all_user_block_results or []):
        for block_idx, block_results in enumerate(user_blocks or []):
            for solve_idx, solve_result in enumerate(block_results or []):
                kkt_history = list(solve_result.get("kkt_history", []))
                if len(kkt_history) == 0:
                    continue
                segment_id += 1
                solve_status = str(solve_result.get("solve_status", "unknown"))
                status_counts[solve_status] = int(status_counts.get(solve_status, 0)) + 1
                n_kl = int(solve_result.get("n_kl", 0))
                transmitted_bits = int(solve_result.get("B_l", 0))

                for epoch_row_idx, epoch_row in enumerate(kkt_history):
                    is_segment_end = bool(epoch_row_idx == (len(kkt_history) - 1))
                    rows.append(
                        {
                            "user": int(user_idx),
                            "block": int(block_idx),
                            "solve_index": int(solve_idx),
                            "solve_segment_id": int(segment_id),
                            "epoch": int(epoch_row.get("epoch", epoch_row_idx + 1)),
                            "n_kl": int(n_kl),
                            "transmitted_bits": int(transmitted_bits),
                            "kkt_primal_residual": float(epoch_row.get("primal_residual", np.nan)),
                            "kkt_complementarity_residual": float(epoch_row.get("complementarity_residual", np.nan)),
                            "kkt_stationarity_residual": float(epoch_row.get("stationarity_residual", np.nan)),
                            "rate_gap": float(epoch_row.get("rate_gap", np.nan)),
                            "power_gap": float(epoch_row.get("power_gap", np.nan)),
                            "rate_violation": float(epoch_row.get("rate_violation", np.nan)),
                            "power_violation": float(epoch_row.get("power_violation", np.nan)),
                            "rate": float(epoch_row.get("rate", np.nan)),
                            "power": float(epoch_row.get("power", np.nan)),
                            "lambda_rate": float(epoch_row.get("lambda_rate", np.nan)),
                            "lambda_power": float(epoch_row.get("lambda_power", np.nan)),
                            "solve_status": solve_status if is_segment_end else "",
                            "solve_segment_end": bool(is_segment_end),
                        }
                    )

    return rows, status_counts


def compute_summary_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Derive latency, service, asynchronality, link-quality, and convergence summaries."""
    initial_latency = [float(x) for x in result["initial_latency"]]
    final_latency = [float(x) for x in result["final_latency"]]
    K = len(final_latency)

    latency_reduction_per_user_percent: list[float] = []
    for init_val, final_val in zip(initial_latency, final_latency):
        if init_val > 0:
            reduction = ((init_val - final_val) / init_val) * 100.0
        else:
            reduction = 0.0
        latency_reduction_per_user_percent.append(float(reduction))

    initial_total_latency = float(sum(initial_latency))
    final_total_latency = float(sum(final_latency))
    if initial_total_latency > 0:
        total_latency_reduction_percent = ((initial_total_latency - final_total_latency) / initial_total_latency) * 100.0
    else:
        total_latency_reduction_percent = 0.0

    initial_async_matrix, initial_async_pairs, initial_async_sum = _pairwise_latency_diffs(initial_latency)
    final_async_matrix, final_async_pairs, final_async_sum = _pairwise_latency_diffs(final_latency)
    if initial_async_sum > 0:
        async_reduction_percent = ((initial_async_sum - final_async_sum) / initial_async_sum) * 100.0
    else:
        async_reduction_percent = 0.0

    initial_snr_db = [float(x) for x in result.get("initial_snr_db", [])]
    final_snr_db = [float(x) for x in result.get("final_snr_db", [])]
    initial_sinr_db = [float(x) for x in result.get("initial_sinr_db", [])]
    final_sinr_db = [float(x) for x in result.get("final_sinr_db", [])]
    blocks_per_user = [len(v) for v in result.get("final_n_kl", [[] for _ in range(K)])]
    total_n = [int(x) for x in result.get("final_n", [0 for _ in range(K)])]
    served_bits = [int(sum(v)) for v in result.get("B_kl", [[] for _ in range(K)])]
    initial_served_bits = [int(sum(v)) for v in result.get("initial_B_kl", [[] for _ in range(K)])]
    skipped_blocks_per_user = [
        int(v)
        for v in result.get(
            "skipped_blocks_per_user",
            [0 for _ in range(K)],
        )
    ]
    scenario_mode = str(
        result.get(
            "scenario_mode",
            result.get("experiment_scenario_mode", ""),
        )
    )
    scenario_block_targets = np.asarray(result.get("scenario_block_targets", []), dtype=int)
    target_bits_per_user = [0 for _ in range(K)]
    initial_unserved_bits = [0 for _ in range(K)]
    final_unserved_bits = [0 for _ in range(K)]
    partially_served_blocks_per_user = [0 for _ in range(K)]
    zero_service_blocks_per_user = [0 for _ in range(K)]

    if scenario_mode == STREAMING_MODE and scenario_block_targets.ndim == 2 and scenario_block_targets.shape[0] == K:
        target_bits_per_user = list(map(int, scenario_block_targets.sum(axis=1, dtype=int)))
        initial_b_kl = [list(map(int, values)) for values in result.get("initial_B_kl", [[] for _ in range(K)])]
        final_b_kl = [list(map(int, values)) for values in result.get("B_kl", [[] for _ in range(K)])]
        for k in range(K):
            targets = scenario_block_targets[int(k)].tolist()
            initial_served = initial_b_kl[int(k)] if int(k) < len(initial_b_kl) else []
            final_served = final_b_kl[int(k)] if int(k) < len(final_b_kl) else []
            initial_unserved_bits[int(k)] = int(
                sum(max(int(t) - int(s), 0) for t, s in zip(targets, initial_served))
            )
            final_unserved_bits[int(k)] = int(
                sum(max(int(t) - int(s), 0) for t, s in zip(targets, final_served))
            )
            partially_served_blocks_per_user[int(k)] = int(
                sum(1 for t, s in zip(targets, final_served) if int(t) > 0 and 0 < int(s) < int(t))
            )
            zero_service_blocks_per_user[int(k)] = int(
                sum(1 for t, s in zip(targets, final_served) if int(t) > 0 and int(s) <= 0)
            )

    per_user_summary = []
    for k in range(K):
        target_bits = int(target_bits_per_user[k]) if k < len(target_bits_per_user) else 0
        user_served_bits = int(served_bits[k])
        missed_bits = int(final_unserved_bits[k])
        successful_blocks = max(
            int(blocks_per_user[k])
            - int(partially_served_blocks_per_user[k])
            - int(zero_service_blocks_per_user[k]),
            0,
        )
        per_user_summary.append(
            {
                "user": int(k),
                "initial_latency": initial_latency[k],
                "final_latency": final_latency[k],
                "latency_reduction_percent": latency_reduction_per_user_percent[k],
                "initial_snr_db": initial_snr_db[k] if k < len(initial_snr_db) else 0.0,
                "final_snr_db": final_snr_db[k] if k < len(final_snr_db) else 0.0,
                "initial_sinr_db": initial_sinr_db[k] if k < len(initial_sinr_db) else 0.0,
                "final_sinr_db": final_sinr_db[k] if k < len(final_sinr_db) else 0.0,
                "blocks": int(blocks_per_user[k]),
                "total_n": int(total_n[k]),
                "target_bits": target_bits,
                "initial_served_bits": int(initial_served_bits[k]),
                "served_bits": user_served_bits,
                "initial_unserved_bits": int(initial_unserved_bits[k]),
                "unserved_bits": int(final_unserved_bits[k]),
                "missed_bits": missed_bits,
                "delivery_ratio": (
                    float(user_served_bits) / float(target_bits) if target_bits > 0 else 1.0
                ),
                "successful_blocks": successful_blocks,
                "deadline_miss_fraction": (
                    float(int(partially_served_blocks_per_user[k]) + int(zero_service_blocks_per_user[k]))
                    / float(blocks_per_user[k])
                    if int(blocks_per_user[k]) > 0
                    else 0.0
                ),
                "average_selected_blocklength": (
                    float(total_n[k]) / float(blocks_per_user[k])
                    if int(blocks_per_user[k]) > 0
                    else 0.0
                ),
                "partially_served_blocks": int(partially_served_blocks_per_user[k]),
                "zero_service_blocks": int(zero_service_blocks_per_user[k]),
                "skipped_blocks": int(skipped_blocks_per_user[k]) if k < len(skipped_blocks_per_user) else 0,
            }
        )

    baseline_comparison_metrics: dict[str, dict[str, Any]] = {}
    for baseline_name, baseline_payload in result.get("baseline_references", {}).items():
        if not isinstance(baseline_payload, dict):
            continue
        ref_latency = baseline_payload.get("latency", [])
        if not isinstance(ref_latency, Sequence):
            continue
        baseline_comparison_metrics[str(baseline_name)] = _compute_reference_latency_metrics(
            ref_latency,
            final_latency,
        )
    dispersion_diagnostics = _compute_dispersion_diagnostics(result)

    return {
        "initial_total_latency": initial_total_latency,
        "final_total_latency": final_total_latency,
        "initial_avg_latency": float(initial_total_latency / max(K, 1)),
        "final_avg_latency": float(final_total_latency / max(K, 1)),
        "initial_max_latency": float(max(initial_latency) if initial_latency else 0.0),
        "final_max_latency": float(max(final_latency) if final_latency else 0.0),
        "initial_min_latency": float(min(initial_latency) if initial_latency else 0.0),
        "final_min_latency": float(min(final_latency) if final_latency else 0.0),
        "latency_reduction_per_user_percent": latency_reduction_per_user_percent,
        "total_latency_reduction_percent": float(total_latency_reduction_percent),
        "initial_asynchronality_matrix": initial_async_matrix,
        "final_asynchronality_matrix": final_async_matrix,
        "initial_asynchronality_pairs": initial_async_pairs,
        "final_asynchronality_pairs": final_async_pairs,
        "initial_asynchronality_sum": float(initial_async_sum),
        "final_asynchronality_sum": float(final_async_sum),
        "asynchronality_reduction_percent": float(async_reduction_percent),
        "initial_avg_snr_db": _mean_for_served_users(initial_snr_db, initial_served_bits),
        "final_avg_snr_db": _mean_for_served_users(final_snr_db, served_bits),
        "initial_avg_sinr_db": _mean_for_served_users(initial_sinr_db, initial_served_bits),
        "final_avg_sinr_db": _mean_for_served_users(final_sinr_db, served_bits),
        "scenario_mode": scenario_mode,
        "target_bits_per_user": target_bits_per_user,
        "initial_unserved_bits_per_user": initial_unserved_bits,
        "unserved_bits_per_user": final_unserved_bits,
        "missed_bits_per_user": final_unserved_bits,
        "total_delivery_ratio": (
            float(sum(served_bits)) / float(sum(target_bits_per_user))
            if sum(target_bits_per_user) > 0
            else 1.0
        ),
        "partially_served_blocks_per_user": partially_served_blocks_per_user,
        "zero_service_blocks_per_user": zero_service_blocks_per_user,
        "skipped_blocks_per_user": skipped_blocks_per_user,
        "per_user_summary": per_user_summary,
        "baseline_comparison_metrics": baseline_comparison_metrics,
        "dispersion_diagnostics": dispersion_diagnostics,
    }


def build_precoder_net_result(
    test_uplinksystem: Any,
    test_data_dict: dict[str, Any],
    *,
    method_name: str,
    cfg_path: str,
    test_seed: int,
    train_seeds: Sequence[int],
    train_artifact: dict[str, Any],
    initial_R_fbl: Sequence[Any],
    initial_n_kl: Sequence[Any],
    initial_n: Sequence[float],
    initial_latency: Sequence[float],
    initial_snr_db: Sequence[float],
    initial_sinr_db: Sequence[float],
    initial_bits_per_symbol: Sequence[float],
    initial_B_kl: Sequence[Sequence[int]] | None = None,
    initial_bits_per_symbol_by_block: Sequence[Sequence[float]] | None = None,
    initial_interference_diag: dict[str, Any] | None = None,
    final_interference_diag: dict[str, Any] | None = None,
    uplink_rate_model: str | None = None,
    naive_full_t_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble uplink Monte Carlo training/testing outputs into the canonical result record."""
    _, final_snr_db = test_uplinksystem.get_SNR()
    _, final_sinr_db = test_uplinksystem.get_SINR()
    if final_interference_diag is None:
        final_interference_diag = collect_uplink_interference_diagnostics(test_uplinksystem)
    final_bits_per_symbol = [
        np.asarray(test_data_dict["B_kl_star_test"][user], dtype=np.float64)
        / np.asarray(test_uplinksystem.n_kl[user], dtype=np.float64)
        for user in range(test_uplinksystem.K)
    ]
    achieved_final_r_fbl = [
        list(map(float, values))
        for values in test_data_dict.get("R_star_test", test_uplinksystem.R_fbl)
    ]
    epoch_history, kkt_status_counts = _flatten_uplink_epoch_history(
        test_data_dict.get("all_user_block_results_test", [])
    )

    result = {
        "method_name": method_name,
        "cfg_path": cfg_path,
        "seed": int(test_seed),
        "train_seeds": [int(v) for v in train_seeds],
        "training_dataset_sizes": [int(v) for v in train_artifact.get("training_dataset_sizes", [])],
        "training_sample_counts_per_user": [
            int(v)
            for v in train_artifact.get(
                "training_sample_counts_per_user",
                train_artifact.get("training_dataset_sizes", []),
            )
        ],
        "training_dataset_summary": train_artifact.get("training_dataset_summary", {}),
        "post_training_summary": train_artifact.get("post_training_summary", {}),
        "precoder_net_training_losses": train_artifact.get("precoder_net_training_losses", []),
        "precoder_net_training_history": train_artifact.get("precoder_net_training_history", {}),
        "precoder_parameterization": train_artifact.get("precoder_parameterization", "unknown"),
        "training_objective": train_artifact.get("training_objective", "unknown"),
        "monte_carlo_training_style": str(
            train_artifact.get(
                "monte_carlo_training_style",
                train_artifact.get("post_training_summary", {}).get(
                    "monte_carlo_training_style",
                    "rollout_query_objective",
                ),
            )
        ),
        "uplink_objective_mode": str(
            train_artifact.get(
                "uplink_objective_mode",
                train_artifact.get("post_training_summary", {}).get("uplink_objective_mode", "unknown"),
            )
        ),
        "beam_reward_mode": str(
            train_artifact.get(
                "beam_reward_mode",
                train_artifact.get("post_training_summary", {}).get("beam_reward_mode", "unknown"),
            )
        ),
        "uplink_rate_model": str(uplink_rate_model or "unknown"),
        "user_model_specs": train_artifact.get("user_model_specs", []),
        "initial_latency": list(map(float, initial_latency)),
        "final_latency": list(map(float, test_uplinksystem.latency)),
        "initial_n": list(map(float, initial_n)),
        "final_n": list(map(int, test_uplinksystem.n)),
        "initial_n_kl": [list(map(float, np.atleast_1d(v))) for v in initial_n_kl],
        "final_n_kl": [list(map(int, v)) for v in test_uplinksystem.n_kl],
        "initial_B_kl": [list(map(int, values)) for values in (initial_B_kl or [[] for _ in range(test_uplinksystem.K)])],
        "initial_R_fbl": [np.asarray(v).tolist() for v in initial_R_fbl],
        "final_R_fbl": achieved_final_r_fbl,
        "committed_final_R_fbl": [np.asarray(v).tolist() for v in test_uplinksystem.R_fbl],
        "C": [list(map(float, values)) for values in test_uplinksystem.C],
        "V": [list(map(float, values)) for values in test_uplinksystem.V],
        "initial_snr_db": list(map(float, initial_snr_db)),
        "final_snr_db": list(map(float, final_snr_db)),
        "initial_sinr_db": list(map(float, initial_sinr_db)),
        "final_sinr_db": list(map(float, final_sinr_db)),
        "initial_bits_per_symbol": list(map(float, initial_bits_per_symbol)),
        "initial_bits_per_symbol_by_block": [
            list(map(float, values))
            for values in (initial_bits_per_symbol_by_block or [[] for _ in range(test_uplinksystem.K)])
        ],
        "final_bits_per_symbol": [list(map(float, vals)) for vals in final_bits_per_symbol],
        "B_kl": [list(map(int, values)) for values in test_data_dict["B_kl_star_test"]],
        "n_kl": [list(map(int, values)) for values in test_data_dict["n_star_test"]],
        "R_fbl": [list(map(float, values)) for values in test_data_dict["R_star_test"]],
        "blocks_per_user": [len(v) for v in test_uplinksystem.n_kl],
        "initial_schedule_source": "random_precoder_baseline",
        "baseline_references": {
            "random_precoder_baseline": {
                "schedule_source": "random_precoder_baseline",
                "latency": list(map(float, initial_latency)),
                "n": list(map(float, initial_n)),
                "n_kl": [list(map(float, np.atleast_1d(v))) for v in initial_n_kl],
                "B_kl": [list(map(int, values)) for values in (initial_B_kl or [[] for _ in range(test_uplinksystem.K)])],
                "R_fbl": [np.asarray(v).tolist() for v in initial_R_fbl],
                "bits_per_symbol": list(map(float, initial_bits_per_symbol)),
                "snr_db": list(map(float, initial_snr_db)),
                "sinr_db": list(map(float, initial_sinr_db)),
            },
        },
        "scenario_mode": test_data_dict.get("scenario_mode", ""),
        "scenario_block_targets": test_data_dict.get("scenario_block_targets", []),
        "initial_interference_diag": initial_interference_diag,
        "final_interference_diag": final_interference_diag,
        "epoch_history": epoch_history,
        "kkt_solve_status_counts": kkt_status_counts,
        "skipped_blocks_per_user": [
            int(v)
            for v in test_data_dict.get(
                "skipped_blocks_per_user",
                [0 for _ in range(test_uplinksystem.K)],
            )
        ],
        "initial_skipped_blocks_per_user": [
            int(v)
            for v in train_artifact.get(
                "initial_skipped_blocks_per_user",
                [0 for _ in range(test_uplinksystem.K)],
            )
        ],
    }
    if isinstance(naive_full_t_baseline, dict):
        result["baseline_references"]["naive_full_T_baseline"] = {
            "schedule_source": str(
                naive_full_t_baseline.get("initial_schedule_source", "naive_full_T_baseline")
            ),
            "latency": [float(v) for v in naive_full_t_baseline.get("initial_latency", [])],
            "n": [float(v) for v in naive_full_t_baseline.get("initial_n", [])],
            "n_kl": [list(map(float, np.atleast_1d(v))) for v in naive_full_t_baseline.get("initial_n_kl", [])],
            "B_kl": [list(map(int, values)) for values in naive_full_t_baseline.get("initial_B_kl", [])],
            "R_fbl": [np.asarray(v).tolist() for v in naive_full_t_baseline.get("initial_R_fbl", [])],
            "bits_per_symbol": [float(v) for v in naive_full_t_baseline.get("initial_bits_per_symbol", [])],
            "snr_db": [float(v) for v in naive_full_t_baseline.get("initial_snr_db", [])],
            "sinr_db": [float(v) for v in naive_full_t_baseline.get("initial_sinr_db", [])],
            "skipped_blocks_per_user": [int(v) for v in naive_full_t_baseline.get("skipped_blocks_per_user", [])],
        }
    result["summary_metrics"] = compute_summary_metrics(result)
    return result


def build_convergence_result(
    uplinksystem: Any,
    convergence_data_dict: dict[str, Any],
    *,
    method_name: str,
    cfg_path: str,
    seed: int,
    initial_R_fbl: Sequence[Any],
    initial_n_kl: Sequence[Any],
    initial_n: Sequence[float],
    initial_latency: Sequence[float],
    initial_snr_db: Sequence[float],
    initial_sinr_db: Sequence[float],
    initial_bits_per_symbol: Sequence[float],
    initial_B_kl: Sequence[Sequence[int]] | None = None,
    initial_bits_per_symbol_by_block: Sequence[Sequence[float]] | None = None,
    initial_interference_diag: dict[str, Any] | None = None,
    final_interference_diag: dict[str, Any] | None = None,
    sim_cfg: dict[str, Any] | None = None,
    naive_full_t_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble an uplink training-only schedule and baseline comparisons for saving."""
    _, final_snr_db = uplinksystem.get_SNR()
    _, final_sinr_db = uplinksystem.get_SINR()
    if final_interference_diag is None:
        final_interference_diag = collect_uplink_interference_diagnostics(uplinksystem)
    final_bits_per_symbol = [
        np.asarray(convergence_data_dict["B_kl_star"][user], dtype=np.float64)
        / np.asarray(uplinksystem.n_kl[user], dtype=np.float64)
        for user in range(uplinksystem.K)
    ]
    achieved_final_r_fbl = [
        list(map(float, values))
        for values in convergence_data_dict.get("R_star", uplinksystem.R_fbl)
    ]
    epoch_history, kkt_status_counts = _flatten_uplink_epoch_history(
        convergence_data_dict.get("all_user_block_results_train", [])
    )

    result = {
        "method_name": method_name,
        "cfg_path": cfg_path,
        "seed": int(seed),
        "convergence_precoder_update_mode": convergence_data_dict.get("convergence_precoder_update_mode", "precoder_net"),
        "precoder_parameterization": convergence_data_dict.get("precoder_parameterization", "unknown"),
        "initial_latency": list(map(float, initial_latency)),
        "final_latency": list(map(float, uplinksystem.latency)),
        "initial_n": list(map(float, initial_n)),
        "final_n": list(map(int, uplinksystem.n)),
        "initial_n_kl": [list(map(float, np.atleast_1d(v))) for v in initial_n_kl],
        "final_n_kl": [list(map(int, values)) for values in uplinksystem.n_kl],
        "initial_B_kl": [list(map(int, values)) for values in (initial_B_kl or [[] for _ in range(uplinksystem.K)])],
        "initial_R_fbl": [np.asarray(v).tolist() for v in initial_R_fbl],
        "final_R_fbl": achieved_final_r_fbl,
        "committed_final_R_fbl": [np.asarray(v).tolist() for v in uplinksystem.R_fbl],
        "C": [list(map(float, values)) for values in uplinksystem.C],
        "V": [list(map(float, values)) for values in uplinksystem.V],
        "initial_snr_db": list(map(float, initial_snr_db)),
        "final_snr_db": list(map(float, final_snr_db)),
        "initial_sinr_db": list(map(float, initial_sinr_db)),
        "final_sinr_db": list(map(float, final_sinr_db)),
        "initial_bits_per_symbol": list(map(float, initial_bits_per_symbol)),
        "initial_bits_per_symbol_by_block": [
            list(map(float, values))
            for values in (initial_bits_per_symbol_by_block or [[] for _ in range(uplinksystem.K)])
        ],
        "final_bits_per_symbol": [list(map(float, values)) for values in final_bits_per_symbol],
        "B_kl": [list(map(int, values)) for values in convergence_data_dict["B_kl_star"]],
        "n_kl": [list(map(int, values)) for values in convergence_data_dict["n_star"]],
        "R_fbl": [list(map(float, values)) for values in convergence_data_dict["R_star"]],
        "blocks_per_user": [len(values) for values in uplinksystem.n_kl],
        "initial_schedule_source": "random_precoder_baseline",
        "baseline_references": {
            "random_precoder_baseline": {
                "schedule_source": "random_precoder_baseline",
                "latency": list(map(float, initial_latency)),
                "n": list(map(float, initial_n)),
                "n_kl": [list(map(float, np.atleast_1d(v))) for v in initial_n_kl],
                "B_kl": [list(map(int, values)) for values in (initial_B_kl or [[] for _ in range(uplinksystem.K)])],
                "R_fbl": [np.asarray(v).tolist() for v in initial_R_fbl],
                "bits_per_symbol": list(map(float, initial_bits_per_symbol)),
                "snr_db": list(map(float, initial_snr_db)),
                "sinr_db": list(map(float, initial_sinr_db)),
            },
        },
        "scenario_mode": convergence_data_dict.get("scenario_mode", ""),
        "scenario_block_targets": convergence_data_dict.get("scenario_block_targets", []),
        "initial_interference_diag": initial_interference_diag,
        "final_interference_diag": final_interference_diag,
        "epoch_history": epoch_history,
        "kkt_solve_status_counts": kkt_status_counts,
        "convergence_stopping_rule": (
            str(sim_cfg.get("convergence_stopping_rule", "unknown"))
            if sim_cfg is not None
            else "unknown"
        ),
        "kkt_tolerances": {
            "primal": float(sim_cfg.get("kkt_primal_tolerance", np.nan)) if sim_cfg is not None else np.nan,
            "complementarity": float(sim_cfg.get("kkt_complementarity_tolerance", np.nan)) if sim_cfg is not None else np.nan,
            "stationarity": float(sim_cfg.get("kkt_stationarity_tolerance", np.nan)) if sim_cfg is not None else np.nan,
        },
        "uplink_rate_model": str(sim_cfg.get("uplink_rate_model", "unknown")) if sim_cfg is not None else "unknown",
        "uplink_objective_mode": str(sim_cfg.get("uplink_objective_mode", "unknown")) if sim_cfg is not None else "unknown",
        "beam_reward_mode": str(sim_cfg.get("beam_reward_mode", "unknown")) if sim_cfg is not None else "unknown",
        "skipped_blocks_per_user": [
            int(v)
            for v in convergence_data_dict.get(
                "skipped_blocks_per_user",
                [0 for _ in range(uplinksystem.K)],
            )
        ],
    }
    if isinstance(naive_full_t_baseline, dict):
        result["baseline_references"]["naive_full_T_baseline"] = {
            "schedule_source": str(
                naive_full_t_baseline.get("initial_schedule_source", "naive_full_T_baseline")
            ),
            "latency": [float(v) for v in naive_full_t_baseline.get("initial_latency", [])],
            "n": [float(v) for v in naive_full_t_baseline.get("initial_n", [])],
            "n_kl": [list(map(float, np.atleast_1d(v))) for v in naive_full_t_baseline.get("initial_n_kl", [])],
            "B_kl": [list(map(int, values)) for values in naive_full_t_baseline.get("initial_B_kl", [])],
            "R_fbl": [np.asarray(v).tolist() for v in naive_full_t_baseline.get("initial_R_fbl", [])],
            "bits_per_symbol": [float(v) for v in naive_full_t_baseline.get("initial_bits_per_symbol", [])],
            "snr_db": [float(v) for v in naive_full_t_baseline.get("initial_snr_db", [])],
            "sinr_db": [float(v) for v in naive_full_t_baseline.get("initial_sinr_db", [])],
            "skipped_blocks_per_user": [int(v) for v in naive_full_t_baseline.get("skipped_blocks_per_user", [])],
        }
    result["summary_metrics"] = compute_summary_metrics(result)
    return result


def _sum_per_user_summary_field(metrics: dict[str, Any], field: str) -> int:
    return int(sum(int(row.get(field, 0)) for row in metrics.get("per_user_summary", [])))


def _get_baseline_comparison_metric(
    metrics: dict[str, Any],
    baseline_name: str,
) -> dict[str, Any]:
    baseline_metrics = metrics.get("baseline_comparison_metrics", {})
    if not isinstance(baseline_metrics, dict):
        return {}
    value = baseline_metrics.get(baseline_name, {})
    return value if isinstance(value, dict) else {}


def _build_uplink_final_test_section_lines(result: dict[str, Any]) -> list[str]:
    metrics = result["summary_metrics"]
    random_baseline_metrics = _get_baseline_comparison_metric(metrics, "random_precoder_baseline")
    naive_full_t_metrics = _get_baseline_comparison_metric(metrics, "naive_full_T_baseline")
    dispersion_diagnostics = metrics.get("dispersion_diagnostics", {})
    lines = [
        "Final test results",
        f"Initial total latency (random baseline): {metrics['initial_total_latency']:.6f}",
        f"Final total latency: {metrics['final_total_latency']:.6f}",
        (
            "Latency reduction vs random precoder baseline (%): "
            f"{random_baseline_metrics.get('total_latency_reduction_percent', metrics['total_latency_reduction_percent']):.4f}"
        ),
        f"Initial avg latency (random baseline): {metrics['initial_avg_latency']:.6f}",
        f"Final avg latency: {metrics['final_avg_latency']:.6f}",
        f"Initial asynchronality sum (random baseline): {metrics['initial_asynchronality_sum']:.6f}",
        f"Final asynchronality sum: {metrics['final_asynchronality_sum']:.6f}",
        f"Asynchronality reduction (%): {metrics['asynchronality_reduction_percent']:.4f}",
        _format_optional_db("Initial avg served-user SNR (dB)", metrics["initial_avg_snr_db"]),
        _format_optional_db("Final avg served-user SNR (dB)", metrics["final_avg_snr_db"]),
        _format_optional_db("Initial avg served-user SINR (dB)", metrics["initial_avg_sinr_db"]),
        _format_optional_db("Final avg served-user SINR (dB)", metrics["final_avg_sinr_db"]),
        (
            "Final served-block avg capacity C: "
            f"{float(dispersion_diagnostics.get('avg_capacity', 0.0)):.6f}"
        ),
        (
            "Final served-block avg dispersion V: "
            f"{float(dispersion_diagnostics.get('avg_dispersion', 0.0)):.6f}"
        ),
        (
            "Final served-block avg finite-blocklength penalty: "
            f"{float(dispersion_diagnostics.get('avg_fbl_penalty', 0.0)):.6f}"
        ),
        (
            "Final served-block avg penalty/C ratio: "
            f"{float(dispersion_diagnostics.get('avg_penalty_over_capacity', 0.0)):.6f}"
        ),
        f"Total served bits: {_sum_per_user_summary_field(metrics, 'served_bits')}",
        f"Total skipped blocks: {_sum_per_user_summary_field(metrics, 'skipped_blocks')}",
    ]
    if naive_full_t_metrics:
        lines.extend(
            [
                f"Initial total latency (naive full-T baseline): {naive_full_t_metrics.get('initial_total_latency', 0.0):.6f}",
                (
                    "Latency reduction vs naive full-T baseline (%): "
                    f"{naive_full_t_metrics.get('total_latency_reduction_percent', 0.0):.4f}"
                ),
            ]
        )
    if metrics.get("scenario_mode", "") == STREAMING_MODE:
        lines.extend(
            [
                f"Total target bits: {_sum_per_user_summary_field(metrics, 'target_bits')}",
                f"Total unserved bits: {_sum_per_user_summary_field(metrics, 'unserved_bits')}",
                f"Overall delivery ratio: {float(metrics.get('total_delivery_ratio', 0.0)):.6f}",
                f"Total partially served blocks: {_sum_per_user_summary_field(metrics, 'partially_served_blocks')}",
                f"Total zero-service blocks: {_sum_per_user_summary_field(metrics, 'zero_service_blocks')}",
            ]
        )
    return lines


def _build_uplink_per_user_test_lines(result: dict[str, Any]) -> list[str]:
    metrics = result["summary_metrics"]
    naive_full_t_metrics = _get_baseline_comparison_metric(metrics, "naive_full_T_baseline")
    naive_per_user = naive_full_t_metrics.get("latency_reduction_per_user_percent", [])
    lines = ["Per-user final test details"]
    for row in metrics["per_user_summary"]:
        parts = [
            f"User {row['user']}",
            f"init_lat={row['initial_latency']:.6f}",
            f"final_lat={row['final_latency']:.6f}",
            f"lat_red_vs_random={row['latency_reduction_percent']:.4f}%",
            (
                f"init_snr={row['initial_snr_db']:.4f} dB"
                if int(row["initial_served_bits"]) > 0
                else "init_snr=n/a"
            ),
            (
                f"final_snr={row['final_snr_db']:.4f} dB"
                if int(row["served_bits"]) > 0
                else "final_snr=n/a"
            ),
            (
                f"init_sinr={row['initial_sinr_db']:.4f} dB"
                if int(row["initial_served_bits"]) > 0
                else "init_sinr=n/a"
            ),
            (
                f"final_sinr={row['final_sinr_db']:.4f} dB"
                if int(row["served_bits"]) > 0
                else "final_sinr=n/a"
            ),
            f"blocks={row['blocks']}",
            f"total_n={row['total_n']}",
            f"served_bits={row['served_bits']}",
            f"skipped_blocks={row.get('skipped_blocks', 0)}",
        ]
        if int(row["user"]) < len(naive_per_user):
            parts.append(f"lat_red_vs_fullT={float(naive_per_user[int(row['user'])]):.4f}%")
        if metrics.get("scenario_mode", "") == STREAMING_MODE:
            parts.extend(
                [
                    f"target_bits={row.get('target_bits', 0)}",
                    f"init_unserved_bits={row.get('initial_unserved_bits', 0)}",
                    f"unserved_bits={row.get('unserved_bits', 0)}",
                    f"delivery_ratio={float(row.get('delivery_ratio', 0.0)):.6f}",
                    f"successful_blocks={row.get('successful_blocks', 0)}",
                    f"deadline_miss_fraction={float(row.get('deadline_miss_fraction', 0.0)):.6f}",
                    f"avg_n_kl={float(row.get('average_selected_blocklength', 0.0)):.4f}",
                    f"partial_blocks={row.get('partially_served_blocks', 0)}",
                    f"zero_service_blocks={row.get('zero_service_blocks', 0)}",
                ]
            )
        lines.append(" | ".join(parts))
    return lines


def build_dispersion_diagnostic_lines(result: dict[str, Any]) -> list[str]:
    metrics = result.get("summary_metrics", {})
    diagnostics = metrics.get("dispersion_diagnostics", {}) if isinstance(metrics, dict) else {}
    if not isinstance(diagnostics, dict) or int(diagnostics.get("accepted_block_count", 0)) <= 0:
        return []

    lines = [
        "Uplink dispersion diagnostics",
        "",
        "Terms",
        "Capacity C: realized-block conditional capacity used by the simulator.",
        "Dispersion V: realized-block conditional dispersion term used by the simulator.",
        "Finite-blocklength penalty: C - R_fbl for the accepted served block.",
        "Penalty/C ratio: fraction of realized capacity removed by the finite-blocklength penalty.",
        "",
        f"Accepted served blocks: {int(diagnostics.get('accepted_block_count', 0))}",
        f"Average capacity C: {float(diagnostics.get('avg_capacity', 0.0)):.6f}",
        f"Average dispersion V: {float(diagnostics.get('avg_dispersion', 0.0)):.6f}",
        f"Average finite-blocklength penalty: {float(diagnostics.get('avg_fbl_penalty', 0.0)):.6f}",
        f"Average penalty/C ratio: {float(diagnostics.get('avg_penalty_over_capacity', 0.0)):.6f}",
        f"Maximum penalty/C ratio: {float(diagnostics.get('max_penalty_over_capacity', 0.0)):.6f}",
        "",
        "Per accepted block",
    ]
    for row in diagnostics.get("accepted_block_records", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            " | ".join(
                [
                    f"user={int(row.get('user', -1))}",
                    f"block={int(row.get('block', -1))}",
                    f"served_bits={int(row.get('served_bits', 0))}",
                    f"n_kl={int(row.get('n_kl', 0))}",
                    f"C={float(row.get('capacity', np.nan)):.6f}",
                    f"V={float(row.get('dispersion', np.nan)):.6f}",
                    f"penalty={float(row.get('fbl_penalty', np.nan)):.6f}",
                    f"penalty_over_C={float(row.get('penalty_over_capacity', np.nan)):.6f}",
                    f"achieved_R_fbl={float(row.get('achieved_rate', np.nan)):.6f}",
                    f"required_rate={float(row.get('required_rate', np.nan)):.6f}",
                    f"margin={float(row.get('rate_margin', np.nan)):.6f}",
                ]
            )
        )
    return lines


def build_convergence_summary_lines(result: dict[str, Any]) -> list[str]:
    metrics = result["summary_metrics"]
    lines = [
        "Uplink optimizer summary",
        "",
        "Setup",
        f"Method: {result.get('method_name', 'unknown')}",
        f"Config: {result.get('cfg_path', 'unknown')}",
        f"Seed: {int(result.get('seed', 0))}",
        f"Run started at: {result.get('run_started_at_local', 'unknown')}",
        f"Run completed at: {result.get('run_completed_at_local', 'unknown')}",
        f"Scenario: {result.get('scenario_mode', 'unknown')}",
        f"Uplink rate model: {result.get('uplink_rate_model', 'unknown')}",
        f"Uplink objective mode: {result.get('uplink_objective_mode', 'unknown')}",
        f"Beam reward mode: {result.get('beam_reward_mode', 'unknown')}",
        f"Convergence precoder update mode: {result.get('convergence_precoder_update_mode', 'unknown')}",
        f"Precoder parameterization: {result.get('precoder_parameterization', 'unknown')}",
        f"Initial schedule source: {result.get('initial_schedule_source', 'unknown')}",
    ]
    lines.extend([""])
    lines.extend(_build_uplink_final_test_section_lines(result))
    lines.extend([""])
    lines.extend(_build_uplink_per_user_test_lines(result))

    if metrics["initial_asynchronality_pairs"]:
        lines.extend(["", "Per-pair asynchronality"])
        for init_pair, final_pair in zip(metrics["initial_asynchronality_pairs"], metrics["final_asynchronality_pairs"]):
            lines.append(
                " | ".join(
                    [
                        f"Users {init_pair['user_i']}-{init_pair['user_j']}",
                        f"initial_diff={init_pair['abs_latency_diff']:.6f}",
                        f"final_diff={final_pair['abs_latency_diff']:.6f}",
                    ]
                )
            )

    lines.extend(format_experiment_cost_lines(result.get("experiment_cost")))
    return lines


def build_summary_lines(result: dict[str, Any]) -> list[str]:
    metrics = result["summary_metrics"]
    dataset_summary = result.get("training_dataset_summary", {})
    post_training_summary = result.get("post_training_summary", {})
    lines = [
        "Uplink optimizer summary",
        "",
        "Setup",
        f"Method: {result.get('method_name', 'unknown')}",
        f"Config: {result.get('cfg_path', 'unknown')}",
        f"Test seed: {int(result.get('seed', 0))}",
        f"Train seeds: {result.get('train_seeds', [])}",
        f"Run started at: {result.get('run_started_at_local', 'unknown')}",
        f"Run completed at: {result.get('run_completed_at_local', 'unknown')}",
        f"Scenario: {result.get('experiment_scenario_mode', 'unknown')}",
        f"Uplink rate model: {result.get('uplink_rate_model', 'unknown')}",
        f"Uplink objective mode: {result.get('uplink_objective_mode', 'unknown')}",
        f"Beam reward mode: {result.get('beam_reward_mode', 'unknown')}",
        f"Precoder parameterization: {result.get('precoder_parameterization', 'unknown')}",
        f"Training objective: {result.get('training_objective', 'unknown')}",
        f"Monte Carlo training style: {result.get('monte_carlo_training_style', 'unknown')}",
        f"Test n search strategy: {result.get('test_n_search_strategy', 'config_default')}",
        f"Test n search direction: {result.get('test_n_search_direction', 'config_default')}",
        f"Initial schedule source: {result.get('initial_schedule_source', 'unknown')}",
        f"Training dataset sample kind: {dataset_summary.get('training_sample_kind', 'unknown') if isinstance(dataset_summary, dict) else 'unknown'}",
        f"Training dataset sample unit: {dataset_summary.get('training_sample_unit', 'unknown') if isinstance(dataset_summary, dict) else 'unknown'}",
        f"Training dataset total samples: {int(dataset_summary.get('total_training_samples', 0)) if isinstance(dataset_summary, dict) else 0}",
        f"Training SNR ranges by user (dB): {dataset_summary.get('training_snr_db_ranges', []) if isinstance(dataset_summary, dict) else []}",
        f"Training sample counts per user: {result.get('training_sample_counts_per_user', result.get('training_dataset_sizes', []))}",
    ]
    if result.get("reused_training_artifact"):
        lines.append(f"Reused training artifact: {result.get('reused_training_artifact')}")
    lines.extend([""])
    lines.extend(_build_uplink_final_test_section_lines(result))
    if isinstance(post_training_summary, dict) and len(post_training_summary) > 0:
        lines.extend(
            [
                "",
                "Training results",
                f"Epochs requested: {int(post_training_summary.get('epochs_requested', 0))}",
                f"Configured max epochs: {int(post_training_summary.get('configured_max_epochs', post_training_summary.get('epochs_requested', 0)))}",
                f"Per-user epochs completed: {post_training_summary.get('per_user_epochs_completed', [])}",
                f"Per-user training solve status: {post_training_summary.get('per_user_training_solve_status', [])}",
                f"Training sample unit: {post_training_summary.get('training_sample_unit', 'unknown')}",
                f"Base training dataset: {post_training_summary.get('base_dataset_kind', 'unknown')}",
                f"Training samples: {int(post_training_summary.get('total_training_samples', 0))}",
                f"Monte Carlo training style: {post_training_summary.get('monte_carlo_training_style', result.get('monte_carlo_training_style', 'unknown'))}",
                f"Rollout anchor-bits mode: {post_training_summary.get('rollout_anchor_bits_mode', 'unknown')}",
                f"Last epoch mean per-user rollout rate: {float(post_training_summary.get('last_epoch_mean_user_rollout_rate', post_training_summary.get('final_avg_user_rate', 0.0))):.6f}",
                f"Best epoch mean per-user rollout rate: {float(post_training_summary.get('best_epoch_mean_user_rollout_rate', post_training_summary.get('best_avg_user_rate', 0.0))):.6f}",
                f"Last epoch mean per-user objective loss: {float(post_training_summary.get('last_epoch_mean_user_rollout_objective_loss', 0.0)):.6f}",
                f"Best epoch mean per-user objective loss: {float(post_training_summary.get('best_epoch_mean_user_rollout_objective_loss', 0.0)):.6f}",
                (
                    "Per-user last epoch avg rate over rollout queries: "
                    f"{post_training_summary.get('per_user_last_epoch_avg_rate_over_rollout_queries', post_training_summary.get('per_user_final_rate', []))}"
                ),
                (
                    "Per-user last epoch avg objective loss over rollout queries: "
                    f"{post_training_summary.get('per_user_last_epoch_avg_objective_loss_over_rollout_queries', [])}"
                ),
                (
                    "Last epoch feasible rollout queries: "
                    f"{int(post_training_summary.get('last_epoch_feasible_rollout_queries', 0))} / "
                    f"{max(int(post_training_summary.get('last_epoch_total_rollout_queries', 0)), 0)}"
                ),
            ]
        )
    lines.extend([""])
    lines.extend(_build_uplink_per_user_test_lines(result))

    if metrics["initial_asynchronality_pairs"]:
        lines.extend(["", "Per-pair asynchronality"])
        for init_pair, final_pair in zip(metrics["initial_asynchronality_pairs"], metrics["final_asynchronality_pairs"]):
            lines.append(
                " | ".join(
                    [
                        f"Users {init_pair['user_i']}-{init_pair['user_j']}",
                        f"initial_diff={init_pair['abs_latency_diff']:.6f}",
                        f"final_diff={final_pair['abs_latency_diff']:.6f}",
                    ]
                )
            )
    if isinstance(post_training_summary, dict) and len(post_training_summary) > 0:
        lines.extend(
            [
                "",
                "Additional training details",
                f"Cumulative rollout queries by n_kl: {post_training_summary.get('cumulative_rollout_queries_by_n_kl', {}).get('global_rollout_queries_by_n_kl_over_all_epochs', {})}",
                f"Cumulative frontier rollout queries by n_kl: {post_training_summary.get('cumulative_frontier_rollout_queries_by_n_kl', {}).get('global_frontier_rollout_queries_by_n_kl_over_all_epochs', {})}",
                f"Train-eval initial blocks per user: {post_training_summary.get('train_eval_initial_blocks_per_user', [])}",
                f"Train-eval final blocks per user: {post_training_summary.get('train_eval_blocks_per_user', [])}",
                f"Train-eval initial total n per user: {post_training_summary.get('train_eval_initial_total_n_per_user', [])}",
                f"Train-eval final total n per user: {post_training_summary.get('train_eval_total_n_per_user', [])}",
                (
                    "Train-eval total latency reduction (%): "
                    f"{float(post_training_summary.get('train_eval_total_latency_reduction_percent', 0.0)):.4f}"
                ),
                f"Train-eval initial selected n_kl summary: {post_training_summary.get('train_eval_initial_selected_n_kl_summary', {})}",
                f"Train-eval final selected n_kl summary: {post_training_summary.get('train_eval_selected_n_kl_summary', {})}",
            ]
        )
    lines.extend(format_experiment_cost_lines(result.get("experiment_cost")))
    lines.extend(
        [
            "",
            "Terminology",
            "- channel episode: one (seed, user, block=0) channel realization stored in the base dataset",
            "- rollout query: one visited (episode, n_kl) state generated online from the current precoder net",
            "- initial schedule: the random-precoder baseline used for the before-optimization uplink latency",
        ]
    )
    return lines


def build_training_dataset_summary_lines(dataset_summary: dict[str, Any]) -> list[str]:
    lines = [
        "Uplink training dataset summary",
        f"Training sample kind: {dataset_summary.get('training_sample_kind', 'unknown')}",
        f"Total training samples: {int(dataset_summary.get('total_training_samples', 0))}",
        f"Training sample unit: {dataset_summary.get('training_sample_unit', 'unknown')}",
        f"Base dataset kind: {dataset_summary.get('base_dataset_kind', 'unknown')}",
        f"Training scenario modes: {dataset_summary.get('scenario_modes', [])}",
        f"Training SNR ranges by user (dB): {dataset_summary.get('training_snr_db_ranges', [])}",
        f"Training samples by seed: {dataset_summary.get('training_samples_by_seed', {})}",
        f"Training samples by user SNR (dB): {dataset_summary.get('training_samples_by_user_snr_db', [])}",
        f"Training samples per user: {dataset_summary.get('training_samples_per_user', [])}",
    ]
    lines.extend(
        [
            "",
            "Terminology",
            "- payload channel episode: one seed-based payload rollout anchor; later blocks are generated online",
            "- streaming block sample: one independent seed-based block with a fresh per-block bit target",
        ]
    )
    return lines


def build_post_training_summary_lines(post_training_summary: dict[str, Any]) -> list[str]:
    lines = [
        "Uplink post-training summary",
        f"Train-eval seed: {int(post_training_summary.get('train_eval_seed', 0))}",
        f"Train-eval SNRs by user (dB): {post_training_summary.get('train_eval_snr_db_by_user', [])}",
        f"Run started at: {post_training_summary.get('run_started_at_local', 'unknown')}",
        f"Training started at: {post_training_summary.get('training_started_at_local', 'unknown')}",
        f"Training completed at: {post_training_summary.get('training_completed_at_local', 'unknown')}",
        f"Epochs requested: {int(post_training_summary.get('epochs_requested', 0))}",
        f"Configured max epochs: {int(post_training_summary.get('configured_max_epochs', post_training_summary.get('epochs_requested', 0)))}",
        f"Training sample unit: {post_training_summary.get('training_sample_unit', 'unknown')}",
        f"Base training dataset: {post_training_summary.get('base_dataset_kind', 'unknown')}",
        f"Monte Carlo training style: {post_training_summary.get('monte_carlo_training_style', 'unknown')}",
        f"Rollout anchor-bits mode: {post_training_summary.get('rollout_anchor_bits_mode', 'unknown')}",
        f"Per-user num epochs: {post_training_summary.get('per_user_num_epochs', [])}",
        f"Per-user epochs completed: {post_training_summary.get('per_user_epochs_completed', [])}",
        f"Per-user training solve status: {post_training_summary.get('per_user_training_solve_status', [])}",
        f"Per-user restored solution source: {post_training_summary.get('per_user_restored_solution_source', [])}",
        f"Training samples: {int(post_training_summary.get('total_training_samples', 0))}",
        f"Last epoch mean per-user rollout rate: {float(post_training_summary.get('last_epoch_mean_user_rollout_rate', post_training_summary.get('final_avg_user_rate', 0.0))):.6f}",
        f"Best epoch mean per-user rollout rate: {float(post_training_summary.get('best_epoch_mean_user_rollout_rate', post_training_summary.get('best_avg_user_rate', 0.0))):.6f}",
        f"Last epoch mean per-user objective loss: {float(post_training_summary.get('last_epoch_mean_user_rollout_objective_loss', 0.0)):.6f}",
        f"Best epoch mean per-user objective loss: {float(post_training_summary.get('best_epoch_mean_user_rollout_objective_loss', 0.0)):.6f}",
        (
            "Per-user last epoch avg rate over rollout queries: "
            f"{post_training_summary.get('per_user_last_epoch_avg_rate_over_rollout_queries', post_training_summary.get('per_user_final_rate', []))}"
        ),
        (
            "Per-user last epoch avg objective loss over rollout queries: "
            f"{post_training_summary.get('per_user_last_epoch_avg_objective_loss_over_rollout_queries', [])}"
        ),
        (
            "Last epoch feasible rollout queries: "
            f"{int(post_training_summary.get('last_epoch_feasible_rollout_queries', 0))} / "
            f"{max(int(post_training_summary.get('last_epoch_total_rollout_queries', 0)), 0)}"
        ),
        f"Per-user best objective loss: {post_training_summary.get('per_user_best_objective_loss', [])}",
        f"Per-user final KKT primal residual: {post_training_summary.get('per_user_final_kkt_primal_residual', [])}",
        f"Per-user final KKT complementarity residual: {post_training_summary.get('per_user_final_kkt_complementarity_residual', [])}",
        f"Per-user final KKT stationarity residual: {post_training_summary.get('per_user_final_kkt_stationarity_residual', [])}",
        f"Cumulative rollout queries by n_kl: {post_training_summary.get('cumulative_rollout_queries_by_n_kl', {}).get('global_rollout_queries_by_n_kl_over_all_epochs', {})}",
        f"Cumulative frontier rollout queries by n_kl: {post_training_summary.get('cumulative_frontier_rollout_queries_by_n_kl', {}).get('global_frontier_rollout_queries_by_n_kl_over_all_epochs', {})}",
        f"Last epoch rollout queries by n_kl: {post_training_summary.get('final_epoch_rollout_query_summary', {}).get('global_rollout_queries_by_n_kl', {})}",
        f"Last epoch frontier rollout queries by n_kl: {post_training_summary.get('final_epoch_rollout_query_summary', {}).get('global_frontier_rollout_queries_by_n_kl', {})}",
        f"Train-eval initial latency: {post_training_summary.get('train_eval_initial_latency', [])}",
        f"Train-eval final latency: {post_training_summary.get('train_eval_final_latency', [])}",
        f"Train-eval initial blocks per user: {post_training_summary.get('train_eval_initial_blocks_per_user', [])}",
        f"Train-eval final blocks per user: {post_training_summary.get('train_eval_blocks_per_user', [])}",
        f"Train-eval initial total n per user: {post_training_summary.get('train_eval_initial_total_n_per_user', [])}",
        f"Train-eval final total n per user: {post_training_summary.get('train_eval_total_n_per_user', [])}",
        f"Train-eval initial served bits per user: {post_training_summary.get('train_eval_initial_served_bits_per_user', [])}",
        f"Train-eval final served bits per user: {post_training_summary.get('train_eval_served_bits_per_user', [])}",
        f"Train-eval initial skipped blocks per user: {post_training_summary.get('train_eval_initial_skipped_blocks_per_user', [])}",
        f"Train-eval final skipped blocks per user: {post_training_summary.get('train_eval_skipped_blocks_per_user', [])}",
        (
            "Train-eval total latency reduction (%): "
            f"{float(post_training_summary.get('train_eval_total_latency_reduction_percent', 0.0)):.4f}"
        ),
        "Train-eval initial selected n_kl summary:",
        f"{post_training_summary.get('train_eval_initial_selected_n_kl_summary', {})}",
        "Train-eval selected n_kl summary:",
        f"{post_training_summary.get('train_eval_selected_n_kl_summary', {})}",
    ]
    lines.extend(format_experiment_cost_lines(post_training_summary.get("experiment_cost")))
    lines.extend(
        [
            "",
            "Terminology",
            "- train-eval: evaluation of the trained precoder nets on the first training seed",
            "- channel episode: one (seed, user, block=0) channel realization stored in the base dataset",
            "- rollout query: one visited (episode, n_kl) state generated online from the current precoder net",
            "- initial schedule: the random-precoder baseline used for the before-training latency",
        ]
    )
    return lines
