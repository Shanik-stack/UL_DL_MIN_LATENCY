"""Result assembly for downlink Monte Carlo evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from latency_optimization.results.metrics import build_schedule_reference
from latency_optimization.results.persistence import make_serializable

from ..config import validate_shared_bs_streaming_blocklength_input_mode
from ..model_service import describe_precoder_parameterization
from ..precoders.checkpoints import export_user_model_specs


def build_evaluation_result(
    *,
    system: Any,
    sim_params: dict[str, Any],
    method_name: str,
    objective_name: str,
    weight_strategy: str,
    allocation_mode: str,
    model_scope: str,
    n_plan: Sequence[Sequence[int]],
    bits_plan: Sequence[Sequence[int]],
    rate_plan: Sequence[Sequence[float]],
    initial_latency: Sequence[float],
    initial_plan: dict[str, Any],
    full_block_latency: Sequence[float],
    full_block_plan: dict[str, Any],
    initial_snr_db: Sequence[float],
    final_snr_db: Sequence[float],
    initial_sinr_db: Sequence[float],
    final_sinr_db: Sequence[float],
    initial_interference: dict[str, Any],
    final_interference: dict[str, Any],
    outer_history: Sequence[dict[str, Any]],
    epoch_history: Sequence[Any],
    rate_points: Sequence[Any],
    training_history: dict[str, Any] | None,
    train_seeds: Sequence[int] | None,
    training_dataset_sizes: Sequence[int] | None,
    skipped_blocks_per_user: Sequence[int],
    evaluation_cost_counters: dict[str, Any],
    evaluation_wall_time_seconds: float,
    scenario_mode: str,
    scenario_block_targets: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    history = training_history or {}
    result = {
        "method_name": str(method_name),
        "objective_mode": str(objective_name),
        "allocation_mode": str(allocation_mode),
        "weight_strategy": str(weight_strategy),
        "precoder_parameterization": describe_precoder_parameterization(
            model_scope,
            "precoder_net",
            uses_blocklength_input=True,
        ),
        "downlink_precoder_net_scope": str(model_scope),
        "shared_bs_streaming_blocklength_input_mode": (
            validate_shared_bs_streaming_blocklength_input_mode(
                sim_params.get("shared_bs_streaming_blocklength_input_mode", "joint_blocklength_vector")
            )
            if str(model_scope) == "bs_shared_net"
            else "not_applicable"
        ),
        "user_model_specs": export_user_model_specs(
            system.Nr,
            system.Nb,
            system.dk,
            uses_blocklength_input=True,
            context_k=system.K,
            context_max_nr=int(np.max(system.Nr)),
            context_max_nb=int(np.max(system.Nb)),
            context_max_dk=int(np.max(system.dk)),
            model_scope=model_scope,
        ),
        "n_kl": [list(map(int, values)) for values in n_plan],
        "B_kl": [list(map(int, values)) for values in bits_plan],
        "R_fbl": [list(map(float, values)) for values in system.R_fbl],
        "R_alloc": [list(map(float, values)) for values in rate_plan],
        "initial_latency": list(map(float, initial_latency)),
        "initial_plan": initial_plan,
        "initial_baseline_completed": bool(initial_plan.get("completed", True)),
        "initial_baseline_failure": str(initial_plan.get("failure_reason", "")),
        "initial_schedule_source": "random_precoder_baseline",
        "baseline_references": {
            "random_precoder_baseline": build_schedule_reference(
                "random_precoder_baseline",
                initial_latency,
                initial_plan,
                snr_db=initial_snr_db,
                sinr_db=initial_sinr_db,
            ),
            "naive_full_T_baseline": build_schedule_reference(
                "naive_full_T_baseline",
                full_block_latency,
                full_block_plan,
                snr_db=initial_snr_db,
                sinr_db=initial_sinr_db,
            ),
        },
        "initial_interference_diag": initial_interference,
        "final_latency": system.latency.tolist(),
        "initial_snr_db": list(map(float, initial_snr_db)),
        "final_snr_db": list(map(float, final_snr_db)),
        "initial_sinr_db": list(map(float, initial_sinr_db)),
        "final_sinr_db": list(map(float, final_sinr_db)),
        "final_interference_diag": final_interference,
        "outer_history": list(outer_history),
        "epoch_history": list(epoch_history),
        "rate_points": list(rate_points),
        "blocks_per_user": [len(values) for values in n_plan],
        "precoder_net_training_losses": [
            list(map(float, row)) for row in history.get("per_user_objective_loss", [])
        ],
        "precoder_net_training_history": make_serializable(history),
        "train_seeds": [int(value) for value in (train_seeds or [])],
        "training_dataset_sizes": [int(value) for value in (training_dataset_sizes or [])],
        "training_active_user_case_counts_per_user": [
            int(value) for value in (training_dataset_sizes or [])
        ],
        "skipped_blocks_per_user": [int(value) for value in skipped_blocks_per_user],
        "evaluation_cost_counters": evaluation_cost_counters,
        "core_evaluation_wall_time_seconds": float(evaluation_wall_time_seconds),
        "scenario_mode": str(scenario_mode),
    }
    if scenario_block_targets is not None:
        result["scenario_block_targets"] = [list(map(int, row)) for row in scenario_block_targets]
    return result


__all__ = ["build_evaluation_result"]
