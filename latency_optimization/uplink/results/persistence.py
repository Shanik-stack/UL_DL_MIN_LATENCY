"""Write uplink test results in human-readable and machine-readable forms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from latency_optimization.experiments.scenarios import STREAMING_MODE


def _json_compatible(value):
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_pairwise_asynchronality(latencies, stream) -> float:
    total = 0.0
    for first in range(len(latencies)):
        for second in range(first + 1, len(latencies)):
            difference = abs(float(latencies[first]) - float(latencies[second]))
            total += difference
            stream.write(f"User {first} - User {second}: {difference}\n")
    return total


def save_test_results_to_txt(
    test_uplinksystem,
    test_data_dict,
    initial_Rfbl,
    initial_n_kl,
    initial_n,
    initial_latency,
    initial_snr_db,
    initial_sinr_db,
    initial_bits_per_symbol,
    save_dir,
    filename,
    initial_B_kl=None,
    initial_bits_per_symbol_by_block=None,
):
    """Persist the complete before/after uplink evaluation summary."""
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / filename

    final_bits_per_symbol = [
        np.asarray(test_data_dict["B_kl_star_test"][user], dtype=np.float64)
        / np.asarray(test_uplinksystem.n_kl[user], dtype=np.float64)
        for user in range(test_uplinksystem.K)
    ]
    final_rfbl = [
        list(map(float, values))
        for values in test_data_dict.get("R_star_test", test_uplinksystem.R_fbl)
    ]
    _, final_snr_db = test_uplinksystem.get_SNR()
    _, final_sinr_db = test_uplinksystem.get_SINR()

    initial_latency_array = np.asarray(initial_latency, dtype=float)
    final_latency_array = np.asarray(test_uplinksystem.latency, dtype=float)
    latency_reduction_per_user = np.divide(
        initial_latency_array - final_latency_array,
        initial_latency_array,
        out=np.zeros_like(initial_latency_array),
        where=initial_latency_array != 0,
    ) * 100.0
    initial_total_latency = float(np.sum(initial_latency_array))
    final_total_latency = float(np.sum(final_latency_array))
    latency_reduction_percent = (
        100.0 * (initial_total_latency - final_total_latency) / initial_total_latency
        if initial_total_latency > 0.0
        else 0.0
    )

    with text_path.open("w", encoding="utf-8") as stream:
        for user in range(test_uplinksystem.K):
            initial_user_bits = (
                list(initial_B_kl[user])
                if initial_B_kl is not None and user < len(initial_B_kl)
                else [int(test_uplinksystem.B[user])]
            )
            initial_user_bps = (
                list(initial_bits_per_symbol_by_block[user])
                if initial_bits_per_symbol_by_block is not None
                and user < len(initial_bits_per_symbol_by_block)
                else [float(initial_bits_per_symbol[user])]
            )
            stream.write(f"\n|| ---------------- USER {user} ---------------- ||\n")
            stream.write(f"R_fbl: Initial {initial_Rfbl[user]} -> Final {final_rfbl[user]}\n")
            stream.write(f"n_kl: Initial {initial_n_kl[user]} -> Final {test_uplinksystem.n_kl[user]}\n")
            stream.write(f"N_k: Initial {initial_n[user]} -> Final {test_uplinksystem.n[user]}\n")
            stream.write(
                f"B_tx_kl: Initial {initial_user_bits} "
                f"-> Final {test_data_dict['B_kl_star_test'][user]}\n"
            )
            stream.write(
                f"Bits per symbol: Initial avg {initial_bits_per_symbol[user]}, "
                f"per block {initial_user_bps} -> Final {final_bits_per_symbol[user]}\n"
            )
            stream.write(
                f"SNR_dB: Initial {initial_snr_db[user]:.4f} "
                f"-> Final {final_snr_db[user]:.4f}\n"
            )
            stream.write(
                f"SINR_dB: Initial {initial_sinr_db[user]:.4f} "
                f"-> Final {final_sinr_db[user]:.4f}\n"
            )

            if str(test_data_dict.get("scenario_mode", "")) == STREAMING_MODE:
                targets = np.asarray(test_data_dict.get("scenario_block_targets", []), dtype=int)
                if targets.ndim == 2 and user < targets.shape[0]:
                    target_bits = targets[user].tolist()
                    final_bits = list(map(int, test_data_dict["B_kl_star_test"][user]))
                    initial_bits = (
                        list(map(int, initial_B_kl[user]))
                        if initial_B_kl is not None and user < len(initial_B_kl)
                        else [0] * len(target_bits)
                    )
                    initial_unserved = [max(t - b, 0) for t, b in zip(target_bits, initial_bits)]
                    final_unserved = [max(t - b, 0) for t, b in zip(target_bits, final_bits)]
                    stream.write(f"Target bits per block: {target_bits}\n")
                    stream.write(f"Initial unserved bits per block: {initial_unserved}\n")
                    stream.write(f"Final unserved bits per block: {final_unserved}\n")

        stream.write(f"\nInitial latency per user:\n{initial_latency_array}\n")
        stream.write(f"\nFinal latency per user:\n{final_latency_array}\n")
        stream.write(f"Latency reduction per user (%): {latency_reduction_per_user}\n")
        stream.write(f"Total latency reduction (%): {latency_reduction_percent}\n")
        stream.write("\nInitial asynchronality:\n")
        initial_asynchronality = _write_pairwise_asynchronality(initial_latency_array, stream)
        stream.write(f"Initial sum of asynchronality: {initial_asynchronality}\n")
        stream.write("\nFinal asynchronality:\n")
        final_asynchronality = _write_pairwise_asynchronality(final_latency_array, stream)
        stream.write(f"Final sum of asynchronality: {final_asynchronality}\n")
        asynchronality_reduction = (
            100.0 * (initial_asynchronality - final_asynchronality) / initial_asynchronality
            if initial_asynchronality > 0.0
            else 0.0
        )
        stream.write(f"Asynchronality reduction (%): {asynchronality_reduction}\n")

    result = {
        "initial_latency": initial_latency_array,
        "final_latency": final_latency_array,
        "initial_n": initial_n,
        "final_n": test_uplinksystem.n,
        "initial_n_kl": initial_n_kl,
        "final_n_kl": test_uplinksystem.n_kl,
        "initial_Rfbl": initial_Rfbl,
        "final_Rfbl": final_rfbl,
        "committed_final_Rfbl": test_uplinksystem.R_fbl,
        "initial_B_kl": initial_B_kl,
        "target_snr_db": test_uplinksystem.snr_db,
        "initial_snr_db": initial_snr_db,
        "final_snr_db": final_snr_db,
        "initial_sinr_db": initial_sinr_db,
        "final_sinr_db": final_sinr_db,
        "initial_bits_per_symbol": initial_bits_per_symbol,
        "initial_bits_per_symbol_by_block": initial_bits_per_symbol_by_block,
        "final_bits_per_symbol": final_bits_per_symbol,
        "B_tx_final": test_data_dict["B_kl_star_test"],
        "scenario_mode": test_data_dict.get("scenario_mode", ""),
        "scenario_block_targets": test_data_dict.get("scenario_block_targets", []),
        "latency_reduction_per_user_percent": latency_reduction_per_user,
        "total_latency_reduction_percent": latency_reduction_percent,
        "initial_asynchronality_sum": initial_asynchronality,
        "final_asynchronality_sum": final_asynchronality,
        "asynchronality_reduction_percent": asynchronality_reduction,
    }
    json_path = output_dir / filename.replace(".txt", ".json")
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(_json_compatible(result), stream, indent=4)
    print(f"Test results saved to: {text_path}")


__all__ = ["save_test_results_to_txt"]
