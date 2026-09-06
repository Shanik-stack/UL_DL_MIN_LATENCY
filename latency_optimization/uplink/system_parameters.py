"""Validate uplink configuration values and derive system parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class SystemParams:
    K: int
    B: np.ndarray
    fs: np.ndarray
    f_carrier: Optional[np.ndarray]
    v: Optional[np.ndarray]
    P: np.ndarray
    NR: np.ndarray
    NT: np.ndarray
    snr_db: np.ndarray
    desired_CNR: np.ndarray
    epsilon: np.ndarray
    D_s: Optional[np.ndarray]
    C_T: Optional[np.ndarray]
    T: np.ndarray
    L: np.ndarray
    n_kl: list[list[int]]
    n: np.ndarray
    latency: np.ndarray
    dk: np.ndarray
    initial_latency: np.ndarray
    initial_bits_per_symbol: list[float]


def _as_user_array(value: Sequence, user_count: int, name: str, dtype=None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    expected_shape = (user_count,)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    return array


def _derive_coherence_blocklengths(
    user_count: int,
    symbol_rates: np.ndarray,
    carrier_frequencies: Sequence[float],
    user_speeds: Sequence[float],
    propagation_speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frequencies = _as_user_array(carrier_frequencies, user_count, "f_carrier", dtype=float)
    speeds = _as_user_array(user_speeds, user_count, "v", dtype=float)
    doppler_shifts = frequencies * speeds / propagation_speed
    if np.any(doppler_shifts <= 0):
        raise ValueError("All Doppler shifts must be positive; check f_carrier and v.")
    coherence_times = 1.0 / doppler_shifts
    blocklengths = np.asarray(coherence_times * symbol_rates, dtype=int)
    return frequencies, speeds, doppler_shifts, coherence_times, blocklengths


def initialize_system_params(
    K: int,
    *,
    B: Sequence[int],
    fs: Sequence[float],
    P: Sequence[float],
    Nr: Sequence[int],
    Nt: Sequence[int],
    snr_db: Sequence[float],
    epsilon: Sequence[float],
    f_carrier: Optional[Sequence[float]] = None,
    v: Optional[Sequence[float]] = None,
    desired_CNR: Optional[Sequence[float]] = None,
    c: float = 299_792_458.0,
    initial_bits_per_symbol: Optional[Sequence[float]] = None,
    T: Optional[Sequence[int]] = None,
) -> dict:
    """Build the validated per-user arrays consumed by :class:`UplinkSystem`."""
    user_count = int(K)
    payload_bits = _as_user_array(B, user_count, "B", dtype=int)
    symbol_rates = _as_user_array(fs, user_count, "fs", dtype=float)
    powers = _as_user_array(P, user_count, "P", dtype=float)
    receive_antennas = _as_user_array(Nr, user_count, "Nr", dtype=int)
    transmit_antennas = _as_user_array(Nt, user_count, "Nt", dtype=int)
    target_snr_db = _as_user_array(snr_db, user_count, "snr_db", dtype=float)
    error_probabilities = _as_user_array(epsilon, user_count, "epsilon", dtype=float)
    desired_cnr = (
        np.zeros(user_count, dtype=float)
        if desired_CNR is None
        else _as_user_array(desired_CNR, user_count, "desired_CNR", dtype=float)
    )

    if T is None:
        if f_carrier is None or v is None:
            raise ValueError("f_carrier and v are required when T is not configured.")
        frequencies, speeds, doppler_shifts, coherence_times, blocklengths = (
            _derive_coherence_blocklengths(
                user_count,
                symbol_rates,
                f_carrier,
                v,
                float(c),
            )
        )
    else:
        frequencies = None
        speeds = None
        doppler_shifts = None
        coherence_times = None
        blocklengths = _as_user_array(T, user_count, "T", dtype=int)

    if np.any(blocklengths <= 0):
        raise ValueError("All coherence blocklengths T must be positive.")
    if np.any(symbol_rates <= 0):
        raise ValueError("All symbol rates fs must be positive.")

    block_counts = np.ones(user_count, dtype=int)
    blocklengths_by_user = [[int(blocklengths[user])] for user in range(user_count)]
    total_channel_uses = blocklengths.copy()
    latency = total_channel_uses / symbol_rates
    spatial_streams = np.minimum(receive_antennas, transmit_antennas)

    if initial_bits_per_symbol is None:
        initial_bps = payload_bits.astype(float) / blocklengths.astype(float)
    else:
        initial_bps = _as_user_array(
            initial_bits_per_symbol,
            user_count,
            "initial_bits_per_symbol",
            dtype=float,
        )
    initial_latency = payload_bits / np.maximum(initial_bps, 1e-12) / symbol_rates

    return asdict(
        SystemParams(
            K=user_count,
            B=payload_bits,
            fs=symbol_rates,
            f_carrier=frequencies,
            v=speeds,
            P=powers,
            NR=receive_antennas,
            NT=transmit_antennas,
            snr_db=target_snr_db,
            desired_CNR=desired_cnr,
            epsilon=error_probabilities,
            D_s=doppler_shifts,
            C_T=coherence_times,
            T=blocklengths,
            L=block_counts,
            n_kl=blocklengths_by_user,
            n=total_channel_uses,
            latency=latency,
            dk=spatial_streams,
            initial_latency=initial_latency,
            initial_bits_per_symbol=initial_bps.tolist(),
        )
    )


__all__ = ["SystemParams", "initialize_system_params"]
