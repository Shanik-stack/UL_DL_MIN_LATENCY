"""Torch-only inference for uplink precoder models."""

from __future__ import annotations

import torch
from torch import nn

from latency_optimization.precoders.parameters import as_complex_tensor
from latency_optimization.precoders.power import normalize_matrix_power_torch


def network_output_to_precoder(
    network_output: torch.Tensor,
    transmit_antennas: int,
    streams: int,
) -> torch.Tensor:
    real_imaginary = network_output.squeeze(0).reshape(
        2, int(transmit_antennas), int(streams)
    )
    return torch.complex(real_imaginary[0], real_imaginary[1])


def infer_precoder(
    model: nn.Module,
    channel,
    blocklength: int,
    noise_variance: float,
    error_probability: float,
    transmit_antennas: int,
    streams: int,
    power_limit: float,
) -> torch.Tensor:
    channel_tensor = as_complex_tensor(channel, device=next(model.parameters()).device)
    output = model(channel_tensor, blocklength, noise_variance, error_probability)
    precoder = network_output_to_precoder(output, transmit_antennas, streams)
    return normalize_matrix_power_torch(precoder, power_limit)


__all__ = ["infer_precoder", "network_output_to_precoder"]
