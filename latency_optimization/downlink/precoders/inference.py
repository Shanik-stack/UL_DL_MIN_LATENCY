"""Convert downlink MLP outputs into per-user complex precoders."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .models import model_outputs_full_bs_precoder


def _complex_precoder(output: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    shaped = output.squeeze(0).view(2, rows, columns)
    return (shaped[0] + 1j * shaped[1]).to(torch.complex64)


def _split_full_bs_precoder(
    output: torch.Tensor,
    transmit_antennas: Sequence[int],
    streams: Sequence[int],
    *,
    output_rows: int,
    output_stream_slot: int,
) -> list[torch.Tensor]:
    full = _complex_precoder(output, output_rows, len(streams) * output_stream_slot)
    return [
        full[: int(transmit_antennas[user]), user * output_stream_slot : user * output_stream_slot + int(streams[user])]
        for user in range(len(streams))
    ]


def infer_raw_precoder_torch(
    model: nn.Module,
    channel: torch.Tensor,
    nb: int,
    dk: int,
    *,
    user_index=None,
) -> torch.Tensor:
    output = model(channel, user_index=user_index)
    full = _complex_precoder(output, int(model.output_nb), int(model.output_dk))
    return full[: int(nb), : int(dk)]


def infer_raw_precoder_torch_with_blocklength(
    model: nn.Module,
    channels,
    blocklength: int,
    active_mask,
    noise_plus_interference_covariance: torch.Tensor,
    epsilon: float,
    nb: int,
    dk: int,
    *,
    user_index=None,
) -> torch.Tensor:
    output = model(
        channels,
        int(blocklength),
        active_mask,
        noise_plus_interference_covariance,
        float(epsilon),
        user_index=user_index,
    )
    full = _complex_precoder(output, int(model.output_nb), int(model.output_dk))
    return full[: int(nb), : int(dk)]


def infer_raw_bs_precoders_torch(
    model: nn.Module,
    channels,
    active_mask,
    nb: Sequence[int],
    dk: Sequence[int],
) -> list[torch.Tensor]:
    if not model_outputs_full_bs_precoder(model):
        raise ValueError("A full-BS-output model is required.")
    return _split_full_bs_precoder(
        model(channels, active_mask),
        nb,
        dk,
        output_rows=int(model.output_nb),
        output_stream_slot=int(model.output_dk),
    )


def infer_raw_bs_precoders_torch_with_blocklength(
    model: nn.Module,
    channels,
    blocklengths,
    active_mask,
    noise_variances,
    epsilons,
    nb: Sequence[int],
    dk: Sequence[int],
) -> list[torch.Tensor]:
    if not model_outputs_full_bs_precoder(model):
        raise ValueError("A full-BS-output model is required.")
    return _split_full_bs_precoder(
        model(channels, blocklengths, active_mask, noise_variances, epsilons),
        nb,
        dk,
        output_rows=int(model.output_nb),
        output_stream_slot=int(model.output_dk),
    )


__all__ = [
    "infer_raw_bs_precoders_torch",
    "infer_raw_bs_precoders_torch_with_blocklength",
    "infer_raw_precoder_torch",
    "infer_raw_precoder_torch_with_blocklength",
]
