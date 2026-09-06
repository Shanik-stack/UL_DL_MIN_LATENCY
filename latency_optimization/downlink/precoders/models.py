"""The two supported downlink MLP precoder contracts."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn as nn

from latency_optimization.core.validation import require_choice
from latency_optimization.runtime import DEVICE

DOWNLINK_PRECODER_NET_SCOPES = {"per_user_nets", "bs_shared_net"}
CHANNEL_ONLY_INPUT = "channel_only"
BLOCKLENGTH_AWARE_INPUT = "block_context_noise_epsilon_n"
DOWNLINK_MODEL_INPUT_MODES = {CHANNEL_ONLY_INPUT, BLOCKLENGTH_AWARE_INPUT}


def validate_downlink_precoder_net_scope(scope: str) -> str:
    return require_choice(scope, DOWNLINK_PRECODER_NET_SCOPES, "downlink_precoder_net_scope")


def model_outputs_full_bs_precoder(model: nn.Module) -> bool:
    return bool(getattr(model, "outputs_full_bs_precoder", False))


def pad_complex_matrix(matrix: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    padded = torch.zeros((rows, columns), dtype=torch.complex64, device=matrix.device)
    row_count = min(int(matrix.shape[0]), int(rows))
    column_count = min(int(matrix.shape[1]), int(columns))
    padded[:row_count, :column_count] = matrix[:row_count, :column_count].to(torch.complex64)
    return padded


def _channels_as_sequence(channels: Sequence[torch.Tensor] | torch.Tensor) -> list[torch.Tensor]:
    if isinstance(channels, torch.Tensor):
        if channels.dim() == 2:
            return [channels]
        if channels.dim() == 3:
            return [channels[index] for index in range(int(channels.shape[0]))]
        raise ValueError(f"Expected a rank-2 channel or rank-3 channel block, got rank {channels.dim()}.")
    converted = [torch.as_tensor(channel, dtype=torch.complex64, device=DEVICE) for channel in channels]
    if not converted:
        raise ValueError("A channel block must contain at least one user channel.")
    return converted


def _encode_channel_block(
    channels: Sequence[torch.Tensor] | torch.Tensor,
    *,
    user_count: int,
    max_receive_antennas: int,
    max_transmit_antennas: int,
) -> tuple[torch.Tensor, torch.device]:
    channel_list = _channels_as_sequence(channels)
    device = channel_list[0].device
    padded = [
        pad_complex_matrix(channel_list[user].to(device), max_receive_antennas, max_transmit_antennas)
        if user < len(channel_list)
        else torch.zeros(
            (max_receive_antennas, max_transmit_antennas),
            dtype=torch.complex64,
            device=device,
        )
        for user in range(user_count)
    ]
    stacked = torch.stack(padded)
    return torch.cat([stacked.real.reshape(1, -1), stacked.imag.reshape(1, -1)], dim=1), device


def _encode_vector(
    values: Sequence[int | float] | torch.Tensor,
    *,
    length: int,
    device: torch.device,
    transform=None,
) -> torch.Tensor:
    vector = torch.as_tensor(values, dtype=torch.float32, device=device).reshape(1, -1)
    if int(vector.shape[1]) != int(length):
        raise ValueError(f"Expected {length} values, got {int(vector.shape[1])}.")
    return transform(vector) if transform is not None else vector


def _mlp(
    input_size: int,
    output_size: int,
    minimum_width: int,
    output_multipliers: tuple[int, int, int] = (8, 4, 2),
) -> nn.Sequential:
    widths = (
        max(minimum_width, output_multipliers[0] * output_size),
        max(minimum_width // 2, output_multipliers[1] * output_size),
        max(minimum_width // 4, output_multipliers[2] * output_size),
    )
    return nn.Sequential(
        nn.Linear(input_size, widths[0]),
        nn.ReLU(),
        nn.Linear(widths[0], widths[1]),
        nn.ReLU(),
        nn.Linear(widths[1], widths[2]),
        nn.ReLU(),
        nn.Linear(widths[2], output_size),
    )


class PerUserChannelMlp(nn.Module):
    """Maps one user's channel to that user's complex precoder."""

    def __init__(self, receive_antennas: int, transmit_antennas: int, streams: int):
        super().__init__()
        self.output_nb = int(transmit_antennas)
        self.output_dk = int(streams)
        input_size = 2 * int(receive_antennas) * int(transmit_antennas)
        output_size = 2 * self.output_nb * self.output_dk
        self.net = _mlp(input_size, output_size, 256)

    def forward(self, channel: torch.Tensor, *, user_index=None) -> torch.Tensor:
        del user_index
        flattened = channel.to(torch.complex64).reshape(1, -1)
        return self.net(torch.cat([flattened.real, flattened.imag], dim=1))


class PerUserBlocklengthMlp(nn.Module):
    """Maps a full block context and one user's blocklength to its precoder."""

    def __init__(
        self,
        receive_antennas: int,
        transmit_antennas: int,
        streams: int,
        *,
        user_count: int,
        max_receive_antennas: int,
        max_transmit_antennas: int,
    ):
        super().__init__()
        self.k_count = int(user_count)
        self.max_nr = int(max_receive_antennas)
        self.max_nb = int(max_transmit_antennas)
        self.output_nb = int(transmit_antennas)
        self.output_dk = int(streams)
        input_size = 2 * self.k_count * self.max_nr * self.max_nb + self.k_count + 2 * self.max_nr**2 + 2
        output_size = 2 * self.output_nb * self.output_dk
        self.net = _mlp(input_size, output_size, 256)

    def forward(
        self,
        channels: Sequence[torch.Tensor] | torch.Tensor,
        blocklength: int | float,
        active_mask: Sequence[int | float] | torch.Tensor,
        noise_plus_interference_covariance: torch.Tensor,
        epsilon: float,
        *,
        user_index=None,
    ) -> torch.Tensor:
        del user_index
        encoded_channels, device = _encode_channel_block(
            channels,
            user_count=self.k_count,
            max_receive_antennas=self.max_nr,
            max_transmit_antennas=self.max_nb,
        )
        mask = _encode_vector(active_mask, length=self.k_count, device=device)
        covariance = pad_complex_matrix(
            noise_plus_interference_covariance.to(device, dtype=torch.complex64),
            self.max_nr,
            self.max_nr,
        )
        encoded_covariance = torch.cat([covariance.real.reshape(1, -1), covariance.imag.reshape(1, -1)], dim=1)
        metadata = encoded_channels.new_tensor([[math.log1p(max(float(blocklength), 1.0)), float(epsilon)]])
        return self.net(torch.cat([encoded_channels, mask, encoded_covariance, metadata], dim=1))


class SharedBsChannelMlp(nn.Module):
    """Maps all user channels to the full base-station precoder."""

    def __init__(self, *, user_count: int, max_receive_antennas: int, transmit_antennas: int, max_streams: int):
        super().__init__()
        self.k_count = int(user_count)
        self.max_nr = int(max_receive_antennas)
        self.max_nb = int(transmit_antennas)
        self.output_nb = self.max_nb
        self.output_dk = int(max_streams)
        self.outputs_full_bs_precoder = True
        output_size = 2 * self.output_nb * self.k_count * self.output_dk
        input_size = 2 * self.k_count * self.max_nr * self.max_nb + self.k_count
        self.net = _mlp(input_size, output_size, 512, (4, 2, 1))

    def forward(self, channels, active_mask) -> torch.Tensor:
        encoded_channels, device = _encode_channel_block(
            channels,
            user_count=self.k_count,
            max_receive_antennas=self.max_nr,
            max_transmit_antennas=self.max_nb,
        )
        mask = _encode_vector(active_mask, length=self.k_count, device=device)
        return self.net(torch.cat([encoded_channels, mask], dim=1))


class SharedBsBlocklengthMlp(SharedBsChannelMlp):
    """Maps joint channels, blocklengths, noise, and activity to the full BS precoder."""

    def __init__(self, *, user_count: int, max_receive_antennas: int, transmit_antennas: int, max_streams: int):
        nn.Module.__init__(self)
        self.k_count = int(user_count)
        self.max_nr = int(max_receive_antennas)
        self.max_nb = int(transmit_antennas)
        self.output_nb = self.max_nb
        self.output_dk = int(max_streams)
        self.outputs_full_bs_precoder = True
        output_size = 2 * self.output_nb * self.k_count * self.output_dk
        input_size = 2 * self.k_count * self.max_nr * self.max_nb + 4 * self.k_count
        self.net = _mlp(input_size, output_size, 512, (4, 2, 1))

    def forward(self, channels, blocklengths, active_mask, noise_variances, epsilons) -> torch.Tensor:
        encoded_channels, device = _encode_channel_block(
            channels,
            user_count=self.k_count,
            max_receive_antennas=self.max_nr,
            max_transmit_antennas=self.max_nb,
        )
        mask = _encode_vector(active_mask, length=self.k_count, device=device)
        n_values = _encode_vector(
            blocklengths,
            length=self.k_count,
            device=device,
            transform=lambda value: torch.log1p(torch.clamp(value, min=1.0)),
        )
        noise = _encode_vector(
            noise_variances,
            length=self.k_count,
            device=device,
            transform=lambda value: torch.log1p(torch.clamp(value, min=0.0)),
        )
        epsilon = _encode_vector(epsilons, length=self.k_count, device=device)
        return self.net(torch.cat([encoded_channels, mask, n_values, noise, epsilon], dim=1))


def build_user_precoder_net(nr: int, nb: int, dk: int, *, device: torch.device = DEVICE) -> nn.Module:
    return PerUserChannelMlp(nr, nb, dk).to(device)


def build_user_precoder_net_with_blocklength(
    nr: int,
    nb: int,
    dk: int,
    *,
    k_count: int,
    max_nr: int,
    max_nb: int,
    device: torch.device = DEVICE,
) -> nn.Module:
    return PerUserBlocklengthMlp(
        nr,
        nb,
        dk,
        user_count=k_count,
        max_receive_antennas=max_nr,
        max_transmit_antennas=max_nb,
    ).to(device)


def build_shared_bs_precoder_net(
    *, k_count: int, max_nr: int, max_nb: int, max_dk: int, device: torch.device = DEVICE
) -> nn.Module:
    return SharedBsChannelMlp(
        user_count=k_count,
        max_receive_antennas=max_nr,
        transmit_antennas=max_nb,
        max_streams=max_dk,
    ).to(device)


def build_shared_bs_precoder_net_with_blocklength(
    *, k_count: int, max_nr: int, max_nb: int, max_dk: int, device: torch.device = DEVICE
) -> nn.Module:
    return SharedBsBlocklengthMlp(
        user_count=k_count,
        max_receive_antennas=max_nr,
        transmit_antennas=max_nb,
        max_streams=max_dk,
    ).to(device)
