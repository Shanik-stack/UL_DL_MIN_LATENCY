"""The supported blocklength-aware uplink MLP."""

from __future__ import annotations

import torch
from torch import nn

from latency_optimization.runtime import DEVICE


UPLINK_MODEL_INPUT_MODE = "channel_sigma_epsilon_n"


class UplinkPrecoderMlp(nn.Module):
    """Map ``(H_kl, n_kl, sigma_k^2, epsilon_k)`` to user k's precoder."""

    def __init__(self, receive_antennas: int, transmit_antennas: int, streams: int):
        super().__init__()
        self.receive_antennas = int(receive_antennas)
        self.transmit_antennas = int(transmit_antennas)
        self.streams = int(streams)
        self.uses_blocklength_input = True
        self.input_mode = UPLINK_MODEL_INPUT_MODE

        channel_features = 2 * self.receive_antennas * self.transmit_antennas
        output_features = 2 * self.transmit_antennas * self.streams
        widths = (
            max(256, 8 * output_features),
            max(128, 4 * output_features),
            max(64, 2 * output_features),
        )
        self.network = nn.Sequential(
            nn.Linear(channel_features + 3, widths[0]),
            nn.ReLU(),
            nn.Linear(widths[0], widths[1]),
            nn.ReLU(),
            nn.Linear(widths[1], widths[2]),
            nn.ReLU(),
            nn.Linear(widths[2], output_features),
        )

    def forward(
        self,
        channel: torch.Tensor,
        blocklength: int | float | torch.Tensor,
        noise_variance: float | torch.Tensor,
        error_probability: float | torch.Tensor,
    ) -> torch.Tensor:
        flattened = channel.to(torch.complex64).reshape(1, -1)
        channel_features = torch.cat([flattened.real, flattened.imag], dim=1)
        metadata = torch.stack(
            [
                torch.log1p(torch.clamp(torch.as_tensor(blocklength, device=channel.device), min=1.0)),
                torch.log(torch.clamp(torch.as_tensor(noise_variance, device=channel.device), min=1e-12)),
                torch.as_tensor(error_probability, device=channel.device),
            ]
        ).to(dtype=channel_features.dtype).reshape(1, 3)
        return self.network(torch.cat([channel_features, metadata], dim=1))


def build_user_precoder_net(
    receive_antennas: int,
    transmit_antennas: int,
    streams: int,
    *,
    device: torch.device = DEVICE,
) -> UplinkPrecoderMlp:
    return UplinkPrecoderMlp(receive_antennas, transmit_antennas, streams).to(device)


__all__ = ["UPLINK_MODEL_INPUT_MODE", "UplinkPrecoderMlp", "build_user_precoder_net"]
