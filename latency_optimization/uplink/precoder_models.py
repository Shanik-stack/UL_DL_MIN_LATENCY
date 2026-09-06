"""The n-aware uplink MLP and its checkpoint/inference operations."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from latency_optimization.precoders.power import normalize_matrix_power_torch
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
        hidden_1 = max(256, 8 * output_features)
        hidden_2 = max(128, 4 * output_features)
        hidden_3 = max(64, 2 * output_features)
        self.network = nn.Sequential(
            nn.Linear(channel_features + 3, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, hidden_3),
            nn.ReLU(),
            nn.Linear(hidden_3, output_features),
        )

    def forward(
        self,
        channel: torch.Tensor,
        blocklength: int | float,
        noise_variance: float,
        error_probability: float,
    ) -> torch.Tensor:
        flattened_channel = channel.reshape(1, -1)
        channel_features = torch.cat(
            [flattened_channel.real, flattened_channel.imag],
            dim=1,
        )
        metadata = channel_features.new_tensor(
            [[
                np.log1p(max(float(blocklength), 1.0)),
                np.log(max(float(noise_variance), 1e-12)),
                float(error_probability),
            ]]
        )
        return self.network(torch.cat([channel_features, metadata], dim=1))


def _network_output_to_precoder(
    network_output: torch.Tensor,
    transmit_antennas: int,
    streams: int,
) -> torch.Tensor:
    if network_output.dim() == 2:
        network_output = network_output.squeeze(0)
    real_imaginary = network_output.view(2, int(transmit_antennas), int(streams))
    return (real_imaginary[0] + 1j * real_imaginary[1]).to(torch.complex64)


def build_user_precoder_net_with_blocklength_and_sigma(
    Nr: int,
    Nt: int,
    dk: int,
    *,
    device: torch.device = DEVICE,
) -> UplinkPrecoderMlp:
    return UplinkPrecoderMlp(Nr, Nt, dk).to(device)


def infer_precoder_torch_with_blocklength_and_sigma(
    model: nn.Module,
    H_kl: torch.Tensor,
    n_kl: int,
    sigma2: float,
    epsilon: float,
    Nt: int,
    dk: int,
    P: float,
) -> torch.Tensor:
    network_output = model(H_kl, int(n_kl), float(sigma2), float(epsilon))
    precoder = _network_output_to_precoder(network_output, Nt, dk)
    return normalize_matrix_power_torch(precoder, P)


def infer_precoder_numpy_with_blocklength_and_sigma(
    model: nn.Module,
    H_kl: np.ndarray,
    n_kl: int,
    sigma2: float,
    epsilon: float,
    Nt: int,
    dk: int,
    P: float,
    *,
    device: torch.device = DEVICE,
) -> np.ndarray:
    with torch.no_grad():
        channel = torch.as_tensor(H_kl, dtype=torch.complex64, device=device)
        precoder = infer_precoder_torch_with_blocklength_and_sigma(
            model,
            channel,
            n_kl,
            sigma2,
            epsilon,
            Nt,
            dk,
            P,
        )
    return precoder.detach().cpu().numpy().astype(np.complex128)


def export_user_model_specs(
    NR: Sequence[int],
    NT: Sequence[int],
    dk: Sequence[int],
    *,
    uses_blocklength_input: bool = True,
    input_mode: str = UPLINK_MODEL_INPUT_MODE,
) -> list[dict[str, int | bool | str]]:
    if not uses_blocklength_input or input_mode != UPLINK_MODEL_INPUT_MODE:
        raise ValueError(
            "Uplink supports only the n-aware channel_sigma_epsilon_n precoder model."
        )
    return [
        {
            "Nr": int(NR[user]),
            "Nt": int(NT[user]),
            "dk": int(dk[user]),
            "uses_blocklength_input": True,
            "input_mode": UPLINK_MODEL_INPUT_MODE,
        }
        for user in range(len(NR))
    ]


def export_user_model_states(models: Sequence[nn.Module]) -> list[dict[str, Any]]:
    return [
        {name: value.detach().cpu() for name, value in model.state_dict().items()}
        for model in models
    ]


def load_user_precoder_models(
    model_specs: Sequence[dict[str, Any]],
    model_states: Sequence[dict[str, Any]],
    *,
    device: torch.device = DEVICE,
) -> list[nn.Module]:
    if len(model_specs) != len(model_states):
        raise ValueError("Uplink model specifications and states must have equal lengths.")
    models: list[nn.Module] = []
    for spec, state in zip(model_specs, model_states):
        if spec["input_mode"] != UPLINK_MODEL_INPUT_MODE:
            raise ValueError(
                f"Unsupported uplink model input_mode {spec['input_mode']!r}; "
                f"expected {UPLINK_MODEL_INPUT_MODE!r}."
            )
        model = build_user_precoder_net_with_blocklength_and_sigma(
            int(spec["Nr"]),
            int(spec["Nt"]),
            int(spec["dk"]),
            device=device,
        )
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


__all__ = [
    "UPLINK_MODEL_INPUT_MODE",
    "UplinkPrecoderMlp",
    "build_user_precoder_net_with_blocklength_and_sigma",
    "export_user_model_specs",
    "export_user_model_states",
    "infer_precoder_numpy_with_blocklength_and_sigma",
    "infer_precoder_torch_with_blocklength_and_sigma",
    "load_user_precoder_models",
]
