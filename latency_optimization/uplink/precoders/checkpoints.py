"""Strict checkpoint persistence for the uplink MLP."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from latency_optimization.runtime import DEVICE

from .models import UPLINK_MODEL_INPUT_MODE, build_user_precoder_net


def export_user_model_specs(
    receive_antennas: Sequence[int],
    transmit_antennas: Sequence[int],
    streams: Sequence[int],
) -> list[dict[str, int | str]]:
    return [
        {
            "Nr": int(receive_antennas[user]),
            "Nt": int(transmit_antennas[user]),
            "dk": int(streams[user]),
            "input_mode": UPLINK_MODEL_INPUT_MODE,
        }
        for user in range(len(receive_antennas))
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
        raise ValueError("Uplink model specification and state counts must match.")
    models = []
    for spec, state in zip(model_specs, model_states):
        if spec["input_mode"] != UPLINK_MODEL_INPUT_MODE:
            raise ValueError(
                f"Unsupported uplink model input_mode {spec['input_mode']!r}; "
                f"expected {UPLINK_MODEL_INPUT_MODE!r}."
            )
        model = build_user_precoder_net(
            int(spec["Nr"]), int(spec["Nt"]), int(spec["dk"]), device=device
        )
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


__all__ = ["export_user_model_specs", "export_user_model_states", "load_user_precoder_models"]
