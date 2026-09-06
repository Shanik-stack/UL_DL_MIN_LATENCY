"""Strict serialization for the supported downlink MLPs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from latency_optimization.core.validation import require_choice
from latency_optimization.runtime import DEVICE

from .models import (
    BLOCKLENGTH_AWARE_INPUT,
    CHANNEL_ONLY_INPUT,
    DOWNLINK_MODEL_INPUT_MODES,
    build_shared_bs_precoder_net,
    build_shared_bs_precoder_net_with_blocklength,
    build_user_precoder_net,
    build_user_precoder_net_with_blocklength,
    validate_downlink_precoder_net_scope,
)


def export_user_model_specs(
    nr: Sequence[int],
    nb: Sequence[int],
    dk: Sequence[int],
    *,
    uses_blocklength_input: bool = False,
    context_k: int | None = None,
    context_max_nr: int | None = None,
    context_max_nb: int | None = None,
    model_scope: str,
    context_max_dk: int | None = None,
) -> list[dict[str, int | bool | str]]:
    scope = validate_downlink_precoder_net_scope(model_scope)
    input_mode = BLOCKLENGTH_AWARE_INPUT if uses_blocklength_input else CHANNEL_ONLY_INPUT
    return [
        {
            "nr": int(nr[user]),
            "nb": int(nb[user]),
            "dk": int(dk[user]),
            "input_mode": input_mode,
            "context_k": int(context_k if context_k is not None else len(nr)),
            "context_max_nr": int(context_max_nr if context_max_nr is not None else max(nr)),
            "context_max_nb": int(context_max_nb if context_max_nb is not None else max(nb)),
            "context_max_dk": int(context_max_dk if context_max_dk is not None else max(dk)),
            "model_scope": scope,
        }
        for user in range(len(nr))
    ]


def export_user_model_states(models: Sequence[nn.Module]) -> list[dict[str, Any]]:
    return [{key: value.detach().cpu() for key, value in model.state_dict().items()} for model in models]


def _build_model_from_spec(spec: dict[str, Any], device: torch.device) -> nn.Module:
    scope = validate_downlink_precoder_net_scope(str(spec["model_scope"]))
    input_mode = require_choice(str(spec["input_mode"]), DOWNLINK_MODEL_INPUT_MODES, "downlink model input_mode")
    common = {
        "k_count": int(spec["context_k"]),
        "max_nr": int(spec["context_max_nr"]),
        "max_nb": int(spec["context_max_nb"]),
        "device": device,
    }
    if scope == "bs_shared_net":
        builder = (
            build_shared_bs_precoder_net_with_blocklength
            if input_mode == BLOCKLENGTH_AWARE_INPUT
            else build_shared_bs_precoder_net
        )
        return builder(max_dk=int(spec["context_max_dk"]), **common)
    if input_mode == BLOCKLENGTH_AWARE_INPUT:
        return build_user_precoder_net_with_blocklength(
            int(spec["nr"]),
            int(spec["nb"]),
            int(spec["dk"]),
            **common,
        )
    return build_user_precoder_net(
        int(spec["nr"]),
        int(spec["nb"]),
        int(spec["dk"]),
        device=device,
    )


def load_user_precoder_models(
    model_specs: Sequence[dict[str, Any]],
    model_states: Sequence[dict[str, Any]],
    *,
    device: torch.device = DEVICE,
) -> list[nn.Module]:
    if len(model_specs) != len(model_states):
        raise ValueError("Model specification and state counts must match.")
    if not model_specs:
        return []

    scope = validate_downlink_precoder_net_scope(str(model_specs[0]["model_scope"]))
    if scope == "bs_shared_net":
        shared_model = _build_model_from_spec(model_specs[0], device)
        shared_model.load_state_dict(model_states[0])
        shared_model.eval()
        return [shared_model for _ in model_specs]

    models = []
    for spec, state in zip(model_specs, model_states):
        model = _build_model_from_spec(spec, device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models
