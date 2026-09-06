"""Framework-neutral checkpoint operations for precoder networks."""

from __future__ import annotations

from collections.abc import Sequence

import torch


ModelState = dict[str, torch.Tensor]


def clone_model_state(model: torch.nn.Module) -> ModelState:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def clone_model_states(models: Sequence[torch.nn.Module]) -> list[ModelState]:
    return [clone_model_state(model) for model in models]


def restore_model_states(models: Sequence[torch.nn.Module], states: Sequence[ModelState]) -> None:
    for model, state in zip(models, states):
        model.load_state_dict(state)


def relative_model_state_change(model: torch.nn.Module, previous_state: ModelState | None) -> float:
    if previous_state is None:
        return float("inf")
    return relative_models_state_change([model], [previous_state])


def relative_models_state_change(
    models: Sequence[torch.nn.Module],
    previous_states: Sequence[ModelState] | None,
) -> float:
    if previous_states is None:
        return float("inf")
    delta_squared = 0.0
    reference_squared = 0.0
    for model, previous_state in zip(models, previous_states):
        for name, current_value in model.state_dict().items():
            current = current_value.detach().cpu()
            previous = previous_state[name]
            delta_squared += float(torch.sum((current - previous).pow(2)).item())
            reference_squared += float(torch.sum(previous.pow(2)).item())
    return float(delta_squared**0.5 / max(reference_squared**0.5, 1.0e-12))


__all__ = [
    "ModelState",
    "clone_model_state",
    "clone_model_states",
    "relative_model_state_change",
    "relative_models_state_change",
    "restore_model_states",
]
