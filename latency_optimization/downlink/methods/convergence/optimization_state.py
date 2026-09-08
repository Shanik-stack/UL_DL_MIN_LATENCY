"""Checkpoint and restore state for one downlink block solve."""

from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np
import torch

from ...simulation.block_state import clone_precoders


def capture_active_block_solver_state(
    precoders: list[list[np.ndarray]],
    user_models: Sequence[torch.nn.Module],
    model_optimizers: Sequence[torch.optim.Optimizer],
    active_users: Sequence[int],
) -> dict[str, Any]:
    """Copy only the state that can change while solving one block."""
    active_ids = [int(user) for user in active_users]
    if len(user_models) == 0 or len(model_optimizers) == 0:
        model_states: dict[int, dict[str, torch.Tensor]] = {}
        optimizer_states: dict[int, dict[str, Any]] = {}
    else:
        model_states = {
            user: {key: value.detach().cpu().clone() for key, value in user_models[user].state_dict().items()}
            for user in active_ids
        }
        optimizer_states = {
            user: copy.deepcopy(model_optimizers[user].state_dict())
            for user in active_ids
        }
    return {
        "precoders": clone_precoders(precoders),
        "model_states": model_states,
        "optimizer_states": optimizer_states,
    }


def restore_active_block_solver_state(
    precoders: list[list[np.ndarray]],
    user_models: Sequence[torch.nn.Module],
    model_optimizers: Sequence[torch.optim.Optimizer],
    active_users: Sequence[int],
    state: dict[str, Any],
) -> None:
    """Restore a checkpoint after a reduced-blocklength candidate fails."""
    precoders[:] = clone_precoders(state["precoders"])
    if len(user_models) == 0 or len(model_optimizers) == 0:
        return
    for user in active_users:
        user_id = int(user)
        user_models[user_id].load_state_dict(state["model_states"][user_id])
        model_optimizers[user_id].load_state_dict(state["optimizer_states"][user_id])


__all__ = ["capture_active_block_solver_state", "restore_active_block_solver_state"]
