"""Construction and inference service for downlink precoder models."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch

from latency_optimization.runtime import DEVICE

from .block_state import channels_for_block
from .precoders.models import (
    build_shared_bs_precoder_net,
    build_user_precoder_net,
    model_outputs_full_bs_precoder,
    validate_downlink_precoder_net_scope,
)
from .precoders.inference import (
    infer_raw_bs_precoders_numpy,
    infer_raw_bs_precoders_torch,
    infer_raw_precoder_numpy,
)
from .system import DownlinkSystem


def describe_precoder_parameterization(
    model_scope: str,
    update_mode: str,
    *,
    uses_blocklength_input: bool = False,
) -> str:
    if update_mode == "direct_precoder":
        return "direct_active_block_precoders"
    if validate_downlink_precoder_net_scope(model_scope) == "bs_shared_net":
        return "bs_shared_block_context_to_full_precoder_mlp"
    return (
        "per_user_block_context_to_precoder_mlp"
        if uses_blocklength_input
        else "per_user_channel_to_precoder_mlp"
    )


def initial_baseline_model_scope() -> str:
    return "per_user_nets"


def build_precoder_models(
    system: DownlinkSystem,
    *,
    initialization_seed: int | None = None,
    model_scope: str,
) -> list[torch.nn.Module]:
    resolved_scope = validate_downlink_precoder_net_scope(model_scope)

    def construct() -> list[torch.nn.Module]:
        if resolved_scope == "bs_shared_net":
            shared_model = build_shared_bs_precoder_net(
                k_count=system.K,
                max_nr=int(np.max(system.Nr)),
                max_nb=int(np.max(system.Nb)),
                max_dk=int(np.max(system.dk)),
                device=DEVICE,
            )
            return [shared_model for _ in range(system.K)]
        return [
            build_user_precoder_net(
                nr=int(system.Nr[user]),
                nb=int(system.Nb[user]),
                dk=int(system.dk[user]),
                device=DEVICE,
            )
            for user in range(system.K)
        ]

    if initialization_seed is None:
        return construct()

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(int(initialization_seed))
        np.random.seed(int(initialization_seed) % (2**32 - 1))
        torch.manual_seed(int(initialization_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(initialization_seed))
        return construct()
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def build_model_optimizers(
    models: Sequence[torch.nn.Module],
    *,
    learning_rate: float,
) -> list[torch.optim.Optimizer]:
    by_model_identity: dict[int, torch.optim.Optimizer] = {}
    optimizers: list[torch.optim.Optimizer] = []
    for model in models:
        identity = id(model)
        if identity not in by_model_identity:
            by_model_identity[identity] = torch.optim.Adam(
                model.parameters(),
                lr=float(learning_rate),
            )
        optimizers.append(by_model_identity[identity])
    return optimizers


def models_output_full_bs_precoder(models: Sequence[torch.nn.Module]) -> bool:
    return bool(models) and model_outputs_full_bs_precoder(models[0])


def active_mask_for_users(system: DownlinkSystem, active_users: Sequence[int]) -> list[int]:
    active = {int(user) for user in active_users}
    return [int(user in active) for user in range(system.K)]


def infer_shared_block_precoders_numpy(
    system: DownlinkSystem,
    shared_model: torch.nn.Module,
    block: int,
    active_users: Sequence[int],
) -> dict[int, np.ndarray]:
    beams = infer_raw_bs_precoders_numpy(
        shared_model,
        channels_for_block(system, block),
        active_mask_for_users(system, active_users),
        system.Nb,
        system.dk,
        device=DEVICE,
    )
    return {int(user): np.asarray(beams[int(user)], dtype=np.complex128) for user in active_users}


def infer_shared_block_precoders_torch(
    system: DownlinkSystem,
    shared_model: torch.nn.Module,
    block: int,
    active_users: Sequence[int],
) -> dict[int, torch.Tensor]:
    channel_tensors = [
        torch.as_tensor(channel, dtype=torch.complex64, device=DEVICE)
        for channel in channels_for_block(system, block)
    ]
    active_mask = torch.as_tensor(
        active_mask_for_users(system, active_users),
        dtype=torch.float32,
        device=DEVICE,
    )
    beams = infer_raw_bs_precoders_torch(
        shared_model,
        channel_tensors,
        active_mask,
        system.Nb,
        system.dk,
    )
    return {int(user): beams[int(user)] for user in active_users}


def build_precoder_snapshot(
    system: DownlinkSystem,
    models: Sequence[torch.nn.Module],
) -> list[list[np.ndarray]]:
    if models_output_full_bs_precoder(models):
        snapshot: list[list[np.ndarray]] = [[] for _ in range(system.K)]
        max_blocks = max((len(user_blocks) for user_blocks in system.H), default=0)
        for block in range(max_blocks):
            block_precoders = infer_shared_block_precoders_numpy(
                system,
                models[0],
                block,
                list(range(system.K)),
            )
            for user in range(system.K):
                snapshot[user].append(block_precoders[user])
            system.project_block_precoders_to_power(snapshot, block)
        return snapshot

    snapshot = [
        [
            infer_raw_precoder_numpy(
                models[user],
                np.asarray(system.H[user][block], dtype=np.complex64),
                nb=int(system.Nb[user]),
                dk=int(system.dk[user]),
                device=DEVICE,
                user_index=user,
            )
            for block in range(len(system.H[user]))
        ]
        for user in range(system.K)
    ]
    for block in range(max((len(user_blocks) for user_blocks in snapshot), default=0)):
        system.project_block_precoders_to_power(snapshot, block)
    return snapshot


def refresh_block_precoders(
    system: DownlinkSystem,
    precoders: list[list[np.ndarray]],
    models: Sequence[torch.nn.Module],
    active_users: Sequence[int],
    block: int,
) -> None:
    block = int(block)
    if models_output_full_bs_precoder(models):
        block_precoders = infer_shared_block_precoders_numpy(
            system,
            models[0],
            block,
            active_users,
        )
        for user in active_users:
            user = int(user)
            if block >= len(precoders[user]):
                raise ValueError(f"User {user} has no precoder slot for block {block}.")
            precoders[user][block] = block_precoders[user]
    else:
        for user in active_users:
            user = int(user)
            if block >= len(precoders[user]):
                raise ValueError(f"User {user} has no precoder slot for block {block}.")
            precoders[user][block] = infer_raw_precoder_numpy(
                models[user],
                np.asarray(system.H[user][block], dtype=np.complex64),
                nb=int(system.Nb[user]),
                dk=int(system.dk[user]),
                device=DEVICE,
                user_index=user,
            )
    system.project_block_precoders_to_power(
        precoders,
        block,
        active_users=[int(user) for user in active_users],
    )


__all__ = [
    "active_mask_for_users",
    "build_model_optimizers",
    "build_precoder_models",
    "build_precoder_snapshot",
    "describe_precoder_parameterization",
    "infer_shared_block_precoders_numpy",
    "infer_shared_block_precoders_torch",
    "initial_baseline_model_scope",
    "models_output_full_bs_precoder",
    "refresh_block_precoders",
]
