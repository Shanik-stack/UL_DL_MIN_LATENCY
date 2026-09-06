"""Objective-only uplink precoder optimization."""

from __future__ import annotations

import copy

import torch
from torch import nn

from latency_optimization.core.validation import require_choice
from latency_optimization.optimization.stopping import objective_convergence_status
from latency_optimization.precoders.parameters import complex_tensor_from_parameter
from latency_optimization.precoders.power import cap_matrix_power_torch
from latency_optimization.results.console import format_progress_log_line
from latency_optimization.runtime import DEVICE

from ..objective import FiniteBlocklengthRateObjective
from ..precoder_models import infer_precoder_torch_with_blocklength_and_sigma


CONVERGENCE_PRECODER_UPDATE_MODES = {"precoder_net", "direct_precoder"}


def validate_convergence_precoder_update_mode(raw_mode: str) -> str:
    return require_choice(
        raw_mode,
        CONVERGENCE_PRECODER_UPDATE_MODES,
        "convergence_precoder_update_mode",
    )


def _project_precoder_power(precoder: torch.Tensor, power_limit: float) -> torch.Tensor:
    return cap_matrix_power_torch(precoder, power_limit)


def optimize_precoder_for_nl(
    precoder_net: nn.Module | None,
    loss_fn: FiniteBlocklengthRateObjective,
    Nt: int,
    dk: int,
    max_epochs: int,
    optimizer: torch.optim.Optimizer,
    precoder_change_tolerance: float = 1e-5,
    print_every_epoch: int = 1,
    verbose: bool = True,
    log_context: dict[str, object] | None = None,
    *,
    precoder_param: torch.nn.Parameter | None = None,
    update_mode: str = "precoder_net",
) -> dict[str, object]:
    """Maximize FBL rate for one user, channel block, and blocklength."""
    update_mode = validate_convergence_precoder_update_mode(update_mode)
    if update_mode == "precoder_net" and precoder_net is None:
        raise ValueError("precoder_net is required when update_mode='precoder_net'.")
    if update_mode == "direct_precoder" and precoder_param is None:
        raise ValueError("precoder_param is required when update_mode='direct_precoder'.")

    max_epochs = max(1, int(max_epochs))
    print_every_epoch = max(1, int(print_every_epoch))
    loss_curve: list[float] = []
    convergence_history: list[dict[str, float]] = []
    previous_precoder: torch.Tensor | None = None
    best_loss = float("inf")
    solve_status = "max_epochs_reached"

    def build_precoder() -> torch.Tensor:
        if update_mode == "direct_precoder":
            return _project_precoder_power(
                complex_tensor_from_parameter(precoder_param),
                loss_fn.P,
            )
        return infer_precoder_torch_with_blocklength_and_sigma(
            precoder_net,
            loss_fn.H_kl,
            int(loss_fn.n_kl),
            loss_fn.sigma2,
            loss_fn.epsilon,
            Nt,
            dk,
            loss_fn.P,
        )

    def capture_state() -> dict[str, object]:
        model_state = (
            precoder_param.detach().cpu().clone()
            if update_mode == "direct_precoder"
            else {
                key: value.detach().cpu().clone()
                for key, value in precoder_net.state_dict().items()
            }
        )
        return {
            "model": model_state,
            "optimizer": copy.deepcopy(optimizer.state_dict()),
        }

    def restore_state(state: dict[str, object]) -> None:
        if update_mode == "direct_precoder":
            with torch.no_grad():
                precoder_param.copy_(state["model"].to(DEVICE, dtype=precoder_param.dtype))
        else:
            precoder_net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])

    best_state = capture_state()
    for epoch_index in range(max_epochs):
        precoder = build_precoder()
        optimizer.zero_grad()
        loss, rate, reward, power, rate_gap, power_gap, rate_violation, power_violation = loss_fn(
            precoder
        )
        loss.backward()

        primal_residual = max(
            float(rate_violation.detach().cpu()),
            float(power_violation.detach().cpu()),
        )
        if previous_precoder is None:
            precoder_change = float("inf")
        else:
            numerator = float(
                torch.linalg.norm(precoder.detach() - previous_precoder, ord="fro").cpu()
            )
            denominator = max(
                float(torch.linalg.norm(previous_precoder, ord="fro").cpu()),
                1e-12,
            )
            precoder_change = numerator / denominator

        objective_value = float(loss.detach().cpu())
        loss_curve.append(objective_value)
        convergence_history.append(
            {
                "epoch": float(epoch_index + 1),
                "primal_residual": primal_residual,
                "precoder_change": precoder_change,
                "rate_gap": float(rate_gap.detach().cpu()),
                "power_gap": float(power_gap.detach().cpu()),
                "rate_violation": float(rate_violation.detach().cpu()),
                "power_violation": float(power_violation.detach().cpu()),
                "rate": float(rate.detach().cpu()),
                "power": float(power.detach().cpu()),
            }
        )
        previous_precoder = precoder.detach().clone()

        epoch_status = objective_convergence_status(
            precoder_change,
            precoder_change_tolerance,
            has_previous_state=epoch_index > 0,
        )
        if verbose and (
            (epoch_index + 1) % print_every_epoch == 0
            or epoch_index == 0
            or epoch_status != "running"
        ):
            print(
                format_progress_log_line(
                    "[UL Convergence]",
                    phase="optimize",
                    epoch=f"{epoch_index + 1}/{max_epochs}",
                    objective=objective_value,
                    rate=float(rate.detach().cpu()),
                    power=float(power.detach().cpu()),
                    rate_or_power_violation=primal_residual,
                    precoder_change=precoder_change,
                    status=epoch_status,
                    **dict(log_context or {}),
                )
            )

        if objective_value < best_loss:
            best_loss = objective_value
            best_state = capture_state()
        if epoch_status == "objective_stationary":
            solve_status = epoch_status
            break
        if epoch_index + 1 < max_epochs:
            optimizer.step()

    restore_state(best_state)
    if solve_status == "max_epochs_reached":
        solve_status = "max_epochs_best_objective"

    with torch.no_grad():
        final_precoder = build_precoder()
        loss, rate, reward, power, rate_gap, power_gap, rate_violation, power_violation = loss_fn(
            final_precoder
        )

    return {
        "F": final_precoder.detach(),
        "R_fbl": float(rate.detach().item()),
        "beam_reward": float(reward.detach().item()),
        "F_power": float(power.detach().item()),
        "rate_gap": float(rate_gap.detach().item()),
        "power_gap": float(power_gap.detach().item()),
        "rate_violation": float(rate_violation.detach().item()),
        "power_violation": float(power_violation.detach().item()),
        "loss_curve": loss_curve,
        "solve_status": solve_status,
        "kkt_history": convergence_history,
        "final_primal_residual": max(
            float(rate_violation.detach().cpu()),
            float(power_violation.detach().cpu()),
        ),
    }
