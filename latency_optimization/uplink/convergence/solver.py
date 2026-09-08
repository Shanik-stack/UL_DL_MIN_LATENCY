"""Objective-only uplink precoder optimization."""

from __future__ import annotations

import copy
from typing import Mapping

import torch
from torch import nn

from latency_optimization.core.validation import require_choice
from latency_optimization.optimization.stopping import KktResiduals, convergence_status_from_config
from latency_optimization.precoders.parameters import complex_tensor_from_parameter
from latency_optimization.precoders.power import cap_matrix_power_torch
from latency_optimization.results.console import format_progress_log_line
from latency_optimization.runtime import DEVICE

from ..objective import UplinkPrecoderObjective
from ..precoders.inference import infer_precoder


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
    loss_fn: UplinkPrecoderObjective,
    Nt: int,
    dk: int,
    max_epochs: int,
    optimizer: torch.optim.Optimizer,
    stopping_config: Mapping[str, object],
    print_every_epoch: int = 1,
    verbose: bool = True,
    log_context: dict[str, object] | None = None,
    *,
    precoder_param: torch.nn.Parameter | None = None,
    update_mode: str = "precoder_net",
) -> dict[str, object]:
    """Solve one user's beamforming problem for a fixed channel and ``n_kl``.

    What: optimize either the complex precoder entries directly or the weights of
    that user's precoder network. Each epoch builds the current beam, evaluates the
    uplink FBL objective and its rate/power constraints, performs one gradient
    update, projects the beam to ``||F_kl||_F^2 <= P_k``, and then measures the
    updated state.

    Why: allocation decides which blocklengths to test, while this function answers
    the continuous subproblem: whether a beam can support the requested bits at one
    chosen ``n_kl``. Keeping that numerical solve here gives payload and streaming
    allocation identical convergence and checkpoint behavior.

    Returns: the selected feasible (or best available) precoder, its FBL rate and
    power, the loss history, KKT residual history, and the reason optimization
    stopped. The caller uses those values to accept or reject the tested ``n_kl``.
    """
    update_mode = validate_convergence_precoder_update_mode(update_mode)
    if update_mode == "precoder_net" and precoder_net is None:
        raise ValueError("precoder_net is required when update_mode='precoder_net'.")
    if update_mode == "direct_precoder" and precoder_param is None:
        raise ValueError("precoder_param is required when update_mode='direct_precoder'.")

    max_epochs = max(1, int(max_epochs))
    print_every_epoch = max(1, int(print_every_epoch))
    loss_curve: list[float] = []
    convergence_history: list[dict[str, float]] = []
    best_loss = float("inf")
    solve_status = "max_epochs_reached"

    def build_precoder() -> torch.Tensor:
        if update_mode == "direct_precoder":
            return _project_precoder_power(
                complex_tensor_from_parameter(precoder_param),
                loss_fn.P,
            )
        return infer_precoder(
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
        optimizer.zero_grad()
        training_precoder = build_precoder()
        precoder_before_update = training_precoder.detach().clone()
        training_objective = loss_fn(training_precoder)
        training_objective["loss"].backward()
        optimizer.step()

        # Re-evaluate the updated state for stopping and checkpoint selection.
        with torch.no_grad():
            precoder = build_precoder()
            objective = loss_fn(precoder)
        loss = objective["loss"]
        rate = objective["rate"]
        power = objective["power"]
        rate_gap = objective["rate_gap"]
        power_gap = objective["power_gap"]
        rate_violation = objective["rate_violation"]
        power_violation = objective["power_violation"]
        primal_residual = max(
            float(rate_violation.detach().cpu()),
            float(power_violation.detach().cpu()),
        )
        numerator = float(
            torch.linalg.norm(precoder.detach() - precoder_before_update, ord="fro").cpu()
        )
        denominator = max(
            float(torch.linalg.norm(precoder_before_update, ord="fro").cpu()),
            1e-12,
        )
        precoder_change = numerator / denominator

        objective_value = float(loss.detach().cpu())
        loss_curve.append(objective_value)
        convergence_history.append(
            {
                "epoch": float(epoch_index + 1),
                "primal_residual": primal_residual,
                "complementarity_residual": 0.0,
                "precoder_change": precoder_change,
                "stationarity_residual": precoder_change,
                "rate_gap": float(rate_gap.detach().cpu()),
                "power_gap": float(power_gap.detach().cpu()),
                "rate_violation": float(rate_violation.detach().cpu()),
                "power_violation": float(power_violation.detach().cpu()),
                "rate": float(rate.detach().cpu()),
                "power": float(power.detach().cpu()),
            }
        )
        epoch_status = convergence_status_from_config(
            stopping_config,
            precoder_change=precoder_change,
            has_previous_state=True,
            residuals=KktResiduals(primal_residual, 0.0, precoder_change),
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
        if epoch_status != "running":
            solve_status = epoch_status
            break

    restore_state(best_state)
    if solve_status == "max_epochs_reached":
        solve_status = "max_epochs_best_objective"

    with torch.no_grad():
        final_precoder = build_precoder()
        objective = loss_fn(final_precoder)
        rate = objective["rate"]
        reward = objective["reward"]
        power = objective["power"]
        rate_gap = objective["rate_gap"]
        power_gap = objective["power_gap"]
        rate_violation = objective["rate_violation"]
        power_violation = objective["power_violation"]

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
        "final_complementarity_residual": 0.0,
        "final_stationarity_residual": float(
            convergence_history[-1]["stationarity_residual"] if convergence_history else float("inf")
        ),
    }
