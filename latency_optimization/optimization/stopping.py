"""Shared convergence checks for direct and neural precoder solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


OBJECTIVE_STATIONARITY = "objective_stationarity"
KKT_RESIDUALS = "kkt_residuals"
CONVERGENCE_STOPPING_RULES = {OBJECTIVE_STATIONARITY, KKT_RESIDUALS}


@dataclass(frozen=True)
class KktResiduals:
    """Primal, complementarity, and precoder-stationarity diagnostics."""

    primal: float
    complementarity: float
    stationarity: float


@dataclass(frozen=True)
class KktTolerances:
    """Configured thresholds for :class:`KktResiduals`."""

    primal: float
    complementarity: float
    stationarity: float


def objective_convergence_status(
    precoder_change: float,
    tolerance: float,
    *,
    has_previous_state: bool,
) -> str:
    """Stop when the relative precoder change falls below its tolerance."""
    if float(tolerance) < 0.0:
        raise ValueError("precoder_change_tolerance must be non-negative.")
    if has_previous_state and float(precoder_change) <= float(tolerance):
        return "objective_stationary"
    return "running"


def classify_convergence(
    residuals: KktResiduals,
    tolerances: KktTolerances,
    *,
    has_previous_state: bool,
    constraints_enabled: bool,
) -> str:
    """Classify a residual-based solve without assuming a dual implementation.

    In the retained objective-only solvers, complementarity is zero because no
    dual variables are optimized.  Streaming still needs the primal check: it
    prevents a nearly unchanged but infeasible precoder from being accepted.
    """
    for name, value in vars(tolerances).items():
        if float(value) < 0.0:
            raise ValueError(f"KKT {name} tolerance must be non-negative.")
    if not has_previous_state:
        return "running"
    if float(residuals.stationarity) > float(tolerances.stationarity):
        return "running"
    if constraints_enabled and (
        float(residuals.primal) > float(tolerances.primal)
        or float(residuals.complementarity) > float(tolerances.complementarity)
    ):
        return "stationary_infeasible"
    return "kkt_converged"


def kkt_tolerances_from_config(config: Mapping[str, object]) -> KktTolerances:
    """Build strict KKT thresholds from one normalized simulation config."""
    return KktTolerances(
        float(config["kkt_primal_tolerance"]),
        float(config["kkt_complementarity_tolerance"]),
        float(config["kkt_stationarity_tolerance"]),
    )


def convergence_status_from_config(
    config: Mapping[str, object],
    *,
    precoder_change: float,
    has_previous_state: bool,
    residuals: KktResiduals,
) -> str:
    """Apply the one configured stopping rule used by a method."""
    return convergence_status(
        str(config["convergence_stopping_rule"]),
        precoder_change=precoder_change,
        precoder_change_tolerance=float(config["precoder_change_tolerance"]),
        has_previous_state=has_previous_state,
        residuals=residuals,
        kkt_tolerances=kkt_tolerances_from_config(config),
    )


def convergence_status(
    stopping_rule: str,
    *,
    precoder_change: float,
    precoder_change_tolerance: float,
    has_previous_state: bool,
    residuals: KktResiduals | None = None,
    kkt_tolerances: KktTolerances | None = None,
) -> str:
    """Apply the configured stopping rule with strict required inputs."""
    if stopping_rule not in CONVERGENCE_STOPPING_RULES:
        choices = ", ".join(sorted(CONVERGENCE_STOPPING_RULES))
        raise ValueError(f"Unsupported convergence_stopping_rule {stopping_rule!r}; use one of {choices}.")
    if stopping_rule == OBJECTIVE_STATIONARITY:
        return objective_convergence_status(
            precoder_change,
            precoder_change_tolerance,
            has_previous_state=has_previous_state,
        )
    if residuals is None or kkt_tolerances is None:
        raise ValueError("kkt_residuals stopping requires residuals and kkt_tolerances.")
    return classify_convergence(
        residuals,
        kkt_tolerances,
        has_previous_state=has_previous_state,
        constraints_enabled=True,
    )


__all__ = [
    "CONVERGENCE_STOPPING_RULES",
    "KKT_RESIDUALS",
    "OBJECTIVE_STATIONARITY",
    "KktResiduals",
    "KktTolerances",
    "classify_convergence",
    "convergence_status_from_config",
    "convergence_status",
    "kkt_tolerances_from_config",
    "objective_convergence_status",
]
