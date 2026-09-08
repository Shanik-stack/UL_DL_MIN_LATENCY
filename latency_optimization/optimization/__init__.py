"""Optimization utilities shared by uplink and downlink."""

from .convergence_criteria import (
    CONVERGENCE_STOPPING_RULES,
    KKT_RESIDUALS,
    OBJECTIVE_STATIONARITY,
    KktResiduals,
    KktTolerances,
    classify_convergence,
    convergence_status_from_config,
    convergence_status,
    kkt_tolerances_from_config,
    objective_convergence_status,
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
