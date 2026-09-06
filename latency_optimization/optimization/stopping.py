"""Stopping rule shared by objective-only neural and direct solvers."""


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


__all__ = ["objective_convergence_status"]
