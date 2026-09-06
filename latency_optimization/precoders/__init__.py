"""Precoder representations, constraints, and provider contracts."""

from .power import (
    POWER_PROJECTION_SAFETY_MARGIN,
    cap_matrix_power_numpy,
    cap_matrix_power_torch,
    joint_power_scale_numpy,
    joint_power_scale_torch,
    normalize_matrix_power_numpy,
    normalize_matrix_power_torch,
)

__all__ = [
    "POWER_PROJECTION_SAFETY_MARGIN",
    "cap_matrix_power_numpy",
    "cap_matrix_power_torch",
    "joint_power_scale_numpy",
    "joint_power_scale_torch",
    "normalize_matrix_power_numpy",
    "normalize_matrix_power_torch",
]
