"""Strict validation helpers for public configuration values."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any


def require_choice(value: Any, choices: Collection[str], option_name: str) -> str:
    """Return an exact supported string or raise a descriptive error."""
    if not isinstance(value, str) or value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{option_name} must be exactly one of: {expected}; got {value!r}.")
    return value


def require_bool(value: Any, option_name: str) -> bool:
    """Reject string and numeric substitutes for YAML booleans."""
    if type(value) is not bool:
        raise ValueError(f"{option_name} must be true or false; got {value!r}.")
    return value


__all__ = ["require_bool", "require_choice"]
