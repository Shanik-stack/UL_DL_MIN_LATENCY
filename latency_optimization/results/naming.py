"""Stable, compact result names for the exact configured experiment values."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from .persistence import make_serializable


METHOD_TAGS = {
    "convergence_per_epoch_baseline": "conv",
    "monte_carlo_precoder_net_train_test": "mc",
    "monte_carlo_precoder_net_test": "mc_test",
    "small_exhaustive_payload_compare": "exhaustive",
}
OBJECTIVE_TAGS = {
    "unweighted_sum_rate": "unweighted",
    "inverse_cnr_weighted_sum_rate": "inverse_cnr",
}
SCOPE_TAGS = {
    "bs_shared_net": "shared_bs",
    "per_user_nets": "per_user",
}
UPDATE_MODE_TAGS = {
    "precoder_net": "net",
    "direct_precoder": "direct",
}
TRAINING_STYLE_TAGS = {
    "rollout_query_objective": "rollout",
}


def _safe_tag_token(value: str) -> str:
    """Make an already-selected value safe for use in a filename."""
    token = re.sub(r"[^A-Za-z0-9._-]", "_", str(value))
    token = re.sub(r"_+", "_", token).strip("_-")
    return token


def _tag_from_exact_value(value: str, tags: dict[str, str], option_name: str) -> str:
    if not isinstance(value, str) or value not in tags:
        expected = ", ".join(sorted(tags))
        raise ValueError(f"{option_name} must be exactly one of: {expected}; got {value!r}.")
    return tags[value]


def format_method_tag(method_name: str) -> str:
    return _tag_from_exact_value(method_name, METHOD_TAGS, "method")


def format_objective_tag(objective_mode: str) -> str:
    return _tag_from_exact_value(objective_mode, OBJECTIVE_TAGS, "objective mode")


def format_scope_tag(scope_name: str) -> str:
    return _tag_from_exact_value(scope_name, SCOPE_TAGS, "precoder-net scope")


def format_update_mode_tag(update_mode: str) -> str:
    return _tag_from_exact_value(update_mode, UPDATE_MODE_TAGS, "precoder update mode")


def format_training_style_tag(training_style: str) -> str:
    return _tag_from_exact_value(training_style, TRAINING_STYLE_TAGS, "Monte Carlo training style")


def join_tag_parts(*parts: str | None) -> str:
    return "_".join(_safe_tag_token(part) for part in parts if part)


def compact_cfg_stem(cfg_name: str) -> str:
    stem = os.path.splitext(os.path.basename(str(cfg_name)))[0]
    for prefix in ("downlink_", "uplink_", "config_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return _safe_tag_token(stem) or "cfg"


def build_config_content_hash(config_obj: Any, *, length: int = 12) -> str:
    canonical = json.dumps(
        make_serializable(config_obj),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[: max(4, int(length))]


def make_method_result_tag(
    method_name: str,
    cfg_name: str,
    *,
    seed: int | None = None,
    cfg_hash: str | None = None,
) -> str:
    parts = [_safe_tag_token(method_name), compact_cfg_stem(cfg_name)]
    if cfg_hash:
        parts.append(f"h{_safe_tag_token(cfg_hash)}")
    if seed is not None:
        parts.append(f"s{int(seed)}")
    return "__".join(part for part in parts if part)


__all__ = [
    "build_config_content_hash",
    "compact_cfg_stem",
    "format_method_tag",
    "format_objective_tag",
    "format_scope_tag",
    "format_training_style_tag",
    "format_update_mode_tag",
    "join_tag_parts",
    "make_method_result_tag",
]
