from __future__ import annotations

from typing import Any, Callable

from .validation import require_choice


N_SEARCH_DIRECTIONS = {"ascending", "descending"}
N_SEARCH_STRATEGIES = {"fixed_step", "coarse_to_fine", "exponential", "binary"}
MONTE_CARLO_SEARCH_PHASES = {"training", "testing"}


def validate_n_search_direction(value: Any) -> str:
    """Validate whether candidate n values are visited upward or downward."""
    return require_choice(value, N_SEARCH_DIRECTIONS, "n_search_direction")


def validate_n_search_strategy(value: Any, *, allow_only_fixed_step: bool = False) -> str:
    """Validate a search strategy and enforce fixed-step-only training when requested."""
    strategy = require_choice(value, N_SEARCH_STRATEGIES, "n_search_strategy")
    if allow_only_fixed_step and strategy != "fixed_step":
        raise ValueError(
            "Monte Carlo currently supports only simulation.n_search_strategy='fixed_step' "
            "so rollout queries stay aligned with the visited training states."
        )
    return strategy


def build_n_search_config(
    *,
    n_min: int,
    n_max: int,
    fine_step: int,
    direction: Any,
    strategy: Any,
    coarse_step: int | None = None,
    exponential_factor: int | None = None,
    allow_only_fixed_step: bool = False,
) -> dict[str, int | str]:
    """Normalize one link-independent blocklength-search specification."""
    n_min_int = int(n_min)
    n_max_int = int(n_max)
    fine_step_int = max(1, int(fine_step))
    coarse_step_int = max(fine_step_int, int(coarse_step if coarse_step is not None else fine_step_int))
    exponential_factor_int = max(2, int(exponential_factor if exponential_factor is not None else 2))
    return {
        "n_min": n_min_int,
        "n_max": n_max_int,
        "fine_step": fine_step_int,
        "direction": validate_n_search_direction(direction),
        "strategy": validate_n_search_strategy(strategy, allow_only_fixed_step=allow_only_fixed_step),
        "coarse_step": coarse_step_int,
        "exponential_factor": exponential_factor_int,
    }


def build_monte_carlo_n_search_config(
    simulation: dict[str, Any],
    *,
    n_min: int,
    n_max: int,
    phase: str,
) -> dict[str, int | str]:
    """Resolve the shared Monte Carlo blocklength-search settings."""
    resolved_phase = require_choice(phase, MONTE_CARLO_SEARCH_PHASES, "monte_carlo_search_phase")
    fine_step = int(simulation["n_kl_step"])
    if resolved_phase == "training":
        return build_n_search_config(
            n_min=n_min,
            n_max=n_max,
            fine_step=fine_step,
            direction=simulation.get("n_search_direction", "descending"),
            strategy=simulation.get("n_search_strategy", "fixed_step"),
            coarse_step=simulation.get("n_search_coarse_step", fine_step),
            exponential_factor=simulation.get("n_search_exponential_factor", 2),
            allow_only_fixed_step=True,
        )

    return build_n_search_config(
        n_min=n_min,
        n_max=n_max,
        fine_step=fine_step,
        direction=simulation.get(
            "monte_carlo_test_n_search_direction",
            simulation.get("n_search_direction", "descending"),
        ),
        strategy=simulation.get(
            "monte_carlo_test_n_search_strategy",
            simulation.get("n_search_strategy", "fixed_step"),
        ),
        coarse_step=simulation.get(
            "monte_carlo_test_n_search_coarse_step",
            simulation.get("n_search_coarse_step", fine_step),
        ),
        exponential_factor=simulation.get(
            "monte_carlo_test_n_search_exponential_factor",
            simulation.get("n_search_exponential_factor", 2),
        ),
    )


def _descending_candidates(n_min: int, n_max: int, step: int) -> list[int]:
    values: list[int] = []
    candidate = int(n_max) - int(step)
    while candidate >= int(n_min):
        values.append(int(candidate))
        candidate -= int(step)
    if int(n_min) < int(n_max) and int(n_min) not in values:
        values.append(int(n_min))
    return values


def _ascending_candidates(n_min: int, n_max: int, step: int) -> list[int]:
    values: list[int] = []
    candidate = int(n_min)
    while candidate < int(n_max):
        values.append(int(candidate))
        candidate += int(step)
    return values


def build_fixed_step_n_candidates(
    *,
    n_min: int,
    n_max: int,
    step: int,
    direction: Any,
) -> list[int]:
    """Generate deterministic ascending or descending fixed-step n candidates."""
    resolved_direction = validate_n_search_direction(direction)
    if resolved_direction == "descending":
        return _descending_candidates(int(n_min), int(n_max), max(1, int(step)))
    return _ascending_candidates(int(n_min), int(n_max), max(1, int(step)))


def _exponential_descending_candidates(
    n_min: int,
    n_max: int,
    fine_step: int,
    factor: int,
) -> list[int]:
    values: list[int] = []
    offset = int(fine_step)
    seen: set[int] = set()
    while True:
        candidate = max(int(n_min), int(n_max) - int(offset))
        if candidate in seen:
            break
        seen.add(candidate)
        values.append(int(candidate))
        if candidate <= int(n_min):
            break
        offset *= int(factor)
    return values


def _exponential_ascending_candidates(
    n_min: int,
    n_max: int,
    fine_step: int,
    factor: int,
) -> list[int]:
    values: list[int] = []
    offset = 0
    seen: set[int] = set()
    while True:
        candidate = min(int(n_max) - 1, int(n_min) + int(offset))
        if candidate < int(n_min):
            candidate = int(n_min)
        if candidate in seen:
            break
        seen.add(candidate)
        values.append(int(candidate))
        if candidate >= int(n_max) - 1:
            break
        offset = int(fine_step) if offset == 0 else int(offset) * int(factor)
    return values


def _refine_descending_candidates(
    last_feasible_n: int,
    lower_exclusive: int,
    fine_step: int,
) -> list[int]:
    values: list[int] = []
    candidate = int(last_feasible_n) - int(fine_step)
    while candidate > int(lower_exclusive):
        values.append(int(candidate))
        candidate -= int(fine_step)
    return values


def _refine_ascending_candidates(
    last_infeasible_n: int,
    upper_feasible_n: int,
    fine_step: int,
) -> list[int]:
    values: list[int] = []
    candidate = int(last_infeasible_n) + int(fine_step)
    while candidate < int(upper_feasible_n):
        values.append(int(candidate))
        candidate += int(fine_step)
    return values


def run_n_frontier_search(
    search_cfg: dict[str, int | str],
    evaluate_candidate: Callable[[int, str], dict[str, Any]],
) -> dict[str, Any]:
    """Search candidate blocklengths and retain the feasible frontier.

    The callback owns link-specific beam/rate evaluation; this function owns
    candidate ordering, coarse-to-fine refinement, and visited-state tracking.
    """
    n_min = int(search_cfg["n_min"])
    n_max = int(search_cfg["n_max"])
    fine_step = int(search_cfg["fine_step"])
    coarse_step = int(search_cfg["coarse_step"])
    exponential_factor = int(search_cfg["exponential_factor"])
    direction = str(search_cfg["direction"])
    strategy = str(search_cfg["strategy"])

    accepted: list[dict[str, Any]] = []
    visited: list[dict[str, Any]] = []
    visited_n_values: set[int] = set()
    frontier_rejected: dict[str, Any] | None = None

    def test_candidate(candidate_n: int, stage: str) -> tuple[bool, dict[str, Any]] | None:
        n_int = int(candidate_n)
        if n_int <= 0 or n_int >= n_max or n_int in visited_n_values:
            return None
        visited_n_values.add(n_int)
        result = evaluate_candidate(int(n_int), str(stage))
        feasible = bool(result.get("feasible", False))
        event = {
            "n_kl": int(n_int),
            "stage": str(stage),
            "feasible": bool(feasible),
            "result": result,
        }
        visited.append(event)
        if feasible:
            accepted.append(event)
        return feasible, event

    if strategy == "binary":
        candidate_grid = _ascending_candidates(n_min, n_max, fine_step)
        low_idx = 0
        high_idx = len(candidate_grid) - 1
        best_feasible_n = int(n_max)
        while low_idx <= high_idx:
            mid_idx = (low_idx + high_idx) // 2
            candidate_n = int(candidate_grid[mid_idx])
            tested = test_candidate(candidate_n, "binary")
            if tested is None:
                break
            feasible, event = tested
            if feasible:
                best_feasible_n = int(candidate_n)
                high_idx = mid_idx - 1
            else:
                frontier_rejected = event
                low_idx = mid_idx + 1
        best_n = int(best_feasible_n)
    elif direction == "descending":
        if strategy == "fixed_step":
            probe_candidates = _descending_candidates(n_min, n_max, fine_step)
            refine_candidates: list[int] = []
        elif strategy == "coarse_to_fine":
            probe_candidates = _descending_candidates(n_min, n_max, coarse_step)
            refine_candidates = []
        else:
            probe_candidates = _exponential_descending_candidates(
                n_min,
                n_max,
                fine_step,
                exponential_factor,
            )
            refine_candidates = []

        last_feasible_n = int(n_max)
        lower_frontier = int(n_min) - int(fine_step)
        for candidate_n in probe_candidates:
            tested = test_candidate(
                int(candidate_n),
                "fixed_step" if strategy == "fixed_step" else "probe",
            )
            if tested is None:
                continue
            feasible, event = tested
            if feasible:
                last_feasible_n = int(candidate_n)
                continue
            frontier_rejected = event
            lower_frontier = int(candidate_n)
            break

        if strategy in {"coarse_to_fine", "exponential"}:
            refine_candidates = _refine_descending_candidates(last_feasible_n, lower_frontier, fine_step)
            for candidate_n in refine_candidates:
                tested = test_candidate(int(candidate_n), "refine")
                if tested is None:
                    continue
                feasible, event = tested
                if feasible:
                    last_feasible_n = int(candidate_n)
                    continue
                frontier_rejected = event
                break

        best_n = int(last_feasible_n)
    else:
        if strategy == "fixed_step":
            probe_candidates = _ascending_candidates(n_min, n_max, fine_step)
            refine_candidates = []
        elif strategy == "coarse_to_fine":
            probe_candidates = _ascending_candidates(n_min, n_max, coarse_step)
            refine_candidates = []
        else:
            probe_candidates = _exponential_ascending_candidates(
                n_min,
                n_max,
                fine_step,
                exponential_factor,
            )
            refine_candidates = []

        last_infeasible_n = int(n_min) - int(fine_step)
        upper_feasible_n: int | None = None
        for candidate_n in probe_candidates:
            tested = test_candidate(
                int(candidate_n),
                "fixed_step" if strategy == "fixed_step" else "probe",
            )
            if tested is None:
                continue
            feasible, event = tested
            if feasible:
                upper_feasible_n = int(candidate_n)
                break
            frontier_rejected = event
            last_infeasible_n = int(candidate_n)

        if upper_feasible_n is None:
            upper_feasible_n = int(n_max)

        if strategy in {"coarse_to_fine", "exponential"}:
            refine_candidates = _refine_ascending_candidates(
                last_infeasible_n,
                upper_feasible_n,
                fine_step,
            )
            for candidate_n in refine_candidates:
                tested = test_candidate(int(candidate_n), "refine")
                if tested is None:
                    continue
                feasible, event = tested
                if feasible:
                    upper_feasible_n = int(candidate_n)
                    break
                frontier_rejected = event
                last_infeasible_n = int(candidate_n)

        best_n = int(upper_feasible_n)

    return {
        "best_n": int(best_n),
        "accepted": accepted,
        "frontier_rejected": frontier_rejected,
        "visited": visited,
    }
