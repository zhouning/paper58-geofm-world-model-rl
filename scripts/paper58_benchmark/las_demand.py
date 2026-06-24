from __future__ import annotations

import numpy as np


class DemandValidationError(ValueError):
    """Raised when a demand or constraint configuration cannot be satisfied."""


def derive_observed_demand(end_map: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(np.asarray(end_map), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def derive_demand(
    start_map: np.ndarray,
    end_map: np.ndarray,
    prediction_map: np.ndarray,
    demand_source: str = "observed_end",
) -> dict[int, int]:
    if demand_source == "observed_end":
        return derive_observed_demand(end_map)
    if demand_source == "paper58_prediction":
        return derive_observed_demand(prediction_map)
    if demand_source == "start_persistence":
        return derive_observed_demand(start_map)
    raise DemandValidationError(
        "unsupported demand_source "
        f"{demand_source!r}; expected one of: observed_end, paper58_prediction, start_persistence"
    )


def validate_total_demand(start_map: np.ndarray, demand: dict[int, int]) -> None:
    total = int(sum(int(value) for value in demand.values()))
    n_cells = int(np.asarray(start_map).size)
    if total != n_cells:
        raise DemandValidationError(f"demand total {total} does not match raster cells {n_cells}")
    for cls, count in demand.items():
        if int(count) < 0:
            raise DemandValidationError(f"demand for class {int(cls)} is negative: {int(count)}")


def build_editable_mask(
    start_map: np.ndarray,
    exclusion_mask: np.ndarray | None = None,
    immutable_classes: set[int] | None = None,
) -> np.ndarray:
    start = np.asarray(start_map)
    editable = np.ones(start.shape, dtype=bool)
    if exclusion_mask is not None:
        exclusion = np.asarray(exclusion_mask, dtype=bool)
        if exclusion.shape != start.shape:
            raise DemandValidationError(
                f"exclusion mask shape {exclusion.shape} does not match start map shape {start.shape}"
            )
        editable &= ~exclusion
    for cls in sorted(immutable_classes or set()):
        editable &= start != int(cls)
    return editable


def remaining_editable_demand(
    start_map: np.ndarray,
    demand: dict[int, int],
    editable_mask: np.ndarray,
) -> dict[int, int]:
    start = np.asarray(start_map)
    editable = np.asarray(editable_mask, dtype=bool)
    if editable.shape != start.shape:
        raise DemandValidationError(f"editable mask shape {editable.shape} does not match start map shape {start.shape}")
    validate_total_demand(start, demand)
    fixed = start[~editable]
    fixed_values, fixed_counts = np.unique(fixed, return_counts=True)
    fixed_by_class = {int(value): int(count) for value, count in zip(fixed_values, fixed_counts)}
    remaining: dict[int, int] = {}
    for cls, target_count in sorted((int(cls), int(count)) for cls, count in demand.items()):
        fixed_count = fixed_by_class.get(cls, 0)
        if fixed_count > target_count:
            raise DemandValidationError(f"fixed class {cls} count {fixed_count} exceeds demand {target_count}")
        remaining[cls] = target_count - fixed_count
    editable_total = int(np.count_nonzero(editable))
    remaining_total = int(sum(remaining.values()))
    if remaining_total != editable_total:
        raise DemandValidationError(
            f"remaining editable demand {remaining_total} does not match editable cells {editable_total}"
        )
    return remaining
