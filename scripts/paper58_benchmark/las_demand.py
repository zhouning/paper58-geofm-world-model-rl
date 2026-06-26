from __future__ import annotations

import numpy as np


class DemandValidationError(ValueError):
    """Raised when a demand or constraint configuration cannot be satisfied."""


def derive_observed_demand(end_map: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(np.asarray(end_map), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def _round_expected_counts(expected: np.ndarray, class_values: list[int], total: int) -> dict[int, int]:
    base = np.floor(expected).astype(np.int64)
    deficit = int(total - int(np.sum(base)))
    if deficit > 0:
        remainders = expected - base
        order = sorted(range(len(class_values)), key=lambda index: (-float(remainders[index]), int(class_values[index])))
        for index in order[:deficit]:
            base[index] += 1
    elif deficit < 0:
        remainders = expected - base
        order = sorted(range(len(class_values)), key=lambda index: (float(remainders[index]), -int(class_values[index])))
        for index in order[: abs(deficit)]:
            if base[index] > 0:
                base[index] -= 1
    return {int(cls): int(base[index]) for index, cls in enumerate(class_values)}


def project_transition_prior_demand(
    start_map: np.ndarray,
    class_values: list[int],
    transition_prior: dict[tuple[int, int], float],
) -> dict[int, int]:
    start = np.asarray(start_map)
    classes = [int(cls) for cls in class_values]
    if not classes:
        raise DemandValidationError("class_values must be non-empty for transition_prior demand")
    values, counts = np.unique(start, return_counts=True)
    start_counts = {int(value): int(count) for value, count in zip(values, counts)}
    expected = np.zeros(len(classes), dtype=np.float64)
    class_to_index = {int(cls): index for index, cls in enumerate(classes)}
    for from_cls in classes:
        count = int(start_counts.get(from_cls, 0))
        if count == 0:
            continue
        row = np.array([float(transition_prior.get((from_cls, to_cls), 0.0)) for to_cls in classes], dtype=np.float64)
        row_sum = float(np.sum(row))
        if row_sum <= 0.0:
            expected[class_to_index[from_cls]] += count
        else:
            expected += count * (row / row_sum)
    return _round_expected_counts(expected, classes, int(start.size))


def blend_demand_counts(
    primary_demand: dict[int, int],
    secondary_demand: dict[int, int],
    class_values: list[int],
    secondary_weight: float,
    total: int,
) -> dict[int, int]:
    weight = float(secondary_weight)
    if weight < 0.0 or weight > 1.0:
        raise DemandValidationError(f"demand_blend_weight must be in [0, 1]: {weight}")
    classes = [int(cls) for cls in class_values]
    if not classes:
        raise DemandValidationError("class_values must be non-empty for blended demand")
    primary = np.array([int(primary_demand.get(cls, 0)) for cls in classes], dtype=np.float64)
    secondary = np.array([int(secondary_demand.get(cls, 0)) for cls in classes], dtype=np.float64)
    expected = (1.0 - weight) * primary + weight * secondary
    return _round_expected_counts(expected, classes, int(total))


def adaptive_transition_prior_blend_demand(
    start_map: np.ndarray,
    prediction_map: np.ndarray,
    class_values: list[int],
    transition_prior: dict[tuple[int, int], float],
    demand_blend_weight: float,
    adaptive_demand_l1_threshold: float,
    adaptive_demand_change_fraction_high: float,
) -> dict[int, int]:
    threshold = float(adaptive_demand_l1_threshold)
    if threshold < 0.0:
        raise DemandValidationError(f"adaptive_demand_l1_threshold must be non-negative: {threshold}")
    change_high = float(adaptive_demand_change_fraction_high)
    if change_high < 0.0 or change_high > 1.0:
        raise DemandValidationError(f"adaptive_demand_change_fraction_high must be in [0, 1]: {change_high}")

    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    if start.shape != prediction.shape:
        raise DemandValidationError(
            f"prediction map shape {prediction.shape} does not match start map shape {start.shape}"
        )
    total = int(start.size)
    prior_demand = project_transition_prior_demand(start, class_values, transition_prior)
    prediction_demand = derive_observed_demand(prediction)
    classes = [int(cls) for cls in class_values]
    l1 = sum(abs(int(prior_demand.get(cls, 0)) - int(prediction_demand.get(cls, 0))) for cls in classes)
    l1_fraction = (float(l1) / float(total)) if total > 0 else 0.0
    prediction_change_fraction = (float(np.count_nonzero(prediction != start)) / float(total)) if total > 0 else 0.0
    if l1_fraction >= threshold and prediction_change_fraction <= change_high:
        return blend_demand_counts(
            prior_demand,
            prediction_demand,
            class_values,
            secondary_weight=demand_blend_weight,
            total=total,
        )
    return prior_demand


def adaptive_transition_prior_blend_should_blend(
    start_map: np.ndarray,
    prediction_map: np.ndarray,
    class_values: list[int],
    transition_prior: dict[tuple[int, int], float],
    adaptive_demand_l1_threshold: float,
    adaptive_demand_change_fraction_high: float,
) -> bool:
    threshold = float(adaptive_demand_l1_threshold)
    if threshold < 0.0:
        raise DemandValidationError(f"adaptive_demand_l1_threshold must be non-negative: {threshold}")
    change_high = float(adaptive_demand_change_fraction_high)
    if change_high < 0.0 or change_high > 1.0:
        raise DemandValidationError(f"adaptive_demand_change_fraction_high must be in [0, 1]: {change_high}")

    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    if start.shape != prediction.shape:
        raise DemandValidationError(
            f"prediction map shape {prediction.shape} does not match start map shape {start.shape}"
        )
    total = int(start.size)
    prior_demand = project_transition_prior_demand(start, class_values, transition_prior)
    prediction_demand = derive_observed_demand(prediction)
    classes = [int(cls) for cls in class_values]
    l1 = sum(abs(int(prior_demand.get(cls, 0)) - int(prediction_demand.get(cls, 0))) for cls in classes)
    l1_fraction = (float(l1) / float(total)) if total > 0 else 0.0
    prediction_change_fraction = (float(np.count_nonzero(prediction != start)) / float(total)) if total > 0 else 0.0
    return l1_fraction >= threshold and prediction_change_fraction <= change_high


def _gross_change_budget_from_source(
    start_map: np.ndarray,
    end_map: np.ndarray,
    prediction_map: np.ndarray,
    target_demand: dict[int, int] | None,
    change_budget_source: str,
) -> int | None:
    start = np.asarray(start_map)
    end = np.asarray(end_map)
    prediction = np.asarray(prediction_map)
    if change_budget_source == "paper58_prediction":
        return int(np.count_nonzero(prediction != start))
    if change_budget_source == "observed_end":
        return int(np.count_nonzero(end != start))
    if change_budget_source == "demand_delta":
        if target_demand is None:
            raise DemandValidationError("demand_delta change budget requires target_demand")
        return minimum_change_budget_from_demand(start, target_demand)
    if change_budget_source == "none":
        return None
    raise DemandValidationError(
        "unsupported change_budget_source "
        f"{change_budget_source!r}; expected one of: paper58_prediction, observed_end, demand_delta, none"
    )


def _reciprocal_churn_fraction_from_prediction(start_map: np.ndarray, prediction_map: np.ndarray) -> float:
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    if start.shape != prediction.shape:
        raise DemandValidationError(
            f"prediction map shape {prediction.shape} does not match start map shape {start.shape}"
        )
    transitions: dict[tuple[int, int], int] = {}
    for from_cls, to_cls in zip(start.ravel(), prediction.ravel()):
        from_key = int(from_cls)
        to_key = int(to_cls)
        if from_key == to_key:
            continue
        key = (from_key, to_key)
        transitions[key] = transitions.get(key, 0) + 1
    gross_changes = int(sum(transitions.values()))
    if gross_changes <= 0:
        return 0.0
    reciprocal = 0
    visited: set[tuple[int, int]] = set()
    for (from_cls, to_cls), count in transitions.items():
        pair = (min(from_cls, to_cls), max(from_cls, to_cls))
        if pair in visited:
            continue
        reciprocal += 2 * min(count, int(transitions.get((to_cls, from_cls), 0)))
        visited.add(pair)
    return float(reciprocal) / float(gross_changes)


def select_change_budget_scale(
    start_map: np.ndarray,
    end_map: np.ndarray,
    prediction_map: np.ndarray,
    target_demand: dict[int, int] | None = None,
    change_budget_source: str = "paper58_prediction",
    base_change_budget_scale: float = 1.0,
    adaptive_change_budget_scale: float | None = None,
    adaptive_change_budget_fraction_low: float = 0.0,
    adaptive_change_budget_fraction_high: float = 1.0,
    adaptive_churn_budget_scale: float | None = None,
    adaptive_churn_fraction_high: float = 0.5,
    demand_source: str = "observed_end",
    class_values: list[int] | None = None,
    transition_prior: dict[tuple[int, int], float] | None = None,
    adaptive_demand_l1_threshold: float | None = None,
    adaptive_demand_change_fraction_high: float = 1.0,
) -> float:
    base_scale = float(base_change_budget_scale)
    if base_scale < 0.0:
        raise DemandValidationError(f"base_change_budget_scale must be non-negative: {base_scale}")
    if adaptive_change_budget_scale is None and adaptive_churn_budget_scale is None:
        return base_scale

    adaptive_scale = None
    if adaptive_change_budget_scale is not None:
        adaptive_scale = float(adaptive_change_budget_scale)
        if adaptive_scale < 0.0:
            raise DemandValidationError(f"adaptive_change_budget_scale must be non-negative: {adaptive_scale}")

    churn_scale = None
    if adaptive_churn_budget_scale is not None:
        churn_scale = float(adaptive_churn_budget_scale)
        if churn_scale < 0.0:
            raise DemandValidationError(f"adaptive_churn_budget_scale must be non-negative: {churn_scale}")

    low = float(adaptive_change_budget_fraction_low)
    high = float(adaptive_change_budget_fraction_high)
    if low < 0.0 or high > 1.0:
        raise DemandValidationError(
            f"adaptive_change_budget_fraction bounds must be in [0, 1]: low={low}, high={high}"
        )
    if low >= high:
        raise DemandValidationError(
            "adaptive_change_budget_fraction_low must be smaller than adaptive_change_budget_fraction_high"
        )
    churn_high = float(adaptive_churn_fraction_high)
    if churn_high < 0.0 or churn_high > 1.0:
        raise DemandValidationError(
            f"adaptive_churn_fraction_high must be in [0, 1]: {churn_high}"
        )

    gross_budget = _gross_change_budget_from_source(start_map, end_map, prediction_map, target_demand, change_budget_source)
    if gross_budget is None:
        return base_scale
    total = int(np.asarray(start_map).size)
    change_fraction = (float(gross_budget) / float(total)) if total > 0 else 0.0

    if churn_scale is not None:
        churn_fraction = _reciprocal_churn_fraction_from_prediction(start_map, prediction_map)
        if churn_fraction >= churn_high:
            return churn_scale

    if adaptive_scale is not None and change_fraction >= high:
        return adaptive_scale
    if adaptive_scale is not None and change_fraction <= low:
        if demand_source != "transition_prior_adaptive_blend":
            return adaptive_scale
        if class_values is None or transition_prior is None or adaptive_demand_l1_threshold is None:
            raise DemandValidationError(
                "adaptive change budget low-fraction gate requires class_values, transition_prior, and adaptive_demand_l1_threshold"
            )
        if not adaptive_transition_prior_blend_should_blend(
            start_map,
            prediction_map,
            class_values,
            transition_prior,
            adaptive_demand_l1_threshold,
            adaptive_demand_change_fraction_high,
        ):
            return adaptive_scale
    return base_scale


def minimum_change_budget_from_demand(start_map: np.ndarray, target_demand: dict[int, int]) -> int:
    start = np.asarray(start_map)
    values, counts = np.unique(start, return_counts=True)
    start_counts = {int(value): int(count) for value, count in zip(values, counts)}
    classes = sorted(set(start_counts) | {int(cls) for cls in target_demand})
    deficits = [
        max(0, int(target_demand.get(cls, 0)) - int(start_counts.get(cls, 0)))
        for cls in classes
    ]
    return int(sum(deficits))


def derive_demand(
    start_map: np.ndarray,
    end_map: np.ndarray,
    prediction_map: np.ndarray,
    demand_source: str = "observed_end",
    class_values: list[int] | None = None,
    transition_prior: dict[tuple[int, int], float] | None = None,
    demand_blend_weight: float = 0.0,
    adaptive_demand_l1_threshold: float | None = None,
    adaptive_demand_change_fraction_high: float = 1.0,
) -> dict[int, int]:
    if demand_source == "observed_end":
        return derive_observed_demand(end_map)
    if demand_source == "paper58_prediction":
        return derive_observed_demand(prediction_map)
    if demand_source == "start_persistence":
        return derive_observed_demand(start_map)
    if demand_source == "transition_prior":
        if class_values is None or transition_prior is None:
            raise DemandValidationError("transition_prior demand requires class_values and transition_prior")
        return project_transition_prior_demand(start_map, class_values, transition_prior)
    if demand_source == "transition_prior_blend":
        if class_values is None or transition_prior is None:
            raise DemandValidationError("transition_prior_blend demand requires class_values and transition_prior")
        prior_demand = project_transition_prior_demand(start_map, class_values, transition_prior)
        prediction_demand = derive_observed_demand(prediction_map)
        return blend_demand_counts(
            prior_demand,
            prediction_demand,
            class_values,
            secondary_weight=demand_blend_weight,
            total=int(np.asarray(start_map).size),
        )
    if demand_source == "transition_prior_adaptive_blend":
        if class_values is None or transition_prior is None:
            raise DemandValidationError("transition_prior_adaptive_blend demand requires class_values and transition_prior")
        if adaptive_demand_l1_threshold is None:
            raise DemandValidationError(
                "transition_prior_adaptive_blend demand requires adaptive_demand_l1_threshold"
            )
        return adaptive_transition_prior_blend_demand(
            start_map,
            prediction_map,
            class_values,
            transition_prior,
            demand_blend_weight=demand_blend_weight,
            adaptive_demand_l1_threshold=adaptive_demand_l1_threshold,
            adaptive_demand_change_fraction_high=adaptive_demand_change_fraction_high,
        )
    raise DemandValidationError(
        "unsupported demand_source "
        f"{demand_source!r}; expected one of: observed_end, paper58_prediction, "
        "start_persistence, transition_prior, transition_prior_blend, transition_prior_adaptive_blend"
    )


def derive_change_budget(
    start_map: np.ndarray,
    end_map: np.ndarray,
    prediction_map: np.ndarray,
    target_demand: dict[int, int],
    change_budget_source: str = "paper58_prediction",
    change_budget_scale: float = 1.0,
) -> int | None:
    scale = float(change_budget_scale)
    if scale < 0.0:
        raise DemandValidationError(f"change_budget_scale must be non-negative: {scale}")
    budget = _gross_change_budget_from_source(start_map, end_map, prediction_map, target_demand, change_budget_source)
    if budget is None:
        return None
    minimum_budget = minimum_change_budget_from_demand(start_map, target_demand)
    return max(minimum_budget, int(round(budget * scale)))


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
