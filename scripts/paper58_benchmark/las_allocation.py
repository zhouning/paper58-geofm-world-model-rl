from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.paper58_benchmark.las_demand import (
    DemandValidationError,
    build_editable_mask,
    remaining_editable_demand,
)


@dataclass(frozen=True)
class LASAllocationResult:
    simulated_map: np.ndarray
    target_demand: dict[int, int]
    achieved_demand: dict[int, int]
    unmet_demand: dict[int, int]
    selected_transitions: list[dict[str, int | float]]
    constraint_violations: list[str]


def _counts(label_map: np.ndarray, class_values: list[int]) -> dict[int, int]:
    arr = np.asarray(label_map)
    return {int(cls): int(np.count_nonzero(arr == int(cls))) for cls in class_values}


def _is_allowed(from_cls: int, to_cls: int, allowed_transitions: set[tuple[int, int]] | None) -> bool:
    if from_cls == to_cls:
        return True
    if allowed_transitions is None:
        return True
    return (int(from_cls), int(to_cls)) in allowed_transitions


def _remaining_demand_is_feasible(
    start_map: np.ndarray,
    assigned: np.ndarray,
    remaining: dict[int, int],
    allowed_transitions: set[tuple[int, int]] | None,
) -> bool:
    slots: list[int] = []
    for to_cls, needed in sorted(remaining.items()):
        slots.extend([int(to_cls)] * int(needed))
    if not slots:
        return True

    unassigned_pixels = [(int(row), int(col)) for row, col in np.argwhere(~assigned)]
    if len(unassigned_pixels) < len(slots):
        return False

    edges: list[list[int]] = []
    for row, col in unassigned_pixels:
        from_cls = int(start_map[row, col])
        edges.append(
            [
                slot_index
                for slot_index, to_cls in enumerate(slots)
                if _is_allowed(from_cls, to_cls, allowed_transitions)
            ]
        )

    slot_to_pixel = [-1] * len(slots)

    def find_match(pixel_index: int, seen: list[bool]) -> bool:
        for slot_index in edges[pixel_index]:
            if seen[slot_index]:
                continue
            seen[slot_index] = True
            matched_pixel = slot_to_pixel[slot_index]
            if matched_pixel == -1 or find_match(matched_pixel, seen):
                slot_to_pixel[slot_index] = pixel_index
                return True
        return False

    matched_slots = 0
    for pixel_index in range(len(unassigned_pixels)):
        if find_match(pixel_index, [False] * len(slots)):
            matched_slots += 1
            if matched_slots == len(slots):
                return True
    return False


def allocate_demand_constrained(
    start_map: np.ndarray,
    suitability: np.ndarray,
    class_values: list[int],
    target_demand: dict[int, int],
    exclusion_mask: np.ndarray | None = None,
    immutable_classes: set[int] | None = None,
    allowed_transitions: set[tuple[int, int]] | None = None,
) -> LASAllocationResult:
    start = np.asarray(start_map)
    scores = np.asarray(suitability, dtype=np.float32)
    if scores.shape != start.shape + (len(class_values),):
        raise DemandValidationError(
            f"suitability shape {scores.shape} does not match start/class shape {start.shape + (len(class_values),)}"
        )
    demand = {int(cls): int(count) for cls, count in target_demand.items()}
    editable = build_editable_mask(start, exclusion_mask=exclusion_mask, immutable_classes=immutable_classes)
    remaining = remaining_editable_demand(start, demand, editable)
    class_to_col = {int(cls): index for index, cls in enumerate(class_values)}
    simulated = start.copy()
    assigned = ~editable
    selected: list[dict[str, int | float]] = []

    candidates: list[tuple[float, int, int, int, int]] = []
    for row, col in np.argwhere(editable):
        from_cls = int(start[row, col])
        for to_cls in class_values:
            to_key = int(to_cls)
            if remaining.get(to_key, 0) <= 0:
                continue
            if not _is_allowed(from_cls, to_key, allowed_transitions):
                continue
            score = float(scores[row, col, class_to_col[to_key]])
            if from_cls == to_key:
                score += 1e-6
            candidates.append((score, int(row), int(col), from_cls, to_key))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[4]))
    for score, row, col, from_cls, to_cls in candidates:
        if assigned[row, col]:
            continue
        if remaining.get(to_cls, 0) <= 0:
            continue
        tentative_assigned = assigned.copy()
        tentative_assigned[row, col] = True
        tentative_remaining = remaining.copy()
        tentative_remaining[to_cls] -= 1
        if not _remaining_demand_is_feasible(start, tentative_assigned, tentative_remaining, allowed_transitions):
            continue
        simulated[row, col] = to_cls
        assigned[row, col] = True
        remaining[to_cls] -= 1
        if from_cls != to_cls:
            selected.append(
                {
                    "row": row,
                    "col": col,
                    "from_class": from_cls,
                    "to_class": to_cls,
                    "score": score,
                }
            )

    violations: list[str] = []
    if np.any(~assigned):
        for row, col in np.argwhere(~assigned):
            from_cls = int(start[row, col])
            assigned_class = None
            for to_cls in class_values:
                to_key = int(to_cls)
                if remaining.get(to_key, 0) > 0 and _is_allowed(from_cls, to_key, allowed_transitions):
                    assigned_class = to_key
                    break
            if assigned_class is None:
                violations.append(f"no feasible class for pixel ({int(row)}, {int(col)}) from class {from_cls}")
                simulated[row, col] = from_cls
            else:
                simulated[row, col] = assigned_class
                remaining[assigned_class] -= 1
            assigned[row, col] = True

    achieved = _counts(simulated, class_values)
    unmet = {int(cls): max(0, int(demand.get(int(cls), 0)) - int(achieved.get(int(cls), 0))) for cls in class_values}
    for cls in class_values:
        target = int(demand.get(int(cls), 0))
        actual = int(achieved.get(int(cls), 0))
        if target != actual:
            violations.append(f"class {int(cls)} demand target {target} achieved {actual}")

    return LASAllocationResult(
        simulated_map=simulated,
        target_demand=demand,
        achieved_demand=achieved,
        unmet_demand=unmet,
        selected_transitions=selected,
        constraint_violations=violations,
    )
