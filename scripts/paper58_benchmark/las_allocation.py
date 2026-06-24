from __future__ import annotations

from collections import deque
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
    source_counts: dict[int, int],
    remaining: dict[int, int],
    allowed_transitions: set[tuple[int, int]] | None,
) -> bool:
    required = {int(cls): int(needed) for cls, needed in remaining.items() if int(needed) > 0}
    total_required = int(sum(required.values()))
    if total_required == 0:
        return True

    available = {int(cls): int(count) for cls, count in source_counts.items() if int(count) > 0}
    if int(sum(available.values())) < total_required:
        return False
    if allowed_transitions is None:
        return True

    source_classes = sorted(available)
    target_classes = sorted(required)
    source_offset = 1
    target_offset = source_offset + len(source_classes)
    sink = target_offset + len(target_classes)
    graph = [[0] * (sink + 1) for _ in range(sink + 1)]

    for index, from_cls in enumerate(source_classes):
        graph[0][source_offset + index] = available[from_cls]
    for source_index, from_cls in enumerate(source_classes):
        for target_index, to_cls in enumerate(target_classes):
            if _is_allowed(from_cls, to_cls, allowed_transitions):
                graph[source_offset + source_index][target_offset + target_index] = available[from_cls]
    for index, to_cls in enumerate(target_classes):
        graph[target_offset + index][sink] = required[to_cls]

    flow = 0
    while True:
        parent = [-1] * len(graph)
        parent[0] = 0
        queue: deque[int] = deque([0])
        while queue and parent[sink] == -1:
            node = queue.popleft()
            for next_node, capacity in enumerate(graph[node]):
                if capacity > 0 and parent[next_node] == -1:
                    parent[next_node] = node
                    queue.append(next_node)
                    if next_node == sink:
                        break
        if parent[sink] == -1:
            break

        increment = total_required - flow
        node = sink
        while node != 0:
            prev = parent[node]
            increment = min(increment, graph[prev][node])
            node = prev
        node = sink
        while node != 0:
            prev = parent[node]
            graph[prev][node] -= increment
            graph[node][prev] += increment
            node = prev
        flow += increment
        if flow == total_required:
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
    editable_values, editable_counts = np.unique(start[editable], return_counts=True)
    unassigned_source_counts = {
        int(value): int(count) for value, count in zip(editable_values, editable_counts, strict=False)
    }

    stable_candidates: list[tuple[float, float, int, int, int]] = []
    for row, col in np.argwhere(editable):
        from_cls = int(start[row, col])
        if remaining.get(from_cls, 0) <= 0 or from_cls not in class_to_col:
            continue
        stay_score = float(scores[row, col, class_to_col[from_cls]])
        best_change_score: float | None = None
        for to_cls in class_values:
            to_key = int(to_cls)
            if to_key == from_cls:
                continue
            if remaining.get(to_key, 0) <= 0:
                continue
            if not _is_allowed(from_cls, to_key, allowed_transitions):
                continue
            change_score = float(scores[row, col, class_to_col[to_key]])
            best_change_score = change_score if best_change_score is None else max(best_change_score, change_score)
        persistence_margin = float("inf") if best_change_score is None else stay_score - best_change_score
        stable_candidates.append((persistence_margin, stay_score, int(row), int(col), from_cls))

    stable_candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    for _, _, row, col, from_cls in stable_candidates:
        if assigned[row, col]:
            continue
        if remaining.get(from_cls, 0) <= 0:
            continue
        tentative_source_counts = unassigned_source_counts.copy()
        tentative_source_counts[from_cls] = tentative_source_counts.get(from_cls, 0) - 1
        tentative_remaining = remaining.copy()
        tentative_remaining[from_cls] -= 1
        if not _remaining_demand_is_feasible(tentative_source_counts, tentative_remaining, allowed_transitions):
            continue
        assigned[row, col] = True
        unassigned_source_counts[from_cls] -= 1
        remaining[from_cls] -= 1

    candidates: list[tuple[float, int, int, int, int]] = []
    for row, col in np.argwhere(editable):
        if assigned[row, col]:
            continue
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
        tentative_source_counts = unassigned_source_counts.copy()
        tentative_source_counts[from_cls] = tentative_source_counts.get(from_cls, 0) - 1
        tentative_remaining = remaining.copy()
        tentative_remaining[to_cls] -= 1
        if not _remaining_demand_is_feasible(tentative_source_counts, tentative_remaining, allowed_transitions):
            continue
        simulated[row, col] = to_cls
        assigned[row, col] = True
        unassigned_source_counts[from_cls] -= 1
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
                if from_cls != assigned_class:
                    selected.append(
                        {
                            "row": int(row),
                            "col": int(col),
                            "from_class": from_cls,
                            "to_class": assigned_class,
                            "score": float(scores[row, col, class_to_col[assigned_class]]),
                        }
                    )
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
