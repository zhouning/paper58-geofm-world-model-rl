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


def _neighborhood_affinity_cube(label_map: np.ndarray, class_values: list[int]) -> np.ndarray:
    labels = np.asarray(label_map)
    affinity = np.zeros(labels.shape + (len(class_values),), dtype=np.float32)
    neighbor_counts = np.zeros(labels.shape, dtype=np.float32)
    if labels.size == 0:
        return affinity

    height, width = labels.shape
    class_to_col = {int(cls): index for index, cls in enumerate(class_values)}
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if row_offset == 0 and col_offset == 0:
                continue
            src_row_start = max(0, -row_offset)
            src_row_end = min(height, height - row_offset)
            dst_row_start = max(0, row_offset)
            dst_row_end = min(height, height + row_offset)
            src_col_start = max(0, -col_offset)
            src_col_end = min(width, width - col_offset)
            dst_col_start = max(0, col_offset)
            dst_col_end = min(width, width + col_offset)
            source = labels[src_row_start:src_row_end, src_col_start:src_col_end]
            target = affinity[dst_row_start:dst_row_end, dst_col_start:dst_col_end]
            for cls, class_index in class_to_col.items():
                target[..., class_index] += source == cls
            neighbor_counts[dst_row_start:dst_row_end, dst_col_start:dst_col_end] += 1.0

    valid = neighbor_counts > 0
    affinity[valid] /= neighbor_counts[valid, None]
    return affinity


def _latent_neighborhood_affinity_cube(
    label_map: np.ndarray,
    embedding_grid: np.ndarray,
    class_values: list[int],
) -> np.ndarray:
    labels = np.asarray(label_map)
    embeddings = np.asarray(embedding_grid, dtype=np.float32)
    if embeddings.ndim != 3:
        raise DemandValidationError(f"embedding_grid must be shaped (H, W, D), got {embeddings.shape}")
    if embeddings.shape[:2] != labels.shape:
        raise DemandValidationError(
            f"embedding_grid spatial shape {embeddings.shape[:2]} does not match start map shape {labels.shape}"
        )
    affinity = np.zeros(labels.shape + (len(class_values),), dtype=np.float32)
    class_counts = np.zeros_like(affinity, dtype=np.float32)
    if labels.size == 0:
        return affinity

    height, width = labels.shape
    class_to_col = {int(cls): index for index, cls in enumerate(class_values)}
    embedding_norm = np.linalg.norm(embeddings, axis=-1)
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if row_offset == 0 and col_offset == 0:
                continue
            src_row_start = max(0, -row_offset)
            src_row_end = min(height, height - row_offset)
            dst_row_start = max(0, row_offset)
            dst_row_end = min(height, height + row_offset)
            src_col_start = max(0, -col_offset)
            src_col_end = min(width, width - col_offset)
            dst_col_start = max(0, col_offset)
            dst_col_end = min(width, width + col_offset)

            source_labels = labels[src_row_start:src_row_end, src_col_start:src_col_end]
            source_embeddings = embeddings[src_row_start:src_row_end, src_col_start:src_col_end]
            target_embeddings = embeddings[dst_row_start:dst_row_end, dst_col_start:dst_col_end]
            source_norm = embedding_norm[src_row_start:src_row_end, src_col_start:src_col_end]
            target_norm = embedding_norm[dst_row_start:dst_row_end, dst_col_start:dst_col_end]
            denom = source_norm * target_norm
            similarity = np.zeros(source_labels.shape, dtype=np.float32)
            valid = denom > 0.0
            cosine = np.sum(source_embeddings * target_embeddings, axis=-1)
            similarity[valid] = np.maximum(cosine[valid] / denom[valid], 0.0)

            target_affinity = affinity[dst_row_start:dst_row_end, dst_col_start:dst_col_end]
            target_counts = class_counts[dst_row_start:dst_row_end, dst_col_start:dst_col_end]
            for cls, class_index in class_to_col.items():
                mask = source_labels == cls
                target_affinity[..., class_index] += np.where(mask, similarity, 0.0)
                target_counts[..., class_index] += mask

    valid_counts = class_counts > 0
    affinity[valid_counts] /= class_counts[valid_counts]
    return affinity


def _allocation_score(
    scores: np.ndarray,
    neighborhood: np.ndarray,
    neighborhood_weight: float,
    latent_neighborhood: np.ndarray,
    latent_neighborhood_weight: float,
    row: int,
    col: int,
    target_cls: int,
    class_to_col: dict[int, int],
) -> float:
    class_index = class_to_col[int(target_cls)]
    return float(
        scores[row, col, class_index]
        + neighborhood_weight * neighborhood[row, col, class_index]
        + latent_neighborhood_weight * latent_neighborhood[row, col, class_index]
    )


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


def _add_balanced_change_swaps(
    simulated: np.ndarray,
    start: np.ndarray,
    scores: np.ndarray,
    neighborhood: np.ndarray,
    neighborhood_weight: float,
    latent_neighborhood: np.ndarray,
    latent_neighborhood_weight: float,
    class_values: list[int],
    class_to_col: dict[int, int],
    editable: np.ndarray,
    selected: list[dict[str, int | float]],
    allowed_transitions: set[tuple[int, int]] | None,
    target_change_pixels: int | None,
    balanced_swap_min_margin: float | None,
    balanced_swap_min_base_score: float | None,
) -> None:
    if target_change_pixels is None:
        return
    target = int(target_change_pixels)
    if target < 0:
        raise DemandValidationError(f"target_change_pixels must be non-negative: {target}")
    remaining_extra = target - int(np.count_nonzero(simulated != start))
    if remaining_extra < 2:
        return

    stable_coords = [
        (int(row), int(col))
        for row, col in np.argwhere(editable & (simulated == start))
        if int(start[row, col]) in class_to_col
    ]
    known_classes = {int(cls) for cls in class_values}
    swap_candidates: list[tuple[float, int, int, int, int, int, int]] = []
    for left_index, (left_row, left_col) in enumerate(stable_coords):
        left_cls = int(start[left_row, left_col])
        if left_cls not in known_classes:
            continue
        for right_row, right_col in stable_coords[left_index + 1 :]:
            right_cls = int(start[right_row, right_col])
            if right_cls == left_cls or right_cls not in known_classes:
                continue
            if not _is_allowed(left_cls, right_cls, allowed_transitions):
                continue
            if not _is_allowed(right_cls, left_cls, allowed_transitions):
                continue
            left_score = _allocation_score(
                scores,
                neighborhood,
                neighborhood_weight,
                latent_neighborhood,
                latent_neighborhood_weight,
                left_row,
                left_col,
                right_cls,
                class_to_col,
            )
            right_score = _allocation_score(
                scores,
                neighborhood,
                neighborhood_weight,
                latent_neighborhood,
                latent_neighborhood_weight,
                right_row,
                right_col,
                left_cls,
                class_to_col,
            )
            left_stay_score = _allocation_score(
                scores,
                neighborhood,
                neighborhood_weight,
                latent_neighborhood,
                latent_neighborhood_weight,
                left_row,
                left_col,
                left_cls,
                class_to_col,
            )
            right_stay_score = _allocation_score(
                scores,
                neighborhood,
                neighborhood_weight,
                latent_neighborhood,
                latent_neighborhood_weight,
                right_row,
                right_col,
                right_cls,
                class_to_col,
            )
            swap_margin = (left_score + right_score) - (left_stay_score + right_stay_score)
            if balanced_swap_min_margin is not None and swap_margin < balanced_swap_min_margin:
                continue
            left_base_score = float(scores[left_row, left_col, class_to_col[right_cls]])
            right_base_score = float(scores[right_row, right_col, class_to_col[left_cls]])
            if (
                balanced_swap_min_base_score is not None
                and left_base_score + right_base_score < balanced_swap_min_base_score
            ):
                continue
            swap_candidates.append(
                (
                    left_score + right_score,
                    left_row,
                    left_col,
                    left_cls,
                    right_row,
                    right_col,
                    right_cls,
                )
            )

    swap_candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[4], item[5]))
    used: set[tuple[int, int]] = set()
    for _, left_row, left_col, left_cls, right_row, right_col, right_cls in swap_candidates:
        if remaining_extra < 2:
            break
        left = (left_row, left_col)
        right = (right_row, right_col)
        if left in used or right in used:
            continue
        if simulated[left_row, left_col] != left_cls or simulated[right_row, right_col] != right_cls:
            continue
        simulated[left_row, left_col] = right_cls
        simulated[right_row, right_col] = left_cls
        used.add(left)
        used.add(right)
        remaining_extra -= 2
        selected.append(
            {
                "row": left_row,
                "col": left_col,
                "from_class": left_cls,
                "to_class": right_cls,
                "score": _allocation_score(
                    scores,
                    neighborhood,
                    neighborhood_weight,
                    latent_neighborhood,
                    latent_neighborhood_weight,
                    left_row,
                    left_col,
                    right_cls,
                    class_to_col,
                ),
            }
        )
        selected.append(
            {
                "row": right_row,
                "col": right_col,
                "from_class": right_cls,
                "to_class": left_cls,
                "score": _allocation_score(
                    scores,
                    neighborhood,
                    neighborhood_weight,
                    latent_neighborhood,
                    latent_neighborhood_weight,
                    right_row,
                    right_col,
                    left_cls,
                    class_to_col,
                ),
            }
        )


def allocate_demand_constrained(
    start_map: np.ndarray,
    suitability: np.ndarray,
    class_values: list[int],
    target_demand: dict[int, int],
    exclusion_mask: np.ndarray | None = None,
    immutable_classes: set[int] | None = None,
    allowed_transitions: set[tuple[int, int]] | None = None,
    target_change_pixels: int | None = None,
    neighborhood_weight: float = 0.0,
    embedding_grid: np.ndarray | None = None,
    latent_neighborhood_weight: float = 0.0,
    balanced_swap_min_margin: float | None = None,
    balanced_swap_min_base_score: float | None = None,
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
    weight = float(neighborhood_weight)
    if weight < 0.0:
        raise DemandValidationError(f"neighborhood_weight must be non-negative: {weight}")
    latent_weight = float(latent_neighborhood_weight)
    if latent_weight < 0.0:
        raise DemandValidationError(f"latent_neighborhood_weight must be non-negative: {latent_weight}")
    margin_floor = None if balanced_swap_min_margin is None else float(balanced_swap_min_margin)
    if margin_floor is not None and margin_floor < 0.0:
        raise DemandValidationError(f"balanced_swap_min_margin must be non-negative: {margin_floor}")
    base_score_floor = None if balanced_swap_min_base_score is None else float(balanced_swap_min_base_score)
    if base_score_floor is not None and base_score_floor < 0.0:
        raise DemandValidationError(f"balanced_swap_min_base_score must be non-negative: {base_score_floor}")
    neighborhood = _neighborhood_affinity_cube(start, class_values) if weight > 0.0 else np.zeros_like(scores)
    if latent_weight > 0.0:
        if embedding_grid is None:
            raise DemandValidationError("embedding_grid is required when latent_neighborhood_weight is positive")
        latent_neighborhood = _latent_neighborhood_affinity_cube(start, embedding_grid, class_values)
    else:
        latent_neighborhood = np.zeros_like(scores)
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
        stay_score = _allocation_score(
            scores,
            neighborhood,
            weight,
            latent_neighborhood,
            latent_weight,
            int(row),
            int(col),
            from_cls,
            class_to_col,
        )
        best_change_score: float | None = None
        for to_cls in class_values:
            to_key = int(to_cls)
            if to_key == from_cls:
                continue
            if remaining.get(to_key, 0) <= 0:
                continue
            if not _is_allowed(from_cls, to_key, allowed_transitions):
                continue
            change_score = _allocation_score(
                scores,
                neighborhood,
                weight,
                latent_neighborhood,
                latent_weight,
                int(row),
                int(col),
                to_key,
                class_to_col,
            )
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
            score = _allocation_score(
                scores,
                neighborhood,
                weight,
                latent_neighborhood,
                latent_weight,
                int(row),
                int(col),
                to_key,
                class_to_col,
            )
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
                            "score": _allocation_score(
                                scores,
                                neighborhood,
                                weight,
                                latent_neighborhood,
                                latent_weight,
                                int(row),
                                int(col),
                                assigned_class,
                                class_to_col,
                            ),
                        }
                    )
            assigned[row, col] = True

    _add_balanced_change_swaps(
        simulated,
        start,
        scores,
        neighborhood,
        weight,
        latent_neighborhood,
        latent_weight,
        class_values,
        class_to_col,
        editable,
        selected,
        allowed_transitions,
        target_change_pixels,
        margin_floor,
        base_score_floor,
    )

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
