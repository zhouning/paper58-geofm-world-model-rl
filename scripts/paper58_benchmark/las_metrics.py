from __future__ import annotations

import numpy as np

from scripts.paper58_benchmark.evaluate_benchmark import binary_change_metrics


def _validated_maps(*maps: np.ndarray) -> list[np.ndarray]:
    arrays = [np.asarray(map_value) for map_value in maps]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"shape mismatch: {sorted(shapes)}")
    return arrays


def figure_of_merit(start_map: np.ndarray, true_map: np.ndarray, pred_map: np.ndarray) -> float:
    start, true, pred = _validated_maps(start_map, true_map, pred_map)
    true_change = true != start
    pred_change = pred != start
    union = true_change | pred_change
    if not np.any(union):
        return 1.0
    intersection = true_change & (pred == true)
    return float(np.count_nonzero(intersection) / np.count_nonzero(union))


def transition_accuracy(start_map: np.ndarray, true_map: np.ndarray, pred_map: np.ndarray) -> float:
    start, true, pred = _validated_maps(start_map, true_map, pred_map)
    changed = true != start
    if not np.any(changed):
        return 1.0
    correct = (pred == true) & changed
    return float(np.count_nonzero(correct) / np.count_nonzero(changed))


def quantity_disagreement(true_map: np.ndarray, pred_map: np.ndarray) -> float:
    true, pred = _validated_maps(true_map, pred_map)
    total = int(true.size)
    if total == 0:
        return 0.0
    classes = sorted({int(value) for value in np.unique(true)} | {int(value) for value in np.unique(pred)})
    difference = 0
    for cls in classes:
        difference += abs(int(np.count_nonzero(true == cls)) - int(np.count_nonzero(pred == cls)))
    return float(0.5 * difference / total)


def allocation_disagreement(true_map: np.ndarray, pred_map: np.ndarray) -> float:
    true, pred = _validated_maps(true_map, pred_map)
    total = int(true.size)
    if total == 0:
        return 0.0
    total_disagreement = 1.0 - float(np.count_nonzero(true == pred) / total)
    return float(max(0.0, total_disagreement - quantity_disagreement(true, pred)))


def method_metric_row(
    method: str,
    area: str,
    tier: str,
    stratum: str,
    start_map: np.ndarray,
    true_map: np.ndarray,
    pred_map: np.ndarray,
) -> dict[str, int | float | str]:
    start, true, pred = _validated_maps(start_map, true_map, pred_map)
    change = binary_change_metrics(true != start, pred != start)
    return {
        "method": method,
        "area": area,
        "tier": tier,
        "stratum": stratum,
        "n_pixels": int(start.size),
        "true_change_pixels": int(np.count_nonzero(true != start)),
        "pred_change_pixels": int(np.count_nonzero(pred != start)),
        "change_precision": change["precision"],
        "change_recall": change["recall"],
        "change_f1": change["f1"],
        "fom": figure_of_merit(start, true, pred),
        "transition_accuracy": transition_accuracy(start, true, pred),
        "quantity_disagreement": quantity_disagreement(true, pred),
        "allocation_disagreement": allocation_disagreement(true, pred),
    }
