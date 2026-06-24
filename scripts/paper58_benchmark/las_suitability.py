from __future__ import annotations

import numpy as np


def class_values_from_maps(*maps: np.ndarray) -> list[int]:
    values: set[int] = set()
    for arr in maps:
        for value in np.unique(np.asarray(arr)):
            values.add(int(value))
    return sorted(values)


def one_hot_probability_cube(
    labels: np.ndarray,
    class_values: list[int],
    confidence: float = 0.95,
    floor: float = 0.01,
) -> np.ndarray:
    label_arr = np.asarray(labels)
    if not class_values:
        raise ValueError("class_values must be non-empty")
    if len(class_values) == 1:
        return np.ones(label_arr.shape + (1,), dtype=np.float32)
    cube = np.full(label_arr.shape + (len(class_values),), float(floor), dtype=np.float32)
    class_to_col = {int(cls): index for index, cls in enumerate(class_values)}
    for cls, col in class_to_col.items():
        cube[label_arr == cls, col] = float(confidence)
    return cube


def transition_prior_from_pairs(
    training_pairs: list[tuple[np.ndarray, np.ndarray]],
    class_values: list[int],
) -> dict[tuple[int, int], float]:
    counts: dict[int, dict[int, int]] = {}
    for start, end in training_pairs:
        start_arr = np.asarray(start)
        end_arr = np.asarray(end)
        if start_arr.shape != end_arr.shape:
            continue
        for from_cls, to_cls in zip(start_arr.ravel(), end_arr.ravel()):
            from_key = int(from_cls)
            to_key = int(to_cls)
            counts.setdefault(from_key, {})
            counts[from_key][to_key] = counts[from_key].get(to_key, 0) + 1

    prior: dict[tuple[int, int], float] = {}
    for from_cls in class_values:
        row = counts.get(int(from_cls), {})
        total = int(sum(row.values()))
        if total == 0:
            continue
        for to_cls in class_values:
            prior[(int(from_cls), int(to_cls))] = row.get(int(to_cls), 0) / total
    return prior


def _embedding_change_pressure(
    embedding_start: np.ndarray | None,
    embedding_forecast: np.ndarray | None,
    target_shape: tuple[int, int],
) -> np.ndarray:
    if embedding_start is None or embedding_forecast is None:
        return np.zeros(target_shape, dtype=np.float32)
    start = np.asarray(embedding_start, dtype=np.float32)
    forecast = np.asarray(embedding_forecast, dtype=np.float32)
    if start.shape != forecast.shape:
        raise ValueError(f"embedding shape mismatch: start={start.shape}, forecast={forecast.shape}")
    if start.shape[:2] != target_shape:
        raise ValueError(f"embedding grid shape {start.shape[:2]} does not match target shape {target_shape}")
    pressure = np.linalg.norm(forecast - start, axis=-1)
    max_value = float(np.max(pressure)) if pressure.size else 0.0
    if max_value <= 0.0:
        return np.zeros(target_shape, dtype=np.float32)
    return (pressure / max_value).astype(np.float32)


def build_transition_suitability(
    start_map: np.ndarray,
    class_values: list[int],
    forecast_probs: np.ndarray,
    start_probs: np.ndarray | None = None,
    embedding_start: np.ndarray | None = None,
    embedding_forecast: np.ndarray | None = None,
    allowed_transitions: set[tuple[int, int]] | None = None,
    transition_prior: dict[tuple[int, int], float] | None = None,
    forecast_prob_weight: float = 1.0,
    probability_gain_weight: float = 0.5,
    change_pressure_weight: float = 0.25,
    transition_prior_weight: float = 0.25,
) -> np.ndarray:
    start = np.asarray(start_map)
    forecast = np.asarray(forecast_probs, dtype=np.float32)
    if forecast.shape != start.shape + (len(class_values),):
        raise ValueError(
            f"forecast_probs shape {forecast.shape} does not match start/class shape "
            f"{start.shape + (len(class_values),)}"
        )
    if start_probs is None:
        start_prob_arr = np.zeros_like(forecast)
    else:
        start_prob_arr = np.asarray(start_probs, dtype=np.float32)
        if start_prob_arr.shape != forecast.shape:
            raise ValueError(
                f"start_probs shape {start_prob_arr.shape} does not match forecast_probs shape {forecast.shape}"
            )

    pressure = _embedding_change_pressure(embedding_start, embedding_forecast, start.shape)
    class_to_col = {int(cls): index for index, cls in enumerate(class_values)}
    suitability = np.zeros_like(forecast, dtype=np.float32)
    prior = transition_prior or {}

    for from_cls in np.unique(start):
        from_key = int(from_cls)
        from_mask = start == from_key
        for to_cls in class_values:
            to_key = int(to_cls)
            if from_key == to_key:
                continue
            if allowed_transitions is not None and (from_key, to_key) not in allowed_transitions:
                continue
            col = class_to_col[to_key]
            probability_gain = np.maximum(forecast[..., col] - start_prob_arr[..., col], 0.0)
            base_score = (
                forecast_prob_weight * forecast[..., col]
                + probability_gain_weight * probability_gain
                + change_pressure_weight * pressure
                + transition_prior_weight * float(prior.get((from_key, to_key), 0.0))
            )
            suitability[from_mask, col] = base_score[from_mask]
    return np.maximum(suitability, 0.0).astype(np.float32)
