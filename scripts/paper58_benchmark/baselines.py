from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def persistence_prediction(start_map: np.ndarray) -> np.ndarray:
    return np.asarray(start_map).copy()


def spatial_shuffle_prediction(prediction: np.ndarray, seed: int = 20260617) -> np.ndarray:
    pred = np.asarray(prediction)
    rng = np.random.default_rng(seed)
    return rng.permutation(pred.ravel()).reshape(pred.shape)


def label_only_transition_prior(
    target_start: np.ndarray,
    training_pairs: list[tuple[np.ndarray, np.ndarray]],
    seed: int = 20260618,
) -> np.ndarray:
    target = np.asarray(target_start)
    predicted = target.copy()
    transitions: dict[int, dict[int, int]] = {}
    for train_start, train_end in training_pairs:
        if train_start.shape != train_end.shape:
            continue
        for start_cls, end_cls in zip(train_start.ravel(), train_end.ravel()):
            start_key = int(start_cls)
            end_key = int(end_cls)
            transitions.setdefault(start_key, {})
            transitions[start_key][end_key] = transitions[start_key].get(end_key, 0) + 1
    if not transitions:
        return predicted

    rng = np.random.default_rng(seed)
    flat_start = target.ravel()
    flat_pred = predicted.ravel()
    for start_cls in sorted(np.unique(flat_start)):
        class_counts = transitions.get(int(start_cls))
        if not class_counts:
            continue
        indices = np.flatnonzero(flat_start == start_cls)
        shuffled_indices = rng.permutation(indices)
        end_classes = sorted(class_counts)
        counts = np.array([class_counts[end_cls] for end_cls in end_classes], dtype=float)
        fractions = counts / counts.sum()
        allocation = np.floor(fractions * indices.size).astype(int)
        remainder = int(indices.size - allocation.sum())
        if remainder > 0:
            residual = fractions * indices.size - allocation
            for position in np.argsort(-residual)[:remainder]:
                allocation[position] += 1
        cursor = 0
        for end_cls, n_assign in zip(end_classes, allocation):
            selected = shuffled_indices[cursor : cursor + int(n_assign)]
            flat_pred[selected] = end_cls
            cursor += int(n_assign)
    return flat_pred.reshape(target.shape)


def leave_one_region_temporal_prior(
    target_area: str,
    target_start: np.ndarray,
    training_rows: list[dict],
    seed: int = 20260619,
) -> np.ndarray:
    target = np.asarray(target_start)
    rates = []
    changed_end_classes = []
    for row in training_rows:
        if str(row.get("area")) == target_area:
            continue
        start = np.asarray(row["start"])
        end = np.asarray(row["end"])
        if start.shape != end.shape:
            continue
        changed = end != start
        rates.append(float(np.mean(changed)))
        changed_end_classes.extend(int(value) for value in end[changed].ravel())
    if not rates or not changed_end_classes:
        return target.copy()

    rng = np.random.default_rng(seed)
    n_change = int(round(float(np.mean(rates)) * target.size))
    n_change = max(0, min(n_change, target.size))
    pred = target.copy().ravel()
    if n_change == 0:
        return pred.reshape(target.shape)
    indices = rng.choice(np.arange(target.size), size=n_change, replace=False)
    candidates = np.array(changed_end_classes, dtype=pred.dtype)
    for index in indices:
        alternatives = candidates[candidates != pred[index]]
        choices = alternatives if alternatives.size else candidates
        pred[index] = rng.choice(choices)
    return pred.reshape(target.shape)


def fit_linear_embedding_delta(
    train_start: np.ndarray,
    train_end: np.ndarray,
    test_start: np.ndarray,
    alpha: float = 1e-3,
) -> np.ndarray:
    train_start_arr = np.asarray(train_start, dtype=np.float32)
    train_end_arr = np.asarray(train_end, dtype=np.float32)
    test_start_arr = np.asarray(test_start, dtype=np.float32)
    if train_start_arr.shape != train_end_arr.shape:
        raise ValueError(f"Shape mismatch: train_start={train_start_arr.shape}, train_end={train_end_arr.shape}")
    feature_dim = train_start_arr.shape[-1]
    x_train = train_start_arr.reshape(-1, feature_dim)
    y_delta = (train_end_arr - train_start_arr).reshape(-1, feature_dim)
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_delta)
    x_test = test_start_arr.reshape(-1, feature_dim)
    delta = model.predict(x_test).reshape(test_start_arr.shape)
    return test_start_arr + delta.astype(np.float32, copy=False)
