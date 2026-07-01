from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper58_benchmark.apply_paper58_calibrated_transition_exactness_gate import (
    discover_calibration_cases,
)
from scripts.paper58_benchmark.apply_paper58_change_gate import class_aligned_neighborhood
from scripts.paper58_benchmark.sweep_paper58_change_gate import load_case_from_change_gate_dir


PREDICTION_RE = re.compile(r"^(?P<area>.+)_lulc_pred_(?P<start_year>\d{4})_(?P<end_year>\d{4})\.npy$")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _valid_mask(start_map: np.ndarray, prediction_map: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    if start.shape != prediction.shape:
        raise ValueError(f"shape mismatch: start={start.shape}, prediction={prediction.shape}")
    valid = (start != 0) & (prediction != 0) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != start.shape:
        raise ValueError(f"valid_mask shape {valid.shape} does not match start shape {start.shape}")
    return valid


def change_demand_features(
    start_map: np.ndarray,
    prediction_map: np.ndarray,
    selector_class: int = 5,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    valid = _valid_mask(start, prediction, valid_mask)
    total = int(np.count_nonzero(valid))
    if total <= 0:
        return np.zeros(5, dtype=np.float32)
    predicted_change = valid & (prediction != start)
    start_selector = valid & (start == int(selector_class))
    pred_selector = valid & (prediction == int(selector_class))
    start_values = start[valid].ravel()
    _, counts = np.unique(start_values, return_counts=True)
    probabilities = counts.astype(np.float64) / float(total)
    entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
    entropy_norm = entropy / max(float(np.log(max(len(counts), 2))), 1e-12)
    return np.array(
        [
            np.count_nonzero(predicted_change) / total,
            np.count_nonzero(start_selector) / total,
            np.count_nonzero(pred_selector) / total,
            entropy_norm,
            min(1.0, len(counts) / 10.0),
        ],
        dtype=np.float32,
    )


def fit_change_fraction_regression(
    feature_rows: np.ndarray,
    target_change_fractions: np.ndarray,
    ridge: float = 1e-3,
) -> dict[str, Any]:
    features = np.asarray(feature_rows, dtype=np.float64)
    targets = np.asarray(target_change_fractions, dtype=np.float64).reshape(-1)
    if features.ndim != 2:
        raise ValueError(f"feature_rows must be 2D, got {features.shape}")
    if targets.shape[0] != features.shape[0]:
        raise ValueError(f"target length {targets.shape[0]} does not match features {features.shape[0]}")
    if features.shape[0] == 0:
        raise ValueError("at least one calibration row is required")
    penalty = float(ridge)
    if penalty < 0.0:
        raise ValueError(f"ridge must be non-negative: {penalty}")
    x = np.concatenate([np.ones((features.shape[0], 1), dtype=np.float64), features], axis=1)
    if penalty == 0.0:
        weights = np.linalg.lstsq(x, targets, rcond=None)[0]
    else:
        regularizer = np.eye(x.shape[1], dtype=np.float64) * penalty
        regularizer[0, 0] = 0.0
        weights = np.linalg.solve(x.T @ x + regularizer, x.T @ targets)
    return {
        "weights": weights.astype(np.float64),
        "ridge": penalty,
        "n_features": int(features.shape[1]),
    }


def predict_change_fraction(
    model: dict[str, Any],
    features: np.ndarray,
    min_fraction: float = 0.0,
    max_fraction: float = 1.0,
) -> float:
    weights = np.asarray(model["weights"], dtype=np.float64)
    row = np.asarray(features, dtype=np.float64).reshape(-1)
    if weights.shape[0] != row.shape[0] + 1:
        raise ValueError(f"feature length {row.shape[0]} does not match model weights {weights.shape[0]}")
    lower = float(min_fraction)
    upper = float(max_fraction)
    if lower < 0.0 or upper > 1.0 or lower > upper:
        raise ValueError(f"invalid fraction bounds: min={lower}, max={upper}")
    raw = float(weights[0] + row @ weights[1:])
    return float(min(upper, max(lower, raw)))


def fit_change_ratio_demand_model(
    predicted_change_fractions: np.ndarray,
    target_change_fractions: np.ndarray,
    quantile: float = 0.25,
    multiplier: float = 1.0,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    predicted = np.asarray(predicted_change_fractions, dtype=np.float64).reshape(-1)
    targets = np.asarray(target_change_fractions, dtype=np.float64).reshape(-1)
    if predicted.shape[0] != targets.shape[0]:
        raise ValueError(f"target length {targets.shape[0]} does not match predicted {predicted.shape[0]}")
    if predicted.shape[0] == 0:
        raise ValueError("at least one calibration row is required")
    q = float(quantile)
    mult = float(multiplier)
    eps = float(epsilon)
    if q < 0.0 or q > 1.0:
        raise ValueError(f"quantile must be in [0, 1]: {q}")
    if mult < 0.0:
        raise ValueError(f"multiplier must be non-negative: {mult}")
    usable = predicted > eps
    if not np.any(usable):
        raise ValueError("at least one calibration row must have positive predicted change fraction")
    ratios = np.maximum(targets[usable], 0.0) / np.maximum(predicted[usable], eps)
    base_ratio = float(np.quantile(ratios, q))
    return {
        "base_ratio": base_ratio,
        "effective_ratio": float(base_ratio * mult),
        "quantile": q,
        "multiplier": mult,
        "n_rows": int(predicted.shape[0]),
        "n_usable_rows": int(np.count_nonzero(usable)),
    }


def predict_ratio_change_fraction(
    model: dict[str, Any],
    candidate_change_fraction: float,
    min_fraction: float = 0.0,
    max_fraction: float = 1.0,
) -> float:
    lower = float(min_fraction)
    upper = float(max_fraction)
    if lower < 0.0 or upper > 1.0 or lower > upper:
        raise ValueError(f"invalid fraction bounds: min={lower}, max={upper}")
    candidate = max(0.0, float(candidate_change_fraction))
    raw = candidate * float(model["effective_ratio"])
    return float(min(upper, max(lower, raw)))


def fit_adaptive_ratio_cap_model(
    predicted_change_fractions: np.ndarray,
    target_change_fractions: np.ndarray,
    high_candidate_threshold: float = 0.5,
    high_candidate_quantile: float = 0.5,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    predicted = np.asarray(predicted_change_fractions, dtype=np.float64).reshape(-1)
    targets = np.asarray(target_change_fractions, dtype=np.float64).reshape(-1)
    if predicted.shape[0] != targets.shape[0]:
        raise ValueError(f"target length {targets.shape[0]} does not match predicted {predicted.shape[0]}")
    if predicted.shape[0] == 0:
        raise ValueError("at least one calibration row is required")
    threshold = float(high_candidate_threshold)
    quantile = float(high_candidate_quantile)
    eps = float(epsilon)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"high_candidate_threshold must be in [0, 1]: {threshold}")
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError(f"high_candidate_quantile must be in [0, 1]: {quantile}")
    usable = predicted > eps
    if not np.any(usable):
        raise ValueError("at least one calibration row must have positive predicted change fraction")
    high = usable & (predicted >= threshold)
    ratio_source = high if np.any(high) else usable
    ratios = np.maximum(targets[ratio_source], 0.0) / np.maximum(predicted[ratio_source], eps)
    return {
        "high_candidate_threshold": threshold,
        "high_candidate_ratio": float(np.quantile(ratios, quantile)),
        "high_candidate_quantile": quantile,
        "n_rows": int(predicted.shape[0]),
        "n_high_candidate_rows": int(np.count_nonzero(high)),
        "n_usable_rows": int(np.count_nonzero(usable)),
    }


def predict_ratio_change_fraction_v2(
    ratio_model: dict[str, Any],
    cap_model: dict[str, Any] | None,
    candidate_change_fraction: float,
    min_fraction: float = 0.0,
    max_fraction: float = 1.0,
) -> float:
    candidate = max(0.0, float(candidate_change_fraction))
    raw = candidate * float(ratio_model["effective_ratio"])
    if cap_model is not None and candidate >= float(cap_model["high_candidate_threshold"]):
        raw = min(raw, candidate * float(cap_model["high_candidate_ratio"]))
    lower = float(min_fraction)
    upper = float(max_fraction)
    if lower < 0.0 or upper > 1.0 or lower > upper:
        raise ValueError(f"invalid fraction bounds: min={lower}, max={upper}")
    return float(min(upper, max(lower, raw)))


def ratio_model_for_region_scale(
    model: dict[str, Any],
    valid_pixels: int,
    large_region_valid_pixel_threshold: int | None,
    large_region_ratio_multiplier: float | None,
) -> tuple[dict[str, Any], float, bool]:
    base_multiplier = float(model["multiplier"])
    threshold = large_region_valid_pixel_threshold
    large_multiplier = large_region_ratio_multiplier
    if threshold is None or large_multiplier is None or int(valid_pixels) < int(threshold):
        return model, base_multiplier, False
    scaled_model = dict(model)
    selected_multiplier = float(large_multiplier)
    scaled_model["multiplier"] = selected_multiplier
    scaled_model["effective_ratio"] = float(scaled_model["base_ratio"] * selected_multiplier)
    return scaled_model, selected_multiplier, True


def fit_transition_reliability_weights(
    calibration_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    smoothing: float = 1.0,
    min_weight: float = 0.2,
) -> dict[str, Any]:
    smooth = float(smoothing)
    floor = float(min_weight)
    if smooth < 0.0:
        raise ValueError(f"smoothing must be non-negative: {smooth}")
    if floor < 0.0 or floor > 1.0:
        raise ValueError(f"min_weight must be in [0, 1]: {floor}")
    if not calibration_rows:
        raise ValueError("at least one calibration row is required")

    predicted_by_pair: dict[tuple[int, int], int] = {}
    hits_by_pair: dict[tuple[int, int], int] = {}
    total_predicted = 0
    total_hits = 0
    for index, (start_map, prediction_map, end_map) in enumerate(calibration_rows):
        start = np.asarray(start_map)
        prediction = np.asarray(prediction_map)
        end = np.asarray(end_map)
        if start.shape != prediction.shape or start.shape != end.shape:
            raise ValueError(
                f"calibration row {index} shape mismatch: "
                f"start={start.shape}, prediction={prediction.shape}, end={end.shape}"
            )
        valid = (start != 0) & (prediction != 0) & (end != 0)
        candidates = valid & (prediction != start)
        hits = candidates & (prediction == end)
        total_predicted += int(np.count_nonzero(candidates))
        total_hits += int(np.count_nonzero(hits))
        for from_cls, to_cls in zip(start[candidates].ravel(), prediction[candidates].ravel(), strict=False):
            key = (int(from_cls), int(to_cls))
            predicted_by_pair[key] = predicted_by_pair.get(key, 0) + 1
        for from_cls, to_cls in zip(start[hits].ravel(), prediction[hits].ravel(), strict=False):
            key = (int(from_cls), int(to_cls))
            hits_by_pair[key] = hits_by_pair.get(key, 0) + 1

    default_precision = float(total_hits / total_predicted) if total_predicted else 0.0
    default_weight = floor + (1.0 - floor) * default_precision
    weights: dict[tuple[int, int], float] = {}
    pair_stats: dict[tuple[int, int], dict[str, Any]] = {}
    for key, predicted_count in sorted(predicted_by_pair.items()):
        hit_count = hits_by_pair.get(key, 0)
        precision = float((hit_count + smooth * default_precision) / (predicted_count + smooth))
        weight = floor + (1.0 - floor) * precision
        weights[key] = float(min(1.0, max(floor, weight)))
        pair_stats[key] = {
            "predicted": int(predicted_count),
            "hits": int(hit_count),
            "smoothed_precision": precision,
            "weight": weights[key],
        }
    return {
        "weights": weights,
        "default_weight": float(default_weight),
        "pair_stats": pair_stats,
        "smoothing": smooth,
        "min_weight": floor,
        "total_predicted": int(total_predicted),
        "total_hits": int(total_hits),
    }


def transition_reliability_score(model: dict[str, Any], from_class: int, to_class: int) -> float:
    weights = model.get("weights", {})
    return float(weights.get((int(from_class), int(to_class)), model["default_weight"]))


def transition_reliability_map(
    start_map: np.ndarray,
    prediction_map: np.ndarray,
    model: dict[str, Any],
) -> np.ndarray:
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    if start.shape != prediction.shape:
        raise ValueError(f"shape mismatch: start={start.shape}, prediction={prediction.shape}")
    reliability = np.full(start.shape, float(model["default_weight"]), dtype=np.float32)
    for from_cls, to_cls in sorted(model.get("weights", {})):
        mask = (start == int(from_cls)) & (prediction == int(to_cls))
        reliability[mask] = transition_reliability_score(model, int(from_cls), int(to_cls))
    return reliability


def _class_neighborhood_fraction_window(label_map: np.ndarray, class_values: list[int], window_size: int) -> dict[int, np.ndarray]:
    labels = np.asarray(label_map)
    size = int(window_size)
    if size < 3 or size % 2 == 0:
        raise ValueError(f"window_size must be an odd integer >= 3: {window_size}")
    radius = size // 2
    fractions: dict[int, np.ndarray] = {}
    height, width = labels.shape
    for cls in class_values:
        mask = (labels == int(cls)).astype(np.float32)
        support = np.zeros(labels.shape, dtype=np.float32)
        counts = np.zeros(labels.shape, dtype=np.float32)
        for row_offset in range(-radius, radius + 1):
            for col_offset in range(-radius, radius + 1):
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
                support[dst_row_start:dst_row_end, dst_col_start:dst_col_end] += mask[
                    src_row_start:src_row_end,
                    src_col_start:src_col_end,
                ]
                counts[dst_row_start:dst_row_end, dst_col_start:dst_col_end] += 1.0
        fractions[int(cls)] = np.divide(support, counts, out=np.zeros_like(support), where=counts > 0.0)
    return fractions


def multi_scale_class_aligned_neighborhood(
    start_map: np.ndarray,
    prediction_map: np.ndarray,
    classes: list[int],
    window_sizes: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    if start.shape != prediction.shape:
        raise ValueError(f"shape mismatch: start={start.shape}, prediction={prediction.shape}")
    windows = [3] if window_sizes is None else [int(size) for size in window_sizes]
    if not windows:
        raise ValueError("window_sizes must be non-empty")
    target = np.zeros(start.shape, dtype=np.float32)
    source = np.zeros(start.shape, dtype=np.float32)
    for window in windows:
        neighborhoods = _class_neighborhood_fraction_window(start, classes, window)
        target_scale = np.zeros(start.shape, dtype=np.float32)
        source_scale = np.zeros(start.shape, dtype=np.float32)
        for cls, support in neighborhoods.items():
            target_scale[prediction == int(cls)] = support[prediction == int(cls)]
            source_scale[start == int(cls)] = support[start == int(cls)]
        target += target_scale
        source += source_scale
    divisor = float(len(windows))
    return target / divisor, source / divisor


def filter_calibration_cases_by_area(
    calibration_cases: list[Any],
    calibration_areas: list[str] | None,
) -> tuple[list[Any], list[dict[str, str]]]:
    if calibration_areas is None:
        return list(calibration_cases), []
    allowed = {str(area) for area in calibration_areas}
    selected: list[Any] = []
    skipped: list[dict[str, str]] = []
    for case in calibration_cases:
        area = str(case.area)
        if area in allowed:
            selected.append(case)
        else:
            skipped.append({"area": area, "reason": "not_in_calibration_area_whitelist"})
    return selected, skipped


def apply_demand_calibrated_spatial_gate(
    start_map: np.ndarray,
    prediction_map: np.ndarray,
    score_map: np.ndarray,
    target_change_fraction: float,
    valid_mask: np.ndarray | None = None,
    target_neighborhood_weight: float = 0.0,
    source_neighborhood_penalty: float = 0.0,
    transition_reliability_model: dict[str, Any] | None = None,
    transition_weight_strength: float = 0.0,
    neighborhood_window_sizes: list[int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    score = np.asarray(score_map, dtype=np.float32)
    if start.shape != prediction.shape or start.shape != score.shape:
        raise ValueError(f"shape mismatch: start={start.shape}, prediction={prediction.shape}, score={score.shape}")
    valid = _valid_mask(start, prediction, valid_mask)
    target_fraction = float(target_change_fraction)
    if target_fraction < 0.0 or target_fraction > 1.0:
        raise ValueError(f"target_change_fraction must be in [0, 1]: {target_fraction}")
    target_weight = float(target_neighborhood_weight)
    source_penalty = float(source_neighborhood_penalty)
    reliability_strength = float(transition_weight_strength)
    if target_weight < 0.0 or source_penalty < 0.0:
        raise ValueError(
            "target_neighborhood_weight and source_neighborhood_penalty must be non-negative: "
            f"target={target_weight}, source={source_penalty}"
        )
    if reliability_strength < 0.0:
        raise ValueError(f"transition_weight_strength must be non-negative: {reliability_strength}")

    candidates = valid & (prediction != start)
    candidate_indices = np.flatnonzero(candidates.ravel())
    valid_pixels = int(np.count_nonzero(valid))
    target_pixels = min(int(round(valid_pixels * target_fraction)), int(candidate_indices.size))
    gated = start.copy()
    kept_indices = np.array([], dtype=np.int64)
    if target_pixels > 0 and candidate_indices.size > 0:
        rank_score = score.copy()
        if target_weight > 0.0 or source_penalty > 0.0:
            classes = sorted({int(value) for value in np.unique(start[valid])} | {int(value) for value in np.unique(prediction[valid])})
            if neighborhood_window_sizes is None:
                target_support, source_support = class_aligned_neighborhood(start, prediction, classes)
            else:
                target_support, source_support = multi_scale_class_aligned_neighborhood(
                    start,
                    prediction,
                    classes,
                    window_sizes=neighborhood_window_sizes,
                )
            rank_score = rank_score + target_weight * target_support - source_penalty * source_support
        if transition_reliability_model is not None and reliability_strength > 0.0:
            reliability = transition_reliability_map(start, prediction, transition_reliability_model)
            rank_score = rank_score + reliability_strength * reliability
        order = candidate_indices[np.argsort(rank_score.ravel()[candidate_indices])[::-1]]
        kept_indices = order[:target_pixels]
        gated.ravel()[kept_indices] = prediction.ravel()[kept_indices]
    diagnostics = {
        "valid_pixels": valid_pixels,
        "candidate_change_pixels": int(candidate_indices.size),
        "target_change_fraction": target_fraction,
        "target_change_pixels": int(target_pixels),
        "kept_change_pixels": int(kept_indices.size),
        "target_neighborhood_weight": target_weight,
        "source_neighborhood_penalty": source_penalty,
        "transition_weight_strength": reliability_strength,
        "neighborhood_window_sizes": None if neighborhood_window_sizes is None else [int(size) for size in neighborhood_window_sizes],
    }
    return gated.astype(prediction.dtype, copy=False), diagnostics


def _true_change_fraction(start_map: np.ndarray, end_map: np.ndarray, prediction_map: np.ndarray) -> float:
    start = np.asarray(start_map)
    end = np.asarray(end_map)
    prediction = np.asarray(prediction_map)
    if start.shape != end.shape or start.shape != prediction.shape:
        raise ValueError(f"shape mismatch: start={start.shape}, end={end.shape}, prediction={prediction.shape}")
    valid = (start != 0) & (end != 0) & (prediction != 0)
    total = int(np.count_nonzero(valid))
    return float(np.count_nonzero(valid & (end != start)) / total) if total else 0.0


def _load_source_prediction(source_prediction_dir: Path, area: str, start_year: int, end_year: int) -> np.ndarray:
    path = Path(source_prediction_dir) / f"{area}_lulc_pred_{int(start_year)}_{int(end_year)}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing source prediction for {area}: {path}")
    return np.load(path).astype(np.int32, copy=False)


def run_spatial_demand_allocation(
    source_prediction_dir: Path,
    change_gate_dirs: list[Path],
    calibration_label_dir: Path,
    calibration_prediction_dir: Path,
    output_dir: Path,
    selector_class: int = 5,
    ridge: float = 1e-3,
    min_fraction: float = 0.0,
    max_fraction: float = 0.35,
    target_neighborhood_weight: float = 0.75,
    source_neighborhood_penalty: float = 0.25,
    calibration_areas: list[str] | None = None,
    demand_strategy: str = "regression",
    ratio_quantile: float = 0.25,
    ratio_multiplier: float = 1.0,
    enable_transition_reliability: bool = False,
    transition_reliability_strength: float = 0.0,
    neighborhood_window_sizes: list[int] | None = None,
    enable_adaptive_ratio_cap: bool = False,
    high_candidate_threshold: float = 0.5,
    high_candidate_quantile: float = 0.5,
    large_region_valid_pixel_threshold: int | None = None,
    large_region_ratio_multiplier: float | None = None,
) -> dict[str, Any]:
    large_threshold = None if large_region_valid_pixel_threshold is None else int(large_region_valid_pixel_threshold)
    large_multiplier = None if large_region_ratio_multiplier is None else float(large_region_ratio_multiplier)
    if (large_threshold is None) != (large_multiplier is None):
        raise ValueError(
            "large_region_valid_pixel_threshold and large_region_ratio_multiplier must be provided together"
        )
    if large_threshold is not None and large_threshold < 1:
        raise ValueError(f"large_region_valid_pixel_threshold must be positive: {large_threshold}")
    if large_multiplier is not None and large_multiplier < 0.0:
        raise ValueError(f"large_region_ratio_multiplier must be non-negative: {large_multiplier}")

    calibration_cases, skipped_calibration = discover_calibration_cases(
        Path(calibration_label_dir),
        Path(calibration_prediction_dir),
    )
    calibration_cases, whitelist_skipped = filter_calibration_cases_by_area(calibration_cases, calibration_areas)
    skipped_calibration = [*skipped_calibration, *whitelist_skipped]
    if not calibration_cases:
        raise ValueError("no usable calibration cases were discovered")
    calibration_features = np.stack(
        [
            change_demand_features(case.start_map, case.prediction_map, selector_class=selector_class)
            for case in calibration_cases
        ],
        axis=0,
    )
    calibration_targets = np.array(
        [_true_change_fraction(case.start_map, case.end_map, case.prediction_map) for case in calibration_cases],
        dtype=np.float32,
    )
    strategy = str(demand_strategy)
    if strategy == "regression":
        model = fit_change_fraction_regression(calibration_features, calibration_targets, ridge=ridge)
    elif strategy == "ratio":
        model = fit_change_ratio_demand_model(
            calibration_features[:, 0],
            calibration_targets,
            quantile=ratio_quantile,
            multiplier=ratio_multiplier,
        )
    else:
        raise ValueError(f"unsupported demand_strategy: {strategy}")
    if strategy != "ratio" and large_threshold is not None:
        raise ValueError("large-region ratio multiplier is only supported with demand_strategy=ratio")
    calibration_transition_rows = [
        (case.start_map, case.prediction_map, case.end_map)
        for case in calibration_cases
    ]
    transition_reliability_model = (
        fit_transition_reliability_weights(calibration_transition_rows)
        if bool(enable_transition_reliability)
        else None
    )
    adaptive_ratio_cap_model = (
        fit_adaptive_ratio_cap_model(
            calibration_features[:, 0],
            calibration_targets,
            high_candidate_threshold=high_candidate_threshold,
            high_candidate_quantile=high_candidate_quantile,
        )
        if bool(enable_adaptive_ratio_cap) and strategy == "ratio"
        else None
    )

    output = Path(output_dir)
    predictions_dir = output / "predictions"
    diagnostics_dir = output / "diagnostics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for change_gate_dir in change_gate_dirs:
        case = load_case_from_change_gate_dir(Path(change_gate_dir))
        source_prediction = _load_source_prediction(Path(source_prediction_dir), case.area, case.start_year, case.end_year)
        if source_prediction.shape != case.start_map.shape:
            raise ValueError(
                f"shape mismatch for {case.area}: prediction={source_prediction.shape}, start={case.start_map.shape}"
            )
        valid = (case.start_map != 0) & (source_prediction != 0)
        valid_pixels = int(np.count_nonzero(valid))
        features = change_demand_features(case.start_map, source_prediction, selector_class=selector_class)
        ratio_multiplier_for_case = None
        effective_ratio_for_case = None
        large_region_ratio_multiplier_applied = False
        if strategy == "regression":
            target_fraction = predict_change_fraction(
                model,
                features,
                min_fraction=min_fraction,
                max_fraction=max_fraction,
            )
        else:
            case_ratio_model, ratio_multiplier_for_case, large_region_ratio_multiplier_applied = ratio_model_for_region_scale(
                model,
                valid_pixels=valid_pixels,
                large_region_valid_pixel_threshold=large_threshold,
                large_region_ratio_multiplier=large_multiplier,
            )
            effective_ratio_for_case = float(case_ratio_model["effective_ratio"])
            if adaptive_ratio_cap_model is None:
                target_fraction = predict_ratio_change_fraction(
                    case_ratio_model,
                    float(features[0]),
                    min_fraction=min_fraction,
                    max_fraction=max_fraction,
                )
            else:
                target_fraction = predict_ratio_change_fraction_v2(
                    case_ratio_model,
                    adaptive_ratio_cap_model,
                    float(features[0]),
                    min_fraction=min_fraction,
                    max_fraction=max_fraction,
                )
        gated, gate_diagnostics = apply_demand_calibrated_spatial_gate(
            case.start_map,
            source_prediction,
            case.score_map,
            target_change_fraction=target_fraction,
            valid_mask=valid,
            target_neighborhood_weight=target_neighborhood_weight,
            source_neighborhood_penalty=source_neighborhood_penalty,
            transition_reliability_model=transition_reliability_model,
            transition_weight_strength=transition_reliability_strength,
            neighborhood_window_sizes=neighborhood_window_sizes,
        )
        prediction_path = predictions_dir / f"{case.area}_lulc_pred_{case.start_year}_{case.end_year}.npy"
        np.save(prediction_path, gated)
        diagnostic_payload = {
            "area": case.area,
            "start_year": case.start_year,
            "end_year": case.end_year,
            "source_prediction_dir": Path(source_prediction_dir),
            "source_change_gate_dir": Path(change_gate_dir),
            "features": features,
            "predicted_target_change_fraction": target_fraction,
            "gate_diagnostics": gate_diagnostics,
            "ratio_multiplier_for_case": ratio_multiplier_for_case,
            "effective_ratio_for_case": effective_ratio_for_case,
            "large_region_valid_pixel_threshold": large_threshold,
            "large_region_ratio_multiplier_applied": large_region_ratio_multiplier_applied,
            "adaptive_ratio_cap_enabled": adaptive_ratio_cap_model is not None,
            "transition_reliability_enabled": transition_reliability_model is not None,
            "output_prediction": prediction_path,
        }
        _write_json(diagnostics_dir / f"{case.area}_spatial_demand_allocation.json", diagnostic_payload)
        cases.append(diagnostic_payload)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paper58_spatial_demand_allocation",
        "source_prediction_dir": Path(source_prediction_dir),
        "change_gate_dirs": [Path(path) for path in change_gate_dirs],
        "calibration_label_dir": Path(calibration_label_dir),
        "calibration_prediction_dir": Path(calibration_prediction_dir),
        "calibration_areas": None if calibration_areas is None else [str(area) for area in calibration_areas],
        "usable_calibration_area_count": len(calibration_cases),
        "skipped_calibration_cases": skipped_calibration,
        "parameters": {
            "selector_class": int(selector_class),
            "ridge": float(ridge),
            "min_fraction": float(min_fraction),
            "max_fraction": float(max_fraction),
            "target_neighborhood_weight": float(target_neighborhood_weight),
            "source_neighborhood_penalty": float(source_neighborhood_penalty),
            "demand_strategy": strategy,
            "ratio_quantile": float(ratio_quantile),
            "ratio_multiplier": float(ratio_multiplier),
            "enable_transition_reliability": bool(enable_transition_reliability),
            "transition_reliability_strength": float(transition_reliability_strength),
            "neighborhood_window_sizes": None if neighborhood_window_sizes is None else [int(size) for size in neighborhood_window_sizes],
            "enable_adaptive_ratio_cap": bool(enable_adaptive_ratio_cap),
            "high_candidate_threshold": float(high_candidate_threshold),
            "high_candidate_quantile": float(high_candidate_quantile),
            "large_region_valid_pixel_threshold": large_threshold,
            "large_region_ratio_multiplier": large_multiplier,
        },
        "demand_model": model,
        "transition_reliability_model": transition_reliability_model,
        "adaptive_ratio_cap_model": adaptive_ratio_cap_model,
        "calibration_target_change_fraction_mean": float(np.mean(calibration_targets)),
        "calibration_predicted_change_fraction_mean": float(np.mean(calibration_features[:, 0])),
        "n_cases": len(cases),
        "cases": cases,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply Paper58 demand-calibrated spatial allocation gate.")
    parser.add_argument("--source-prediction-dir", type=Path, required=True)
    parser.add_argument("--change-gate-dir", action="append", type=Path, required=True)
    parser.add_argument("--calibration-label-dir", type=Path, required=True)
    parser.add_argument("--calibration-prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selector-class", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--min-fraction", type=float, default=0.0)
    parser.add_argument("--max-fraction", type=float, default=0.35)
    parser.add_argument("--target-neighborhood-weight", type=float, default=0.75)
    parser.add_argument("--source-neighborhood-penalty", type=float, default=0.25)
    parser.add_argument("--calibration-area", action="append", default=None)
    parser.add_argument("--demand-strategy", choices=["regression", "ratio"], default="regression")
    parser.add_argument("--ratio-quantile", type=float, default=0.25)
    parser.add_argument("--ratio-multiplier", type=float, default=1.0)
    parser.add_argument("--enable-transition-reliability", action="store_true")
    parser.add_argument("--transition-reliability-strength", type=float, default=0.0)
    parser.add_argument("--neighborhood-window-size", action="append", type=int, default=None)
    parser.add_argument("--enable-adaptive-ratio-cap", action="store_true")
    parser.add_argument("--high-candidate-threshold", type=float, default=0.5)
    parser.add_argument("--high-candidate-quantile", type=float, default=0.5)
    parser.add_argument("--large-region-valid-pixel-threshold", type=int, default=None)
    parser.add_argument("--large-region-ratio-multiplier", type=float, default=None)
    args = parser.parse_args(argv)
    manifest = run_spatial_demand_allocation(
        source_prediction_dir=args.source_prediction_dir,
        change_gate_dirs=args.change_gate_dir,
        calibration_label_dir=args.calibration_label_dir,
        calibration_prediction_dir=args.calibration_prediction_dir,
        output_dir=args.output_dir,
        selector_class=args.selector_class,
        ridge=args.ridge,
        min_fraction=args.min_fraction,
        max_fraction=args.max_fraction,
        target_neighborhood_weight=args.target_neighborhood_weight,
        source_neighborhood_penalty=args.source_neighborhood_penalty,
        calibration_areas=args.calibration_area,
        demand_strategy=args.demand_strategy,
        ratio_quantile=args.ratio_quantile,
        ratio_multiplier=args.ratio_multiplier,
        enable_transition_reliability=args.enable_transition_reliability,
        transition_reliability_strength=args.transition_reliability_strength,
        neighborhood_window_sizes=args.neighborhood_window_size,
        enable_adaptive_ratio_cap=args.enable_adaptive_ratio_cap,
        high_candidate_threshold=args.high_candidate_threshold,
        high_candidate_quantile=args.high_candidate_quantile,
        large_region_valid_pixel_threshold=args.large_region_valid_pixel_threshold,
        large_region_ratio_multiplier=args.large_region_ratio_multiplier,
    )
    print(
        "Paper58 spatial demand allocation complete: "
        f"calibration={manifest['usable_calibration_area_count']}, "
        f"cases={manifest['n_cases']}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
