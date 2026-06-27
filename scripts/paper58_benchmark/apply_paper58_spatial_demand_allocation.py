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
    if target_weight < 0.0 or source_penalty < 0.0:
        raise ValueError(
            "target_neighborhood_weight and source_neighborhood_penalty must be non-negative: "
            f"target={target_weight}, source={source_penalty}"
        )

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
            target_support, source_support = class_aligned_neighborhood(start, prediction, classes)
            rank_score = rank_score + target_weight * target_support - source_penalty * source_support
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
) -> dict[str, Any]:
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
        features = change_demand_features(case.start_map, source_prediction, selector_class=selector_class)
        if strategy == "regression":
            target_fraction = predict_change_fraction(
                model,
                features,
                min_fraction=min_fraction,
                max_fraction=max_fraction,
            )
        else:
            target_fraction = predict_ratio_change_fraction(
                model,
                float(features[0]),
                min_fraction=min_fraction,
                max_fraction=max_fraction,
            )
        valid = (case.start_map != 0) & (source_prediction != 0)
        gated, gate_diagnostics = apply_demand_calibrated_spatial_gate(
            case.start_map,
            source_prediction,
            case.score_map,
            target_change_fraction=target_fraction,
            valid_mask=valid,
            target_neighborhood_weight=target_neighborhood_weight,
            source_neighborhood_penalty=source_neighborhood_penalty,
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
        },
        "demand_model": model,
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
    )
    print(
        "Paper58 spatial demand allocation complete: "
        f"calibration={manifest['usable_calibration_area_count']}, "
        f"cases={manifest['n_cases']}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
