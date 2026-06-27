from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper58_benchmark.apply_paper58_calibrated_transition_exactness_gate import (
    _filter_calibration_cases,
    _parse_exclude_terms,
    discover_calibration_cases,
)
from scripts.paper58_benchmark.apply_paper58_change_gate import class_aligned_neighborhood
from scripts.paper58_benchmark.apply_paper58_transition_reliability_gate import load_case_from_change_gate_dir


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def estimate_class_count_ratios(
    calibration_cases: list[Any],
    class_values: list[int],
    smoothing: float = 0.0,
) -> tuple[dict[int, float], dict[str, Any]]:
    smooth = float(smoothing)
    if smooth < 0.0:
        raise ValueError(f"smoothing must be non-negative: {smooth}")
    classes = [int(cls) for cls in class_values]
    if not classes:
        raise ValueError("class_values must be non-empty")
    predicted_counts = {cls: 0 for cls in classes}
    end_counts = {cls: 0 for cls in classes}
    for index, case in enumerate(calibration_cases):
        start = np.asarray(case.start_map)
        end = np.asarray(case.end_map)
        prediction = np.asarray(case.prediction_map)
        if start.shape != end.shape or start.shape != prediction.shape:
            raise ValueError(
                f"calibration case {index} shape mismatch: "
                f"start={start.shape}, end={end.shape}, prediction={prediction.shape}"
            )
        valid = (start != 0) & (end != 0) & (prediction != 0)
        for cls in classes:
            predicted_counts[cls] += int(np.count_nonzero(valid & (prediction == cls)))
            end_counts[cls] += int(np.count_nonzero(valid & (end == cls)))
    ratios: dict[int, float] = {}
    for cls in classes:
        denominator = predicted_counts[cls] + smooth
        numerator = end_counts[cls] + smooth
        ratios[cls] = float(numerator / denominator) if denominator > 0.0 else 1.0
    diagnostics = {
        "calibration_case_count": len(calibration_cases),
        "class_values": classes,
        "smoothing": smooth,
        "predicted_counts": predicted_counts,
        "end_counts": end_counts,
        "class_count_ratios": ratios,
    }
    return ratios, diagnostics


def class4_collapse_class5_expansion_triggered(
    start_map: np.ndarray,
    prediction_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    min_class4_start_fraction: float = 0.03,
    max_class4_pred_to_start_ratio: float = 0.30,
    min_class5_expansion_fraction: float = 0.03,
) -> bool:
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    if start.shape != prediction.shape:
        raise ValueError(f"shape mismatch: start={start.shape}, prediction={prediction.shape}")
    valid = (start != 0) & (prediction != 0) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != start.shape:
        raise ValueError(f"valid_mask shape {valid.shape} does not match start shape {start.shape}")
    total = int(np.count_nonzero(valid))
    if total <= 0:
        return False
    class4_start = int(np.count_nonzero(valid & (start == 4)))
    class4_prediction = int(np.count_nonzero(valid & (prediction == 4)))
    class5_start = int(np.count_nonzero(valid & (start == 5)))
    class5_prediction = int(np.count_nonzero(valid & (prediction == 5)))
    class4_fraction = float(class4_start / total)
    class4_ratio = float(class4_prediction / class4_start) if class4_start > 0 else 1.0
    class5_expansion_fraction = float((class5_prediction - class5_start) / total)
    return (
        class4_fraction >= float(min_class4_start_fraction)
        and class4_ratio <= float(max_class4_pred_to_start_ratio)
        and class5_expansion_fraction >= float(min_class5_expansion_fraction)
    )


def apply_calibrated_class_budget_gate(
    start_map: np.ndarray,
    prediction_map: np.ndarray,
    score_map: np.ndarray,
    class_count_ratios: dict[int, float],
    budget_strength: float = 0.1,
    gated_classes: list[int] | None = None,
    min_overbudget_pixels: int = 100,
    valid_mask: np.ndarray | None = None,
    target_support_weight: float = 0.75,
    source_support_penalty: float = 0.75,
    target_neighborhood: np.ndarray | None = None,
    source_neighborhood: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    score = np.asarray(score_map, dtype=np.float32)
    if start.shape != prediction.shape or start.shape != score.shape:
        raise ValueError(f"shape mismatch: start={start.shape}, prediction={prediction.shape}, score={score.shape}")
    strength = float(budget_strength)
    if strength < 0.0 or strength > 1.0:
        raise ValueError(f"budget_strength must be in [0, 1]: {strength}")
    min_over = int(min_overbudget_pixels)
    if min_over < 1:
        raise ValueError(f"min_overbudget_pixels must be positive: {min_over}")
    target_weight = float(target_support_weight)
    source_penalty = float(source_support_penalty)
    if target_weight < 0.0 or source_penalty < 0.0:
        raise ValueError(
            "target_support_weight and source_support_penalty must be non-negative: "
            f"target={target_weight}, source={source_penalty}"
        )
    valid = (start != 0) & (prediction != 0) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != start.shape:
        raise ValueError(f"valid_mask shape {valid.shape} does not match start shape {start.shape}")
    if target_neighborhood is None or source_neighborhood is None:
        class_values = sorted({int(value) for value in np.unique(start)} | {int(value) for value in np.unique(prediction)})
        computed_target, computed_source = class_aligned_neighborhood(start, prediction, class_values)
        target_support = computed_target if target_neighborhood is None else np.asarray(target_neighborhood, dtype=np.float32)
        source_support = computed_source if source_neighborhood is None else np.asarray(source_neighborhood, dtype=np.float32)
    else:
        target_support = np.asarray(target_neighborhood, dtype=np.float32)
        source_support = np.asarray(source_neighborhood, dtype=np.float32)
    if target_support.shape != start.shape or source_support.shape != start.shape:
        raise ValueError(
            f"neighborhood shape mismatch: target={target_support.shape}, "
            f"source={source_support.shape}, start={start.shape}"
        )

    gated = prediction.copy()
    rank_score = score + target_weight * target_support - source_penalty * source_support
    class_rows: list[dict[str, Any]] = []
    reverted_total = 0
    for cls in [int(value) for value in (gated_classes or sorted(class_count_ratios))]:
        current_count = int(np.count_nonzero(valid & (gated == cls)))
        ratio = float(class_count_ratios.get(cls, 1.0))
        if ratio < 0.0:
            raise ValueError(f"class_count_ratio must be non-negative for class {cls}: {ratio}")
        calibrated_budget = int(round(current_count * ratio))
        target_budget = int(round((1.0 - strength) * current_count + strength * calibrated_budget))
        overbudget = max(0, current_count - target_budget)
        reverted = 0
        candidate_indices = np.flatnonzero((valid & (gated == cls) & (gated != start)).ravel())
        if overbudget >= min_over and candidate_indices.size > 0:
            revert_count = min(overbudget, int(candidate_indices.size))
            order = candidate_indices[np.argsort(rank_score.ravel()[candidate_indices])]
            reverted_indices = order[:revert_count]
            gated.ravel()[reverted_indices] = start.ravel()[reverted_indices]
            reverted = int(reverted_indices.size)
            reverted_total += reverted
        class_rows.append(
            {
                "target_class": cls,
                "class_count_ratio": ratio,
                "current_count": current_count,
                "calibrated_budget": calibrated_budget,
                "target_budget": target_budget,
                "overbudget_pixels": overbudget,
                "candidate_pixels": int(candidate_indices.size),
                "reverted_pixels": reverted,
            }
        )

    diagnostics = {
        "candidate_change_pixels": int(np.count_nonzero(valid & (prediction != start))),
        "reverted_pixels": int(reverted_total),
        "budget_strength": strength,
        "gated_classes": [int(value) for value in (gated_classes or sorted(class_count_ratios))],
        "min_overbudget_pixels": min_over,
        "target_support_weight": target_weight,
        "source_support_penalty": source_penalty,
        "class_rows": class_rows,
    }
    return gated.astype(prediction.dtype, copy=False), diagnostics


def _load_source_prediction(source_prediction_dir: Path, area: str, start_year: int, end_year: int) -> np.ndarray:
    path = Path(source_prediction_dir) / f"{area}_lulc_pred_{start_year}_{end_year}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing source prediction for {area}: {path}")
    return np.load(path).astype(np.int32, copy=False)


def run_calibrated_class_budget_gate(
    source_prediction_dir: Path,
    change_gate_dirs: list[Path],
    calibration_label_dir: Path,
    calibration_prediction_dir: Path,
    output_dir: Path,
    budget_strength: float = 0.1,
    gated_classes: list[int] | None = None,
    min_overbudget_pixels: int = 100,
    ratio_smoothing: float = 0.0,
    excluded_name_terms_by_area: dict[str, list[str]] | None = None,
    anomaly_trigger: bool = True,
    min_class4_start_fraction: float = 0.03,
    max_class4_pred_to_start_ratio: float = 0.30,
    min_class5_expansion_fraction: float = 0.03,
) -> dict[str, Any]:
    metadata_cases = [load_case_from_change_gate_dir(Path(path)) for path in change_gate_dirs]
    calibration_cases, skipped_calibration = discover_calibration_cases(
        Path(calibration_label_dir),
        Path(calibration_prediction_dir),
    )
    if not calibration_cases:
        raise ValueError("no usable calibration cases were discovered")

    output = Path(output_dir)
    predictions_dir = output / "predictions"
    diagnostics_dir = output / "diagnostics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    case_summaries: list[dict[str, Any]] = []
    classes = [1, 2, 4, 5, 7, 8, 9, 10, 11]
    for case in metadata_cases:
        source_prediction = _load_source_prediction(
            Path(source_prediction_dir),
            case.area,
            case.start_year,
            case.end_year,
        )
        if source_prediction.shape != case.start_map.shape:
            raise ValueError(
                f"shape mismatch for {case.area}: prediction={source_prediction.shape}, start={case.start_map.shape}"
            )
        target_calibration_cases = _filter_calibration_cases(
            case.area,
            calibration_cases,
            excluded_name_terms_by_area,
        )
        if not target_calibration_cases:
            raise ValueError(f"no calibration cases remain after exclusions for {case.area}")
        class_ratios, ratio_diagnostics = estimate_class_count_ratios(
            target_calibration_cases,
            class_values=classes,
            smoothing=ratio_smoothing,
        )
        valid_mask = (case.start_map != 0) & (source_prediction != 0)
        triggered = True
        if anomaly_trigger:
            triggered = class4_collapse_class5_expansion_triggered(
                case.start_map,
                source_prediction,
                valid_mask=valid_mask,
                min_class4_start_fraction=min_class4_start_fraction,
                max_class4_pred_to_start_ratio=max_class4_pred_to_start_ratio,
                min_class5_expansion_fraction=min_class5_expansion_fraction,
            )
        if triggered:
            gated, gate_diagnostics = apply_calibrated_class_budget_gate(
                case.start_map,
                source_prediction,
                case.score_map,
                class_count_ratios=class_ratios,
                budget_strength=budget_strength,
                gated_classes=gated_classes or [5],
                min_overbudget_pixels=min_overbudget_pixels,
                valid_mask=valid_mask,
            )
        else:
            gated = source_prediction.copy()
            gate_diagnostics = {
                "candidate_change_pixels": int(np.count_nonzero(valid_mask & (source_prediction != case.start_map))),
                "reverted_pixels": 0,
                "budget_strength": float(budget_strength),
                "gated_classes": [int(value) for value in (gated_classes or [5])],
                "min_overbudget_pixels": int(min_overbudget_pixels),
                "class_rows": [],
            }
        prediction_path = predictions_dir / f"{case.area}_lulc_pred_{case.start_year}_{case.end_year}.npy"
        np.save(prediction_path, gated)
        diagnostic_payload = {
            "area": case.area,
            "start_year": case.start_year,
            "end_year": case.end_year,
            "source_prediction_dir": Path(source_prediction_dir),
            "source_change_gate_dir": case.source_dir,
            "calibration_label_dir": Path(calibration_label_dir),
            "calibration_prediction_dir": Path(calibration_prediction_dir),
            "calibration_area_count": len(target_calibration_cases),
            "calibration_areas": [calibration.area for calibration in target_calibration_cases],
            "excluded_name_terms": (excluded_name_terms_by_area or {}).get(case.area, []),
            "anomaly_triggered": bool(triggered),
            "ratio_diagnostics": ratio_diagnostics,
            "gate_diagnostics": gate_diagnostics,
            "output_prediction": prediction_path,
        }
        _write_json(diagnostics_dir / f"{case.area}_calibrated_class_budget_gate.json", diagnostic_payload)
        case_summaries.append(
            {
                "area": case.area,
                "start_year": case.start_year,
                "end_year": case.end_year,
                "output_prediction": prediction_path,
                "calibration_area_count": len(target_calibration_cases),
                "anomaly_triggered": bool(triggered),
                "candidate_change_pixels": gate_diagnostics["candidate_change_pixels"],
                "reverted_pixels": gate_diagnostics["reverted_pixels"],
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paper58_calibrated_class_budget_gate",
        "n_cases": len(metadata_cases),
        "source_prediction_dir": Path(source_prediction_dir),
        "change_gate_dirs": [Path(path) for path in change_gate_dirs],
        "calibration_label_dir": Path(calibration_label_dir),
        "calibration_prediction_dir": Path(calibration_prediction_dir),
        "usable_calibration_area_count": len(calibration_cases),
        "skipped_calibration_cases": skipped_calibration,
        "excluded_name_terms_by_area": excluded_name_terms_by_area or {},
        "parameters": {
            "budget_strength": float(budget_strength),
            "gated_classes": [int(value) for value in (gated_classes or [5])],
            "min_overbudget_pixels": int(min_overbudget_pixels),
            "ratio_smoothing": float(ratio_smoothing),
            "anomaly_trigger": bool(anomaly_trigger),
            "min_class4_start_fraction": float(min_class4_start_fraction),
            "max_class4_pred_to_start_ratio": float(max_class4_pred_to_start_ratio),
            "min_class5_expansion_fraction": float(min_class5_expansion_fraction),
        },
        "cases": case_summaries,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _parse_classes(values: str | None) -> list[int] | None:
    if values is None or values.strip() == "":
        return None
    return [int(value.strip()) for value in values.split(",") if value.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply Paper58 calibrated class-budget gate.")
    parser.add_argument("--source-prediction-dir", type=Path, required=True)
    parser.add_argument("--change-gate-dir", action="append", type=Path, required=True)
    parser.add_argument("--calibration-label-dir", type=Path, required=True)
    parser.add_argument("--calibration-prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-strength", type=float, default=0.1)
    parser.add_argument("--gated-classes", default="5")
    parser.add_argument("--min-overbudget-pixels", type=int, default=100)
    parser.add_argument("--ratio-smoothing", type=float, default=0.0)
    parser.add_argument("--disable-anomaly-trigger", action="store_true")
    parser.add_argument("--min-class4-start-fraction", type=float, default=0.03)
    parser.add_argument("--max-class4-pred-to-start-ratio", type=float, default=0.30)
    parser.add_argument("--min-class5-expansion-fraction", type=float, default=0.03)
    parser.add_argument(
        "--exclude-term",
        action="append",
        default=None,
        help="Optional calibration exclusion formatted as target_area=substring; repeat as needed.",
    )
    args = parser.parse_args(argv)
    manifest = run_calibrated_class_budget_gate(
        source_prediction_dir=args.source_prediction_dir,
        change_gate_dirs=args.change_gate_dir,
        calibration_label_dir=args.calibration_label_dir,
        calibration_prediction_dir=args.calibration_prediction_dir,
        output_dir=args.output_dir,
        budget_strength=args.budget_strength,
        gated_classes=_parse_classes(args.gated_classes),
        min_overbudget_pixels=args.min_overbudget_pixels,
        ratio_smoothing=args.ratio_smoothing,
        excluded_name_terms_by_area=_parse_exclude_terms(args.exclude_term),
        anomaly_trigger=not args.disable_anomaly_trigger,
        min_class4_start_fraction=args.min_class4_start_fraction,
        max_class4_pred_to_start_ratio=args.max_class4_pred_to_start_ratio,
        min_class5_expansion_fraction=args.min_class5_expansion_fraction,
    )
    print(
        "Paper58 calibrated class-budget gate complete: "
        f"cases={manifest['n_cases']}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
