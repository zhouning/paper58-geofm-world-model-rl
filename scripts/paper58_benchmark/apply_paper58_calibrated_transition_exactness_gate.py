from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper58_benchmark.apply_paper58_change_gate import (
    apply_transition_exactness_reversion_gate,
    class_aligned_neighborhood,
    estimate_transition_reliability,
)
from scripts.paper58_benchmark.apply_paper58_transition_reliability_gate import (
    TransitionReliabilityGateCase,
    load_case_from_change_gate_dir,
)


PREDICTION_RE = re.compile(r"^(?P<area>.+)_lulc_pred_(?P<start_year>\d{4})_(?P<end_year>\d{4})\.npy$")


@dataclass(frozen=True)
class CalibrationCase:
    area: str
    start_year: int
    end_year: int
    start_map: np.ndarray
    end_map: np.ndarray
    prediction_map: np.ndarray


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


def discover_calibration_cases(
    calibration_label_dir: Path,
    calibration_prediction_dir: Path,
) -> tuple[list[CalibrationCase], list[dict[str, str]]]:
    cases: list[CalibrationCase] = []
    skipped: list[dict[str, str]] = []
    label_root = Path(calibration_label_dir)
    prediction_root = Path(calibration_prediction_dir)
    for prediction_path in sorted(prediction_root.glob("*_lulc_pred_*_*.npy")):
        match = PREDICTION_RE.match(prediction_path.name)
        if match is None:
            skipped.append({"path": prediction_path.as_posix(), "reason": "filename_not_supported"})
            continue
        area = str(match.group("area"))
        start_year = int(match.group("start_year"))
        end_year = int(match.group("end_year"))
        start_path = label_root / f"{area}_lulc_{start_year}.npy"
        end_path = label_root / f"{area}_lulc_{end_year}.npy"
        if not start_path.exists() or not end_path.exists():
            skipped.append({"area": area, "reason": "missing_start_or_end_label"})
            continue
        try:
            start = np.load(start_path).astype(np.int32, copy=False)
            end = np.load(end_path).astype(np.int32, copy=False)
            prediction = np.load(prediction_path).astype(np.int32, copy=False)
        except Exception as exc:
            skipped.append({"area": area, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if start.shape != end.shape or start.shape != prediction.shape or start.ndim != 2:
            skipped.append(
                {
                    "area": area,
                    "reason": f"shape_mismatch: start={start.shape}, end={end.shape}, pred={prediction.shape}",
                }
            )
            continue
        cases.append(
            CalibrationCase(
                area=area,
                start_year=start_year,
                end_year=end_year,
                start_map=start,
                end_map=end,
                prediction_map=prediction,
            )
        )
    return cases, skipped


def _load_source_prediction(source_prediction_dir: Path, case: TransitionReliabilityGateCase) -> np.ndarray:
    path = Path(source_prediction_dir) / f"{case.area}_lulc_pred_{case.start_year}_{case.end_year}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing source prediction for {case.area}: {path}")
    prediction = np.load(path).astype(np.int32, copy=False)
    if prediction.shape != case.start_map.shape:
        raise ValueError(
            f"shape mismatch for {case.area}: prediction={prediction.shape}, start={case.start_map.shape}"
        )
    return prediction


def _calibration_tuples(cases: list[CalibrationCase]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [(case.start_map, case.end_map, case.prediction_map) for case in cases]


def _excluded_terms_for_area(area: str, excluded_name_terms_by_area: dict[str, list[str]] | None) -> list[str]:
    if not excluded_name_terms_by_area:
        return []
    return [str(term).lower() for term in excluded_name_terms_by_area.get(area, [])]


def _filter_calibration_cases(
    target_area: str,
    calibration_cases: list[CalibrationCase],
    excluded_name_terms_by_area: dict[str, list[str]] | None,
) -> list[CalibrationCase]:
    terms = _excluded_terms_for_area(target_area, excluded_name_terms_by_area)
    if not terms:
        return list(calibration_cases)
    return [case for case in calibration_cases if not any(term in case.area.lower() for term in terms)]


def apply_transition_spatial_exactness_reversion_gate(
    start_map: np.ndarray,
    prediction_map: np.ndarray,
    score_map: np.ndarray,
    exactness_by_transition: dict[tuple[int, int], float],
    global_exactness: float,
    max_revert_fraction: float = 0.4,
    min_group_size: int = 100,
    valid_mask: np.ndarray | None = None,
    target_support_weight: float = 0.0,
    source_support_penalty: float = 0.0,
    target_neighborhood: np.ndarray | None = None,
    source_neighborhood: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    score = np.asarray(score_map, dtype=np.float32)
    if start.shape != prediction.shape or start.shape != score.shape:
        raise ValueError(f"shape mismatch: start={start.shape}, prediction={prediction.shape}, score={score.shape}")
    global_exact = float(global_exactness)
    max_revert = float(max_revert_fraction)
    min_group = int(min_group_size)
    target_weight = float(target_support_weight)
    source_penalty = float(source_support_penalty)
    if global_exact < 0.0 or global_exact > 1.0:
        raise ValueError(f"global_exactness must be in [0, 1]: {global_exact}")
    if max_revert < 0.0 or max_revert > 1.0:
        raise ValueError(f"max_revert_fraction must be in [0, 1]: {max_revert}")
    if min_group < 1:
        raise ValueError(f"min_group_size must be positive: {min_group}")
    if target_weight < 0.0 or source_penalty < 0.0:
        raise ValueError(
            "target_support_weight and source_support_penalty must be non-negative: "
            f"target={target_weight}, source={source_penalty}"
        )
    valid = np.ones(start.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
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

    candidates = valid & (prediction != start)
    rank_score = score + target_weight * target_support - source_penalty * source_support
    gated = prediction.copy()
    reverted_total = 0
    transition_rows: list[dict[str, Any]] = []
    denominator = max(global_exact, 1e-6)
    for from_cls in sorted({int(value) for value in np.unique(start[candidates])}):
        from_mask = candidates & (start == from_cls)
        for to_cls in sorted({int(value) for value in np.unique(prediction[from_mask])}):
            group = from_mask & (prediction == to_cls)
            indices = np.flatnonzero(group.ravel())
            if indices.size == 0:
                continue
            exactness = float(exactness_by_transition.get((from_cls, to_cls), global_exact))
            revert_fraction = 0.0
            reverted = 0
            if indices.size >= min_group and exactness < global_exact:
                revert_fraction = max_revert * (global_exact - exactness) / denominator
                revert_count = int(round(indices.size * revert_fraction))
                if revert_count > 0:
                    order = indices[np.argsort(rank_score.ravel()[indices])]
                    reverted_indices = order[:revert_count]
                    gated.ravel()[reverted_indices] = start.ravel()[reverted_indices]
                    reverted = int(reverted_indices.size)
                    reverted_total += reverted
            transition_rows.append(
                {
                    "from_class": from_cls,
                    "to_class": to_cls,
                    "candidate_pixels": int(indices.size),
                    "exactness": exactness,
                    "revert_fraction": float(revert_fraction),
                    "reverted_pixels": reverted,
                    "target_support_mean": float(np.mean(target_support.ravel()[indices])),
                    "source_support_mean": float(np.mean(source_support.ravel()[indices])),
                    "rank_score_min": float(np.min(rank_score.ravel()[indices])),
                    "rank_score_max": float(np.max(rank_score.ravel()[indices])),
                }
            )

    diagnostics = {
        "candidate_change_pixels": int(np.count_nonzero(candidates)),
        "reverted_change_pixels": int(reverted_total),
        "global_exactness": global_exact,
        "max_revert_fraction": max_revert,
        "min_group_size": min_group,
        "target_support_weight": target_weight,
        "source_support_penalty": source_penalty,
        "transition_group_count": len(transition_rows),
        "transition_groups": transition_rows,
    }
    return gated.astype(prediction.dtype, copy=False), diagnostics


def run_calibrated_transition_exactness_gate(
    source_prediction_dir: Path,
    change_gate_dirs: list[Path],
    calibration_label_dir: Path,
    calibration_prediction_dir: Path,
    output_dir: Path,
    max_revert_fraction: float = 0.4,
    min_group_size: int = 100,
    smoothing: float = 200.0,
    excluded_name_terms_by_area: dict[str, list[str]] | None = None,
    target_support_weight: float = 0.0,
    source_support_penalty: float = 0.0,
) -> dict[str, Any]:
    metadata_cases = [load_case_from_change_gate_dir(Path(path)) for path in change_gate_dirs]
    cases = [
        TransitionReliabilityGateCase(
            area=case.area,
            start_year=case.start_year,
            end_year=case.end_year,
            start_map=case.start_map,
            end_map=case.end_map,
            prediction_map=_load_source_prediction(Path(source_prediction_dir), case),
            score_map=case.score_map,
            source_dir=case.source_dir,
        )
        for case in metadata_cases
    ]
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
    for case in cases:
        target_calibration_cases = _filter_calibration_cases(
            case.area,
            calibration_cases,
            excluded_name_terms_by_area,
        )
        if not target_calibration_cases:
            raise ValueError(f"no calibration cases remain after exclusions for {case.area}")
        exactness, training_diagnostics = estimate_transition_reliability(
            _calibration_tuples(target_calibration_cases),
            alpha_exact=1.0,
            smoothing=smoothing,
        )
        valid_mask = (case.start_map != 0) & (case.prediction_map != 0)
        if target_support_weight > 0.0 or source_support_penalty > 0.0:
            gated, gate_diagnostics = apply_transition_spatial_exactness_reversion_gate(
                case.start_map,
                case.prediction_map,
                case.score_map,
                exactness_by_transition=exactness,
                global_exactness=float(training_diagnostics["global_reliability"]),
                max_revert_fraction=max_revert_fraction,
                min_group_size=min_group_size,
                valid_mask=valid_mask,
                target_support_weight=target_support_weight,
                source_support_penalty=source_support_penalty,
            )
        else:
            gated, gate_diagnostics = apply_transition_exactness_reversion_gate(
                case.start_map,
                case.prediction_map,
                case.score_map,
                exactness_by_transition=exactness,
                global_exactness=float(training_diagnostics["global_reliability"]),
                max_revert_fraction=max_revert_fraction,
                min_group_size=min_group_size,
                valid_mask=valid_mask,
            )
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
            "excluded_name_terms": _excluded_terms_for_area(case.area, excluded_name_terms_by_area),
            "training_diagnostics": training_diagnostics,
            "gate_diagnostics": gate_diagnostics,
            "exactness_by_transition": {
                f"{from_cls}->{to_cls}": value for (from_cls, to_cls), value in sorted(exactness.items())
            },
            "output_prediction": prediction_path,
        }
        _write_json(diagnostics_dir / f"{case.area}_calibrated_transition_exactness_gate.json", diagnostic_payload)
        case_summaries.append(
            {
                "area": case.area,
                "start_year": case.start_year,
                "end_year": case.end_year,
                "output_prediction": prediction_path,
                "source_change_gate_dir": case.source_dir,
                "calibration_area_count": len(target_calibration_cases),
                "candidate_change_pixels": gate_diagnostics["candidate_change_pixels"],
                "reverted_change_pixels": gate_diagnostics["reverted_change_pixels"],
                "global_exactness": training_diagnostics["global_reliability"],
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "paper58_calibrated_transition_spatial_exactness_gate"
            if target_support_weight > 0.0 or source_support_penalty > 0.0
            else "paper58_calibrated_transition_exactness_gate"
        ),
        "n_cases": len(cases),
        "source_prediction_dir": Path(source_prediction_dir),
        "change_gate_dirs": [Path(path) for path in change_gate_dirs],
        "calibration_label_dir": Path(calibration_label_dir),
        "calibration_prediction_dir": Path(calibration_prediction_dir),
        "usable_calibration_area_count": len(calibration_cases),
        "skipped_calibration_cases": skipped_calibration,
        "excluded_name_terms_by_area": excluded_name_terms_by_area or {},
        "parameters": {
            "max_revert_fraction": float(max_revert_fraction),
            "min_group_size": int(min_group_size),
            "smoothing": float(smoothing),
            "target_support_weight": float(target_support_weight),
            "source_support_penalty": float(source_support_penalty),
        },
        "cases": case_summaries,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _parse_exclude_terms(values: list[str] | None) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"exclude term must be formatted as area=term, got {value!r}")
        area, term = value.split("=", 1)
        parsed.setdefault(area, []).append(term)
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply Paper58 transition exactness gate calibrated from external independent cases."
    )
    parser.add_argument("--source-prediction-dir", type=Path, required=True)
    parser.add_argument("--change-gate-dir", action="append", type=Path, required=True)
    parser.add_argument("--calibration-label-dir", type=Path, required=True)
    parser.add_argument("--calibration-prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-revert-fraction", type=float, default=0.4)
    parser.add_argument("--min-group-size", type=int, default=100)
    parser.add_argument("--smoothing", type=float, default=200.0)
    parser.add_argument("--target-support-weight", type=float, default=0.0)
    parser.add_argument("--source-support-penalty", type=float, default=0.0)
    parser.add_argument(
        "--exclude-term",
        action="append",
        default=None,
        help="Optional calibration exclusion formatted as target_area=substring; repeat as needed.",
    )
    args = parser.parse_args(argv)
    manifest = run_calibrated_transition_exactness_gate(
        source_prediction_dir=args.source_prediction_dir,
        change_gate_dirs=args.change_gate_dir,
        calibration_label_dir=args.calibration_label_dir,
        calibration_prediction_dir=args.calibration_prediction_dir,
        output_dir=args.output_dir,
        max_revert_fraction=args.max_revert_fraction,
        min_group_size=args.min_group_size,
        smoothing=args.smoothing,
        excluded_name_terms_by_area=_parse_exclude_terms(args.exclude_term),
        target_support_weight=args.target_support_weight,
        source_support_penalty=args.source_support_penalty,
    )
    print(
        "Paper58 calibrated transition-exactness gate complete: "
        f"cases={manifest['n_cases']}, calibration={manifest['usable_calibration_area_count']}, "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
