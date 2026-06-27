from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper58_benchmark.apply_paper58_change_gate import (
    apply_transition_exactness_reversion_gate,
    estimate_transition_reliability,
)
from scripts.paper58_benchmark.apply_paper58_transition_reliability_gate import (
    TransitionReliabilityGateCase,
    load_case_from_change_gate_dir,
)


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


def _training_tuples(cases: list[TransitionReliabilityGateCase]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [(case.start_map, case.end_map, case.prediction_map) for case in cases]


def run_leave_one_area_transition_exactness_gate(
    source_prediction_dir: Path,
    change_gate_dirs: list[Path],
    output_dir: Path,
    max_revert_fraction: float = 0.4,
    min_group_size: int = 100,
    smoothing: float = 200.0,
) -> dict[str, Any]:
    metadata_cases = [load_case_from_change_gate_dir(Path(path)) for path in change_gate_dirs]
    cases = [
        replace(case, prediction_map=_load_source_prediction(Path(source_prediction_dir), case))
        for case in metadata_cases
    ]
    output = Path(output_dir)
    predictions_dir = output / "predictions"
    diagnostics_dir = output / "diagnostics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    case_summaries: list[dict[str, Any]] = []
    for case in cases:
        training_cases = [other for other in cases if other.area != case.area]
        exactness, training_diagnostics = estimate_transition_reliability(
            _training_tuples(training_cases),
            alpha_exact=1.0,
            smoothing=smoothing,
        )
        valid_mask = (case.start_map != 0) & (case.prediction_map != 0)
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
            "training_area_count": len(training_cases),
            "training_areas": [training.area for training in training_cases],
            "training_diagnostics": training_diagnostics,
            "gate_diagnostics": gate_diagnostics,
            "exactness_by_transition": {
                f"{from_cls}->{to_cls}": value for (from_cls, to_cls), value in sorted(exactness.items())
            },
            "output_prediction": prediction_path,
        }
        _write_json(diagnostics_dir / f"{case.area}_transition_exactness_gate.json", diagnostic_payload)
        case_summaries.append(
            {
                "area": case.area,
                "start_year": case.start_year,
                "end_year": case.end_year,
                "output_prediction": prediction_path,
                "source_change_gate_dir": case.source_dir,
                "training_areas": diagnostic_payload["training_areas"],
                "candidate_change_pixels": gate_diagnostics["candidate_change_pixels"],
                "reverted_change_pixels": gate_diagnostics["reverted_change_pixels"],
                "global_exactness": training_diagnostics["global_reliability"],
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paper58_transition_exactness_reversion_gate",
        "n_cases": len(cases),
        "source_prediction_dir": Path(source_prediction_dir),
        "change_gate_dirs": [Path(path) for path in change_gate_dirs],
        "parameters": {
            "max_revert_fraction": float(max_revert_fraction),
            "min_group_size": int(min_group_size),
            "smoothing": float(smoothing),
        },
        "cases": case_summaries,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply leave-one-area transition exactness reversion gate from saved score rasters."
    )
    parser.add_argument("--source-prediction-dir", type=Path, required=True)
    parser.add_argument("--change-gate-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-revert-fraction", type=float, default=0.4)
    parser.add_argument("--min-group-size", type=int, default=100)
    parser.add_argument("--smoothing", type=float, default=200.0)
    args = parser.parse_args(argv)
    manifest = run_leave_one_area_transition_exactness_gate(
        source_prediction_dir=args.source_prediction_dir,
        change_gate_dirs=args.change_gate_dir,
        output_dir=args.output_dir,
        max_revert_fraction=args.max_revert_fraction,
        min_group_size=args.min_group_size,
        smoothing=args.smoothing,
    )
    print(
        "Paper58 transition-exactness gate complete: "
        f"cases={manifest['n_cases']}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
