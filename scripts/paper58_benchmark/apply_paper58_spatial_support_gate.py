from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper58_benchmark.apply_paper58_change_gate import apply_spatial_support_reversion_gate
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


def run_spatial_support_reversion_gate(
    source_prediction_dir: Path,
    change_gate_dirs: list[Path],
    output_dir: Path,
    target_neighborhood_threshold: float = 0.25,
    score_quantile: float = 0.4,
    max_revert_fraction: float = 0.03,
) -> dict[str, Any]:
    cases = [load_case_from_change_gate_dir(Path(path)) for path in change_gate_dirs]
    output = Path(output_dir)
    predictions_dir = output / "predictions"
    diagnostics_dir = output / "diagnostics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    case_summaries: list[dict[str, Any]] = []
    for case in cases:
        source_prediction = _load_source_prediction(Path(source_prediction_dir), case)
        valid_mask = (case.start_map != 0) & (source_prediction != 0)
        gated, diagnostics = apply_spatial_support_reversion_gate(
            case.start_map,
            source_prediction,
            case.score_map,
            target_neighborhood_threshold=target_neighborhood_threshold,
            score_quantile=score_quantile,
            max_revert_fraction=max_revert_fraction,
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
            "diagnostics": diagnostics,
            "output_prediction": prediction_path,
        }
        _write_json(diagnostics_dir / f"{case.area}_spatial_support_reversion_gate.json", diagnostic_payload)
        case_summaries.append(
            {
                "area": case.area,
                "start_year": case.start_year,
                "end_year": case.end_year,
                "output_prediction": prediction_path,
                "source_change_gate_dir": case.source_dir,
                "candidate_change_pixels": diagnostics["candidate_change_pixels"],
                "eligible_revert_pixels": diagnostics["eligible_revert_pixels"],
                "reverted_change_pixels": diagnostics["reverted_change_pixels"],
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paper58_spatial_support_reversion_gate",
        "n_cases": len(cases),
        "source_prediction_dir": Path(source_prediction_dir),
        "change_gate_dirs": [Path(path) for path in change_gate_dirs],
        "parameters": {
            "target_neighborhood_threshold": float(target_neighborhood_threshold),
            "score_quantile": float(score_quantile),
            "max_revert_fraction": float(max_revert_fraction),
        },
        "cases": case_summaries,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply Paper58 spatial support reversion gate on top of saved Paper58 predictions."
    )
    parser.add_argument("--source-prediction-dir", type=Path, required=True)
    parser.add_argument("--change-gate-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-neighborhood-threshold", type=float, default=0.25)
    parser.add_argument("--score-quantile", type=float, default=0.4)
    parser.add_argument("--max-revert-fraction", type=float, default=0.03)
    args = parser.parse_args(argv)
    manifest = run_spatial_support_reversion_gate(
        source_prediction_dir=args.source_prediction_dir,
        change_gate_dirs=args.change_gate_dir,
        output_dir=args.output_dir,
        target_neighborhood_threshold=args.target_neighborhood_threshold,
        score_quantile=args.score_quantile,
        max_revert_fraction=args.max_revert_fraction,
    )
    print(
        "Paper58 spatial-support reversion gate complete: "
        f"cases={manifest['n_cases']}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
