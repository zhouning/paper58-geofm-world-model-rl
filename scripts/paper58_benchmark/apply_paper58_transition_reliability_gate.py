from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper58_benchmark.apply_paper58_change_gate import (
    apply_transition_reliability_gate,
    estimate_transition_reliability,
)


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TransitionReliabilityGateCase:
    area: str
    start_year: int
    end_year: int
    start_map: np.ndarray
    end_map: np.ndarray
    prediction_map: np.ndarray
    score_map: np.ndarray
    source_dir: Path


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


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_case_from_change_gate_dir(change_gate_dir: Path) -> TransitionReliabilityGateCase:
    root = Path(change_gate_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    area = str(manifest["area"])
    start_year = int(manifest["start_year"])
    end_year = int(manifest["end_year"])
    start_path = _resolve_path(manifest["inputs"]["start"])
    prediction_path = _resolve_path(manifest["inputs"]["paper58_prediction"])
    end_path = start_path.with_name(f"{area}_lulc_{end_year}.npy")
    score_path = root / "diagnostics" / f"{area}_change_gate_score.npy"
    return TransitionReliabilityGateCase(
        area=area,
        start_year=start_year,
        end_year=end_year,
        start_map=np.load(start_path).astype(np.int32, copy=False),
        end_map=np.load(end_path).astype(np.int32, copy=False),
        prediction_map=np.load(prediction_path).astype(np.int32, copy=False),
        score_map=np.load(score_path).astype(np.float32, copy=False),
        source_dir=root,
    )


def _training_tuples(cases: list[TransitionReliabilityGateCase]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [(case.start_map, case.end_map, case.prediction_map) for case in cases]


def run_leave_one_area_transition_reliability_gate(
    change_gate_dirs: list[Path],
    output_dir: Path,
    base_keep_fraction: float = 0.8,
    reliability_slope: float = 0.55,
    min_keep_fraction: float = 0.45,
    max_keep_fraction: float = 1.0,
    alpha_exact: float = 0.5,
    smoothing: float = 500.0,
) -> dict[str, Any]:
    cases = [load_case_from_change_gate_dir(Path(path)) for path in change_gate_dirs]
    output = Path(output_dir)
    predictions_dir = output / "predictions"
    diagnostics_dir = output / "diagnostics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    case_summaries: list[dict[str, Any]] = []
    for case in cases:
        training_cases = [other for other in cases if other.area != case.area]
        reliability, training_diagnostics = estimate_transition_reliability(
            _training_tuples(training_cases),
            alpha_exact=alpha_exact,
            smoothing=smoothing,
        )
        valid_mask = (case.start_map != 0) & (case.prediction_map != 0)
        gated, gate_diagnostics = apply_transition_reliability_gate(
            case.start_map,
            case.prediction_map,
            case.score_map,
            reliability_by_transition=reliability,
            global_reliability=float(training_diagnostics["global_reliability"]),
            base_keep_fraction=base_keep_fraction,
            reliability_slope=reliability_slope,
            min_keep_fraction=min_keep_fraction,
            max_keep_fraction=max_keep_fraction,
            valid_mask=valid_mask,
        )
        prediction_path = predictions_dir / f"{case.area}_lulc_pred_{case.start_year}_{case.end_year}.npy"
        np.save(prediction_path, gated)
        diagnostic_payload = {
            "area": case.area,
            "start_year": case.start_year,
            "end_year": case.end_year,
            "source_dir": case.source_dir,
            "training_area_count": len(training_cases),
            "training_areas": [training.area for training in training_cases],
            "training_diagnostics": training_diagnostics,
            "gate_diagnostics": gate_diagnostics,
            "reliability_by_transition": {
                f"{from_cls}->{to_cls}": value for (from_cls, to_cls), value in sorted(reliability.items())
            },
            "output_prediction": prediction_path,
        }
        _write_json(diagnostics_dir / f"{case.area}_transition_reliability_gate.json", diagnostic_payload)
        case_summaries.append(
            {
                "area": case.area,
                "start_year": case.start_year,
                "end_year": case.end_year,
                "output_prediction": prediction_path,
                "source_dir": case.source_dir,
                "training_areas": diagnostic_payload["training_areas"],
                "candidate_change_pixels": gate_diagnostics["candidate_change_pixels"],
                "kept_change_pixels": gate_diagnostics["kept_change_pixels"],
                "global_reliability": training_diagnostics["global_reliability"],
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paper58_transition_reliability_gate",
        "n_cases": len(cases),
        "parameters": {
            "base_keep_fraction": float(base_keep_fraction),
            "reliability_slope": float(reliability_slope),
            "min_keep_fraction": float(min_keep_fraction),
            "max_keep_fraction": float(max_keep_fraction),
            "alpha_exact": float(alpha_exact),
            "smoothing": float(smoothing),
        },
        "change_gate_dirs": [Path(path) for path in change_gate_dirs],
        "cases": case_summaries,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply Paper58 leave-one-area transition-reliability change gate from saved score rasters."
    )
    parser.add_argument("--change-gate-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-keep-fraction", type=float, default=0.8)
    parser.add_argument("--reliability-slope", type=float, default=0.55)
    parser.add_argument("--min-keep-fraction", type=float, default=0.45)
    parser.add_argument("--max-keep-fraction", type=float, default=1.0)
    parser.add_argument("--alpha-exact", type=float, default=0.5)
    parser.add_argument("--smoothing", type=float, default=500.0)
    args = parser.parse_args(argv)
    manifest = run_leave_one_area_transition_reliability_gate(
        change_gate_dirs=args.change_gate_dir,
        output_dir=args.output_dir,
        base_keep_fraction=args.base_keep_fraction,
        reliability_slope=args.reliability_slope,
        min_keep_fraction=args.min_keep_fraction,
        max_keep_fraction=args.max_keep_fraction,
        alpha_exact=args.alpha_exact,
        smoothing=args.smoothing,
    )
    print(
        "Paper58 transition-reliability gate complete: "
        f"cases={manifest['n_cases']}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
