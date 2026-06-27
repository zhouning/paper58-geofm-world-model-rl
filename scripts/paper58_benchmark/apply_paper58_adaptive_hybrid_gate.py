from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

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


def _case_map(change_gate_dirs: list[Path]) -> dict[str, TransitionReliabilityGateCase]:
    cases = [load_case_from_change_gate_dir(Path(path)) for path in change_gate_dirs]
    return {case.area: case for case in cases}


def _load_gate_prediction(case: TransitionReliabilityGateCase, source_dir: Path) -> np.ndarray:
    path = Path(source_dir) / "predictions" / f"{case.area}_lulc_pred_{case.start_year}_{case.end_year}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing gate prediction for {case.area}: {path}")
    prediction = np.load(path).astype(np.int32, copy=False)
    if prediction.shape != case.start_map.shape:
        raise ValueError(
            f"shape mismatch for {case.area}: prediction={prediction.shape}, start={case.start_map.shape}"
        )
    return prediction


def _load_transition_prediction(transition_gate_dir: Path, case: TransitionReliabilityGateCase) -> np.ndarray:
    path = Path(transition_gate_dir) / "predictions" / f"{case.area}_lulc_pred_{case.start_year}_{case.end_year}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing transition prediction for {case.area}: {path}")
    prediction = np.load(path).astype(np.int32, copy=False)
    if prediction.shape != case.start_map.shape:
        raise ValueError(
            f"shape mismatch for {case.area}: transition={prediction.shape}, start={case.start_map.shape}"
        )
    return prediction


def _load_global_reliability(transition_gate_dir: Path, case: TransitionReliabilityGateCase) -> float:
    path = Path(transition_gate_dir) / "diagnostics" / f"{case.area}_transition_reliability_gate.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["training_diagnostics"]["global_reliability"])


def _change_rate(start_map: np.ndarray, prediction_map: np.ndarray) -> float:
    valid = (start_map != 0) & (prediction_map != 0)
    denominator = int(np.count_nonzero(valid))
    if denominator == 0:
        return 0.0
    return float(np.count_nonzero(valid & (prediction_map != start_map)) / denominator)


def run_adaptive_hybrid_gate(
    transition_gate_dir: Path,
    low_change_gate_dirs: list[Path],
    high_change_gate_dirs: list[Path],
    output_dir: Path,
    mild_gate_dirs: list[Path] | None = None,
    low_reliability_threshold: float = 0.168,
    low_change_rate_threshold: float = 0.115,
    high_change_rate_threshold: float = 0.15,
) -> dict[str, Any]:
    low_reliability = float(low_reliability_threshold)
    low_change_rate = float(low_change_rate_threshold)
    high_change_rate = float(high_change_rate_threshold)
    if low_reliability < 0.0 or low_reliability > 1.0:
        raise ValueError(f"low_reliability_threshold must be in [0, 1]: {low_reliability}")
    if low_change_rate < 0.0 or low_change_rate > 1.0:
        raise ValueError(f"low_change_rate_threshold must be in [0, 1]: {low_change_rate}")
    if high_change_rate < 0.0 or high_change_rate > 1.0 or high_change_rate < low_change_rate:
        raise ValueError(
            f"high_change_rate_threshold must be in [low_change_rate_threshold, 1]: {high_change_rate}"
        )

    low_cases = _case_map(low_change_gate_dirs)
    high_cases = _case_map(high_change_gate_dirs)
    if set(low_cases) != set(high_cases):
        raise ValueError("low and high change-gate cases must contain the same areas")
    mild_sources: dict[str, Path] = {}
    if mild_gate_dirs is not None:
        if len(mild_gate_dirs) == 1 and (Path(mild_gate_dirs[0]) / "predictions").exists():
            mild_sources = {area: Path(mild_gate_dirs[0]) for area in high_cases}
        else:
            mild_cases = _case_map(mild_gate_dirs)
            if set(mild_cases) != set(high_cases):
                raise ValueError("mild gate cases must contain the same areas as low and high cases")
            mild_sources = {area: case.source_dir for area, case in mild_cases.items()}

    output = Path(output_dir)
    predictions_dir = output / "predictions"
    diagnostics_dir = output / "diagnostics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    case_summaries: list[dict[str, Any]] = []
    for area in sorted(high_cases):
        case = high_cases[area]
        transition_prediction = _load_transition_prediction(Path(transition_gate_dir), case)
        reliability = _load_global_reliability(Path(transition_gate_dir), case)
        change_rate = _change_rate(case.start_map, transition_prediction)
        if reliability < low_reliability:
            selected_branch = "transition_reliability_gate"
            selected_prediction = transition_prediction
            selected_source = Path(transition_gate_dir)
        elif change_rate < low_change_rate:
            selected_branch = "low_change_gate"
            selected_prediction = _load_gate_prediction(low_cases[area], low_cases[area].source_dir)
            selected_source = low_cases[area].source_dir
        elif mild_gate_dirs is not None and change_rate < high_change_rate:
            selected_branch = "mild_spatial_support_gate"
            selected_source = mild_sources[area]
            selected_prediction = _load_gate_prediction(case, selected_source)
        else:
            selected_branch = "high_change_gate"
            selected_prediction = _load_gate_prediction(case, case.source_dir)
            selected_source = case.source_dir

        prediction_path = predictions_dir / f"{area}_lulc_pred_{case.start_year}_{case.end_year}.npy"
        np.save(prediction_path, selected_prediction)
        diagnostic = {
            "area": area,
            "start_year": case.start_year,
            "end_year": case.end_year,
            "selected_branch": selected_branch,
            "selected_source": selected_source,
            "global_reliability": reliability,
            "transition_change_rate": change_rate,
            "low_reliability_threshold": low_reliability,
            "low_change_rate_threshold": low_change_rate,
            "high_change_rate_threshold": high_change_rate,
            "output_prediction": prediction_path,
        }
        _write_json(diagnostics_dir / f"{area}_adaptive_hybrid_gate.json", diagnostic)
        case_summaries.append(diagnostic)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paper58_adaptive_hybrid_gate",
        "n_cases": len(case_summaries),
        "transition_gate_dir": Path(transition_gate_dir),
        "low_change_gate_dirs": [Path(path) for path in low_change_gate_dirs],
        "mild_gate_dirs": [Path(path) for path in (mild_gate_dirs or [])],
        "high_change_gate_dirs": [Path(path) for path in high_change_gate_dirs],
        "parameters": {
            "low_reliability_threshold": low_reliability,
            "low_change_rate_threshold": low_change_rate,
            "high_change_rate_threshold": high_change_rate,
        },
        "cases": case_summaries,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply a non-oracle adaptive Paper58 gate portfolio.")
    parser.add_argument("--transition-gate-dir", type=Path, required=True)
    parser.add_argument("--low-change-gate-dir", action="append", type=Path, required=True)
    parser.add_argument("--mild-gate-dir", action="append", type=Path, default=None)
    parser.add_argument("--high-change-gate-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-reliability-threshold", type=float, default=0.168)
    parser.add_argument("--low-change-rate-threshold", type=float, default=0.115)
    parser.add_argument("--high-change-rate-threshold", type=float, default=0.15)
    args = parser.parse_args(argv)
    manifest = run_adaptive_hybrid_gate(
        transition_gate_dir=args.transition_gate_dir,
        low_change_gate_dirs=args.low_change_gate_dir,
        mild_gate_dirs=args.mild_gate_dir,
        high_change_gate_dirs=args.high_change_gate_dir,
        output_dir=args.output_dir,
        low_reliability_threshold=args.low_reliability_threshold,
        low_change_rate_threshold=args.low_change_rate_threshold,
        high_change_rate_threshold=args.high_change_rate_threshold,
    )
    print(
        "Paper58 adaptive hybrid gate complete: "
        f"cases={manifest['n_cases']}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
