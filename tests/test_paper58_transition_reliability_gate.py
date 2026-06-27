import json
from pathlib import Path

import numpy as np


def _write_case(root: Path, area: str, start: np.ndarray, end: np.ndarray, prediction: np.ndarray, score: np.ndarray) -> Path:
    case_dir = root / area
    labels = case_dir / "labels"
    predictions = case_dir / "predictions"
    diagnostics = case_dir / "diagnostics"
    labels.mkdir(parents=True)
    predictions.mkdir()
    diagnostics.mkdir()
    np.save(labels / f"{area}_lulc_2020.npy", start)
    np.save(labels / f"{area}_lulc_2021.npy", end)
    np.save(predictions / f"{area}_lulc_pred_2020_2021.npy", prediction)
    np.save(diagnostics / f"{area}_change_gate_score.npy", score)
    (case_dir / "manifest.json").write_text(
        json.dumps(
            {
                "area": area,
                "start_year": 2020,
                "end_year": 2021,
                "inputs": {
                    "start": str(labels / f"{area}_lulc_2020.npy"),
                    "paper58_prediction": str(predictions / f"{area}_lulc_pred_2020_2021.npy"),
                },
            }
        ),
        encoding="utf-8",
    )
    return case_dir


def test_run_leave_one_area_transition_reliability_gate_writes_predictions(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.apply_paper58_transition_reliability_gate import (
        run_leave_one_area_transition_reliability_gate,
    )

    case_a = _write_case(
        tmp_path,
        "case_a",
        start=np.array([[1, 1]], dtype=np.int32),
        end=np.array([[5, 1]], dtype=np.int32),
        prediction=np.array([[5, 5]], dtype=np.int32),
        score=np.array([[0.9, 0.8]], dtype=np.float32),
    )
    case_b = _write_case(
        tmp_path,
        "case_b",
        start=np.array([[1, 1]], dtype=np.int32),
        end=np.array([[5, 5]], dtype=np.int32),
        prediction=np.array([[5, 5]], dtype=np.int32),
        score=np.array([[0.7, 0.6]], dtype=np.float32),
    )

    result = run_leave_one_area_transition_reliability_gate(
        change_gate_dirs=[case_a, case_b],
        output_dir=tmp_path / "out",
        base_keep_fraction=0.5,
        min_keep_fraction=0.0,
        max_keep_fraction=1.0,
        reliability_slope=0.5,
        alpha_exact=1.0,
        smoothing=0.0,
    )

    assert result["n_cases"] == 2
    assert (tmp_path / "out" / "predictions" / "case_a_lulc_pred_2020_2021.npy").exists()
    assert (tmp_path / "out" / "diagnostics" / "case_a_transition_reliability_gate.json").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
