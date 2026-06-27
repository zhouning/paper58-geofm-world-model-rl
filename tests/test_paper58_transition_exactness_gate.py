import json
from pathlib import Path

import numpy as np


def _write_case(root: Path, area: str, start: np.ndarray, end: np.ndarray, score: np.ndarray) -> Path:
    case_dir = root / area
    labels = case_dir / "labels"
    predictions = case_dir / "predictions"
    diagnostics = case_dir / "diagnostics"
    labels.mkdir(parents=True)
    predictions.mkdir()
    diagnostics.mkdir()
    np.save(labels / f"{area}_lulc_2020.npy", start)
    np.save(labels / f"{area}_lulc_2021.npy", end)
    np.save(predictions / f"{area}_lulc_pred_2020_2021.npy", end)
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


def test_run_transition_exactness_gate_writes_predictions(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.apply_paper58_transition_exactness_gate import (
        run_leave_one_area_transition_exactness_gate,
    )

    case_a = _write_case(
        tmp_path,
        "case_a",
        start=np.array([[1, 1]], dtype=np.int32),
        end=np.array([[1, 1]], dtype=np.int32),
        score=np.array([[0.1, 0.9]], dtype=np.float32),
    )
    case_b = _write_case(
        tmp_path,
        "case_b",
        start=np.array([[1, 1]], dtype=np.int32),
        end=np.array([[1, 5]], dtype=np.int32),
        score=np.array([[0.1, 0.9]], dtype=np.float32),
    )
    source_predictions = tmp_path / "source_predictions"
    source_predictions.mkdir()
    np.save(source_predictions / "case_a_lulc_pred_2020_2021.npy", np.array([[5, 5]], dtype=np.int32))
    np.save(source_predictions / "case_b_lulc_pred_2020_2021.npy", np.array([[5, 5]], dtype=np.int32))

    manifest = run_leave_one_area_transition_exactness_gate(
        source_prediction_dir=source_predictions,
        change_gate_dirs=[case_a, case_b],
        output_dir=tmp_path / "out",
        max_revert_fraction=0.5,
        min_group_size=1,
        smoothing=0.0,
    )

    assert manifest["method"] == "paper58_transition_exactness_reversion_gate"
    assert (tmp_path / "out" / "predictions" / "case_a_lulc_pred_2020_2021.npy").exists()
    assert (tmp_path / "out" / "diagnostics" / "case_a_transition_exactness_gate.json").exists()
