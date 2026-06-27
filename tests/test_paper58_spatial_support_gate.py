import json
from pathlib import Path

import numpy as np


def _write_change_gate_case(root: Path, area: str, start: np.ndarray, end: np.ndarray, score: np.ndarray) -> Path:
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


def test_run_spatial_support_reversion_gate_writes_predictions(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.apply_paper58_spatial_support_gate import (
        run_spatial_support_reversion_gate,
    )

    area = "case_a"
    source_dir = _write_change_gate_case(
        tmp_path,
        area,
        start=np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 5],
            ],
            dtype=np.int32,
        ),
        end=np.array(
            [
                [1, 1, 1],
                [1, 5, 1],
                [1, 1, 5],
            ],
            dtype=np.int32,
        ),
        score=np.array(
            [
                [0.1, 0.9, 0.9],
                [0.9, 0.1, 0.9],
                [0.9, 0.9, 0.9],
            ],
            dtype=np.float32,
        ),
    )
    prediction_dir = tmp_path / "v3_predictions"
    prediction_dir.mkdir()
    np.save(
        prediction_dir / f"{area}_lulc_pred_2020_2021.npy",
        np.array(
            [
                [5, 1, 1],
                [1, 5, 1],
                [1, 1, 5],
            ],
            dtype=np.int32,
        ),
    )

    manifest = run_spatial_support_reversion_gate(
        source_prediction_dir=prediction_dir,
        change_gate_dirs=[source_dir],
        output_dir=tmp_path / "out",
        target_neighborhood_threshold=0.25,
        score_quantile=1.0,
        max_revert_fraction=1.0,
    )

    out_prediction = tmp_path / "out" / "predictions" / f"{area}_lulc_pred_2020_2021.npy"
    assert manifest["method"] == "paper58_spatial_support_reversion_gate"
    assert out_prediction.exists()
    assert np.load(out_prediction).tolist() == [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 5],
    ]
    assert (tmp_path / "out" / "diagnostics" / f"{area}_spatial_support_reversion_gate.json").exists()
