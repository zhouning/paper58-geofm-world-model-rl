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


def _write_calibration_case(root: Path, area: str, start: np.ndarray, end: np.ndarray, prediction: np.ndarray) -> None:
    labels = root / "labels"
    predictions = root / "predictions"
    labels.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(exist_ok=True)
    np.save(labels / f"{area}_lulc_2020.npy", start)
    np.save(labels / f"{area}_lulc_2021.npy", end)
    np.save(predictions / f"{area}_lulc_pred_2020_2021.npy", prediction)


def test_run_calibrated_transition_exactness_gate_uses_external_calibration(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.apply_paper58_calibrated_transition_exactness_gate import (
        run_calibrated_transition_exactness_gate,
    )

    target_start = np.array([[1, 1, 1]], dtype=np.int32)
    target_end = np.array([[1, 1, 1]], dtype=np.int32)
    target_score = np.array([[0.1, 0.9, 0.2]], dtype=np.float32)
    target_case = _write_change_gate_case(tmp_path, "target", target_start, target_end, target_score)

    source_predictions = tmp_path / "source_predictions"
    source_predictions.mkdir()
    np.save(source_predictions / "target_lulc_pred_2020_2021.npy", np.array([[5, 5, 1]], dtype=np.int32))

    calibration_root = tmp_path / "calibration"
    _write_calibration_case(
        calibration_root,
        "calib_low",
        start=np.array([[1, 1]], dtype=np.int32),
        end=np.array([[1, 1]], dtype=np.int32),
        prediction=np.array([[5, 5]], dtype=np.int32),
    )
    _write_calibration_case(
        calibration_root,
        "calib_high",
        start=np.array([[2, 2]], dtype=np.int32),
        end=np.array([[2, 5]], dtype=np.int32),
        prediction=np.array([[2, 5]], dtype=np.int32),
    )

    manifest = run_calibrated_transition_exactness_gate(
        source_prediction_dir=source_predictions,
        change_gate_dirs=[target_case],
        calibration_label_dir=calibration_root / "labels",
        calibration_prediction_dir=calibration_root / "predictions",
        output_dir=tmp_path / "out",
        max_revert_fraction=0.5,
        min_group_size=1,
        smoothing=0.0,
    )

    prediction = np.load(tmp_path / "out" / "predictions" / "target_lulc_pred_2020_2021.npy")
    assert prediction.tolist() == [[1, 5, 1]]
    assert manifest["method"] == "paper58_calibrated_transition_exactness_gate"
    assert manifest["cases"][0]["calibration_area_count"] == 2
    diagnostic = json.loads(
        (tmp_path / "out" / "diagnostics" / "target_calibrated_transition_exactness_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic["calibration_areas"] == ["calib_high", "calib_low"]


def test_transition_spatial_exactness_gate_ranks_by_neighborhood_support() -> None:
    from scripts.paper58_benchmark.apply_paper58_calibrated_transition_exactness_gate import (
        apply_transition_spatial_exactness_reversion_gate,
    )

    start = np.array([[1, 1, 1, 1]], dtype=np.int32)
    prediction = np.array([[5, 5, 5, 1]], dtype=np.int32)
    score = np.array([[0.1, 0.1, 0.1, 0.0]], dtype=np.float32)
    target_support = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    source_support = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

    gated, diagnostics = apply_transition_spatial_exactness_reversion_gate(
        start,
        prediction,
        score,
        exactness_by_transition={(1, 5): 0.0},
        global_exactness=0.5,
        max_revert_fraction=0.5,
        min_group_size=1,
        target_support_weight=1.0,
        source_support_penalty=1.0,
        target_neighborhood=target_support,
        source_neighborhood=source_support,
    )

    assert gated.tolist() == [[1, 5, 1, 1]]
    assert diagnostics["reverted_change_pixels"] == 2
    assert diagnostics["transition_groups"][0]["reverted_pixels"] == 2
