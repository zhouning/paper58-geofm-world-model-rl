import json
from pathlib import Path

import numpy as np


def _write_gate_case(root: Path, area: str, start: np.ndarray, prediction: np.ndarray, end: np.ndarray) -> Path:
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
    np.save(diagnostics / f"{area}_change_gate_score.npy", np.ones(start.shape, dtype=np.float32))
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


def _write_transition_prediction(root: Path, area: str, prediction: np.ndarray, global_reliability: float) -> None:
    (root / "predictions").mkdir(parents=True, exist_ok=True)
    (root / "diagnostics").mkdir(exist_ok=True)
    np.save(root / "predictions" / f"{area}_lulc_pred_2020_2021.npy", prediction)
    (root / "diagnostics" / f"{area}_transition_reliability_gate.json").write_text(
        json.dumps({"training_diagnostics": {"global_reliability": global_reliability}}),
        encoding="utf-8",
    )


def test_run_adaptive_hybrid_gate_selects_non_oracle_branches(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.apply_paper58_adaptive_hybrid_gate import run_adaptive_hybrid_gate

    start = np.array([[1, 1, 1, 1]], dtype=np.int32)
    end = np.array([[1, 1, 1, 1]], dtype=np.int32)
    low_prediction = np.array([[5, 1, 1, 1]], dtype=np.int32)
    mild_prediction = np.array([[5, 5, 5, 1]], dtype=np.int32)
    high_prediction = np.array([[5, 5, 1, 1]], dtype=np.int32)
    transition_prediction = np.array([[5, 5, 5, 5]], dtype=np.int32)

    low_dirs = []
    mild_dirs = []
    high_dirs = []
    transition_dir = tmp_path / "transition"
    for area, global_reliability in [
        ("low_rel", 0.10),
        ("low_pressure", 0.20),
        ("mid_pressure", 0.20),
        ("high_pressure", 0.20),
    ]:
        low_dirs.append(_write_gate_case(tmp_path / "low", area, start, low_prediction, end))
        mild_dirs.append(_write_gate_case(tmp_path / "mild", area, start, mild_prediction, end))
        high_dirs.append(_write_gate_case(tmp_path / "high", area, start, high_prediction, end))
        if area == "low_pressure":
            prediction = np.array([[5, 1, 1, 1]], dtype=np.int32)
        elif area == "mid_pressure":
            prediction = np.array([[5, 5, 1, 1]], dtype=np.int32)
        else:
            prediction = transition_prediction
        _write_transition_prediction(transition_dir, area, prediction, global_reliability)

    manifest = run_adaptive_hybrid_gate(
        transition_gate_dir=transition_dir,
        low_change_gate_dirs=low_dirs,
        mild_gate_dirs=mild_dirs,
        high_change_gate_dirs=high_dirs,
        output_dir=tmp_path / "out",
        low_reliability_threshold=0.15,
        low_change_rate_threshold=0.40,
        high_change_rate_threshold=0.70,
    )

    selected = {row["area"]: row["selected_branch"] for row in manifest["cases"]}
    assert selected == {
        "high_pressure": "high_change_gate",
        "low_pressure": "low_change_gate",
        "low_rel": "transition_reliability_gate",
        "mid_pressure": "mild_spatial_support_gate",
    }
    assert np.load(tmp_path / "out" / "predictions" / "low_rel_lulc_pred_2020_2021.npy").tolist() == [[5, 5, 5, 5]]
    assert np.load(tmp_path / "out" / "predictions" / "mid_pressure_lulc_pred_2020_2021.npy").tolist() == [[5, 5, 5, 1]]
    assert (tmp_path / "out" / "diagnostics" / "high_pressure_adaptive_hybrid_gate.json").exists()
