import json
from pathlib import Path

import numpy as np

from scripts.paper58_benchmark.evaluate_las import evaluate_las


def _provenance_fields() -> dict:
    return {
        "bbox": [120.0, 30.0, 120.1, 30.1],
        "data_source": "test_source",
        "development_contact_status": "none",
        "contact_evidence": "synthetic no-contact evidence",
        "expected_role": "positive_change_candidate",
    }


def test_evaluate_las_writes_method_rows(tmp_path: Path):
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 3]], dtype=np.int32)
    paper58_pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
    flus_pred = np.array([[1, 1], [2, 3]], dtype=np.int32)

    label_start = tmp_path / "start.npy"
    label_end = tmp_path / "end.npy"
    pred_path = tmp_path / "paper58.npy"
    flus_path = tmp_path / "flus.npy"
    np.save(label_start, start)
    np.save(label_end, end)
    np.save(pred_path, paper58_pred)
    np.save(flus_path, flus_pred)

    registry = tmp_path / "benchmark_registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "external",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Wetland",
                        **_provenance_fields(),
                        "label_start_path": str(label_start),
                        "label_end_path": str(label_end),
                        "prediction_path": str(pred_path),
                        "flus_prediction_path": str(flus_path),
                        "qc_status": "include",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "las_out"

    result = evaluate_las(registry_path=registry, output_dir=output_dir)

    assert result["summary"]["n_evaluated_rows"] == 1
    assert result["summary"]["methods"] == ["flus", "paper58_direct", "paper58_las"]
    assert (output_dir / "las_metrics_by_method.csv").exists()
    assert (output_dir / "las_summary.json").exists()
    assert (output_dir / "simulated" / "external_2020_2021_paper58_las.npy").exists()


def test_evaluate_las_keeps_failure_rows_visible(tmp_path: Path):
    registry = tmp_path / "benchmark_registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "excluded",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Urban",
                        **_provenance_fields(),
                        "label_start_path": "missing_start.npy",
                        "label_end_path": "missing_end.npy",
                        "prediction_path": "missing_pred.npy",
                        "qc_status": "class_collapse",
                        "excluded_reason": "synthetic excluded row",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "las_out"

    result = evaluate_las(registry_path=registry, output_dir=output_dir)

    assert result["summary"]["n_evaluated_rows"] == 0
    failures = (output_dir / "las_failures.csv").read_text(encoding="utf-8")
    assert "excluded" in failures
    assert "class_collapse" in failures
