import json
from pathlib import Path

import pytest

from scripts.paper58_benchmark.attach_flus_predictions import attach_flus_predictions


def _row(area: str = "liaohe_delta_wetland_holdout") -> dict:
    return {
        "area": area,
        "start_year": 2020,
        "end_year": 2021,
        "tier": "tier1",
        "stratum": "Wetland",
        "bbox": [120.0, 30.0, 120.1, 30.1],
        "data_source": "test",
        "development_contact_status": "none",
        "contact_evidence": "synthetic no-contact evidence",
        "expected_role": "positive_change_candidate",
        "label_start_path": "start.npy",
        "label_end_path": "end.npy",
        "prediction_path": "paper58.npy",
        "qc_status": "include",
    }


def test_attach_flus_predictions_adds_matching_path(tmp_path: Path):
    registry_path = tmp_path / "benchmark_registry.json"
    prediction_dir = tmp_path / "flus"
    output_path = tmp_path / "registry_with_flus.json"
    prediction_dir.mkdir()
    flus_path = prediction_dir / "liaohe_delta_wetland_holdout_2020_2021_flus.tif"
    flus_path.write_bytes(b"placeholder")
    registry_path.write_text(json.dumps({"rows": [_row()]}), encoding="utf-8")

    summary = attach_flus_predictions(registry_path, prediction_dir, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["rows"][0]["flus_prediction_path"] == str(flus_path)
    assert summary == {
        "n_rows": 1,
        "n_matched": 1,
        "n_missing": 0,
        "missing": [],
    }


def test_attach_flus_predictions_reports_missing_rows(tmp_path: Path):
    registry_path = tmp_path / "benchmark_registry.json"
    prediction_dir = tmp_path / "flus"
    output_path = tmp_path / "registry_with_flus.json"
    prediction_dir.mkdir()
    registry_path.write_text(json.dumps({"rows": [_row(area="missing_area")]}), encoding="utf-8")

    summary = attach_flus_predictions(registry_path, prediction_dir, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "flus_prediction_path" not in payload["rows"][0]
    assert summary["n_rows"] == 1
    assert summary["n_matched"] == 0
    assert summary["n_missing"] == 1
    assert summary["missing"] == [{"area": "missing_area", "start_year": 2020, "end_year": 2021}]


def test_attach_flus_predictions_strict_mode_rejects_missing_rows(tmp_path: Path):
    registry_path = tmp_path / "benchmark_registry.json"
    prediction_dir = tmp_path / "flus"
    output_path = tmp_path / "registry_with_flus.json"
    prediction_dir.mkdir()
    registry_path.write_text(json.dumps({"rows": [_row(area="missing_area")]}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing FLUS predictions for 1 row"):
        attach_flus_predictions(registry_path, prediction_dir, output_path, strict=True)
