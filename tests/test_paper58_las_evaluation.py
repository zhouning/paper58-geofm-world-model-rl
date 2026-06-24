import csv
import json
from pathlib import Path

import numpy as np
import pytest

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


def test_evaluate_las_accepts_geotiff_flus_prediction(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 3]], dtype=np.int32)
    paper58_pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
    flus_pred = np.array([[1, 1], [2, 3]], dtype=np.int32)
    label_start = tmp_path / "start.npy"
    label_end = tmp_path / "end.npy"
    pred_path = tmp_path / "paper58.npy"
    flus_path = tmp_path / "flus.tif"
    np.save(label_start, start)
    np.save(label_end, end)
    np.save(pred_path, paper58_pred)
    with rasterio.open(
        flus_path,
        "w",
        driver="GTiff",
        height=flus_pred.shape[0],
        width=flus_pred.shape[1],
        count=1,
        dtype=flus_pred.dtype,
        transform=rasterio.transform.from_origin(0, flus_pred.shape[0], 1, 1),
    ) as dataset:
        dataset.write(flus_pred, 1)
    registry = tmp_path / "benchmark_registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "external_geotiff",
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

    result = evaluate_las(registry_path=registry, output_dir=tmp_path / "las_out")

    assert result["summary"]["methods"] == ["flus", "paper58_direct", "paper58_las"]


def test_evaluate_las_uses_paper58_gross_change_budget(tmp_path: Path):
    start = np.array([[1, 2]], dtype=np.int32)
    end = np.array([[2, 1]], dtype=np.int32)
    paper58_pred = np.array([[2, 1]], dtype=np.int32)

    label_start = tmp_path / "start.npy"
    label_end = tmp_path / "end.npy"
    pred_path = tmp_path / "paper58.npy"
    np.save(label_start, start)
    np.save(label_end, end)
    np.save(pred_path, paper58_pred)

    registry = tmp_path / "benchmark_registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "gross_budget",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Urban",
                        **_provenance_fields(),
                        "label_start_path": str(label_start),
                        "label_end_path": str(label_end),
                        "prediction_path": str(pred_path),
                        "qc_status": "include",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "las_out"

    evaluate_las(registry_path=registry, output_dir=output_dir)

    simulated = np.load(output_dir / "simulated" / "gross_budget_2020_2021_paper58_las.npy")
    assert simulated.tolist() == [[2, 1]]


def test_evaluate_las_accepts_neighborhood_weight(tmp_path: Path):
    start = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]], dtype=np.int32)
    end = np.array([[1, 1, 2], [1, 2, 2], [1, 1, 2]], dtype=np.int32)
    paper58_pred = end.copy()
    label_start = tmp_path / "start.npy"
    label_end = tmp_path / "end.npy"
    pred_path = tmp_path / "paper58.npy"
    np.save(label_start, start)
    np.save(label_end, end)
    np.save(pred_path, paper58_pred)
    registry = tmp_path / "benchmark_registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "neighborhood",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Urban",
                        **_provenance_fields(),
                        "label_start_path": str(label_start),
                        "label_end_path": str(label_end),
                        "prediction_path": str(pred_path),
                        "qc_status": "include",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_las(registry_path=registry, output_dir=tmp_path / "las_out", neighborhood_weight=1.0)

    assert result["summary"]["n_evaluated_rows"] == 1


def test_evaluate_las_accepts_latent_neighborhood_weight(tmp_path: Path):
    start = np.array([[1, 2, 1], [1, 2, 1]], dtype=np.int32)
    end = np.array([[2, 2, 1], [1, 2, 1]], dtype=np.int32)
    paper58_pred = end.copy()
    embeddings = np.zeros((2, 3, 2), dtype=np.float32)
    embeddings[:, 0, :] = np.array([1.0, 0.0], dtype=np.float32)
    embeddings[:, 1, :] = np.array([1.0, 0.0], dtype=np.float32)
    embeddings[:, 2, :] = np.array([0.0, 1.0], dtype=np.float32)
    label_start = tmp_path / "start.npy"
    label_end = tmp_path / "end.npy"
    pred_path = tmp_path / "paper58.npy"
    embedding_path = tmp_path / "embedding_start.npy"
    np.save(label_start, start)
    np.save(label_end, end)
    np.save(pred_path, paper58_pred)
    np.save(embedding_path, embeddings)
    registry = tmp_path / "benchmark_registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "latent_neighborhood",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Urban",
                        **_provenance_fields(),
                        "label_start_path": str(label_start),
                        "label_end_path": str(label_end),
                        "prediction_path": str(pred_path),
                        "embedding_start_path": str(embedding_path),
                        "qc_status": "include",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_las(
        registry_path=registry,
        output_dir=tmp_path / "las_out",
        latent_neighborhood_weight=1.0,
    )

    assert result["summary"]["n_evaluated_rows"] == 1


def test_evaluate_las_metric_rows_keep_temporal_pair_keys(tmp_path: Path):
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
    rows = []
    for start_year, end_year in [(2020, 2021), (2021, 2022)]:
        label_start = tmp_path / f"start_{start_year}.npy"
        label_end = tmp_path / f"end_{end_year}.npy"
        pred_path = tmp_path / f"paper58_{start_year}_{end_year}.npy"
        np.save(label_start, start)
        np.save(label_end, end)
        np.save(pred_path, pred)
        rows.append(
            {
                "area": "repeat_area",
                "start_year": start_year,
                "end_year": end_year,
                "tier": "tier1",
                "stratum": "Wetland",
                **_provenance_fields(),
                "label_start_path": str(label_start),
                "label_end_path": str(label_end),
                "prediction_path": str(pred_path),
                "qc_status": "include",
            }
        )
    registry = tmp_path / "benchmark_registry.json"
    registry.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    output_dir = tmp_path / "las_out"

    evaluate_las(registry_path=registry, output_dir=output_dir)

    with (output_dir / "las_metrics_by_method.csv").open(encoding="utf-8") as f:
        metric_rows = list(csv.DictReader(f))
    pairs = {(row["area"], row["start_year"], row["end_year"]) for row in metric_rows}
    assert ("repeat_area", "2020", "2021") in pairs
    assert ("repeat_area", "2021", "2022") in pairs
    summary_payload = json.loads((output_dir / "las_summary.json").read_text(encoding="utf-8"))
    json_pairs = {
        (row["area"], row["start_year"], row["end_year"])
        for row in summary_payload["metrics"]
    }
    assert ("repeat_area", 2020, 2021) in json_pairs
    assert ("repeat_area", 2021, 2022) in json_pairs


def test_evaluate_las_relocates_cross_machine_registry_paths(tmp_path: Path, monkeypatch):
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
    label_dir = tmp_path / "data" / "independent_change_labels" / "labels"
    pred_dir = tmp_path / "data" / "independent_change_labels" / "predicted"
    label_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)
    label_start = label_dir / "relocated_lulc_2020.npy"
    label_end = label_dir / "relocated_lulc_2021.npy"
    pred_path = pred_dir / "relocated_lulc_pred_2020_2021.npy"
    np.save(label_start, start)
    np.save(label_end, end)
    np.save(pred_path, pred)
    monkeypatch.chdir(tmp_path)
    registry = tmp_path / "benchmark_registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "relocated",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Wetland",
                        **_provenance_fields(),
                        "label_start_path": "D:/other/machine/data/independent_change_labels/labels/relocated_lulc_2020.npy",
                        "label_end_path": "D:/other/machine/data/independent_change_labels/labels/relocated_lulc_2021.npy",
                        "prediction_path": "D:/other/machine/data/independent_change_labels/predicted/relocated_lulc_pred_2020_2021.npy",
                        "qc_status": "include",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_las(registry_path=registry, output_dir=tmp_path / "las_out")

    assert result["summary"]["n_evaluated_rows"] == 1


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


def test_evaluate_las_keeps_include_runtime_failures_visible(tmp_path: Path):
    registry = tmp_path / "benchmark_registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "missing_inputs",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Urban",
                        **_provenance_fields(),
                        "label_start_path": str(tmp_path / "missing_start.npy"),
                        "label_end_path": str(tmp_path / "missing_end.npy"),
                        "prediction_path": str(tmp_path / "missing_pred.npy"),
                        "qc_status": "include",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "las_out"

    result = evaluate_las(registry_path=registry, output_dir=output_dir)

    assert result["summary"]["n_evaluated_rows"] == 0
    assert result["summary"]["n_failed_rows"] == 1
    failures = (output_dir / "las_failures.csv").read_text(encoding="utf-8")
    assert "missing_inputs" in failures
    assert "runtime_failure" in failures


def test_evaluate_las_keeps_late_row_runtime_failures_visible(tmp_path: Path):
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 3]], dtype=np.int32)
    paper58_pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
    label_start = tmp_path / "start.npy"
    label_end = tmp_path / "end.npy"
    pred_path = tmp_path / "paper58.npy"
    np.save(label_start, start)
    np.save(label_end, end)
    np.save(pred_path, paper58_pred)
    registry = tmp_path / "benchmark_registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "bad_year",
                        "start_year": "not_a_year",
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Urban",
                        **_provenance_fields(),
                        "label_start_path": str(label_start),
                        "label_end_path": str(label_end),
                        "prediction_path": str(pred_path),
                        "qc_status": "include",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "las_out"

    result = evaluate_las(registry_path=registry, output_dir=output_dir)

    assert result["summary"]["n_evaluated_rows"] == 0
    assert result["summary"]["n_failed_rows"] == 1
    failures = (output_dir / "las_failures.csv").read_text(encoding="utf-8")
    assert "bad_year" in failures
    assert "runtime_failure" in failures
