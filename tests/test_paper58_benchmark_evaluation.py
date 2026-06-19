import json
from pathlib import Path

import numpy as np
import pytest

from scripts.paper58_benchmark.evaluate_benchmark import (
    binary_change_metrics,
    evaluate_benchmark,
    evaluate_registry_row,
    _read_registry,
    _transition_training_pairs,
)


def test_binary_change_metrics_reports_f1():
    true_change = np.array([[False, True], [True, False]])
    pred_change = np.array([[False, True], [False, True]])

    result = binary_change_metrics(true_change, pred_change)

    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_evaluate_registry_row_computes_primary_and_spatial_advantages(tmp_path: Path):
    labels = tmp_path / "labels"
    predicted = tmp_path / "predicted"
    embeddings = tmp_path / "embeddings"
    labels.mkdir()
    predicted.mkdir()
    embeddings.mkdir()
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
    np.save(labels / "external_lulc_2020.npy", start)
    np.save(labels / "external_lulc_2021.npy", end)
    np.save(predicted / "external_lulc_pred_2020_2021.npy", pred)
    np.save(embeddings / "external_emb_2020.npy", np.zeros((2, 2, 64), dtype=np.float32))
    np.save(embeddings / "external_emb_2021.npy", np.ones((2, 2, 64), dtype=np.float32))

    row = {
        "area": "external",
        "start_year": 2020,
        "end_year": 2021,
        "tier": "tier1",
        "stratum": "Wetland",
        "label_start_path": str(labels / "external_lulc_2020.npy"),
        "label_end_path": str(labels / "external_lulc_2021.npy"),
        "prediction_path": str(predicted / "external_lulc_pred_2020_2021.npy"),
        "embedding_start_path": str(embeddings / "external_emb_2020.npy"),
        "embedding_end_path": str(embeddings / "external_emb_2021.npy"),
        "qc_status": "include",
        "excluded_reason": "",
    }

    metrics = evaluate_registry_row(row, transition_training_pairs=[], temporal_training_rows=[])

    assert metrics["model_change_f1"] == pytest.approx(2 / 3)
    assert metrics["persistence_change_f1"] == pytest.approx(0.0)
    assert metrics["primary_change_advantage"] >= 0.0
    assert "spatial_change_advantage" in metrics
    assert "embedding_advantage" in metrics


def test_transition_training_pairs_excludes_all_same_area_rows(tmp_path: Path):
    target_start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    target_end = np.array([[1, 2], [2, 3]], dtype=np.int32)
    same_area_start = np.array([[3, 3], [4, 4]], dtype=np.int32)
    same_area_end = np.array([[3, 4], [4, 5]], dtype=np.int32)
    other_area_start = np.array([[5, 5], [6, 6]], dtype=np.int32)
    other_area_end = np.array([[5, 6], [6, 7]], dtype=np.int32)

    rows = []
    for area, start, end in [
        ("external", target_start, target_end),
        ("external", same_area_start, same_area_end),
        ("other", other_area_start, other_area_end),
    ]:
        start_path = tmp_path / f"{area}_{int(start[0, 0])}_start.npy"
        end_path = tmp_path / f"{area}_{int(end[0, 0])}_end.npy"
        np.save(start_path, start)
        np.save(end_path, end)
        rows.append(
            {
                "area": area,
                "start_year": 2020 if area == "external" else 2019,
                "end_year": 2021 if area == "external" else 2020,
                "qc_status": "include",
                "label_start_path": str(start_path),
                "label_end_path": str(end_path),
            }
        )

    target = rows[0]
    pairs = _transition_training_pairs(rows, target, target_start.shape)

    assert len(pairs) == 1
    assert np.array_equal(pairs[0][0], other_area_start)
    assert np.array_equal(pairs[0][1], other_area_end)


def test_read_registry_rejects_malformed_payloads(tmp_path: Path):
    registry_path = tmp_path / "benchmark_registry.json"
    registry_path.write_text(json.dumps([{"rows": []}]), encoding="utf-8")

    with pytest.raises(ValueError, match="benchmark_registry\\.json must contain an object"):
        _read_registry(registry_path)

    registry_path.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark_registry\\.json must contain a 'rows' list"):
        _read_registry(registry_path)

    registry_path.write_text(json.dumps({"rows": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark_registry\\.json must contain a 'rows' list"):
        _read_registry(registry_path)


def test_read_registry_rejects_non_dict_and_missing_required_fields(tmp_path: Path):
    registry_path = tmp_path / "benchmark_registry.json"
    registry_path.write_text(json.dumps({"rows": [1]}), encoding="utf-8")

    with pytest.raises(ValueError, match="registry row 0 must be an object"):
        _read_registry(registry_path)

    registry_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "external",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Wetland",
                        "label_start_path": "a.npy",
                        "label_end_path": "b.npy",
                        "prediction_path": "c.npy",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="registry row 0 missing required field: qc_status"):
        _read_registry(registry_path)


def test_evaluate_benchmark_writes_outputs_and_does_not_pool_tiers(tmp_path: Path):
    labels = tmp_path / "labels"
    predicted = tmp_path / "predicted"
    embeddings = tmp_path / "embeddings"
    output = tmp_path / "out"
    for path in (labels, predicted, embeddings, output):
        path.mkdir()

    for area, tier in [("external", "tier1"), ("bishan", "tier2")]:
        start = np.array([[1, 1], [2, 2]], dtype=np.int32)
        end = np.array([[1, 2], [2, 3]], dtype=np.int32)
        pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
        np.save(labels / f"{area}_lulc_2020.npy", start)
        np.save(labels / f"{area}_lulc_2021.npy", end)
        np.save(predicted / f"{area}_lulc_pred_2020_2021.npy", pred)
        np.save(embeddings / f"{area}_emb_2020.npy", np.zeros((2, 2, 64), dtype=np.float32))
        np.save(embeddings / f"{area}_emb_2021.npy", np.ones((2, 2, 64), dtype=np.float32))

    registry = {
        "rows": [
            {
                "area": area,
                "start_year": 2020,
                "end_year": 2021,
                "tier": tier,
                "stratum": "Wetland" if tier == "tier1" else "Mixed",
                "label_start_path": str(labels / f"{area}_lulc_2020.npy"),
                "label_end_path": str(labels / f"{area}_lulc_2021.npy"),
                "prediction_path": str(predicted / f"{area}_lulc_pred_2020_2021.npy"),
                "embedding_start_path": str(embeddings / f"{area}_emb_2020.npy"),
                "embedding_end_path": str(embeddings / f"{area}_emb_2021.npy"),
                "qc_status": "include",
                "excluded_reason": "",
            }
            for area, tier in [("external", "tier1"), ("bishan", "tier2")]
        ]
    }
    registry_path = output / "benchmark_registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    result = evaluate_benchmark(registry_path=registry_path, output_dir=output, n_boot=100)

    assert result["summary"]["n_evaluated"] == 2
    assert result["summary_by_tier"]["tier1"]["n"] == 1
    assert result["summary_by_tier"]["tier2"]["n"] == 1
    assert (output / "benchmark_metrics_by_pair.csv").exists()
    assert (output / "benchmark_gate_report.json").exists()


def test_evaluate_benchmark_keeps_same_stratum_separate_by_tier(tmp_path: Path):
    labels = tmp_path / "labels"
    predicted = tmp_path / "predicted"
    embeddings = tmp_path / "embeddings"
    output = tmp_path / "out"
    for path in (labels, predicted, embeddings, output):
        path.mkdir()

    for area, tier in [("external", "tier1"), ("bishan", "tier2")]:
        start = np.array([[1, 1], [2, 2]], dtype=np.int32)
        end = np.array([[1, 2], [2, 3]], dtype=np.int32)
        pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
        np.save(labels / f"{area}_lulc_2020.npy", start)
        np.save(labels / f"{area}_lulc_2021.npy", end)
        np.save(predicted / f"{area}_lulc_pred_2020_2021.npy", pred)
        np.save(embeddings / f"{area}_emb_2020.npy", np.zeros((2, 2, 64), dtype=np.float32))
        np.save(embeddings / f"{area}_emb_2021.npy", np.ones((2, 2, 64), dtype=np.float32))

    registry = {
        "rows": [
            {
                "area": area,
                "start_year": 2020,
                "end_year": 2021,
                "tier": tier,
                "stratum": "Mixed",
                "label_start_path": str(labels / f"{area}_lulc_2020.npy"),
                "label_end_path": str(labels / f"{area}_lulc_2021.npy"),
                "prediction_path": str(predicted / f"{area}_lulc_pred_2020_2021.npy"),
                "embedding_start_path": str(embeddings / f"{area}_emb_2020.npy"),
                "embedding_end_path": str(embeddings / f"{area}_emb_2021.npy"),
                "qc_status": "include",
                "excluded_reason": "",
            }
            for area, tier in [("external", "tier1"), ("bishan", "tier2")]
        ]
    }
    registry_path = output / "benchmark_registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    result = evaluate_benchmark(registry_path=registry_path, output_dir=output, n_boot=100)

    assert set(result["summary_by_stratum"]) == {"tier1", "tier2"}
    assert set(result["summary_by_stratum"]["tier1"]) == {"Mixed"}
    assert set(result["summary_by_stratum"]["tier2"]) == {"Mixed"}

    csv_lines = (output / "benchmark_summary_by_stratum.csv").read_text(encoding="utf-8").splitlines()
    header = csv_lines[0].split(",")
    assert header[:2] == ["tier", "stratum"]
    assert len(csv_lines) == 3
