import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.paper58_benchmark.evaluate_las import _row_neighborhood_weight, evaluate_las


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


def test_evaluate_las_can_use_demand_delta_change_budget(tmp_path: Path):
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
                        "area": "demand_delta_budget",
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

    result = evaluate_las(
        registry_path=registry,
        output_dir=output_dir,
        change_budget_source="demand_delta",
        change_budget_scale=0.5,
    )

    simulated = np.load(output_dir / "simulated" / "demand_delta_budget_2020_2021_paper58_las.npy")
    assert simulated.tolist() == [[1, 2]]
    assert result["summary"]["change_budget_source"] == "demand_delta"
    assert result["summary"]["change_budget_scale"] == 0.5


def test_evaluate_las_accepts_adaptive_change_budget_gate(tmp_path: Path):
    start = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int32)
    end = start.copy()
    paper58_pred = np.array([[2, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int32)

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
                        "area": "adaptive_budget",
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

    result = evaluate_las(
        registry_path=registry,
        output_dir=tmp_path / "las_out",
        change_budget_scale=0.82,
        adaptive_change_budget_scale=0.85,
        adaptive_change_budget_fraction_low=0.13,
        adaptive_change_budget_fraction_high=0.30,
    )

    assert result["summary"]["adaptive_change_budget_scale"] == 0.85
    assert result["summary"]["adaptive_change_budget_fraction_low"] == 0.13
    assert result["summary"]["adaptive_change_budget_fraction_high"] == 0.3


def test_evaluate_las_accepts_adaptive_churn_budget_gate(tmp_path: Path):
    start = np.array([[5, 5, 7, 7, 1]], dtype=np.int32)
    end = start.copy()
    paper58_pred = np.array([[7, 7, 5, 5, 1]], dtype=np.int32)

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
                        "area": "adaptive_churn",
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

    result = evaluate_las(
        registry_path=registry,
        output_dir=tmp_path / "las_out",
        adaptive_churn_budget_scale=0.55,
        adaptive_churn_fraction_high=0.75,
    )

    assert result["summary"]["adaptive_churn_budget_scale"] == 0.55
    assert result["summary"]["adaptive_churn_fraction_high"] == 0.75


def test_evaluate_las_passes_balanced_swap_min_margin(tmp_path: Path):
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
                        "area": "margin_budget",
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

    result = evaluate_las(
        registry_path=registry,
        output_dir=output_dir,
        balanced_swap_min_margin=9.0,
    )

    simulated = np.load(output_dir / "simulated" / "margin_budget_2020_2021_paper58_las.npy")
    assert simulated.tolist() == [[1, 2]]
    assert result["summary"]["balanced_swap_min_margin"] == 9.0


def test_evaluate_las_passes_balanced_swap_min_base_score(tmp_path: Path):
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
                        "area": "base_score_budget",
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

    result = evaluate_las(
        registry_path=registry,
        output_dir=output_dir,
        balanced_swap_min_base_score=9.0,
    )

    simulated = np.load(output_dir / "simulated" / "base_score_budget_2020_2021_paper58_las.npy")
    assert simulated.tolist() == [[1, 2]]
    assert result["summary"]["balanced_swap_min_base_score"] == 9.0


def test_evaluate_las_passes_balanced_swap_min_side_base_score(tmp_path: Path):
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
                        "area": "side_base_score_budget",
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

    result = evaluate_las(
        registry_path=registry,
        output_dir=output_dir,
        balanced_swap_min_side_base_score=9.0,
    )

    simulated = np.load(output_dir / "simulated" / "side_base_score_budget_2020_2021_paper58_las.npy")
    assert simulated.tolist() == [[1, 2]]
    assert result["summary"]["balanced_swap_min_side_base_score"] == 9.0


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


def test_evaluate_las_accepts_adaptive_neighborhood_weight(tmp_path: Path):
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
                        "area": "adaptive_neighborhood",
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

    result = evaluate_las(
        registry_path=registry,
        output_dir=tmp_path / "las_out",
        neighborhood_weight=0.5,
        adaptive_neighborhood_weight=2.5,
        adaptive_change_fraction_low=0.05,
        adaptive_change_fraction_high=0.20,
    )

    assert result["summary"]["n_evaluated_rows"] == 1
    assert result["summary"]["adaptive_neighborhood_weight"] == 2.5
    assert result["summary"]["adaptive_change_fraction_low"] == 0.05
    assert result["summary"]["adaptive_change_fraction_high"] == 0.2


def test_row_neighborhood_weight_switches_on_change_fraction_range():
    assert _row_neighborhood_weight(10, 100, 2.0, 2.5, 0.10, 0.20) == 2.5
    assert _row_neighborhood_weight(19, 100, 2.0, 2.5, 0.10, 0.20) == 2.5
    assert _row_neighborhood_weight(9, 100, 2.0, 2.5, 0.10, 0.20) == 2.0
    assert _row_neighborhood_weight(20, 100, 2.0, 2.5, 0.10, 0.20) == 2.0
    assert _row_neighborhood_weight(None, 100, 2.0, 2.5, 0.10, 0.20) == 2.0
    assert _row_neighborhood_weight(10, 0, 2.0, 2.5, 0.10, 0.20) == 2.0


@pytest.mark.parametrize(
    ("low", "high", "message"),
    [
        (-0.01, 0.20, "bounds must be in \\[0, 1\\]"),
        (0.10, 1.01, "bounds must be in \\[0, 1\\]"),
        (0.20, 0.20, "must be smaller"),
        (0.30, 0.20, "must be smaller"),
    ],
)
def test_row_neighborhood_weight_rejects_invalid_adaptive_bounds(low: float, high: float, message: str):
    with pytest.raises(ValueError, match=message):
        _row_neighborhood_weight(10, 100, 2.0, 2.5, low, high)


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


def test_evaluate_las_uses_embedding_change_pressure_for_suitability(tmp_path: Path):
    start = np.array([[1, 1]], dtype=np.int32)
    end = np.array([[1, 2]], dtype=np.int32)
    paper58_pred = np.array([[1, 1]], dtype=np.int32)
    embedding_start = np.zeros((1, 2, 1), dtype=np.float32)
    embedding_end = np.array([[[0.0], [5.0]]], dtype=np.float32)

    label_start = tmp_path / "start.npy"
    label_end = tmp_path / "end.npy"
    pred_path = tmp_path / "paper58.npy"
    embedding_start_path = tmp_path / "embedding_start.npy"
    embedding_end_path = tmp_path / "embedding_end.npy"
    np.save(label_start, start)
    np.save(label_end, end)
    np.save(pred_path, paper58_pred)
    np.save(embedding_start_path, embedding_start)
    np.save(embedding_end_path, embedding_end)

    registry = tmp_path / "benchmark_registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "embedding_pressure",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Urban",
                        **_provenance_fields(),
                        "label_start_path": str(label_start),
                        "label_end_path": str(label_end),
                        "prediction_path": str(pred_path),
                        "embedding_start_path": str(embedding_start_path),
                        "embedding_end_path": str(embedding_end_path),
                        "qc_status": "include",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "las_out"

    result = evaluate_las(
        registry_path=registry,
        output_dir=output_dir,
        suitability_change_pressure_weight=1.0,
    )

    simulated = np.load(output_dir / "simulated" / "embedding_pressure_2020_2021_paper58_las.npy")
    assert simulated.tolist() == [[1, 2]]
    assert result["summary"]["suitability_change_pressure_weight"] == 1.0


def test_evaluate_las_accepts_suitability_score_weights(tmp_path: Path):
    start = np.array([[1, 1]], dtype=np.int32)
    end = np.array([[1, 2]], dtype=np.int32)
    paper58_pred = np.array([[1, 2]], dtype=np.int32)

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
                        "area": "suitability_weights",
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

    result = evaluate_las(
        registry_path=registry,
        output_dir=tmp_path / "las_out",
        suitability_forecast_prob_weight=0.75,
        suitability_probability_gain_weight=0.25,
        suitability_transition_prior_weight=0.10,
    )

    assert result["summary"]["suitability_forecast_prob_weight"] == 0.75
    assert result["summary"]["suitability_probability_gain_weight"] == 0.25
    assert result["summary"]["suitability_transition_prior_weight"] == 0.10


def test_evaluate_las_can_use_paper58_prediction_demand_without_observed_end(tmp_path: Path):
    start = np.array([[1, 1, 2]], dtype=np.int32)
    end = np.array([[2, 2, 2]], dtype=np.int32)
    paper58_pred = np.array([[1, 1, 2]], dtype=np.int32)
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
                        "area": "non_oracle_demand",
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

    result = evaluate_las(
        registry_path=registry,
        output_dir=output_dir,
        demand_source="paper58_prediction",
    )

    simulated = np.load(output_dir / "simulated" / "non_oracle_demand_2020_2021_paper58_las.npy")
    values, counts = np.unique(simulated, return_counts=True)
    assert dict(zip(values.tolist(), counts.tolist())) == {1: 2, 2: 1}
    assert result["summary"]["demand_source"] == "paper58_prediction"


def test_evaluate_las_accepts_transition_prior_demand(tmp_path: Path):
    rows = []
    for area, start, end, pred in [
        (
            "transition_target_a",
            np.array([[1, 1], [2, 2]], dtype=np.int32),
            np.array([[1, 2], [2, 2]], dtype=np.int32),
            np.array([[1, 2], [2, 2]], dtype=np.int32),
        ),
        (
            "transition_target_b",
            np.array([[1, 1], [2, 2]], dtype=np.int32),
            np.array([[2, 2], [1, 2]], dtype=np.int32),
            np.array([[2, 2], [1, 2]], dtype=np.int32),
        ),
    ]:
        start_path = tmp_path / f"{area}_start.npy"
        end_path = tmp_path / f"{area}_end.npy"
        pred_path = tmp_path / f"{area}_pred.npy"
        np.save(start_path, start)
        np.save(end_path, end)
        np.save(pred_path, pred)
        rows.append(
            {
                "area": area,
                "start_year": 2020,
                "end_year": 2021,
                "tier": "tier1",
                "stratum": "Urban",
                **_provenance_fields(),
                "label_start_path": str(start_path),
                "label_end_path": str(end_path),
                "prediction_path": str(pred_path),
                "qc_status": "include",
            }
        )
    registry = tmp_path / "benchmark_registry.json"
    registry.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    result = evaluate_las(
        registry_path=registry,
        output_dir=tmp_path / "las_out",
        demand_source="transition_prior",
    )

    assert result["summary"]["n_evaluated_rows"] == 2
    assert result["summary"]["demand_source"] == "transition_prior"


def test_evaluate_las_accepts_transition_prior_blend_demand(tmp_path: Path):
    rows = []
    for area, start, end, pred in [
        (
            "blend_target_a",
            np.array([[1, 1], [2, 2]], dtype=np.int32),
            np.array([[1, 2], [2, 2]], dtype=np.int32),
            np.array([[1, 1], [1, 2]], dtype=np.int32),
        ),
        (
            "blend_target_b",
            np.array([[1, 1], [2, 2]], dtype=np.int32),
            np.array([[2, 2], [1, 2]], dtype=np.int32),
            np.array([[2, 2], [2, 2]], dtype=np.int32),
        ),
    ]:
        start_path = tmp_path / f"{area}_start.npy"
        end_path = tmp_path / f"{area}_end.npy"
        pred_path = tmp_path / f"{area}_pred.npy"
        np.save(start_path, start)
        np.save(end_path, end)
        np.save(pred_path, pred)
        rows.append(
            {
                "area": area,
                "start_year": 2020,
                "end_year": 2021,
                "tier": "tier1",
                "stratum": "Urban",
                **_provenance_fields(),
                "label_start_path": str(start_path),
                "label_end_path": str(end_path),
                "prediction_path": str(pred_path),
                "qc_status": "include",
            }
        )
    registry = tmp_path / "benchmark_registry.json"
    registry.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    result = evaluate_las(
        registry_path=registry,
        output_dir=tmp_path / "las_out",
        demand_source="transition_prior_blend",
        demand_blend_weight=0.5,
    )

    assert result["summary"]["n_evaluated_rows"] == 2
    assert result["summary"]["demand_source"] == "transition_prior_blend"
    assert result["summary"]["demand_blend_weight"] == 0.5


def test_evaluate_las_accepts_transition_prior_adaptive_blend_demand(tmp_path: Path):
    rows = []
    for area, start, end, pred in [
        (
            "adaptive_blend_target_a",
            np.array([[1, 1], [2, 2]], dtype=np.int32),
            np.array([[1, 2], [2, 2]], dtype=np.int32),
            np.array([[1, 1], [1, 2]], dtype=np.int32),
        ),
        (
            "adaptive_blend_target_b",
            np.array([[1, 1], [2, 2]], dtype=np.int32),
            np.array([[2, 2], [1, 2]], dtype=np.int32),
            np.array([[2, 2], [2, 2]], dtype=np.int32),
        ),
    ]:
        start_path = tmp_path / f"{area}_start.npy"
        end_path = tmp_path / f"{area}_end.npy"
        pred_path = tmp_path / f"{area}_pred.npy"
        np.save(start_path, start)
        np.save(end_path, end)
        np.save(pred_path, pred)
        rows.append(
            {
                "area": area,
                "start_year": 2020,
                "end_year": 2021,
                "tier": "tier1",
                "stratum": "Urban",
                **_provenance_fields(),
                "label_start_path": str(start_path),
                "label_end_path": str(end_path),
                "prediction_path": str(pred_path),
                "qc_status": "include",
            }
        )
    registry = tmp_path / "benchmark_registry.json"
    registry.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    result = evaluate_las(
        registry_path=registry,
        output_dir=tmp_path / "las_out",
        demand_source="transition_prior_adaptive_blend",
        demand_blend_weight=0.75,
        adaptive_demand_l1_threshold=0.14,
        adaptive_demand_change_fraction_high=0.22,
    )

    assert result["summary"]["n_evaluated_rows"] == 2
    assert result["summary"]["demand_source"] == "transition_prior_adaptive_blend"
    assert result["summary"]["demand_blend_weight"] == 0.75
    assert result["summary"]["adaptive_demand_l1_threshold"] == 0.14
    assert result["summary"]["adaptive_demand_change_fraction_high"] == 0.22


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
