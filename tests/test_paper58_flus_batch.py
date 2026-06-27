import json
from pathlib import Path

import numpy as np
import pytest

from scripts.paper58_benchmark.run_flus_batch import run_flus_batch


def _provenance_fields() -> dict:
    return {
        "bbox": [120.0, 30.0, 120.1, 30.1],
        "data_source": "test_source",
        "development_contact_status": "none",
        "contact_evidence": "synthetic no-contact evidence",
        "expected_role": "positive_change_candidate",
    }


def _write_geotiff(path: Path, array: np.ndarray) -> None:
    rasterio = pytest.importorskip("rasterio")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        transform=rasterio.transform.from_origin(0, array.shape[0], 1, 1),
    ) as dataset:
        dataset.write(array, 1)


def test_run_flus_batch_decodes_console_output_to_registry_prediction_name(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    start = np.array([[1, 5], [1, 5]], dtype=np.int32)
    end = np.array([[5, 5], [1, 1]], dtype=np.int32)
    pred = np.array([[5, 5], [1, 1]], dtype=np.int32)
    start_path = tmp_path / "start.npy"
    end_path = tmp_path / "end.npy"
    pred_path = tmp_path / "pred.npy"
    np.save(start_path, start)
    np.save(end_path, end)
    np.save(pred_path, pred)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "noncontiguous",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Wetland",
                        **_provenance_fields(),
                        "label_start_path": str(start_path),
                        "label_end_path": str(end_path),
                        "prediction_path": str(pred_path),
                        "qc_status": "include",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_console(case_dir: Path) -> None:
        _write_geotiff(case_dir / "simresult_2021.tif", np.array([[2, 2], [1, 1]], dtype=np.int32))

    summary = run_flus_batch(
        registry_path=registry,
        case_root=tmp_path / "cases",
        prediction_dir=tmp_path / "predictions",
        flus_executable=tmp_path / "flus_console",
        console_runner=fake_console,
    )

    prediction_path = tmp_path / "predictions" / "noncontiguous_2020_2021_flus.tif"
    assert summary["n_rows"] == 1
    assert summary["n_ran"] == 1
    assert summary["n_failed"] == 0
    assert summary["demand_source"] == "observed_end"
    assert summary["failures"] == []
    assert prediction_path.exists()
    with rasterio.open(prediction_path) as dataset:
        assert dataset.read(1).tolist() == [[5, 5], [1, 1]]


def test_run_flus_batch_skips_excluded_rows(tmp_path: Path):
    start = np.array([[1]], dtype=np.int32)
    start_path = tmp_path / "start.npy"
    np.save(start_path, start)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "excluded",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Wetland",
                        **_provenance_fields(),
                        "label_start_path": str(start_path),
                        "label_end_path": str(start_path),
                        "prediction_path": str(start_path),
                        "qc_status": "exclude",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    calls: list[Path] = []

    summary = run_flus_batch(
        registry_path=registry,
        case_root=tmp_path / "cases",
        prediction_dir=tmp_path / "predictions",
        flus_executable=tmp_path / "flus_console",
        console_runner=calls.append,
    )

    assert calls == []
    assert summary["n_rows"] == 0
    assert summary["n_ran"] == 0


def test_run_flus_batch_can_use_paper58_prediction_demand(tmp_path: Path):
    start = np.array([[1, 5], [1, 5]], dtype=np.int32)
    end = np.array([[5, 5], [5, 5]], dtype=np.int32)
    pred = np.array([[1, 1], [1, 5]], dtype=np.int32)
    start_path = tmp_path / "start.npy"
    end_path = tmp_path / "end.npy"
    pred_path = tmp_path / "pred.npy"
    np.save(start_path, start)
    np.save(end_path, end)
    np.save(pred_path, pred)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "non_oracle_flus",
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
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_console(case_dir: Path) -> None:
        assert (case_dir / "CCregionMakovChain.csv").read_text(encoding="utf-8") == "year,type1,type2\n2021,3,1\n"
        _write_geotiff(case_dir / "simresult.tif", np.array([[1, 1], [1, 2]], dtype=np.int32))

    summary = run_flus_batch(
        registry_path=registry,
        case_root=tmp_path / "cases",
        prediction_dir=tmp_path / "predictions",
        flus_executable=tmp_path / "flus_console",
        demand_source="paper58_prediction",
        console_runner=fake_console,
    )

    assert summary["n_ran"] == 1


def test_run_flus_batch_can_use_transition_prior_blend_demand(tmp_path: Path):
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 2]], dtype=np.int32)
    pred = np.array([[1, 1], [1, 2]], dtype=np.int32)
    start_path = tmp_path / "start.npy"
    end_path = tmp_path / "end.npy"
    pred_path = tmp_path / "pred.npy"
    np.save(start_path, start)
    np.save(end_path, end)
    np.save(pred_path, pred)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "blend_flus",
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
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_console(case_dir: Path) -> None:
        assert (case_dir / "CCregionMakovChain.csv").read_text(encoding="utf-8") == "year,type1,type2\n2021,3,1\n"
        _write_geotiff(case_dir / "simresult.tif", np.array([[1, 1], [1, 2]], dtype=np.int32))

    summary = run_flus_batch(
        registry_path=registry,
        case_root=tmp_path / "cases",
        prediction_dir=tmp_path / "predictions",
        flus_executable=tmp_path / "flus_console",
        demand_source="transition_prior_blend",
        demand_blend_weight=0.5,
        console_runner=fake_console,
    )

    assert summary["n_ran"] == 1
    assert summary["demand_source"] == "transition_prior_blend"
    assert summary["demand_blend_weight"] == 0.5


def test_run_flus_batch_can_use_transition_prior_adaptive_blend_demand(tmp_path: Path):
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 2]], dtype=np.int32)
    pred = np.array([[1, 1], [1, 2]], dtype=np.int32)
    start_path = tmp_path / "start.npy"
    end_path = tmp_path / "end.npy"
    pred_path = tmp_path / "pred.npy"
    np.save(start_path, start)
    np.save(end_path, end)
    np.save(pred_path, pred)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "adaptive_blend_flus",
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
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_console(case_dir: Path) -> None:
        assert (case_dir / "CCregionMakovChain.csv").read_text(encoding="utf-8") == "year,type1,type2\n2021,3,1\n"
        _write_geotiff(case_dir / "simresult.tif", np.array([[1, 1], [1, 2]], dtype=np.int32))

    summary = run_flus_batch(
        registry_path=registry,
        case_root=tmp_path / "cases",
        prediction_dir=tmp_path / "predictions",
        flus_executable=tmp_path / "flus_console",
        demand_source="transition_prior_adaptive_blend",
        demand_blend_weight=0.75,
        adaptive_demand_l1_threshold=0.14,
        adaptive_demand_change_fraction_high=0.30,
        console_runner=fake_console,
    )

    assert summary["n_ran"] == 1
    assert summary["demand_source"] == "transition_prior_adaptive_blend"
    assert summary["demand_blend_weight"] == 0.75
    assert summary["adaptive_demand_l1_threshold"] == 0.14
    assert summary["adaptive_demand_change_fraction_high"] == 0.30
