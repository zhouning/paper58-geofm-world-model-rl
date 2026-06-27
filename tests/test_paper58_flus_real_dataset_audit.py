from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")


def _write_tif(path: Path, array: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        transform=rasterio.transform.from_origin(100.0, 200.0, 80.0, 80.0),
        crs="EPSG:32649",
    ) as dataset:
        dataset.write(array, 1)


def test_landuse_raster_summary_reads_grid_and_counts(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.audit_flus_real_dataset import summarize_landuse_raster

    path = tmp_path / "landuse2005.tif"
    _write_tif(path, np.array([[1, 1, 5], [2, 5, 5]], dtype=np.uint8))

    summary = summarize_landuse_raster(path, year=2005, class_names={1: "Arable", 2: "Woodland", 5: "Construction"})

    assert summary["year"] == 2005
    assert summary["width"] == 3
    assert summary["height"] == 2
    assert summary["pixel_count"] == 6
    assert summary["crs"] == "EPSG:32649"
    assert summary["class_counts"] == [
        {"class_value": 1, "class_name": "Arable", "pixels": 2, "share": pytest.approx(2 / 6)},
        {"class_value": 2, "class_name": "Woodland", "pixels": 1, "share": pytest.approx(1 / 6)},
        {"class_value": 5, "class_name": "Construction", "pixels": 3, "share": pytest.approx(3 / 6)},
    ]


def test_landuse_raster_summary_excludes_zero_outside_boundary(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.audit_flus_real_dataset import summarize_landuse_raster

    path = tmp_path / "landuse2005.tif"
    _write_tif(path, np.array([[0, 1, 5], [0, 5, 5]], dtype=np.uint8))

    summary = summarize_landuse_raster(path, year=2005, class_names={1: "Arable", 5: "Construction"})

    assert summary["pixel_count"] == 6
    assert summary["valid_pixel_count"] == 4
    assert summary["outside_pixel_count"] == 2
    assert summary["class_counts"] == [
        {"class_value": 1, "class_name": "Arable", "pixels": 1, "share": pytest.approx(1 / 4)},
        {"class_value": 5, "class_name": "Construction", "pixels": 3, "share": pytest.approx(3 / 4)},
    ]


def test_transition_counts_report_changed_and_persistent_pixels() -> None:
    from scripts.paper58_benchmark.audit_flus_real_dataset import transition_count_rows

    start = np.array([[1, 1, 5], [2, 5, 5]], dtype=np.uint8)
    end = np.array([[1, 5, 5], [2, 1, 3]], dtype=np.uint8)

    rows = transition_count_rows(start, end, 2000, 2005, class_names={1: "Arable", 2: "Woodland", 3: "Meadow", 5: "Construction"})

    assert rows == [
        {
            "period": "2000_2005",
            "from_class": 1,
            "from_name": "Arable",
            "to_class": 1,
            "to_name": "Arable",
            "pixels": 1,
            "changed": False,
        },
        {
            "period": "2000_2005",
            "from_class": 1,
            "from_name": "Arable",
            "to_class": 5,
            "to_name": "Construction",
            "pixels": 1,
            "changed": True,
        },
        {
            "period": "2000_2005",
            "from_class": 2,
            "from_name": "Woodland",
            "to_class": 2,
            "to_name": "Woodland",
            "pixels": 1,
            "changed": False,
        },
        {
            "period": "2000_2005",
            "from_class": 5,
            "from_name": "Construction",
            "to_class": 1,
            "to_name": "Arable",
            "pixels": 1,
            "changed": True,
        },
        {
            "period": "2000_2005",
            "from_class": 5,
            "from_name": "Construction",
            "to_class": 3,
            "to_name": "Meadow",
            "pixels": 1,
            "changed": True,
        },
        {
            "period": "2000_2005",
            "from_class": 5,
            "from_name": "Construction",
            "to_class": 5,
            "to_name": "Construction",
            "pixels": 1,
            "changed": False,
        },
    ]


def test_transition_counts_exclude_zero_outside_boundary() -> None:
    from scripts.paper58_benchmark.audit_flus_real_dataset import transition_count_rows, transition_summary

    start = np.array([[0, 1, 5], [2, 0, 5]], dtype=np.uint8)
    end = np.array([[0, 5, 5], [2, 1, 3]], dtype=np.uint8)

    rows = transition_count_rows(start, end, 2000, 2005, class_names={1: "Arable", 2: "Woodland", 3: "Meadow", 5: "Construction"})
    summary = transition_summary(start, end, 2000, 2005)

    assert [row["pixels"] for row in rows] == [1, 1, 1, 1]
    assert {row["from_class"] for row in rows} == {1, 2, 5}
    assert {row["to_class"] for row in rows} == {2, 3, 5}
    assert summary["n_pixels"] == 4
    assert summary["changed_pixels"] == 2
    assert summary["changed_share"] == pytest.approx(0.5)


def test_protocol_summary_uses_earliest_pair_for_calibration_and_latest_pair_for_validation() -> None:
    from scripts.paper58_benchmark.audit_flus_real_dataset import comparison_protocol

    protocol = comparison_protocol([2000, 2005, 2006])

    assert protocol["calibration"] == {
        "start_year": 2000,
        "end_year": 2005,
        "role": "calibrate transition demand and suitability from real FLUS-style inputs",
    }
    assert protocol["validation"] == {
        "start_year": 2005,
        "end_year": 2006,
        "role": "matched Paper58 vs GeoSOS-FLUS simulation target",
    }
