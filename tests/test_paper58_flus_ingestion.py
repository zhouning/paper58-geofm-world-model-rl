from pathlib import Path

import numpy as np
import pytest

from scripts.paper58_benchmark.flus import FLUSIngestionError, load_flus_prediction


def test_load_flus_prediction_accepts_npy(tmp_path: Path):
    path = tmp_path / "flus.npy"
    arr = np.array([[1, 2], [2, 3]], dtype=np.int32)
    np.save(path, arr)

    loaded = load_flus_prediction(path, expected_shape=(2, 2), allowed_classes={1, 2, 3})

    assert np.array_equal(loaded, arr)


def test_load_flus_prediction_accepts_csv(tmp_path: Path):
    path = tmp_path / "flus.csv"
    path.write_text("1,2\n2,3\n", encoding="utf-8")

    loaded = load_flus_prediction(path, expected_shape=(2, 2), allowed_classes={1, 2, 3})

    assert loaded.dtype == np.int32
    assert loaded.tolist() == [[1, 2], [2, 3]]


def test_load_flus_prediction_accepts_geotiff(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    path = tmp_path / "flus.tif"
    arr = np.array([[1, 2], [2, 3]], dtype=np.int32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        transform=rasterio.transform.from_origin(0, arr.shape[0], 1, 1),
    ) as dataset:
        dataset.write(arr, 1)

    loaded = load_flus_prediction(path, expected_shape=(2, 2), allowed_classes={1, 2, 3})

    assert loaded.dtype == np.int32
    assert loaded.tolist() == [[1, 2], [2, 3]]


def test_load_flus_prediction_accepts_single_row_csv(tmp_path: Path):
    path = tmp_path / "flus.csv"
    path.write_text("1,2\n", encoding="utf-8")

    loaded = load_flus_prediction(path, expected_shape=(1, 2), allowed_classes={1, 2})

    assert loaded.tolist() == [[1, 2]]


def test_load_flus_prediction_rejects_shape_mismatch(tmp_path: Path):
    path = tmp_path / "flus.npy"
    np.save(path, np.ones((2, 3), dtype=np.int32))

    with pytest.raises(FLUSIngestionError, match="FLUS prediction shape"):
        load_flus_prediction(path, expected_shape=(2, 2), allowed_classes={1})


def test_load_flus_prediction_rejects_unknown_classes(tmp_path: Path):
    path = tmp_path / "flus.npy"
    np.save(path, np.array([[1, 9]], dtype=np.int32))

    with pytest.raises(FLUSIngestionError, match="unknown FLUS classes: \\[9\\]"):
        load_flus_prediction(path, expected_shape=(1, 2), allowed_classes={1, 2})


def test_load_flus_prediction_rejects_non_integral_labels(tmp_path: Path):
    path = tmp_path / "flus.npy"
    np.save(path, np.array([[1.0, 1.9]], dtype=np.float32))

    with pytest.raises(FLUSIngestionError, match="non-integer FLUS labels"):
        load_flus_prediction(path, expected_shape=(1, 2), allowed_classes={1, 2})
