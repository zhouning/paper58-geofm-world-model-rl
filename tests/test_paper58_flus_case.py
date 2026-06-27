from pathlib import Path

import numpy as np
import pytest

from scripts.paper58_benchmark.flus_case import (
    FLUSCaseError,
    decode_flus_geotiff,
    decode_flus_labels,
    find_flus_simulation_result,
    write_flus_case,
)


def test_write_flus_case_exports_console_inputs(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    start = np.array([[1, 2], [1, 2]], dtype=np.int32)
    probability = np.zeros((2, 2, 2), dtype=np.float32)
    probability[:, :, 0] = 0.25
    probability[:, :, 1] = 0.75

    paths = write_flus_case(
        output_dir=tmp_path,
        start_map=start,
        probability_cube=probability,
        class_values=[1, 2],
        future_demand={1: 2, 2: 2},
        end_year=2021,
    )

    assert paths.landuse_path == tmp_path / "landuse.tif"
    assert paths.probability_path == tmp_path / "probability.tif"
    assert paths.restrict_path == tmp_path / "restrict.tif"
    assert paths.simulation_config_path == tmp_path / "CCregionsimlog.txt"
    assert paths.markov_chain_path == tmp_path / "CCregionMakovChain.csv"
    with rasterio.open(paths.landuse_path) as dataset:
        assert dataset.count == 1
        assert dataset.read(1).tolist() == [[1, 2], [1, 2]]
    with rasterio.open(paths.probability_path) as dataset:
        assert dataset.count == 2
        assert dataset.read(1).shape == (2, 2)
    with rasterio.open(paths.restrict_path) as dataset:
        assert dataset.count == 1
        assert dataset.read(1).tolist() == [[1, 1], [1, 1]]
    assert paths.markov_chain_path.read_text(encoding="utf-8") == "year,type1,type2\n2021,2,2\n"
    config = paths.simulation_config_path.read_text(encoding="utf-8")
    assert "[Path of land use data]\nlanduse.tif" in config
    assert "[Path of probability data]\nprobability.tif" in config
    assert "[Future Pixels]\n2\n2" in config


def test_write_flus_case_encodes_non_contiguous_classes(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    start = np.array([[1, 5], [1, 5]], dtype=np.int32)
    probability = np.zeros((2, 2, 2), dtype=np.float32)
    probability[:, :, 0] = 0.25
    probability[:, :, 1] = 0.75

    paths = write_flus_case(
        output_dir=tmp_path,
        start_map=start,
        probability_cube=probability,
        class_values=[1, 5],
        future_demand={1: 2, 5: 2},
        end_year=2021,
    )

    with rasterio.open(paths.landuse_path) as dataset:
        assert dataset.read(1).tolist() == [[1, 2], [1, 2]]
    assert paths.class_mapping_path.read_text(encoding="utf-8") == (
        '{\n'
        '  "encoded_to_original": {\n'
        '    "1": 1,\n'
        '    "2": 5\n'
        '  },\n'
        '  "original_to_encoded": {\n'
        '    "1": 1,\n'
        '    "5": 2\n'
        "  }\n"
        "}"
    )


def test_decode_flus_labels_restores_original_classes():
    encoded = np.array([[1, 2], [2, 1]], dtype=np.int32)

    decoded = decode_flus_labels(encoded, class_values=[1, 5])

    assert decoded.tolist() == [[1, 5], [5, 1]]


def test_decode_flus_geotiff_restores_original_classes(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    encoded_path = tmp_path / "encoded.tif"
    decoded_path = tmp_path / "decoded.tif"
    encoded = np.array([[1, 2], [2, 1]], dtype=np.int32)
    with rasterio.open(
        encoded_path,
        "w",
        driver="GTiff",
        height=encoded.shape[0],
        width=encoded.shape[1],
        count=1,
        dtype=encoded.dtype,
        transform=rasterio.transform.from_origin(0, encoded.shape[0], 1, 1),
    ) as dataset:
        dataset.write(encoded, 1)

    decode_flus_geotiff(encoded_path, decoded_path, class_values=[1, 5])

    with rasterio.open(decoded_path) as dataset:
        assert dataset.read(1).tolist() == [[1, 5], [5, 1]]


def test_find_flus_simulation_result_accepts_configured_name(tmp_path: Path):
    output = tmp_path / "simresult.tif"
    output.touch()

    assert find_flus_simulation_result(tmp_path, end_year=2021) == output


def test_find_flus_simulation_result_accepts_year_suffixed_name(tmp_path: Path):
    output = tmp_path / "simresult_2021.tif"
    output.touch()

    assert find_flus_simulation_result(tmp_path, end_year=2021) == output


def test_write_flus_case_rejects_wrong_probability_shape(tmp_path: Path):
    start = np.array([[1, 2], [1, 2]], dtype=np.int32)
    probability = np.zeros((2, 2, 3), dtype=np.float32)

    with pytest.raises(FLUSCaseError, match="probability_cube shape"):
        write_flus_case(
            output_dir=tmp_path,
            start_map=start,
            probability_cube=probability,
            class_values=[1, 2],
            future_demand={1: 2, 2: 2},
            end_year=2021,
        )


def test_write_flus_case_rejects_wrong_demand_total(tmp_path: Path):
    start = np.array([[1, 2], [1, 2]], dtype=np.int32)
    probability = np.zeros((2, 2, 2), dtype=np.float32)

    with pytest.raises(FLUSCaseError, match="future demand total 3 does not match raster cells 4"):
        write_flus_case(
            output_dir=tmp_path,
            start_map=start,
            probability_cube=probability,
            class_values=[1, 2],
            future_demand={1: 1, 2: 2},
            end_year=2021,
        )
