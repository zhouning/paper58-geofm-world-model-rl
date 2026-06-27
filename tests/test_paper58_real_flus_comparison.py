from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")


def _write_multiband_tif(path: Path, data: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        transform=rasterio.transform.from_origin(0, data.shape[1], 1, 1),
    ) as dataset:
        dataset.write(data)


def test_read_dynamicworld_sample_extracts_labels_probabilities_and_valid_mask(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.run_real_flus_comparison import read_dynamicworld_sample

    tif = tmp_path / "sample.tif"
    data = np.zeros((14, 2, 2), dtype=np.uint16)
    data[0] = np.array([[1, 2], [0, 5]], dtype=np.uint16)
    data[1] = np.array([[1, 3], [0, 5]], dtype=np.uint16)
    data[2] = np.array([[9000, 100], [0, 100]], dtype=np.uint16)
    data[3] = np.array([[100, 9000], [0, 100]], dtype=np.uint16)
    data[8] = np.array([[8500, 100], [0, 100]], dtype=np.uint16)
    data[9] = np.array([[100, 8500], [0, 100]], dtype=np.uint16)
    _write_multiband_tif(tif, data)

    sample = read_dynamicworld_sample("toy_area", tif, 2020, 2021)

    assert sample.area == "toy_area"
    assert sample.start_year == 2020
    assert sample.end_year == 2021
    assert sample.start.tolist() == [[1, 2], [0, 5]]
    assert sample.end.tolist() == [[1, 3], [0, 5]]
    assert sample.valid_mask.tolist() == [[True, True], [False, True]]
    assert sample.probability_cube.shape == (2, 2, 6)
    assert sample.probability_cube[0, 0, 0] == pytest.approx(0.85)


def test_transition_prior_proxy_preserves_outside_mask_and_uses_training_distribution() -> None:
    from scripts.paper58_benchmark.run_real_flus_comparison import RealSample, make_transition_prior_proxy

    sample = RealSample(
        area="target",
        start_year=2020,
        end_year=2021,
        source="unit",
        start=np.array([[1, 1, 0], [2, 2, 2]], dtype=np.int32),
        end=np.array([[1, 2, 0], [2, 3, 3]], dtype=np.int32),
        probability_cube=None,
        valid_mask=np.array([[True, True, False], [True, True, True]]),
    )
    train_start = np.array([[1, 1, 2], [2, 2, 2]], dtype=np.int32)
    train_end = np.array([[2, 2, 3], [3, 3, 3]], dtype=np.int32)

    pred = make_transition_prior_proxy(sample, [(train_start, train_end)], seed=3)

    assert pred.shape == sample.start.shape
    assert pred[0, 2] == 0
    assert set(np.unique(pred[sample.valid_mask])).issubset({2, 3})


def test_evaluate_predictions_scores_only_valid_pixels() -> None:
    from scripts.paper58_benchmark.run_real_flus_comparison import RealSample, evaluate_predictions

    sample = RealSample(
        area="target",
        start_year=2020,
        end_year=2021,
        source="unit",
        start=np.array([[1, 0], [2, 2]], dtype=np.int32),
        end=np.array([[2, 0], [2, 3]], dtype=np.int32),
        probability_cube=None,
        valid_mask=np.array([[True, False], [True, True]]),
    )
    paper58 = np.array([[2, 9], [2, 2]], dtype=np.int32)
    flus = np.array([[1, 9], [2, 3]], dtype=np.int32)

    rows = evaluate_predictions(sample, paper58_prediction=paper58, flus_prediction=flus)

    by_method = {row["method"]: row for row in rows}
    assert by_method["paper58_proxy"]["n_pixels"] == 3
    assert by_method["paper58_proxy"]["change_f1"] == pytest.approx(2 / 3)
    assert by_method["geosos_flus_console"]["change_f1"] == pytest.approx(2 / 3)


def test_write_visual_report_emits_metrics_and_spatial_panel(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.run_real_flus_comparison import RealSample, write_visual_report

    sample = RealSample(
        area="target",
        start_year=2020,
        end_year=2021,
        source="unit",
        start=np.array([[1, 1], [2, 2]], dtype=np.int32),
        end=np.array([[1, 2], [2, 3]], dtype=np.int32),
        probability_cube=None,
        valid_mask=np.ones((2, 2), dtype=bool),
    )
    paper58 = np.array([[1, 2], [2, 2]], dtype=np.int32)
    rows = write_visual_report(
        output_dir=tmp_path,
        samples=[sample],
        paper58_predictions={"target": paper58},
        flus_predictions={},
        metric_rows=[],
        notes=["unit-test note"],
    )

    assert rows["n_spatial_panels"] == 1
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "metrics_by_method.csv").exists()
    assert (tmp_path / "figures" / "spatial_target.png").exists()

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "如何阅读误差图" in readme
    assert "总体对比结论" in readme
    assert "Paper58 与 GeoSOS-FLUS 的算法原理差异" in readme
    assert "潜空间世界模型" in readme
    assert "元胞自动机" in readme
    assert "单区域判读" in readme

    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert '<html lang="zh-CN">' in html
    assert "如何阅读误差图" in html
