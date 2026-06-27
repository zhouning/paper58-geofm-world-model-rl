from pathlib import Path

import numpy as np


def test_discover_same_grid_samples_loads_true_paper58_predictions(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.run_true_paper58_flus_same_grid_comparison import discover_same_grid_samples

    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    labels.mkdir()
    predictions.mkdir()
    start = np.array([[1, 2], [1, 2]], dtype=np.int32)
    end = np.array([[1, 5], [1, 2]], dtype=np.int32)
    pred = np.array([[1, 5], [2, 2]], dtype=np.int32)
    np.save(labels / "strict_area_lulc_2020.npy", start)
    np.save(labels / "strict_area_lulc_2021.npy", end)
    np.save(predictions / "strict_area_lulc_pred_2020_2021.npy", pred)

    samples = discover_same_grid_samples(predictions, labels)

    assert len(samples) == 1
    assert samples[0].area == "strict_area"
    assert samples[0].start_year == 2020
    assert samples[0].end_year == 2021
    assert samples[0].paper58_prediction.tolist() == pred.tolist()
    assert samples[0].valid_mask.tolist() == [[True, True], [True, True]]


def test_write_same_grid_report_explains_strict_evidence_boundary(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.run_true_paper58_flus_same_grid_comparison import (
        SameGridSample,
        write_same_grid_report,
    )

    sample = SameGridSample(
        area="strict_area",
        start_year=2020,
        end_year=2021,
        start=np.array([[1, 2], [1, 2]], dtype=np.int32),
        end=np.array([[1, 5], [1, 2]], dtype=np.int32),
        paper58_prediction=np.array([[1, 5], [2, 2]], dtype=np.int32),
        valid_mask=np.ones((2, 2), dtype=bool),
    )
    flus_prediction = np.array([[1, 2], [1, 2]], dtype=np.int32)

    result = write_same_grid_report(
        output_dir=tmp_path,
        samples=[sample],
        flus_predictions={"strict_area": flus_prediction},
        notes=["单元测试说明"],
    )

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert result["n_samples"] == 1
    assert "完整 Paper58 latent-dynamics" in readme
    assert "同网格严格对比" in readme
    assert "GeoSOS-FLUS 的适宜性层" in readme
    assert "总体对比结论" in readme
    assert "单区域判读" in readme
    assert '<html lang="zh-CN">' in html
    assert "单元测试说明" in html


def test_write_same_grid_report_includes_extra_method_predictions(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.run_true_paper58_flus_same_grid_comparison import (
        SameGridSample,
        write_same_grid_report,
    )

    sample = SameGridSample(
        area="strict_area",
        start_year=2020,
        end_year=2021,
        start=np.array([[1, 2], [1, 2]], dtype=np.int32),
        end=np.array([[1, 5], [1, 2]], dtype=np.int32),
        paper58_prediction=np.array([[1, 5], [2, 2]], dtype=np.int32),
        valid_mask=np.ones((2, 2), dtype=bool),
    )
    extra_prediction = np.array([[1, 5], [1, 2]], dtype=np.int32)

    write_same_grid_report(
        output_dir=tmp_path,
        samples=[sample],
        flus_predictions={},
        extra_predictions={"paper58_change_gate": {"strict_area": extra_prediction}},
    )

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    metrics = (tmp_path / "metrics_by_method.csv").read_text(encoding="utf-8")
    assert "`paper58_change_gate`" in readme
    assert "paper58_change_gate" in metrics
