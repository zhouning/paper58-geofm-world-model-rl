from pathlib import Path

import numpy as np


def test_start_class_fraction_selector_uses_only_start_map(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.apply_paper58_start_class_fraction_selector import (
        run_start_class_fraction_selector,
    )

    labels = tmp_path / "labels"
    fallback = tmp_path / "fallback"
    semantic = tmp_path / "semantic"
    labels.mkdir()
    fallback.mkdir()
    semantic.mkdir()

    high_area = "high_class5"
    low_area = "low_class5"
    np.save(labels / f"{high_area}_lulc_2020.npy", np.array([[5, 5], [5, 1]], dtype=np.int32))
    np.save(labels / f"{low_area}_lulc_2020.npy", np.array([[5, 1], [1, 1]], dtype=np.int32))

    np.save(fallback / f"{high_area}_lulc_pred_2020_2021.npy", np.full((2, 2), 10, dtype=np.int32))
    np.save(semantic / f"{high_area}_lulc_pred_2020_2021.npy", np.full((2, 2), 20, dtype=np.int32))
    np.save(fallback / f"{low_area}_lulc_pred_2020_2021.npy", np.full((2, 2), 30, dtype=np.int32))
    np.save(semantic / f"{low_area}_lulc_pred_2020_2021.npy", np.full((2, 2), 40, dtype=np.int32))

    manifest = run_start_class_fraction_selector(
        labels_dir=labels,
        fallback_prediction_dir=fallback,
        semantic_prediction_dir=semantic,
        output_dir=tmp_path / "out",
        selector_class=5,
        max_semantic_fraction=0.25,
        start_year=2020,
        end_year=2021,
    )

    assert np.load(tmp_path / "out" / "predictions" / f"{high_area}_lulc_pred_2020_2021.npy").tolist() == [
        [10, 10],
        [10, 10],
    ]
    assert np.load(tmp_path / "out" / "predictions" / f"{low_area}_lulc_pred_2020_2021.npy").tolist() == [
        [40, 40],
        [40, 40],
    ]
    choices = {row["area"]: row for row in manifest["cases"]}
    assert choices[high_area]["selected_branch"] == "fallback"
    assert choices[low_area]["selected_branch"] == "semantic"
    assert "end_label" not in choices[high_area]
