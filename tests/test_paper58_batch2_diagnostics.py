from pathlib import Path

import numpy as np

from scripts.paper58_benchmark.make_batch2_diagnostics import (
    best_shift_diagnostic,
    make_batch2_alignment_table,
)


def test_best_shift_diagnostic_finds_known_change_offset():
    true_change = np.zeros((5, 5), dtype=bool)
    model_change = np.zeros((5, 5), dtype=bool)
    true_change[3, 3] = True
    model_change[1, 1] = True

    diagnostic = best_shift_diagnostic(true_change, model_change, max_shift=3)

    assert diagnostic["raw_change_f1"] == 0.0
    assert diagnostic["best_shift_change_f1"] == 1.0
    assert diagnostic["best_dy"] == 2
    assert diagnostic["best_dx"] == 2


def test_make_batch2_alignment_table_writes_shift_diagnostics(tmp_path: Path):
    labels = tmp_path / "labels"
    predictions = tmp_path / "predicted"
    output = tmp_path / "diagnostics"
    labels.mkdir()
    predictions.mkdir()

    start = np.zeros((5, 5), dtype=np.int32)
    end = start.copy()
    pred = start.copy()
    end[3, 3] = 1
    pred[1, 1] = 1
    np.save(labels / "toy_lulc_2020.npy", start)
    np.save(labels / "toy_lulc_2021.npy", end)
    np.save(predictions / "toy_lulc_pred_2020_2021.npy", pred)

    rows = make_batch2_alignment_table(
        out_dir=output,
        labels_dir=labels,
        predictions_dir=predictions,
        areas=["toy"],
        start_year=2020,
        end_year=2021,
        max_shift=3,
    )

    assert rows == [
        {
            "area": "toy",
            "raw_change_f1": 0.0,
            "best_shift_change_f1": 1.0,
            "best_dy": 2,
            "best_dx": 2,
            "centroid_true_y": 3.0,
            "centroid_true_x": 3.0,
            "centroid_model_y": 1.0,
            "centroid_model_x": 1.0,
        }
    ]
    assert (output / "batch2_spatial_alignment_shift.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "area,raw_change_f1,best_shift_change_f1,best_dy,best_dx,"
        "centroid_true_y,centroid_true_x,centroid_model_y,centroid_model_x"
    )
