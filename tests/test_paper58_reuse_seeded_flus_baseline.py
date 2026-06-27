import csv
from pathlib import Path

import numpy as np


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_reuse_seeded_flus_baseline_writes_challenger_summary(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.reuse_seeded_flus_baseline import reuse_seeded_flus_baseline

    area = "case_a"
    labels = tmp_path / "labels"
    paper58_predictions = tmp_path / "paper58_predictions"
    challenger_predictions = tmp_path / "challenger_predictions"
    labels.mkdir()
    paper58_predictions.mkdir()
    challenger_predictions.mkdir()
    start = np.array([[1, 1, 1]], dtype=np.int32)
    end = np.array([[5, 1, 1]], dtype=np.int32)
    base_prediction = np.array([[1, 1, 1]], dtype=np.int32)
    challenger_prediction = np.array([[5, 1, 1]], dtype=np.int32)
    np.save(labels / f"{area}_lulc_2020.npy", start)
    np.save(labels / f"{area}_lulc_2021.npy", end)
    np.save(paper58_predictions / f"{area}_lulc_pred_2020_2021.npy", base_prediction)
    np.save(challenger_predictions / f"{area}_lulc_pred_2020_2021.npy", challenger_prediction)
    fields = [
        "seed",
        "method",
        "area",
        "start_year",
        "end_year",
        "source",
        "tier",
        "stratum",
        "n_pixels",
        "true_change_pixels",
        "pred_change_pixels",
        "change_precision",
        "change_recall",
        "change_f1",
        "fom",
        "transition_accuracy",
        "quantity_disagreement",
        "allocation_disagreement",
    ]
    _write_csv(
        tmp_path / "baseline" / "seeded_metrics_by_method.csv",
        [
            {
                "seed": 1,
                "method": "geosos_flus_console",
                "area": area,
                "start_year": 2020,
                "end_year": 2021,
                "source": "test",
                "tier": "same_grid",
                "stratum": "test",
                "n_pixels": 3,
                "true_change_pixels": 1,
                "pred_change_pixels": 0,
                "change_precision": 0.0,
                "change_recall": 0.0,
                "change_f1": 0.0,
                "fom": 0.0,
                "transition_accuracy": 0.0,
                "quantity_disagreement": 0.3333333333,
                "allocation_disagreement": 0.3333333333,
            },
            {
                "seed": 2,
                "method": "geosos_flus_console",
                "area": area,
                "start_year": 2020,
                "end_year": 2021,
                "source": "test",
                "tier": "same_grid",
                "stratum": "test",
                "n_pixels": 3,
                "true_change_pixels": 1,
                "pred_change_pixels": 0,
                "change_precision": 0.0,
                "change_recall": 0.0,
                "change_f1": 0.0,
                "fom": 0.0,
                "transition_accuracy": 0.0,
                "quantity_disagreement": 0.3333333333,
                "allocation_disagreement": 0.3333333333,
            },
        ],
        fields,
    )

    result = reuse_seeded_flus_baseline(
        baseline_seeded_dir=tmp_path / "baseline",
        paper58_predictions_dir=paper58_predictions,
        labels_dir=labels,
        challenger_prediction_dir=challenger_predictions,
        output_dir=tmp_path / "out",
        challenger="paper58_v4",
    )

    assert result["n_rows"] == 4
    readme = (tmp_path / "out" / "README.md").read_text(encoding="utf-8")
    assert "`paper58_v4` 在 4 个指标上" in readme
    assert (tmp_path / "out" / "seeded_delta_summary.csv").exists()
