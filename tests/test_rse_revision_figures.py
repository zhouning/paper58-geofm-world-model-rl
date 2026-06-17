from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from scripts.rse_revision.make_revision_figures import (
    load_revision_inputs,
    load_spatial_change_case,
)


def test_load_revision_inputs_reads_summary_and_area_rows(tmp_path: Path):
    summary = tmp_path / "revision_summary.json"
    alpha = tmp_path / "alphaearth_area_metrics.csv"
    prithvi = tmp_path / "prithvi_area_metrics.csv"
    category = tmp_path / "alphaearth_category_summary.csv"
    planning = tmp_path / "planning_baseline_summary.csv"
    transfer = tmp_path / "transfer_planning_summary.csv"
    decoder = tmp_path / "world_model_lulc_decode.json"

    summary.write_text('{"alphaearth": {"n_areas": 1}, "prithvi": {"n_areas": 1}}', encoding="utf-8")
    alpha.write_text(
        "area,persistence,model,advantage,change_pixel_advantage\n"
        "a,0.95,0.96,0.01,\n",
        encoding="utf-8",
    )
    prithvi.write_text(
        "area,persistence,model,advantage,change_pixel_advantage\n"
        "p,1.0,0.99999,-0.00001,\n",
        encoding="utf-8",
    )
    category.write_text(
        "category,n,mean_advantage,median_advantage,n_positive,n_negative,areas\n"
        "Urban,1,0.01,0.01,1,0,a\n",
        encoding="utf-8",
    )
    planning.write_text(
        "configuration,n,slope_mean,slope_std,cont_mean,cont_std,reward_mean,reward_std,slope_retention_vs_full\n"
        "full,1,-1.0,0.0,0.01,0.0,1.0,0.0,1.0\n",
        encoding="utf-8",
    )
    transfer.write_text(
        "configuration,n,reward_mean,reward_std,crop_change_mean,crop_change_std,reward_delta_vs_random,reward_delta_vs_greedy\n"
        "transfer,1,-7.0,0.0,-3.5,0.0,5.0,13.0\n",
        encoding="utf-8",
    )
    decoder.write_text(
        '{"n_samples": 2, "classes": [10, 30], "overall_accuracy": 0.5, '
        '"confusion_matrix": [[1, 0], [1, 0]], '
        '"classification_report": {"macro avg": {"f1-score": 0.3333333333333333}}}',
        encoding="utf-8",
    )

    data = load_revision_inputs(tmp_path, decoder_path=decoder)

    assert data["summary"]["alphaearth"]["n_areas"] == 1
    assert data["alphaearth_rows"][0]["area"] == "a"
    assert data["alphaearth_rows"][0]["advantage"] == 0.01
    assert data["prithvi_rows"][0]["area"] == "p"
    assert data["decoder"]["overall_accuracy"] == 0.5
    assert data["category_rows"][0]["category"] == "Urban"
    assert data["planning_baseline_rows"][0]["configuration"] == "full"
    assert data["transfer_planning_rows"][0]["configuration"] == "transfer"
    assert data["transfer_planning_rows"][0]["reward_delta_vs_random"] == 5.0


def test_load_spatial_change_case_builds_error_masks(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    predicted_dir = tmp_path / "predicted"
    labels_dir.mkdir()
    predicted_dir.mkdir()

    start = np.array([[1, 1, 2], [2, 3, 3]])
    end = np.array([[1, 2, 2], [3, 3, 1]])
    pred = np.array([[1, 2, 2], [2, 1, 1]])
    np.save(labels_dir / "toy_lulc_2020.npy", start)
    np.save(labels_dir / "toy_lulc_2021.npy", end)
    np.save(predicted_dir / "toy_lulc_pred_2020_2021.npy", pred)

    case = load_spatial_change_case(labels_dir, predicted_dir, "toy", 2020, 2021)

    assert case["start"].shape == (2, 3)
    assert int(case["true_change"].sum()) == 3
    assert int(case["model_change"].sum()) == 3
    assert int(case["change_hit"].sum()) == 2
    assert int(case["change_miss"].sum()) == 1
    assert int(case["false_alarm"].sum()) == 1
