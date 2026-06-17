import json
from pathlib import Path

import pytest

from scripts.rse_revision.audit_empirical_pipeline import audit_empirical_pipeline
from scripts.rse_revision.build_revision_results import (
    summarize_planning_runs,
    summarize_rows_by_category,
    summarize_transfer_planning_runs,
)


def test_summarize_rows_by_category_groups_valid_area_advantages():
    rows = [
        {"area": "yangtze_delta", "advantage": 0.01},
        {"area": "jing_jin_ji", "advantage": -0.02},
        {"area": "poyang_lake", "advantage": 0.03},
        {"area": "unknown_area", "advantage": 0.99},
    ]

    result = summarize_rows_by_category(rows)

    assert result["Urban"]["n"] == 2
    assert result["Urban"]["mean_advantage"] == pytest.approx(-0.005)
    assert result["Urban"]["n_positive"] == 1
    assert result["Wetland"]["mean_advantage"] == pytest.approx(0.03)
    assert "unknown_area" not in result


def test_summarize_planning_runs_reads_json_files_and_computes_retention(tmp_path: Path):
    intervention_dir = tmp_path / "intervention"
    dual_dir = tmp_path / "dual_rep"
    intervention_dir.mkdir()
    dual_dir.mkdir()

    for seed, slope in enumerate([-0.10, -0.20]):
        (intervention_dir / f"ppo_eval_seed{seed}.json").write_text(
            json.dumps({"seed": seed, "mean_slope": slope, "mean_cont": 0.01, "mean_reward": -10.0}),
            encoding="utf-8",
        )
    for label, slopes in {
        "full": [-1.0, -1.2],
        "dropout0.3": [-0.8, -1.0],
        "dropout1.0": [-0.2, -0.3],
    }.items():
        for seed, slope in enumerate(slopes):
            (dual_dir / f"{label}_eval_seed{seed}.json").write_text(
                json.dumps({"seed": seed, "mean_slope": slope, "mean_cont": 0.02, "mean_reward": 5.0}),
                encoding="utf-8",
            )

    result = summarize_planning_runs(
        {
            "embedding_intervention": (intervention_dir, "ppo_eval_seed*.json"),
            "full": (dual_dir, "full_eval_seed*.json"),
            "dropout0.3": (dual_dir, "dropout0.3_eval_seed*.json"),
            "dropout1.0": (dual_dir, "dropout1.0_eval_seed*.json"),
        }
    )

    assert result["embedding_intervention"]["n"] == 2
    assert result["embedding_intervention"]["slope_mean"] == pytest.approx(-0.15)
    assert result["dropout0.3"]["slope_retention_vs_full"] == pytest.approx(0.8181818)
    assert result["dropout1.0"]["slope_retention_vs_full"] == pytest.approx(0.2272727)


def test_summarize_transfer_planning_runs_reads_reward_style_jsons(tmp_path: Path):
    transfer_dir = tmp_path / "transfer_heping"
    transfer_dir.mkdir()

    payloads = {
        "random": [(-10.0, -5.0), (-14.0, -7.0)],
        "greedy": [(-20.0, -6.0), (-20.0, -6.0)],
        "transfer": [(-8.0, -4.0), (-6.0, -3.0)],
    }
    for label, rows in payloads.items():
        for seed, (reward, crop_change) in enumerate(rows):
            (transfer_dir / f"{label}_eval_seed{seed}.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "label": label,
                        "mean_reward": reward,
                        "mean_cropland_change": crop_change,
                    }
                ),
                encoding="utf-8",
            )

    result = summarize_transfer_planning_runs(transfer_dir)

    assert result["random"]["n"] == 2
    assert result["random"]["reward_mean"] == pytest.approx(-12.0)
    assert result["transfer"]["reward_mean"] == pytest.approx(-7.0)
    assert result["transfer"]["reward_delta_vs_random"] == pytest.approx(5.0)
    assert result["transfer"]["reward_delta_vs_greedy"] == pytest.approx(13.0)
    assert result["transfer"]["crop_change_mean"] == pytest.approx(-3.5)


def test_audit_empirical_pipeline_marks_completed_diagnostics_and_missing_change_labels(tmp_path: Path):
    results_dir = tmp_path / "revision_results"
    decoder_path = tmp_path / "world_model_lulc_decode.json"
    independent_labels_dir = tmp_path / "independent_change_labels"
    output_dir = tmp_path / "audit_out"
    results_dir.mkdir()

    (results_dir / "revision_summary.json").write_text(
        json.dumps(
            {
                "alphaearth": {"n_areas": 10, "advantage": {"mean": 0.0047}},
                "prithvi": {"n_areas": 17, "advantage": {"mean": -0.000002}},
                "planning_baselines": {"full": {"n": 15}, "dropout0.3": {"n": 15}},
                "transfer_planning": {"transfer": {"n": 15}},
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "alphaearth_area_metrics.csv",
        "prithvi_area_metrics.csv",
        "alphaearth_category_summary.csv",
        "planning_baseline_summary.csv",
        "transfer_planning_summary.csv",
    ):
        (results_dir / name).write_text("header\n", encoding="utf-8")
    decoder_path.write_text(
        json.dumps({"n_samples": 5119, "overall_accuracy": 0.7699, "classification_report": {}}),
        encoding="utf-8",
    )

    report = audit_empirical_pipeline(
        results_dir=results_dir,
        decoder_path=decoder_path,
        independent_change_labels_dir=independent_labels_dir,
        output_dir=output_dir,
    )

    stages = {stage["stage_id"]: stage for stage in report["stages"]}
    assert stages["area_level_embedding_dynamics"]["status"] == "complete"
    assert stages["encoder_replacement_ablation"]["status"] == "complete"
    assert stages["semantic_decoder_validation"]["status"] == "diagnostic"
    assert stages["planning_feature_dropout"]["status"] == "diagnostic"
    assert stages["independent_change_label_validation"]["status"] == "missing"
    assert stages["independent_change_label_validation"]["claim_supported"] is False
    assert report["summary"]["n_complete"] == 2
    assert report["summary"]["n_diagnostic"] == 3
    assert report["summary"]["n_missing"] == 1
    assert (output_dir / "empirical_pipeline_audit.json").exists()
    assert (output_dir / "empirical_pipeline_audit.csv").exists()


def test_audit_empirical_pipeline_uses_completed_independent_change_summary(tmp_path: Path):
    results_dir = tmp_path / "revision_results"
    decoder_path = tmp_path / "world_model_lulc_decode.json"
    independent_labels_dir = tmp_path / "independent_change_labels"
    output_dir = tmp_path / "audit_out"
    results_dir.mkdir()
    independent_labels_dir.mkdir()

    (results_dir / "revision_summary.json").write_text(
        json.dumps(
            {
                "alphaearth": {"n_areas": 10, "advantage": {"mean": 0.0047}},
                "prithvi": {"n_areas": 17, "advantage": {"mean": -0.000002}},
                "planning_baselines": {"full": {"n": 15}, "dropout0.3": {"n": 15}},
                "transfer_planning": {"transfer": {"n": 15}},
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "alphaearth_area_metrics.csv",
        "prithvi_area_metrics.csv",
        "alphaearth_category_summary.csv",
        "planning_baseline_summary.csv",
        "transfer_planning_summary.csv",
    ):
        (results_dir / name).write_text("header\n", encoding="utf-8")
    decoder_path.write_text(json.dumps({"n_samples": 10, "overall_accuracy": 0.8}), encoding="utf-8")
    (results_dir / "independent_change_validation_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "n_area_year_pairs": 2,
                "mean_metrics": {
                    "model_change_f1": 0.42,
                    "shuffled_model_change_f1": 0.28,
                    "transition_prior_change_f1": 0.16,
                    "persistence_change_f1": 0.0,
                    "model_changed_pixel_accuracy": 0.31,
                },
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "independent_change_validation_by_area.csv").write_text("area,start_year,end_year\n", encoding="utf-8")

    report = audit_empirical_pipeline(
        results_dir=results_dir,
        decoder_path=decoder_path,
        independent_change_labels_dir=independent_labels_dir,
        output_dir=output_dir,
    )

    stages = {stage["stage_id"]: stage for stage in report["stages"]}
    independent_stage = stages["independent_change_label_validation"]
    assert independent_stage["status"] == "complete"
    assert independent_stage["claim_supported"] is True
    assert "n_pairs=2" in independent_stage["metric_summary"]
    assert "model change F1=0.42" in independent_stage["metric_summary"]
    assert "shuffled change F1=0.28" in independent_stage["metric_summary"]
    assert "transition-prior change F1=0.16" in independent_stage["metric_summary"]
    assert report["summary"]["independent_change_label_validation_complete"] is True
