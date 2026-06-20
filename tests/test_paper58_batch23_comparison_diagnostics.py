import csv
import json
from pathlib import Path

from scripts.paper58_benchmark.make_batch23_comparison_diagnostics import (
    build_comparison_diagnostics,
)


def _write_results_dir(path: Path, metrics_rows: list[dict], gate: dict) -> None:
    path.mkdir(parents=True)
    fields = [
        "area",
        "start_year",
        "end_year",
        "tier",
        "stratum",
        "true_change_pixels",
        "model_change_f1",
        "primary_change_advantage",
        "spatial_shuffle_change_f1",
        "spatial_change_advantage",
    ]
    with (path / "benchmark_metrics_by_pair.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics_rows)
    (path / "benchmark_gate_report.json").write_text(json.dumps(gate), encoding="utf-8")


def test_build_comparison_diagnostics_writes_ranked_and_summary_outputs(tmp_path: Path):
    batch2 = tmp_path / "batch2"
    batch3 = tmp_path / "batch3"
    output = tmp_path / "out"
    _write_results_dir(
        batch2,
        [
            {
                "area": "xiong_an",
                "start_year": 2020,
                "end_year": 2021,
                "tier": "tier1",
                "stratum": "Urban",
                "true_change_pixels": 100,
                "model_change_f1": 0.20,
                "primary_change_advantage": 0.10,
                "spatial_shuffle_change_f1": 0.30,
                "spatial_change_advantage": -0.10,
            },
            {
                "area": "beibu",
                "start_year": 2020,
                "end_year": 2021,
                "tier": "tier1",
                "stratum": "Urban",
                "true_change_pixels": 50,
                "model_change_f1": 0.35,
                "primary_change_advantage": 0.12,
                "spatial_shuffle_change_f1": 0.20,
                "spatial_change_advantage": 0.15,
            },
        ],
        {
            "status": "fail",
            "tier1_spatial_change": {"ci_low": -0.02},
            "tier1_primary_change": {"ci_low": 0.05},
            "positive_tier1_strata": 2,
        },
    )
    _write_results_dir(
        batch3,
        [
            {
                "area": "suzhou",
                "start_year": 2020,
                "end_year": 2021,
                "tier": "tier1",
                "stratum": "Urban",
                "true_change_pixels": 40,
                "model_change_f1": 0.16,
                "primary_change_advantage": 0.02,
                "spatial_shuffle_change_f1": 0.17,
                "spatial_change_advantage": -0.01,
            },
            {
                "area": "fuzhou",
                "start_year": 2020,
                "end_year": 2021,
                "tier": "tier1",
                "stratum": "Urban",
                "true_change_pixels": 10,
                "model_change_f1": 0.42,
                "primary_change_advantage": 0.42,
                "spatial_shuffle_change_f1": 0.08,
                "spatial_change_advantage": 0.34,
            },
        ],
        {
            "status": "pass",
            "tier1_spatial_change": {"ci_low": 0.06},
            "tier1_primary_change": {"ci_low": 0.10},
            "positive_tier1_strata": 5,
        },
    )

    summary = build_comparison_diagnostics(
        batch2_results_dir=batch2,
        batch3_results_dir=batch3,
        output_dir=output,
        n_boot=100,
    )

    assert summary["gate_comparison"] == {
        "batch2_status": "fail",
        "batch2_primary_ci_low": 0.05,
        "batch2_spatial_ci_low": -0.02,
        "batch2_positive_tier1_strata": 2,
        "batch3_status": "pass",
        "batch3_primary_ci_low": 0.10,
        "batch3_spatial_ci_low": 0.06,
        "batch3_positive_tier1_strata": 5,
    }
    assert summary["urban_comparison"]["batch2"]["n"] == 2
    assert summary["urban_comparison"]["batch2"]["n_negative_spatial"] == 1
    assert summary["urban_comparison"]["batch3"]["mean_spatial_change_advantage"] == 0.165
    ranked_lines = (output / "batch23_spatial_advantage_ranked.csv").read_text(encoding="utf-8").splitlines()
    assert ranked_lines[0].startswith("batch,area,stratum")
    assert ranked_lines[1].startswith("batch2,xiong_an,Urban")
    assert ranked_lines[-1].startswith("batch3,fuzhou,Urban")
    assert (output / "batch23_comparison_summary.json").exists()
