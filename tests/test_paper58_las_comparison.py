import csv
from pathlib import Path

import pytest

from scripts.paper58_benchmark.summarize_las_comparison import summarize_las_comparison


def _write_metrics(path: Path) -> None:
    rows = [
        {
            "method": "paper58_las",
            "area": "a",
            "start_year": "2020",
            "end_year": "2021",
            "tier": "tier1",
            "stratum": "Wetland",
            "change_f1": "0.60",
            "fom": "0.30",
            "change_recall": "0.70",
            "quantity_disagreement": "0.00",
            "allocation_disagreement": "0.20",
        },
        {
            "method": "flus",
            "area": "a",
            "start_year": "2020",
            "end_year": "2021",
            "tier": "tier1",
            "stratum": "Wetland",
            "change_f1": "0.40",
            "fom": "0.20",
            "change_recall": "0.30",
            "quantity_disagreement": "0.10",
            "allocation_disagreement": "0.10",
        },
        {
            "method": "paper58_las",
            "area": "b",
            "start_year": "2020",
            "end_year": "2021",
            "tier": "tier1",
            "stratum": "Urban",
            "change_f1": "0.20",
            "fom": "0.10",
            "change_recall": "0.50",
            "quantity_disagreement": "0.00",
            "allocation_disagreement": "0.30",
        },
        {
            "method": "flus",
            "area": "b",
            "start_year": "2020",
            "end_year": "2021",
            "tier": "tier1",
            "stratum": "Urban",
            "change_f1": "0.30",
            "fom": "0.12",
            "change_recall": "0.20",
            "quantity_disagreement": "0.05",
            "allocation_disagreement": "0.20",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_las_comparison_writes_paired_advantages(tmp_path: Path):
    metrics = tmp_path / "las_metrics_by_method.csv"
    output = tmp_path / "comparison"
    _write_metrics(metrics)

    summary = summarize_las_comparison(metrics_path=metrics, output_dir=output, n_boot=100)

    assert summary["n_pairs"] == 2
    assert summary["method_means"]["paper58_las"]["change_f1"] == pytest.approx(0.40)
    assert summary["method_means"]["flus"]["change_f1"] == pytest.approx(0.35)
    assert summary["advantages"]["change_f1"]["mean_advantage"] == pytest.approx(0.05)
    assert summary["advantages"]["change_f1"]["sign_test"]["n_positive"] == 1
    assert summary["advantages"]["change_f1"]["sign_test"]["n_negative"] == 1
    assert (output / "las_comparison_summary.json").exists()
    assert (output / "las_comparison_by_area.csv").exists()


def test_summarize_las_comparison_requires_matched_methods(tmp_path: Path):
    metrics = tmp_path / "las_metrics_by_method.csv"
    _write_metrics(metrics)

    with pytest.raises(ValueError, match="no matched method pairs"):
        summarize_las_comparison(metrics_path=metrics, output_dir=tmp_path / "comparison", baseline_method="missing")
