from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from scripts.paper58_benchmark.make_benchmark_figures import load_benchmark_outputs, make_benchmark_figures


def test_load_benchmark_outputs_fails_when_required_files_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Missing benchmark result files"):
        load_benchmark_outputs(tmp_path)


def test_make_benchmark_figures_writes_pdf_and_png(tmp_path: Path):
    results_dir = tmp_path / "results"
    figure_dir = tmp_path / "figures"
    results_dir.mkdir()
    (results_dir / "benchmark_metrics_by_pair.csv").write_text(
        "area,start_year,end_year,tier,stratum,primary_change_advantage,spatial_change_advantage,model_change_f1,best_non_neural_change_f1\n"
        "external,2020,2021,tier1,Wetland,0.20,0.10,0.50,0.30\n",
        encoding="utf-8",
    )
    (results_dir / "benchmark_gate_report.json").write_text(
        '{"status": "pass", "positive_tier1_strata": 1}',
        encoding="utf-8",
    )

    make_benchmark_figures(results_dir=results_dir, figure_dir=figure_dir)

    assert (figure_dir / "fig_paper58_benchmark_gate.pdf").exists()
    assert (figure_dir / "fig_paper58_benchmark_gate.png").exists()
