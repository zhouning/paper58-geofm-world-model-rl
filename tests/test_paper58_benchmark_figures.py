from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from scripts.paper58_benchmark.make_benchmark_figures import load_benchmark_outputs, make_benchmark_figures


def _write_gate_report(results_dir: Path) -> None:
    (results_dir / "benchmark_gate_report.json").write_text(
        '{"status": "pass", "positive_tier1_strata": 1}',
        encoding="utf-8",
    )


def test_load_benchmark_outputs_fails_when_required_files_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Missing benchmark result files"):
        load_benchmark_outputs(tmp_path)


def test_load_benchmark_outputs_reports_missing_required_columns(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "benchmark_metrics_by_pair.csv").write_text(
        "start_year,end_year,primary_change_advantage,spatial_change_advantage,model_change_f1,best_non_neural_change_f1\n"
        "2020,2021,0.20,0.10,0.50,0.30\n",
        encoding="utf-8",
    )
    _write_gate_report(results_dir)

    with pytest.raises(
        ValueError,
        match=r"benchmark_metrics_by_pair\.csv missing required columns: area, tier",
    ):
        load_benchmark_outputs(results_dir)


def test_load_benchmark_outputs_reports_invalid_numeric_value_with_row_and_field(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "benchmark_metrics_by_pair.csv").write_text(
        "area,start_year,end_year,tier,stratum,primary_change_advantage,spatial_change_advantage,model_change_f1,best_non_neural_change_f1\n"
        "external,2020,2021,tier1,Wetland,NA,0.10,0.50,0.30\n",
        encoding="utf-8",
    )
    _write_gate_report(results_dir)

    with pytest.raises(
        ValueError,
        match=r"benchmark_metrics_by_pair\.csv row 2 has invalid primary_change_advantage: 'NA'",
    ):
        load_benchmark_outputs(results_dir)


def test_load_benchmark_outputs_reports_missing_numeric_value_with_row_and_field(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "benchmark_metrics_by_pair.csv").write_text(
        "area,start_year,end_year,tier,stratum,primary_change_advantage,spatial_change_advantage,model_change_f1,best_non_neural_change_f1\n"
        "external,2020,2021,tier1,Wetland,0.20,0.10,0.50\n",
        encoding="utf-8",
    )
    _write_gate_report(results_dir)

    with pytest.raises(ValueError) as excinfo:
        load_benchmark_outputs(results_dir)

    message = str(excinfo.value)
    assert "benchmark_metrics_by_pair.csv" in message
    assert "row 2" in message
    assert "best_non_neural_change_f1" in message
    assert "None" in message


def test_load_benchmark_outputs_reports_extra_csv_cells(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "benchmark_metrics_by_pair.csv").write_text(
        "area,start_year,end_year,tier,stratum,primary_change_advantage,spatial_change_advantage,model_change_f1,best_non_neural_change_f1\n"
        "external,2020,2021,tier1,Wetland,0.20,0.10,0.50,0.30,unexpected\n",
        encoding="utf-8",
    )
    _write_gate_report(results_dir)

    with pytest.raises(
        ValueError,
        match=r"benchmark_metrics_by_pair\.csv row 2 has extra columns: \['unexpected'\]",
    ):
        load_benchmark_outputs(results_dir)


@pytest.mark.parametrize(
    ("row", "field", "value"),
    [
        ("external,2020,2021,tier1,Wetland,NaN,0.10,0.50,0.30\n", "primary_change_advantage", "NaN"),
        ("external,2020,2021,tier1,Wetland,0.20,inf,0.50,0.30\n", "spatial_change_advantage", "inf"),
    ],
)
def test_load_benchmark_outputs_reports_non_finite_numeric_value(
    tmp_path: Path,
    row: str,
    field: str,
    value: str,
):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "benchmark_metrics_by_pair.csv").write_text(
        "area,start_year,end_year,tier,stratum,primary_change_advantage,spatial_change_advantage,model_change_f1,best_non_neural_change_f1\n"
        + row,
        encoding="utf-8",
    )
    _write_gate_report(results_dir)

    with pytest.raises(
        ValueError,
        match=rf"benchmark_metrics_by_pair\.csv row 2 has invalid {field}: '{value}'",
    ):
        load_benchmark_outputs(results_dir)


def test_load_benchmark_outputs_reports_invalid_gate_json(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "benchmark_metrics_by_pair.csv").write_text(
        "area,start_year,end_year,tier,stratum,primary_change_advantage,spatial_change_advantage,model_change_f1,best_non_neural_change_f1\n"
        "external,2020,2021,tier1,Wetland,0.20,0.10,0.50,0.30\n",
        encoding="utf-8",
    )
    (results_dir / "benchmark_gate_report.json").write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"benchmark_gate_report\.json is not valid JSON"):
        load_benchmark_outputs(results_dir)


def test_load_benchmark_outputs_requires_gate_json_object(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "benchmark_metrics_by_pair.csv").write_text(
        "area,start_year,end_year,tier,stratum,primary_change_advantage,spatial_change_advantage,model_change_f1,best_non_neural_change_f1\n"
        "external,2020,2021,tier1,Wetland,0.20,0.10,0.50,0.30\n",
        encoding="utf-8",
    )
    (results_dir / "benchmark_gate_report.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match=r"benchmark_gate_report\.json must contain a JSON object"):
        load_benchmark_outputs(results_dir)


def test_make_benchmark_figures_writes_pdf_and_png(tmp_path: Path):
    results_dir = tmp_path / "results"
    figure_dir = tmp_path / "figures"
    results_dir.mkdir()
    (results_dir / "benchmark_metrics_by_pair.csv").write_text(
        "area,start_year,end_year,tier,stratum,primary_change_advantage,spatial_change_advantage,model_change_f1,best_non_neural_change_f1\n"
        "external,2020,2021,tier1,Wetland,0.20,0.10,0.50,0.30\n",
        encoding="utf-8",
    )
    _write_gate_report(results_dir)

    make_benchmark_figures(results_dir=results_dir, figure_dir=figure_dir)

    assert (figure_dir / "fig_paper58_benchmark_gate.pdf").stat().st_size > 0
    assert (figure_dir / "fig_paper58_benchmark_gate.png").stat().st_size > 0


def test_make_benchmark_figures_uses_all_rows_when_no_tier1_rows(tmp_path: Path):
    results_dir = tmp_path / "results"
    figure_dir = tmp_path / "figures"
    results_dir.mkdir()
    (results_dir / "benchmark_metrics_by_pair.csv").write_text(
        "area,start_year,end_year,tier,stratum,primary_change_advantage,spatial_change_advantage,model_change_f1,best_non_neural_change_f1\n"
        "external,2020,2021,tier2,Wetland,0.20,0.10,0.50,0.30\n",
        encoding="utf-8",
    )
    _write_gate_report(results_dir)

    make_benchmark_figures(results_dir=results_dir, figure_dir=figure_dir)

    assert (figure_dir / "fig_paper58_benchmark_gate.pdf").stat().st_size > 0
    assert (figure_dir / "fig_paper58_benchmark_gate.png").stat().st_size > 0
