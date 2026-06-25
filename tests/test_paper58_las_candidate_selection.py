import csv
from pathlib import Path

import pytest

from scripts.paper58_benchmark.select_las_candidate_loo import (
    select_las_candidates_leave_one_area_out,
)


FIELDS = [
    "area",
    "start_year",
    "end_year",
    "tier",
    "stratum",
    "change_f1_advantage",
    "fom_advantage",
    "change_recall_advantage",
    "transition_accuracy_advantage",
    "quantity_disagreement_advantage",
    "allocation_disagreement_advantage",
]


def _row(area: str, change_f1: float, fom: float = 0.0) -> dict[str, object]:
    return {
        "area": area,
        "start_year": "2020",
        "end_year": "2021",
        "tier": "tier1",
        "stratum": "Urban",
        "change_f1_advantage": change_f1,
        "fom_advantage": fom,
        "change_recall_advantage": change_f1 + 0.1,
        "transition_accuracy_advantage": change_f1 + 0.2,
        "quantity_disagreement_advantage": 0.0,
        "allocation_disagreement_advantage": -0.01,
    }


def _write_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_leave_one_area_out_selection_does_not_use_held_out_area(tmp_path: Path):
    fixed = tmp_path / "fixed" / "las_comparison_by_area.csv"
    adaptive = tmp_path / "adaptive" / "las_comparison_by_area.csv"
    _write_comparison(
        fixed,
        [
            _row("area_a", 0.50),
            _row("area_b", 0.10),
            _row("area_c", 0.10),
        ],
    )
    _write_comparison(
        adaptive,
        [
            _row("area_a", 0.00),
            _row("area_b", 0.30),
            _row("area_c", 0.30),
        ],
    )

    result = select_las_candidates_leave_one_area_out(
        {"fixed": fixed, "adaptive": adaptive},
        output_dir=tmp_path / "loo",
        primary_metric="change_f1",
        n_boot=100,
    )

    selected = {row["area"]: row["selected_candidate"] for row in result["selected_rows"]}
    assert selected == {
        "area_a": "adaptive",
        "area_b": "fixed",
        "area_c": "fixed",
    }
    assert result["summary"]["selected_candidate_counts"] == {"adaptive": 1, "fixed": 2}
    assert result["summary"]["holdout_advantages"]["change_f1"]["mean_advantage"] == pytest.approx(0.0666666667)
    assert (tmp_path / "loo" / "loo_selected_candidates.csv").exists()
    assert (tmp_path / "loo" / "loo_selection_summary.json").exists()


def test_leave_one_area_out_selection_uses_tie_break_metrics(tmp_path: Path):
    lower_fom = tmp_path / "lower_fom" / "las_comparison_by_area.csv"
    higher_fom = tmp_path / "higher_fom" / "las_comparison_by_area.csv"
    _write_comparison(
        lower_fom,
        [
            _row("area_a", 0.20, 0.01),
            _row("area_b", 0.20, 0.01),
            _row("area_c", 0.20, 0.01),
        ],
    )
    _write_comparison(
        higher_fom,
        [
            _row("area_a", 0.20, 0.10),
            _row("area_b", 0.20, 0.10),
            _row("area_c", 0.20, 0.10),
        ],
    )

    result = select_las_candidates_leave_one_area_out(
        {"lower_fom": lower_fom, "higher_fom": higher_fom},
        output_dir=tmp_path / "loo",
        primary_metric="change_f1",
        tie_break_metrics=["fom"],
        n_boot=100,
    )

    assert {row["selected_candidate"] for row in result["selected_rows"]} == {"higher_fom"}
