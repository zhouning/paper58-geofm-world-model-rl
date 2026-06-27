import csv
from pathlib import Path

import pytest


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_seeded_dir(
    root: Path,
    area: str,
    deltas: dict[str, tuple[float, int, bool]],
) -> Path:
    out = root / area
    _write_csv(
        out / "seeded_metric_summary_by_method.csv",
        [
            {
                "method": "paper58_change_gate",
                "area": area,
                "n": 5,
                "mean_change_f1": 0.3,
                "std_change_f1": 0.0,
                "min_change_f1": 0.3,
                "max_change_f1": 0.3,
                "mean_fom": 0.1,
                "std_fom": 0.0,
                "min_fom": 0.1,
                "max_fom": 0.1,
                "mean_transition_accuracy": 0.2,
                "std_transition_accuracy": 0.0,
                "min_transition_accuracy": 0.2,
                "max_transition_accuracy": 0.2,
                "mean_allocation_disagreement": 0.05,
                "std_allocation_disagreement": 0.0,
                "min_allocation_disagreement": 0.05,
                "max_allocation_disagreement": 0.05,
            }
        ],
    )
    _write_csv(
        out / "seeded_delta_summary.csv",
        [
            {
                "challenger": "paper58_change_gate",
                "baseline": "geosos_flus_console",
                "metric": metric,
                "n": 5,
                "mean_delta": mean_delta,
                "std_delta": 0.01,
                "min_delta": mean_delta - 0.01,
                "max_delta": mean_delta + 0.01,
                "n_better": n_better,
                "better_rate": n_better / 5,
                "higher_is_better": higher_is_better,
            }
            for metric, (mean_delta, n_better, higher_is_better) in deltas.items()
        ],
    )
    return out


def test_cross_area_summary_counts_metric_direction_and_seed_wins(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.summarize_seeded_flus_cross_area import summarize_cross_area

    area_a = _write_seeded_dir(
        tmp_path,
        "area_a",
        {
            "change_f1": (0.10, 5, True),
            "fom": (-0.02, 1, True),
            "transition_accuracy": (0.04, 4, True),
            "allocation_disagreement": (-0.03, 5, False),
        },
    )
    area_b = _write_seeded_dir(
        tmp_path,
        "area_b",
        {
            "change_f1": (-0.05, 0, True),
            "fom": (0.01, 3, True),
            "transition_accuracy": (-0.06, 0, True),
            "allocation_disagreement": (0.02, 0, False),
        },
    )

    summary = summarize_cross_area([area_a, area_b])
    by_metric = {row["metric"]: row for row in summary.metric_rows}

    assert by_metric["change_f1"]["n_areas"] == 2
    assert by_metric["change_f1"]["area_better_count"] == 1
    assert by_metric["change_f1"]["weighted_seed_better_rate"] == pytest.approx(0.5)
    assert by_metric["change_f1"]["mean_area_delta"] == pytest.approx(0.025)
    assert by_metric["allocation_disagreement"]["area_better_count"] == 1
    assert by_metric["allocation_disagreement"]["mean_area_benefit"] == pytest.approx(0.005)


def test_cross_area_report_contains_chinese_conclusion_and_links(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.summarize_seeded_flus_cross_area import (
        summarize_cross_area,
        write_cross_area_report,
    )

    area_a = _write_seeded_dir(
        tmp_path,
        "area_a",
        {
            "change_f1": (0.10, 5, True),
            "fom": (0.02, 5, True),
            "transition_accuracy": (-0.01, 0, True),
            "allocation_disagreement": (-0.03, 5, False),
        },
    )
    same_grid = tmp_path / "same_grid_area_a"
    same_grid.mkdir()
    (same_grid / "README.md").write_text("# same grid\n", encoding="utf-8")

    summary = summarize_cross_area([area_a])
    output = tmp_path / "report"
    write_cross_area_report(summary, output, same_grid_report_dirs=[same_grid], create_figures=False)

    text = (output / "README.md").read_text(encoding="utf-8")
    assert "跨区域稳健性结论" in text
    assert "不能表述为 Paper58 已经在所有指标上完全超过 GeoSOS-FLUS" in text
    assert "same_grid_area_a/README.md" in text
