from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from scripts.paper58_benchmark.schema import write_csv, write_json
from scripts.paper58_benchmark.statistics import clustered_bootstrap_ci, paired_sign_test


DEFAULT_METRICS = [
    "change_f1",
    "fom",
    "change_recall",
    "transition_accuracy",
    "quantity_disagreement",
    "allocation_disagreement",
]
CONTEXT_FIELDS = ["area", "start_year", "end_year", "tier", "stratum"]


def _read_comparison_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _advantage_key(metric: str) -> str:
    if metric.endswith("_advantage"):
        return metric
    return f"{metric}_advantage"


def _candidate_rows_by_area(candidate_paths: dict[str, Path]) -> dict[str, dict[str, dict[str, str]]]:
    if not candidate_paths:
        raise ValueError("at least one candidate comparison file is required")

    by_candidate: dict[str, dict[str, dict[str, str]]] = {}
    for candidate, path in sorted(candidate_paths.items()):
        rows = _read_comparison_rows(Path(path))
        if not rows:
            raise ValueError(f"{candidate} has no comparison rows: {path}")
        by_area: dict[str, dict[str, str]] = {}
        for row in rows:
            area = str(row.get("area", "")).strip()
            if not area:
                raise ValueError(f"{candidate} contains a row without area: {path}")
            if area in by_area:
                raise ValueError(f"{candidate} contains duplicate area {area}: {path}")
            by_area[area] = row
        by_candidate[candidate] = by_area
    return by_candidate


def _common_areas(by_candidate: dict[str, dict[str, dict[str, str]]]) -> list[str]:
    area_sets = [set(rows) for rows in by_candidate.values()]
    common = sorted(set.intersection(*area_sets)) if area_sets else []
    if not common:
        raise ValueError("candidate comparison files have no common areas")
    missing = {
        candidate: sorted(set(common).symmetric_difference(set(rows)))
        for candidate, rows in by_candidate.items()
        if set(rows) != set(common)
    }
    if missing:
        details = "; ".join(f"{candidate}: {areas}" for candidate, areas in sorted(missing.items()))
        raise ValueError(f"candidate comparison files must contain the same areas; mismatches: {details}")
    return common


def _metric_mean(rows: list[dict[str, str]], metric: str) -> float:
    key = _advantage_key(metric)
    values = [value for value in (_finite_float(row.get(key)) for row in rows) if value is not None]
    if not values:
        return float("-inf")
    return float(mean(values))


def _selection_score(
    candidate_rows: list[dict[str, str]],
    primary_metric: str,
    tie_break_metrics: list[str],
) -> tuple[float, ...]:
    return tuple(_metric_mean(candidate_rows, metric) for metric in [primary_metric, *tie_break_metrics])


def _select_candidate_for_area(
    held_out_area: str,
    by_candidate: dict[str, dict[str, dict[str, str]]],
    primary_metric: str,
    tie_break_metrics: list[str],
) -> tuple[str, tuple[float, ...], int]:
    ranked: list[tuple[tuple[float, ...], str, int]] = []
    for candidate, rows_by_area in by_candidate.items():
        train_rows = [row for area, row in rows_by_area.items() if area != held_out_area]
        ranked.append((_selection_score(train_rows, primary_metric, tie_break_metrics), candidate, len(train_rows)))
    ranked.sort(key=lambda item: (*item[0], item[1]), reverse=True)
    score, candidate, n_train = ranked[0]
    return candidate, score, n_train


def _selected_row(
    held_out_area: str,
    selected_candidate: str,
    selected_score: tuple[float, ...],
    selection_train_n: int,
    by_candidate: dict[str, dict[str, dict[str, str]]],
    primary_metric: str,
    tie_break_metrics: list[str],
    metrics: list[str],
) -> dict[str, Any]:
    source = by_candidate[selected_candidate][held_out_area]
    row: dict[str, Any] = {
        field: source.get(field, "")
        for field in CONTEXT_FIELDS
    }
    row["selected_candidate"] = selected_candidate
    row["selection_train_n"] = int(selection_train_n)
    for index, metric in enumerate([primary_metric, *tie_break_metrics]):
        row[f"selection_train_{_advantage_key(metric)}_mean"] = selected_score[index]
    for metric in metrics:
        key = _advantage_key(metric)
        row[key] = _finite_float(source.get(key))
    return row


def _advantage_summary(rows: list[dict[str, Any]], metric: str, n_boot: int, seed: int) -> dict[str, Any]:
    key = _advantage_key(metric)
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    ci_rows = [{"area": row["area"], key: float(row[key])} for row in rows if isinstance(row.get(key), (int, float))]
    sign = paired_sign_test(values)
    ci = clustered_bootstrap_ci(ci_rows, key, n_boot=n_boot, seed=seed)
    return {
        "n": len(values),
        "mean_advantage": float(mean(values)) if values else None,
        "n_positive": int(sum(value > 0 for value in values)),
        "n_negative": int(sum(value < 0 for value in values)),
        "bootstrap_ci": ci,
        "sign_test": sign,
    }


def select_las_candidates_leave_one_area_out(
    candidate_paths: dict[str, Path],
    output_dir: Path,
    primary_metric: str = "change_f1",
    tie_break_metrics: list[str] | None = None,
    metrics: list[str] | None = None,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    selected_tie_break_metrics = list(tie_break_metrics or [])
    selected_metrics = list(metrics or DEFAULT_METRICS)
    by_candidate = _candidate_rows_by_area(candidate_paths)
    areas = _common_areas(by_candidate)

    selected_rows: list[dict[str, Any]] = []
    for area in areas:
        candidate, score, n_train = _select_candidate_for_area(
            area,
            by_candidate,
            primary_metric,
            selected_tie_break_metrics,
        )
        selected_rows.append(
            _selected_row(
                area,
                candidate,
                score,
                n_train,
                by_candidate,
                primary_metric,
                selected_tie_break_metrics,
                selected_metrics,
            )
        )

    output = Path(output_dir)
    selection_fields = [
        *CONTEXT_FIELDS,
        "selected_candidate",
        "selection_train_n",
        *[
            f"selection_train_{_advantage_key(metric)}_mean"
            for metric in [primary_metric, *selected_tie_break_metrics]
        ],
        *[_advantage_key(metric) for metric in selected_metrics],
    ]
    write_csv(output / "loo_selected_candidates.csv", selected_rows, selection_fields)

    summary = {
        "n_areas": len(areas),
        "candidate_names": sorted(candidate_paths),
        "primary_metric": primary_metric,
        "tie_break_metrics": selected_tie_break_metrics,
        "metrics": selected_metrics,
        "selected_candidate_counts": dict(sorted(Counter(row["selected_candidate"] for row in selected_rows).items())),
        "holdout_advantages": {
            metric: _advantage_summary(selected_rows, metric, n_boot=n_boot, seed=seed + index)
            for index, metric in enumerate(selected_metrics)
        },
    }
    write_json(output / "loo_selection_summary.json", summary)
    return {"selected_rows": selected_rows, "summary": summary}


def _parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("candidate name cannot be empty")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-area-out selection audit for LAS candidate settings.")
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        required=True,
        help="Candidate comparison CSV as NAME=PATH. Repeat for multiple candidates.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-metric", default="change_f1")
    parser.add_argument("--tie-break-metric", action="append", default=[])
    parser.add_argument("--metric", action="append", default=None)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidate_paths = dict(args.candidate)
    result = select_las_candidates_leave_one_area_out(
        candidate_paths,
        output_dir=args.output_dir,
        primary_metric=args.primary_metric,
        tie_break_metrics=list(args.tie_break_metric),
        metrics=args.metric,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    primary = result["summary"]["holdout_advantages"].get(args.primary_metric, {})
    print(
        "LOAO LAS candidate selection: "
        f"{result['summary']['n_areas']} area(s), "
        f"{args.primary_metric}_advantage={primary.get('mean_advantage')}"
    )


if __name__ == "__main__":
    main()
