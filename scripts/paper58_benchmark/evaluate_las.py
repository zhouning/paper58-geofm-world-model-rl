from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper58_benchmark.evaluate_benchmark import _read_registry
from scripts.paper58_benchmark.flus import load_flus_prediction
from scripts.paper58_benchmark.las_allocation import allocate_demand_constrained
from scripts.paper58_benchmark.las_demand import derive_observed_demand
from scripts.paper58_benchmark.las_metrics import method_metric_row
from scripts.paper58_benchmark.las_suitability import (
    build_transition_suitability,
    class_values_from_maps,
    one_hot_probability_cube,
    transition_prior_from_pairs,
)
from scripts.paper58_benchmark.schema import DEFAULT_BENCHMARK_DIR, write_csv, write_json


LAS_METRIC_FIELDS = [
    "method",
    "area",
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


def _path(value: object) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _load_array(path_value: object) -> np.ndarray:
    path = _path(path_value)
    if path is None:
        raise FileNotFoundError("missing array path")
    return np.load(path)


def _transition_training_pairs(
    rows: list[dict[str, Any]],
    target: dict[str, Any],
    target_shape: tuple[int, ...],
) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for row in rows:
        if row.get("area") == target.get("area"):
            continue
        if row.get("qc_status") != "include":
            continue
        start_path = _path(row.get("label_start_path"))
        end_path = _path(row.get("label_end_path"))
        if start_path is None or end_path is None or not start_path.exists() or not end_path.exists():
            continue
        start = np.load(start_path)
        end = np.load(end_path)
        if start.shape == end.shape == target_shape:
            pairs.append((start, end))
    return pairs


def _write_failures(path: Path, registry_rows: list[dict[str, Any]]) -> None:
    fields = ["area", "start_year", "end_year", "tier", "qc_status", "excluded_reason"]
    failures = [
        {field: row.get(field, "") for field in fields}
        for row in registry_rows
        if row.get("qc_status") != "include"
    ]
    write_csv(path, failures, fields)


def _write_selected_transitions(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    fields = ["area", "start_year", "end_year", "row", "col", "from_class", "to_class", "score"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _method_names(metric_rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["method"]) for row in metric_rows})


def evaluate_las(
    registry_path: Path = DEFAULT_BENCHMARK_DIR / "benchmark_registry.json",
    output_dir: Path = DEFAULT_BENCHMARK_DIR.parent / "las_results",
) -> dict[str, Any]:
    registry_rows = _read_registry(Path(registry_path))
    included_rows = [row for row in registry_rows if row.get("qc_status") == "include"]
    output = Path(output_dir)
    simulated_dir = output / "simulated"
    output.mkdir(parents=True, exist_ok=True)
    simulated_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, int | float | str]] = []

    for row in included_rows:
        start = _load_array(row.get("label_start_path")).astype(np.int32, copy=False)
        end = _load_array(row.get("label_end_path")).astype(np.int32, copy=False)
        paper58_pred = _load_array(row.get("prediction_path")).astype(np.int32, copy=False)
        if start.shape != end.shape or start.shape != paper58_pred.shape:
            raise ValueError(
                f"shape mismatch for {row.get('area')}: start={start.shape}, "
                f"end={end.shape}, pred={paper58_pred.shape}"
            )

        class_values = class_values_from_maps(start, end, paper58_pred)
        flus_path = _path(row.get("flus_prediction_path"))
        flus_pred = None
        if flus_path is not None:
            flus_pred = load_flus_prediction(flus_path, expected_shape=start.shape, allowed_classes=set(class_values))
            class_values = class_values_from_maps(start, end, paper58_pred, flus_pred)

        start_probs = one_hot_probability_cube(start, class_values, confidence=0.95, floor=0.01)
        forecast_probs = one_hot_probability_cube(paper58_pred, class_values, confidence=0.95, floor=0.01)
        prior = transition_prior_from_pairs(_transition_training_pairs(included_rows, row, start.shape), class_values)
        suitability = build_transition_suitability(
            start,
            class_values=class_values,
            forecast_probs=forecast_probs,
            start_probs=start_probs,
            transition_prior=prior,
        )
        allocation = allocate_demand_constrained(
            start,
            suitability,
            class_values=class_values,
            target_demand=derive_observed_demand(end),
        )

        area = str(row.get("area"))
        start_year = int(row.get("start_year"))
        end_year = int(row.get("end_year"))
        np.save(
            simulated_dir / f"{area}_{start_year}_{end_year}_paper58_las.npy",
            allocation.simulated_map.astype(np.int32, copy=False),
        )

        metric_rows.append(
            method_metric_row(
                "paper58_direct",
                area,
                str(row.get("tier")),
                str(row.get("stratum")),
                start,
                end,
                paper58_pred,
            )
        )
        metric_rows.append(
            method_metric_row(
                "paper58_las",
                area,
                str(row.get("tier")),
                str(row.get("stratum")),
                start,
                end,
                allocation.simulated_map,
            )
        )
        if flus_pred is not None:
            metric_rows.append(
                method_metric_row(
                    "flus",
                    area,
                    str(row.get("tier")),
                    str(row.get("stratum")),
                    start,
                    end,
                    flus_pred,
                )
            )

        for selected in allocation.selected_transitions:
            selected_rows.append(
                {
                    "area": area,
                    "start_year": start_year,
                    "end_year": end_year,
                    **selected,
                }
            )

    summary = {
        "n_registry_rows": len(registry_rows),
        "n_evaluated_rows": len(included_rows),
        "n_metric_rows": len(metric_rows),
        "methods": _method_names(metric_rows),
    }
    result = {"summary": summary, "metrics": metric_rows}
    write_csv(output / "las_metrics_by_method.csv", metric_rows, LAS_METRIC_FIELDS)
    write_json(output / "las_summary.json", result)
    _write_failures(output / "las_failures.csv", registry_rows)
    _write_selected_transitions(output / "las_selected_transitions.csv", selected_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Paper58-LAS against Paper58 direct and FLUS-compatible outputs.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_BENCHMARK_DIR / "benchmark_registry.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_DIR.parent / "las_results")
    args = parser.parse_args()
    result = evaluate_las(registry_path=args.registry, output_dir=args.output_dir)
    print(
        "Paper58-LAS evaluation: "
        f"{result['summary']['n_evaluated_rows']} evaluated row(s), "
        f"methods={','.join(result['summary']['methods'])}"
    )


if __name__ == "__main__":
    main()
