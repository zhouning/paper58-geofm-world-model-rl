from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper58_benchmark.evaluate_benchmark import _read_registry
from scripts.paper58_benchmark.flus import load_flus_prediction
from scripts.paper58_benchmark.las_allocation import allocate_demand_constrained
from scripts.paper58_benchmark.las_demand import derive_change_budget, derive_demand
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
    "start_year",
    "end_year",
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
    candidate = Path(str(value))
    if candidate.exists():
        return candidate
    normalized = str(value).replace("\\", "/")
    for marker in ("data/", "paper/", "experiments/"):
        if marker in normalized:
            relocated = Path.cwd() / normalized[normalized.index(marker):]
            if relocated.exists():
                return relocated
    return candidate


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


def _failure_record(row: dict[str, Any], qc_status: str, reason: str) -> dict[str, Any]:
    fields = ["area", "start_year", "end_year", "tier", "qc_status", "excluded_reason"]
    record = {field: row.get(field, "") for field in fields}
    record["qc_status"] = qc_status
    record["excluded_reason"] = reason
    return record


def _write_failures(path: Path, failures: list[dict[str, Any]]) -> None:
    fields = ["area", "start_year", "end_year", "tier", "qc_status", "excluded_reason"]
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


def _with_years(metric_row: dict[str, Any], start_year: int, end_year: int) -> dict[str, Any]:
    return {**metric_row, "start_year": start_year, "end_year": end_year}


def _row_neighborhood_weight(
    target_change_pixels: int | None,
    n_pixels: int,
    neighborhood_weight: float,
    adaptive_neighborhood_weight: float | None,
    adaptive_change_fraction_low: float,
    adaptive_change_fraction_high: float,
) -> float:
    if adaptive_neighborhood_weight is None or target_change_pixels is None or n_pixels <= 0:
        return float(neighborhood_weight)
    if adaptive_change_fraction_low < 0.0 or adaptive_change_fraction_high > 1.0:
        raise ValueError("adaptive change fraction bounds must be in [0, 1]")
    if adaptive_change_fraction_low >= adaptive_change_fraction_high:
        raise ValueError("adaptive_change_fraction_low must be smaller than adaptive_change_fraction_high")
    change_fraction = float(target_change_pixels) / float(n_pixels)
    if adaptive_change_fraction_low <= change_fraction < adaptive_change_fraction_high:
        return float(adaptive_neighborhood_weight)
    return float(neighborhood_weight)


def evaluate_las(
    registry_path: Path = DEFAULT_BENCHMARK_DIR / "benchmark_registry.json",
    output_dir: Path = DEFAULT_BENCHMARK_DIR.parent / "las_results",
    neighborhood_weight: float = 0.0,
    adaptive_neighborhood_weight: float | None = None,
    adaptive_change_fraction_low: float = 0.0,
    adaptive_change_fraction_high: float = 1.0,
    latent_neighborhood_weight: float = 0.0,
    demand_source: str = "observed_end",
    demand_blend_weight: float = 0.0,
    adaptive_demand_l1_threshold: float | None = None,
    adaptive_demand_change_fraction_high: float = 1.0,
    change_budget_source: str = "paper58_prediction",
    change_budget_scale: float = 1.0,
    balanced_swap_min_margin: float | None = None,
    balanced_swap_min_base_score: float | None = None,
) -> dict[str, Any]:
    registry_rows = _read_registry(Path(registry_path))
    included_rows = [row for row in registry_rows if row.get("qc_status") == "include"]
    output = Path(output_dir)
    simulated_dir = output / "simulated"
    output.mkdir(parents=True, exist_ok=True)
    simulated_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, int | float | str]] = []
    failure_rows = [
        _failure_record(row, str(row.get("qc_status", "")), str(row.get("excluded_reason", "")))
        for row in registry_rows
        if row.get("qc_status") != "include"
    ]
    evaluated_rows = 0

    for row in included_rows:
        try:
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
            embedding_start = None
            if latent_neighborhood_weight > 0.0:
                embedding_start = _load_array(row.get("embedding_start_path")).astype(np.float32, copy=False)
            target_demand = derive_demand(
                start,
                end,
                paper58_pred,
                demand_source=demand_source,
                class_values=class_values,
                transition_prior=prior,
                demand_blend_weight=demand_blend_weight,
                adaptive_demand_l1_threshold=adaptive_demand_l1_threshold,
                adaptive_demand_change_fraction_high=adaptive_demand_change_fraction_high,
            )
            target_change_pixels = derive_change_budget(
                start,
                end,
                paper58_pred,
                target_demand,
                change_budget_source=change_budget_source,
                change_budget_scale=change_budget_scale,
            )
            allocation = allocate_demand_constrained(
                start,
                suitability,
                class_values=class_values,
                target_demand=target_demand,
                target_change_pixels=target_change_pixels,
                neighborhood_weight=_row_neighborhood_weight(
                    target_change_pixels,
                    int(start.size),
                    neighborhood_weight,
                    adaptive_neighborhood_weight,
                    adaptive_change_fraction_low,
                    adaptive_change_fraction_high,
                ),
                embedding_grid=embedding_start,
                latent_neighborhood_weight=latent_neighborhood_weight,
                balanced_swap_min_margin=balanced_swap_min_margin,
                balanced_swap_min_base_score=balanced_swap_min_base_score,
            )
        except Exception as exc:
            failure_rows.append(_failure_record(row, "runtime_failure", f"{type(exc).__name__}: {exc}"))
            continue

        try:
            area = str(row.get("area"))
            start_year = int(row.get("start_year"))
            end_year = int(row.get("end_year"))
            row_metric_rows = [
                _with_years(
                    method_metric_row(
                        "paper58_direct",
                        area,
                        str(row.get("tier")),
                        str(row.get("stratum")),
                        start,
                        end,
                        paper58_pred,
                    ),
                    start_year,
                    end_year,
                )
            ]
            row_metric_rows.append(
                _with_years(
                    method_metric_row(
                        "paper58_las",
                        area,
                        str(row.get("tier")),
                        str(row.get("stratum")),
                        start,
                        end,
                        allocation.simulated_map,
                    ),
                    start_year,
                    end_year,
                )
            )
            if flus_pred is not None:
                row_metric_rows.append(
                    _with_years(
                        method_metric_row(
                            "flus",
                            area,
                            str(row.get("tier")),
                            str(row.get("stratum")),
                            start,
                            end,
                            flus_pred,
                        ),
                        start_year,
                        end_year,
                    )
                )

            row_selected_rows = [
                {
                    "area": area,
                    "start_year": start_year,
                    "end_year": end_year,
                    **selected,
                }
                for selected in allocation.selected_transitions
            ]
            np.save(
                simulated_dir / f"{area}_{start_year}_{end_year}_paper58_las.npy",
                allocation.simulated_map.astype(np.int32, copy=False),
            )
        except Exception as exc:
            failure_rows.append(_failure_record(row, "runtime_failure", f"{type(exc).__name__}: {exc}"))
            continue

        metric_rows.extend(row_metric_rows)
        selected_rows.extend(row_selected_rows)
        evaluated_rows += 1

    summary = {
        "n_registry_rows": len(registry_rows),
        "n_evaluated_rows": evaluated_rows,
        "n_failed_rows": len(failure_rows),
        "n_metric_rows": len(metric_rows),
        "methods": _method_names(metric_rows),
        "demand_source": demand_source,
        "demand_blend_weight": float(demand_blend_weight),
        "adaptive_demand_l1_threshold": adaptive_demand_l1_threshold,
        "adaptive_demand_change_fraction_high": float(adaptive_demand_change_fraction_high),
        "adaptive_neighborhood_weight": adaptive_neighborhood_weight,
        "adaptive_change_fraction_low": float(adaptive_change_fraction_low),
        "adaptive_change_fraction_high": float(adaptive_change_fraction_high),
        "change_budget_source": change_budget_source,
        "change_budget_scale": float(change_budget_scale),
        "balanced_swap_min_margin": balanced_swap_min_margin,
        "balanced_swap_min_base_score": balanced_swap_min_base_score,
    }
    result = {"summary": summary, "metrics": metric_rows}
    write_csv(output / "las_metrics_by_method.csv", metric_rows, LAS_METRIC_FIELDS)
    write_json(output / "las_summary.json", result)
    _write_failures(output / "las_failures.csv", failure_rows)
    _write_selected_transitions(output / "las_selected_transitions.csv", selected_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Paper58-LAS against Paper58 direct and FLUS-compatible outputs.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_BENCHMARK_DIR / "benchmark_registry.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_DIR.parent / "las_results")
    parser.add_argument(
        "--neighborhood-weight",
        type=float,
        default=0.0,
        help="Weight for class-neighborhood affinity during LAS allocation.",
    )
    parser.add_argument(
        "--latent-neighborhood-weight",
        type=float,
        default=0.0,
        help="Weight for start-embedding semantic neighborhood affinity during LAS allocation.",
    )
    parser.add_argument(
        "--adaptive-neighborhood-weight",
        type=float,
        default=None,
        help="Optional alternate neighborhood weight for rows whose predicted change fraction falls in a target range.",
    )
    parser.add_argument(
        "--adaptive-change-fraction-low",
        type=float,
        default=0.0,
        help="Lower bound of predicted change fraction for applying adaptive-neighborhood-weight.",
    )
    parser.add_argument(
        "--adaptive-change-fraction-high",
        type=float,
        default=1.0,
        help="Upper bound of predicted change fraction for applying adaptive-neighborhood-weight.",
    )
    parser.add_argument(
        "--demand-source",
        choices=[
            "observed_end",
            "paper58_prediction",
            "start_persistence",
            "transition_prior",
            "transition_prior_blend",
            "transition_prior_adaptive_blend",
        ],
        default="observed_end",
        help="Demand source for LAS allocation. observed_end is oracle demand; paper58_prediction is non-oracle.",
    )
    parser.add_argument(
        "--demand-blend-weight",
        type=float,
        default=0.0,
        help="Weight of Paper58 prediction class counts when demand_source=transition_prior_blend.",
    )
    parser.add_argument(
        "--adaptive-demand-l1-threshold",
        type=float,
        default=None,
        help=(
            "L1 demand-disagreement fraction required before transition_prior_adaptive_blend "
            "uses blended demand."
        ),
    )
    parser.add_argument(
        "--adaptive-demand-change-fraction-high",
        type=float,
        default=1.0,
        help=(
            "Maximum Paper58 prediction change fraction allowed before "
            "transition_prior_adaptive_blend uses blended demand."
        ),
    )
    parser.add_argument(
        "--change-budget-source",
        choices=["paper58_prediction", "observed_end", "demand_delta", "none"],
        default="paper58_prediction",
        help="Gross change budget for balanced LAS swaps. demand_delta uses only the minimum change implied by demand.",
    )
    parser.add_argument(
        "--change-budget-scale",
        type=float,
        default=1.0,
        help="Scale applied to the selected gross change budget, bounded below by the demand-delta budget.",
    )
    parser.add_argument(
        "--balanced-swap-min-margin",
        type=float,
        default=None,
        help=(
            "Optional evidence margin floor for extra balanced swaps. "
            "A swap pair is allowed only when change score exceeds persistence score by at least this value."
        ),
    )
    parser.add_argument(
        "--balanced-swap-min-base-score",
        type=float,
        default=None,
        help=(
            "Optional raw Paper58/GeoFM suitability floor for extra balanced swaps. "
            "The pair-level base score sum must meet this floor before neighborhood terms are added."
        ),
    )
    args = parser.parse_args()
    result = evaluate_las(
        registry_path=args.registry,
        output_dir=args.output_dir,
        neighborhood_weight=args.neighborhood_weight,
        adaptive_neighborhood_weight=args.adaptive_neighborhood_weight,
        adaptive_change_fraction_low=args.adaptive_change_fraction_low,
        adaptive_change_fraction_high=args.adaptive_change_fraction_high,
        latent_neighborhood_weight=args.latent_neighborhood_weight,
        demand_source=args.demand_source,
        demand_blend_weight=args.demand_blend_weight,
        adaptive_demand_l1_threshold=args.adaptive_demand_l1_threshold,
        adaptive_demand_change_fraction_high=args.adaptive_demand_change_fraction_high,
        change_budget_source=args.change_budget_source,
        change_budget_scale=args.change_budget_scale,
        balanced_swap_min_margin=args.balanced_swap_min_margin,
        balanced_swap_min_base_score=args.balanced_swap_min_base_score,
    )
    print(
        "Paper58-LAS evaluation: "
        f"{result['summary']['n_evaluated_rows']} evaluated row(s), "
        f"methods={','.join(result['summary']['methods'])}"
    )


if __name__ == "__main__":
    main()
