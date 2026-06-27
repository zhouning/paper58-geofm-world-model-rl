from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper58_benchmark.apply_paper58_spatial_demand_allocation import (
    _true_change_fraction,
    apply_demand_calibrated_spatial_gate,
    change_demand_features,
    fit_change_ratio_demand_model,
    predict_ratio_change_fraction,
)
from scripts.paper58_benchmark.las_metrics import method_metric_row
from scripts.paper58_benchmark.sweep_paper58_change_gate import GateSweepCase, load_case_from_change_gate_dir


METRIC_FIELDS = [
    "change_f1",
    "fom",
    "transition_accuracy",
    "allocation_disagreement",
]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_ready(row.get(field, "")) for field in fields})


def ratio_parameter_grid(
    ratio_quantiles: list[float],
    ratio_multipliers: list[float],
    min_fractions: list[float],
    max_fractions: list[float],
    target_neighborhood_weights: list[float],
    source_neighborhood_penalties: list[float],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for ratio_quantile in ratio_quantiles:
        for ratio_multiplier in ratio_multipliers:
            for min_fraction in min_fractions:
                for max_fraction in max_fractions:
                    for target_weight in target_neighborhood_weights:
                        for source_penalty in source_neighborhood_penalties:
                            rows.append(
                                {
                                    "ratio_quantile": float(ratio_quantile),
                                    "ratio_multiplier": float(ratio_multiplier),
                                    "min_fraction": float(min_fraction),
                                    "max_fraction": float(max_fraction),
                                    "target_neighborhood_weight": float(target_weight),
                                    "source_neighborhood_penalty": float(source_penalty),
                                }
                            )
    return rows


def _rank_parameter_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["mean_fom"]),
            float(row["mean_allocation_disagreement"]),
            -float(row["mean_change_f1"]),
            -float(row.get("mean_transition_accuracy", 0.0)),
        ),
    )


def select_best_parameter_row(
    rows: list[dict[str, Any]],
    *,
    min_mean_transition_accuracy: float | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one parameter row is required")
    candidates = rows
    if min_mean_transition_accuracy is not None:
        eligible = [
            row
            for row in rows
            if float(row.get("mean_transition_accuracy", 0.0)) >= float(min_mean_transition_accuracy)
        ]
        if eligible:
            candidates = eligible
    return _rank_parameter_rows(candidates)[0]


def _case_target_change_fraction(case: GateSweepCase) -> float:
    return _true_change_fraction(case.start_map, case.end_map, case.prediction_map)


def _case_predicted_change_fraction(case: GateSweepCase) -> float:
    return float(change_demand_features(case.start_map, case.prediction_map)[0])


def _valid_mask(case: GateSweepCase) -> np.ndarray:
    return (case.start_map != 0) & (case.end_map != 0) & (case.prediction_map != 0)


def evaluate_ratio_parameters_leave_one(
    cases: list[GateSweepCase],
    parameters: dict[str, float],
) -> dict[str, Any]:
    if len(cases) < 2:
        raise ValueError("at least two calibration cases are required for leave-one-area tuning")
    rows: list[dict[str, Any]] = []
    target_fractions: list[float] = []
    for holdout_index, holdout in enumerate(cases):
        training_cases = [case for index, case in enumerate(cases) if index != holdout_index]
        predicted_fractions = np.array([_case_predicted_change_fraction(case) for case in training_cases], dtype=np.float32)
        target_change_fractions = np.array([_case_target_change_fraction(case) for case in training_cases], dtype=np.float32)
        model = fit_change_ratio_demand_model(
            predicted_fractions,
            target_change_fractions,
            quantile=float(parameters["ratio_quantile"]),
            multiplier=float(parameters["ratio_multiplier"]),
        )
        candidate_fraction = _case_predicted_change_fraction(holdout)
        target_fraction = predict_ratio_change_fraction(
            model,
            candidate_fraction,
            min_fraction=float(parameters["min_fraction"]),
            max_fraction=float(parameters["max_fraction"]),
        )
        valid = _valid_mask(holdout)
        gated, diagnostics = apply_demand_calibrated_spatial_gate(
            holdout.start_map,
            holdout.prediction_map,
            holdout.score_map,
            target_change_fraction=target_fraction,
            valid_mask=valid,
            target_neighborhood_weight=float(parameters["target_neighborhood_weight"]),
            source_neighborhood_penalty=float(parameters["source_neighborhood_penalty"]),
        )
        metric = method_metric_row(
            method="paper58_spatial_demand_ratio_loo",
            area=holdout.area,
            tier="external_calibration_leave_one",
            stratum="ratio_demand_tuning",
            start_map=holdout.start_map[valid],
            true_map=holdout.end_map[valid],
            pred_map=gated[valid],
        )
        rows.append(
            {
                **metric,
                "start_year": holdout.start_year,
                "end_year": holdout.end_year,
                "predicted_target_change_fraction": target_fraction,
                "candidate_change_pixels": diagnostics["candidate_change_pixels"],
                "target_change_pixels": diagnostics["target_change_pixels"],
                "kept_change_pixels": diagnostics["kept_change_pixels"],
            }
        )
        target_fractions.append(target_fraction)

    summary = {key: float(value) for key, value in parameters.items()}
    for metric in METRIC_FIELDS:
        summary[f"mean_{metric}"] = float(np.mean([float(row[metric]) for row in rows]))
    summary["mean_predicted_target_change_fraction"] = float(np.mean(target_fractions))
    summary["n_cases"] = len(cases)
    return {**summary, "case_rows": rows}


def discover_change_gate_dirs(paths: list[Path], roots: list[Path]) -> list[Path]:
    dirs = [Path(path) for path in paths]
    for root in roots:
        dirs.extend(sorted(path for path in Path(root).iterdir() if (path / "manifest.json").exists()))
    unique: dict[str, Path] = {}
    for path in dirs:
        unique[path.resolve().as_posix()] = path
    return list(unique.values())


def run_ratio_tuning(
    change_gate_dirs: list[Path],
    output_dir: Path,
    grid: list[dict[str, float]],
    *,
    min_mean_transition_accuracy: float | None = None,
) -> dict[str, Any]:
    cases = [load_case_from_change_gate_dir(Path(path)) for path in change_gate_dirs]
    summary_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for index, parameters in enumerate(grid):
        result = evaluate_ratio_parameters_leave_one(cases, parameters)
        row = {key: value for key, value in result.items() if key != "case_rows"}
        row["parameter_index"] = index
        summary_rows.append(row)
        for case_row in result["case_rows"]:
            case_rows.append({**parameters, "parameter_index": index, **case_row})
    best = select_best_parameter_row(
        summary_rows,
        min_mean_transition_accuracy=min_mean_transition_accuracy,
    )
    transition_floor_eligible_rows = None
    if min_mean_transition_accuracy is not None:
        transition_floor_eligible_rows = sum(
            1
            for row in summary_rows
            if float(row.get("mean_transition_accuracy", 0.0)) >= float(min_mean_transition_accuracy)
        )
    output = Path(output_dir)
    _write_csv(
        output / "ratio_tuning_summary.csv",
        summary_rows,
        [
            "parameter_index",
            "ratio_quantile",
            "ratio_multiplier",
            "min_fraction",
            "max_fraction",
            "target_neighborhood_weight",
            "source_neighborhood_penalty",
            "n_cases",
            "mean_change_f1",
            "mean_fom",
            "mean_transition_accuracy",
            "mean_allocation_disagreement",
            "mean_predicted_target_change_fraction",
        ],
    )
    _write_csv(
        output / "ratio_tuning_case_metrics.csv",
        case_rows,
        [
            "parameter_index",
            "ratio_quantile",
            "ratio_multiplier",
            "min_fraction",
            "max_fraction",
            "target_neighborhood_weight",
            "source_neighborhood_penalty",
            "method",
            "area",
            "start_year",
            "end_year",
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
            "predicted_target_change_fraction",
            "candidate_change_pixels",
            "target_change_pixels",
            "kept_change_pixels",
        ],
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paper58_spatial_demand_ratio_leave_one_tuning",
        "change_gate_dirs": [Path(path) for path in change_gate_dirs],
        "n_cases": len(cases),
        "n_parameter_rows": len(grid),
        "selection_rule": (
            "if min_mean_transition_accuracy is reachable, restrict to rows meeting it; "
            "then maximize mean_fom, minimize mean_allocation_disagreement, maximize mean_change_f1"
        ),
        "min_mean_transition_accuracy": min_mean_transition_accuracy,
        "transition_floor_eligible_rows": transition_floor_eligible_rows,
        "best_parameters": best,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune Paper58 spatial-demand ratio parameters on external leave-one-area cases.")
    parser.add_argument("--change-gate-dir", action="append", type=Path, default=[])
    parser.add_argument("--change-gate-root", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ratio-quantiles", default="0.25,0.5,0.75")
    parser.add_argument("--ratio-multipliers", default="0.8,1.0,1.2,1.5,2.0")
    parser.add_argument("--min-fractions", default="0.0,0.03,0.05,0.08")
    parser.add_argument("--max-fractions", default="0.25,0.35,0.5")
    parser.add_argument("--target-neighborhood-weights", default="0.0,0.5,1.0")
    parser.add_argument("--source-neighborhood-penalties", default="0.0,0.25,0.5")
    parser.add_argument("--min-mean-transition-accuracy", type=float, default=None)
    args = parser.parse_args(argv)
    change_gate_dirs = discover_change_gate_dirs(args.change_gate_dir, args.change_gate_root)
    if not change_gate_dirs:
        raise ValueError("no change-gate dirs discovered")
    grid = ratio_parameter_grid(
        ratio_quantiles=_float_list(args.ratio_quantiles),
        ratio_multipliers=_float_list(args.ratio_multipliers),
        min_fractions=_float_list(args.min_fractions),
        max_fractions=_float_list(args.max_fractions),
        target_neighborhood_weights=_float_list(args.target_neighborhood_weights),
        source_neighborhood_penalties=_float_list(args.source_neighborhood_penalties),
    )
    manifest = run_ratio_tuning(
        change_gate_dirs,
        args.output_dir,
        grid,
        min_mean_transition_accuracy=args.min_mean_transition_accuracy,
    )
    print(
        "Paper58 ratio-demand tuning complete: "
        f"cases={manifest['n_cases']}, parameter_rows={manifest['n_parameter_rows']}, "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
