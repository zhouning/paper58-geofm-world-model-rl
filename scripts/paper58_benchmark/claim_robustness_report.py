from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METRICS = ["change_f1", "fom", "transition_accuracy", "allocation_disagreement"]
LOWER_IS_BETTER = {"allocation_disagreement", "quantity_disagreement"}


@dataclass(frozen=True)
class GateThresholds:
    required_mean_metric_wins: int = 4
    required_seed_wins: int = 5
    required_seed_count: int = 5
    min_allocation_seed_wins: int = 3
    min_fom_paired_wins: int = 66


def _to_float(value: Any) -> float:
    return float(value)


def _to_int(value: Any) -> int:
    return int(float(value))


def metric_is_better(metric: str, challenger_value: float, baseline_value: float) -> bool:
    if metric in LOWER_IS_BETTER:
        return challenger_value < baseline_value
    return challenger_value > baseline_value


def mean_metric_advantages(
    metric_summary_rows: list[dict[str, Any]],
    *,
    challenger: str,
    baseline: str,
) -> list[dict[str, Any]]:
    by_method = {str(row["method"]): row for row in metric_summary_rows}
    if challenger not in by_method:
        raise ValueError(f"missing challenger method in metric summary: {challenger}")
    if baseline not in by_method:
        raise ValueError(f"missing baseline method in metric summary: {baseline}")
    challenger_row = by_method[challenger]
    baseline_row = by_method[baseline]
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        challenger_value = _to_float(challenger_row[f"mean_{metric}"])
        baseline_value = _to_float(baseline_row[f"mean_{metric}"])
        rows.append(
            {
                "metric": metric,
                "challenger": challenger,
                "baseline": baseline,
                "challenger_mean": challenger_value,
                "baseline_mean": baseline_value,
                "delta": challenger_value - baseline_value,
                "higher_is_better": metric not in LOWER_IS_BETTER,
                "better": metric_is_better(metric, challenger_value, baseline_value),
            }
        )
    return rows


def _summary_by_metric(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["metric"]): row for row in rows}


def _gate_row(gate: str, metric: str, observed: int, required: int, passed: bool, description: str) -> dict[str, Any]:
    return {
        "gate": gate,
        "metric": metric,
        "observed": int(observed),
        "required": int(required),
        "passed": bool(passed),
        "description": description,
    }


def evaluate_acceptance_gates(
    mean_advantages: list[dict[str, Any]],
    seeded_overall_summary: list[dict[str, Any]],
    paired_summary: list[dict[str, Any]],
    *,
    thresholds: GateThresholds | None = None,
) -> dict[str, Any]:
    limits = thresholds or GateThresholds()
    gates: list[dict[str, Any]] = []

    mean_wins = sum(1 for row in mean_advantages if bool(row["better"]))
    gates.append(
        _gate_row(
            "mean_4_of_4",
            "all",
            mean_wins,
            limits.required_mean_metric_wins,
            mean_wins >= limits.required_mean_metric_wins,
            "24-township mean metrics must remain 4/4 better than fixed geosos_flus_console.",
        )
    )

    seeded = _summary_by_metric(seeded_overall_summary)
    for metric in ("change_f1", "fom", "transition_accuracy"):
        row = seeded.get(metric, {})
        observed = _to_int(row.get("n_better", 0))
        required = limits.required_seed_wins
        gates.append(
            _gate_row(
                "seed_5_of_5",
                metric,
                observed,
                required,
                observed >= required and _to_int(row.get("n", 0)) >= limits.required_seed_count,
                f"{metric} must keep 5/5 seeded mean wins.",
            )
        )
    allocation_row = seeded.get("allocation_disagreement", {})
    allocation_observed = _to_int(allocation_row.get("n_better", 0))
    gates.append(
        _gate_row(
            "allocation_seed_wins",
            "allocation_disagreement",
            allocation_observed,
            limits.min_allocation_seed_wins,
            allocation_observed >= limits.min_allocation_seed_wins,
            "Allocation disagreement must keep at least 3/5 seeded mean wins.",
        )
    )

    paired = _summary_by_metric(paired_summary)
    fom_row = paired.get("fom", {})
    fom_observed = _to_int(fom_row.get("n_better", 0))
    gates.append(
        _gate_row(
            "paired_fom_wins",
            "fom",
            fom_observed,
            limits.min_fom_paired_wins,
            fom_observed >= limits.min_fom_paired_wins,
            "FoM area-by-seed paired wins must improve beyond the current 61/120 baseline.",
        )
    )

    return {
        "passed": all(bool(row["passed"]) for row in gates),
        "thresholds": asdict(limits),
        "gates": gates,
    }
