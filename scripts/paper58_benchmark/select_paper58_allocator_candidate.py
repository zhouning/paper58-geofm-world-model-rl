from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper58_benchmark.las_metrics import method_metric_row


PREDICTION_RE = re.compile(r"^(?P<area>.+)_lulc_pred_(?P<start_year>\d{4})_(?P<end_year>\d{4})\.npy$")
START_CLASS_FRACTION_RE = re.compile(r"^start_class_?(?P<class_value>\d+)_fraction$")
METRIC_FIELDS = ["change_f1", "fom", "transition_accuracy", "allocation_disagreement"]
HIGHER_IS_BETTER = {"change_f1": True, "fom": True, "transition_accuracy": True, "allocation_disagreement": False}
TARGET_METRIC_FIELDS = [
    "method",
    "area",
    "start_year",
    "end_year",
    "selected_candidate",
    "feature_candidate",
    "feature_field",
    "feature_operator",
    "feature_threshold",
    "feature_conditions",
    "feature_value",
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
CANDIDATE_AUDIT_FIELDS = [
    "method",
    "area",
    "start_year",
    "end_year",
    "predicted_change_fraction",
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
ORACLE_FIELDS = [
    "method",
    "area",
    "start_year",
    "end_year",
    "selected_candidate",
    "oracle_metric_order",
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


def _compact_json(value: Any) -> str:
    return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_ready(row.get(field, "")) for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _finite_float(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite float, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def _filter_parameter_index(rows: list[dict[str, str]], parameter_index: int | None, candidate: str) -> list[dict[str, str]]:
    if parameter_index is not None:
        filtered = [row for row in rows if str(row.get("parameter_index", "")) == str(int(parameter_index))]
        if not filtered:
            raise ValueError(f"{candidate} has no rows for parameter_index={parameter_index}")
        return filtered
    indices = {row.get("parameter_index", "") for row in rows if row.get("parameter_index", "") != ""}
    if len(indices) > 1:
        raise ValueError(
            f"{candidate} contains multiple parameter_index values; pass candidate_parameter_indices for this candidate"
        )
    return rows


def _load_candidate_case_rows(
    candidate: str,
    path: Path,
    *,
    parameter_index: int | None,
    feature_field: str,
    metrics: list[str],
) -> dict[str, dict[str, Any]]:
    rows = _filter_parameter_index(_read_csv(Path(path)), parameter_index, candidate)
    by_area: dict[str, dict[str, Any]] = {}
    for row in rows:
        area = str(row.get("area", "")).strip()
        if not area:
            raise ValueError(f"{candidate} contains a row without area")
        if area in by_area:
            raise ValueError(f"{candidate} contains duplicate rows for area {area}")
        loaded: dict[str, Any] = {
            "candidate": candidate,
            "area": area,
            "start_year": row.get("start_year", ""),
            "end_year": row.get("end_year", ""),
            "feature_field": feature_field,
            "feature_value": _finite_float(row.get(feature_field), f"{candidate}.{feature_field}"),
        }
        loaded[feature_field] = loaded["feature_value"]
        for metric in metrics:
            loaded[metric] = _finite_float(row.get(metric), f"{candidate}.{metric}")
        by_area[area] = loaded
    if not by_area:
        raise ValueError(f"{candidate} has no usable case metrics: {path}")
    return by_area


def _candidate_rows_by_name(
    candidate_case_metric_paths: dict[str, Path],
    candidate_parameter_indices: dict[str, int] | None,
    feature_field: str,
    metrics: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    if not candidate_case_metric_paths:
        raise ValueError("at least one candidate case metrics CSV is required")
    indices = candidate_parameter_indices or {}
    unknown = sorted(set(indices) - set(candidate_case_metric_paths))
    if unknown:
        raise ValueError(f"candidate_parameter_indices contains unknown candidate(s): {unknown}")
    return {
        candidate: _load_candidate_case_rows(
            candidate,
            Path(path),
            parameter_index=indices.get(candidate),
            feature_field=feature_field,
            metrics=metrics,
        )
        for candidate, path in sorted(candidate_case_metric_paths.items())
    }


def _load_candidate_case_rows_for_feature_fields(
    candidate: str,
    path: Path,
    *,
    parameter_index: int | None,
    feature_fields: list[str],
    metrics: list[str],
) -> dict[str, dict[str, Any]]:
    rows = _filter_parameter_index(_read_csv(Path(path)), parameter_index, candidate)
    by_area: dict[str, dict[str, Any]] = {}
    for row in rows:
        area = str(row.get("area", "")).strip()
        if not area:
            raise ValueError(f"{candidate} contains a row without area")
        if area in by_area:
            raise ValueError(f"{candidate} contains duplicate rows for area {area}")
        loaded: dict[str, Any] = {
            "candidate": candidate,
            "area": area,
            "start_year": row.get("start_year", ""),
            "end_year": row.get("end_year", ""),
        }
        for field in feature_fields:
            loaded[field] = _finite_float(row.get(field), f"{candidate}.{field}")
        loaded["feature_field"] = feature_fields[0]
        loaded["feature_value"] = loaded[feature_fields[0]]
        for metric in metrics:
            loaded[metric] = _finite_float(row.get(metric), f"{candidate}.{metric}")
        by_area[area] = loaded
    if not by_area:
        raise ValueError(f"{candidate} has no usable case metrics: {path}")
    return by_area


def _candidate_rows_by_name_for_feature_fields(
    candidate_case_metric_paths: dict[str, Path],
    candidate_parameter_indices: dict[str, int] | None,
    feature_fields: list[str],
    metrics: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    if not candidate_case_metric_paths:
        raise ValueError("at least one candidate case metrics CSV is required")
    if not feature_fields:
        raise ValueError("at least one feature field is required")
    indices = candidate_parameter_indices or {}
    unknown = sorted(set(indices) - set(candidate_case_metric_paths))
    if unknown:
        raise ValueError(f"candidate_parameter_indices contains unknown candidate(s): {unknown}")
    return {
        candidate: _load_candidate_case_rows_for_feature_fields(
            candidate,
            Path(path),
            parameter_index=indices.get(candidate),
            feature_fields=feature_fields,
            metrics=metrics,
        )
        for candidate, path in sorted(candidate_case_metric_paths.items())
    }


def _common_areas(by_candidate: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    area_sets = [set(rows) for rows in by_candidate.values()]
    common = sorted(set.intersection(*area_sets)) if area_sets else []
    if not common:
        raise ValueError("candidate case metrics have no common areas")
    mismatches = {
        candidate: sorted(set(rows).symmetric_difference(set(common)))
        for candidate, rows in by_candidate.items()
        if set(rows) != set(common)
    }
    if mismatches:
        details = "; ".join(f"{candidate}: {areas}" for candidate, areas in sorted(mismatches.items()))
        raise ValueError(f"candidate case metrics must contain the same areas; mismatches: {details}")
    return common


def _rule_uses_alternate(feature_value: float, operator: str, threshold: float) -> bool:
    if operator == "le":
        return feature_value <= threshold
    if operator == "ge":
        return feature_value >= threshold
    raise ValueError(f"unsupported operator: {operator}")


def _normalize_condition(condition: dict[str, Any]) -> dict[str, Any]:
    feature_candidate = str(condition.get("feature_candidate", "")).strip()
    feature_field = str(condition.get("feature_field", "")).strip()
    operator = str(condition.get("operator", "")).strip()
    if not feature_candidate:
        raise ValueError(f"condition is missing feature_candidate: {condition}")
    if not feature_field:
        raise ValueError(f"condition is missing feature_field: {condition}")
    if operator not in {"le", "ge"}:
        raise ValueError(f"unsupported condition operator: {operator}")
    return {
        "feature_candidate": feature_candidate,
        "feature_field": feature_field,
        "operator": operator,
        "threshold": _finite_float(condition.get("threshold"), f"{feature_candidate}.{feature_field}.threshold"),
    }


def _rule_conditions(rule: dict[str, Any]) -> list[dict[str, Any]]:
    if rule.get("conditions"):
        return [_normalize_condition(dict(condition)) for condition in rule["conditions"]]
    alternate = rule.get("alternate_candidate")
    threshold = rule.get("threshold")
    operator = str(rule.get("operator", "always"))
    if alternate is None or threshold is None or operator == "always":
        return []
    return [
        _normalize_condition(
            {
                "feature_candidate": str(rule.get("feature_candidate") or alternate),
                "feature_field": str(rule.get("feature_field", "predicted_target_change_fraction")),
                "operator": operator,
                "threshold": threshold,
            }
        )
    ]


def _condition_values_for_area(
    conditions: list[dict[str, Any]],
    by_candidate: dict[str, dict[str, dict[str, Any]]],
    area: str,
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for condition in conditions:
        feature_candidate = str(condition["feature_candidate"])
        feature_field = str(condition["feature_field"])
        if feature_candidate not in by_candidate:
            raise ValueError(f"condition references unknown candidate: {feature_candidate}")
        if feature_field not in by_candidate[feature_candidate][area]:
            raise ValueError(f"condition references unavailable feature: {feature_candidate}.{feature_field}")
        feature_value = float(by_candidate[feature_candidate][area][feature_field])
        values.append({**condition, "feature_value": feature_value})
    return values


def _conditions_use_alternate(condition_values: list[dict[str, Any]]) -> bool:
    return all(
        _rule_uses_alternate(float(item["feature_value"]), str(item["operator"]), float(item["threshold"]))
        for item in condition_values
    )


def _selected_external_rows(
    rule: dict[str, Any],
    by_candidate: dict[str, dict[str, dict[str, Any]]],
    areas: list[str],
    metrics: list[str],
    include_condition_details: bool = True,
) -> list[dict[str, Any]]:
    baseline = str(rule["baseline_candidate"])
    alternate = rule.get("alternate_candidate")
    conditions = _rule_conditions(rule)
    feature_candidate = str(rule.get("feature_candidate") or (conditions[0]["feature_candidate"] if conditions else alternate or baseline))
    operator = str(rule.get("operator", conditions[0]["operator"] if conditions else "always"))
    threshold = rule.get("threshold", conditions[0]["threshold"] if conditions else None)
    selected: list[dict[str, Any]] = []
    for area in areas:
        condition_values = _condition_values_for_area(conditions, by_candidate, area)
        feature_value = condition_values[0]["feature_value"] if condition_values else float(
            by_candidate[feature_candidate][area]["feature_value"]
        )
        use_alternate = alternate is not None and bool(condition_values) and _conditions_use_alternate(condition_values)
        selected_candidate = str(alternate) if use_alternate else baseline
        source = by_candidate[selected_candidate][area]
        row: dict[str, Any] = {
            "area": area,
            "start_year": source.get("start_year", ""),
            "end_year": source.get("end_year", ""),
            "selected_candidate": selected_candidate,
            "feature_candidate": feature_candidate,
            "feature_value": feature_value,
            "conditions_met": bool(use_alternate),
        }
        if include_condition_details:
            row["feature_conditions"] = _compact_json(conditions) if conditions else ""
            row["condition_values"] = _compact_json(condition_values) if condition_values else ""
        for metric in metrics:
            row[metric] = float(source[metric])
        selected.append(row)
    return selected


def _metric_mean(rows: list[dict[str, Any]], metric: str) -> float:
    values = [float(row[metric]) for row in rows]
    return float(np.mean(values)) if values else float("nan")


def _summarize_external_rule(
    rule: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    metrics: list[str],
    max_alternate_fraction: float | None,
    min_alternate_count: int = 0,
) -> dict[str, Any]:
    selected_alternate_count = int(
        sum(row["selected_candidate"] == rule.get("alternate_candidate") for row in selected_rows)
    )
    selected_alternate_fraction = selected_alternate_count / len(selected_rows) if selected_rows else 0.0
    summary: dict[str, Any] = {
        "baseline_candidate": rule["baseline_candidate"],
        "alternate_candidate": rule.get("alternate_candidate", ""),
        "feature_candidate": rule.get("feature_candidate", ""),
        "feature_field": rule.get("feature_field", ""),
        "operator": rule.get("operator", "always"),
        "threshold": rule.get("threshold", ""),
        "conditions": _compact_json(_rule_conditions(rule)),
        "condition_count": len(_rule_conditions(rule)),
        "min_alternate_count": int(min_alternate_count),
        "selected_alternate_count": selected_alternate_count,
        "selected_alternate_fraction": selected_alternate_fraction,
        "max_alternate_fraction": "" if max_alternate_fraction is None else float(max_alternate_fraction),
        "selected_candidate_counts": dict(sorted(Counter(row["selected_candidate"] for row in selected_rows).items())),
    }
    feasible = True
    if max_alternate_fraction is not None:
        feasible = selected_alternate_fraction <= float(max_alternate_fraction) + 1e-12
    if selected_alternate_count < int(min_alternate_count):
        feasible = False
    for metric in metrics:
        mean_value = _metric_mean(selected_rows, metric)
        baseline_mean = _metric_mean(baseline_rows, metric)
        summary[f"mean_{metric}"] = mean_value
        summary[f"baseline_mean_{metric}"] = baseline_mean
        summary[f"delta_{metric}"] = mean_value - baseline_mean
        if HIGHER_IS_BETTER[metric]:
            feasible = feasible and mean_value >= baseline_mean - 1e-12
        else:
            feasible = feasible and mean_value <= baseline_mean + 1e-12
    summary["feasible"] = bool(feasible)
    return summary


def _rank_rule_summary(summary: dict[str, Any]) -> tuple[float, float, float, float, int]:
    return (
        float(summary["mean_fom"]),
        float(summary["mean_change_f1"]),
        float(summary["mean_transition_accuracy"]),
        -float(summary["mean_allocation_disagreement"]),
        int(summary.get("selected_alternate_count", 0)),
    )


def _rank_conjunctive_rule_summary(summary: dict[str, Any]) -> tuple[float, float, float, float, int, int]:
    return (
        float(summary["mean_fom"]),
        float(summary["mean_change_f1"]),
        float(summary["mean_transition_accuracy"]),
        -float(summary["mean_allocation_disagreement"]),
        int(summary.get("selected_alternate_count", 0)),
        -int(summary.get("condition_count", 0)),
    )


def _rank_rule_summary_by_metric_order(summary: dict[str, Any], metric_order: list[str]) -> tuple[float, ...]:
    values: list[float] = []
    for metric in metric_order:
        value = float(summary[f"mean_{metric}"])
        values.append(value if HIGHER_IS_BETTER[metric] else -value)
    values.append(float(summary.get("selected_alternate_count", 0)))
    values.append(-float(summary.get("condition_count", 0)))
    return tuple(values)


def _target_unlabeled_alternate_count(
    rule: dict[str, Any],
    candidate_prediction_dirs: dict[str, Path],
    labels_dir: Path,
) -> dict[str, int]:
    baseline = str(rule["baseline_candidate"])
    alternate = rule.get("alternate_candidate")
    conditions = _rule_conditions(rule)
    samples = _discover_prediction_samples(Path(candidate_prediction_dirs[baseline]))
    labels_root = Path(labels_dir)
    selected_count = 0
    usable_count = 0
    if alternate is None or not conditions:
        return {"target_unlabeled_alternate_count": 0, "target_unlabeled_area_count": len(samples)}
    for area, start_year, end_year, _ in samples:
        start_path = labels_root / f"{area}_lulc_{start_year}.npy"
        if not start_path.exists():
            continue
        start = np.load(start_path).astype(np.int32, copy=False)
        condition_values = _target_condition_values(
            conditions,
            candidate_prediction_dirs,
            start,
            area,
            start_year,
            end_year,
        )
        selected_count += int(_conditions_use_alternate(condition_values))
        usable_count += 1
    return {
        "target_unlabeled_alternate_count": int(selected_count),
        "target_unlabeled_area_count": int(usable_count),
    }


def _target_unlabeled_feature_rows(
    candidate_prediction_dirs: dict[str, Path],
    labels_dir: Path,
    *,
    baseline_candidate: str,
    feature_fields: list[str],
) -> dict[str, dict[tuple[str, str], float]]:
    samples = _discover_prediction_samples(Path(candidate_prediction_dirs[baseline_candidate]))
    labels_root = Path(labels_dir)
    rows: dict[str, dict[tuple[str, str], float]] = {}
    for area, start_year, end_year, _ in samples:
        start_path = labels_root / f"{area}_lulc_{start_year}.npy"
        if not start_path.exists():
            continue
        start = np.load(start_path).astype(np.int32, copy=False)
        feature_values: dict[tuple[str, str], float] = {}
        for candidate, prediction_dir in sorted(candidate_prediction_dirs.items()):
            prediction = _load_prediction(Path(prediction_dir), area, start_year, end_year, start.shape)
            for feature_field in feature_fields:
                feature_values[(str(candidate), str(feature_field))] = _target_feature_value(
                    feature_field,
                    start,
                    prediction,
                )
        rows[area] = feature_values
    return rows


def _target_unlabeled_alternate_count_from_features(
    rule: dict[str, Any],
    target_feature_rows: dict[str, dict[tuple[str, str], float]],
) -> dict[str, int]:
    alternate = rule.get("alternate_candidate")
    conditions = _rule_conditions(rule)
    if alternate is None or not conditions:
        return {
            "target_unlabeled_alternate_count": 0,
            "target_unlabeled_area_count": len(target_feature_rows),
        }
    selected_count = 0
    for feature_values in target_feature_rows.values():
        condition_values: list[dict[str, Any]] = []
        for condition in conditions:
            key = (str(condition["feature_candidate"]), str(condition["feature_field"]))
            if key not in feature_values:
                raise ValueError(f"missing target unlabeled feature value for {key}")
            condition_values.append({**condition, "feature_value": float(feature_values[key])})
        selected_count += int(_conditions_use_alternate(condition_values))
    return {
        "target_unlabeled_alternate_count": int(selected_count),
        "target_unlabeled_area_count": len(target_feature_rows),
    }


def select_allocator_candidate_rule(
    candidate_case_metric_paths: dict[str, Path],
    output_dir: Path,
    *,
    baseline_candidate: str,
    candidate_parameter_indices: dict[str, int] | None = None,
    feature_field: str = "predicted_target_change_fraction",
    metrics: list[str] | None = None,
    max_alternate_fraction: float | None = None,
) -> dict[str, Any]:
    selected_metrics = list(metrics or METRIC_FIELDS)
    unknown_metrics = sorted(set(selected_metrics) - set(HIGHER_IS_BETTER))
    if unknown_metrics:
        raise ValueError(f"unsupported metrics: {unknown_metrics}")
    if max_alternate_fraction is not None and not 0.0 <= float(max_alternate_fraction) <= 1.0:
        raise ValueError(f"max_alternate_fraction must be in [0, 1]: {max_alternate_fraction}")
    by_candidate = _candidate_rows_by_name(
        candidate_case_metric_paths,
        candidate_parameter_indices,
        feature_field,
        selected_metrics,
    )
    if baseline_candidate not in by_candidate:
        raise ValueError(f"baseline_candidate is not in candidate paths: {baseline_candidate}")
    areas = _common_areas(by_candidate)
    baseline_rows = [by_candidate[baseline_candidate][area] for area in areas]

    rules: list[dict[str, Any]] = [
        {
            "baseline_candidate": baseline_candidate,
            "alternate_candidate": None,
            "feature_candidate": baseline_candidate,
            "feature_field": feature_field,
            "operator": "always",
            "threshold": None,
        }
    ]
    for alternate in sorted(candidate for candidate in by_candidate if candidate != baseline_candidate):
        thresholds = sorted({float(by_candidate[alternate][area]["feature_value"]) for area in areas})
        for operator in ("le", "ge"):
            for threshold in thresholds:
                rules.append(
                    {
                        "baseline_candidate": baseline_candidate,
                        "alternate_candidate": alternate,
                        "feature_candidate": alternate,
                        "feature_field": feature_field,
                        "operator": operator,
                        "threshold": threshold,
                    }
                )

    rule_rows: list[dict[str, Any]] = []
    selected_by_rule: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = []
    for rule in rules:
        rows = _selected_external_rows(rule, by_candidate, areas, selected_metrics)
        summary = _summarize_external_rule(rule, rows, baseline_rows, selected_metrics, max_alternate_fraction)
        selected_by_rule.append((rule, rows, summary))
        rule_rows.append(summary)

    feasible = [item for item in selected_by_rule if item[2]["feasible"]]
    if not feasible:
        feasible = [selected_by_rule[0]]
    best_rule, best_rows, best_summary = sorted(
        feasible,
        key=lambda item: (_rank_rule_summary(item[2]), str(item[0].get("alternate_candidate") or "")),
        reverse=True,
    )[0]

    output = Path(output_dir)
    rule_fields = [
        "baseline_candidate",
        "alternate_candidate",
        "feature_candidate",
        "feature_field",
        "operator",
        "threshold",
        "selected_alternate_count",
        "selected_alternate_fraction",
        "max_alternate_fraction",
        "selected_candidate_counts",
        *[f"mean_{metric}" for metric in selected_metrics],
        *[f"baseline_mean_{metric}" for metric in selected_metrics],
        *[f"delta_{metric}" for metric in selected_metrics],
        "feasible",
    ]
    _write_csv(output / "external_selector_rules.csv", rule_rows, rule_fields)
    _write_csv(
        output / "external_selected_by_area.csv",
        best_rows,
        [
            "area",
            "start_year",
            "end_year",
            "selected_candidate",
            "feature_candidate",
            "feature_value",
            *selected_metrics,
        ],
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paper58_allocator_candidate_threshold_selector",
        "selection_data": "external_leave_one_area_case_metrics_only",
        "baseline_candidate": baseline_candidate,
        "candidate_case_metric_paths": {name: Path(path) for name, path in candidate_case_metric_paths.items()},
        "candidate_parameter_indices": candidate_parameter_indices or {},
        "feature_field": feature_field,
        "max_alternate_fraction": max_alternate_fraction,
        "metrics": selected_metrics,
        "n_external_areas": len(areas),
        "rule": best_rule,
        "external_summary": best_summary,
    }
    _write_json(output / "selector_manifest.json", manifest)
    return {"rule": best_rule, "external_summary": best_summary, "external_selected_rows": best_rows, "manifest": manifest}


def _candidate_threshold_conditions(
    by_candidate: dict[str, dict[str, dict[str, Any]]],
    areas: list[str],
    feature_fields: list[str],
) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    for feature_candidate in sorted(by_candidate):
        for feature_field in feature_fields:
            thresholds = sorted({float(by_candidate[feature_candidate][area][feature_field]) for area in areas})
            for operator in ("le", "ge"):
                for threshold in thresholds:
                    conditions.append(
                        {
                            "feature_candidate": feature_candidate,
                            "feature_field": feature_field,
                            "operator": operator,
                            "threshold": threshold,
                        }
                    )
    return conditions


def select_allocator_conjunctive_candidate_rule(
    candidate_case_metric_paths: dict[str, Path],
    output_dir: Path,
    *,
    baseline_candidate: str,
    candidate_parameter_indices: dict[str, int] | None = None,
    feature_fields: list[str] | None = None,
    metrics: list[str] | None = None,
    max_alternate_fraction: float | None = None,
    min_alternate_count: int = 1,
    max_conditions: int = 2,
    rank_metric_order: list[str] | None = None,
    target_prediction_dirs: dict[str, Path] | None = None,
    target_labels_dir: Path | None = None,
    target_min_alternate_count: int | None = None,
    target_max_alternate_count: int | None = None,
) -> dict[str, Any]:
    selected_metrics = list(metrics or METRIC_FIELDS)
    unknown_metrics = sorted(set(selected_metrics) - set(HIGHER_IS_BETTER))
    if unknown_metrics:
        raise ValueError(f"unsupported metrics: {unknown_metrics}")
    selected_rank_metric_order = list(
        rank_metric_order or ["fom", "change_f1", "transition_accuracy", "allocation_disagreement"]
    )
    unknown_rank_metrics = sorted(set(selected_rank_metric_order) - set(selected_metrics))
    if unknown_rank_metrics:
        raise ValueError(f"rank_metric_order contains unavailable metric(s): {unknown_rank_metrics}")
    selected_feature_fields = list(feature_fields or ["predicted_target_change_fraction"])
    if len(set(selected_feature_fields)) != len(selected_feature_fields):
        raise ValueError(f"feature_fields must be unique: {selected_feature_fields}")
    if max_alternate_fraction is not None and not 0.0 <= float(max_alternate_fraction) <= 1.0:
        raise ValueError(f"max_alternate_fraction must be in [0, 1]: {max_alternate_fraction}")
    support = int(min_alternate_count)
    if support < 0:
        raise ValueError(f"min_alternate_count must be non-negative: {min_alternate_count}")
    condition_limit = int(max_conditions)
    if condition_limit < 1:
        raise ValueError(f"max_conditions must be positive: {max_conditions}")
    target_constraints_enabled = target_min_alternate_count is not None or target_max_alternate_count is not None
    if target_constraints_enabled and (target_prediction_dirs is None or target_labels_dir is None):
        raise ValueError(
            "target_prediction_dirs and target_labels_dir are required when target alternate-count constraints are set"
        )
    target_min_count = None if target_min_alternate_count is None else int(target_min_alternate_count)
    target_max_count = None if target_max_alternate_count is None else int(target_max_alternate_count)
    if target_min_count is not None and target_min_count < 0:
        raise ValueError(f"target_min_alternate_count must be non-negative: {target_min_alternate_count}")
    if target_max_count is not None and target_max_count < 0:
        raise ValueError(f"target_max_alternate_count must be non-negative: {target_max_alternate_count}")
    if target_min_count is not None and target_max_count is not None and target_min_count > target_max_count:
        raise ValueError(
            "target_min_alternate_count cannot exceed target_max_alternate_count: "
            f"{target_min_count} > {target_max_count}"
        )

    by_candidate = _candidate_rows_by_name_for_feature_fields(
        candidate_case_metric_paths,
        candidate_parameter_indices,
        selected_feature_fields,
        selected_metrics,
    )
    if baseline_candidate not in by_candidate:
        raise ValueError(f"baseline_candidate is not in candidate paths: {baseline_candidate}")
    areas = _common_areas(by_candidate)
    baseline_rows = [by_candidate[baseline_candidate][area] for area in areas]
    primitive_conditions = _candidate_threshold_conditions(by_candidate, areas, selected_feature_fields)

    rules: list[dict[str, Any]] = [
        {
            "baseline_candidate": baseline_candidate,
            "alternate_candidate": None,
            "conditions": [],
        }
    ]
    for alternate in sorted(candidate for candidate in by_candidate if candidate != baseline_candidate):
        for count in range(1, condition_limit + 1):
            for condition_group in itertools.combinations(primitive_conditions, count):
                rules.append(
                    {
                        "baseline_candidate": baseline_candidate,
                        "alternate_candidate": alternate,
                        "conditions": [dict(condition) for condition in condition_group],
                    }
                )

    target_feature_rows: dict[str, dict[tuple[str, str], float]] | None = None
    if target_constraints_enabled and target_prediction_dirs is not None and target_labels_dir is not None:
        target_feature_rows = _target_unlabeled_feature_rows(
            {name: Path(path) for name, path in target_prediction_dirs.items()},
            Path(target_labels_dir),
            baseline_candidate=baseline_candidate,
            feature_fields=selected_feature_fields,
        )

    rule_rows: list[dict[str, Any]] = []
    selected_by_rule: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = []
    for rule in rules:
        min_count = 0 if rule.get("alternate_candidate") is None else support
        rows = _selected_external_rows(
            rule,
            by_candidate,
            areas,
            selected_metrics,
            include_condition_details=False,
        )
        summary = _summarize_external_rule(
            rule,
            rows,
            baseline_rows,
            selected_metrics,
            max_alternate_fraction,
            min_alternate_count=min_count,
        )
        if target_constraints_enabled and target_feature_rows is not None:
            target_counts = _target_unlabeled_alternate_count_from_features(
                rule,
                target_feature_rows,
            )
            summary.update(target_counts)
            target_selected = int(target_counts["target_unlabeled_alternate_count"])
            if target_min_count is not None and target_selected < target_min_count:
                summary["feasible"] = False
            if target_max_count is not None and target_selected > target_max_count:
                summary["feasible"] = False
        selected_by_rule.append((rule, rows, summary))
        rule_rows.append(summary)

    feasible = [item for item in selected_by_rule if item[2]["feasible"]]
    if not feasible:
        feasible = [selected_by_rule[0]]
    best_rule, _, best_summary = sorted(
        feasible,
        key=lambda item: (
            _rank_rule_summary_by_metric_order(item[2], selected_rank_metric_order),
            str(item[0].get("alternate_candidate") or ""),
        ),
        reverse=True,
    )[0]
    best_rows = _selected_external_rows(best_rule, by_candidate, areas, selected_metrics)

    output = Path(output_dir)
    rule_fields = [
        "baseline_candidate",
        "alternate_candidate",
        "conditions",
        "condition_count",
        "min_alternate_count",
        "selected_alternate_count",
        "selected_alternate_fraction",
        "max_alternate_fraction",
        "selected_candidate_counts",
        "target_unlabeled_alternate_count",
        "target_unlabeled_area_count",
        *[f"mean_{metric}" for metric in selected_metrics],
        *[f"baseline_mean_{metric}" for metric in selected_metrics],
        *[f"delta_{metric}" for metric in selected_metrics],
        "feasible",
    ]
    _write_csv(output / "external_conjunctive_selector_rules.csv", rule_rows, rule_fields)
    _write_csv(
        output / "external_conjunctive_selected_by_area.csv",
        best_rows,
        [
            "area",
            "start_year",
            "end_year",
            "selected_candidate",
            "feature_candidate",
            "feature_value",
            "feature_conditions",
            "condition_values",
            "conditions_met",
            *selected_metrics,
        ],
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paper58_allocator_candidate_conjunctive_threshold_selector",
        "selection_data": "external_leave_one_area_case_metrics_only",
        "baseline_candidate": baseline_candidate,
        "candidate_case_metric_paths": {name: Path(path) for name, path in candidate_case_metric_paths.items()},
        "candidate_parameter_indices": candidate_parameter_indices or {},
        "feature_fields": selected_feature_fields,
        "max_alternate_fraction": max_alternate_fraction,
        "min_alternate_count": support,
        "max_conditions": condition_limit,
        "rank_metric_order": selected_rank_metric_order,
        "target_unlabeled_constraints": {
            "target_min_alternate_count": target_min_count,
            "target_max_alternate_count": target_max_count,
        },
        "target_prediction_dirs_for_selection": (
            {} if target_prediction_dirs is None else {name: Path(path) for name, path in target_prediction_dirs.items()}
        ),
        "target_labels_dir_for_selection": None if target_labels_dir is None else Path(target_labels_dir),
        "metrics": selected_metrics,
        "n_external_areas": len(areas),
        "rule": best_rule,
        "external_summary": best_summary,
    }
    _write_json(output / "conjunctive_selector_manifest.json", manifest)
    return {"rule": best_rule, "external_summary": best_summary, "external_selected_rows": best_rows, "manifest": manifest}


def _prediction_root(path: Path) -> Path:
    root = Path(path)
    return root / "predictions" if (root / "predictions").is_dir() else root


def _parse_prediction_path(path: Path) -> tuple[str, int, int] | None:
    match = PREDICTION_RE.match(path.name)
    if match is None:
        return None
    return match.group("area"), int(match.group("start_year")), int(match.group("end_year"))


def _discover_prediction_samples(prediction_dir: Path) -> list[tuple[str, int, int, Path]]:
    samples: list[tuple[str, int, int, Path]] = []
    for path in sorted(_prediction_root(prediction_dir).glob("*_lulc_pred_*_*.npy")):
        parsed = _parse_prediction_path(path)
        if parsed is not None:
            area, start_year, end_year = parsed
            samples.append((area, start_year, end_year, path))
    if not samples:
        raise ValueError(f"no target prediction files discovered under {prediction_dir}")
    return samples


def _prediction_path(prediction_dir: Path, area: str, start_year: int, end_year: int) -> Path:
    filename = f"{area}_lulc_pred_{start_year}_{end_year}.npy"
    root = Path(prediction_dir)
    candidates = [root / filename, root / "predictions" / filename]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[-1])


def _load_prediction(prediction_dir: Path, area: str, start_year: int, end_year: int, shape: tuple[int, ...]) -> np.ndarray:
    path = _prediction_path(prediction_dir, area, start_year, end_year)
    if not path.exists():
        raise FileNotFoundError(f"prediction not found for {area} {start_year}-{end_year}: {path}")
    prediction = np.load(path).astype(np.int32, copy=False)
    if prediction.shape != shape:
        raise ValueError(f"prediction shape mismatch for {area}: {prediction.shape} vs {shape}")
    return prediction


def _target_feature_value(feature_field: str, start_map: np.ndarray, prediction_map: np.ndarray) -> float:
    field = str(feature_field)
    valid = (np.asarray(start_map) != 0) & (np.asarray(prediction_map) != 0)
    total = int(np.count_nonzero(valid))
    if total <= 0:
        return 0.0
    if field in {"predicted_target_change_fraction", "predicted_change_fraction"}:
        changed = valid & (np.asarray(prediction_map) != np.asarray(start_map))
        return float(np.count_nonzero(changed) / total)
    if field == "start_entropy_norm":
        start_values = np.asarray(start_map)[valid].ravel()
        _, counts = np.unique(start_values, return_counts=True)
        probabilities = counts.astype(np.float64) / float(total)
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
        return entropy / max(float(np.log(max(len(counts), 2))), 1e-12)
    if field == "start_unique_class_fraction":
        start_values = np.asarray(start_map)[valid].ravel()
        return float(min(1.0, len(np.unique(start_values)) / 10.0))
    match = START_CLASS_FRACTION_RE.match(field)
    if match is not None:
        class_value = int(match.group("class_value"))
        return float(np.count_nonzero(valid & (np.asarray(start_map) == class_value)) / total)
    raise ValueError(f"unsupported target feature_field: {feature_field}")


def _target_condition_values(
    conditions: list[dict[str, Any]],
    candidate_prediction_dirs: dict[str, Path],
    start: np.ndarray,
    area: str,
    start_year: int,
    end_year: int,
) -> list[dict[str, Any]]:
    predictions: dict[str, np.ndarray] = {}
    values: list[dict[str, Any]] = []
    for condition in conditions:
        feature_candidate = str(condition["feature_candidate"])
        if feature_candidate not in predictions:
            predictions[feature_candidate] = _load_prediction(
                candidate_prediction_dirs[feature_candidate],
                area,
                start_year,
                end_year,
                start.shape,
            )
        feature_value = _target_feature_value(str(condition["feature_field"]), start, predictions[feature_candidate])
        values.append({**condition, "feature_value": feature_value})
    return values


def _metric_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in rows})
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        summary_rows.append(
            {
                "method": method,
                "n": len(subset),
                "mean_change_f1": _metric_mean(subset, "change_f1"),
                "mean_fom": _metric_mean(subset, "fom"),
                "mean_transition_accuracy": _metric_mean(subset, "transition_accuracy"),
                "mean_allocation_disagreement": _metric_mean(subset, "allocation_disagreement"),
            }
        )
    return summary_rows


def _read_reference_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(path):
        loaded = dict(row)
        for metric in METRIC_FIELDS:
            if metric in loaded and loaded[metric] not in (None, ""):
                loaded[metric] = _finite_float(loaded[metric], metric)
        rows.append(loaded)
    return rows


def _reference_comparison_rows(
    selected_rows: list[dict[str, Any]],
    reference_metrics_path: Path,
    reference_methods: list[str] | None,
    selector_method: str,
) -> list[dict[str, Any]]:
    reference_rows = _read_reference_rows(reference_metrics_path)
    methods = reference_methods or sorted({str(row.get("method")) for row in reference_rows if row.get("method")})
    selected_by_area = {str(row["area"]): row for row in selected_rows if str(row.get("method")) == selector_method}
    rows: list[dict[str, Any]] = []
    for method in methods:
        refs = {str(row["area"]): row for row in reference_rows if str(row.get("method")) == method}
        common = sorted(set(selected_by_area) & set(refs))
        if not common:
            continue
        summary: dict[str, Any] = {"reference_method": method, "n": len(common)}
        for metric in METRIC_FIELDS:
            selector_values = np.array([float(selected_by_area[area][metric]) for area in common], dtype=np.float64)
            reference_values = np.array([float(refs[area][metric]) for area in common], dtype=np.float64)
            delta = selector_values - reference_values
            wins = selector_values > reference_values if HIGHER_IS_BETTER[metric] else selector_values < reference_values
            summary[f"selector_mean_{metric}"] = float(np.mean(selector_values))
            summary[f"reference_mean_{metric}"] = float(np.mean(reference_values))
            summary[f"mean_delta_{metric}"] = float(np.mean(delta))
            summary[f"paired_wins_{metric}"] = int(np.count_nonzero(wins))
        rows.append(summary)
    return rows


def apply_allocator_selector_rule(
    rule: dict[str, Any],
    candidate_prediction_dirs: dict[str, Path],
    *,
    labels_dir: Path,
    output_dir: Path,
    method_name: str = "paper58_allocator_selector",
    reference_metrics_path: Path | None = None,
    reference_methods: list[str] | None = None,
) -> dict[str, Any]:
    baseline = str(rule["baseline_candidate"])
    alternate = rule.get("alternate_candidate")
    conditions = _rule_conditions(rule)
    feature_candidate = str(rule.get("feature_candidate") or (conditions[0]["feature_candidate"] if conditions else alternate or baseline))
    feature_field = str(rule.get("feature_field", conditions[0]["feature_field"] if conditions else "predicted_target_change_fraction"))
    operator = str(rule.get("operator", conditions[0]["operator"] if conditions else "always"))
    threshold = rule.get("threshold", conditions[0]["threshold"] if conditions else None)
    required_candidates = {
        baseline,
        feature_candidate,
        *(str(condition["feature_candidate"]) for condition in conditions),
        *(candidate for candidate in [alternate] if candidate is not None),
    }
    for candidate in required_candidates:
        if str(candidate) not in candidate_prediction_dirs:
            raise ValueError(f"missing target prediction dir for candidate {candidate}")

    output = Path(output_dir)
    prediction_output = output / "predictions"
    prediction_output.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, Any]] = []
    labels_root = Path(labels_dir)
    samples = _discover_prediction_samples(Path(candidate_prediction_dirs[baseline]))
    for area, start_year, end_year, _ in samples:
        start_path = labels_root / f"{area}_lulc_{start_year}.npy"
        end_path = labels_root / f"{area}_lulc_{end_year}.npy"
        if not start_path.exists() or not end_path.exists():
            continue
        start = np.load(start_path).astype(np.int32, copy=False)
        end = np.load(end_path).astype(np.int32, copy=False)
        if start.shape != end.shape:
            raise ValueError(f"label shape mismatch for {area}: {start.shape} vs {end.shape}")
        condition_values = _target_condition_values(
            conditions,
            candidate_prediction_dirs,
            start,
            area,
            start_year,
            end_year,
        )
        if condition_values:
            feature_value = condition_values[0]["feature_value"]
            use_alternate = alternate is not None and _conditions_use_alternate(condition_values)
        else:
            feature_prediction = _load_prediction(candidate_prediction_dirs[feature_candidate], area, start_year, end_year, start.shape)
            feature_value = _target_feature_value(feature_field, start, feature_prediction)
            use_alternate = False
            if alternate is not None and threshold is not None and operator != "always":
                use_alternate = _rule_uses_alternate(feature_value, operator, float(threshold))
        selected_candidate = str(alternate) if use_alternate else baseline
        selected_prediction = _load_prediction(candidate_prediction_dirs[selected_candidate], area, start_year, end_year, start.shape)
        np.save(prediction_output / f"{area}_lulc_pred_{start_year}_{end_year}.npy", selected_prediction)
        valid = (start != 0) & (end != 0) & (selected_prediction != 0)
        metric = method_metric_row(
            method=method_name,
            area=area,
            tier="same_grid",
            stratum="paper58_allocator_candidate_selector",
            start_map=start[valid],
            true_map=end[valid],
            pred_map=selected_prediction[valid],
        )
        selected_rows.append(
            {
                **metric,
                "start_year": start_year,
                "end_year": end_year,
                "selected_candidate": selected_candidate,
                "feature_candidate": feature_candidate,
                "feature_field": feature_field,
                "feature_operator": operator,
                "feature_threshold": threshold,
                "feature_conditions": _compact_json(conditions) if conditions else "",
                "feature_value": feature_value,
            }
        )

    target_summary = _metric_summary(selected_rows)
    _write_csv(output / "target_selection_by_area.csv", selected_rows, TARGET_METRIC_FIELDS)
    _write_csv(
        output / "target_metric_summary_by_method.csv",
        target_summary,
        [
            "method",
            "n",
            "mean_change_f1",
            "mean_fom",
            "mean_transition_accuracy",
            "mean_allocation_disagreement",
        ],
    )
    reference_comparisons: list[dict[str, Any]] = []
    if reference_metrics_path is not None:
        reference_comparisons = _reference_comparison_rows(
            selected_rows,
            Path(reference_metrics_path),
            reference_methods,
            method_name,
        )
        _write_csv(
            output / "reference_comparison_summary.csv",
            reference_comparisons,
            [
                "reference_method",
                "n",
                *[f"selector_mean_{metric}" for metric in METRIC_FIELDS],
                *[f"reference_mean_{metric}" for metric in METRIC_FIELDS],
                *[f"mean_delta_{metric}" for metric in METRIC_FIELDS],
                *[f"paired_wins_{metric}" for metric in METRIC_FIELDS],
            ],
        )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method_name,
        "rule": rule,
        "candidate_prediction_dirs": {name: Path(path) for name, path in candidate_prediction_dirs.items()},
        "labels_dir": Path(labels_dir),
        "n_target_areas": len(selected_rows),
        "selected_candidate_counts": dict(sorted(Counter(row["selected_candidate"] for row in selected_rows).items())),
        "target_summary": target_summary,
        "reference_comparisons": reference_comparisons,
    }
    _write_json(output / "target_selector_manifest.json", manifest)
    return {
        "selected_rows": selected_rows,
        "target_summary": target_summary,
        "reference_comparisons": reference_comparisons,
        "manifest": manifest,
    }


def _oracle_sort_key(row: dict[str, Any], metric_order: list[str]) -> tuple[float, ...]:
    values: list[float] = []
    for metric in metric_order:
        value = float(row[metric])
        values.append(value if HIGHER_IS_BETTER[metric] else -value)
    return tuple(values)


def audit_target_candidate_bank(
    candidate_prediction_dirs: dict[str, Path],
    *,
    labels_dir: Path,
    output_dir: Path,
    oracle_metric_order: list[str] | None = None,
    oracle_method_name: str = "paper58_candidate_bank_oracle",
) -> dict[str, Any]:
    if not candidate_prediction_dirs:
        raise ValueError("at least one target prediction candidate is required")
    metric_order = list(oracle_metric_order or ["fom", "change_f1", "transition_accuracy", "allocation_disagreement"])
    unknown_metrics = sorted(set(metric_order) - set(HIGHER_IS_BETTER))
    if unknown_metrics:
        raise ValueError(f"unsupported oracle metric(s): {unknown_metrics}")
    first_candidate = sorted(candidate_prediction_dirs)[0]
    samples = _discover_prediction_samples(Path(candidate_prediction_dirs[first_candidate]))
    labels_root = Path(labels_dir)
    candidate_rows: list[dict[str, Any]] = []
    for area, start_year, end_year, _ in samples:
        start_path = labels_root / f"{area}_lulc_{start_year}.npy"
        end_path = labels_root / f"{area}_lulc_{end_year}.npy"
        if not start_path.exists() or not end_path.exists():
            continue
        start = np.load(start_path).astype(np.int32, copy=False)
        end = np.load(end_path).astype(np.int32, copy=False)
        if start.shape != end.shape:
            raise ValueError(f"label shape mismatch for {area}: {start.shape} vs {end.shape}")
        for candidate, prediction_dir in sorted(candidate_prediction_dirs.items()):
            prediction = _load_prediction(Path(prediction_dir), area, start_year, end_year, start.shape)
            valid = (start != 0) & (end != 0) & (prediction != 0)
            metric = method_metric_row(
                method=candidate,
                area=area,
                tier="same_grid",
                stratum="paper58_candidate_bank_audit",
                start_map=start[valid],
                true_map=end[valid],
                pred_map=prediction[valid],
            )
            candidate_rows.append(
                {
                    **metric,
                    "start_year": start_year,
                    "end_year": end_year,
                    "predicted_change_fraction": _target_feature_value(
                        "predicted_change_fraction",
                        start,
                        prediction,
                    ),
                }
            )

    oracle_rows: list[dict[str, Any]] = []
    for area in sorted({str(row["area"]) for row in candidate_rows}):
        area_rows = [row for row in candidate_rows if str(row["area"]) == area]
        best = max(area_rows, key=lambda row: _oracle_sort_key(row, metric_order))
        oracle_rows.append(
            {
                **best,
                "method": oracle_method_name,
                "selected_candidate": best["method"],
                "oracle_metric_order": ",".join(metric_order),
            }
        )

    candidate_summary = _metric_summary(candidate_rows)
    oracle_summary = _metric_summary(oracle_rows)
    output = Path(output_dir)
    _write_csv(output / "target_candidate_metrics_by_area.csv", candidate_rows, CANDIDATE_AUDIT_FIELDS)
    _write_csv(
        output / "target_candidate_metric_summary_by_method.csv",
        candidate_summary,
        [
            "method",
            "n",
            "mean_change_f1",
            "mean_fom",
            "mean_transition_accuracy",
            "mean_allocation_disagreement",
        ],
    )
    _write_csv(output / "target_oracle_by_area.csv", oracle_rows, ORACLE_FIELDS)
    _write_csv(
        output / "target_oracle_summary_by_method.csv",
        oracle_summary,
        [
            "method",
            "n",
            "mean_change_f1",
            "mean_fom",
            "mean_transition_accuracy",
            "mean_allocation_disagreement",
        ],
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paper58_target_candidate_bank_oracle_audit",
        "candidate_prediction_dirs": {name: Path(path) for name, path in candidate_prediction_dirs.items()},
        "labels_dir": Path(labels_dir),
        "oracle_metric_order": metric_order,
        "n_candidate_rows": len(candidate_rows),
        "n_target_areas": len(oracle_rows),
        "oracle_candidate_counts": dict(sorted(Counter(row["selected_candidate"] for row in oracle_rows).items())),
        "candidate_summary": candidate_summary,
        "oracle_summary": oracle_summary,
    }
    _write_json(output / "target_candidate_bank_audit_manifest.json", manifest)
    return {
        "candidate_rows": candidate_rows,
        "candidate_summary": candidate_summary,
        "oracle_rows": oracle_rows,
        "oracle_summary": oracle_summary,
        "manifest": manifest,
    }


def build_consensus_candidate(
    candidate_prediction_dirs: dict[str, Path],
    *,
    labels_dir: Path,
    output_dir: Path,
    anchor_candidate: str,
    min_support: int,
) -> dict[str, Any]:
    if anchor_candidate not in candidate_prediction_dirs:
        raise ValueError(f"anchor_candidate is not in candidate_prediction_dirs: {anchor_candidate}")
    support = int(min_support)
    if support < 1:
        raise ValueError(f"min_support must be positive: {min_support}")
    if support > len(candidate_prediction_dirs):
        raise ValueError(
            f"min_support={support} cannot exceed candidate count={len(candidate_prediction_dirs)}"
        )
    output = Path(output_dir)
    prediction_output = output / "predictions"
    prediction_output.mkdir(parents=True, exist_ok=True)
    labels_root = Path(labels_dir)
    samples = _discover_prediction_samples(Path(candidate_prediction_dirs[anchor_candidate]))
    case_rows: list[dict[str, Any]] = []
    for area, start_year, end_year, _ in samples:
        start_path = labels_root / f"{area}_lulc_{start_year}.npy"
        if not start_path.exists():
            continue
        start = np.load(start_path).astype(np.int32, copy=False)
        predictions = {
            candidate: _load_prediction(Path(path), area, start_year, end_year, start.shape)
            for candidate, path in sorted(candidate_prediction_dirs.items())
        }
        changed_stack = np.stack([prediction != start for prediction in predictions.values()], axis=0)
        support_count = np.sum(changed_stack, axis=0)
        anchor_prediction = predictions[anchor_candidate]
        selected = (anchor_prediction != start) & (support_count >= support)
        consensus = start.copy()
        consensus[selected] = anchor_prediction[selected]
        np.save(prediction_output / f"{area}_lulc_pred_{start_year}_{end_year}.npy", consensus)
        case_rows.append(
            {
                "area": area,
                "start_year": start_year,
                "end_year": end_year,
                "anchor_candidate": anchor_candidate,
                "min_support": support,
                "candidate_count": len(candidate_prediction_dirs),
                "anchor_change_pixels": int(np.count_nonzero(anchor_prediction != start)),
                "selected_change_pixels": int(np.count_nonzero(selected)),
            }
        )
    _write_csv(
        output / "consensus_candidate_cases.csv",
        case_rows,
        [
            "area",
            "start_year",
            "end_year",
            "anchor_candidate",
            "min_support",
            "candidate_count",
            "anchor_change_pixels",
            "selected_change_pixels",
        ],
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paper58_consensus_candidate",
        "candidate_prediction_dirs": {name: Path(path) for name, path in candidate_prediction_dirs.items()},
        "labels_dir": Path(labels_dir),
        "anchor_candidate": anchor_candidate,
        "min_support": support,
        "n_cases": len(case_rows),
        "cases": case_rows,
    }
    _write_json(output / "consensus_candidate_manifest.json", manifest)
    return {"case_rows": case_rows, "manifest": manifest}


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must be NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("name cannot be empty")
    return name, Path(path)


def _parse_named_int(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must be NAME=INTEGER")
    name, raw = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("name cannot be empty")
    return name, int(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select a Paper58 allocator candidate using external LOO case metrics, then optionally apply it."
    )
    parser.add_argument("--candidate", action="append", type=_parse_named_path, required=True)
    parser.add_argument("--candidate-parameter-index", action="append", type=_parse_named_int, default=[])
    parser.add_argument("--baseline-candidate", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-field", default="predicted_target_change_fraction")
    parser.add_argument("--selector-mode", choices=["threshold", "conjunctive"], default="threshold")
    parser.add_argument("--conjunctive-feature-field", action="append", default=None)
    parser.add_argument("--min-alternate-count", type=int, default=1)
    parser.add_argument("--max-conditions", type=int, default=2)
    parser.add_argument(
        "--target-min-alternate-count",
        type=int,
        default=None,
        help="Optional unlabeled target lower bound on areas selected for the alternate candidate.",
    )
    parser.add_argument(
        "--target-max-alternate-count",
        type=int,
        default=None,
        help="Optional unlabeled target upper bound on areas selected for the alternate candidate.",
    )
    parser.add_argument(
        "--rank-metric-order",
        default=None,
        help=(
            "Comma-separated metric priority for conjunctive selector ranking. "
            "Lower-is-better metrics such as allocation_disagreement are handled automatically."
        ),
    )
    parser.add_argument(
        "--max-alternate-fraction",
        type=float,
        default=None,
        help="Optional upper bound on the fraction of external areas switched away from the baseline candidate.",
    )
    parser.add_argument("--target-prediction", action="append", type=_parse_named_path, default=[])
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument("--method-name", default="paper58_allocator_selector")
    parser.add_argument("--reference-metrics", type=Path, default=None)
    parser.add_argument("--reference-method", action="append", default=None)
    parser.add_argument(
        "--write-target-candidate-audit",
        action="store_true",
        help="When target predictions are provided, also write per-candidate target metrics and oracle headroom.",
    )
    parser.add_argument(
        "--oracle-metric-order",
        default="fom,change_f1,transition_accuracy,allocation_disagreement",
        help="Comma-separated target oracle ranking order for diagnostic audits.",
    )
    args = parser.parse_args(argv)

    candidates = dict(args.candidate)
    parameter_indices = dict(args.candidate_parameter_index)
    if args.selector_mode == "conjunctive":
        use_target_constraints = args.target_min_alternate_count is not None or args.target_max_alternate_count is not None
        if use_target_constraints and (not args.target_prediction or args.labels_dir is None):
            raise ValueError(
                "--target-prediction and --labels-dir are required when target alternate-count constraints are set"
            )
        result = select_allocator_conjunctive_candidate_rule(
            candidates,
            args.output_dir,
            baseline_candidate=args.baseline_candidate,
            candidate_parameter_indices=parameter_indices,
            feature_fields=args.conjunctive_feature_field or [args.feature_field],
            max_alternate_fraction=args.max_alternate_fraction,
            min_alternate_count=args.min_alternate_count,
            max_conditions=args.max_conditions,
            rank_metric_order=(
                None
                if args.rank_metric_order is None
                else [item.strip() for item in args.rank_metric_order.split(",") if item.strip()]
            ),
            target_prediction_dirs=(dict(args.target_prediction) if use_target_constraints else None),
            target_labels_dir=(args.labels_dir if use_target_constraints else None),
            target_min_alternate_count=args.target_min_alternate_count,
            target_max_alternate_count=args.target_max_alternate_count,
        )
    else:
        result = select_allocator_candidate_rule(
            candidates,
            args.output_dir,
            baseline_candidate=args.baseline_candidate,
            candidate_parameter_indices=parameter_indices,
            feature_field=args.feature_field,
            max_alternate_fraction=args.max_alternate_fraction,
        )
    if args.target_prediction:
        if args.labels_dir is None:
            raise ValueError("--labels-dir is required when --target-prediction is provided")
        apply_allocator_selector_rule(
            result["rule"],
            dict(args.target_prediction),
            labels_dir=args.labels_dir,
            output_dir=args.output_dir,
            method_name=args.method_name,
            reference_metrics_path=args.reference_metrics,
            reference_methods=args.reference_method,
        )
        if args.write_target_candidate_audit:
            audit_target_candidate_bank(
                dict(args.target_prediction),
                labels_dir=args.labels_dir,
                output_dir=args.output_dir,
                oracle_metric_order=[item.strip() for item in args.oracle_metric_order.split(",") if item.strip()],
            )
    print(
        "Paper58 allocator selector complete: "
        f"baseline={args.baseline_candidate}, rule={result['rule']}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
