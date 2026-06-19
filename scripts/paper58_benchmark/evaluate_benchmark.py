from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.paper58_benchmark.baselines import (
    label_only_transition_prior,
    leave_one_region_temporal_prior,
    persistence_prediction,
    spatial_shuffle_prediction,
)
from scripts.paper58_benchmark.holdouts import VALID_CONTACT_STATUS
from scripts.paper58_benchmark.schema import DEFAULT_BENCHMARK_DIR, write_csv, write_json
from scripts.paper58_benchmark.statistics import gate_report, summarize_by_tier_and_stratum


METRIC_FIELDS = [
    "area",
    "start_year",
    "end_year",
    "tier",
    "stratum",
    "n_pixels",
    "true_change_pixels",
    "true_change_pct",
    "model_change_precision",
    "model_change_recall",
    "model_change_f1",
    "persistence_change_f1",
    "spatial_shuffle_change_f1",
    "transition_prior_change_f1",
    "temporal_prior_change_f1",
    "best_non_neural_change_f1",
    "primary_change_advantage",
    "spatial_change_advantage",
    "embedding_model_cosine",
    "embedding_persistence_cosine",
    "embedding_advantage",
]

REQUIRED_REGISTRY_FIELDS = [
    "area",
    "start_year",
    "end_year",
    "tier",
    "stratum",
    "bbox",
    "data_source",
    "development_contact_status",
    "contact_evidence",
    "expected_role",
    "label_start_path",
    "label_end_path",
    "prediction_path",
    "qc_status",
]


def _normalized_string(value: object) -> str:
    return str(value).strip().lower() if isinstance(value, str) else str(value)


def _path(value: object) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _load_array(path_value: object) -> np.ndarray:
    path = _path(path_value)
    if path is None:
        raise FileNotFoundError("Missing array path in benchmark registry row")
    return np.load(path)


def binary_change_metrics(true_change: np.ndarray, pred_change: np.ndarray) -> dict:
    true = true_change.astype(bool).ravel()
    pred = pred_change.astype(bool).ravel()
    tp = int(np.count_nonzero(true & pred))
    fp = int(np.count_nonzero(~true & pred))
    fn = int(np.count_nonzero(true & ~pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    a2 = np.asarray(a, dtype=float).reshape(-1, a.shape[-1])
    b2 = np.asarray(b, dtype=float).reshape(-1, b.shape[-1])
    numerator = np.sum(a2 * b2, axis=1)
    denom = np.linalg.norm(a2, axis=1) * np.linalg.norm(b2, axis=1)
    valid = denom > 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(numerator[valid] / denom[valid]))


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _transition_training_pairs(
    rows: list[dict],
    target: dict,
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


def _temporal_training_rows(rows: list[dict], target: dict) -> list[dict]:
    training: list[dict] = []
    for row in rows:
        if row.get("qc_status") != "include":
            continue
        start_path = _path(row.get("label_start_path"))
        end_path = _path(row.get("label_end_path"))
        if start_path is None or end_path is None or not start_path.exists() or not end_path.exists():
            continue
        training.append(
            {
                "area": row.get("area"),
                "start": np.load(start_path),
                "end": np.load(end_path),
            }
        )
    return training


def evaluate_registry_row(
    row: dict,
    transition_training_pairs: list[tuple[np.ndarray, np.ndarray]],
    temporal_training_rows: list[dict],
) -> dict:
    start = _load_array(row.get("label_start_path"))
    end = _load_array(row.get("label_end_path"))
    model_pred = _load_array(row.get("prediction_path"))
    if start.shape != end.shape or start.shape != model_pred.shape:
        raise ValueError(
            f"Shape mismatch for {row.get('area')} {row.get('start_year')}-{row.get('end_year')}: "
            f"start={start.shape}, end={end.shape}, pred={model_pred.shape}"
        )

    true_change = end != start
    model_change = model_pred != start
    persistence_pred = persistence_prediction(start)
    persistence_change = persistence_pred != start
    shuffle_pred = spatial_shuffle_prediction(model_pred)
    shuffle_change = shuffle_pred != start
    transition_prior_pred = label_only_transition_prior(start, transition_training_pairs)
    transition_prior_change = transition_prior_pred != start
    temporal_prior_pred = leave_one_region_temporal_prior(str(row.get("area")), start, temporal_training_rows)
    temporal_prior_change = temporal_prior_pred != start

    model_metrics = binary_change_metrics(true_change, model_change)
    persistence_metrics = binary_change_metrics(true_change, persistence_change)
    shuffle_metrics = binary_change_metrics(true_change, shuffle_change)
    transition_metrics = binary_change_metrics(true_change, transition_prior_change)
    temporal_metrics = binary_change_metrics(true_change, temporal_prior_change)
    best_non_neural = max(
        persistence_metrics["f1"],
        transition_metrics["f1"],
        temporal_metrics["f1"],
    )

    embedding_persistence_cosine = None
    embedding_model_cosine = None
    embedding_advantage = None
    emb_start_path = _path(row.get("embedding_start_path"))
    emb_end_path = _path(row.get("embedding_end_path"))
    if (
        emb_start_path is not None
        and emb_end_path is not None
        and emb_start_path.exists()
        and emb_end_path.exists()
    ):
        emb_start = np.load(emb_start_path)
        emb_end = np.load(emb_end_path)
        embedding_persistence_cosine = mean_cosine(emb_start, emb_end)

    return {
        "area": row.get("area"),
        "start_year": int(row.get("start_year")),
        "end_year": int(row.get("end_year")),
        "tier": row.get("tier"),
        "stratum": row.get("stratum"),
        "n_pixels": int(start.size),
        "true_change_pixels": int(np.count_nonzero(true_change)),
        "true_change_pct": float(np.count_nonzero(true_change) / start.size) if start.size else 0.0,
        "model_change_precision": model_metrics["precision"],
        "model_change_recall": model_metrics["recall"],
        "model_change_f1": model_metrics["f1"],
        "persistence_change_f1": persistence_metrics["f1"],
        "spatial_shuffle_change_f1": shuffle_metrics["f1"],
        "transition_prior_change_f1": transition_metrics["f1"],
        "temporal_prior_change_f1": temporal_metrics["f1"],
        "best_non_neural_change_f1": best_non_neural,
        "primary_change_advantage": model_metrics["f1"] - best_non_neural,
        "spatial_change_advantage": model_metrics["f1"] - shuffle_metrics["f1"],
        "embedding_model_cosine": embedding_model_cosine,
        "embedding_persistence_cosine": embedding_persistence_cosine,
        "embedding_advantage": embedding_advantage,
    }


def _read_registry(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark_registry.json must contain an object")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("benchmark_registry.json must contain a 'rows' list")

    validated_rows: list[dict] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"registry row {index} must be an object")
        for field in REQUIRED_REGISTRY_FIELDS:
            if field not in row:
                raise ValueError(f"registry row {index} missing required field: {field}")
        row["tier"] = _normalized_string(row.get("tier"))
        row["development_contact_status"] = _normalized_string(row.get("development_contact_status"))
        if isinstance(row.get("contact_evidence"), str):
            row["contact_evidence"] = row["contact_evidence"].strip()
        if row["development_contact_status"] not in VALID_CONTACT_STATUS:
            raise ValueError(
                f"registry row {index} invalid development_contact_status: {row['development_contact_status']}"
            )
        if row.get("tier") == "tier1" and row.get("development_contact_status") != "none":
            raise ValueError(f"tier1 row {index} requires development_contact_status='none'")
        if row.get("tier") == "tier1" and not str(row.get("contact_evidence", "")).strip():
            raise ValueError(f"tier1 row {index} requires non-empty contact_evidence")
        validated_rows.append(row)
    return validated_rows


def _write_summary_csv(path: Path, summary: dict) -> None:
    fields = ["group", "n", "mean", "median", "n_positive", "n_negative"]
    rows = [{"group": group, **values} for group, values in summary.items()]
    write_csv(path, rows, fields)


def _write_nested_summary_csv(path: Path, summary: dict[str, dict[str, dict]]) -> None:
    fields = ["tier", "stratum", "n", "mean", "median", "n_positive", "n_negative"]
    rows = [
        {"tier": tier, "stratum": stratum, **values}
        for tier, strata in summary.items()
        for stratum, values in strata.items()
    ]
    write_csv(path, rows, fields)


def _write_failures_csv(path: Path, registry_rows: list[dict]) -> None:
    fields = ["area", "start_year", "end_year", "tier", "qc_status", "excluded_reason"]
    failures = [
        {field: row.get(field) for field in fields}
        for row in registry_rows
        if row.get("qc_status") != "include"
    ]
    write_csv(path, failures, fields)


def evaluate_benchmark(
    registry_path: Path = DEFAULT_BENCHMARK_DIR / "benchmark_registry.json",
    output_dir: Path = DEFAULT_BENCHMARK_DIR,
    n_boot: int = 5000,
) -> dict:
    registry_rows = _read_registry(Path(registry_path))
    included_rows = [row for row in registry_rows if row.get("qc_status") == "include"]
    metric_rows = []
    for row in included_rows:
        start = _load_array(row.get("label_start_path"))
        metrics = evaluate_registry_row(
            row,
            transition_training_pairs=_transition_training_pairs(included_rows, row, start.shape),
            temporal_training_rows=_temporal_training_rows(included_rows, row),
        )
        metric_rows.append(metrics)

    primary_summary = summarize_by_tier_and_stratum(metric_rows, "primary_change_advantage")
    spatial_summary = summarize_by_tier_and_stratum(metric_rows, "spatial_change_advantage")
    stratum_summary_by_tier: dict[str, dict[str, dict]] = {}
    for tier in primary_summary["by_tier"]:
        tier_rows = [row for row in metric_rows if row.get("tier") == tier]
        stratum_summary_by_tier[tier] = summarize_by_tier_and_stratum(tier_rows, "primary_change_advantage")["by_stratum"]
    gates = gate_report(metric_rows, n_boot=n_boot)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_registry_rows": len(registry_rows),
        "n_evaluated": len(metric_rows),
        "n_failures": len(registry_rows) - len(included_rows),
        "model_change_f1_mean": float(np.mean([row["model_change_f1"] for row in metric_rows])) if metric_rows else None,
        "primary_change_advantage_mean": (
            float(np.mean([row["primary_change_advantage"] for row in metric_rows])) if metric_rows else None
        ),
    }
    result = {
        "summary": summary,
        "summary_by_tier": primary_summary["by_tier"],
        "summary_by_stratum": stratum_summary_by_tier,
        "spatial_summary_by_tier": spatial_summary["by_tier"],
        "gate_report": gates,
    }

    write_csv(output_dir / "benchmark_metrics_by_pair.csv", metric_rows, METRIC_FIELDS)
    write_json(output_dir / "benchmark_summary.json", result)
    _write_summary_csv(output_dir / "benchmark_summary_by_tier.csv", result["summary_by_tier"])
    _write_nested_summary_csv(output_dir / "benchmark_summary_by_stratum.csv", result["summary_by_stratum"])
    write_json(output_dir / "benchmark_gate_report.json", gates)
    _write_failures_csv(output_dir / "benchmark_failures.csv", registry_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Paper58 external benchmark.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_BENCHMARK_DIR / "benchmark_registry.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()
    result = evaluate_benchmark(args.registry, args.output_dir, args.n_boot)
    print(
        "Benchmark evaluation: "
        f"{result['summary']['n_evaluated']} evaluated pair(s), "
        f"gate status={result['gate_report']['status']}"
    )


if __name__ == "__main__":
    main()
