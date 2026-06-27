from __future__ import annotations

import argparse
import csv
import math
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
LOWER_IS_BETTER = {"quantity_disagreement", "allocation_disagreement"}
PAIR_FIELDS = ["area", "start_year", "end_year"]
CONTEXT_FIELDS = ["tier", "stratum"]


def _read_rows(path: Path) -> list[dict[str, str]]:
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


def _pair_key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(str(row.get(field, "")) for field in PAIR_FIELDS)  # type: ignore[return-value]


def _advantage(challenger: float, baseline: float, metric: str) -> float:
    if metric in LOWER_IS_BETTER:
        return baseline - challenger
    return challenger - baseline


def _mean_by_method(rows: list[dict[str, str]], metrics: list[str]) -> dict[str, dict[str, float]]:
    methods = sorted({row.get("method", "") for row in rows if row.get("method")})
    result: dict[str, dict[str, float]] = {}
    for method in methods:
        method_rows = [row for row in rows if row.get("method") == method]
        result[method] = {}
        for metric in metrics:
            values = [
                number
                for number in (_finite_float(row.get(metric)) for row in method_rows)
                if number is not None
            ]
            if values:
                result[method][metric] = float(mean(values))
    return result


def _paired_rows(
    rows: list[dict[str, str]],
    metrics: list[str],
    challenger_method: str,
    baseline_method: str,
) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        by_pair.setdefault(_pair_key(row), {})[str(row.get("method"))] = row

    paired: list[dict[str, Any]] = []
    for key, method_rows in sorted(by_pair.items()):
        challenger = method_rows.get(challenger_method)
        baseline = method_rows.get(baseline_method)
        if challenger is None or baseline is None:
            continue
        record: dict[str, Any] = {
            "area": key[0],
            "start_year": key[1],
            "end_year": key[2],
        }
        for field in CONTEXT_FIELDS:
            record[field] = challenger.get(field, baseline.get(field, ""))
        for metric in metrics:
            challenger_value = _finite_float(challenger.get(metric))
            baseline_value = _finite_float(baseline.get(metric))
            if challenger_value is None or baseline_value is None:
                continue
            record[f"{metric}_{challenger_method}"] = challenger_value
            record[f"{metric}_{baseline_method}"] = baseline_value
            record[f"{metric}_advantage"] = _advantage(challenger_value, baseline_value, metric)
        paired.append(record)
    return paired


def _advantage_summary(paired_rows: list[dict[str, Any]], metric: str, n_boot: int, seed: int) -> dict[str, Any]:
    key = f"{metric}_advantage"
    values = [float(row[key]) for row in paired_rows if key in row]
    ci_rows = [{"area": row["area"], key: float(row[key])} for row in paired_rows if key in row]
    sign = paired_sign_test(values)
    ci = clustered_bootstrap_ci(ci_rows, key, n_boot=n_boot, seed=seed)
    return {
        "direction": "baseline_minus_challenger" if metric in LOWER_IS_BETTER else "challenger_minus_baseline",
        "n": len(values),
        "mean_advantage": float(mean(values)) if values else None,
        "n_positive": int(sum(value > 0 for value in values)),
        "n_negative": int(sum(value < 0 for value in values)),
        "bootstrap_ci": ci,
        "sign_test": sign,
    }


def summarize_las_comparison(
    metrics_path: Path,
    output_dir: Path,
    challenger_method: str = "paper58_las",
    baseline_method: str = "flus",
    metrics: list[str] | None = None,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    selected_metrics = list(metrics or DEFAULT_METRICS)
    rows = _read_rows(Path(metrics_path))
    paired = _paired_rows(rows, selected_metrics, challenger_method, baseline_method)
    if not paired:
        raise ValueError(f"no matched method pairs for {challenger_method} vs {baseline_method}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    by_area_fields = [*PAIR_FIELDS, *CONTEXT_FIELDS]
    for metric in selected_metrics:
        by_area_fields.extend(
            [
                f"{metric}_{challenger_method}",
                f"{metric}_{baseline_method}",
                f"{metric}_advantage",
            ]
        )
    write_csv(output / "las_comparison_by_area.csv", paired, by_area_fields)

    summary = {
        "challenger_method": challenger_method,
        "baseline_method": baseline_method,
        "n_pairs": len(paired),
        "metrics": selected_metrics,
        "method_means": _mean_by_method(rows, selected_metrics),
        "advantages": {
            metric: _advantage_summary(paired, metric, n_boot=n_boot, seed=seed + index)
            for index, metric in enumerate(selected_metrics)
        },
    }
    write_json(output / "las_comparison_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize matched Paper58-LAS vs FLUS comparison metrics.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--challenger-method", default="paper58_las")
    parser.add_argument("--baseline-method", default="flus")
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()
    summary = summarize_las_comparison(
        metrics_path=args.metrics,
        output_dir=args.output_dir,
        challenger_method=args.challenger_method,
        baseline_method=args.baseline_method,
        n_boot=args.n_boot,
    )
    f1 = summary["advantages"].get("change_f1", {})
    fom = summary["advantages"].get("fom", {})
    print(
        "LAS comparison: "
        f"{summary['n_pairs']} matched pair(s), "
        f"change_f1_advantage={f1.get('mean_advantage')}, "
        f"fom_advantage={fom.get('mean_advantage')}"
    )


if __name__ == "__main__":
    main()
