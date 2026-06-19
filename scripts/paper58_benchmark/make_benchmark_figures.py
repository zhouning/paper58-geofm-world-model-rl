from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.paper58_benchmark.schema import DEFAULT_BENCHMARK_DIR


REQUIRED_METRIC_COLUMNS = (
    "area",
    "start_year",
    "end_year",
    "tier",
    "primary_change_advantage",
    "spatial_change_advantage",
    "model_change_f1",
    "best_non_neural_change_f1",
)

NUMERIC_METRIC_COLUMNS = (
    "primary_change_advantage",
    "spatial_change_advantage",
    "model_change_f1",
    "best_non_neural_change_f1",
)


def _read_metrics(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or ())
        missing = [column for column in REQUIRED_METRIC_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"{path.name} missing required columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path.name} row {row_number} has extra columns: {row[None]!r}")
            parsed = dict(row)
            for key in NUMERIC_METRIC_COLUMNS:
                try:
                    parsed_value = float(parsed[key])
                    if not math.isfinite(parsed_value):
                        raise ValueError
                    parsed[key] = parsed_value
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path.name} row {row_number} has invalid {key}: {parsed[key]!r}"
                    ) from exc
            rows.append(parsed)
    return rows


def _read_gate_report(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def load_benchmark_outputs(results_dir: Path = DEFAULT_BENCHMARK_DIR) -> dict:
    results_dir = Path(results_dir)
    metrics_path = results_dir / "benchmark_metrics_by_pair.csv"
    gate_path = results_dir / "benchmark_gate_report.json"
    missing = [str(path) for path in (metrics_path, gate_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing benchmark result files: " + ", ".join(missing))
    return {
        "metrics": _read_metrics(metrics_path),
        "gate": _read_gate_report(gate_path),
    }


def make_benchmark_figures(
    results_dir: Path = DEFAULT_BENCHMARK_DIR,
    figure_dir: Path | None = None,
) -> None:
    data = load_benchmark_outputs(results_dir)
    figure_dir = Path(figure_dir) if figure_dir is not None else Path(results_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in data["metrics"] if row["tier"] == "tier1"]
    if not rows:
        rows = data["metrics"]
    labels = [f"{row['area']}\n{row['start_year']}-{row['end_year']}" for row in rows]
    x = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(max(5.0, len(rows) * 0.75), 3.2))
    primary = [row["primary_change_advantage"] for row in rows]
    spatial = [row["spatial_change_advantage"] for row in rows]
    width = 0.35
    ax.bar([i - width / 2 for i in x], primary, width=width, label="Model - best non-neural")
    ax.bar([i + width / 2 for i in x], spatial, width=width, label="Model - spatial shuffle")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Change-F1 advantage")
    ax.set_title(f"Paper58 benchmark gate status: {data['gate'].get('status', 'unknown')}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_paper58_benchmark_gate.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_paper58_benchmark_gate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make Paper58 benchmark figures.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--figure-dir", type=Path, default=None)
    args = parser.parse_args()
    make_benchmark_figures(args.results_dir, args.figure_dir)
    print(f"Wrote Paper58 benchmark figures from {args.results_dir}")


if __name__ == "__main__":
    main()
