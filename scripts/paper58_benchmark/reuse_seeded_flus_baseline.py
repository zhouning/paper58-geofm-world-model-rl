from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper58_benchmark.las_metrics import method_metric_row
from scripts.paper58_benchmark.run_seeded_flus_replicates import (
    METRICS,
    _write_csv,
    summarize_seeded_deltas,
    summarize_seeded_method_metrics,
    write_seeded_report,
)
from scripts.paper58_benchmark.run_true_paper58_flus_same_grid_comparison import (
    METRIC_FIELDS,
    SameGridSample,
    discover_same_grid_samples,
)


SEEDED_FIELDS = ["seed", *METRIC_FIELDS]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _read_seeded_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(f):
            converted: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"seed", "start_year", "end_year", "n_pixels", "true_change_pixels", "pred_change_pixels"}:
                    converted[key] = int(float(value))
                elif key in {
                    "change_precision",
                    "change_recall",
                    "change_f1",
                    "fom",
                    "transition_accuracy",
                    "quantity_disagreement",
                    "allocation_disagreement",
                }:
                    converted[key] = float(value)
                else:
                    converted[key] = value
            rows.append(converted)
    return rows


def _load_challenger_prediction(sample: SameGridSample, challenger_prediction_dir: Path) -> np.ndarray:
    path = Path(challenger_prediction_dir) / f"{sample.area}_lulc_pred_{sample.start_year}_{sample.end_year}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing challenger prediction for {sample.area}: {path}")
    prediction = np.load(path).astype(np.int32, copy=False)
    if prediction.shape != sample.start.shape:
        raise ValueError(
            f"shape mismatch for {sample.area}: prediction={prediction.shape}, start={sample.start.shape}"
        )
    return prediction


def _challenger_metric_row(sample: SameGridSample, prediction: np.ndarray, challenger: str) -> dict[str, Any]:
    valid = np.asarray(sample.valid_mask, dtype=bool)
    row = method_metric_row(
        challenger,
        sample.area,
        "same_grid",
        sample.source,
        sample.start[valid],
        sample.end[valid],
        prediction[valid],
    )
    return {**row, "start_year": sample.start_year, "end_year": sample.end_year, "source": sample.source}


def reuse_seeded_flus_baseline(
    baseline_seeded_dir: Path,
    paper58_predictions_dir: Path,
    labels_dir: Path,
    challenger_prediction_dir: Path,
    output_dir: Path,
    challenger: str,
    baseline: str = "geosos_flus_console",
) -> dict[str, Any]:
    baseline_rows = [
        row
        for row in _read_seeded_rows(Path(baseline_seeded_dir) / "seeded_metrics_by_method.csv")
        if str(row.get("method")) == baseline
    ]
    if not baseline_rows:
        raise ValueError(f"no `{baseline}` rows found under {baseline_seeded_dir}")

    samples = {
        sample.area: sample
        for sample in discover_same_grid_samples(
            paper58_predictions_dir=Path(paper58_predictions_dir),
            labels_dir=Path(labels_dir),
        )
    }
    challenger_rows_by_area: dict[str, dict[str, Any]] = {}
    for area in sorted({str(row["area"]) for row in baseline_rows}):
        sample = samples.get(area)
        if sample is None:
            raise ValueError(f"baseline area `{area}` is missing from Paper58 samples")
        prediction = _load_challenger_prediction(sample, Path(challenger_prediction_dir))
        challenger_rows_by_area[area] = _challenger_metric_row(sample, prediction, challenger)

    all_rows: list[dict[str, Any]] = []
    seeds = sorted({int(row["seed"]) for row in baseline_rows})
    for row in baseline_rows:
        all_rows.append(row)
        challenger_row = dict(challenger_rows_by_area[str(row["area"])])
        challenger_row["seed"] = int(row["seed"])
        all_rows.append(challenger_row)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "seeded_metrics_by_method.csv", all_rows, SEEDED_FIELDS)
    method_summary = summarize_seeded_method_metrics(all_rows)
    method_fields = [
        "method",
        "area",
        "n",
        *[field for metric in METRICS for field in (f"mean_{metric}", f"std_{metric}", f"min_{metric}", f"max_{metric}")],
    ]
    _write_csv(output / "seeded_metric_summary_by_method.csv", method_summary, method_fields)
    delta_summary = summarize_seeded_deltas(all_rows, challenger=challenger, baseline=baseline)
    _write_csv(
        output / "seeded_delta_summary.csv",
        delta_summary,
        [
            "challenger",
            "baseline",
            "metric",
            "n",
            "mean_delta",
            "std_delta",
            "min_delta",
            "max_delta",
            "n_better",
            "better_rate",
            "higher_is_better",
        ],
    )
    write_seeded_report(output, seeds, method_summary, delta_summary, challenger, baseline)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_seeded_dir": Path(baseline_seeded_dir),
        "paper58_predictions_dir": Path(paper58_predictions_dir),
        "labels_dir": Path(labels_dir),
        "challenger_prediction_dir": Path(challenger_prediction_dir),
        "output_dir": output,
        "challenger": challenger,
        "baseline": baseline,
        "seeds": seeds,
        "n_rows": len(all_rows),
        "method_summary": method_summary,
        "delta_summary": delta_summary,
    }
    (output / "run_manifest.json").write_text(json.dumps(_json_ready(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reuse seeded GeoSOS-FLUS baseline rows for a deterministic challenger.")
    parser.add_argument("--baseline-seeded-dir", type=Path, required=True)
    parser.add_argument("--paper58-predictions-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--challenger-prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--baseline", default="geosos_flus_console")
    args = parser.parse_args(argv)
    result = reuse_seeded_flus_baseline(
        baseline_seeded_dir=args.baseline_seeded_dir,
        paper58_predictions_dir=args.paper58_predictions_dir,
        labels_dir=args.labels_dir,
        challenger_prediction_dir=args.challenger_prediction_dir,
        output_dir=args.output_dir,
        challenger=args.challenger,
        baseline=args.baseline,
    )
    print(
        "Reused seeded FLUS baseline report complete: "
        f"seeds={len(result['seeds'])}, rows={result['n_rows']}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
