from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_DIR = ROOT / "data" / "independent_change_labels" / "labels"
DEFAULT_PREDICTIONS_DIR = ROOT / "data" / "independent_change_labels" / "predicted"
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "rse_submission_paper58" / "revision_results"

LABEL_RE = re.compile(r"^(?P<area>.+)_lulc_(?P<year>\d{4})\.npy$")
PREDICTION_RE = re.compile(r"^(?P<area>.+)_lulc_pred_(?P<start_year>\d{4})_(?P<end_year>\d{4})\.npy$")

BY_AREA_FIELDS = [
    "area",
    "start_year",
    "end_year",
    "n_pixels",
    "true_change_pixels",
    "true_change_pct",
    "model_change_precision",
    "model_change_recall",
    "model_change_f1",
    "persistence_change_precision",
    "persistence_change_recall",
    "persistence_change_f1",
    "model_end_accuracy",
    "persistence_end_accuracy",
    "model_changed_pixel_accuracy",
    "persistence_changed_pixel_accuracy",
    "shuffled_model_change_f1",
    "shuffled_model_end_accuracy",
    "shuffled_model_changed_pixel_accuracy",
    "transition_prior_change_precision",
    "transition_prior_change_recall",
    "transition_prior_change_f1",
    "transition_prior_end_accuracy",
    "transition_prior_changed_pixel_accuracy",
    "transition_prior_transition_exact_match",
    "transition_prior_area_bias_mae",
    "model_transition_exact_match",
    "persistence_transition_exact_match",
    "model_area_bias_mae",
    "persistence_area_bias_mae",
]

SHUFFLE_SEED = 20260617
TRANSITION_PRIOR_SEED = 20260618


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true = y_true.astype(bool).ravel()
    pred = y_pred.astype(bool).ravel()
    tp = int(np.count_nonzero(true & pred))
    fp = int(np.count_nonzero(~true & pred))
    fn = int(np.count_nonzero(true & ~pred))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _area_bias_mae(true_end: np.ndarray, pred_end: np.ndarray) -> float:
    classes = np.union1d(true_end.ravel(), pred_end.ravel())
    n_pixels = true_end.size
    if n_pixels == 0:
        return 0.0
    errors = []
    for cls in classes:
        true_frac = np.count_nonzero(true_end == cls) / n_pixels
        pred_frac = np.count_nonzero(pred_end == cls) / n_pixels
        errors.append(abs(pred_frac - true_frac))
    return float(mean(errors)) if errors else 0.0


def _shuffle_prediction(pred_end: np.ndarray, area: str, start_year: int, end_year: int) -> np.ndarray:
    _ = (area, start_year, end_year)
    rng = np.random.default_rng(SHUFFLE_SEED)
    return rng.permutation(pred_end.ravel()).reshape(pred_end.shape)


def _transition_prior_prediction(
    target_start: np.ndarray,
    training_pairs: list[tuple[np.ndarray, np.ndarray]],
    seed: int = TRANSITION_PRIOR_SEED,
) -> np.ndarray:
    """Predict end labels from leave-out empirical class-transition frequencies."""
    target = np.asarray(target_start)
    predicted = target.copy()
    if not training_pairs:
        return predicted

    transitions: dict[int, dict[int, int]] = {}
    for train_start, train_end in training_pairs:
        if train_start.shape != train_end.shape:
            continue
        for start_cls, end_cls in zip(train_start.ravel(), train_end.ravel()):
            start_key = int(start_cls)
            end_key = int(end_cls)
            transitions.setdefault(start_key, {})
            transitions[start_key][end_key] = transitions[start_key].get(end_key, 0) + 1

    rng = np.random.default_rng(seed)
    flat_start = target.ravel()
    flat_pred = predicted.ravel()
    for start_cls in sorted(np.unique(flat_start)):
        class_counts = transitions.get(int(start_cls))
        if not class_counts:
            continue
        indices = np.flatnonzero(flat_start == start_cls)
        if indices.size == 0:
            continue
        shuffled_indices = rng.permutation(indices)
        end_classes = sorted(class_counts)
        counts = np.array([class_counts[end_cls] for end_cls in end_classes], dtype=float)
        fractions = counts / counts.sum()
        allocation = np.floor(fractions * indices.size).astype(int)
        remainder = int(indices.size - allocation.sum())
        if remainder > 0:
            residual_order = np.argsort(-(fractions * indices.size - allocation))
            for position in residual_order[:remainder]:
                allocation[position] += 1
        cursor = 0
        for end_cls, n_assign in zip(end_classes, allocation):
            if n_assign <= 0:
                continue
            selected = shuffled_indices[cursor : cursor + int(n_assign)]
            flat_pred[selected] = end_cls
            cursor += int(n_assign)
    return flat_pred.reshape(target.shape)


def _transition_counts(true_start: np.ndarray, true_end: np.ndarray, pred_end: np.ndarray) -> list[dict]:
    true_change = true_end != true_start
    rows = []
    for start_cls, end_cls, pred_cls in zip(
        true_start[true_change].ravel(),
        true_end[true_change].ravel(),
        pred_end[true_change].ravel(),
    ):
        rows.append(
            {
                "true_transition": f"{int(start_cls)}->{int(end_cls)}",
                "predicted_transition": f"{int(start_cls)}->{int(pred_cls)}",
            }
        )
    grouped: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["true_transition"], row["predicted_transition"])
        grouped[key] = grouped.get(key, 0) + 1
    return [
        {"true_transition": true_t, "predicted_transition": pred_t, "n": count}
        for (true_t, pred_t), count in sorted(grouped.items())
    ]


def evaluate_pair(
    area: str,
    start_year: int,
    end_year: int,
    true_start: np.ndarray,
    true_end: np.ndarray,
    pred_end: np.ndarray,
    transition_prior_end: np.ndarray | None = None,
) -> dict:
    if true_start.shape != true_end.shape or true_start.shape != pred_end.shape:
        raise ValueError(
            f"Shape mismatch for {area} {start_year}-{end_year}: "
            f"start={true_start.shape}, end={true_end.shape}, pred={pred_end.shape}"
        )
    if transition_prior_end is not None and transition_prior_end.shape != true_start.shape:
        raise ValueError(
            f"Shape mismatch for transition prior {area} {start_year}-{end_year}: "
            f"start={true_start.shape}, prior={transition_prior_end.shape}"
        )

    true_change = true_end != true_start
    model_change = pred_end != true_start
    persistence_end = true_start
    persistence_change = persistence_end != true_start
    shuffled_end = _shuffle_prediction(pred_end, area, start_year, end_year)
    shuffled_change = shuffled_end != true_start
    if transition_prior_end is None:
        transition_prior_end = persistence_end
    transition_prior_change = transition_prior_end != true_start

    model_change_metrics = _binary_metrics(true_change, model_change)
    persistence_change_metrics = _binary_metrics(true_change, persistence_change)
    shuffled_change_metrics = _binary_metrics(true_change, shuffled_change)
    transition_prior_change_metrics = _binary_metrics(true_change, transition_prior_change)
    changed_mask = true_change
    true_changed_end = true_end[changed_mask]
    pred_changed_end = pred_end[changed_mask]
    persistence_changed_end = persistence_end[changed_mask]
    shuffled_changed_end = shuffled_end[changed_mask]
    transition_prior_changed_end = transition_prior_end[changed_mask]

    model_changed_accuracy = _accuracy(true_changed_end, pred_changed_end)
    persistence_changed_accuracy = _accuracy(true_changed_end, persistence_changed_end)
    shuffled_changed_accuracy = _accuracy(true_changed_end, shuffled_changed_end)
    transition_prior_changed_accuracy = _accuracy(true_changed_end, transition_prior_changed_end)

    return {
        "area": area,
        "start_year": int(start_year),
        "end_year": int(end_year),
        "n_pixels": int(true_start.size),
        "true_change_pixels": int(np.count_nonzero(true_change)),
        "true_change_pct": float(np.count_nonzero(true_change) / true_start.size) if true_start.size else 0.0,
        "model_change_precision": model_change_metrics["precision"],
        "model_change_recall": model_change_metrics["recall"],
        "model_change_f1": model_change_metrics["f1"],
        "persistence_change_precision": persistence_change_metrics["precision"],
        "persistence_change_recall": persistence_change_metrics["recall"],
        "persistence_change_f1": persistence_change_metrics["f1"],
        "model_end_accuracy": _accuracy(true_end, pred_end),
        "persistence_end_accuracy": _accuracy(true_end, persistence_end),
        "model_changed_pixel_accuracy": model_changed_accuracy,
        "persistence_changed_pixel_accuracy": persistence_changed_accuracy,
        "shuffled_model_change_f1": shuffled_change_metrics["f1"],
        "shuffled_model_end_accuracy": _accuracy(true_end, shuffled_end),
        "shuffled_model_changed_pixel_accuracy": shuffled_changed_accuracy,
        "transition_prior_change_precision": transition_prior_change_metrics["precision"],
        "transition_prior_change_recall": transition_prior_change_metrics["recall"],
        "transition_prior_change_f1": transition_prior_change_metrics["f1"],
        "transition_prior_end_accuracy": _accuracy(true_end, transition_prior_end),
        "transition_prior_changed_pixel_accuracy": transition_prior_changed_accuracy,
        "transition_prior_transition_exact_match": transition_prior_changed_accuracy,
        "transition_prior_area_bias_mae": _area_bias_mae(true_end, transition_prior_end),
        "model_transition_exact_match": model_changed_accuracy,
        "persistence_transition_exact_match": persistence_changed_accuracy,
        "model_area_bias_mae": _area_bias_mae(true_end, pred_end),
        "persistence_area_bias_mae": _area_bias_mae(true_end, persistence_end),
        "transition_counts": _transition_counts(true_start, true_end, pred_end),
    }


def _discover_label_files(labels_dir: Path) -> dict[str, dict[int, Path]]:
    discovered: dict[str, dict[int, Path]] = {}
    if not labels_dir.exists():
        return discovered
    for path in sorted(labels_dir.glob("*_lulc_*.npy")):
        match = LABEL_RE.match(path.name)
        if not match:
            continue
        area = match.group("area")
        year = int(match.group("year"))
        discovered.setdefault(area, {})[year] = path
    return discovered


def _discover_prediction_files(predictions_dir: Path) -> dict[str, dict[tuple[int, int], Path]]:
    discovered: dict[str, dict[tuple[int, int], Path]] = {}
    if not predictions_dir.exists():
        return discovered
    for path in sorted(predictions_dir.glob("*_lulc_pred_*_*.npy")):
        match = PREDICTION_RE.match(path.name)
        if not match:
            continue
        area = match.group("area")
        start_year = int(match.group("start_year"))
        end_year = int(match.group("end_year"))
        discovered.setdefault(area, {})[(start_year, end_year)] = path
    return discovered


def _load_transition_prior_training_pairs(
    label_files: dict[str, dict[int, Path]],
    target_area: str,
    target_start_year: int,
    target_end_year: int,
    target_shape: tuple[int, ...],
) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for area, years_to_path in sorted(label_files.items()):
        years = sorted(years_to_path)
        for start_year, end_year in zip(years, years[1:]):
            if area == target_area and start_year == target_start_year and end_year == target_end_year:
                continue
            start = np.load(years_to_path[start_year])
            end = np.load(years_to_path[end_year])
            if start.shape != end.shape or start.shape != target_shape:
                continue
            pairs.append((start, end))
    return pairs


def _candidate_prediction_paths(predictions_dir: Path, area: str, start_year: int, end_year: int) -> list[Path]:
    return [
        predictions_dir / f"{area}_lulc_pred_{start_year}_{end_year}.npy",
        predictions_dir / f"{area}_lulc_{end_year}_pred.npy",
        predictions_dir / f"{area}_lulc_{end_year}.npy",
    ]


def _find_prediction_path(predictions_dir: Path, area: str, start_year: int, end_year: int) -> Path | None:
    for path in _candidate_prediction_paths(predictions_dir, area, start_year, end_year):
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _write_by_area_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BY_AREA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in BY_AREA_FIELDS})


def _write_transition_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["area", "start_year", "end_year", "true_transition", "predicted_transition", "n"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for item in row.get("transition_counts", []):
                writer.writerow(
                    {
                        "area": row["area"],
                        "start_year": row["start_year"],
                        "end_year": row["end_year"],
                        **item,
                    }
                )


def _mean_metrics(rows: list[dict]) -> dict:
    metrics = [field for field in BY_AREA_FIELDS if field not in {"area", "start_year", "end_year"}]
    summary = {}
    for metric in metrics:
        values = []
        for row in rows:
            value = row.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        summary[metric] = float(mean(values)) if values else None
    return summary


def evaluate_independent_change_validation(
    labels_dir: Path = DEFAULT_LABELS_DIR,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    labels_dir = Path(labels_dir)
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)

    label_files = _discover_label_files(labels_dir)
    prediction_files = _discover_prediction_files(predictions_dir)
    rows: list[dict] = []
    skipped_pairs: list[dict] = []
    if prediction_files:
        for area, year_pairs in sorted(prediction_files.items()):
            years_to_path = label_files.get(area, {})
            for (start_year, end_year), pred_path in sorted(year_pairs.items()):
                if start_year not in years_to_path or end_year not in years_to_path:
                    skipped_pairs.append(
                        {
                            "area": area,
                            "start_year": start_year,
                            "end_year": end_year,
                            "reason": "missing_label",
                            "expected_label_files": [
                                str(labels_dir / f"{area}_lulc_{start_year}.npy"),
                                str(labels_dir / f"{area}_lulc_{end_year}.npy"),
                            ],
                        }
                    )
                    continue
                try:
                    true_start = np.load(years_to_path[start_year])
                    true_end = np.load(years_to_path[end_year])
                    transition_prior_end = _transition_prior_prediction(
                        target_start=true_start,
                        training_pairs=_load_transition_prior_training_pairs(
                            label_files,
                            area,
                            start_year,
                            end_year,
                            true_start.shape,
                        ),
                    )
                    row = evaluate_pair(
                        area=area,
                        start_year=start_year,
                        end_year=end_year,
                        true_start=true_start,
                        true_end=true_end,
                        pred_end=np.load(pred_path),
                        transition_prior_end=transition_prior_end,
                    )
                except ValueError as exc:
                    skipped_pairs.append(
                        {
                            "area": area,
                            "start_year": start_year,
                            "end_year": end_year,
                            "reason": "invalid_pair",
                            "message": str(exc),
                        }
                    )
                    continue
                row["label_start_file"] = str(years_to_path[start_year])
                row["label_end_file"] = str(years_to_path[end_year])
                row["prediction_file"] = str(pred_path)
                rows.append(row)
    else:
        for area, years_to_path in sorted(label_files.items()):
            years = sorted(years_to_path)
            for start_year, end_year in zip(years, years[1:]):
                pred_path = _find_prediction_path(predictions_dir, area, start_year, end_year)
                if pred_path is None:
                    skipped_pairs.append(
                        {
                            "area": area,
                            "start_year": start_year,
                            "end_year": end_year,
                            "reason": "missing_prediction",
                            "expected_prediction_files": [
                                str(path) for path in _candidate_prediction_paths(predictions_dir, area, start_year, end_year)
                            ],
                        }
                    )
                    continue
                try:
                    true_start = np.load(years_to_path[start_year])
                    true_end = np.load(years_to_path[end_year])
                    transition_prior_end = _transition_prior_prediction(
                        target_start=true_start,
                        training_pairs=_load_transition_prior_training_pairs(
                            label_files,
                            area,
                            start_year,
                            end_year,
                            true_start.shape,
                        ),
                    )
                    row = evaluate_pair(
                        area=area,
                        start_year=start_year,
                        end_year=end_year,
                        true_start=true_start,
                        true_end=true_end,
                        pred_end=np.load(pred_path),
                        transition_prior_end=transition_prior_end,
                    )
                except ValueError as exc:
                    skipped_pairs.append(
                        {
                            "area": area,
                            "start_year": start_year,
                            "end_year": end_year,
                            "reason": "invalid_pair",
                            "message": str(exc),
                        }
                    )
                    continue
                row["label_start_file"] = str(years_to_path[start_year])
                row["label_end_file"] = str(years_to_path[end_year])
                row["prediction_file"] = str(pred_path)
                rows.append(row)

    status = "complete" if rows else "missing_predictions" if skipped_pairs else "missing_labels"
    summary = {
        "status": status,
        "labels_dir": str(labels_dir),
        "predictions_dir": str(predictions_dir),
        "n_area_year_pairs": len(rows),
        "n_skipped_pairs": len(skipped_pairs),
        "mean_metrics": _mean_metrics(rows),
        "pairs": rows,
        "skipped_pairs": skipped_pairs,
        "notes": [
            "The persistence baseline predicts the end-year LULC map as identical to the start-year map.",
            "Change metrics treat a pixel as changed when the end-year class differs from the start-year class.",
            "Transition exact-match accuracy is computed only on true changed pixels.",
            f"The shuffled-model negative control uses fixed seed {SHUFFLE_SEED} to preserve the predicted class histogram while disrupting spatial correspondence.",
            "The label-only transition prior baseline estimates start-class to end-class frequencies from other label pairs with the same grid shape and contains no embedding or model-prediction information.",
            "No categorical transition claim should be made unless n_area_year_pairs is greater than zero.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "independent_change_validation_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_by_area_csv(output_dir / "independent_change_validation_by_area.csv", rows)
    _write_transition_csv(output_dir / "independent_change_validation_transitions.csv", rows)
    return _json_ready(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate independent LULC change-label validation for Paper58.")
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = evaluate_independent_change_validation(args.labels_dir, args.predictions_dir, args.output_dir)
    print(
        "Independent change validation: "
        f"{summary['status']}, "
        f"{summary['n_area_year_pairs']} evaluated pair(s), "
        f"{summary['n_skipped_pairs']} skipped pair(s)"
    )


if __name__ == "__main__":
    main()
