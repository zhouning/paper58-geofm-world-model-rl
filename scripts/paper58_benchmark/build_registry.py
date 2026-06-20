from __future__ import annotations

import argparse
import re
from dataclasses import fields
from pathlib import Path

import numpy as np

from scripts.paper58_benchmark.holdouts import (
    DEFAULT_HOLDOUT_MANIFEST,
    HoldoutArea,
    load_holdout_manifest,
    manifest_lookup,
)
from scripts.paper58_benchmark.schema import (
    DEFAULT_BENCHMARK_DIR,
    DEFAULT_EXPERIMENT_DATA_DIR,
    DEFAULT_INDEPENDENT_EMBEDDINGS_DIR,
    DEFAULT_LABELS_DIR,
    DEFAULT_PREDICTIONS_DIR,
    BenchmarkRow,
    area_stratum,
    assign_tier,
    row_to_dict,
    write_csv,
    write_json,
)


LABEL_RE = re.compile(r"^(?P<area>.+)_lulc_(?P<year>\d{4})\.npy$")
PRED_RE = re.compile(r"^(?P<area>.+)_lulc_pred_(?P<start_year>\d{4})_(?P<end_year>\d{4})\.npy$")
REGISTRY_FIELDS = [field.name for field in fields(BenchmarkRow)]


def parse_label_filename(name: str) -> tuple[str, int] | None:
    match = LABEL_RE.match(name)
    if match is None:
        return None
    return match.group("area"), int(match.group("year"))


def parse_prediction_filename(name: str) -> tuple[str, int, int] | None:
    match = PRED_RE.match(name)
    if match is None:
        return None
    return match.group("area"), int(match.group("start_year")), int(match.group("end_year"))


def _discover_labels(labels_dir: Path) -> dict[str, dict[int, Path]]:
    discovered: dict[str, dict[int, Path]] = {}
    for path in sorted(labels_dir.glob("*_lulc_*.npy")):
        parsed = parse_label_filename(path.name)
        if parsed is None:
            continue
        area, year = parsed
        discovered.setdefault(area, {})[year] = path
    return discovered


def _discover_predictions(predictions_dir: Path) -> list[tuple[str, int, int, Path]]:
    rows = []
    for path in sorted(predictions_dir.glob("*_lulc_pred_*_*.npy")):
        parsed = parse_prediction_filename(path.name)
        if parsed is None:
            continue
        area, start_year, end_year = parsed
        rows.append((area, start_year, end_year, path))
    return rows


def _embedding_area_aliases(area: str) -> list[str]:
    aliases = [area]
    if area == "banzhucun":
        aliases.append("village")
    return aliases


def _find_embedding_path(area: str, year: int, independent_embeddings_dir: Path, experiment_data_dir: Path) -> Path | None:
    candidates = []
    for alias in _embedding_area_aliases(area):
        candidates.append(independent_embeddings_dir / f"{alias}_emb_{year}.npy")
        candidates.append(experiment_data_dir / f"{alias}_emb_{year}.npy")
        candidates.append(experiment_data_dir / "prithvi" / f"{alias}_emb_{year}.npy")
        candidates.append(experiment_data_dir / alias / f"{alias}_emb_{year}.npy")
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _find_context_path(area: str, independent_embeddings_dir: Path, experiment_data_dir: Path) -> Path | None:
    candidates = []
    for alias in _embedding_area_aliases(area):
        candidates.append(independent_embeddings_dir / f"{alias}_context.npy")
        candidates.append(experiment_data_dir / f"{alias}_context.npy")
        candidates.append(experiment_data_dir / alias / f"{alias}_context.npy")
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _load_holdout_lookup(path: Path | None) -> dict[str, HoldoutArea]:
    if path is None or not Path(path).exists():
        return {}
    return {area.lower(): record for area, record in manifest_lookup(load_holdout_manifest(Path(path))).items()}


def _filter_predictions_to_holdouts(
    predictions: list[tuple[str, int, int, Path]],
    holdouts: dict[str, HoldoutArea],
) -> list[tuple[str, int, int, Path]]:
    return [
        (area, start_year, end_year, prediction_path)
        for area, start_year, end_year, prediction_path in predictions
        if area.lower() in holdouts
    ]


def _shape(path: Path | None) -> tuple[int, ...] | None:
    if path is None:
        return None
    return tuple(np.load(path, mmap_mode="r").shape)


def _class_collapse(prediction: np.ndarray, true_change_pixels: int) -> bool:
    return true_change_pixels > 0 and np.unique(prediction).size <= 1


def _build_row(
    area: str,
    start_year: int,
    end_year: int,
    prediction_path: Path,
    labels: dict[str, dict[int, Path]],
    independent_embeddings_dir: Path,
    experiment_data_dir: Path,
    holdouts: dict[str, HoldoutArea],
) -> BenchmarkRow:
    label_start_path = labels.get(area, {}).get(start_year)
    label_end_path = labels.get(area, {}).get(end_year)
    embedding_start_path = _find_embedding_path(area, start_year, independent_embeddings_dir, experiment_data_dir)
    embedding_end_path = _find_embedding_path(area, end_year, independent_embeddings_dir, experiment_data_dir)
    context_path = _find_context_path(area, independent_embeddings_dir, experiment_data_dir)
    holdout = holdouts.get(area.lower())
    if holdout and (start_year not in holdout.years or end_year not in holdout.years):
        holdout = None
    bbox = holdout.bbox if holdout else None
    data_source = holdout.data_source if holdout else ""
    development_contact_status = holdout.development_contact_status if holdout else "uncertain"
    contact_evidence = holdout.contact_evidence if holdout else "area missing from Paper58 holdout manifest"
    expected_role = holdout.expected_role if holdout else "review_required"
    stratum = holdout.stratum if holdout else area_stratum(area)
    tier = assign_tier(area, development_contact_status=development_contact_status)
    qc_status = "include"
    excluded_reason = ""
    n_pixels = 0
    true_change_pixels = 0
    true_change_pct = 0.0
    label_shape = None
    prediction_shape = _shape(prediction_path)
    embedding_shape = _shape(embedding_start_path)

    if label_start_path is None or label_end_path is None:
        qc_status = "exclude"
        excluded_reason = "missing_label"
    else:
        start = np.load(label_start_path)
        end = np.load(label_end_path)
        pred = np.load(prediction_path)
        label_shape = tuple(start.shape)
        prediction_shape = tuple(pred.shape)
        n_pixels = int(start.size)
        if start.shape != end.shape:
            qc_status = "exclude"
            excluded_reason = "label_shape_mismatch"
        elif start.shape != pred.shape:
            qc_status = "exclude"
            excluded_reason = "label_prediction_shape_mismatch"
        elif start.size == 0:
            qc_status = "exclude"
            excluded_reason = "empty_arrays"
        else:
            true_change = end != start
            true_change_pixels = int(np.count_nonzero(true_change))
            true_change_pct = float(true_change_pixels / start.size)
            if true_change_pixels == 0:
                qc_status = "negative_control"
                excluded_reason = "zero_reference_change"
            elif _class_collapse(pred, true_change_pixels):
                qc_status = "exclude"
                excluded_reason = "class_collapse"

    if qc_status == "include" and (embedding_start_path is None or embedding_end_path is None):
        qc_status = "exclude"
        excluded_reason = "missing_embedding"

    return BenchmarkRow(
        area=area,
        start_year=start_year,
        end_year=end_year,
        tier=tier,
        stratum=stratum,
        bbox=bbox,
        data_source=data_source,
        development_contact_status=development_contact_status,
        contact_evidence=contact_evidence,
        expected_role=expected_role,
        label_start_path=label_start_path,
        label_end_path=label_end_path,
        prediction_path=prediction_path,
        embedding_start_path=embedding_start_path,
        embedding_end_path=embedding_end_path,
        context_path=context_path,
        label_shape=label_shape,
        prediction_shape=prediction_shape,
        embedding_shape=embedding_shape,
        n_pixels=n_pixels,
        true_change_pixels=true_change_pixels,
        true_change_pct=true_change_pct,
        qc_status=qc_status,
        excluded_reason=excluded_reason,
    )


def build_registry(
    labels_dir: Path = DEFAULT_LABELS_DIR,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
    independent_embeddings_dir: Path = DEFAULT_INDEPENDENT_EMBEDDINGS_DIR,
    experiment_data_dir: Path = DEFAULT_EXPERIMENT_DATA_DIR,
    output_dir: Path = DEFAULT_BENCHMARK_DIR,
    holdout_manifest_path: Path | None = DEFAULT_HOLDOUT_MANIFEST,
    filter_to_holdout_manifest: bool = False,
) -> list[BenchmarkRow]:
    labels = _discover_labels(Path(labels_dir))
    predictions = _discover_predictions(Path(predictions_dir))
    holdouts = _load_holdout_lookup(holdout_manifest_path)
    if filter_to_holdout_manifest:
        predictions = _filter_predictions_to_holdouts(predictions, holdouts)
    rows = [
        _build_row(
            area=area,
            start_year=start_year,
            end_year=end_year,
            prediction_path=prediction_path,
            labels=labels,
            independent_embeddings_dir=Path(independent_embeddings_dir),
            experiment_data_dir=Path(experiment_data_dir),
            holdouts=holdouts,
        )
        for area, start_year, end_year, prediction_path in predictions
    ]
    dict_rows = [row_to_dict(row) for row in rows]
    output_dir = Path(output_dir)
    write_json(output_dir / "benchmark_registry.json", {"rows": dict_rows})
    write_csv(output_dir / "benchmark_registry.csv", dict_rows, REGISTRY_FIELDS)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Paper58 external benchmark registry.")
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--independent-embeddings-dir", type=Path, default=DEFAULT_INDEPENDENT_EMBEDDINGS_DIR)
    parser.add_argument("--experiment-data-dir", type=Path, default=DEFAULT_EXPERIMENT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--holdout-manifest", type=Path, default=DEFAULT_HOLDOUT_MANIFEST)
    parser.add_argument(
        "--filter-to-holdout-manifest",
        action="store_true",
        help="Only include prediction rows whose area appears in the holdout manifest.",
    )
    args = parser.parse_args()
    rows = build_registry(
        labels_dir=args.labels_dir,
        predictions_dir=args.predictions_dir,
        independent_embeddings_dir=args.independent_embeddings_dir,
        experiment_data_dir=args.experiment_data_dir,
        output_dir=args.output_dir,
        holdout_manifest_path=args.holdout_manifest,
        filter_to_holdout_manifest=args.filter_to_holdout_manifest,
    )
    included = sum(row.qc_status == "include" for row in rows)
    print(f"Benchmark registry: {len(rows)} candidate pair(s), {included} included pair(s)")


if __name__ == "__main__":
    main()
