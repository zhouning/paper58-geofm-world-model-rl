from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import scripts.paper58_benchmark.make_batch2_diagnostics as batch2_diagnostics


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH5_RESULTS_DIR = ROOT / "paper" / "rse_submission_paper58" / "benchmark_results_batch5"
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "rse_submission_paper58" / "diagnostics_batch5_liaohe"
DEFAULT_LABELS_DIR = ROOT / "data" / "independent_change_labels" / "labels"
DEFAULT_EMBEDDINGS_DIR = ROOT / "data" / "independent_change_labels" / "embeddings"
DEFAULT_PREDICTIONS_DIR = ROOT / "data" / "independent_change_labels" / "predicted"
DEFAULT_DECODER_PATH = ROOT / "src" / "adk_world_model" / "weights" / "lulc_decoder_v1.pkl"
DEFAULT_WEIGHTS_PATH = ROOT / "src" / "adk_world_model" / "weights" / "latent_dynamics_v1.pt"

FOCUS_AREA = "liaohe_delta_wetland_holdout"
WETLAND_COMPARISON_AREAS = [
    "erlong_lake_margin_holdout",
    "honghu_lake_margin_holdout",
    "zhalong_wetland_edge_holdout",
]
SECONDARY_RISK_AREAS = ["wenan_lakeplain_newtown_holdout"]

METRIC_FIELDS = [
    "area",
    "start_year",
    "end_year",
    "tier",
    "stratum",
    "true_change_pixels",
    "model_change_f1",
    "primary_change_advantage",
    "spatial_shuffle_change_f1",
    "spatial_change_advantage",
]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _read_metrics(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [field for field in METRIC_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name} missing required columns: {', '.join(missing)}")
        rows = []
        for row in reader:
            rows.append(
                {
                    "area": row["area"],
                    "start_year": int(row["start_year"]),
                    "end_year": int(row["end_year"]),
                    "tier": row["tier"],
                    "stratum": row["stratum"],
                    "true_change_pixels": int(row["true_change_pixels"]),
                    "model_change_f1": float(row["model_change_f1"]),
                    "primary_change_advantage": float(row["primary_change_advantage"]),
                    "spatial_shuffle_change_f1": float(row["spatial_shuffle_change_f1"]),
                    "spatial_change_advantage": float(row["spatial_change_advantage"]),
                }
            )
    return rows


def _metric_by_area(rows: list[dict]) -> dict[str, dict]:
    return {row["area"]: row for row in rows}


def _load_decoder(decoder_path: Path):
    return batch2_diagnostics._load_decoder(decoder_path)


def _load_model(weights_path: Path):
    return batch2_diagnostics._load_forecast_model(weights_path)


def _first_for_area(rows: list[dict], area: str) -> dict | None:
    return next((row for row in rows if row.get("area") == area), None)


def _top_transition(rows: list[dict], area: str) -> dict | None:
    return next((row for row in rows if row.get("area") == area), None)


def build_liaohe_diagnostics(
    batch5_results_dir: Path = DEFAULT_BATCH5_RESULTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
    decoder=None,
    model=None,
    decoder_path: Path = DEFAULT_DECODER_PATH,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    diagnostics=batch2_diagnostics,
) -> dict:
    batch5_results_dir = Path(batch5_results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _read_metrics(batch5_results_dir / "benchmark_metrics_by_pair.csv")
    metrics_by_area = _metric_by_area(metrics)
    if FOCUS_AREA not in metrics_by_area:
        raise ValueError(f"Batch 5 metrics do not contain focus area: {FOCUS_AREA}")

    diagnostic_areas = [FOCUS_AREA, *WETLAND_COMPARISON_AREAS, *SECONDARY_RISK_AREAS]
    decoder = decoder if decoder is not None else _load_decoder(decoder_path)
    model = model if model is not None else _load_model(weights_path)

    alignment_rows = diagnostics.make_batch2_alignment_table(
        out_dir=output_dir,
        labels_dir=labels_dir,
        predictions_dir=predictions_dir,
        areas=diagnostic_areas,
        output_filename="batch5_liaohe_spatial_alignment_shift.csv",
    )
    diagnostics.make_embedding_decoder_audit_table(
        out_dir=output_dir,
        decoder=decoder,
        labels_dir=labels_dir,
        embeddings_dir=embeddings_dir,
        predictions_dir=predictions_dir,
        areas=diagnostic_areas,
        output_filename="batch5_liaohe_embedding_decoder_audit.csv",
    )
    true_end_confidence_rows = diagnostics.make_decoder_true_end_confidence_table(
        out_dir=output_dir,
        decoder=decoder,
        labels_dir=labels_dir,
        embeddings_dir=embeddings_dir,
        areas=diagnostic_areas,
        output_filename="batch5_liaohe_decoder_true_end_confidence_by_area.csv",
    )
    forecast_true_end_confidence_rows = diagnostics.make_forecast_true_end_confidence_table(
        out_dir=output_dir,
        decoder=decoder,
        model=model,
        labels_dir=labels_dir,
        embeddings_dir=embeddings_dir,
        areas=diagnostic_areas,
        output_filename="batch5_liaohe_forecast_true_end_confidence_by_area.csv",
    )

    alignment_by_area = {row["area"]: row for row in alignment_rows}
    transition_rows: list[dict] = []
    transition_fate_rows: list[dict] = []
    shifted_transition_fate_rows: list[dict] = []
    forecast_transition_fate_rows: list[dict] = []
    for area in diagnostic_areas:
        for row in diagnostics.make_transition_table(
            out_dir=output_dir,
            decoder=decoder,
            labels_dir=labels_dir,
            embeddings_dir=embeddings_dir,
            predictions_dir=predictions_dir,
            area=area,
        ):
            transition_rows.append({"area": area, **row})
        for row in diagnostics.make_transition_fate_table(
            out_dir=output_dir,
            decoder=decoder,
            labels_dir=labels_dir,
            embeddings_dir=embeddings_dir,
            predictions_dir=predictions_dir,
            area=area,
        ):
            transition_fate_rows.append({"area": area, **row})
        alignment = alignment_by_area.get(area, {"best_dy": 0, "best_dx": 0})
        for row in diagnostics.make_shifted_transition_fate_table(
            out_dir=output_dir,
            labels_dir=labels_dir,
            predictions_dir=predictions_dir,
            area=area,
            shift_dy=int(alignment["best_dy"]),
            shift_dx=int(alignment["best_dx"]),
        ):
            shifted_transition_fate_rows.append({"area": area, **row})
        for row in diagnostics.make_forecast_transition_fate_table(
            out_dir=output_dir,
            decoder=decoder,
            model=model,
            labels_dir=labels_dir,
            embeddings_dir=embeddings_dir,
            predictions_dir=predictions_dir,
            area=area,
        ):
            forecast_transition_fate_rows.append({"area": area, **row})

    _write_csv(
        output_dir / "batch5_liaohe_transition_counts_all.csv",
        transition_rows,
        ["area", "source", "start_class", "end_class", "n_pixels"],
    )
    _write_csv(
        output_dir / "batch5_liaohe_transition_fate_all.csv",
        transition_fate_rows,
        [
            "area",
            "true_transition",
            "n_true_pixels",
            "decoded_start_top",
            "decoded_end_top",
            "model_end_top",
            "mean_true_end_prob",
            "median_true_end_prob",
            "top_mean_prob_class",
            "top_mean_prob",
            "second_mean_prob_class",
            "second_mean_prob",
        ],
    )
    _write_csv(
        output_dir / "batch5_liaohe_shifted_transition_fate_all.csv",
        shifted_transition_fate_rows,
        [
            "area",
            "true_transition",
            "n_true_pixels",
            "raw_model_end_top",
            "shifted_model_end_top",
            "raw_match_pixels",
            "shifted_match_pixels",
        ],
    )
    _write_csv(
        output_dir / "batch5_liaohe_forecast_transition_fate_all.csv",
        forecast_transition_fate_rows,
        [
            "area",
            "true_transition",
            "n_true_pixels",
            "observed_end_top",
            "forecast_end_top",
            "observed_mean_true_end_prob",
            "observed_median_true_end_prob",
            "forecast_mean_true_end_prob",
            "forecast_median_true_end_prob",
            "mean_true_end_prob_delta",
            "observed_top_mean_prob_class",
            "observed_top_mean_prob",
            "forecast_top_mean_prob_class",
            "forecast_top_mean_prob",
        ],
    )

    focus_alignment = alignment_by_area.get(FOCUS_AREA)
    focus_top_transition = _top_transition(transition_fate_rows, FOCUS_AREA)
    focus_forecast_top_transition = _top_transition(forecast_transition_fate_rows, FOCUS_AREA)
    focus_class11_confidence = next(
        (
            row
            for row in true_end_confidence_rows
            if row.get("area") == FOCUS_AREA and row.get("true_end_class") == 11
        ),
        None,
    )
    focus_forecast_class11_confidence = next(
        (
            row
            for row in forecast_true_end_confidence_rows
            if row.get("area") == FOCUS_AREA and row.get("true_end_class") == 11
        ),
        None,
    )
    summary = {
        "focus_area": FOCUS_AREA,
        "focus_metrics": metrics_by_area[FOCUS_AREA],
        "focus_alignment": focus_alignment,
        "focus_top_transition_fate": focus_top_transition,
        "focus_forecast_top_transition_fate": focus_forecast_top_transition,
        "focus_class11_confidence": focus_class11_confidence,
        "focus_forecast_class11_confidence": focus_forecast_class11_confidence,
        "wetland_comparison_areas": WETLAND_COMPARISON_AREAS,
        "secondary_risks": [metrics_by_area[area] for area in SECONDARY_RISK_AREAS if area in metrics_by_area],
        "diagnostic_areas": diagnostic_areas,
    }
    (output_dir / "batch5_liaohe_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "batch5_liaohe_summary.txt").write_text(
        "\n".join(
            [
                f"focus_area={FOCUS_AREA}",
                f"focus_primary_change_advantage={metrics_by_area[FOCUS_AREA]['primary_change_advantage']}",
                f"focus_spatial_change_advantage={metrics_by_area[FOCUS_AREA]['spatial_change_advantage']}",
                f"focus_true_change_pixels={metrics_by_area[FOCUS_AREA]['true_change_pixels']}",
                f"focus_best_shift_change_f1={focus_alignment.get('best_shift_change_f1') if focus_alignment else ''}",
                f"focus_best_shift_dy={focus_alignment.get('best_dy') if focus_alignment else ''}",
                f"focus_best_shift_dx={focus_alignment.get('best_dx') if focus_alignment else ''}",
                f"focus_top_transition={focus_top_transition.get('true_transition') if focus_top_transition else ''}",
                f"focus_top_transition_decoded_end={focus_top_transition.get('decoded_end_top') if focus_top_transition else ''}",
                f"focus_top_transition_model_end={focus_top_transition.get('model_end_top') if focus_top_transition else ''}",
                f"focus_class11_mean_true_end_prob={focus_class11_confidence.get('mean_true_end_prob') if focus_class11_confidence else ''}",
                (
                    "focus_forecast_class11_mean_true_end_prob="
                    f"{focus_forecast_class11_confidence.get('forecast_mean_true_end_prob')}"
                    if focus_forecast_class11_confidence
                    else "focus_forecast_class11_mean_true_end_prob="
                ),
                "secondary_risk_areas=" + ",".join(area for area in SECONDARY_RISK_AREAS if area in metrics_by_area),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Paper58 Batch 5 Liaohe wetland diagnostics.")
    parser.add_argument("--batch5-results-dir", type=Path, default=DEFAULT_BATCH5_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--decoder", type=Path, default=DEFAULT_DECODER_PATH)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    args = parser.parse_args()
    build_liaohe_diagnostics(
        batch5_results_dir=args.batch5_results_dir,
        output_dir=args.output_dir,
        labels_dir=args.labels_dir,
        embeddings_dir=args.embeddings_dir,
        predictions_dir=args.predictions_dir,
        decoder_path=args.decoder,
        weights_path=args.weights,
    )
    print(f"Wrote Batch 5 Liaohe diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
