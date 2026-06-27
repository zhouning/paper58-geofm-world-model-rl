from __future__ import annotations

import argparse
import csv
from pathlib import Path

import scripts.paper58_benchmark.make_batch2_diagnostics as batch2_diagnostics


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "rse_submission_paper58" / "diagnostics_batch23_urban"
DEFAULT_LABELS_DIR = ROOT / "data" / "independent_change_labels" / "labels"
DEFAULT_EMBEDDINGS_DIR = ROOT / "data" / "independent_change_labels" / "embeddings"
DEFAULT_PREDICTIONS_DIR = ROOT / "data" / "independent_change_labels" / "predicted"
DEFAULT_DECODER_PATH = ROOT / "src" / "adk_world_model" / "weights" / "lulc_decoder_v1.pkl"
DEFAULT_WEIGHTS_PATH = ROOT / "src" / "adk_world_model" / "weights" / "latent_dynamics_v1.pt"

BATCH2_URBAN_AREAS = [
    "xiong_an_fringe_holdout",
    "beibu_gulf_urban_holdout",
]

BATCH3_URBAN_AREAS = [
    "fuzhou_delta_urban_holdout",
    "nanning_fringe_holdout",
    "suzhou_fringe_holdout",
    "wuhan_outer_ring_holdout",
]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _load_decoder(decoder_path: Path):
    return batch2_diagnostics._load_decoder(decoder_path)


def _load_model(weights_path: Path):
    return batch2_diagnostics._load_forecast_model(weights_path)


def build_urban_contrast_diagnostics(
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_areas = BATCH2_URBAN_AREAS + BATCH3_URBAN_AREAS

    decoder = decoder if decoder is not None else _load_decoder(decoder_path)
    model = model if model is not None else _load_model(weights_path)

    diagnostics.make_batch2_alignment_table(
        out_dir=output_dir,
        labels_dir=labels_dir,
        predictions_dir=predictions_dir,
        areas=all_areas,
    )
    diagnostics.make_embedding_decoder_audit_table(
        out_dir=output_dir,
        decoder=decoder,
        labels_dir=labels_dir,
        embeddings_dir=embeddings_dir,
        predictions_dir=predictions_dir,
        areas=all_areas,
    )
    diagnostics.make_decoder_true_end_confidence_table(
        out_dir=output_dir,
        decoder=decoder,
        labels_dir=labels_dir,
        embeddings_dir=embeddings_dir,
        areas=all_areas,
    )
    diagnostics.make_forecast_true_end_confidence_table(
        out_dir=output_dir,
        decoder=decoder,
        model=model,
        labels_dir=labels_dir,
        embeddings_dir=embeddings_dir,
        areas=all_areas,
    )

    transition_fate_rows: list[dict] = []
    forecast_transition_fate_rows: list[dict] = []
    transition_rows: list[dict] = []
    for area in all_areas:
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
        output_dir / "urban_transition_all.csv",
        transition_rows,
        ["area", "source", "start_class", "end_class", "n_pixels"],
    )
    _write_csv(
        output_dir / "urban_transition_fate_all.csv",
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
        output_dir / "urban_forecast_transition_fate_all.csv",
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

    summary = {
        "n_urban_areas": len(all_areas),
        "batch2_urban_areas": BATCH2_URBAN_AREAS,
        "batch3_urban_areas": BATCH3_URBAN_AREAS,
        "output_dir": str(output_dir),
    }
    (output_dir / "urban_contrast_summary.txt").write_text(
        "\n".join(
            [
                f"n_urban_areas={summary['n_urban_areas']}",
                f"batch2_urban_areas={','.join(BATCH2_URBAN_AREAS)}",
                f"batch3_urban_areas={','.join(BATCH3_URBAN_AREAS)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Paper58 Batch 2 vs Batch 3 urban contrast diagnostics.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--decoder", type=Path, default=DEFAULT_DECODER_PATH)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    args = parser.parse_args()
    build_urban_contrast_diagnostics(
        output_dir=args.output_dir,
        labels_dir=args.labels_dir,
        embeddings_dir=args.embeddings_dir,
        predictions_dir=args.predictions_dir,
        decoder_path=args.decoder,
        weights_path=args.weights,
    )
    print(f"Wrote Batch 2 vs Batch 3 urban contrast diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
