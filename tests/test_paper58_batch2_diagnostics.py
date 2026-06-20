from pathlib import Path

import numpy as np

from scripts.paper58_benchmark.make_batch2_diagnostics import (
    best_shift_diagnostic,
    make_embedding_decoder_audit_table,
    make_batch2_alignment_table,
    make_transition_fate_table,
    make_transition_table,
    transition_count_rows,
)


def test_best_shift_diagnostic_finds_known_change_offset():
    true_change = np.zeros((5, 5), dtype=bool)
    model_change = np.zeros((5, 5), dtype=bool)
    true_change[3, 3] = True
    model_change[1, 1] = True

    diagnostic = best_shift_diagnostic(true_change, model_change, max_shift=3)

    assert diagnostic["raw_change_f1"] == 0.0
    assert diagnostic["best_shift_change_f1"] == 1.0
    assert diagnostic["best_dy"] == 2
    assert diagnostic["best_dx"] == 2


def test_make_batch2_alignment_table_writes_shift_diagnostics(tmp_path: Path):
    labels = tmp_path / "labels"
    predictions = tmp_path / "predicted"
    output = tmp_path / "diagnostics"
    labels.mkdir()
    predictions.mkdir()

    start = np.zeros((5, 5), dtype=np.int32)
    end = start.copy()
    pred = start.copy()
    end[3, 3] = 1
    pred[1, 1] = 1
    np.save(labels / "toy_lulc_2020.npy", start)
    np.save(labels / "toy_lulc_2021.npy", end)
    np.save(predictions / "toy_lulc_pred_2020_2021.npy", pred)

    rows = make_batch2_alignment_table(
        out_dir=output,
        labels_dir=labels,
        predictions_dir=predictions,
        areas=["toy"],
        start_year=2020,
        end_year=2021,
        max_shift=3,
    )

    assert rows == [
        {
            "area": "toy",
            "raw_change_f1": 0.0,
            "best_shift_change_f1": 1.0,
            "best_dy": 2,
            "best_dx": 2,
            "centroid_true_y": 3.0,
            "centroid_true_x": 3.0,
            "centroid_model_y": 1.0,
            "centroid_model_x": 1.0,
        }
    ]
    assert (output / "batch2_spatial_alignment_shift.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "area,raw_change_f1,best_shift_change_f1,best_dy,best_dx,"
        "centroid_true_y,centroid_true_x,centroid_model_y,centroid_model_x"
    )


class ToyDecoder:
    def predict(self, pixels):
        return pixels[:, 0].astype(np.int32)


def _toy_embedding(class_grid: np.ndarray) -> np.ndarray:
    embedding = np.zeros((*class_grid.shape, 64), dtype=np.float32)
    embedding[..., 0] = class_grid
    return embedding


def test_transition_count_rows_counts_source_end_pairs():
    start = np.array([[1, 1, 2], [2, 2, 2]], dtype=np.int32)
    end = np.array([[1, 3, 3], [2, 4, 4]], dtype=np.int32)
    mask = start != end

    rows = transition_count_rows(start, end, mask, source="reference_change", limit=3)

    assert rows == [
        {"source": "reference_change", "start_class": 2, "end_class": 4, "n_pixels": 2},
        {"source": "reference_change", "start_class": 1, "end_class": 3, "n_pixels": 1},
        {"source": "reference_change", "start_class": 2, "end_class": 3, "n_pixels": 1},
    ]


def test_make_embedding_decoder_audit_table_writes_decoded_change_metrics(tmp_path: Path):
    labels = tmp_path / "labels"
    embeddings = tmp_path / "embeddings"
    predictions = tmp_path / "predicted"
    output = tmp_path / "diagnostics"
    labels.mkdir()
    embeddings.mkdir()
    predictions.mkdir()

    start = np.zeros((2, 2), dtype=np.int32)
    end = np.array([[0, 1], [0, 0]], dtype=np.int32)
    pred = np.array([[0, 1], [1, 0]], dtype=np.int32)
    np.save(labels / "toy_lulc_2020.npy", start)
    np.save(labels / "toy_lulc_2021.npy", end)
    np.save(embeddings / "toy_emb_2020.npy", _toy_embedding(start))
    np.save(embeddings / "toy_emb_2021.npy", _toy_embedding(end))
    np.save(predictions / "toy_lulc_pred_2020_2021.npy", pred)

    rows = make_embedding_decoder_audit_table(
        out_dir=output,
        decoder=ToyDecoder(),
        labels_dir=labels,
        embeddings_dir=embeddings,
        predictions_dir=predictions,
        areas=["toy"],
        start_year=2020,
        end_year=2021,
        max_shift=2,
    )

    assert rows == [
        {
            "area": "toy",
            "start_decode_accuracy": 1.0,
            "end_decode_accuracy": 1.0,
            "true_change_pixels": 1,
            "decoded_observed_change_pixels": 1,
            "decoded_observed_change_precision": 1.0,
            "decoded_observed_change_recall": 1.0,
            "decoded_observed_change_f1": 1.0,
            "decoded_observed_best_shift_change_f1": 1.0,
            "decoded_observed_best_dy": 0,
            "decoded_observed_best_dx": 0,
            "model_change_f1_label_start": 2 / 3,
            "model_change_f1_decoded_start": 2 / 3,
            "model_end_accuracy": 0.75,
            "model_start_accuracy": 0.5,
        }
    ]
    assert (output / "batch2_embedding_decoder_audit.csv").exists()


def test_make_transition_table_writes_reference_model_and_decoded_sources(tmp_path: Path):
    labels = tmp_path / "labels"
    embeddings = tmp_path / "embeddings"
    predictions = tmp_path / "predicted"
    output = tmp_path / "diagnostics"
    labels.mkdir()
    embeddings.mkdir()
    predictions.mkdir()

    start = np.array([[0, 0], [0, 0]], dtype=np.int32)
    end = np.array([[0, 1], [0, 0]], dtype=np.int32)
    pred = np.array([[0, 1], [1, 0]], dtype=np.int32)
    np.save(labels / "toy_lulc_2020.npy", start)
    np.save(labels / "toy_lulc_2021.npy", end)
    np.save(embeddings / "toy_emb_2020.npy", _toy_embedding(start))
    np.save(embeddings / "toy_emb_2021.npy", _toy_embedding(end))
    np.save(predictions / "toy_lulc_pred_2020_2021.npy", pred)

    rows = make_transition_table(
        out_dir=output,
        decoder=ToyDecoder(),
        labels_dir=labels,
        embeddings_dir=embeddings,
        predictions_dir=predictions,
        area="toy",
        start_year=2020,
        end_year=2021,
        limit=5,
    )

    assert rows == [
        {"source": "reference_change", "start_class": 0, "end_class": 1, "n_pixels": 1},
        {"source": "model_change", "start_class": 0, "end_class": 1, "n_pixels": 2},
        {"source": "decoded_observed_change", "start_class": 0, "end_class": 1, "n_pixels": 1},
    ]
    assert (output / "toy_transition_counts.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "source,start_class,end_class,n_pixels"
    )


def test_make_transition_fate_table_tracks_true_transition_destinations(tmp_path: Path):
    labels = tmp_path / "labels"
    embeddings = tmp_path / "embeddings"
    predictions = tmp_path / "predicted"
    output = tmp_path / "diagnostics"
    labels.mkdir()
    embeddings.mkdir()
    predictions.mkdir()

    start = np.array([[5, 5], [7, 7]], dtype=np.int32)
    end = np.array([[11, 7], [5, 11]], dtype=np.int32)
    pred = np.array([[5, 7], [7, 5]], dtype=np.int32)
    decoded_start = np.array([[5, 5], [7, 7]], dtype=np.int32)
    decoded_end = np.array([[5, 7], [5, 5]], dtype=np.int32)
    np.save(labels / "toy_lulc_2020.npy", start)
    np.save(labels / "toy_lulc_2021.npy", end)
    np.save(embeddings / "toy_emb_2020.npy", _toy_embedding(decoded_start))
    np.save(embeddings / "toy_emb_2021.npy", _toy_embedding(decoded_end))
    np.save(predictions / "toy_lulc_pred_2020_2021.npy", pred)

    rows = make_transition_fate_table(
        out_dir=output,
        decoder=ToyDecoder(),
        labels_dir=labels,
        embeddings_dir=embeddings,
        predictions_dir=predictions,
        area="toy",
        start_year=2020,
        end_year=2021,
        top_n_true_transitions=4,
    )

    assert {row["true_transition"]: row for row in rows} == {
        "5->11": {
            "true_transition": "5->11",
            "n_true_pixels": 1,
            "decoded_start_top": "5:1",
            "decoded_end_top": "5:1",
            "model_end_top": "5:1",
        },
        "5->7": {
            "true_transition": "5->7",
            "n_true_pixels": 1,
            "decoded_start_top": "5:1",
            "decoded_end_top": "7:1",
            "model_end_top": "7:1",
        },
        "7->5": {
            "true_transition": "7->5",
            "n_true_pixels": 1,
            "decoded_start_top": "7:1",
            "decoded_end_top": "5:1",
            "model_end_top": "7:1",
        },
        "7->11": {
            "true_transition": "7->11",
            "n_true_pixels": 1,
            "decoded_start_top": "7:1",
            "decoded_end_top": "5:1",
            "model_end_top": "5:1",
        },
    }
    assert (output / "toy_transition_fate.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "true_transition,n_true_pixels,decoded_start_top,decoded_end_top,model_end_top"
    )
