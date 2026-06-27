from pathlib import Path

import numpy as np

from scripts.paper58_benchmark.make_batch2_diagnostics import (
    best_shift_diagnostic,
    make_decoder_true_end_confidence_table,
    make_embedding_decoder_audit_table,
    make_batch2_alignment_table,
    make_forecast_transition_fate_table,
    make_forecast_true_end_confidence_table,
    make_shifted_transition_fate_table,
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


def test_make_batch2_alignment_table_accepts_custom_output_filename(tmp_path: Path):
    labels = tmp_path / "labels"
    predictions = tmp_path / "predicted"
    output = tmp_path / "diagnostics"
    labels.mkdir()
    predictions.mkdir()

    start = np.zeros((3, 3), dtype=np.int32)
    end = start.copy()
    pred = start.copy()
    end[2, 2] = 1
    pred[2, 2] = 1
    np.save(labels / "toy_lulc_2020.npy", start)
    np.save(labels / "toy_lulc_2021.npy", end)
    np.save(predictions / "toy_lulc_pred_2020_2021.npy", pred)

    make_batch2_alignment_table(
        out_dir=output,
        labels_dir=labels,
        predictions_dir=predictions,
        areas=["toy"],
        output_filename="batch5_spatial_alignment_shift.csv",
    )

    assert (output / "batch5_spatial_alignment_shift.csv").exists()
    assert not (output / "batch2_spatial_alignment_shift.csv").exists()


class ToyDecoder:
    def predict(self, pixels):
        return pixels[:, 0].astype(np.int32)


def _toy_embedding(class_grid: np.ndarray) -> np.ndarray:
    embedding = np.zeros((*class_grid.shape, 64), dtype=np.float32)
    embedding[..., 0] = class_grid
    return embedding


class ToyProbDecoder:
    classes_ = np.array([5, 7, 11], dtype=np.int32)

    def predict_proba(self, pixels):
        probs = pixels[:, :3].astype(np.float64)
        totals = probs.sum(axis=1, keepdims=True)
        return probs / totals

    def predict(self, pixels):
        probs = self.predict_proba(pixels)
        return self.classes_[np.argmax(probs, axis=1)]


def _toy_prob_embedding(prob_grid: np.ndarray) -> np.ndarray:
    embedding = np.zeros((*prob_grid.shape[:2], 64), dtype=np.float32)
    embedding[..., :3] = prob_grid
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
    start_probs = np.array(
        [
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    end_probs = np.array(
        [
            [[0.90, 0.09, 0.01], [0.20, 0.70, 0.10]],
            [[0.80, 0.15, 0.05], [0.60, 0.30, 0.10]],
        ],
        dtype=np.float32,
    )
    np.save(labels / "toy_lulc_2020.npy", start)
    np.save(labels / "toy_lulc_2021.npy", end)
    np.save(embeddings / "toy_emb_2020.npy", _toy_prob_embedding(start_probs))
    np.save(embeddings / "toy_emb_2021.npy", _toy_prob_embedding(end_probs))
    np.save(predictions / "toy_lulc_pred_2020_2021.npy", pred)

    rows = make_transition_fate_table(
        out_dir=output,
        decoder=ToyProbDecoder(),
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
            "mean_true_end_prob": 0.01,
            "median_true_end_prob": 0.01,
            "top_mean_prob_class": 5,
            "top_mean_prob": 0.9,
            "second_mean_prob_class": 7,
            "second_mean_prob": 0.09,
        },
        "5->7": {
            "true_transition": "5->7",
            "n_true_pixels": 1,
            "decoded_start_top": "5:1",
            "decoded_end_top": "7:1",
            "model_end_top": "7:1",
            "mean_true_end_prob": 0.7,
            "median_true_end_prob": 0.7,
            "top_mean_prob_class": 7,
            "top_mean_prob": 0.7,
            "second_mean_prob_class": 5,
            "second_mean_prob": 0.2,
        },
        "7->5": {
            "true_transition": "7->5",
            "n_true_pixels": 1,
            "decoded_start_top": "7:1",
            "decoded_end_top": "5:1",
            "model_end_top": "7:1",
            "mean_true_end_prob": 0.8,
            "median_true_end_prob": 0.8,
            "top_mean_prob_class": 5,
            "top_mean_prob": 0.8,
            "second_mean_prob_class": 7,
            "second_mean_prob": 0.15,
        },
        "7->11": {
            "true_transition": "7->11",
            "n_true_pixels": 1,
            "decoded_start_top": "7:1",
            "decoded_end_top": "5:1",
            "model_end_top": "5:1",
            "mean_true_end_prob": 0.1,
            "median_true_end_prob": 0.1,
            "top_mean_prob_class": 5,
            "top_mean_prob": 0.6,
            "second_mean_prob_class": 7,
            "second_mean_prob": 0.3,
        },
    }
    assert (output / "toy_transition_fate.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "true_transition,n_true_pixels,decoded_start_top,decoded_end_top,model_end_top,"
        "mean_true_end_prob,median_true_end_prob,top_mean_prob_class,top_mean_prob,"
        "second_mean_prob_class,second_mean_prob"
    )


def test_make_decoder_true_end_confidence_table_sorts_low_confidence_rows_first(tmp_path: Path):
    labels = tmp_path / "labels"
    embeddings = tmp_path / "embeddings"
    output = tmp_path / "diagnostics"
    labels.mkdir()
    embeddings.mkdir()

    start_a = np.array([[5, 5], [5, 5]], dtype=np.int32)
    end_a = np.array([[11, 5], [5, 5]], dtype=np.int32)
    end_a_probs = np.array(
        [
            [[0.90, 0.09, 0.01], [0.80, 0.15, 0.05]],
            [[0.80, 0.15, 0.05], [0.80, 0.15, 0.05]],
        ],
        dtype=np.float32,
    )

    start_b = np.array([[5, 5], [5, 5]], dtype=np.int32)
    end_b = np.array([[11, 11], [5, 5]], dtype=np.int32)
    end_b_probs = np.array(
        [
            [[0.30, 0.20, 0.50], [0.20, 0.20, 0.60]],
            [[0.80, 0.15, 0.05], [0.80, 0.15, 0.05]],
        ],
        dtype=np.float32,
    )

    np.save(labels / "area_a_lulc_2020.npy", start_a)
    np.save(labels / "area_a_lulc_2021.npy", end_a)
    np.save(labels / "area_b_lulc_2020.npy", start_b)
    np.save(labels / "area_b_lulc_2021.npy", end_b)
    np.save(embeddings / "area_a_emb_2021.npy", _toy_prob_embedding(end_a_probs))
    np.save(embeddings / "area_b_emb_2021.npy", _toy_prob_embedding(end_b_probs))

    rows = make_decoder_true_end_confidence_table(
        out_dir=output,
        decoder=ToyProbDecoder(),
        labels_dir=labels,
        embeddings_dir=embeddings,
        areas=["area_a", "area_b"],
        start_year=2020,
        end_year=2021,
    )

    assert rows == [
        {
            "area": "area_a",
            "true_end_class": 11,
            "n_pixels": 1,
            "mean_true_end_prob": 0.01,
            "median_true_end_prob": 0.01,
            "top_pred_class": 5,
            "top_pred_count": 1,
        },
        {
            "area": "area_b",
            "true_end_class": 11,
            "n_pixels": 2,
            "mean_true_end_prob": 0.55,
            "median_true_end_prob": 0.55,
            "top_pred_class": 11,
            "top_pred_count": 2,
        },
    ]
    assert (output / "batch2_decoder_true_end_confidence_by_area.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "area,true_end_class,n_pixels,mean_true_end_prob,median_true_end_prob,top_pred_class,top_pred_count"
    )


def test_make_decoder_true_end_confidence_table_accepts_custom_output_filename(tmp_path: Path):
    labels = tmp_path / "labels"
    embeddings = tmp_path / "embeddings"
    output = tmp_path / "diagnostics"
    labels.mkdir()
    embeddings.mkdir()

    start = np.array([[5, 5]], dtype=np.int32)
    end = np.array([[11, 5]], dtype=np.int32)
    end_probs = np.array([[[0.90, 0.09, 0.01], [0.80, 0.15, 0.05]]], dtype=np.float32)

    np.save(labels / "toy_lulc_2020.npy", start)
    np.save(labels / "toy_lulc_2021.npy", end)
    np.save(embeddings / "toy_emb_2021.npy", _toy_prob_embedding(end_probs))

    make_decoder_true_end_confidence_table(
        out_dir=output,
        decoder=ToyProbDecoder(),
        labels_dir=labels,
        embeddings_dir=embeddings,
        areas=["toy"],
        output_filename="batch5_decoder_true_end_confidence_by_area.csv",
    )

    assert (output / "batch5_decoder_true_end_confidence_by_area.csv").exists()
    assert not (output / "batch2_decoder_true_end_confidence_by_area.csv").exists()


def test_make_forecast_transition_fate_table_compares_observed_and_forecast_probabilities(
    tmp_path: Path, monkeypatch
):
    import scripts.paper58_benchmark.make_batch2_diagnostics as diagnostics

    labels = tmp_path / "labels"
    embeddings = tmp_path / "embeddings"
    predictions = tmp_path / "predicted"
    output = tmp_path / "diagnostics"
    labels.mkdir()
    embeddings.mkdir()
    predictions.mkdir()

    start = np.array([[5, 5], [7, 7]], dtype=np.int32)
    end = np.array([[11, 7], [5, 11]], dtype=np.int32)
    pred = np.array([[5, 7], [5, 5]], dtype=np.int32)
    observed_end_probs = np.array(
        [
            [[0.90, 0.09, 0.01], [0.20, 0.70, 0.10]],
            [[0.80, 0.15, 0.05], [0.60, 0.30, 0.10]],
        ],
        dtype=np.float32,
    )
    forecast_end_probs = np.array(
        [
            [[0.97, 0.025, 0.005], [0.30, 0.65, 0.05]],
            [[0.85, 0.10, 0.05], [0.88, 0.115, 0.005]],
        ],
        dtype=np.float32,
    )

    np.save(labels / "toy_lulc_2020.npy", start)
    np.save(labels / "toy_lulc_2021.npy", end)
    np.save(embeddings / "toy_emb_2021.npy", _toy_prob_embedding(observed_end_probs))
    np.save(predictions / "toy_lulc_pred_2020_2021.npy", pred)
    monkeypatch.setattr(
        diagnostics,
        "_predict_embedding_for_area",
        lambda model, area, start_year, embeddings_dir: _toy_prob_embedding(forecast_end_probs),
    )

    rows = make_forecast_transition_fate_table(
        out_dir=output,
        decoder=ToyProbDecoder(),
        model=object(),
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
            "observed_end_top": "5:1",
            "forecast_end_top": "5:1",
            "observed_mean_true_end_prob": 0.01,
            "observed_median_true_end_prob": 0.01,
            "forecast_mean_true_end_prob": 0.005,
            "forecast_median_true_end_prob": 0.005,
            "mean_true_end_prob_delta": -0.005,
            "observed_top_mean_prob_class": 5,
            "observed_top_mean_prob": 0.9,
            "forecast_top_mean_prob_class": 5,
            "forecast_top_mean_prob": 0.97,
        },
        "5->7": {
            "true_transition": "5->7",
            "n_true_pixels": 1,
            "observed_end_top": "7:1",
            "forecast_end_top": "7:1",
            "observed_mean_true_end_prob": 0.7,
            "observed_median_true_end_prob": 0.7,
            "forecast_mean_true_end_prob": 0.65,
            "forecast_median_true_end_prob": 0.65,
            "mean_true_end_prob_delta": -0.05,
            "observed_top_mean_prob_class": 7,
            "observed_top_mean_prob": 0.7,
            "forecast_top_mean_prob_class": 7,
            "forecast_top_mean_prob": 0.65,
        },
        "7->5": {
            "true_transition": "7->5",
            "n_true_pixels": 1,
            "observed_end_top": "5:1",
            "forecast_end_top": "5:1",
            "observed_mean_true_end_prob": 0.8,
            "observed_median_true_end_prob": 0.8,
            "forecast_mean_true_end_prob": 0.85,
            "forecast_median_true_end_prob": 0.85,
            "mean_true_end_prob_delta": 0.05,
            "observed_top_mean_prob_class": 5,
            "observed_top_mean_prob": 0.8,
            "forecast_top_mean_prob_class": 5,
            "forecast_top_mean_prob": 0.85,
        },
        "7->11": {
            "true_transition": "7->11",
            "n_true_pixels": 1,
            "observed_end_top": "5:1",
            "forecast_end_top": "5:1",
            "observed_mean_true_end_prob": 0.1,
            "observed_median_true_end_prob": 0.1,
            "forecast_mean_true_end_prob": 0.005,
            "forecast_median_true_end_prob": 0.005,
            "mean_true_end_prob_delta": -0.095,
            "observed_top_mean_prob_class": 5,
            "observed_top_mean_prob": 0.6,
            "forecast_top_mean_prob_class": 5,
            "forecast_top_mean_prob": 0.88,
        },
    }
    assert (output / "toy_forecast_transition_fate.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "true_transition,n_true_pixels,observed_end_top,forecast_end_top,"
        "observed_mean_true_end_prob,observed_median_true_end_prob,"
        "forecast_mean_true_end_prob,forecast_median_true_end_prob,mean_true_end_prob_delta,"
        "observed_top_mean_prob_class,observed_top_mean_prob,"
        "forecast_top_mean_prob_class,forecast_top_mean_prob"
    )


def test_make_forecast_true_end_confidence_table_compares_observed_and_forecast_by_area(
    tmp_path: Path, monkeypatch
):
    import scripts.paper58_benchmark.make_batch2_diagnostics as diagnostics

    labels = tmp_path / "labels"
    embeddings = tmp_path / "embeddings"
    output = tmp_path / "diagnostics"
    labels.mkdir()
    embeddings.mkdir()

    start = np.array([[5, 5], [7, 7]], dtype=np.int32)
    end = np.array([[11, 7], [5, 11]], dtype=np.int32)
    observed_end_probs = np.array(
        [
            [[0.90, 0.09, 0.01], [0.20, 0.70, 0.10]],
            [[0.80, 0.15, 0.05], [0.60, 0.30, 0.10]],
        ],
        dtype=np.float32,
    )
    forecast_end_probs = np.array(
        [
            [[0.97, 0.025, 0.005], [0.30, 0.65, 0.05]],
            [[0.85, 0.10, 0.05], [0.88, 0.115, 0.005]],
        ],
        dtype=np.float32,
    )

    np.save(labels / "toy_lulc_2020.npy", start)
    np.save(labels / "toy_lulc_2021.npy", end)
    np.save(embeddings / "toy_emb_2021.npy", _toy_prob_embedding(observed_end_probs))
    monkeypatch.setattr(
        diagnostics,
        "_predict_embedding_for_area",
        lambda model, area, start_year, embeddings_dir: _toy_prob_embedding(forecast_end_probs),
    )

    rows = make_forecast_true_end_confidence_table(
        out_dir=output,
        decoder=ToyProbDecoder(),
        model=object(),
        labels_dir=labels,
        embeddings_dir=embeddings,
        areas=["toy"],
        start_year=2020,
        end_year=2021,
    )

    assert rows == [
        {
            "area": "toy",
            "true_end_class": 5,
            "n_pixels": 1,
            "observed_mean_true_end_prob": 0.8,
            "observed_median_true_end_prob": 0.8,
            "forecast_mean_true_end_prob": 0.85,
            "forecast_median_true_end_prob": 0.85,
            "mean_true_end_prob_delta": 0.05,
            "observed_top_pred_class": 5,
            "observed_top_pred_count": 1,
            "forecast_top_pred_class": 5,
            "forecast_top_pred_count": 1,
        },
        {
            "area": "toy",
            "true_end_class": 7,
            "n_pixels": 1,
            "observed_mean_true_end_prob": 0.7,
            "observed_median_true_end_prob": 0.7,
            "forecast_mean_true_end_prob": 0.65,
            "forecast_median_true_end_prob": 0.65,
            "mean_true_end_prob_delta": -0.05,
            "observed_top_pred_class": 7,
            "observed_top_pred_count": 1,
            "forecast_top_pred_class": 7,
            "forecast_top_pred_count": 1,
        },
        {
            "area": "toy",
            "true_end_class": 11,
            "n_pixels": 2,
            "observed_mean_true_end_prob": 0.055,
            "observed_median_true_end_prob": 0.055,
            "forecast_mean_true_end_prob": 0.005,
            "forecast_median_true_end_prob": 0.005,
            "mean_true_end_prob_delta": -0.05,
            "observed_top_pred_class": 5,
            "observed_top_pred_count": 2,
            "forecast_top_pred_class": 5,
            "forecast_top_pred_count": 2,
        },
    ]
    assert (
        output / "batch2_forecast_true_end_confidence_by_area.csv"
    ).read_text(encoding="utf-8").splitlines()[0] == (
        "area,true_end_class,n_pixels,"
        "observed_mean_true_end_prob,observed_median_true_end_prob,"
        "forecast_mean_true_end_prob,forecast_median_true_end_prob,mean_true_end_prob_delta,"
        "observed_top_pred_class,observed_top_pred_count,"
        "forecast_top_pred_class,forecast_top_pred_count"
    )


def test_make_shifted_transition_fate_table_tracks_best_shifted_transition_destinations(tmp_path: Path):
    labels = tmp_path / "labels"
    predictions = tmp_path / "predicted"
    output = tmp_path / "diagnostics"
    labels.mkdir()
    predictions.mkdir()

    start = np.array(
        [
            [5, 5, 5],
            [5, 5, 5],
            [5, 5, 5],
        ],
        dtype=np.int32,
    )
    end = start.copy()
    end[2, 2] = 11
    pred = start.copy()
    pred[1, 1] = 11
    np.save(labels / "toy_lulc_2020.npy", start)
    np.save(labels / "toy_lulc_2021.npy", end)
    np.save(predictions / "toy_lulc_pred_2020_2021.npy", pred)

    rows = make_shifted_transition_fate_table(
        out_dir=output,
        labels_dir=labels,
        predictions_dir=predictions,
        area="toy",
        start_year=2020,
        end_year=2021,
        shift_dy=1,
        shift_dx=1,
        top_n_true_transitions=3,
    )

    assert rows == [
        {
            "true_transition": "5->11",
            "n_true_pixels": 1,
            "raw_model_end_top": "5:1",
            "shifted_model_end_top": "11:1",
            "raw_match_pixels": 0,
            "shifted_match_pixels": 1,
        }
    ]
    assert (output / "toy_shifted_transition_fate.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "true_transition,n_true_pixels,raw_model_end_top,shifted_model_end_top,raw_match_pixels,shifted_match_pixels"
    )
