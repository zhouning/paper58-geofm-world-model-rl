import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.rse_revision.evaluate_independent_change_validation import (
    _transition_prior_prediction,
    evaluate_independent_change_validation,
)
from scripts.rse_revision.fetch_independent_lulc_labels import fetch_independent_lulc_labels
from scripts.rse_revision.generate_change_validation_predictions import (
    generate_change_validation_predictions,
)


def _write_holdout_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "areas": [
                    {
                        "name": "poyang_lake",
                        "bbox": [116.0, 29.0, 116.1, 29.1],
                        "development_contact_status": "none",
                    },
                    {
                        "area": "ignored_area_key",
                        "bbox": [100.0, 20.0, 100.1, 20.1],
                        "development_contact_status": "none",
                    },
                    {
                        "name": "wuyi_mountain",
                        "bbox": [117.0, 27.0, 117.1, 27.1],
                        "development_contact_status": "active",
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_fetch_change_validation_embeddings_writes_area_grids_and_context(tmp_path: Path, monkeypatch):
    import scripts.rse_revision.fetch_change_validation_embeddings as fetcher

    calls = {"embeddings": [], "context": []}

    def fake_extract_embeddings(bbox, year, scale=500):
        calls["embeddings"].append({"bbox": bbox, "year": year, "scale": scale})
        return np.full((2, 3, 64), year - 2019, dtype=np.float32)

    def fake_extract_terrain_context(bbox, target_shape=None):
        calls["context"].append({"bbox": bbox, "target_shape": target_shape})
        return np.ones((2, 2, 3), dtype=np.float32)

    monkeypatch.setattr(fetcher, "extract_embeddings", fake_extract_embeddings)
    monkeypatch.setattr(fetcher, "extract_terrain_context", fake_extract_terrain_context)

    manifest = fetcher.fetch_change_validation_embeddings(
        areas=["poyang_lake"],
        years=[2020, 2021],
        output_dir=tmp_path / "embeddings",
        manifest_path=tmp_path / "embedding_manifest.json",
        scale=500,
    )

    assert manifest["status"] == "complete"
    assert manifest["n_records"] == 2
    assert manifest["n_context_records"] == 1
    assert np.load(tmp_path / "embeddings" / "poyang_lake_emb_2021.npy").shape == (2, 3, 64)
    assert np.load(tmp_path / "embeddings" / "poyang_lake_context.npy").shape == (2, 2, 3)
    assert calls["embeddings"][0]["bbox"] == [116.0, 29.0, 116.1, 29.1]
    assert calls["context"] == [{"bbox": [116.0, 29.0, 116.1, 29.1], "target_shape": (2, 3)}]


def test_evaluate_independent_change_validation_computes_transition_metrics(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    predictions_dir = tmp_path / "predicted"
    output_dir = tmp_path / "out"
    labels_dir.mkdir()
    predictions_dir.mkdir()

    true_start = np.array([[1, 1, 2], [2, 3, 3]])
    true_end = np.array([[1, 2, 2], [3, 3, 1]])
    pred_end = np.array([[1, 2, 2], [2, 3, 1]])

    np.save(labels_dir / "toy_lulc_2020.npy", true_start)
    np.save(labels_dir / "toy_lulc_2021.npy", true_end)
    np.save(predictions_dir / "toy_lulc_pred_2020_2021.npy", pred_end)

    summary = evaluate_independent_change_validation(
        labels_dir=labels_dir,
        predictions_dir=predictions_dir,
        output_dir=output_dir,
    )

    assert summary["n_area_year_pairs"] == 1
    assert summary["n_skipped_pairs"] == 0
    assert summary["mean_metrics"]["model_change_precision"] == pytest.approx(1.0)
    assert summary["mean_metrics"]["model_change_recall"] == pytest.approx(2 / 3)
    assert summary["mean_metrics"]["model_change_f1"] == pytest.approx(0.8)
    assert summary["mean_metrics"]["persistence_change_f1"] == pytest.approx(0.0)
    assert summary["mean_metrics"]["model_end_accuracy"] == pytest.approx(5 / 6)
    assert summary["mean_metrics"]["persistence_end_accuracy"] == pytest.approx(0.5)
    assert summary["mean_metrics"]["model_changed_pixel_accuracy"] == pytest.approx(2 / 3)
    assert summary["mean_metrics"]["persistence_changed_pixel_accuracy"] == pytest.approx(0.0)
    assert summary["mean_metrics"]["model_area_bias_mae"] == pytest.approx(1 / 9)
    assert summary["mean_metrics"]["shuffled_model_change_f1"] == pytest.approx(2 / 3)
    assert summary["mean_metrics"]["shuffled_model_end_accuracy"] == pytest.approx(0.5)
    assert summary["mean_metrics"]["shuffled_model_changed_pixel_accuracy"] == pytest.approx(1 / 3)

    summary_path = output_dir / "independent_change_validation_summary.json"
    by_area_path = output_dir / "independent_change_validation_by_area.csv"
    assert summary_path.exists()
    assert by_area_path.exists()

    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["n_area_year_pairs"] == 1
    with by_area_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["area"] == "toy"
    assert rows[0]["start_year"] == "2020"
    assert rows[0]["end_year"] == "2021"


def test_transition_prior_prediction_uses_label_only_training_distribution():
    target_start = np.ones((2, 4), dtype=np.int32)
    training_start = np.ones((2, 4), dtype=np.int32)
    training_end = np.array([[1, 2, 1, 2], [1, 2, 1, 2]], dtype=np.int32)

    prior = _transition_prior_prediction(
        target_start=target_start,
        training_pairs=[(training_start, training_end)],
        seed=7,
    )

    assert prior.shape == target_start.shape
    assert np.count_nonzero(prior == 1) == 4
    assert np.count_nonzero(prior == 2) == 4


def test_evaluate_independent_change_validation_reports_transition_prior_baseline(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    predictions_dir = tmp_path / "predicted"
    output_dir = tmp_path / "out"
    labels_dir.mkdir()
    predictions_dir.mkdir()

    np.save(labels_dir / "toy_lulc_2020.npy", np.ones((2, 4), dtype=np.int32))
    np.save(labels_dir / "toy_lulc_2021.npy", np.array([[1, 2, 1, 2], [1, 2, 1, 2]], dtype=np.int32))
    np.save(labels_dir / "reference_lulc_2020.npy", np.ones((2, 4), dtype=np.int32))
    np.save(labels_dir / "reference_lulc_2021.npy", np.array([[1, 2, 1, 2], [1, 2, 1, 2]], dtype=np.int32))
    np.save(predictions_dir / "toy_lulc_pred_2020_2021.npy", np.ones((2, 4), dtype=np.int32))

    summary = evaluate_independent_change_validation(
        labels_dir=labels_dir,
        predictions_dir=predictions_dir,
        output_dir=output_dir,
    )

    assert summary["mean_metrics"]["transition_prior_change_f1"] is not None
    assert summary["mean_metrics"]["transition_prior_end_accuracy"] is not None
    with (output_dir / "independent_change_validation_by_area.csv").open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert "transition_prior_change_f1" in header
    assert "transition_prior_end_accuracy" in header


def test_evaluate_independent_change_validation_reports_missing_predictions(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    predictions_dir = tmp_path / "predicted"
    output_dir = tmp_path / "out"
    labels_dir.mkdir()
    predictions_dir.mkdir()

    np.save(labels_dir / "toy_lulc_2020.npy", np.array([[1, 1], [2, 2]]))
    np.save(labels_dir / "toy_lulc_2021.npy", np.array([[1, 2], [2, 3]]))

    summary = evaluate_independent_change_validation(
        labels_dir=labels_dir,
        predictions_dir=predictions_dir,
        output_dir=output_dir,
    )

    assert summary["n_area_year_pairs"] == 0
    assert summary["n_skipped_pairs"] == 1
    assert summary["skipped_pairs"][0]["reason"] == "missing_prediction"
    assert summary["status"] == "missing_predictions"


def test_evaluate_independent_change_validation_uses_prediction_year_pairs(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    predictions_dir = tmp_path / "predicted"
    output_dir = tmp_path / "out"
    labels_dir.mkdir()
    predictions_dir.mkdir()

    for year, value in [(2017, 1), (2018, 2), (2019, 3), (2020, 4)]:
        np.save(labels_dir / f"toy_lulc_{year}.npy", np.full((2, 2), value, dtype=np.int32))
    np.save(predictions_dir / "toy_lulc_pred_2017_2020.npy", np.full((2, 2), 4, dtype=np.int32))

    summary = evaluate_independent_change_validation(
        labels_dir=labels_dir,
        predictions_dir=predictions_dir,
        output_dir=output_dir,
    )

    assert summary["n_area_year_pairs"] == 1
    assert summary["n_skipped_pairs"] == 0
    assert summary["pairs"][0]["start_year"] == 2017
    assert summary["pairs"][0]["end_year"] == 2020
    assert summary["mean_metrics"]["model_end_accuracy"] == pytest.approx(1.0)


def test_fetch_independent_lulc_labels_writes_cache_and_manifest(tmp_path: Path, monkeypatch):
    import scripts.rse_revision.fetch_independent_lulc_labels as fetcher

    def fake_extract_lulc_labels(bbox, year, scale=10):
        return np.full((2, 3), year - 2019, dtype=np.int32)

    monkeypatch.setattr(fetcher, "extract_lulc_labels", fake_extract_lulc_labels)

    output_dir = tmp_path / "labels"
    manifest_path = tmp_path / "label_manifest.json"
    manifest = fetch_independent_lulc_labels(
        areas=["wuyi_mountain"],
        years=[2020, 2021],
        output_dir=output_dir,
        manifest_path=manifest_path,
    )

    assert manifest["status"] == "complete"
    assert manifest["n_records"] == 2
    assert manifest_path.exists()
    arr = np.load(output_dir / "wuyi_mountain_lulc_2021.npy")
    assert arr.shape == (2, 3)
    assert arr[0, 0] == 2


def test_fetch_independent_lulc_labels_can_use_fixed_scale_extractor(tmp_path: Path, monkeypatch):
    import scripts.rse_revision.fetch_independent_lulc_labels as fetcher

    calls = []

    def fake_fixed_scale_extractor(bbox, year, scale=10):
        calls.append({"bbox": bbox, "year": year, "scale": scale})
        return np.full((4, 5), year - 2019, dtype=np.int32)

    monkeypatch.setattr(fetcher, "_extract_lulc_labels_fixed_scale", fake_fixed_scale_extractor)

    manifest = fetch_independent_lulc_labels(
        areas=["bishan"],
        years=[2020],
        output_dir=tmp_path / "labels",
        manifest_path=tmp_path / "manifest.json",
        scale=500,
        fixed_scale=True,
    )

    assert manifest["scale_mode"] == "fixed"
    assert manifest["records"][0]["shape"] == [4, 5]
    assert calls == [{"bbox": [106.02, 29.38, 106.33, 29.68], "year": 2020, "scale": 500}]


def test_fetch_independent_lulc_labels_uses_cache_aligned_heping_bbox(tmp_path: Path, monkeypatch):
    import scripts.rse_revision.fetch_independent_lulc_labels as fetcher

    calls = []

    def fake_fixed_scale_extractor(bbox, year, scale=10):
        calls.append({"bbox": bbox, "year": year, "scale": scale})
        return np.ones((516, 356), dtype=np.int32)

    monkeypatch.setattr(fetcher, "_extract_lulc_labels_fixed_scale", fake_fixed_scale_extractor)

    manifest = fetch_independent_lulc_labels(
        areas=["heping"],
        years=[2020],
        output_dir=tmp_path / "labels",
        manifest_path=tmp_path / "manifest.json",
        scale=10,
        fixed_scale=True,
    )

    assert manifest["records"][0]["shape"] == [516, 356]
    assert calls == [{"bbox": [106.1133, 29.5997, 106.1452, 29.646], "year": 2020, "scale": 10}]


def test_fetch_independent_lulc_labels_reads_manifest_area(tmp_path: Path, monkeypatch):
    import scripts.rse_revision.fetch_independent_lulc_labels as fetcher

    calls = []

    def fake_extract_lulc_labels(bbox, year, scale=10):
        calls.append({"bbox": bbox, "year": year, "scale": scale})
        return np.full((2, 3), year - 2019, dtype=np.int32)

    monkeypatch.setattr(fetcher, "extract_lulc_labels", fake_extract_lulc_labels)

    manifest_path = tmp_path / "holdout_manifest.json"
    _write_holdout_manifest(manifest_path)

    manifest = fetch_independent_lulc_labels(
        areas=["poyang_lake"],
        years=[2020],
        output_dir=tmp_path / "labels",
        manifest_path=tmp_path / "label_manifest.json",
        area_manifest_path=manifest_path,
    )

    assert manifest["n_records"] == 1
    assert calls == [{"bbox": [116.0, 29.0, 116.1, 29.1], "year": 2020, "scale": 10}]
    assert np.load(tmp_path / "labels" / "poyang_lake_lulc_2020.npy").shape == (2, 3)


def test_fetch_change_validation_embeddings_reads_manifest_area(tmp_path: Path, monkeypatch):
    import scripts.rse_revision.fetch_change_validation_embeddings as fetcher

    calls = {"embeddings": [], "context": []}

    def fake_extract_embeddings(bbox, year, scale=500):
        calls["embeddings"].append({"bbox": bbox, "year": year, "scale": scale})
        return np.full((2, 3, 64), year - 2019, dtype=np.float32)

    def fake_extract_terrain_context(bbox, target_shape=None):
        calls["context"].append({"bbox": bbox, "target_shape": target_shape})
        return np.ones((2, 2, 3), dtype=np.float32)

    monkeypatch.setattr(fetcher, "extract_embeddings", fake_extract_embeddings)
    monkeypatch.setattr(fetcher, "extract_terrain_context", fake_extract_terrain_context)

    manifest_path = tmp_path / "holdout_manifest.json"
    _write_holdout_manifest(manifest_path)

    manifest = fetcher.fetch_change_validation_embeddings(
        areas=["poyang_lake"],
        years=[2020, 2021],
        output_dir=tmp_path / "embeddings",
        manifest_path=tmp_path / "embedding_manifest.json",
        area_manifest_path=manifest_path,
        scale=500,
    )

    assert manifest["n_records"] == 2
    assert manifest["n_context_records"] == 1
    assert calls["embeddings"][0]["bbox"] == [116.0, 29.0, 116.1, 29.1]
    assert calls["context"] == [{"bbox": [116.0, 29.0, 116.1, 29.1], "target_shape": (2, 3)}]


def test_generate_change_validation_predictions_filters_manifest_area(tmp_path: Path, monkeypatch):
    import scripts.rse_revision.generate_change_validation_predictions as predictor

    embedding_dir = tmp_path / "embeddings"
    embedding_dir.mkdir()
    output_dir = tmp_path / "predicted"
    report_path = tmp_path / "prediction_readiness_report.json"
    manifest_path = tmp_path / "holdout_manifest.json"
    _write_holdout_manifest(manifest_path)
    weights_path = tmp_path / "weights.pt"
    decoder_path = tmp_path / "decoder.pkl"
    weights_path.write_text("stub", encoding="utf-8")
    decoder_path.write_text("stub", encoding="utf-8")

    np.save(embedding_dir / "poyang_lake_emb_2020.npy", np.zeros((2, 2, 64), dtype=np.float32))
    np.save(embedding_dir / "poyang_lake_emb_2021.npy", np.zeros((2, 2, 64), dtype=np.float32))
    np.save(embedding_dir / "wuyi_mountain_emb_2020.npy", np.zeros((2, 2, 64), dtype=np.float32))
    np.save(embedding_dir / "wuyi_mountain_emb_2021.npy", np.zeros((2, 2, 64), dtype=np.float32))

    class FakeModel:
        def __call__(self, z, scenario_t, context_t):
            return z

    class FakeDecoder:
        def predict(self, arr):
            return np.ones(arr.shape[0], dtype=np.int32)

    monkeypatch.setattr(predictor, "_load_model", lambda weights_path: FakeModel())
    monkeypatch.setattr(predictor, "_load_decoder", lambda path: FakeDecoder())
    monkeypatch.setattr(predictor, "_predict_next_embedding", lambda model, emb, context: emb)

    report = generate_change_validation_predictions(
        embedding_dirs=[embedding_dir],
        output_dir=output_dir,
        report_path=report_path,
        weights_path=weights_path,
        decoder_path=decoder_path,
        area_manifest_path=manifest_path,
    )

    assert report["status"] == "complete"
    assert report["n_predictions"] == 1
    assert report["records"][0]["area"] == "poyang_lake"
    assert (output_dir / "poyang_lake_lulc_pred_2020_2021.npy").exists()
    assert not (output_dir / "wuyi_mountain_lulc_pred_2020_2021.npy").exists()


def test_generate_change_validation_predictions_reports_missing_model_artifacts(tmp_path: Path):
    embedding_dir = tmp_path / "embeddings"
    output_dir = tmp_path / "predicted"
    report_path = tmp_path / "prediction_readiness_report.json"
    embedding_dir.mkdir()
    np.save(embedding_dir / "toy_emb_2020.npy", np.zeros((2, 2, 64), dtype=np.float32))
    np.save(embedding_dir / "toy_emb_2021.npy", np.zeros((2, 2, 64), dtype=np.float32))

    report = generate_change_validation_predictions(
        embedding_dirs=[embedding_dir],
        output_dir=output_dir,
        report_path=report_path,
        weights_path=tmp_path / "missing_weights.pt",
        decoder_path=tmp_path / "missing_decoder.pkl",
    )

    assert report["status"] == "not_ready"
    assert report["n_predictions"] == 0
    assert {item["component"] for item in report["readiness_failures"]} == {
        "latent_dynamics_weights",
        "lulc_decoder",
    }
    assert report_path.exists()
