import json
from pathlib import Path

import numpy as np

from scripts.paper58_benchmark.holdouts import DEFAULT_HOLDOUT_MANIFEST
from scripts.paper58_benchmark.schema import (
    AREA_STRATA,
    DEVELOPMENT_AREAS,
    BenchmarkRow,
    assign_tier,
    row_to_dict,
)


def test_assign_tier_marks_development_areas_as_tier2():
    assert assign_tier("bishan") == "tier2"
    assert assign_tier("banzhucun") == "tier2"
    assert assign_tier("heping") == "tier2"


def test_assign_tier_requires_manifest_cleared_no_contact_status():
    assert assign_tier("strict_holdout", development_contact_status="none") == "tier1"
    assert assign_tier("strict_holdout", development_contact_status="uncertain") == "review_required"
    assert assign_tier("poyang_lake", development_contact_status="none") == "tier2"
    assert assign_tier("bishan", development_contact_status="none") == "tier2"


def test_benchmark_row_serializes_paths_and_metrics():
    row = BenchmarkRow(
        area="poyang_lake",
        start_year=2020,
        end_year=2021,
        tier="tier1",
        stratum=AREA_STRATA["poyang_lake"],
        bbox=(116.0, 29.0, 116.1, 29.1),
        data_source="ESRI_LULC_10m_and_AlphaEarth",
        development_contact_status="none",
        contact_evidence="toy external row",
        expected_role="positive_change_candidate",
        label_start_path=Path("labels/poyang_lake_lulc_2020.npy"),
        label_end_path=Path("labels/poyang_lake_lulc_2021.npy"),
        prediction_path=Path("predicted/poyang_lake_lulc_pred_2020_2021.npy"),
        embedding_start_path=Path("embeddings/poyang_lake_emb_2020.npy"),
        embedding_end_path=Path("embeddings/poyang_lake_emb_2021.npy"),
        context_path=Path("embeddings/poyang_lake_context.npy"),
        label_shape=(23, 23),
        prediction_shape=(23, 23),
        embedding_shape=(23, 23, 64),
        n_pixels=529,
        true_change_pixels=92,
        true_change_pct=92 / 529,
        qc_status="include",
        excluded_reason="",
    )

    data = row_to_dict(row)

    assert DEVELOPMENT_AREAS == {"banzhucun", "bishan", "heping"}
    assert data["area"] == "poyang_lake"
    assert data["tier"] == "tier1"
    assert data["bbox"] == [116.0, 29.0, 116.1, 29.1]
    assert data["development_contact_status"] == "none"
    assert data["expected_role"] == "positive_change_candidate"
    assert data["label_start_path"] == "labels/poyang_lake_lulc_2020.npy"
    assert data["label_shape"] == [23, 23]
    assert data["embedding_shape"] == [23, 23, 64]
    assert data["true_change_pixels"] == 92


from scripts.paper58_benchmark.build_registry import (
    build_registry,
    parse_label_filename,
    parse_prediction_filename,
)


def test_parse_label_and_prediction_filenames():
    assert parse_label_filename("toy_area_lulc_2020.npy") == ("toy_area", 2020)
    assert parse_prediction_filename("toy_area_lulc_pred_2020_2021.npy") == ("toy_area", 2020, 2021)
    assert parse_label_filename("bad.npy") is None
    assert parse_prediction_filename("bad.npy") is None


def test_build_registry_includes_valid_pair_and_missing_embedding(tmp_path: Path):
    labels = tmp_path / "labels"
    predicted = tmp_path / "predicted"
    embeddings = tmp_path / "embeddings"
    experiment_data = tmp_path / "experiment_data"
    output = tmp_path / "out"
    labels.mkdir()
    predicted.mkdir()
    embeddings.mkdir()
    experiment_data.mkdir()

    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
    np.save(labels / "external_lulc_2020.npy", start)
    np.save(labels / "external_lulc_2021.npy", end)
    np.save(predicted / "external_lulc_pred_2020_2021.npy", pred)
    np.save(embeddings / "external_emb_2020.npy", np.zeros((2, 2, 64), dtype=np.float32))
    np.save(embeddings / "external_emb_2021.npy", np.ones((2, 2, 64), dtype=np.float32))
    np.save(embeddings / "external_context.npy", np.zeros((2, 2, 2), dtype=np.float32))
    np.save(labels / "missing_emb_lulc_2020.npy", start)
    np.save(labels / "missing_emb_lulc_2021.npy", end)
    np.save(predicted / "missing_emb_lulc_pred_2020_2021.npy", pred)

    rows = build_registry(
        labels_dir=labels,
        predictions_dir=predicted,
        independent_embeddings_dir=embeddings,
        experiment_data_dir=experiment_data,
        output_dir=output,
    )

    assert len(rows) == 2
    by_area = {row.area: row for row in rows}
    row = by_area["external"]
    assert row.area == "external"
    assert row.tier == "review_required"
    assert row.qc_status == "include"
    assert row.true_change_pixels == 2
    assert row.embedding_shape == (2, 2, 64)
    missing_embedding_row = by_area["missing_emb"]
    assert missing_embedding_row.qc_status == "exclude"
    assert missing_embedding_row.excluded_reason == "missing_embedding"
    assert (output / "benchmark_registry.json").exists()
    assert (output / "benchmark_registry.csv").exists()


def test_build_registry_excludes_shape_mismatch_and_marks_zero_change_control(tmp_path: Path):
    labels = tmp_path / "labels"
    predicted = tmp_path / "predicted"
    embeddings = tmp_path / "embeddings"
    experiment_data = tmp_path / "experiment_data"
    output = tmp_path / "out"
    for path in (labels, predicted, embeddings, experiment_data):
        path.mkdir()

    np.save(labels / "mismatch_lulc_2020.npy", np.ones((2, 2), dtype=np.int32))
    np.save(labels / "mismatch_lulc_2021.npy", np.ones((2, 2), dtype=np.int32))
    np.save(predicted / "mismatch_lulc_pred_2020_2021.npy", np.ones((3, 3), dtype=np.int32))
    np.save(labels / "steady_lulc_2020.npy", np.ones((2, 2), dtype=np.int32))
    np.save(labels / "steady_lulc_2021.npy", np.ones((2, 2), dtype=np.int32))
    np.save(predicted / "steady_lulc_pred_2020_2021.npy", np.ones((2, 2), dtype=np.int32))

    rows = build_registry(
        labels_dir=labels,
        predictions_dir=predicted,
        independent_embeddings_dir=embeddings,
        experiment_data_dir=experiment_data,
        output_dir=output,
    )

    by_area = {row.area: row for row in rows}
    assert by_area["mismatch"].qc_status == "exclude"
    assert by_area["mismatch"].excluded_reason == "label_prediction_shape_mismatch"
    assert by_area["steady"].qc_status == "negative_control"
    assert by_area["steady"].excluded_reason == "zero_reference_change"


def test_build_registry_includes_embeddings_found_under_experiment_data_prithvi(tmp_path: Path):
    labels = tmp_path / "labels"
    predicted = tmp_path / "predicted"
    embeddings = tmp_path / "embeddings"
    experiment_data = tmp_path / "experiment_data"
    prithvi = experiment_data / "prithvi"
    output = tmp_path / "out"
    for path in (labels, predicted, embeddings, experiment_data, prithvi):
        path.mkdir()

    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
    np.save(labels / "prithvi_only_lulc_2020.npy", start)
    np.save(labels / "prithvi_only_lulc_2021.npy", end)
    np.save(predicted / "prithvi_only_lulc_pred_2020_2021.npy", pred)
    np.save(prithvi / "prithvi_only_emb_2020.npy", np.zeros((2, 2, 64), dtype=np.float32))
    np.save(prithvi / "prithvi_only_emb_2021.npy", np.ones((2, 2, 64), dtype=np.float32))

    rows = build_registry(
        labels_dir=labels,
        predictions_dir=predicted,
        independent_embeddings_dir=embeddings,
        experiment_data_dir=experiment_data,
        output_dir=output,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.qc_status == "include"
    assert row.embedding_start_path == prithvi / "prithvi_only_emb_2020.npy"
    assert row.embedding_end_path == prithvi / "prithvi_only_emb_2021.npy"


def test_build_registry_uses_manifest_provenance_for_tiers(tmp_path: Path):
    labels = tmp_path / "labels"
    predicted = tmp_path / "predicted"
    embeddings = tmp_path / "embeddings"
    experiment_data = tmp_path / "experiment_data"
    output = tmp_path / "out"
    for path in (labels, predicted, embeddings, experiment_data):
        path.mkdir()

    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
    for area in ("strict_external", "uncertain_external", "poyang_lake"):
        np.save(labels / f"{area}_lulc_2020.npy", start)
        np.save(labels / f"{area}_lulc_2021.npy", end)
        np.save(predicted / f"{area}_lulc_pred_2020_2021.npy", pred)
        np.save(embeddings / f"{area}_emb_2020.npy", np.zeros((2, 2, 64), dtype=np.float32))
        np.save(embeddings / f"{area}_emb_2021.npy", np.ones((2, 2, 64), dtype=np.float32))

    manifest_path = tmp_path / "holdouts.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "areas": [
                    {
                        "area": "strict_external",
                        "bbox": [120.1, 30.1, 120.2, 30.2],
                        "stratum": "Urban",
                        "years": [2020, 2021],
                        "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                        "selection_reason": "toy strict holdout",
                        "development_contact_status": "none",
                        "contact_evidence": "toy no-contact evidence",
                        "expected_role": "positive_change_candidate",
                        "notes": "",
                    },
                    {
                        "area": "uncertain_external",
                        "bbox": [121.1, 31.1, 121.2, 31.2],
                        "stratum": "Mixed",
                        "years": [2020, 2021],
                        "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                        "selection_reason": "toy uncertain holdout",
                        "development_contact_status": "uncertain",
                        "contact_evidence": "toy uncertain evidence",
                        "expected_role": "review_only",
                        "notes": "",
                    },
                    {
                        "area": "poyang_lake",
                        "bbox": [116.0, 29.0, 116.1, 29.1],
                        "stratum": "Wetland",
                        "years": [2020, 2021],
                        "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                        "selection_reason": "training-list override example",
                        "development_contact_status": "none",
                        "contact_evidence": "toy manifest claims none but known training must override",
                        "expected_role": "provenance_audit",
                        "notes": "",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = build_registry(
        labels_dir=labels,
        predictions_dir=predicted,
        independent_embeddings_dir=embeddings,
        experiment_data_dir=experiment_data,
        output_dir=output,
        holdout_manifest_path=manifest_path,
    )

    by_area = {row.area: row for row in rows}
    assert by_area["strict_external"].tier == "tier1"
    assert by_area["strict_external"].bbox == (120.1, 30.1, 120.2, 30.2)
    assert by_area["uncertain_external"].tier == "review_required"
    assert by_area["poyang_lake"].tier == "tier2"
    assert by_area["poyang_lake"].development_contact_status == "none"
    assert DEFAULT_HOLDOUT_MANIFEST.exists()


def test_build_registry_requires_manifest_year_coverage_for_tier1(tmp_path: Path):
    labels = tmp_path / "labels"
    predicted = tmp_path / "predicted"
    embeddings = tmp_path / "embeddings"
    experiment_data = tmp_path / "experiment_data"
    output = tmp_path / "out"
    for path in (labels, predicted, embeddings, experiment_data):
        path.mkdir()

    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
    np.save(labels / "strict_external_lulc_2021.npy", start)
    np.save(labels / "strict_external_lulc_2022.npy", end)
    np.save(predicted / "strict_external_lulc_pred_2021_2022.npy", pred)
    np.save(embeddings / "strict_external_emb_2021.npy", np.zeros((2, 2, 64), dtype=np.float32))
    np.save(embeddings / "strict_external_emb_2022.npy", np.ones((2, 2, 64), dtype=np.float32))

    manifest_path = tmp_path / "holdouts.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "areas": [
                    {
                        "area": "strict_external",
                        "bbox": [120.1, 30.1, 120.2, 30.2],
                        "stratum": "Urban",
                        "years": [2020, 2021],
                        "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                        "selection_reason": "toy strict holdout",
                        "development_contact_status": "none",
                        "contact_evidence": "toy no-contact evidence",
                        "expected_role": "positive_change_candidate",
                        "notes": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = build_registry(
        labels_dir=labels,
        predictions_dir=predicted,
        independent_embeddings_dir=embeddings,
        experiment_data_dir=experiment_data,
        output_dir=output,
        holdout_manifest_path=manifest_path,
    )

    row = rows[0]
    assert row.area == "strict_external"
    assert row.tier == "review_required"
    assert row.development_contact_status == "uncertain"
    assert row.bbox is None


def test_build_registry_matches_manifest_area_case_insensitively(tmp_path: Path):
    labels = tmp_path / "labels"
    predicted = tmp_path / "predicted"
    embeddings = tmp_path / "embeddings"
    experiment_data = tmp_path / "experiment_data"
    output = tmp_path / "out"
    for path in (labels, predicted, embeddings, experiment_data):
        path.mkdir()

    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
    np.save(labels / "Strict_External_lulc_2020.npy", start)
    np.save(labels / "Strict_External_lulc_2021.npy", end)
    np.save(predicted / "Strict_External_lulc_pred_2020_2021.npy", pred)
    np.save(embeddings / "Strict_External_emb_2020.npy", np.zeros((2, 2, 64), dtype=np.float32))
    np.save(embeddings / "Strict_External_emb_2021.npy", np.ones((2, 2, 64), dtype=np.float32))

    manifest_path = tmp_path / "holdouts.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "areas": [
                    {
                        "area": "strict_external",
                        "bbox": [120.1, 30.1, 120.2, 30.2],
                        "stratum": "Urban",
                        "years": [2020, 2021],
                        "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                        "selection_reason": "toy strict holdout",
                        "development_contact_status": "none",
                        "contact_evidence": "toy no-contact evidence",
                        "expected_role": "positive_change_candidate",
                        "notes": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = build_registry(
        labels_dir=labels,
        predictions_dir=predicted,
        independent_embeddings_dir=embeddings,
        experiment_data_dir=experiment_data,
        output_dir=output,
        holdout_manifest_path=manifest_path,
    )

    row = rows[0]
    assert row.area == "Strict_External"
    assert row.tier == "tier1"
    assert row.bbox == (120.1, 30.1, 120.2, 30.2)
    assert row.development_contact_status == "none"
