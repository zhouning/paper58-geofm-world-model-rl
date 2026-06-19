from pathlib import Path

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


def test_assign_tier_marks_non_development_areas_as_tier1():
    assert assign_tier("poyang_lake") == "tier1"
    assert assign_tier("new_holdout_area") == "tier1"


def test_benchmark_row_serializes_paths_and_metrics():
    row = BenchmarkRow(
        area="poyang_lake",
        start_year=2020,
        end_year=2021,
        tier="tier1",
        stratum=AREA_STRATA["poyang_lake"],
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
    assert data["label_start_path"] == "labels/poyang_lake_lulc_2020.npy"
    assert data["label_shape"] == [23, 23]
    assert data["embedding_shape"] == [23, 23, 64]
    assert data["true_change_pixels"] == 92
