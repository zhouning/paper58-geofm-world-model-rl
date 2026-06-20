import json
from pathlib import Path

import pytest

from scripts.paper58_benchmark.holdouts import (
    DEFAULT_HOLDOUT_MANIFEST,
    HoldoutArea,
    area_records_for_status,
    load_holdout_manifest,
    manifest_lookup,
    tier_from_provenance,
)
from scripts.paper58_benchmark.build_combined_holdout_manifest import build_combined_holdout_manifest


def _write_manifest(path: Path, areas: list[dict]) -> None:
    path.write_text(json.dumps({"version": 1, "areas": areas}, indent=2), encoding="utf-8")


def test_load_holdout_manifest_parses_required_fields(tmp_path: Path):
    manifest_path = tmp_path / "holdouts.json"
    _write_manifest(
        manifest_path,
        [
            {
                "area": "strict_urban_holdout",
                "bbox": [120.1, 30.1, 120.2, 30.2],
                "stratum": "Urban",
                "years": [2020, 2021],
                "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                "selection_reason": "external urban holdout",
                "development_contact_status": "none",
                "contact_evidence": "not listed in Paper58 training or development areas",
                "expected_role": "positive_change_candidate",
                "notes": "small independent bbox",
            }
        ],
    )

    areas = load_holdout_manifest(manifest_path)

    assert areas == [
        HoldoutArea(
            area="strict_urban_holdout",
            bbox=(120.1, 30.1, 120.2, 30.2),
            stratum="Urban",
            years=(2020, 2021),
            data_source="ESRI_LULC_10m_and_AlphaEarth",
            selection_reason="external urban holdout",
            development_contact_status="none",
            contact_evidence="not listed in Paper58 training or development areas",
            expected_role="positive_change_candidate",
            notes="small independent bbox",
        )
    ]


def test_load_holdout_manifest_rejects_missing_or_invalid_fields(tmp_path: Path):
    manifest_path = tmp_path / "holdouts.json"
    _write_manifest(
        manifest_path,
        [
            {
                "area": "bad_holdout",
                "bbox": [120.1, 30.1, 120.2, 30.2],
                "stratum": "Urban",
                "years": [2020, 2021],
                "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                "selection_reason": "external urban holdout",
                "development_contact_status": "none",
                "expected_role": "positive_change_candidate",
                "notes": "missing contact evidence",
            }
        ],
    )

    with pytest.raises(ValueError, match="missing required field: contact_evidence"):
        load_holdout_manifest(manifest_path)

    _write_manifest(
        manifest_path,
        [
            {
                "area": "bad_status",
                "bbox": [120.1, 30.1, 120.2, 30.2],
                "stratum": "Urban",
                "years": [2020, 2021],
                "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                "selection_reason": "external urban holdout",
                "development_contact_status": "maybe",
                "contact_evidence": "invalid status",
                "expected_role": "positive_change_candidate",
                "notes": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="invalid development_contact_status"):
        load_holdout_manifest(manifest_path)


def test_load_holdout_manifest_rejects_boolean_bbox_values(tmp_path: Path):
    manifest_path = tmp_path / "holdouts.json"
    _write_manifest(
        manifest_path,
        [
            {
                "area": "bad_bbox_bool",
                "bbox": [120.1, True, 120.2, 30.2],
                "stratum": "Urban",
                "years": [2020, 2021],
                "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                "selection_reason": "invalid bbox example",
                "development_contact_status": "none",
                "contact_evidence": "bbox contains bool",
                "expected_role": "review_only",
                "notes": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="bbox must contain four numbers"):
        load_holdout_manifest(manifest_path)


def test_load_holdout_manifest_rejects_boolean_year_values(tmp_path: Path):
    manifest_path = tmp_path / "holdouts.json"
    _write_manifest(
        manifest_path,
        [
            {
                "area": "bad_year_bool",
                "bbox": [120.1, 30.1, 120.2, 30.2],
                "stratum": "Urban",
                "years": [2020, True],
                "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                "selection_reason": "invalid year example",
                "development_contact_status": "none",
                "contact_evidence": "years contains bool",
                "expected_role": "review_only",
                "notes": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="years must contain integers"):
        load_holdout_manifest(manifest_path)


def test_load_holdout_manifest_rejects_non_object_area_entries(tmp_path: Path):
    manifest_path = tmp_path / "holdouts.json"
    _write_manifest(manifest_path, ["not an object"])

    with pytest.raises(ValueError, match="holdout area 0 must be an object"):
        load_holdout_manifest(manifest_path)


def test_tier_from_provenance_blocks_training_development_and_uncertain_contact(tmp_path: Path):
    manifest_path = tmp_path / "holdouts.json"
    _write_manifest(
        manifest_path,
        [
            {
                "area": "strict_urban_holdout",
                "bbox": [120.1, 30.1, 120.2, 30.2],
                "stratum": "Urban",
                "years": [2020, 2021],
                "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                "selection_reason": "external urban holdout",
                "development_contact_status": "none",
                "contact_evidence": "not listed in Paper58 training or development areas",
                "expected_role": "positive_change_candidate",
                "notes": "",
            },
            {
                "area": "uncertain_holdout",
                "bbox": [121.1, 31.1, 121.2, 31.2],
                "stratum": "Mixed",
                "years": [2020, 2021],
                "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                "selection_reason": "uncertain provenance example",
                "development_contact_status": "uncertain",
                "contact_evidence": "provenance not cleared",
                "expected_role": "review_only",
                "notes": "",
            },
            {
                "area": "poyang_lake",
                "bbox": [116.0, 29.0, 116.1, 29.1],
                "stratum": "Wetland",
                "years": [2020, 2021],
                "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                "selection_reason": "current local benchmark row",
                "development_contact_status": "none",
                "contact_evidence": "listed here as none but training-list membership must override it",
                "expected_role": "provenance_audit",
                "notes": "",
            },
        ],
    )
    lookup = manifest_lookup(load_holdout_manifest(manifest_path))

    assert tier_from_provenance("strict_urban_holdout", lookup) == "tier1"
    assert tier_from_provenance("uncertain_holdout", lookup) == "review_required"
    assert tier_from_provenance("poyang_lake", lookup) == "tier2"
    assert tier_from_provenance("bishan", lookup) == "tier2"
    assert tier_from_provenance("unknown_not_in_manifest", lookup) == "review_required"


def test_area_records_for_status_respects_explicit_empty_status_set(tmp_path: Path):
    manifest_path = tmp_path / "holdouts.json"
    _write_manifest(
        manifest_path,
        [
            {
                "area": "strict_urban_holdout",
                "bbox": [120.1, 30.1, 120.2, 30.2],
                "stratum": "Urban",
                "years": [2020, 2021],
                "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                "selection_reason": "external urban holdout",
                "development_contact_status": "none",
                "contact_evidence": "not listed in Paper58 training or development areas",
                "expected_role": "positive_change_candidate",
                "notes": "",
            }
        ],
    )

    assert area_records_for_status(manifest_path, statuses=set()) == []


def test_repository_holdout_manifest_has_minimum_external_batch():
    areas = load_holdout_manifest(DEFAULT_HOLDOUT_MANIFEST)
    lookup = manifest_lookup(areas)
    tier1 = [area for area in areas if tier_from_provenance(area.area, lookup) == "tier1"]

    assert len(tier1) >= 6
    assert len({area.stratum for area in tier1}) >= 4
    assert all(2020 in area.years and 2021 in area.years for area in tier1)
    assert tier_from_provenance("poyang_lake", lookup) == "tier2"
    assert tier_from_provenance("wuyi_mountain", lookup) == "tier2"


def test_repository_batch2_holdout_manifest_has_target_batch_shape():
    manifest_path = DEFAULT_HOLDOUT_MANIFEST.parent / "paper58_holdout_areas_batch2.json"
    areas = load_holdout_manifest(manifest_path)
    lookup = manifest_lookup(areas)
    tier1 = [area for area in areas if tier_from_provenance(area.area, lookup) == "tier1"]

    assert len(areas) == 8
    assert len(tier1) == 8
    assert len({area.stratum for area in tier1}) >= 5
    assert all(area.years == (2020, 2021) for area in tier1)


def test_build_combined_holdout_manifest_merges_two_valid_manifests(tmp_path: Path):
    batch1 = tmp_path / "batch1.json"
    batch2 = tmp_path / "batch2.json"
    output = tmp_path / "combined.json"
    _write_manifest(
        batch1,
        [
            {
                "area": "batch1_area",
                "bbox": [120.1, 30.1, 120.2, 30.2],
                "stratum": "Urban",
                "years": [2020, 2021],
                "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                "selection_reason": "batch1 candidate",
                "development_contact_status": "none",
                "contact_evidence": "not listed in training or development areas",
                "expected_role": "positive_change_candidate",
                "notes": "",
            }
        ],
    )
    _write_manifest(
        batch2,
        [
            {
                "area": "batch2_area",
                "bbox": [121.1, 31.1, 121.2, 31.2],
                "stratum": "Wetland",
                "years": [2020, 2021],
                "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                "selection_reason": "batch2 candidate",
                "development_contact_status": "none",
                "contact_evidence": "not listed in training or development areas",
                "expected_role": "positive_change_candidate",
                "notes": "",
            }
        ],
    )

    combined = build_combined_holdout_manifest([batch1, batch2], output)

    assert [area.area for area in combined] == ["batch1_area", "batch2_area"]
    assert load_holdout_manifest(output) == combined


def test_build_combined_holdout_manifest_rejects_duplicate_area_names(tmp_path: Path):
    batch1 = tmp_path / "batch1.json"
    batch2 = tmp_path / "batch2.json"
    output = tmp_path / "combined.json"
    shared_row = {
        "area": "shared_area",
        "bbox": [120.1, 30.1, 120.2, 30.2],
        "stratum": "Urban",
        "years": [2020, 2021],
        "data_source": "ESRI_LULC_10m_and_AlphaEarth",
        "selection_reason": "duplicate candidate",
        "development_contact_status": "none",
        "contact_evidence": "not listed in training or development areas",
        "expected_role": "positive_change_candidate",
        "notes": "",
    }
    _write_manifest(batch1, [shared_row])
    _write_manifest(batch2, [shared_row])

    with pytest.raises(ValueError, match="duplicate combined holdout area: shared_area"):
        build_combined_holdout_manifest([batch1, batch2], output)
