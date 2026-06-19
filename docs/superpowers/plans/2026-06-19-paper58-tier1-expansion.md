# Paper58 Tier 1 Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a provenance-first Tier 1 expansion workflow, acquire genuinely external holdout evidence, rerun the Paper58 benchmark gate, and prevent manuscript strengthening unless the expanded gate passes.

**Architecture:** Keep the existing `scripts/rse_revision/` acquisition scripts and `scripts/paper58_benchmark/` evaluation package, but add one shared provenance manifest layer consumed by both. The registry becomes the source of truth for tier assignment, and Tier 1 is assigned only when the manifest proves `development_contact_status = none` and the area is absent from known training/development sets.

**Tech Stack:** Python 3, stdlib JSON/CSV/argparse/dataclasses, NumPy, pytest, existing GEE-backed acquisition helpers, existing Paper58 benchmark scripts.

---

## File Structure

Create these files:

- `scripts/paper58_benchmark/holdouts.py`: load, validate, and query holdout area manifests.
- `tests/test_paper58_benchmark_holdouts.py`: unit tests for manifest parsing and strict tier rules.
- `data/independent_change_labels/paper58_holdout_areas.json`: versioned candidate area manifest for current-contact and new external areas.
- `scripts/paper58_benchmark/audit_provenance.py`: write a registry-level provenance audit before and after expansion.
- `tests/test_paper58_benchmark_provenance_audit.py`: tests for the audit output.

Modify these files:

- `scripts/paper58_benchmark/schema.py`: add provenance fields and strict tier assignment.
- `scripts/paper58_benchmark/build_registry.py`: merge registry rows with holdout manifest metadata.
- `scripts/paper58_benchmark/evaluate_benchmark.py`: reject malformed Tier 1 rows whose provenance does not allow Tier 1.
- `scripts/rse_revision/fetch_independent_lulc_labels.py`: allow manifest-provided external bboxes.
- `scripts/rse_revision/fetch_change_validation_embeddings.py`: allow manifest-provided external bboxes.
- `scripts/rse_revision/generate_change_validation_predictions.py`: allow manifest-driven area filtering.
- `tests/test_paper58_benchmark_registry.py`: update row contract and strict Tier 1 tests.
- `tests/test_paper58_benchmark_evaluation.py`: update registry fixtures with provenance fields.
- `tests/test_rse_revision_change_validation.py`: add manifest-driven acquisition/prediction tests.

Do not modify the RSE manuscript or strengthen manuscript claims in this plan.

---

### Task 1: Holdout Manifest Loader And Strict Tier Rules

**Files:**
- Create: `scripts/paper58_benchmark/holdouts.py`
- Create: `tests/test_paper58_benchmark_holdouts.py`

- [ ] **Step 1: Write failing holdout loader tests**

Create `tests/test_paper58_benchmark_holdouts.py`:

```python
import json
from pathlib import Path

import pytest

from scripts.paper58_benchmark.holdouts import (
    HoldoutArea,
    load_holdout_manifest,
    manifest_lookup,
    tier_from_provenance,
)


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
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_holdouts.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.paper58_benchmark.holdouts'`.

- [ ] **Step 3: Implement the holdout loader**

Create `scripts/paper58_benchmark/holdouts.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOLDOUT_MANIFEST = ROOT / "data" / "independent_change_labels" / "paper58_holdout_areas.json"

VALID_CONTACT_STATUS = {"none", "known_contact", "uncertain"}
KNOWN_TRAINING_AREAS = {
    "yangtze_delta",
    "jing_jin_ji",
    "pearl_river",
    "chengdu_plain",
    "northeast_plain",
    "north_china_plain",
    "jianghan_plain",
    "hetao",
    "yunnan_eco",
    "daxinganling",
    "qinghai_edge",
    "wuyi_mountain",
    "guanzhong",
    "minnan_coast",
    "poyang_lake",
}
KNOWN_DEVELOPMENT_AREAS = {"banzhucun", "bishan", "heping"}
REQUIRED_FIELDS = {
    "area",
    "bbox",
    "stratum",
    "years",
    "data_source",
    "selection_reason",
    "development_contact_status",
    "contact_evidence",
    "expected_role",
    "notes",
}


@dataclass(frozen=True)
class HoldoutArea:
    area: str
    bbox: tuple[float, float, float, float]
    stratum: str
    years: tuple[int, ...]
    data_source: str
    selection_reason: str
    development_contact_status: str
    contact_evidence: str
    expected_role: str
    notes: str

    def as_area_record(self) -> dict[str, Any]:
        return {"name": self.area, "bbox": list(self.bbox)}


def _require_string(row: dict[str, Any], field: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ValueError(f"holdout area {index} field {field} must be a string")
    return value.strip()


def _parse_area(row: dict[str, Any], index: int) -> HoldoutArea:
    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        raise ValueError(f"holdout area {index} missing required field: {missing[0]}")

    area = _require_string(row, "area", index).lower()
    bbox_raw = row["bbox"]
    years_raw = row["years"]
    if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
        raise ValueError(f"holdout area {index} bbox must contain four numbers")
    if not all(isinstance(value, (int, float)) for value in bbox_raw):
        raise ValueError(f"holdout area {index} bbox must contain four numbers")
    if not isinstance(years_raw, list) or not years_raw:
        raise ValueError(f"holdout area {index} years must be a non-empty list")
    if not all(isinstance(value, int) for value in years_raw):
        raise ValueError(f"holdout area {index} years must contain integers")

    status = _require_string(row, "development_contact_status", index)
    if status not in VALID_CONTACT_STATUS:
        raise ValueError(f"holdout area {index} invalid development_contact_status: {status}")

    contact_evidence = _require_string(row, "contact_evidence", index)
    if not contact_evidence:
        raise ValueError(f"holdout area {index} contact_evidence must be non-empty")

    return HoldoutArea(
        area=area,
        bbox=tuple(float(value) for value in bbox_raw),
        stratum=_require_string(row, "stratum", index),
        years=tuple(sorted(set(int(value) for value in years_raw))),
        data_source=_require_string(row, "data_source", index),
        selection_reason=_require_string(row, "selection_reason", index),
        development_contact_status=status,
        contact_evidence=contact_evidence,
        expected_role=_require_string(row, "expected_role", index),
        notes=_require_string(row, "notes", index),
    )


def load_holdout_manifest(path: Path = DEFAULT_HOLDOUT_MANIFEST) -> list[HoldoutArea]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("holdout manifest must contain an object")
    areas = payload.get("areas")
    if not isinstance(areas, list):
        raise ValueError("holdout manifest must contain an 'areas' list")
    parsed = [_parse_area(row, index) for index, row in enumerate(areas)]
    seen: set[str] = set()
    for area in parsed:
        if area.area in seen:
            raise ValueError(f"duplicate holdout area: {area.area}")
        seen.add(area.area)
    return parsed


def manifest_lookup(areas: list[HoldoutArea]) -> dict[str, HoldoutArea]:
    return {area.area: area for area in areas}


def tier_from_provenance(area: str, lookup: dict[str, HoldoutArea]) -> str:
    normalized = area.lower()
    if normalized in KNOWN_TRAINING_AREAS or normalized in KNOWN_DEVELOPMENT_AREAS:
        return "tier2"
    record = lookup.get(normalized)
    if record is None:
        return "review_required"
    if record.development_contact_status == "none":
        return "tier1"
    if record.development_contact_status == "known_contact":
        return "tier2"
    return "review_required"


def area_records_for_status(
    manifest_path: Path = DEFAULT_HOLDOUT_MANIFEST,
    statuses: set[str] | None = None,
) -> list[dict[str, Any]]:
    wanted = statuses or {"none"}
    return [
        area.as_area_record()
        for area in load_holdout_manifest(manifest_path)
        if area.development_contact_status in wanted
    ]
```

- [ ] **Step 4: Run holdout tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_holdouts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add scripts/paper58_benchmark/holdouts.py tests/test_paper58_benchmark_holdouts.py
git commit -m "feat: add Paper58 holdout provenance loader"
```

---

### Task 2: Versioned Holdout Area Manifest

**Files:**
- Create: `data/independent_change_labels/paper58_holdout_areas.json`
- Test: `tests/test_paper58_benchmark_holdouts.py`

- [ ] **Step 1: Add a test for the repository manifest**

Append to `tests/test_paper58_benchmark_holdouts.py`:

```python
from scripts.paper58_benchmark.holdouts import DEFAULT_HOLDOUT_MANIFEST


def test_repository_holdout_manifest_has_minimum_external_batch():
    areas = load_holdout_manifest(DEFAULT_HOLDOUT_MANIFEST)
    lookup = manifest_lookup(areas)
    tier1 = [area for area in areas if tier_from_provenance(area.area, lookup) == "tier1"]

    assert len(tier1) >= 6
    assert len({area.stratum for area in tier1}) >= 4
    assert all(2020 in area.years and 2021 in area.years for area in tier1)
    assert tier_from_provenance("poyang_lake", lookup) == "tier2"
    assert tier_from_provenance("wuyi_mountain", lookup) == "tier2"
```

- [ ] **Step 2: Run the repository manifest test and verify it fails**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_holdouts.py::test_repository_holdout_manifest_has_minimum_external_batch -q
```

Expected: FAIL with `FileNotFoundError` for `paper58_holdout_areas.json`.

- [ ] **Step 3: Create the holdout area manifest**

Create `data/independent_change_labels/paper58_holdout_areas.json`:

```json
{
  "version": 1,
  "created": "2026-06-19",
  "purpose": "Paper58 provenance-first Tier 1 expansion candidates",
  "areas": [
    {
      "area": "poyang_lake",
      "bbox": [116.0, 29.0, 116.1, 29.1],
      "stratum": "Wetland",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "current local benchmark row requiring provenance downgrade",
      "development_contact_status": "known_contact",
      "contact_evidence": "area appears in DEFAULT_TRAINING_AREAS in src/adk_world_model/world_model.py",
      "expected_role": "provenance_audit",
      "notes": "do not count as strict Tier 1 without a separate training-contact audit"
    },
    {
      "area": "wuyi_mountain",
      "bbox": [117.6, 27.7, 117.7, 27.8],
      "stratum": "Forest",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "current local benchmark negative control requiring provenance downgrade",
      "development_contact_status": "known_contact",
      "contact_evidence": "area appears in DEFAULT_TRAINING_AREAS in src/adk_world_model/world_model.py",
      "expected_role": "negative_control",
      "notes": "current reference labels have zero change for 2020-2021"
    },
    {
      "area": "tianjin_binhai_holdout",
      "bbox": [117.45, 38.95, 117.55, 39.05],
      "stratum": "Urban",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "independent coastal urban expansion holdout not in Paper58 training list",
      "development_contact_status": "none",
      "contact_evidence": "area name and bbox are absent from DEFAULT_TRAINING_AREAS and known development areas",
      "expected_role": "positive_change_candidate",
      "notes": "first-batch external urban candidate"
    },
    {
      "area": "shenzhen_outer_holdout",
      "bbox": [114.25, 22.65, 114.35, 22.75],
      "stratum": "Urban",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "independent Pearl River urban-edge holdout outside the original pearl_river bbox",
      "development_contact_status": "none",
      "contact_evidence": "area name and bbox are absent from DEFAULT_TRAINING_AREAS and known development areas",
      "expected_role": "reserve_positive_change_candidate",
      "notes": "reserve urban candidate if Tianjin has low reference change"
    },
    {
      "area": "tarim_oasis_holdout",
      "bbox": [80.20, 41.10, 80.30, 41.20],
      "stratum": "Agriculture",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "independent irrigated oasis agriculture holdout outside existing agricultural bboxes",
      "development_contact_status": "none",
      "contact_evidence": "area name and bbox are absent from DEFAULT_TRAINING_AREAS and known development areas",
      "expected_role": "positive_change_candidate",
      "notes": "tests agricultural regime different from northeast/north_china/jianghan/hetao training areas"
    },
    {
      "area": "sanjiang_plain_holdout",
      "bbox": [133.20, 47.20, 133.30, 47.30],
      "stratum": "Agriculture",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "independent northeast agricultural-wetland margin holdout",
      "development_contact_status": "none",
      "contact_evidence": "area name and bbox are absent from DEFAULT_TRAINING_AREAS and known development areas",
      "expected_role": "reserve_positive_change_candidate",
      "notes": "reserve agriculture candidate if Tarim has low reference change"
    },
    {
      "area": "dongting_lake_holdout",
      "bbox": [112.80, 29.20, 112.90, 29.30],
      "stratum": "Wetland",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "independent wetland holdout outside the poyang_lake training bbox",
      "development_contact_status": "none",
      "contact_evidence": "area name and bbox are absent from DEFAULT_TRAINING_AREAS and known development areas",
      "expected_role": "positive_change_candidate",
      "notes": "tests whether wetland signal generalizes beyond current Poyang row"
    },
    {
      "area": "xiaoxinganling_holdout",
      "bbox": [128.80, 47.70, 128.90, 47.80],
      "stratum": "Forest",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "independent forest holdout outside daxinganling and wuyi training bboxes",
      "development_contact_status": "none",
      "contact_evidence": "area name and bbox are absent from DEFAULT_TRAINING_AREAS and known development areas",
      "expected_role": "positive_change_candidate",
      "notes": "forest candidate; may become zero-change negative control"
    },
    {
      "area": "qinling_mountain_holdout",
      "bbox": [108.40, 33.80, 108.50, 33.90],
      "stratum": "Forest",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "independent mountain forest holdout outside guanzhong training bbox",
      "development_contact_status": "none",
      "contact_evidence": "area name and bbox are absent from DEFAULT_TRAINING_AREAS and known development areas",
      "expected_role": "reserve_positive_change_candidate",
      "notes": "reserve forest candidate if Xiaoxinganling has zero reference change"
    },
    {
      "area": "haibei_plateau_holdout",
      "bbox": [100.90, 37.20, 101.00, 37.30],
      "stratum": "Plateau",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "independent plateau holdout outside qinghai_edge training bbox",
      "development_contact_status": "none",
      "contact_evidence": "area name and bbox are absent from DEFAULT_TRAINING_AREAS and known development areas",
      "expected_role": "positive_change_candidate",
      "notes": "plateau candidate; may be low-change and must be treated transparently"
    }
  ]
}
```

- [ ] **Step 4: Run holdout tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_holdouts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add data/independent_change_labels/paper58_holdout_areas.json tests/test_paper58_benchmark_holdouts.py
git commit -m "data: add Paper58 holdout candidate manifest"
```

---

### Task 3: Provenance Fields In Benchmark Schema And Registry

**Files:**
- Modify: `scripts/paper58_benchmark/schema.py`
- Modify: `scripts/paper58_benchmark/build_registry.py`
- Modify: `tests/test_paper58_benchmark_registry.py`

- [ ] **Step 1: Update registry tests for strict provenance**

Modify `tests/test_paper58_benchmark_registry.py` imports to include:

```python
import json

from scripts.paper58_benchmark.holdouts import DEFAULT_HOLDOUT_MANIFEST
```

Replace `test_assign_tier_marks_non_development_areas_as_tier1` with:

```python
def test_assign_tier_requires_manifest_cleared_no_contact_status():
    assert assign_tier("strict_holdout", development_contact_status="none") == "tier1"
    assert assign_tier("strict_holdout", development_contact_status="uncertain") == "review_required"
    assert assign_tier("poyang_lake", development_contact_status="none") == "tier2"
    assert assign_tier("bishan", development_contact_status="none") == "tier2"
```

Update `test_benchmark_row_serializes_paths_and_metrics` by adding these constructor fields:

```python
        bbox=(116.0, 29.0, 116.1, 29.1),
        data_source="ESRI_LULC_10m_and_AlphaEarth",
        development_contact_status="none",
        contact_evidence="toy external row",
        expected_role="positive_change_candidate",
```

Add these assertions:

```python
    assert data["bbox"] == [116.0, 29.0, 116.1, 29.1]
    assert data["development_contact_status"] == "none"
    assert data["expected_role"] == "positive_change_candidate"
```

Append this test:

```python
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
```

- [ ] **Step 2: Run registry tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_registry.py -q
```

Expected: FAIL because `BenchmarkRow` lacks provenance fields and `build_registry()` lacks `holdout_manifest_path`.

- [ ] **Step 3: Update schema**

In `scripts/paper58_benchmark/schema.py`:

- Import the strict provenance constants and helper:

```python
from scripts.paper58_benchmark.holdouts import (
    KNOWN_DEVELOPMENT_AREAS,
    KNOWN_TRAINING_AREAS,
)
```

- Replace `DEVELOPMENT_AREAS` with:

```python
DEVELOPMENT_AREAS = KNOWN_DEVELOPMENT_AREAS
TRAINING_AREAS = KNOWN_TRAINING_AREAS
```

- Add these fields to `BenchmarkRow` immediately after `stratum`:

```python
    bbox: tuple[float, float, float, float] | None
    data_source: str
    development_contact_status: str
    contact_evidence: str
    expected_role: str
```

- Replace `assign_tier()` with:

```python
def assign_tier(area: str, development_contact_status: str | None = None) -> str:
    normalized = area.lower()
    if normalized in TRAINING_AREAS or normalized in DEVELOPMENT_AREAS:
        return "tier2"
    if development_contact_status == "none":
        return "tier1"
    if development_contact_status == "known_contact":
        return "tier2"
    return "review_required"
```

- Update `REGISTRY_FIELDS` construction in `build_registry.py` after the dataclass signature changes by importing dataclass field metadata:

```python
from dataclasses import fields

REGISTRY_FIELDS = [field.name for field in fields(BenchmarkRow)]
```

- [ ] **Step 4: Update build registry to merge manifest metadata**

In `scripts/paper58_benchmark/build_registry.py`:

- Import:

```python
from scripts.paper58_benchmark.holdouts import (
    DEFAULT_HOLDOUT_MANIFEST,
    HoldoutArea,
    load_holdout_manifest,
    manifest_lookup,
)
```

- Add:

```python
def _load_holdout_lookup(path: Path | None) -> dict[str, HoldoutArea]:
    if path is None or not Path(path).exists():
        return {}
    return manifest_lookup(load_holdout_manifest(Path(path)))
```

- Add a `holdouts: dict[str, HoldoutArea]` parameter to `_build_row()`.

- Inside `_build_row()`, before QC:

```python
    holdout = holdouts.get(area)
    bbox = holdout.bbox if holdout else None
    data_source = holdout.data_source if holdout else ""
    development_contact_status = holdout.development_contact_status if holdout else "uncertain"
    contact_evidence = holdout.contact_evidence if holdout else "area missing from Paper58 holdout manifest"
    expected_role = holdout.expected_role if holdout else "review_required"
    stratum = holdout.stratum if holdout else area_stratum(area)
    tier = assign_tier(area, development_contact_status=development_contact_status)
```

- Use `tier=tier`, `stratum=stratum`, and the new metadata fields when constructing `BenchmarkRow`.

- Add a `holdout_manifest_path` parameter to `build_registry()`:

```python
    holdout_manifest_path: Path | None = DEFAULT_HOLDOUT_MANIFEST,
```

- Load holdouts once:

```python
    holdouts = _load_holdout_lookup(holdout_manifest_path)
```

- Pass `holdouts=holdouts` into `_build_row()`.

- Add a CLI argument:

```python
    parser.add_argument("--holdout-manifest", type=Path, default=DEFAULT_HOLDOUT_MANIFEST)
```

- Pass `holdout_manifest_path=args.holdout_manifest`.

- [ ] **Step 5: Run registry tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_registry.py tests/test_paper58_benchmark_holdouts.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

Run:

```powershell
git add scripts/paper58_benchmark/schema.py scripts/paper58_benchmark/build_registry.py tests/test_paper58_benchmark_registry.py
git commit -m "feat: apply provenance tiers in Paper58 registry"
```

---

### Task 4: Manifest-Driven Acquisition Scripts

**Files:**
- Modify: `scripts/rse_revision/fetch_independent_lulc_labels.py`
- Modify: `scripts/rse_revision/fetch_change_validation_embeddings.py`
- Modify: `scripts/rse_revision/generate_change_validation_predictions.py`
- Modify: `tests/test_rse_revision_change_validation.py`

- [ ] **Step 1: Add failing manifest-driven fetch tests**

Append to `tests/test_rse_revision_change_validation.py`:

```python
def _write_holdout_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "areas": [
                    {
                        "area": "toy_holdout",
                        "bbox": [120.1, 30.1, 120.2, 30.2],
                        "stratum": "Urban",
                        "years": [2020, 2021],
                        "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                        "selection_reason": "toy external holdout",
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


def test_fetch_independent_lulc_labels_reads_manifest_area(tmp_path: Path, monkeypatch):
    import scripts.rse_revision.fetch_independent_lulc_labels as fetcher

    manifest_path = tmp_path / "holdouts.json"
    _write_holdout_manifest(manifest_path)
    calls = []

    def fake_fixed_scale_extractor(bbox, year, scale=10):
        calls.append({"bbox": bbox, "year": year, "scale": scale})
        return np.full((2, 3), year - 2019, dtype=np.int32)

    monkeypatch.setattr(fetcher, "_extract_lulc_labels_fixed_scale", fake_fixed_scale_extractor)

    manifest = fetch_independent_lulc_labels(
        areas=["toy_holdout"],
        years=[2020, 2021],
        output_dir=tmp_path / "labels",
        manifest_path=tmp_path / "label_manifest.json",
        area_manifest_path=manifest_path,
        scale=500,
        fixed_scale=True,
    )

    assert manifest["status"] == "complete"
    assert manifest["n_records"] == 2
    assert calls[0] == {"bbox": [120.1, 30.1, 120.2, 30.2], "year": 2020, "scale": 500}


def test_fetch_change_validation_embeddings_reads_manifest_area(tmp_path: Path, monkeypatch):
    import scripts.rse_revision.fetch_change_validation_embeddings as fetcher

    manifest_path = tmp_path / "holdouts.json"
    _write_holdout_manifest(manifest_path)
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
        areas=["toy_holdout"],
        years=[2020, 2021],
        output_dir=tmp_path / "embeddings",
        manifest_path=tmp_path / "embedding_manifest.json",
        area_manifest_path=manifest_path,
        scale=500,
    )

    assert manifest["status"] == "complete"
    assert manifest["n_records"] == 2
    assert calls["embeddings"][0] == {"bbox": [120.1, 30.1, 120.2, 30.2], "year": 2020, "scale": 500}
    assert calls["context"] == [{"bbox": [120.1, 30.1, 120.2, 30.2], "target_shape": (2, 3)}]


def test_generate_change_validation_predictions_filters_manifest_area(tmp_path: Path):
    manifest_path = tmp_path / "holdouts.json"
    _write_holdout_manifest(manifest_path)
    embedding_dir = tmp_path / "embeddings"
    output_dir = tmp_path / "predicted"
    report_path = tmp_path / "prediction_readiness_report.json"
    embedding_dir.mkdir()
    np.save(embedding_dir / "other_area_emb_2020.npy", np.zeros((2, 2, 64), dtype=np.float32))
    np.save(embedding_dir / "other_area_emb_2021.npy", np.zeros((2, 2, 64), dtype=np.float32))

    report = generate_change_validation_predictions(
        embedding_dirs=[embedding_dir],
        output_dir=output_dir,
        report_path=report_path,
        weights_path=tmp_path / "missing_weights.pt",
        decoder_path=tmp_path / "missing_decoder.pkl",
        area_manifest_path=manifest_path,
    )

    assert report["status"] == "not_ready"
    cached_failure = next(item for item in report["readiness_failures"] if item["component"] == "cached_embeddings")
    assert cached_failure["candidate_areas"] == ["toy_holdout"]
```

- [ ] **Step 2: Run manifest-driven fetch tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_rse_revision_change_validation.py::test_fetch_independent_lulc_labels_reads_manifest_area tests/test_rse_revision_change_validation.py::test_fetch_change_validation_embeddings_reads_manifest_area tests/test_rse_revision_change_validation.py::test_generate_change_validation_predictions_filters_manifest_area -q
```

Expected: FAIL because the functions do not accept `area_manifest_path`.

- [ ] **Step 3: Update label fetcher**

In `scripts/rse_revision/fetch_independent_lulc_labels.py`:

- Import:

```python
from scripts.paper58_benchmark.holdouts import DEFAULT_HOLDOUT_MANIFEST, area_records_for_status, load_holdout_manifest
```

- Change `_area_lookup()` to:

```python
def _area_lookup(area_manifest_path: Path | None = None) -> dict[str, dict]:
    areas = {}
    for area in [*DEFAULT_TRAINING_AREAS, *EXTRA_VALIDATION_AREAS]:
        areas[area["name"].lower()] = {"name": area["name"].lower(), "bbox": area["bbox"]}
    if area_manifest_path is not None and Path(area_manifest_path).exists():
        for area in load_holdout_manifest(Path(area_manifest_path)):
            areas[area.area] = {"name": area.area, "bbox": list(area.bbox)}
    return areas
```

- Add function parameter:

```python
    area_manifest_path: Path | None = None,
```

- Use:

```python
    area_map = _area_lookup(area_manifest_path)
```

- Add CLI argument:

```python
    parser.add_argument("--area-manifest", type=Path, default=None)
```

- Pass `area_manifest_path=args.area_manifest`.

- [ ] **Step 4: Update embedding fetcher**

Make the same `_area_lookup(area_manifest_path)` and `area_manifest_path` changes in `scripts/rse_revision/fetch_change_validation_embeddings.py`.

- [ ] **Step 5: Update prediction generator**

In `scripts/rse_revision/generate_change_validation_predictions.py`:

- Import:

```python
from scripts.paper58_benchmark.holdouts import load_holdout_manifest
```

- Add helper:

```python
def _areas_from_manifest(path: Path) -> list[str]:
    return [area.area for area in load_holdout_manifest(path) if area.development_contact_status == "none"]
```

- Add function parameter:

```python
    area_manifest_path: Path | None = None,
```

- Before applying `areas`:

```python
    if areas is None and area_manifest_path is not None:
        areas = _areas_from_manifest(Path(area_manifest_path))
```

- In the no-embeddings readiness failure, include:

```python
                "candidate_areas": sorted(areas or []),
```

- Add CLI argument:

```python
    parser.add_argument("--area-manifest", type=Path, default=None)
```

- Pass `area_manifest_path=args.area_manifest`.

- [ ] **Step 6: Run RSE change-validation tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_rse_revision_change_validation.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

Run:

```powershell
git add scripts/rse_revision/fetch_independent_lulc_labels.py scripts/rse_revision/fetch_change_validation_embeddings.py scripts/rse_revision/generate_change_validation_predictions.py tests/test_rse_revision_change_validation.py
git commit -m "feat: drive Paper58 acquisition from holdout manifest"
```

---

### Task 5: Evaluation Guard For Invalid Tier 1 Provenance

**Files:**
- Modify: `scripts/paper58_benchmark/evaluate_benchmark.py`
- Modify: `tests/test_paper58_benchmark_evaluation.py`

- [ ] **Step 1: Add failing evaluation guard test**

Append to `tests/test_paper58_benchmark_evaluation.py`:

```python
def test_read_registry_rejects_tier1_without_no_contact_provenance(tmp_path: Path):
    registry_path = tmp_path / "benchmark_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "uncertain_external",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Mixed",
                        "bbox": [121.1, 31.1, 121.2, 31.2],
                        "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                        "development_contact_status": "uncertain",
                        "contact_evidence": "not cleared",
                        "expected_role": "review_only",
                        "label_start_path": "a.npy",
                        "label_end_path": "b.npy",
                        "prediction_path": "c.npy",
                        "qc_status": "include",
                        "excluded_reason": "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tier1 row 0 requires development_contact_status='none'"):
        _read_registry(registry_path)
```

Update all registry row fixtures in this file to include:

```python
                "bbox": [120.1, 30.1, 120.2, 30.2],
                "data_source": "ESRI_LULC_10m_and_AlphaEarth",
                "development_contact_status": "none" if tier == "tier1" else "known_contact",
                "contact_evidence": "toy provenance evidence",
                "expected_role": "positive_change_candidate",
```

For single-row manual fixtures, use `development_contact_status="none"` when `tier="tier1"`.

- [ ] **Step 2: Run evaluation tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_evaluation.py -q
```

Expected: FAIL because `_read_registry()` does not validate provenance.

- [ ] **Step 3: Add required provenance fields and validation**

In `scripts/paper58_benchmark/evaluate_benchmark.py`, extend `REQUIRED_REGISTRY_FIELDS`:

```python
    "bbox",
    "data_source",
    "development_contact_status",
    "contact_evidence",
    "expected_role",
```

Inside `_read_registry()`, after required-field checks:

```python
        if row.get("tier") == "tier1" and row.get("development_contact_status") != "none":
            raise ValueError(f"tier1 row {index} requires development_contact_status='none'")
        if row.get("tier") == "tier1" and not str(row.get("contact_evidence", "")).strip():
            raise ValueError(f"tier1 row {index} requires non-empty contact_evidence")
```

- [ ] **Step 4: Run evaluation tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_evaluation.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

Run:

```powershell
git add scripts/paper58_benchmark/evaluate_benchmark.py tests/test_paper58_benchmark_evaluation.py
git commit -m "fix: reject invalid Tier 1 provenance rows"
```

---

### Task 6: Provenance Audit Output

**Files:**
- Create: `scripts/paper58_benchmark/audit_provenance.py`
- Create: `tests/test_paper58_benchmark_provenance_audit.py`

- [ ] **Step 1: Write failing provenance audit tests**

Create `tests/test_paper58_benchmark_provenance_audit.py`:

```python
import json
from pathlib import Path

from scripts.paper58_benchmark.audit_provenance import audit_registry_provenance


def test_audit_registry_provenance_writes_json_and_csv(tmp_path: Path):
    registry_path = tmp_path / "benchmark_registry.json"
    output_dir = tmp_path / "out"
    registry_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "strict_external",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Urban",
                        "development_contact_status": "none",
                        "contact_evidence": "toy no-contact evidence",
                        "qc_status": "include",
                    },
                    {
                        "area": "poyang_lake",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier2",
                        "stratum": "Wetland",
                        "development_contact_status": "known_contact",
                        "contact_evidence": "training-list membership",
                        "qc_status": "include",
                    },
                    {
                        "area": "uncertain_external",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "review_required",
                        "stratum": "Mixed",
                        "development_contact_status": "uncertain",
                        "contact_evidence": "not cleared",
                        "qc_status": "include",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = audit_registry_provenance(registry_path=registry_path, output_dir=output_dir)

    assert report["n_rows"] == 3
    assert report["tier_counts"] == {"review_required": 1, "tier1": 1, "tier2": 1}
    assert report["invalid_tier1_rows"] == []
    assert (output_dir / "benchmark_provenance_audit.json").exists()
    assert (output_dir / "benchmark_provenance_audit.csv").exists()
```

- [ ] **Step 2: Run audit tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_provenance_audit.py -q
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement provenance audit script**

Create `scripts/paper58_benchmark/audit_provenance.py`:

```python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.paper58_benchmark.schema import DEFAULT_BENCHMARK_DIR, write_csv, write_json


AUDIT_FIELDS = [
    "area",
    "start_year",
    "end_year",
    "tier",
    "stratum",
    "development_contact_status",
    "contact_evidence",
    "qc_status",
    "audit_status",
]


def _read_rows(path: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("benchmark registry must contain a rows list")
    return [row for row in rows if isinstance(row, dict)]


def _audit_status(row: dict) -> str:
    if row.get("tier") == "tier1" and row.get("development_contact_status") != "none":
        return "invalid_tier1_contact"
    if row.get("tier") == "tier1" and not str(row.get("contact_evidence", "")).strip():
        return "invalid_tier1_missing_evidence"
    if row.get("tier") == "review_required":
        return "review_required"
    return "ok"


def audit_registry_provenance(
    registry_path: Path = DEFAULT_BENCHMARK_DIR / "benchmark_registry.json",
    output_dir: Path = DEFAULT_BENCHMARK_DIR,
) -> dict:
    rows = _read_rows(Path(registry_path))
    audit_rows = []
    for row in rows:
        audit_rows.append(
            {
                "area": row.get("area"),
                "start_year": row.get("start_year"),
                "end_year": row.get("end_year"),
                "tier": row.get("tier"),
                "stratum": row.get("stratum"),
                "development_contact_status": row.get("development_contact_status"),
                "contact_evidence": row.get("contact_evidence"),
                "qc_status": row.get("qc_status"),
                "audit_status": _audit_status(row),
            }
        )
    invalid = [row for row in audit_rows if str(row["audit_status"]).startswith("invalid_tier1")]
    report = {
        "n_rows": len(rows),
        "tier_counts": dict(sorted(Counter(str(row.get("tier")) for row in rows).items())),
        "invalid_tier1_rows": invalid,
    }
    output_dir = Path(output_dir)
    write_json(output_dir / "benchmark_provenance_audit.json", report)
    write_csv(output_dir / "benchmark_provenance_audit.csv", audit_rows, AUDIT_FIELDS)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Paper58 benchmark provenance tiers.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_BENCHMARK_DIR / "benchmark_registry.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    args = parser.parse_args()
    report = audit_registry_provenance(args.registry, args.output_dir)
    print(
        "Benchmark provenance audit: "
        f"{report['n_rows']} row(s), "
        f"{len(report['invalid_tier1_rows'])} invalid Tier 1 row(s)"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run audit tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_provenance_audit.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 6**

Run:

```powershell
git add scripts/paper58_benchmark/audit_provenance.py tests/test_paper58_benchmark_provenance_audit.py
git commit -m "feat: add Paper58 provenance audit output"
```

---

### Task 7: Local Verification Before Network Acquisition

**Files:**
- No code edits expected.

- [ ] **Step 1: Run the local provenance and benchmark test set**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_holdouts.py tests/test_paper58_benchmark_registry.py tests/test_paper58_benchmark_evaluation.py tests/test_paper58_benchmark_statistics.py tests/test_paper58_benchmark_figures.py tests/test_paper58_benchmark_provenance_audit.py tests/test_rse_revision_change_validation.py -q
```

Expected: PASS.

- [ ] **Step 2: Run whitespace diff check**

Run:

```powershell
git diff --check
```

Expected: no output.

- [ ] **Step 3: Rebuild the registry locally with strict provenance**

Run:

```powershell
python -m scripts.paper58_benchmark.build_registry --holdout-manifest data\independent_change_labels\paper58_holdout_areas.json
```

Expected interpretation:

- Current `poyang_lake` and `wuyi_mountain` are no longer strict Tier 1.
- Before new data acquisition, the gate should be `insufficient_tier1` or fail for lack of strict Tier 1 rows.

- [ ] **Step 4: Run provenance audit locally**

Run:

```powershell
python -m scripts.paper58_benchmark.audit_provenance
```

Expected:

```text
Benchmark provenance audit: 12 row(s), 0 invalid Tier 1 row(s)
```

If the row count differs because new files already exist, inspect `paper\rse_submission_paper58\benchmark_results\benchmark_provenance_audit.csv` before continuing.

- [ ] **Step 5: Evaluate the strict local registry**

Run:

```powershell
python -m scripts.paper58_benchmark.evaluate_benchmark --n-boot 1000
```

Expected: `gate status=insufficient_tier1` or `gate status=fail`. This is not a regression; it is the intended provenance correction.

- [ ] **Step 6: Commit strict provenance outputs if they changed**

Run:

```powershell
git status --short --untracked-files=no
```

If benchmark CSV/JSON outputs changed, commit them:

```powershell
git add paper/rse_submission_paper58/benchmark_results/*.csv paper/rse_submission_paper58/benchmark_results/*.json
git commit -m "data: record strict Paper58 provenance audit"
```

Do not commit large new figures in this task unless they were regenerated intentionally.

---

### Task 8: GEE Acquisition For New Tier 1 Holdouts

**Files:**
- Expected generated outputs under:
  - `data/independent_change_labels/labels/`
  - `data/independent_change_labels/embeddings/`
  - `data/independent_change_labels/label_manifest.json`
  - `data/independent_change_labels/embedding_manifest.json`

- [ ] **Step 1: Fetch ESRI labels for manifest-cleared holdouts**

This step requires GEE/network access. If the command fails with authentication, DNS, quota, or sandbox network errors, rerun it with escalated approval.

Run:

```powershell
python -m scripts.rse_revision.fetch_independent_lulc_labels --area-manifest data\independent_change_labels\paper58_holdout_areas.json --areas tianjin_binhai_holdout,shenzhen_outer_holdout,tarim_oasis_holdout,sanjiang_plain_holdout,dongting_lake_holdout,xiaoxinganling_holdout,qinling_mountain_holdout,haibei_plateau_holdout --years 2020,2021 --scale 500 --fixed-scale
```

Expected:

```text
Independent LULC label fetch: complete, 16 record(s), 0 failure(s)
```

If failures occur, preserve the manifest and failure records. Do not silently remove failed areas from the experiment.

- [ ] **Step 2: Fetch AlphaEarth embeddings and context for manifest-cleared holdouts**

Run:

```powershell
python -m scripts.rse_revision.fetch_change_validation_embeddings --area-manifest data\independent_change_labels\paper58_holdout_areas.json --areas tianjin_binhai_holdout,shenzhen_outer_holdout,tarim_oasis_holdout,sanjiang_plain_holdout,dongting_lake_holdout,xiaoxinganling_holdout,qinling_mountain_holdout,haibei_plateau_holdout --years 2020,2021 --scale 500
```

Expected:

```text
Change-validation embedding fetch: complete, 16 grid(s), 8 context grid(s), 0 failure(s)
```

If some regions fail, continue only if at least 4 provenance-cleared regions across at least 3 strata remain. Otherwise stop the experiment and record the block.

- [ ] **Step 3: Inspect acquisition manifests**

Run:

```powershell
Get-Content -Raw -LiteralPath data\independent_change_labels\label_manifest.json
```

Run:

```powershell
Get-Content -Raw -LiteralPath data\independent_change_labels\embedding_manifest.json
```

Expected interpretation:

- `n_failures` should be `0`, or failures must be documented.
- label and embedding output paths must use the same area names from `paper58_holdout_areas.json`.

- [ ] **Step 4: Commit acquisition manifests and fetched arrays**

Run:

```powershell
git status --short -- data/independent_change_labels
```

If the fetched arrays are reasonably sized for the repository and are already tracked in this project pattern, commit them:

```powershell
git add data/independent_change_labels/paper58_holdout_areas.json data/independent_change_labels/label_manifest.json data/independent_change_labels/embedding_manifest.json data/independent_change_labels/labels data/independent_change_labels/embeddings
git commit -m "data: fetch Paper58 Tier 1 holdout labels and embeddings"
```

If the arrays are too large for Git, stop and record the exact file sizes before deciding whether to use Git LFS or an external artifact path.

---

### Task 9: Prediction Generation For New Holdouts

**Files:**
- Expected generated outputs under:
  - `data/independent_change_labels/predicted/`
  - `data/independent_change_labels/prediction_readiness_report.json`

- [ ] **Step 1: Generate baseline-scenario predictions for manifest-cleared holdouts**

Run:

```powershell
python -m scripts.rse_revision.generate_change_validation_predictions --embedding-dir data\independent_change_labels\embeddings --area-manifest data\independent_change_labels\paper58_holdout_areas.json
```

Expected:

```text
Change-validation prediction generation: complete, 8 prediction(s)
```

If the report says `not_ready`, inspect `data\independent_change_labels\prediction_readiness_report.json`. Do not evaluate missing prediction rows as if they were negative evidence.

- [ ] **Step 2: Inspect prediction readiness report**

Run:

```powershell
Get-Content -Raw -LiteralPath data\independent_change_labels\prediction_readiness_report.json
```

Expected interpretation:

- `status` should be `complete`.
- `n_skipped` should be `0`, or every skipped row must have an explicit reason.

- [ ] **Step 3: Commit generated predictions**

Run:

```powershell
git add data/independent_change_labels/predicted data/independent_change_labels/prediction_readiness_report.json
git commit -m "data: generate Paper58 Tier 1 holdout predictions"
```

---

### Task 10: Expanded Registry, Gate Rerun, And Stop/Go Decision

**Files:**
- Generated outputs under `paper/rse_submission_paper58/benchmark_results/`

- [ ] **Step 1: Build expanded registry**

Run:

```powershell
python -m scripts.paper58_benchmark.build_registry --holdout-manifest data\independent_change_labels\paper58_holdout_areas.json
```

Expected:

```text
Benchmark registry: 20 candidate pair(s), <N> included pair(s)
```

The exact count may vary if some acquisition rows failed. Continue only if there are at least 4 included strict Tier 1 region clusters and at least 3 strict Tier 1 strata with non-zero reference change.

- [ ] **Step 2: Run provenance audit**

Run:

```powershell
python -m scripts.paper58_benchmark.audit_provenance
```

Expected:

```text
Benchmark provenance audit: <N> row(s), 0 invalid Tier 1 row(s)
```

If invalid Tier 1 rows exist, fix the registry/provenance logic before evaluation.

- [ ] **Step 3: Evaluate expanded benchmark**

Run:

```powershell
python -m scripts.paper58_benchmark.evaluate_benchmark --n-boot 5000
```

Expected:

```text
Benchmark evaluation: <N> evaluated pair(s), gate status=<pass|fail|insufficient_tier1>
```

Interpretation:

- `pass`: a separate manuscript-revision plan may be created.
- `fail`: do not strengthen claims; prepare downgrade or negative-results path.
- `insufficient_tier1`: stop and record why a credible Tier 1 set could not be assembled.

- [ ] **Step 4: Generate updated benchmark figures**

Run:

```powershell
python -m scripts.paper58_benchmark.make_benchmark_figures
```

Expected:

```text
Wrote Paper58 benchmark figures from D:\test\paper58-geofm-world-model-rl\paper\rse_submission_paper58\benchmark_results
```

- [ ] **Step 5: Inspect gate report**

Run:

```powershell
Get-Content -Raw -LiteralPath paper\rse_submission_paper58\benchmark_results\benchmark_gate_report.json
```

Required pass conditions:

- `primary_gate_pass = true`
- `spatial_gate_pass = true`
- `strata_gate_pass = true`
- `positive_tier1_strata >= 3`
- `tier1_primary_change.ci_low > 0`
- `tier1_spatial_change.ci_low > 0`

- [ ] **Step 6: Commit expanded benchmark outputs**

Measure figure size:

```powershell
(Get-ChildItem -LiteralPath paper\rse_submission_paper58\benchmark_results\figures -File | Measure-Object -Property Length -Sum).Sum
```

If figure size is less than or equal to 10485760 bytes:

```powershell
git add paper/rse_submission_paper58/benchmark_results
git commit -m "data: run expanded Paper58 Tier 1 benchmark"
```

If figure size is larger:

```powershell
git add paper/rse_submission_paper58/benchmark_results/*.csv paper/rse_submission_paper58/benchmark_results/*.json
git commit -m "data: run expanded Paper58 Tier 1 benchmark"
```

---

### Task 11: Handoff And Submission Decision Record

**Files:**
- Create or modify: `docs/current_work_progress_2026-06-19.md`

- [ ] **Step 1: Update current work progress**

Append a section to `docs/current_work_progress_2026-06-19.md`:

```markdown
## Tier 1 Expansion Result

The strict Tier 1 expansion workflow is documented at:

```text
docs/superpowers/specs/2026-06-19-paper58-tier1-expansion-design.md
docs/superpowers/plans/2026-06-19-paper58-tier1-expansion.md
```

The expanded benchmark gate result is recorded at:

```text
paper/rse_submission_paper58/benchmark_results/benchmark_gate_report.json
```

Decision rule:

- If the expanded gate passed, create a separate manuscript-revision plan that maps every claim to benchmark outputs.
- If the expanded gate failed or remained insufficient, do not strengthen the RSE manuscript. Continue only with claim downgrading, limitations, data availability, and negative/insufficient-evidence reporting.
```

Then add the actual gate status and key numbers from `benchmark_gate_report.json`.

- [ ] **Step 2: Run final relevant verification**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_holdouts.py tests/test_paper58_benchmark_registry.py tests/test_paper58_benchmark_evaluation.py tests/test_paper58_benchmark_statistics.py tests/test_paper58_benchmark_figures.py tests/test_paper58_benchmark_provenance_audit.py tests/test_rse_revision_change_validation.py tests/test_rse_revision_results.py tests/test_rse_revision_change_validation.py -q
```

Expected: PASS.

- [ ] **Step 3: Run diff check**

Run:

```powershell
git diff --check
```

Expected: no output.

- [ ] **Step 4: Commit handoff**

Run:

```powershell
git add docs/current_work_progress_2026-06-19.md
git commit -m "docs: record Paper58 Tier 1 expansion decision"
```

---

## Self-Review Checklist

- Spec coverage: The plan implements provenance manifesting, strict Tier 1 eligibility, manifest-driven acquisition, registry metadata, evaluation guards, provenance audit output, expanded benchmark rerun, and manuscript stop/go rules.
- Scope control: The plan does not change model architecture and does not edit the manuscript toward stronger claims.
- Type consistency: The provenance fields use `bbox`, `data_source`, `development_contact_status`, `contact_evidence`, and `expected_role` consistently across manifest, registry, evaluation, and audit output.
- Evidence discipline: Current nominal Tier 1 rows that appear in training areas are conservatively downgraded before new external evidence is added.
- Network boundary: GEE/data acquisition is isolated to Task 8 and must use approval if sandbox or network restrictions block it.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-19-paper58-tier1-expansion.md`. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using plan checkpoints.

Recommended execution: option 1 through Task 7 first, then pause before Task 8 because GEE/network acquisition may require approval and may produce large data files.
