# Paper58 Batch 2 Tier 1 Stability Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second external Tier 1 holdout batch, evaluate it separately from the first batch, and rerun a combined benchmark without changing the existing gate logic or manuscript.

**Architecture:** Reuse the existing Paper58 provenance, acquisition, prediction, and benchmark pipeline exactly as-is for Tier logic and evaluation. Add one Batch 2 manifest, one small manifest-combining utility, and output-directory-isolated benchmark runs for `Batch 2 only` and `Batch 1 + Batch 2 combined`.

**Tech Stack:** Python 3, stdlib JSON/argparse/pathlib/dataclasses, pytest, existing Paper58 benchmark scripts, existing GEE-backed acquisition helpers.

---

## File Structure

Create these files:

- `data/independent_change_labels/paper58_holdout_areas_batch2.json`: second-batch external candidate manifest.
- `scripts/paper58_benchmark/build_combined_holdout_manifest.py`: combine validated holdout manifests into one derived manifest.

Modify these files:

- `tests/test_paper58_benchmark_holdouts.py`: add tests for Batch 2 manifest loading and combined-manifest behavior.
- `scripts/paper58_benchmark/build_registry.py`: add an opt-in manifest-area filter for isolated Batch 2 and combined registries.
- `tests/test_paper58_benchmark_registry.py`: verify output isolation and manifest-area filtering.
- `docs/current_work_progress_2026-06-20.md`: append Batch 2 stability-check state after execution.

No changes should be made to:

- `scripts/paper58_benchmark/evaluate_benchmark.py`
- `scripts/paper58_benchmark/make_benchmark_figures.py`
- the RSE manuscript

---

### Task 1: Add Combined-Manifest Utility

**Files:**
- Create: `scripts/paper58_benchmark/build_combined_holdout_manifest.py`
- Modify: `tests/test_paper58_benchmark_holdouts.py`

- [ ] **Step 1: Write failing tests for combining manifests**

Append to `tests/test_paper58_benchmark_holdouts.py`:

```python
from scripts.paper58_benchmark.build_combined_holdout_manifest import build_combined_holdout_manifest


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
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_holdouts.py::test_build_combined_holdout_manifest_merges_two_valid_manifests tests/test_paper58_benchmark_holdouts.py::test_build_combined_holdout_manifest_rejects_duplicate_area_names -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.paper58_benchmark.build_combined_holdout_manifest'`.

- [ ] **Step 3: Implement the combined-manifest utility**

Create `scripts/paper58_benchmark/build_combined_holdout_manifest.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.paper58_benchmark.holdouts import HoldoutArea, load_holdout_manifest


def _row(area: HoldoutArea) -> dict:
    return {
        "area": area.area,
        "bbox": list(area.bbox),
        "stratum": area.stratum,
        "years": list(area.years),
        "data_source": area.data_source,
        "selection_reason": area.selection_reason,
        "development_contact_status": area.development_contact_status,
        "contact_evidence": area.contact_evidence,
        "expected_role": area.expected_role,
        "notes": area.notes,
    }


def build_combined_holdout_manifest(
    manifest_paths: list[Path],
    output_path: Path,
) -> list[HoldoutArea]:
    combined: list[HoldoutArea] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        for area in load_holdout_manifest(Path(manifest_path)):
            if area.area in seen:
                raise ValueError(f"duplicate combined holdout area: {area.area}")
            seen.add(area.area)
            combined.append(area)
    payload = {
        "version": 1,
        "created": "2026-06-20",
        "purpose": "Paper58 combined Batch 1 + Batch 2 holdout manifest",
        "source_manifests": [str(Path(path)) for path in manifest_paths],
        "areas": [_row(area) for area in combined],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a combined Paper58 holdout manifest.")
    parser.add_argument("--manifest", dest="manifests", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    combined = build_combined_holdout_manifest(args.manifests, args.output)
    print(f"Combined holdout manifest: {len(combined)} area(s)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the new tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_holdouts.py::test_build_combined_holdout_manifest_merges_two_valid_manifests tests/test_paper58_benchmark_holdouts.py::test_build_combined_holdout_manifest_rejects_duplicate_area_names -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add scripts/paper58_benchmark/build_combined_holdout_manifest.py tests/test_paper58_benchmark_holdouts.py
git commit -m "feat: add Paper58 combined holdout manifest builder"
```

---

### Task 2: Add Batch 2 Holdout Manifest

**Files:**
- Create: `data/independent_change_labels/paper58_holdout_areas_batch2.json`
- Modify: `tests/test_paper58_benchmark_holdouts.py`

- [ ] **Step 1: Add a repository test for the Batch 2 manifest**

Append to `tests/test_paper58_benchmark_holdouts.py`:

```python
def test_repository_batch2_holdout_manifest_has_target_batch_shape():
    manifest_path = DEFAULT_HOLDOUT_MANIFEST.parent / "paper58_holdout_areas_batch2.json"
    areas = load_holdout_manifest(manifest_path)
    lookup = manifest_lookup(areas)
    tier1 = [area for area in areas if tier_from_provenance(area.area, lookup) == "tier1"]

    assert len(areas) == 8
    assert len(tier1) == 8
    assert len({area.stratum for area in tier1}) >= 5
    assert all(area.years == (2020, 2021) for area in tier1)
```

- [ ] **Step 2: Run the new repository test and verify it fails**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_holdouts.py::test_repository_batch2_holdout_manifest_has_target_batch_shape -q
```

Expected: FAIL with `FileNotFoundError` for `paper58_holdout_areas_batch2.json`.

- [ ] **Step 3: Create the Batch 2 manifest**

Create `data/independent_change_labels/paper58_holdout_areas_batch2.json` with 8 new external candidates:

```json
{
  "version": 1,
  "created": "2026-06-20",
  "purpose": "Paper58 Batch 2 external Tier 1 stability candidates",
  "areas": [
    {
      "area": "xiong_an_fringe_holdout",
      "bbox": [115.90, 38.95, 116.00, 39.05],
      "stratum": "Urban",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "independent inland urban-fringe candidate outside Batch 1 urban boxes",
      "development_contact_status": "none",
      "contact_evidence": "area name and bbox are absent from DEFAULT_TRAINING_AREAS, known development areas, and Batch 1 manifest",
      "expected_role": "positive_change_candidate",
      "notes": "Batch 2 urban inland candidate"
    },
    {
      "area": "beibu_gulf_urban_holdout",
      "bbox": [109.15, 21.55, 109.25, 21.65],
      "stratum": "Urban",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "independent coastal urban-edge candidate outside Batch 1 urban boxes",
      "development_contact_status": "none",
      "contact_evidence": "area name and bbox are absent from DEFAULT_TRAINING_AREAS, known development areas, and Batch 1 manifest",
      "expected_role": "reserve_positive_change_candidate",
      "notes": "Batch 2 urban coastal candidate"
    },
    {
      "area": "songnen_plain_holdout",
      "bbox": [124.80, 45.55, 124.90, 45.65],
      "stratum": "Agriculture",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "independent plains agriculture candidate outside Batch 1 agriculture boxes",
      "development_contact_status": "none",
      "contact_evidence": "area name and bbox are absent from DEFAULT_TRAINING_AREAS, known development areas, and Batch 1 manifest",
      "expected_role": "positive_change_candidate",
      "notes": "Batch 2 agriculture plains candidate"
    },
    {
      "area": "hexi_irrigation_holdout",
      "bbox": [100.10, 38.65, 100.20, 38.75],
      "stratum": "Agriculture",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "independent dryland irrigation candidate outside Batch 1 agriculture boxes",
      "development_contact_status": "none",
      "contact_evidence": "area name and bbox are absent from DEFAULT_TRAINING_AREAS, known development areas, and Batch 1 manifest",
      "expected_role": "reserve_positive_change_candidate",
      "notes": "Batch 2 agriculture dryland candidate"
    },
    {
      "area": "changbai_margin_holdout",
      "bbox": [127.60, 42.15, 127.70, 42.25],
      "stratum": "Forest",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "independent forest candidate outside Batch 1 northeast forest boxes",
      "development_contact_status": "none",
      "contact_evidence": "area name and bbox are absent from DEFAULT_TRAINING_AREAS, known development areas, and Batch 1 manifest",
      "expected_role": "positive_change_candidate",
      "notes": "Batch 2 forest candidate"
    },
    {
      "area": "erlong_lake_margin_holdout",
      "bbox": [123.65, 43.25, 123.75, 43.35],
      "stratum": "Wetland",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "independent inland wetland-margin candidate outside Poyang and Dongting contexts",
      "development_contact_status": "none",
      "contact_evidence": "area name and bbox are absent from DEFAULT_TRAINING_AREAS, known development areas, and Batch 1 manifest",
      "expected_role": "positive_change_candidate",
      "notes": "Batch 2 wetland candidate"
    },
    {
      "area": "ordos_grassland_holdout",
      "bbox": [109.45, 39.25, 109.55, 39.35],
      "stratum": "Grassland",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "independent grassland candidate to test ecological heterogeneity",
      "development_contact_status": "none",
      "contact_evidence": "area name and bbox are absent from DEFAULT_TRAINING_AREAS, known development areas, and Batch 1 manifest",
      "expected_role": "positive_change_candidate",
      "notes": "Batch 2 grassland candidate"
    },
    {
      "area": "west_sichuan_plateau_holdout",
      "bbox": [102.55, 31.95, 102.65, 32.05],
      "stratum": "Plateau",
      "years": [2020, 2021],
      "data_source": "ESRI_LULC_10m_and_AlphaEarth",
      "selection_reason": "independent plateau candidate outside Batch 1 plateau and forest boxes",
      "development_contact_status": "none",
      "contact_evidence": "area name and bbox are absent from DEFAULT_TRAINING_AREAS, known development areas, and Batch 1 manifest",
      "expected_role": "positive_change_candidate",
      "notes": "Batch 2 plateau candidate"
    }
  ]
}
```

- [ ] **Step 4: Run the repository manifest tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_holdouts.py::test_repository_batch2_holdout_manifest_has_target_batch_shape tests/test_paper58_benchmark_holdouts.py::test_repository_holdout_manifest_has_minimum_external_batch -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add data/independent_change_labels/paper58_holdout_areas_batch2.json tests/test_paper58_benchmark_holdouts.py
git commit -m "data: add Paper58 Batch 2 holdout manifest"
```

---

### Task 3: Add Explicit Manifest-Area Registry Filtering

**Files:**
- Modify: `scripts/paper58_benchmark/build_registry.py`
- Modify: `tests/test_paper58_benchmark_registry.py`

- [ ] **Step 1: Write failing tests for Batch 2 output isolation and manifest filtering**

Append to `tests/test_paper58_benchmark_registry.py`:

```python
def test_build_registry_can_filter_to_holdout_manifest_areas(tmp_path: Path):
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
    for area in ("strict_external", "old_cached_area"):
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
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    build_registry(
        labels_dir=labels,
        predictions_dir=predicted,
        independent_embeddings_dir=embeddings,
        experiment_data_dir=experiment_data,
        output_dir=output,
        holdout_manifest_path=manifest_path,
        filter_to_holdout_manifest=True,
    )

    rows = json.loads((output / "benchmark_registry.json").read_text(encoding="utf-8"))["rows"]

    assert [row["area"] for row in rows] == ["strict_external"]
    assert rows[0]["tier"] == "tier1"


def test_build_registry_writes_to_requested_output_dir_without_touching_default(tmp_path: Path):
    labels = tmp_path / "labels"
    predicted = tmp_path / "predicted"
    embeddings = tmp_path / "embeddings"
    experiment_data = tmp_path / "experiment_data"
    batch2_output = tmp_path / "batch2_out"
    combined_output = tmp_path / "combined_out"
    for path in (labels, predicted, embeddings, experiment_data):
        path.mkdir()

    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
    np.save(labels / "strict_external_lulc_2020.npy", start)
    np.save(labels / "strict_external_lulc_2021.npy", end)
    np.save(predicted / "strict_external_lulc_pred_2020_2021.npy", pred)
    np.save(embeddings / "strict_external_emb_2020.npy", np.zeros((2, 2, 64), dtype=np.float32))
    np.save(embeddings / "strict_external_emb_2021.npy", np.ones((2, 2, 64), dtype=np.float32))

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

    build_registry(
        labels_dir=labels,
        predictions_dir=predicted,
        independent_embeddings_dir=embeddings,
        experiment_data_dir=experiment_data,
        output_dir=batch2_output,
        holdout_manifest_path=manifest_path,
        filter_to_holdout_manifest=True,
    )
    build_registry(
        labels_dir=labels,
        predictions_dir=predicted,
        independent_embeddings_dir=embeddings,
        experiment_data_dir=experiment_data,
        output_dir=combined_output,
        holdout_manifest_path=manifest_path,
        filter_to_holdout_manifest=True,
    )

    assert (batch2_output / "benchmark_registry.json").exists()
    assert (combined_output / "benchmark_registry.json").exists()
    assert (batch2_output / "benchmark_registry.json").read_text(encoding="utf-8") == (
        combined_output / "benchmark_registry.json"
    ).read_text(encoding="utf-8")
```

- [ ] **Step 2: Run the new registry tests and verify the filter test fails**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_registry.py::test_build_registry_can_filter_to_holdout_manifest_areas tests/test_paper58_benchmark_registry.py::test_build_registry_writes_to_requested_output_dir_without_touching_default -q
```

Expected: FAIL with `TypeError: build_registry() got an unexpected keyword argument 'filter_to_holdout_manifest'`.

- [ ] **Step 3: Implement opt-in filtering in the registry builder**

In `scripts/paper58_benchmark/build_registry.py`, add this helper after `_load_holdout_lookup()`:

```python
def _filter_predictions_to_holdouts(
    predictions: list[tuple[str, int, int, Path]],
    holdouts: dict[str, HoldoutArea],
) -> list[tuple[str, int, int, Path]]:
    return [
        (area, start_year, end_year, prediction_path)
        for area, start_year, end_year, prediction_path in predictions
        if area.lower() in holdouts
    ]
```

Change the `build_registry()` signature to:

```python
def build_registry(
    labels_dir: Path = DEFAULT_LABELS_DIR,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
    independent_embeddings_dir: Path = DEFAULT_INDEPENDENT_EMBEDDINGS_DIR,
    experiment_data_dir: Path = DEFAULT_EXPERIMENT_DATA_DIR,
    output_dir: Path = DEFAULT_BENCHMARK_DIR,
    holdout_manifest_path: Path | None = DEFAULT_HOLDOUT_MANIFEST,
    filter_to_holdout_manifest: bool = False,
) -> list[BenchmarkRow]:
```

Immediately after loading `holdouts`, filter predictions when requested:

```python
    if filter_to_holdout_manifest:
        predictions = _filter_predictions_to_holdouts(predictions, holdouts)
```

In `main()`, add:

```python
    parser.add_argument(
        "--filter-to-holdout-manifest",
        action="store_true",
        help="Only include prediction rows whose area appears in the holdout manifest.",
    )
```

Pass the flag into `build_registry()`:

```python
        filter_to_holdout_manifest=args.filter_to_holdout_manifest,
```

- [ ] **Step 4: Run the new registry tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_registry.py::test_build_registry_can_filter_to_holdout_manifest_areas tests/test_paper58_benchmark_registry.py::test_build_registry_writes_to_requested_output_dir_without_touching_default -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add scripts/paper58_benchmark/build_registry.py tests/test_paper58_benchmark_registry.py
git commit -m "feat: filter Paper58 registry to holdout manifest areas"
```

---

### Task 4: Local Verification Before Batch 2 Network Work

**Files:**
- No code edits expected.

- [ ] **Step 1: Run the local test set that covers new Batch 2 plumbing**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_holdouts.py tests/test_paper58_benchmark_registry.py tests/test_paper58_benchmark_evaluation.py tests/test_paper58_benchmark_figures.py tests/test_paper58_benchmark_provenance_audit.py tests/test_rse_revision_change_validation.py -q
```

Expected: PASS.

- [ ] **Step 2: Run whitespace diff check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 3: Build the combined manifest locally**

Run:

```powershell
python -m scripts.paper58_benchmark.build_combined_holdout_manifest --manifest data\independent_change_labels\paper58_holdout_areas.json --manifest data\independent_change_labels\paper58_holdout_areas_batch2.json --output data\independent_change_labels\paper58_holdout_areas_combined.json
```

Expected:

```text
Combined holdout manifest: 18 area(s)
```

- [ ] **Step 4: Commit the derived combined manifest if it is stable and intentional**

Run:

```powershell
git status --short -- data/independent_change_labels
```

If only the derived combined manifest is new:

```powershell
git add data/independent_change_labels/paper58_holdout_areas_combined.json
git commit -m "data: add Paper58 combined holdout manifest"
```

---

### Task 5: Batch 2 GEE Acquisition

**Files:**
- Generated outputs under:
  - `data/independent_change_labels/labels/`
  - `data/independent_change_labels/embeddings/`
  - `data/independent_change_labels/label_manifest.json`
  - `data/independent_change_labels/embedding_manifest.json`

- [ ] **Step 1: Fetch ESRI labels for Batch 2 candidates**

Run:

```powershell
python -m scripts.rse_revision.fetch_independent_lulc_labels --area-manifest data\independent_change_labels\paper58_holdout_areas_batch2.json --areas xiong_an_fringe_holdout,beibu_gulf_urban_holdout,songnen_plain_holdout,hexi_irrigation_holdout,changbai_margin_holdout,erlong_lake_margin_holdout,ordos_grassland_holdout,west_sichuan_plateau_holdout --years 2020,2021 --scale 500 --fixed-scale
```

Expected:

```text
Independent LULC label fetch: complete, 16 record(s), 0 failure(s)
```

If this fails because of network, GEE authentication, or sandbox boundaries, rerun with escalated approval rather than changing the area set.

- [ ] **Step 2: Fetch embeddings and context for Batch 2 candidates**

Run:

```powershell
python -m scripts.rse_revision.fetch_change_validation_embeddings --area-manifest data\independent_change_labels\paper58_holdout_areas_batch2.json --areas xiong_an_fringe_holdout,beibu_gulf_urban_holdout,songnen_plain_holdout,hexi_irrigation_holdout,changbai_margin_holdout,erlong_lake_margin_holdout,ordos_grassland_holdout,west_sichuan_plateau_holdout --years 2020,2021 --scale 500
```

Expected:

```text
Change-validation embedding fetch: complete, 16 grid(s), 8 context grid(s), 0 failure(s)
```

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

- `n_failures` should be `0`, or failures must remain explicitly listed.
- All Batch 2 area names should match the manifest exactly.

- [ ] **Step 4: Commit acquisition outputs**

Run:

```powershell
git add data/independent_change_labels/label_manifest.json data/independent_change_labels/embedding_manifest.json data/independent_change_labels/labels data/independent_change_labels/embeddings
git commit -m "data: fetch Paper58 Batch 2 holdout labels and embeddings"
```

---

### Task 6: Batch 2 Prediction Generation

**Files:**
- Generated outputs under:
  - `data/independent_change_labels/predicted/`
  - `data/independent_change_labels/prediction_readiness_report.json`

- [ ] **Step 1: Generate predictions for Batch 2 candidates**

Run:

```powershell
python -m scripts.rse_revision.generate_change_validation_predictions --embedding-dir data\independent_change_labels\embeddings --area-manifest data\independent_change_labels\paper58_holdout_areas_batch2.json
```

Expected:

```text
Change-validation prediction generation: complete, 8 prediction(s)
```

- [ ] **Step 2: Inspect the readiness report**

Run:

```powershell
Get-Content -Raw -LiteralPath data\independent_change_labels\prediction_readiness_report.json
```

Expected interpretation:

- `status = complete`
- `n_skipped = 0`, or every skipped area has an explicit reason

- [ ] **Step 3: Commit prediction outputs**

Run:

```powershell
git add data/independent_change_labels/predicted data/independent_change_labels/prediction_readiness_report.json
git commit -m "data: generate Paper58 Batch 2 holdout predictions"
```

---

### Task 7: Batch 2 Only Benchmark

**Files:**
- Generated outputs under:
  - `paper/rse_submission_paper58/benchmark_results_batch2/`

- [ ] **Step 1: Build the Batch 2 only registry**

Run:

```powershell
python -m scripts.paper58_benchmark.build_registry --holdout-manifest data\independent_change_labels\paper58_holdout_areas_batch2.json --filter-to-holdout-manifest --output-dir paper\rse_submission_paper58\benchmark_results_batch2
```

Expected:

```text
Benchmark registry: 8 candidate pair(s), 4 to 8 included pair(s)
```

- [ ] **Step 2: Run the Batch 2 provenance audit**

Run:

```powershell
python -m scripts.paper58_benchmark.audit_provenance --registry paper\rse_submission_paper58\benchmark_results_batch2\benchmark_registry.json --output-dir paper\rse_submission_paper58\benchmark_results_batch2
```

Expected:

```text
Benchmark provenance audit: 8 row(s), 0 invalid Tier 1 row(s)
```

- [ ] **Step 3: Evaluate the Batch 2 only benchmark**

Run:

```powershell
python -m scripts.paper58_benchmark.evaluate_benchmark --registry paper\rse_submission_paper58\benchmark_results_batch2\benchmark_registry.json --output-dir paper\rse_submission_paper58\benchmark_results_batch2 --n-boot 5000
```

Expected:

```text
Benchmark evaluation: 4 to 8 evaluated pair(s), gate status is pass, fail, or insufficient_tier1
```

- [ ] **Step 4: Generate Batch 2 figures**

Run:

```powershell
python -m scripts.paper58_benchmark.make_benchmark_figures --results-dir paper\rse_submission_paper58\benchmark_results_batch2 --figure-dir paper\rse_submission_paper58\benchmark_results_batch2\figures
```

Expected:

```text
Wrote Paper58 benchmark figures from D:\test\paper58-geofm-world-model-rl\.worktrees\paper58-benchmark\paper\rse_submission_paper58\benchmark_results_batch2
```

- [ ] **Step 5: Inspect the Batch 2 gate report**

Run:

```powershell
Get-Content -Raw -LiteralPath paper\rse_submission_paper58\benchmark_results_batch2\benchmark_gate_report.json
```

Stop interpretation:

- If `positive_tier1_strata < 3`, Batch 2 is insufficient or unstable.
- If either `tier1_primary_change.ci_low <= 0` or `tier1_spatial_change.ci_low <= 0`, Batch 2 is not supportive.
- If `status = pass`, Batch 2 independently supports the bounded claim.

- [ ] **Step 6: Commit Batch 2 only benchmark outputs**

Run:

```powershell
git add paper/rse_submission_paper58/benchmark_results_batch2
git commit -m "data: run Paper58 Batch 2 benchmark"
```

---

### Task 8: Combined Batch 1 + Batch 2 Benchmark

**Files:**
- Generated outputs under:
  - `paper/rse_submission_paper58/benchmark_results_combined/`

- [ ] **Step 1: Rebuild the combined manifest**

Run:

```powershell
python -m scripts.paper58_benchmark.build_combined_holdout_manifest --manifest data\independent_change_labels\paper58_holdout_areas.json --manifest data\independent_change_labels\paper58_holdout_areas_batch2.json --output data\independent_change_labels\paper58_holdout_areas_combined.json
```

Expected:

```text
Combined holdout manifest: 18 area(s)
```

- [ ] **Step 2: Build the combined registry**

Run:

```powershell
python -m scripts.paper58_benchmark.build_registry --holdout-manifest data\independent_change_labels\paper58_holdout_areas_combined.json --filter-to-holdout-manifest --output-dir paper\rse_submission_paper58\benchmark_results_combined
```

Expected:

```text
Benchmark registry: 18 candidate pair(s), 11 to 18 included pair(s)
```

- [ ] **Step 3: Run the combined provenance audit**

Run:

```powershell
python -m scripts.paper58_benchmark.audit_provenance --registry paper\rse_submission_paper58\benchmark_results_combined\benchmark_registry.json --output-dir paper\rse_submission_paper58\benchmark_results_combined
```

Expected:

```text
Benchmark provenance audit: 18 row(s), 0 invalid Tier 1 row(s)
```

- [ ] **Step 4: Evaluate the combined benchmark**

Run:

```powershell
python -m scripts.paper58_benchmark.evaluate_benchmark --registry paper\rse_submission_paper58\benchmark_results_combined\benchmark_registry.json --output-dir paper\rse_submission_paper58\benchmark_results_combined --n-boot 5000
```

Expected:

```text
Benchmark evaluation: 11 to 18 evaluated pair(s), gate status is pass, fail, or insufficient_tier1
```

- [ ] **Step 5: Generate combined figures**

Run:

```powershell
python -m scripts.paper58_benchmark.make_benchmark_figures --results-dir paper\rse_submission_paper58\benchmark_results_combined --figure-dir paper\rse_submission_paper58\benchmark_results_combined\figures
```

Expected:

```text
Wrote Paper58 benchmark figures from D:\test\paper58-geofm-world-model-rl\.worktrees\paper58-benchmark\paper\rse_submission_paper58\benchmark_results_combined
```

- [ ] **Step 6: Inspect the combined gate report**

Run:

```powershell
Get-Content -Raw -LiteralPath paper\rse_submission_paper58\benchmark_results_combined\benchmark_gate_report.json
```

Expected interpretation:

- Combined `status` may only be called strengthened evidence if Batch 2 only was not a failure hidden by pooling.
- Tier 1 region clusters should increase relative to Batch 1 alone.
- Positive Tier 1 strata should not decrease relative to Batch 1 alone.

- [ ] **Step 7: Commit combined benchmark outputs**

Run:

```powershell
git add data/independent_change_labels/paper58_holdout_areas_combined.json paper/rse_submission_paper58/benchmark_results_combined
git commit -m "data: run Paper58 combined Batch 1 and Batch 2 benchmark"
```

---

### Task 9: Handoff And Decision Record

**Files:**
- Modify: `docs/current_work_progress_2026-06-20.md`

- [ ] **Step 1: Append Batch 2 stability-check results to the handoff note**

Append a section to `docs/current_work_progress_2026-06-20.md` with:

```markdown
## Batch 2 Tier 1 Stability Check

Design:

```text
docs/superpowers/specs/2026-06-20-paper58-batch2-tier1-design.md
```

Plan:

```text
docs/superpowers/plans/2026-06-20-paper58-batch2-tier1-stability.md
```

Batch 2 only outputs:

```text
paper/rse_submission_paper58/benchmark_results_batch2
```

Combined outputs:

```text
paper/rse_submission_paper58/benchmark_results_combined
```

Decision rule:

- Batch 2 only is the primary stability check.
- Combined pass does not strengthen the evidence if Batch 2 only fails or remains insufficient.
```

Then add the actual Batch 2 only and combined gate statuses, evaluated Tier 1 counts, positive strata counts, and CI lower bounds from the generated gate reports.

- [ ] **Step 2: Run final relevant verification**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_holdouts.py tests/test_paper58_benchmark_registry.py tests/test_paper58_benchmark_evaluation.py tests/test_paper58_benchmark_figures.py tests/test_paper58_benchmark_provenance_audit.py tests/test_rse_revision_change_validation.py -q
```

Expected: PASS.

- [ ] **Step 3: Run diff check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 4: Commit the handoff update**

Run:

```powershell
git add docs/current_work_progress_2026-06-20.md
git commit -m "docs: record Paper58 Batch 2 stability check status"
```

---

## Self-Review Checklist

- Spec coverage: The plan adds a Batch 2 manifest, a combined-manifest utility, separate Batch 2 and combined benchmark outputs, explicit Batch 2 only versus combined interpretation, and a handoff note.
- Placeholder scan: All tasks contain concrete files, commands, expected outputs, and commit points.
- Type consistency: The plan reuses existing `load_holdout_manifest`, `evaluate_benchmark`, `make_benchmark_figures`, and `--output-dir` / `--holdout-manifest` / `--area-manifest` interfaces, and adds only one opt-in `build_registry` filter argument, `filter_to_holdout_manifest`, surfaced as `--filter-to-holdout-manifest`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-20-paper58-batch2-tier1-stability.md`. Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
