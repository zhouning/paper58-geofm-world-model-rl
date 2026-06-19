# Paper58 External Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, reproducible Paper58 benchmark pipeline that turns existing cached labels, predictions, and embeddings into a strict registry, benchmark metrics, gate report, and paper-candidate diagnostic figures.

**Architecture:** Add a new `scripts/paper58_benchmark/` package that stays separate from `scripts/rse_revision/` and writes all new outputs to `paper/rse_submission_paper58/benchmark_results/`. The pipeline is file-first: build a registry from actual files, evaluate only QC-passing rows, compute pair/region-level statistics, then render figures only from benchmark outputs.

**Tech Stack:** Python 3, NumPy, pandas-style CSV via stdlib `csv`, scikit-learn for the ridge embedding baseline, matplotlib for figures, pytest for tests.

---

## File Structure

Create these files:

- `scripts/paper58_benchmark/__init__.py`: package marker and public version string.
- `scripts/paper58_benchmark/schema.py`: constants, area strata, tier rules, dataclasses, JSON/CSV helpers.
- `scripts/paper58_benchmark/baselines.py`: persistence, spatial shuffle, label-only transition prior, leave-one-region temporal prior, ridge embedding delta baseline.
- `scripts/paper58_benchmark/build_registry.py`: scan local files, infer candidate area-year pairs, run QC, write registry JSON/CSV.
- `scripts/paper58_benchmark/statistics.py`: bootstrap, sign tests, stratified summaries, gate report.
- `scripts/paper58_benchmark/evaluate_benchmark.py`: load registry, evaluate rows, write benchmark metrics and summaries.
- `scripts/paper58_benchmark/make_benchmark_figures.py`: generate figures from benchmark outputs, failing if inputs are absent.

Create these tests:

- `tests/test_paper58_benchmark_baselines.py`
- `tests/test_paper58_benchmark_registry.py`
- `tests/test_paper58_benchmark_statistics.py`
- `tests/test_paper58_benchmark_evaluation.py`
- `tests/test_paper58_benchmark_figures.py`

Do not overwrite any files in `paper/rse_submission_paper58/revision_results/`.

---

### Task 1: Package Schema And Registry Row Contract

**Files:**
- Create: `scripts/paper58_benchmark/__init__.py`
- Create: `scripts/paper58_benchmark/schema.py`
- Test: `tests/test_paper58_benchmark_registry.py`

- [ ] **Step 1: Write the failing schema tests**

Add this initial content to `tests/test_paper58_benchmark_registry.py`:

```python
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
```

- [ ] **Step 2: Run the schema tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_registry.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.paper58_benchmark'`.

- [ ] **Step 3: Add the package schema**

Create `scripts/paper58_benchmark/__init__.py`:

```python
"""Strict external benchmark pipeline for Paper58."""

__all__ = ["__version__"]
__version__ = "0.1.0"
```

Create `scripts/paper58_benchmark/schema.py`:

```python
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_DIR = ROOT / "paper" / "rse_submission_paper58" / "benchmark_results"
DEFAULT_LABELS_DIR = ROOT / "data" / "independent_change_labels" / "labels"
DEFAULT_PREDICTIONS_DIR = ROOT / "data" / "independent_change_labels" / "predicted"
DEFAULT_INDEPENDENT_EMBEDDINGS_DIR = ROOT / "data" / "independent_change_labels" / "embeddings"
DEFAULT_EXPERIMENT_DATA_DIR = ROOT / "experiments" / "paper8" / "data"

DEVELOPMENT_AREAS = {"banzhucun", "bishan", "heping"}

AREA_STRATA = {
    "yangtze_delta": "Urban",
    "jing_jin_ji": "Urban",
    "chengdu_plain": "Urban",
    "pearl_river": "Urban",
    "northeast_plain": "Agriculture",
    "north_china_plain": "Agriculture",
    "jianghan_plain": "Agriculture",
    "hetao": "Agriculture",
    "daxinganling": "Forest",
    "wuyi_mountain": "Forest",
    "qinghai_edge": "Plateau",
    "guanzhong": "Mixed",
    "minnan_coast": "Mixed",
    "bishan": "Mixed",
    "banzhucun": "Mixed",
    "heping": "Mixed",
    "poyang_lake": "Wetland",
    "yunnan_eco": "Ecology",
}


@dataclass(frozen=True)
class BenchmarkRow:
    area: str
    start_year: int
    end_year: int
    tier: str
    stratum: str
    label_start_path: Path | None
    label_end_path: Path | None
    prediction_path: Path | None
    embedding_start_path: Path | None
    embedding_end_path: Path | None
    context_path: Path | None
    label_shape: tuple[int, ...] | None
    prediction_shape: tuple[int, ...] | None
    embedding_shape: tuple[int, ...] | None
    n_pixels: int
    true_change_pixels: int
    true_change_pct: float
    qc_status: str
    excluded_reason: str


def assign_tier(area: str) -> str:
    return "tier2" if area.lower() in DEVELOPMENT_AREAS else "tier1"


def area_stratum(area: str) -> str:
    return AREA_STRATA.get(area.lower(), "Unknown")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def row_to_dict(row: BenchmarkRow) -> dict[str, Any]:
    return _json_ready(asdict(row))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
```

- [ ] **Step 4: Run the schema tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_registry.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add scripts/paper58_benchmark/__init__.py scripts/paper58_benchmark/schema.py tests/test_paper58_benchmark_registry.py
git commit -m "feat: add Paper58 benchmark schema"
```

---

### Task 2: Strong Baselines

**Files:**
- Create: `scripts/paper58_benchmark/baselines.py`
- Modify: `tests/test_paper58_benchmark_baselines.py`

- [ ] **Step 1: Write failing baseline tests**

Create `tests/test_paper58_benchmark_baselines.py`:

```python
import numpy as np
import pytest

from scripts.paper58_benchmark.baselines import (
    fit_linear_embedding_delta,
    label_only_transition_prior,
    leave_one_region_temporal_prior,
    persistence_prediction,
    spatial_shuffle_prediction,
)


def test_persistence_prediction_returns_start_map_copy():
    start = np.array([[1, 2], [3, 4]], dtype=np.int32)
    pred = persistence_prediction(start)

    assert np.array_equal(pred, start)
    pred[0, 0] = 9
    assert start[0, 0] == 1


def test_spatial_shuffle_preserves_histogram_and_shape():
    pred = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int32)
    shuffled = spatial_shuffle_prediction(pred, seed=7)

    assert shuffled.shape == pred.shape
    assert sorted(shuffled.ravel().tolist()) == sorted(pred.ravel().tolist())
    assert not np.array_equal(shuffled, pred)


def test_label_only_transition_prior_uses_leave_out_distribution():
    target_start = np.ones((2, 4), dtype=np.int32)
    train_start = np.ones((2, 4), dtype=np.int32)
    train_end = np.array([[1, 2, 1, 2], [1, 2, 1, 2]], dtype=np.int32)

    pred = label_only_transition_prior(target_start, [(train_start, train_end)], seed=7)

    assert pred.shape == target_start.shape
    assert np.count_nonzero(pred == 1) == 4
    assert np.count_nonzero(pred == 2) == 4


def test_leave_one_region_temporal_prior_uses_other_regions_change_rate():
    target_start = np.full((2, 5), 1, dtype=np.int32)
    training_rows = [
        {
            "area": "source_a",
            "start": np.full((2, 5), 1, dtype=np.int32),
            "end": np.array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1]], dtype=np.int32),
        }
    ]

    pred = leave_one_region_temporal_prior("target", target_start, training_rows, seed=11)

    assert pred.shape == target_start.shape
    assert np.count_nonzero(pred != target_start) == 4


def test_linear_embedding_delta_predicts_residual_shape():
    train_start = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    train_end = train_start + np.array([[[0.1, 0.0], [0.0, -0.1]]], dtype=np.float32)
    test_start = np.array([[[1.0, 0.0]]], dtype=np.float32)

    pred = fit_linear_embedding_delta(train_start, train_end, test_start, alpha=1e-3)

    assert pred.shape == test_start.shape
    assert np.isfinite(pred).all()
    assert pred[0, 0, 0] == pytest.approx(1.1, abs=0.05)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_baselines.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.paper58_benchmark.baselines'`.

- [ ] **Step 3: Implement baselines**

Create `scripts/paper58_benchmark/baselines.py`:

```python
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def persistence_prediction(start_map: np.ndarray) -> np.ndarray:
    return np.asarray(start_map).copy()


def spatial_shuffle_prediction(prediction: np.ndarray, seed: int = 20260617) -> np.ndarray:
    pred = np.asarray(prediction)
    rng = np.random.default_rng(seed)
    return rng.permutation(pred.ravel()).reshape(pred.shape)


def label_only_transition_prior(
    target_start: np.ndarray,
    training_pairs: list[tuple[np.ndarray, np.ndarray]],
    seed: int = 20260618,
) -> np.ndarray:
    target = np.asarray(target_start)
    predicted = target.copy()
    transitions: dict[int, dict[int, int]] = {}
    for train_start, train_end in training_pairs:
        if train_start.shape != train_end.shape:
            continue
        for start_cls, end_cls in zip(train_start.ravel(), train_end.ravel()):
            start_key = int(start_cls)
            end_key = int(end_cls)
            transitions.setdefault(start_key, {})
            transitions[start_key][end_key] = transitions[start_key].get(end_key, 0) + 1
    if not transitions:
        return predicted

    rng = np.random.default_rng(seed)
    flat_start = target.ravel()
    flat_pred = predicted.ravel()
    for start_cls in sorted(np.unique(flat_start)):
        class_counts = transitions.get(int(start_cls))
        if not class_counts:
            continue
        indices = np.flatnonzero(flat_start == start_cls)
        shuffled_indices = rng.permutation(indices)
        end_classes = sorted(class_counts)
        counts = np.array([class_counts[end_cls] for end_cls in end_classes], dtype=float)
        fractions = counts / counts.sum()
        allocation = np.floor(fractions * indices.size).astype(int)
        remainder = int(indices.size - allocation.sum())
        if remainder > 0:
            residual = fractions * indices.size - allocation
            for position in np.argsort(-residual)[:remainder]:
                allocation[position] += 1
        cursor = 0
        for end_cls, n_assign in zip(end_classes, allocation):
            selected = shuffled_indices[cursor : cursor + int(n_assign)]
            flat_pred[selected] = end_cls
            cursor += int(n_assign)
    return flat_pred.reshape(target.shape)


def leave_one_region_temporal_prior(
    target_area: str,
    target_start: np.ndarray,
    training_rows: list[dict],
    seed: int = 20260619,
) -> np.ndarray:
    target = np.asarray(target_start)
    rates = []
    changed_end_classes = []
    for row in training_rows:
        if str(row.get("area")) == target_area:
            continue
        start = np.asarray(row["start"])
        end = np.asarray(row["end"])
        if start.shape != end.shape:
            continue
        changed = end != start
        rates.append(float(np.mean(changed)))
        changed_end_classes.extend(int(value) for value in end[changed].ravel())
    if not rates or not changed_end_classes:
        return target.copy()

    rng = np.random.default_rng(seed)
    n_change = int(round(float(np.mean(rates)) * target.size))
    n_change = max(0, min(n_change, target.size))
    pred = target.copy().ravel()
    if n_change == 0:
        return pred.reshape(target.shape)
    indices = rng.choice(np.arange(target.size), size=n_change, replace=False)
    pred[indices] = rng.choice(np.array(changed_end_classes, dtype=pred.dtype), size=n_change, replace=True)
    return pred.reshape(target.shape)


def fit_linear_embedding_delta(
    train_start: np.ndarray,
    train_end: np.ndarray,
    test_start: np.ndarray,
    alpha: float = 1e-3,
) -> np.ndarray:
    train_start_arr = np.asarray(train_start, dtype=np.float32)
    train_end_arr = np.asarray(train_end, dtype=np.float32)
    test_start_arr = np.asarray(test_start, dtype=np.float32)
    if train_start_arr.shape != train_end_arr.shape:
        raise ValueError(f"Shape mismatch: train_start={train_start_arr.shape}, train_end={train_end_arr.shape}")
    feature_dim = train_start_arr.shape[-1]
    x_train = train_start_arr.reshape(-1, feature_dim)
    y_delta = (train_end_arr - train_start_arr).reshape(-1, feature_dim)
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_delta)
    x_test = test_start_arr.reshape(-1, feature_dim)
    delta = model.predict(x_test).reshape(test_start_arr.shape)
    return test_start_arr + delta.astype(np.float32, copy=False)
```

- [ ] **Step 4: Run baseline tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_baselines.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add scripts/paper58_benchmark/baselines.py tests/test_paper58_benchmark_baselines.py
git commit -m "feat: add Paper58 benchmark baselines"
```

---

### Task 3: Registry Scanning And Quality Control

**Files:**
- Create: `scripts/paper58_benchmark/build_registry.py`
- Modify: `tests/test_paper58_benchmark_registry.py`

- [ ] **Step 1: Add failing registry scanner tests**

Append these tests to `tests/test_paper58_benchmark_registry.py`:

```python
import numpy as np

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

    rows = build_registry(
        labels_dir=labels,
        predictions_dir=predicted,
        independent_embeddings_dir=embeddings,
        experiment_data_dir=experiment_data,
        output_dir=output,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.area == "external"
    assert row.tier == "tier1"
    assert row.qc_status == "include"
    assert row.true_change_pixels == 2
    assert row.embedding_shape == (2, 2, 64)
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
```

- [ ] **Step 2: Run registry tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_registry.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.paper58_benchmark.build_registry'`.

- [ ] **Step 3: Implement registry scanner**

Create `scripts/paper58_benchmark/build_registry.py` with these functions:

```python
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from scripts.paper58_benchmark.schema import (
    DEFAULT_BENCHMARK_DIR,
    DEFAULT_EXPERIMENT_DATA_DIR,
    DEFAULT_INDEPENDENT_EMBEDDINGS_DIR,
    DEFAULT_LABELS_DIR,
    DEFAULT_PREDICTIONS_DIR,
    BenchmarkRow,
    area_stratum,
    assign_tier,
    row_to_dict,
    write_csv,
    write_json,
)


LABEL_RE = re.compile(r"^(?P<area>.+)_lulc_(?P<year>\d{4})\.npy$")
PRED_RE = re.compile(r"^(?P<area>.+)_lulc_pred_(?P<start_year>\d{4})_(?P<end_year>\d{4})\.npy$")
REGISTRY_FIELDS = list(row_to_dict(BenchmarkRow("", 0, 0, "", "", None, None, None, None, None, None, None, None, None, 0, 0, 0.0, "", "")).keys())


def parse_label_filename(name: str) -> tuple[str, int] | None:
    match = LABEL_RE.match(name)
    if match is None:
        return None
    return match.group("area"), int(match.group("year"))


def parse_prediction_filename(name: str) -> tuple[str, int, int] | None:
    match = PRED_RE.match(name)
    if match is None:
        return None
    return match.group("area"), int(match.group("start_year")), int(match.group("end_year"))


def _discover_labels(labels_dir: Path) -> dict[str, dict[int, Path]]:
    discovered: dict[str, dict[int, Path]] = {}
    for path in sorted(labels_dir.glob("*_lulc_*.npy")):
        parsed = parse_label_filename(path.name)
        if parsed is None:
            continue
        area, year = parsed
        discovered.setdefault(area, {})[year] = path
    return discovered


def _discover_predictions(predictions_dir: Path) -> list[tuple[str, int, int, Path]]:
    rows = []
    for path in sorted(predictions_dir.glob("*_lulc_pred_*_*.npy")):
        parsed = parse_prediction_filename(path.name)
        if parsed is None:
            continue
        area, start_year, end_year = parsed
        rows.append((area, start_year, end_year, path))
    return rows


def _embedding_area_aliases(area: str) -> list[str]:
    aliases = [area]
    if area == "banzhucun":
        aliases.append("village")
    return aliases


def _find_embedding_path(area: str, year: int, independent_embeddings_dir: Path, experiment_data_dir: Path) -> Path | None:
    candidates = []
    for alias in _embedding_area_aliases(area):
        candidates.append(independent_embeddings_dir / f"{alias}_emb_{year}.npy")
        candidates.append(experiment_data_dir / f"{alias}_emb_{year}.npy")
        candidates.append(experiment_data_dir / alias / f"{alias}_emb_{year}.npy")
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _find_context_path(area: str, independent_embeddings_dir: Path, experiment_data_dir: Path) -> Path | None:
    candidates = []
    for alias in _embedding_area_aliases(area):
        candidates.append(independent_embeddings_dir / f"{alias}_context.npy")
        candidates.append(experiment_data_dir / f"{alias}_context.npy")
        candidates.append(experiment_data_dir / alias / f"{alias}_context.npy")
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _shape(path: Path | None) -> tuple[int, ...] | None:
    if path is None:
        return None
    return tuple(np.load(path, mmap_mode="r").shape)


def _class_collapse(prediction: np.ndarray, true_change_pixels: int) -> bool:
    return true_change_pixels > 0 and np.unique(prediction).size <= 1


def _build_row(
    area: str,
    start_year: int,
    end_year: int,
    prediction_path: Path,
    labels: dict[str, dict[int, Path]],
    independent_embeddings_dir: Path,
    experiment_data_dir: Path,
) -> BenchmarkRow:
    label_start_path = labels.get(area, {}).get(start_year)
    label_end_path = labels.get(area, {}).get(end_year)
    embedding_start_path = _find_embedding_path(area, start_year, independent_embeddings_dir, experiment_data_dir)
    embedding_end_path = _find_embedding_path(area, end_year, independent_embeddings_dir, experiment_data_dir)
    context_path = _find_context_path(area, independent_embeddings_dir, experiment_data_dir)
    qc_status = "include"
    excluded_reason = ""
    n_pixels = 0
    true_change_pixels = 0
    true_change_pct = 0.0
    label_shape = None
    prediction_shape = _shape(prediction_path)
    embedding_shape = _shape(embedding_start_path)

    if label_start_path is None or label_end_path is None:
        qc_status = "exclude"
        excluded_reason = "missing_label"
    else:
        start = np.load(label_start_path)
        end = np.load(label_end_path)
        pred = np.load(prediction_path)
        label_shape = tuple(start.shape)
        prediction_shape = tuple(pred.shape)
        n_pixels = int(start.size)
        if start.shape != end.shape:
            qc_status = "exclude"
            excluded_reason = "label_shape_mismatch"
        elif start.shape != pred.shape:
            qc_status = "exclude"
            excluded_reason = "label_prediction_shape_mismatch"
        elif start.size == 0:
            qc_status = "exclude"
            excluded_reason = "empty_arrays"
        else:
            true_change = end != start
            true_change_pixels = int(np.count_nonzero(true_change))
            true_change_pct = float(true_change_pixels / start.size)
            if true_change_pixels == 0:
                qc_status = "negative_control"
                excluded_reason = "zero_reference_change"
            elif _class_collapse(pred, true_change_pixels):
                qc_status = "exclude"
                excluded_reason = "class_collapse"

    if qc_status == "include" and (embedding_start_path is None or embedding_end_path is None):
        qc_status = "exclude"
        excluded_reason = "missing_embedding"

    return BenchmarkRow(
        area=area,
        start_year=start_year,
        end_year=end_year,
        tier=assign_tier(area),
        stratum=area_stratum(area),
        label_start_path=label_start_path,
        label_end_path=label_end_path,
        prediction_path=prediction_path,
        embedding_start_path=embedding_start_path,
        embedding_end_path=embedding_end_path,
        context_path=context_path,
        label_shape=label_shape,
        prediction_shape=prediction_shape,
        embedding_shape=embedding_shape,
        n_pixels=n_pixels,
        true_change_pixels=true_change_pixels,
        true_change_pct=true_change_pct,
        qc_status=qc_status,
        excluded_reason=excluded_reason,
    )


def build_registry(
    labels_dir: Path = DEFAULT_LABELS_DIR,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
    independent_embeddings_dir: Path = DEFAULT_INDEPENDENT_EMBEDDINGS_DIR,
    experiment_data_dir: Path = DEFAULT_EXPERIMENT_DATA_DIR,
    output_dir: Path = DEFAULT_BENCHMARK_DIR,
) -> list[BenchmarkRow]:
    labels = _discover_labels(Path(labels_dir))
    predictions = _discover_predictions(Path(predictions_dir))
    rows = [
        _build_row(
            area=area,
            start_year=start_year,
            end_year=end_year,
            prediction_path=prediction_path,
            labels=labels,
            independent_embeddings_dir=Path(independent_embeddings_dir),
            experiment_data_dir=Path(experiment_data_dir),
        )
        for area, start_year, end_year, prediction_path in predictions
    ]
    dict_rows = [row_to_dict(row) for row in rows]
    output_dir = Path(output_dir)
    write_json(output_dir / "benchmark_registry.json", {"rows": dict_rows})
    write_csv(output_dir / "benchmark_registry.csv", dict_rows, REGISTRY_FIELDS)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Paper58 external benchmark registry.")
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--independent-embeddings-dir", type=Path, default=DEFAULT_INDEPENDENT_EMBEDDINGS_DIR)
    parser.add_argument("--experiment-data-dir", type=Path, default=DEFAULT_EXPERIMENT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    args = parser.parse_args()
    rows = build_registry(
        labels_dir=args.labels_dir,
        predictions_dir=args.predictions_dir,
        independent_embeddings_dir=args.independent_embeddings_dir,
        experiment_data_dir=args.experiment_data_dir,
        output_dir=args.output_dir,
    )
    included = sum(row.qc_status == "include" for row in rows)
    print(f"Benchmark registry: {len(rows)} candidate pair(s), {included} included pair(s)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run registry tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_registry.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add scripts/paper58_benchmark/build_registry.py tests/test_paper58_benchmark_registry.py
git commit -m "feat: add Paper58 benchmark registry builder"
```

---

### Task 4: Clustered Statistics And Gate Report

**Files:**
- Create: `scripts/paper58_benchmark/statistics.py`
- Test: `tests/test_paper58_benchmark_statistics.py`

- [ ] **Step 1: Write failing statistics tests**

Create `tests/test_paper58_benchmark_statistics.py`:

```python
import pytest

from scripts.paper58_benchmark.statistics import (
    clustered_bootstrap_ci,
    gate_report,
    paired_sign_test,
    summarize_by_tier_and_stratum,
)


def test_clustered_bootstrap_resamples_regions_not_pixels():
    rows = [
        {"area": "a", "tier": "tier1", "primary_change_advantage": 0.10},
        {"area": "a", "tier": "tier1", "primary_change_advantage": 0.20},
        {"area": "b", "tier": "tier1", "primary_change_advantage": -0.05},
    ]

    ci = clustered_bootstrap_ci(rows, "primary_change_advantage", cluster_key="area", n_boot=200, seed=3)

    assert ci["n_rows"] == 3
    assert ci["n_clusters"] == 2
    assert ci["mean"] == pytest.approx((0.10 + 0.20 - 0.05) / 3)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_paired_sign_test_counts_directions():
    result = paired_sign_test([0.1, 0.2, -0.1, 0.0])

    assert result["n_positive"] == 2
    assert result["n_negative"] == 1
    assert result["n_tie"] == 1
    assert 0.0 <= result["two_sided_p"] <= 1.0


def test_summarize_by_tier_and_stratum_separates_groups():
    rows = [
        {"tier": "tier1", "stratum": "Wetland", "primary_change_advantage": 0.2},
        {"tier": "tier1", "stratum": "Forest", "primary_change_advantage": 0.1},
        {"tier": "tier2", "stratum": "Mixed", "primary_change_advantage": -0.1},
    ]

    summary = summarize_by_tier_and_stratum(rows, "primary_change_advantage")

    assert summary["by_tier"]["tier1"]["n"] == 2
    assert summary["by_tier"]["tier1"]["mean"] == pytest.approx(0.15)
    assert summary["by_stratum"]["Wetland"]["n"] == 1
    assert summary["by_stratum"]["Mixed"]["mean"] == pytest.approx(-0.1)


def test_gate_report_requires_positive_tier1_primary_and_spatial_intervals():
    rows = [
        {
            "area": "a",
            "tier": "tier1",
            "stratum": "Wetland",
            "primary_change_advantage": 0.20,
            "spatial_change_advantage": 0.10,
            "embedding_advantage": 0.01,
        },
        {
            "area": "b",
            "tier": "tier1",
            "stratum": "Forest",
            "primary_change_advantage": 0.15,
            "spatial_change_advantage": 0.08,
            "embedding_advantage": 0.02,
        },
        {
            "area": "c",
            "tier": "tier1",
            "stratum": "Urban",
            "primary_change_advantage": 0.12,
            "spatial_change_advantage": 0.07,
            "embedding_advantage": 0.01,
        },
    ]

    report = gate_report(rows, n_boot=200, seed=9)

    assert report["status"] == "pass"
    assert report["tier1_primary_change"]["ci_low"] > 0
    assert report["tier1_spatial_change"]["ci_low"] > 0
    assert report["positive_tier1_strata"] == 3
```

- [ ] **Step 2: Run statistics tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_statistics.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.paper58_benchmark.statistics'`.

- [ ] **Step 3: Implement statistics**

Create `scripts/paper58_benchmark/statistics.py` with functions named in the tests. Use these rules:

```python
from __future__ import annotations

import math
from statistics import mean, median

import numpy as np


def _finite_values(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def clustered_bootstrap_ci(
    rows: list[dict],
    value_key: str,
    cluster_key: str = "area",
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    clean_rows = [row for row in rows if isinstance(row.get(value_key), (int, float)) and math.isfinite(float(row[value_key]))]
    if not clean_rows:
        return {"n_rows": 0, "n_clusters": 0, "mean": None, "median": None, "ci_low": None, "ci_high": None}
    clusters: dict[str, list[dict]] = {}
    for row in clean_rows:
        clusters.setdefault(str(row.get(cluster_key, "")), []).append(row)
    cluster_names = sorted(clusters)
    values = [float(row[value_key]) for row in clean_rows]
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(cluster_names, size=len(cluster_names), replace=True)
        sampled_values = []
        for cluster in sampled_clusters:
            sampled_values.extend(float(row[value_key]) for row in clusters[cluster])
        boot_means.append(float(np.mean(sampled_values)))
    return {
        "n_rows": int(len(clean_rows)),
        "n_clusters": int(len(cluster_names)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
    }


def paired_sign_test(values: list[float]) -> dict:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    n_positive = sum(v > 0 for v in clean)
    n_negative = sum(v < 0 for v in clean)
    n_tie = sum(v == 0 for v in clean)
    n_effective = n_positive + n_negative
    if n_effective == 0:
        p_value = 1.0
    else:
        k = min(n_positive, n_negative)
        tail = sum(math.comb(n_effective, i) for i in range(k + 1)) / (2 ** n_effective)
        p_value = min(1.0, 2.0 * tail)
    return {
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "n_tie": int(n_tie),
        "n_effective": int(n_effective),
        "two_sided_p": float(p_value),
    }


def _simple_summary(rows: list[dict], value_key: str) -> dict:
    values = _finite_values(rows, value_key)
    return {
        "n": len(values),
        "mean": float(mean(values)) if values else None,
        "median": float(median(values)) if values else None,
        "n_positive": int(sum(v > 0 for v in values)),
        "n_negative": int(sum(v < 0 for v in values)),
    }


def summarize_by_tier_and_stratum(rows: list[dict], value_key: str) -> dict:
    by_tier: dict[str, list[dict]] = {}
    by_stratum: dict[str, list[dict]] = {}
    for row in rows:
        by_tier.setdefault(str(row.get("tier", "unknown")), []).append(row)
        by_stratum.setdefault(str(row.get("stratum", "Unknown")), []).append(row)
    return {
        "by_tier": {key: _simple_summary(group, value_key) for key, group in sorted(by_tier.items())},
        "by_stratum": {key: _simple_summary(group, value_key) for key, group in sorted(by_stratum.items())},
    }


def gate_report(rows: list[dict], n_boot: int = 5000, seed: int = 42) -> dict:
    tier1 = [row for row in rows if row.get("tier") == "tier1"]
    primary = clustered_bootstrap_ci(tier1, "primary_change_advantage", n_boot=n_boot, seed=seed)
    spatial = clustered_bootstrap_ci(tier1, "spatial_change_advantage", n_boot=n_boot, seed=seed + 1)
    embedding = clustered_bootstrap_ci(tier1, "embedding_advantage", n_boot=n_boot, seed=seed + 2)
    strata = summarize_by_tier_and_stratum(tier1, "primary_change_advantage")["by_stratum"]
    positive_strata = sum((item.get("mean") is not None and item["mean"] > 0) for item in strata.values())
    primary_pass = primary["ci_low"] is not None and primary["ci_low"] > 0
    spatial_pass = spatial["ci_low"] is not None and spatial["ci_low"] > 0
    strata_pass = positive_strata >= 3
    status = "pass" if primary_pass and spatial_pass and strata_pass else "fail"
    if not tier1:
        status = "insufficient_tier1"
    return {
        "status": status,
        "tier1_primary_change": primary,
        "tier1_spatial_change": spatial,
        "tier1_embedding": embedding,
        "positive_tier1_strata": int(positive_strata),
        "required_positive_tier1_strata": 3,
        "primary_gate_pass": bool(primary_pass),
        "spatial_gate_pass": bool(spatial_pass),
        "strata_gate_pass": bool(strata_pass),
    }
```

- [ ] **Step 4: Run statistics tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_statistics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add scripts/paper58_benchmark/statistics.py tests/test_paper58_benchmark_statistics.py
git commit -m "feat: add Paper58 benchmark statistics"
```

---

### Task 5: Benchmark Evaluation Pipeline

**Files:**
- Create: `scripts/paper58_benchmark/evaluate_benchmark.py`
- Test: `tests/test_paper58_benchmark_evaluation.py`

- [ ] **Step 1: Write failing evaluation tests**

Create `tests/test_paper58_benchmark_evaluation.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.paper58_benchmark.evaluate_benchmark import (
    binary_change_metrics,
    evaluate_benchmark,
    evaluate_registry_row,
)


def test_binary_change_metrics_reports_f1():
    true_change = np.array([[False, True], [True, False]])
    pred_change = np.array([[False, True], [False, True]])

    result = binary_change_metrics(true_change, pred_change)

    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_evaluate_registry_row_computes_primary_and_spatial_advantages(tmp_path: Path):
    labels = tmp_path / "labels"
    predicted = tmp_path / "predicted"
    embeddings = tmp_path / "embeddings"
    labels.mkdir()
    predicted.mkdir()
    embeddings.mkdir()
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
    np.save(labels / "external_lulc_2020.npy", start)
    np.save(labels / "external_lulc_2021.npy", end)
    np.save(predicted / "external_lulc_pred_2020_2021.npy", pred)
    np.save(embeddings / "external_emb_2020.npy", np.zeros((2, 2, 64), dtype=np.float32))
    np.save(embeddings / "external_emb_2021.npy", np.ones((2, 2, 64), dtype=np.float32))

    row = {
        "area": "external",
        "start_year": 2020,
        "end_year": 2021,
        "tier": "tier1",
        "stratum": "Wetland",
        "label_start_path": str(labels / "external_lulc_2020.npy"),
        "label_end_path": str(labels / "external_lulc_2021.npy"),
        "prediction_path": str(predicted / "external_lulc_pred_2020_2021.npy"),
        "embedding_start_path": str(embeddings / "external_emb_2020.npy"),
        "embedding_end_path": str(embeddings / "external_emb_2021.npy"),
        "qc_status": "include",
        "excluded_reason": "",
    }

    metrics = evaluate_registry_row(row, transition_training_pairs=[], temporal_training_rows=[])

    assert metrics["model_change_f1"] == pytest.approx(2 / 3)
    assert metrics["persistence_change_f1"] == pytest.approx(0.0)
    assert metrics["primary_change_advantage"] >= 0.0
    assert "spatial_change_advantage" in metrics
    assert "embedding_advantage" in metrics


def test_evaluate_benchmark_writes_outputs_and_does_not_pool_tiers(tmp_path: Path):
    labels = tmp_path / "labels"
    predicted = tmp_path / "predicted"
    embeddings = tmp_path / "embeddings"
    output = tmp_path / "out"
    for path in (labels, predicted, embeddings, output):
        path.mkdir()

    for area, tier in [("external", "tier1"), ("bishan", "tier2")]:
        start = np.array([[1, 1], [2, 2]], dtype=np.int32)
        end = np.array([[1, 2], [2, 3]], dtype=np.int32)
        pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
        np.save(labels / f"{area}_lulc_2020.npy", start)
        np.save(labels / f"{area}_lulc_2021.npy", end)
        np.save(predicted / f"{area}_lulc_pred_2020_2021.npy", pred)
        np.save(embeddings / f"{area}_emb_2020.npy", np.zeros((2, 2, 64), dtype=np.float32))
        np.save(embeddings / f"{area}_emb_2021.npy", np.ones((2, 2, 64), dtype=np.float32))

    registry = {
        "rows": [
            {
                "area": area,
                "start_year": 2020,
                "end_year": 2021,
                "tier": tier,
                "stratum": "Wetland" if tier == "tier1" else "Mixed",
                "label_start_path": str(labels / f"{area}_lulc_2020.npy"),
                "label_end_path": str(labels / f"{area}_lulc_2021.npy"),
                "prediction_path": str(predicted / f"{area}_lulc_pred_2020_2021.npy"),
                "embedding_start_path": str(embeddings / f"{area}_emb_2020.npy"),
                "embedding_end_path": str(embeddings / f"{area}_emb_2021.npy"),
                "qc_status": "include",
                "excluded_reason": "",
            }
            for area, tier in [("external", "tier1"), ("bishan", "tier2")]
        ]
    }
    registry_path = output / "benchmark_registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    result = evaluate_benchmark(registry_path=registry_path, output_dir=output, n_boot=100)

    assert result["summary"]["n_evaluated"] == 2
    assert result["summary_by_tier"]["tier1"]["n"] == 1
    assert result["summary_by_tier"]["tier2"]["n"] == 1
    assert (output / "benchmark_metrics_by_pair.csv").exists()
    assert (output / "benchmark_gate_report.json").exists()
```

- [ ] **Step 2: Run evaluation tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_evaluation.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.paper58_benchmark.evaluate_benchmark'`.

- [ ] **Step 3: Implement evaluation pipeline**

Create `scripts/paper58_benchmark/evaluate_benchmark.py`:

```python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from scripts.paper58_benchmark.baselines import (
    label_only_transition_prior,
    leave_one_region_temporal_prior,
    persistence_prediction,
    spatial_shuffle_prediction,
)
from scripts.paper58_benchmark.schema import DEFAULT_BENCHMARK_DIR, write_csv, write_json
from scripts.paper58_benchmark.statistics import gate_report, summarize_by_tier_and_stratum


METRIC_FIELDS = [
    "area",
    "start_year",
    "end_year",
    "tier",
    "stratum",
    "n_pixels",
    "true_change_pixels",
    "true_change_pct",
    "model_change_precision",
    "model_change_recall",
    "model_change_f1",
    "persistence_change_f1",
    "spatial_shuffle_change_f1",
    "transition_prior_change_f1",
    "temporal_prior_change_f1",
    "best_non_neural_change_f1",
    "primary_change_advantage",
    "spatial_change_advantage",
    "embedding_model_cosine",
    "embedding_persistence_cosine",
    "embedding_advantage",
]


def _path(value: object) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _load_array(path_value: object) -> np.ndarray:
    path = _path(path_value)
    if path is None:
        raise FileNotFoundError("Missing array path in benchmark registry row")
    return np.load(path)


def binary_change_metrics(true_change: np.ndarray, pred_change: np.ndarray) -> dict:
    true = true_change.astype(bool).ravel()
    pred = pred_change.astype(bool).ravel()
    tp = int(np.count_nonzero(true & pred))
    fp = int(np.count_nonzero(~true & pred))
    fn = int(np.count_nonzero(true & ~pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    a2 = a.reshape(-1, a.shape[-1]).astype(float)
    b2 = b.reshape(-1, b.shape[-1]).astype(float)
    numerator = np.sum(a2 * b2, axis=1)
    denom = np.linalg.norm(a2, axis=1) * np.linalg.norm(b2, axis=1)
    valid = denom > 0
    return float(np.mean(numerator[valid] / denom[valid])) if np.any(valid) else 0.0


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _transition_training_pairs(rows: list[dict], target: dict, target_shape: tuple[int, ...]) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for row in rows:
        if (
            row.get("area") == target.get("area")
            and int(row.get("start_year")) == int(target.get("start_year"))
            and int(row.get("end_year")) == int(target.get("end_year"))
        ):
            continue
        if row.get("qc_status") != "include":
            continue
        start_path = _path(row.get("label_start_path"))
        end_path = _path(row.get("label_end_path"))
        if start_path is None or end_path is None or not start_path.exists() or not end_path.exists():
            continue
        start = np.load(start_path)
        end = np.load(end_path)
        if start.shape == end.shape == target_shape:
            pairs.append((start, end))
    return pairs


def _temporal_training_rows(rows: list[dict], target: dict) -> list[dict]:
    training = []
    for row in rows:
        if row.get("qc_status") != "include":
            continue
        start_path = _path(row.get("label_start_path"))
        end_path = _path(row.get("label_end_path"))
        if start_path is None or end_path is None or not start_path.exists() or not end_path.exists():
            continue
        training.append(
            {
                "area": row.get("area"),
                "start": np.load(start_path),
                "end": np.load(end_path),
            }
        )
    return training


def evaluate_registry_row(
    row: dict,
    transition_training_pairs: list[tuple[np.ndarray, np.ndarray]],
    temporal_training_rows: list[dict],
) -> dict:
    start = _load_array(row.get("label_start_path"))
    end = _load_array(row.get("label_end_path"))
    model_pred = _load_array(row.get("prediction_path"))
    if start.shape != end.shape or start.shape != model_pred.shape:
        raise ValueError(
            f"Shape mismatch for {row.get('area')} {row.get('start_year')}-{row.get('end_year')}: "
            f"start={start.shape}, end={end.shape}, pred={model_pred.shape}"
        )

    true_change = end != start
    model_change = model_pred != start
    persistence_pred = persistence_prediction(start)
    persistence_change = persistence_pred != start
    shuffle_pred = spatial_shuffle_prediction(model_pred)
    shuffle_change = shuffle_pred != start
    transition_prior_pred = label_only_transition_prior(start, transition_training_pairs)
    transition_prior_change = transition_prior_pred != start
    temporal_prior_pred = leave_one_region_temporal_prior(str(row.get("area")), start, temporal_training_rows)
    temporal_prior_change = temporal_prior_pred != start

    model_metrics = binary_change_metrics(true_change, model_change)
    persistence_metrics = binary_change_metrics(true_change, persistence_change)
    shuffle_metrics = binary_change_metrics(true_change, shuffle_change)
    transition_metrics = binary_change_metrics(true_change, transition_prior_change)
    temporal_metrics = binary_change_metrics(true_change, temporal_prior_change)
    best_non_neural = max(
        persistence_metrics["f1"],
        transition_metrics["f1"],
        temporal_metrics["f1"],
    )

    embedding_persistence_cosine = None
    embedding_model_cosine = None
    embedding_advantage = None
    emb_start_path = _path(row.get("embedding_start_path"))
    emb_end_path = _path(row.get("embedding_end_path"))
    if emb_start_path is not None and emb_end_path is not None and emb_start_path.exists() and emb_end_path.exists():
        emb_start = np.load(emb_start_path)
        emb_end = np.load(emb_end_path)
        embedding_persistence_cosine = mean_cosine(emb_start, emb_end)

    return {
        "area": row.get("area"),
        "start_year": int(row.get("start_year")),
        "end_year": int(row.get("end_year")),
        "tier": row.get("tier"),
        "stratum": row.get("stratum"),
        "n_pixels": int(start.size),
        "true_change_pixels": int(np.count_nonzero(true_change)),
        "true_change_pct": float(np.count_nonzero(true_change) / start.size) if start.size else 0.0,
        "model_change_precision": model_metrics["precision"],
        "model_change_recall": model_metrics["recall"],
        "model_change_f1": model_metrics["f1"],
        "persistence_change_f1": persistence_metrics["f1"],
        "spatial_shuffle_change_f1": shuffle_metrics["f1"],
        "transition_prior_change_f1": transition_metrics["f1"],
        "temporal_prior_change_f1": temporal_metrics["f1"],
        "best_non_neural_change_f1": best_non_neural,
        "primary_change_advantage": model_metrics["f1"] - best_non_neural,
        "spatial_change_advantage": model_metrics["f1"] - shuffle_metrics["f1"],
        "embedding_model_cosine": embedding_model_cosine,
        "embedding_persistence_cosine": embedding_persistence_cosine,
        "embedding_advantage": embedding_advantage,
    }


def _read_registry(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rows", []))


def _write_summary_csv(path: Path, summary: dict) -> None:
    fields = ["group", "n", "mean", "median", "n_positive", "n_negative"]
    rows = [{"group": group, **values} for group, values in summary.items()]
    write_csv(path, rows, fields)


def _write_failures_csv(path: Path, registry_rows: list[dict]) -> None:
    fields = ["area", "start_year", "end_year", "tier", "qc_status", "excluded_reason"]
    failures = [
        {field: row.get(field) for field in fields}
        for row in registry_rows
        if row.get("qc_status") != "include"
    ]
    write_csv(path, failures, fields)


def evaluate_benchmark(
    registry_path: Path = DEFAULT_BENCHMARK_DIR / "benchmark_registry.json",
    output_dir: Path = DEFAULT_BENCHMARK_DIR,
    n_boot: int = 5000,
) -> dict:
    registry_rows = _read_registry(Path(registry_path))
    included_rows = [row for row in registry_rows if row.get("qc_status") == "include"]
    metric_rows = []
    for row in included_rows:
        start = _load_array(row.get("label_start_path"))
        metrics = evaluate_registry_row(
            row,
            transition_training_pairs=_transition_training_pairs(included_rows, row, start.shape),
            temporal_training_rows=_temporal_training_rows(included_rows, row),
        )
        metric_rows.append(metrics)

    primary_summary = summarize_by_tier_and_stratum(metric_rows, "primary_change_advantage")
    spatial_summary = summarize_by_tier_and_stratum(metric_rows, "spatial_change_advantage")
    gates = gate_report(metric_rows, n_boot=n_boot)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_registry_rows": len(registry_rows),
        "n_evaluated": len(metric_rows),
        "n_failures": len(registry_rows) - len(included_rows),
        "model_change_f1_mean": (
            float(mean(row["model_change_f1"] for row in metric_rows)) if metric_rows else None
        ),
        "primary_change_advantage_mean": (
            float(mean(row["primary_change_advantage"] for row in metric_rows)) if metric_rows else None
        ),
    }
    result = {
        "summary": summary,
        "summary_by_tier": primary_summary["by_tier"],
        "summary_by_stratum": primary_summary["by_stratum"],
        "spatial_summary_by_tier": spatial_summary["by_tier"],
        "gate_report": gates,
    }

    write_csv(output_dir / "benchmark_metrics_by_pair.csv", metric_rows, METRIC_FIELDS)
    write_json(output_dir / "benchmark_summary.json", result)
    _write_summary_csv(output_dir / "benchmark_summary_by_tier.csv", result["summary_by_tier"])
    _write_summary_csv(output_dir / "benchmark_summary_by_stratum.csv", result["summary_by_stratum"])
    write_json(output_dir / "benchmark_gate_report.json", gates)
    _write_failures_csv(output_dir / "benchmark_failures.csv", registry_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Paper58 external benchmark.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_BENCHMARK_DIR / "benchmark_registry.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()
    result = evaluate_benchmark(args.registry, args.output_dir, args.n_boot)
    print(
        "Benchmark evaluation: "
        f"{result['summary']['n_evaluated']} evaluated pair(s), "
        f"gate status={result['gate_report']['status']}"
    )


if __name__ == "__main__":
    main()
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
git commit -m "feat: add Paper58 benchmark evaluation"
```

---

### Task 6: Benchmark Figures

**Files:**
- Create: `scripts/paper58_benchmark/make_benchmark_figures.py`
- Test: `tests/test_paper58_benchmark_figures.py`

- [ ] **Step 1: Write failing figure tests**

Create `tests/test_paper58_benchmark_figures.py`:

```python
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from scripts.paper58_benchmark.make_benchmark_figures import load_benchmark_outputs, make_benchmark_figures


def test_load_benchmark_outputs_fails_when_required_files_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Missing benchmark result files"):
        load_benchmark_outputs(tmp_path)


def test_make_benchmark_figures_writes_pdf_and_png(tmp_path: Path):
    results_dir = tmp_path / "results"
    figure_dir = tmp_path / "figures"
    results_dir.mkdir()
    (results_dir / "benchmark_metrics_by_pair.csv").write_text(
        "area,start_year,end_year,tier,stratum,primary_change_advantage,spatial_change_advantage,model_change_f1,best_non_neural_change_f1\n"
        "external,2020,2021,tier1,Wetland,0.20,0.10,0.50,0.30\n",
        encoding="utf-8",
    )
    (results_dir / "benchmark_gate_report.json").write_text(
        '{"status": "pass", "positive_tier1_strata": 1}',
        encoding="utf-8",
    )

    make_benchmark_figures(results_dir=results_dir, figure_dir=figure_dir)

    assert (figure_dir / "fig_paper58_benchmark_gate.pdf").exists()
    assert (figure_dir / "fig_paper58_benchmark_gate.png").exists()
```

- [ ] **Step 2: Run figure tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_figures.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.paper58_benchmark.make_benchmark_figures'`.

- [ ] **Step 3: Implement figures**

Create `scripts/paper58_benchmark/make_benchmark_figures.py`. It must:

- Require `benchmark_metrics_by_pair.csv` and `benchmark_gate_report.json`.
- Raise `FileNotFoundError` if either is absent.
- Make `fig_paper58_benchmark_gate.pdf/png`.
- Plot Tier 1 primary and spatial advantages by area-year pair.
- Use `matplotlib.use("Agg")` only in tests, not in the module.

Implementation outline:

```python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.paper58_benchmark.schema import DEFAULT_BENCHMARK_DIR


def _read_metrics(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            for key in ("primary_change_advantage", "spatial_change_advantage", "model_change_f1", "best_non_neural_change_f1"):
                parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def load_benchmark_outputs(results_dir: Path = DEFAULT_BENCHMARK_DIR) -> dict:
    results_dir = Path(results_dir)
    metrics_path = results_dir / "benchmark_metrics_by_pair.csv"
    gate_path = results_dir / "benchmark_gate_report.json"
    missing = [str(path) for path in (metrics_path, gate_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing benchmark result files: " + ", ".join(missing))
    return {
        "metrics": _read_metrics(metrics_path),
        "gate": json.loads(gate_path.read_text(encoding="utf-8")),
    }


def make_benchmark_figures(results_dir: Path = DEFAULT_BENCHMARK_DIR, figure_dir: Path | None = None) -> None:
    data = load_benchmark_outputs(results_dir)
    figure_dir = Path(figure_dir) if figure_dir is not None else Path(results_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in data["metrics"] if row["tier"] == "tier1"]
    if not rows:
        rows = data["metrics"]
    labels = [f"{row['area']}\\n{row['start_year']}-{row['end_year']}" for row in rows]
    x = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(max(5.0, len(rows) * 0.75), 3.2))
    primary = [row["primary_change_advantage"] for row in rows]
    spatial = [row["spatial_change_advantage"] for row in rows]
    width = 0.35
    ax.bar([i - width / 2 for i in x], primary, width=width, label="Model - best non-neural")
    ax.bar([i + width / 2 for i in x], spatial, width=width, label="Model - spatial shuffle")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Change-F1 advantage")
    ax.set_title(f"Paper58 benchmark gate status: {data['gate'].get('status', 'unknown')}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_paper58_benchmark_gate.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_paper58_benchmark_gate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make Paper58 benchmark figures.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--figure-dir", type=Path, default=None)
    args = parser.parse_args()
    make_benchmark_figures(args.results_dir, args.figure_dir)
    print(f"Wrote Paper58 benchmark figures from {args.results_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run figure tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_figures.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 6**

Run:

```powershell
git add scripts/paper58_benchmark/make_benchmark_figures.py tests/test_paper58_benchmark_figures.py
git commit -m "feat: add Paper58 benchmark figures"
```

---

### Task 7: Local End-To-End Benchmark Run

**Files:**
- No code edits are expected in this task.
- If a command fails, capture the exact failure, add a focused failing test to the relevant `tests/test_paper58_benchmark_*.py` file, then fix the corresponding benchmark module before rerunning this task.

- [ ] **Step 1: Run all Paper58 benchmark tests**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_baselines.py tests/test_paper58_benchmark_registry.py tests/test_paper58_benchmark_statistics.py tests/test_paper58_benchmark_evaluation.py tests/test_paper58_benchmark_figures.py -q
```

Expected: PASS.

- [ ] **Step 2: Build the local registry from existing cached files**

Run:

```powershell
python -m scripts.paper58_benchmark.build_registry
```

Expected:

```text
Benchmark registry: 12 candidate pair(s), 11 included pair(s)
```

The expected excluded candidate is `wuyi_mountain` 2020-2021, which should be retained as `qc_status=negative_control` with `excluded_reason=zero_reference_change`. If any other row is excluded, inspect `benchmark_registry.csv` and record the concrete reason before changing code.

- [ ] **Step 3: Evaluate the local benchmark**

Run:

```powershell
python -m scripts.paper58_benchmark.evaluate_benchmark
```

Expected:

```text
Benchmark evaluation: 11 evaluated pair(s), gate status=<pass|fail|insufficient_tier1>
```

The command must write:

```text
paper/rse_submission_paper58/benchmark_results/benchmark_metrics_by_pair.csv
paper/rse_submission_paper58/benchmark_results/benchmark_summary.json
paper/rse_submission_paper58/benchmark_results/benchmark_summary_by_tier.csv
paper/rse_submission_paper58/benchmark_results/benchmark_summary_by_stratum.csv
paper/rse_submission_paper58/benchmark_results/benchmark_gate_report.json
paper/rse_submission_paper58/benchmark_results/benchmark_failures.csv
```

- [ ] **Step 4: Generate benchmark figures**

Run:

```powershell
python -m scripts.paper58_benchmark.make_benchmark_figures
```

Expected:

```text
Wrote Paper58 benchmark figures from D:\test\paper58-geofm-world-model-rl\paper\rse_submission_paper58\benchmark_results
```

Expected figure files:

```text
paper/rse_submission_paper58/benchmark_results/figures/fig_paper58_benchmark_gate.pdf
paper/rse_submission_paper58/benchmark_results/figures/fig_paper58_benchmark_gate.png
```

- [ ] **Step 5: Inspect the gate report before manuscript work**

Run:

```powershell
Get-Content -Raw -LiteralPath "paper\rse_submission_paper58\benchmark_results\benchmark_gate_report.json"
```

Expected interpretation:

- If `status` is `pass`, implementation can proceed to a separate manuscript-revision plan.
- If `status` is `fail` or `insufficient_tier1`, do not modify the manuscript toward stronger claims. Record the result as evidence that more external data or weaker framing is required.

- [ ] **Step 6: Commit benchmark outputs using a fixed size rule**

Run:

```powershell
git status --short
```

Measure generated figure size:

```powershell
(Get-ChildItem -LiteralPath "paper\rse_submission_paper58\benchmark_results\figures" -File | Measure-Object -Property Length -Sum).Sum
```

If the total figure size is less than or equal to 10485760 bytes, commit scripts, tests, CSV, JSON, and figures:

```powershell
git add scripts/paper58_benchmark tests/test_paper58_benchmark_*.py paper/rse_submission_paper58/benchmark_results
git commit -m "feat: run Paper58 local benchmark pipeline"
```

If the total figure size is greater than 10485760 bytes, commit scripts, tests, CSV, and JSON only:

```powershell
git add scripts/paper58_benchmark tests/test_paper58_benchmark_*.py paper/rse_submission_paper58/benchmark_results/*.csv paper/rse_submission_paper58/benchmark_results/*.json
git commit -m "feat: run Paper58 local benchmark pipeline"
```

---

### Task 8: Verification And Handoff Note

**Files:**
- Create: `docs/current_work_progress_2026-06-19.md`

- [ ] **Step 1: Run full relevant verification**

Run:

```powershell
python -m pytest tests/test_paper58_benchmark_baselines.py tests/test_paper58_benchmark_registry.py tests/test_paper58_benchmark_statistics.py tests/test_paper58_benchmark_evaluation.py tests/test_paper58_benchmark_figures.py tests/test_rse_revision_results.py tests/test_rse_revision_change_validation.py -q
```

Expected: PASS.

- [ ] **Step 2: Run diff checks**

Run:

```powershell
git diff --check
```

Expected: no output.

- [ ] **Step 3: Write the handoff note**

Create `docs/current_work_progress_2026-06-19.md` with:

```markdown
# Current Work Progress: 2026-06-19

## Paper58 Benchmark Status

The Paper58 work has moved to an evidence-first external benchmark path. The active design is:

```text
docs/superpowers/specs/2026-06-19-paper58-benchmark-design.md
```

The active implementation plan is:

```text
docs/superpowers/plans/2026-06-19-paper58-external-benchmark.md
```

## Current Rule

Do not strengthen the RSE manuscript claims until the benchmark gate passes. If the gate fails or reports insufficient Tier 1 evidence, manuscript work must be limited to claim downgrading, reference cleanup, data availability cleanup, and limitations.

## Benchmark Outputs

Expected output directory:

```text
paper/rse_submission_paper58/benchmark_results
```

Key files:

```text
benchmark_registry.json
benchmark_metrics_by_pair.csv
benchmark_summary.json
benchmark_gate_report.json
```

## Resume Point

Resume from the next unchecked task in:

```text
docs/superpowers/plans/2026-06-19-paper58-external-benchmark.md
```
```

- [ ] **Step 4: Commit the handoff note**

Run:

```powershell
git add docs/current_work_progress_2026-06-19.md
git commit -m "docs: record Paper58 benchmark handoff"
```

- [ ] **Step 5: Push after user approval**

Run only after the user asks to push:

```powershell
git push origin main
```

Expected: remote `main` advances to include the benchmark plan and implementation commits.

---

## Self-Review Checklist

- Spec coverage: Tasks cover registry, QC, baselines, clustered statistics, evaluation, figures, output separation, gate report, verification, and manuscript gate.
- Scope: This plan intentionally excludes network/GEE expansion and manuscript rewriting. Those need separate plans after the local benchmark gate report exists.
- Type consistency: All output rows use `area`, `start_year`, `end_year`, `tier`, `stratum`, and the metric names listed in Task 5.
- Output separation: All new benchmark outputs go under `paper/rse_submission_paper58/benchmark_results/`.
- Evidence discipline: The plan does not allow stronger RSE manuscript claims before the gate report passes.
