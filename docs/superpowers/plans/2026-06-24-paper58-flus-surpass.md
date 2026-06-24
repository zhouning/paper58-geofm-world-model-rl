# Paper58-LAS GeoSOS-FLUS Surpass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first testable Paper58-LAS simulator and FLUS-facing evaluator while preserving the existing Paper58 architecture and benchmark evidence.

**Architecture:** Keep Paper58 direct prediction untouched. Add a separate simulator path that converts Paper58 prediction artifacts into transition suitability, applies demand-constrained allocation with constraints and latent-neighborhood scoring, and evaluates the resulting simulated LULC map against FLUS-compatible outputs under matched conditions.

**Tech Stack:** Python 3.11+, NumPy, pytest, existing `scripts.paper58_benchmark` package, existing CSV/JSON helpers in `scripts/paper58_benchmark/schema.py`.

---

## Scope

This plan implements the first build only:

- demand derivation and constraint validation;
- transition suitability from Paper58 prediction/probability artifacts;
- deterministic demand-constrained allocation;
- FLUS-facing metrics;
- FLUS prediction ingestion for `.npy` and `.csv`;
- a new `evaluate_las.py` report writer.

This plan does not modify `LatentDynamicsNet`, existing Batch 2/3/4/5 reports, or manuscript claims.

## File Structure

- Create `scripts/paper58_benchmark/las_demand.py`
  - Responsibility: derive target demand, editable masks, and remaining editable demand.
- Create `scripts/paper58_benchmark/las_suitability.py`
  - Responsibility: build class values, one-hot probability cubes, transition priors, and transition suitability tensors.
- Create `scripts/paper58_benchmark/las_allocation.py`
  - Responsibility: allocate editable pixels to target classes under demand and conversion constraints.
- Create `scripts/paper58_benchmark/las_metrics.py`
  - Responsibility: compute FOM, transition accuracy, quantity disagreement, allocation disagreement, and method comparison rows.
- Create `scripts/paper58_benchmark/flus.py`
  - Responsibility: load and validate GeoSOS-FLUS or FLUS-compatible predictions.
- Create `scripts/paper58_benchmark/evaluate_las.py`
  - Responsibility: run Paper58-LAS on benchmark registry rows and write matched-condition reports.
- Create `tests/test_paper58_las_demand.py`
- Create `tests/test_paper58_las_suitability.py`
- Create `tests/test_paper58_las_allocation.py`
- Create `tests/test_paper58_las_metrics.py`
- Create `tests/test_paper58_flus_ingestion.py`
- Create `tests/test_paper58_las_evaluation.py`

## Task 1: Demand And Constraint Helpers

**Files:**
- Create: `scripts/paper58_benchmark/las_demand.py`
- Test: `tests/test_paper58_las_demand.py`

- [ ] **Step 1: Write failing demand tests**

Create `tests/test_paper58_las_demand.py`:

```python
import numpy as np
import pytest

from scripts.paper58_benchmark.las_demand import (
    DemandValidationError,
    build_editable_mask,
    derive_observed_demand,
    remaining_editable_demand,
    validate_total_demand,
)


def test_derive_observed_demand_counts_classes():
    end = np.array([[1, 2, 2], [3, 3, 3]], dtype=np.int32)

    demand = derive_observed_demand(end)

    assert demand == {1: 1, 2: 2, 3: 3}


def test_validate_total_demand_rejects_wrong_total():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)

    with pytest.raises(DemandValidationError, match="demand total 3 does not match raster cells 4"):
        validate_total_demand(start, {1: 1, 2: 2})


def test_build_editable_mask_respects_exclusion_and_immutable_classes():
    start = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32)
    exclusion = np.array([[False, False, True], [False, True, False]])

    editable = build_editable_mask(start, exclusion_mask=exclusion, immutable_classes={3})

    expected = np.array([[True, True, False], [True, False, False]])
    assert np.array_equal(editable, expected)


def test_remaining_editable_demand_subtracts_fixed_pixels():
    start = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int32)
    editable = np.array([[True, True, True], [True, False, False]])
    full_demand = {1: 1, 2: 3, 3: 2}

    remaining = remaining_editable_demand(start, full_demand, editable)

    assert remaining == {1: 1, 2: 3, 3: 0}


def test_remaining_editable_demand_fails_when_fixed_pixels_exceed_demand():
    start = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int32)
    editable = np.array([[True, True, True], [True, False, False]])
    full_demand = {1: 2, 2: 2, 3: 1}

    with pytest.raises(DemandValidationError, match="fixed class 3 count 2 exceeds demand 1"):
        remaining_editable_demand(start, full_demand, editable)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_paper58_las_demand.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.paper58_benchmark.las_demand'`.

- [ ] **Step 3: Implement demand helpers**

Create `scripts/paper58_benchmark/las_demand.py`:

```python
from __future__ import annotations

import numpy as np


class DemandValidationError(ValueError):
    """Raised when a demand or constraint configuration cannot be satisfied."""


def derive_observed_demand(end_map: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(np.asarray(end_map), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def validate_total_demand(start_map: np.ndarray, demand: dict[int, int]) -> None:
    total = int(sum(int(value) for value in demand.values()))
    n_cells = int(np.asarray(start_map).size)
    if total != n_cells:
        raise DemandValidationError(f"demand total {total} does not match raster cells {n_cells}")
    for cls, count in demand.items():
        if int(count) < 0:
            raise DemandValidationError(f"demand for class {int(cls)} is negative: {int(count)}")


def build_editable_mask(
    start_map: np.ndarray,
    exclusion_mask: np.ndarray | None = None,
    immutable_classes: set[int] | None = None,
) -> np.ndarray:
    start = np.asarray(start_map)
    editable = np.ones(start.shape, dtype=bool)
    if exclusion_mask is not None:
        exclusion = np.asarray(exclusion_mask, dtype=bool)
        if exclusion.shape != start.shape:
            raise DemandValidationError(
                f"exclusion mask shape {exclusion.shape} does not match start map shape {start.shape}"
            )
        editable &= ~exclusion
    for cls in sorted(immutable_classes or set()):
        editable &= start != int(cls)
    return editable


def remaining_editable_demand(
    start_map: np.ndarray,
    demand: dict[int, int],
    editable_mask: np.ndarray,
) -> dict[int, int]:
    start = np.asarray(start_map)
    editable = np.asarray(editable_mask, dtype=bool)
    if editable.shape != start.shape:
        raise DemandValidationError(f"editable mask shape {editable.shape} does not match start map shape {start.shape}")
    validate_total_demand(start, demand)
    fixed = start[~editable]
    fixed_values, fixed_counts = np.unique(fixed, return_counts=True)
    fixed_by_class = {int(value): int(count) for value, count in zip(fixed_values, fixed_counts)}
    remaining: dict[int, int] = {}
    for cls, target_count in sorted((int(cls), int(count)) for cls, count in demand.items()):
        fixed_count = fixed_by_class.get(cls, 0)
        if fixed_count > target_count:
            raise DemandValidationError(f"fixed class {cls} count {fixed_count} exceeds demand {target_count}")
        remaining[cls] = target_count - fixed_count
    editable_total = int(np.count_nonzero(editable))
    remaining_total = int(sum(remaining.values()))
    if remaining_total != editable_total:
        raise DemandValidationError(
            f"remaining editable demand {remaining_total} does not match editable cells {editable_total}"
        )
    return remaining
```

- [ ] **Step 4: Run demand tests**

Run:

```bash
python -m pytest tests/test_paper58_las_demand.py -q
```

Expected: PASS, `5 passed`.

- [ ] **Step 5: Commit demand helpers**

Run:

```bash
git add scripts/paper58_benchmark/las_demand.py tests/test_paper58_las_demand.py
git commit -m "feat: add Paper58 LAS demand helpers"
```

Expected: commit succeeds.

## Task 2: Transition Suitability Construction

**Files:**
- Create: `scripts/paper58_benchmark/las_suitability.py`
- Test: `tests/test_paper58_las_suitability.py`

- [ ] **Step 1: Write failing suitability tests**

Create `tests/test_paper58_las_suitability.py`:

```python
import numpy as np

from scripts.paper58_benchmark.las_suitability import (
    build_transition_suitability,
    class_values_from_maps,
    one_hot_probability_cube,
    transition_prior_from_pairs,
)


def test_class_values_from_maps_returns_sorted_unique_ints():
    a = np.array([[3, 1], [2, 1]], dtype=np.int32)
    b = np.array([[5, 3], [2, 5]], dtype=np.int32)

    assert class_values_from_maps(a, b) == [1, 2, 3, 5]


def test_one_hot_probability_cube_uses_confidence_and_floor():
    labels = np.array([[1, 2]], dtype=np.int32)
    cube = one_hot_probability_cube(labels, [1, 2, 3], confidence=0.9, floor=0.05)

    assert cube.shape == (1, 2, 3)
    assert cube[0, 0, 0] == 0.9
    assert cube[0, 0, 1] == 0.05
    assert cube[0, 0, 2] == 0.05
    assert cube[0, 1, 1] == 0.9


def test_transition_prior_from_pairs_normalizes_by_start_class():
    start = np.array([[1, 1, 2, 2]], dtype=np.int32)
    end = np.array([[1, 2, 2, 3]], dtype=np.int32)

    prior = transition_prior_from_pairs([(start, end)], class_values=[1, 2, 3])

    assert prior[(1, 1)] == 0.5
    assert prior[(1, 2)] == 0.5
    assert prior[(2, 2)] == 0.5
    assert prior[(2, 3)] == 0.5


def test_build_transition_suitability_respects_allowed_transitions():
    start = np.array([[1, 2]], dtype=np.int32)
    forecast_probs = np.array([[[0.1, 0.8], [0.7, 0.2]]], dtype=np.float32)
    start_probs = np.array([[[0.8, 0.1], [0.1, 0.8]]], dtype=np.float32)

    suitability = build_transition_suitability(
        start,
        class_values=[1, 2],
        forecast_probs=forecast_probs,
        start_probs=start_probs,
        allowed_transitions={(1, 2)},
    )

    assert suitability.shape == (1, 2, 2)
    assert suitability[0, 0, 1] > 0.0
    assert suitability[0, 0, 0] == 0.0
    assert suitability[0, 1, 0] == 0.0
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_paper58_las_suitability.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.paper58_benchmark.las_suitability'`.

- [ ] **Step 3: Implement suitability helpers**

Create `scripts/paper58_benchmark/las_suitability.py`:

```python
from __future__ import annotations

import numpy as np


def class_values_from_maps(*maps: np.ndarray) -> list[int]:
    values: set[int] = set()
    for arr in maps:
        for value in np.unique(np.asarray(arr)):
            values.add(int(value))
    return sorted(values)


def one_hot_probability_cube(
    labels: np.ndarray,
    class_values: list[int],
    confidence: float = 0.95,
    floor: float = 0.01,
) -> np.ndarray:
    label_arr = np.asarray(labels)
    if not class_values:
        raise ValueError("class_values must be non-empty")
    if len(class_values) == 1:
        return np.ones(label_arr.shape + (1,), dtype=np.float32)
    cube = np.full(label_arr.shape + (len(class_values),), float(floor), dtype=np.float32)
    class_to_col = {int(cls): index for index, cls in enumerate(class_values)}
    for cls, col in class_to_col.items():
        cube[label_arr == cls, col] = float(confidence)
    return cube


def transition_prior_from_pairs(
    training_pairs: list[tuple[np.ndarray, np.ndarray]],
    class_values: list[int],
) -> dict[tuple[int, int], float]:
    counts: dict[int, dict[int, int]] = {}
    for start, end in training_pairs:
        start_arr = np.asarray(start)
        end_arr = np.asarray(end)
        if start_arr.shape != end_arr.shape:
            continue
        for from_cls, to_cls in zip(start_arr.ravel(), end_arr.ravel()):
            from_key = int(from_cls)
            to_key = int(to_cls)
            counts.setdefault(from_key, {})
            counts[from_key][to_key] = counts[from_key].get(to_key, 0) + 1

    prior: dict[tuple[int, int], float] = {}
    for from_cls in class_values:
        row = counts.get(int(from_cls), {})
        total = int(sum(row.values()))
        if total == 0:
            continue
        for to_cls in class_values:
            prior[(int(from_cls), int(to_cls))] = row.get(int(to_cls), 0) / total
    return prior


def _embedding_change_pressure(
    embedding_start: np.ndarray | None,
    embedding_forecast: np.ndarray | None,
    target_shape: tuple[int, int],
) -> np.ndarray:
    if embedding_start is None or embedding_forecast is None:
        return np.zeros(target_shape, dtype=np.float32)
    start = np.asarray(embedding_start, dtype=np.float32)
    forecast = np.asarray(embedding_forecast, dtype=np.float32)
    if start.shape != forecast.shape:
        raise ValueError(f"embedding shape mismatch: start={start.shape}, forecast={forecast.shape}")
    if start.shape[:2] != target_shape:
        raise ValueError(f"embedding grid shape {start.shape[:2]} does not match target shape {target_shape}")
    pressure = np.linalg.norm(forecast - start, axis=-1)
    max_value = float(np.max(pressure)) if pressure.size else 0.0
    if max_value <= 0.0:
        return np.zeros(target_shape, dtype=np.float32)
    return (pressure / max_value).astype(np.float32)


def build_transition_suitability(
    start_map: np.ndarray,
    class_values: list[int],
    forecast_probs: np.ndarray,
    start_probs: np.ndarray | None = None,
    embedding_start: np.ndarray | None = None,
    embedding_forecast: np.ndarray | None = None,
    allowed_transitions: set[tuple[int, int]] | None = None,
    transition_prior: dict[tuple[int, int], float] | None = None,
    forecast_prob_weight: float = 1.0,
    probability_gain_weight: float = 0.5,
    change_pressure_weight: float = 0.25,
    transition_prior_weight: float = 0.25,
) -> np.ndarray:
    start = np.asarray(start_map)
    forecast = np.asarray(forecast_probs, dtype=np.float32)
    if forecast.shape != start.shape + (len(class_values),):
        raise ValueError(
            f"forecast_probs shape {forecast.shape} does not match start/class shape {start.shape + (len(class_values),)}"
        )
    if start_probs is None:
        start_prob_arr = np.zeros_like(forecast)
    else:
        start_prob_arr = np.asarray(start_probs, dtype=np.float32)
        if start_prob_arr.shape != forecast.shape:
            raise ValueError(f"start_probs shape {start_prob_arr.shape} does not match forecast_probs shape {forecast.shape}")

    pressure = _embedding_change_pressure(embedding_start, embedding_forecast, start.shape)
    class_to_col = {int(cls): index for index, cls in enumerate(class_values)}
    suitability = np.zeros_like(forecast, dtype=np.float32)
    prior = transition_prior or {}

    for from_cls in np.unique(start):
        from_key = int(from_cls)
        from_mask = start == from_key
        for to_cls in class_values:
            to_key = int(to_cls)
            if from_key == to_key:
                continue
            if allowed_transitions is not None and (from_key, to_key) not in allowed_transitions:
                continue
            col = class_to_col[to_key]
            probability_gain = np.maximum(forecast[..., col] - start_prob_arr[..., col], 0.0)
            base_score = (
                forecast_prob_weight * forecast[..., col]
                + probability_gain_weight * probability_gain
                + change_pressure_weight * pressure
                + transition_prior_weight * float(prior.get((from_key, to_key), 0.0))
            )
            suitability[from_mask, col] = base_score[from_mask]
    return np.maximum(suitability, 0.0).astype(np.float32)
```

- [ ] **Step 4: Run suitability tests**

Run:

```bash
python -m pytest tests/test_paper58_las_suitability.py -q
```

Expected: PASS, `4 passed`.

- [ ] **Step 5: Commit suitability helpers**

Run:

```bash
git add scripts/paper58_benchmark/las_suitability.py tests/test_paper58_las_suitability.py
git commit -m "feat: add Paper58 LAS suitability builder"
```

Expected: commit succeeds.

## Task 3: Demand-Constrained Allocation

**Files:**
- Create: `scripts/paper58_benchmark/las_allocation.py`
- Test: `tests/test_paper58_las_allocation.py`

- [ ] **Step 1: Write failing allocation tests**

Create `tests/test_paper58_las_allocation.py`:

```python
import numpy as np

from scripts.paper58_benchmark.las_allocation import allocate_demand_constrained


def test_allocate_demand_constrained_meets_target_demand():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    class_values = [1, 2]
    suitability = np.zeros((2, 2, 2), dtype=np.float32)
    suitability[0, 0, 1] = 0.9
    suitability[0, 1, 1] = 0.8
    target_demand = {1: 2, 2: 2}

    result = allocate_demand_constrained(start, suitability, class_values, target_demand)

    assert result.achieved_demand == target_demand
    assert result.unmet_demand == {1: 0, 2: 0}
    assert result.simulated_map.shape == start.shape


def test_allocate_respects_exclusion_mask():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    class_values = [1, 2]
    suitability = np.zeros((2, 2, 2), dtype=np.float32)
    suitability[:, :, 1] = 1.0
    target_demand = {1: 1, 2: 3}
    exclusion = np.array([[True, False], [False, False]])

    result = allocate_demand_constrained(
        start,
        suitability,
        class_values,
        target_demand,
        exclusion_mask=exclusion,
    )

    assert result.simulated_map[0, 0] == 1
    assert result.achieved_demand == target_demand


def test_allocate_respects_forbidden_transition():
    start = np.array([[1, 2], [2, 2]], dtype=np.int32)
    class_values = [1, 2, 3]
    suitability = np.ones((2, 2, 3), dtype=np.float32)
    target_demand = {1: 1, 2: 1, 3: 2}

    result = allocate_demand_constrained(
        start,
        suitability,
        class_values,
        target_demand,
        allowed_transitions={(2, 3)},
    )

    assert result.simulated_map[0, 0] == 1
    assert np.count_nonzero(result.simulated_map == 3) == 2
    assert result.constraint_violations == []
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_paper58_las_allocation.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.paper58_benchmark.las_allocation'`.

- [ ] **Step 3: Implement allocation**

Create `scripts/paper58_benchmark/las_allocation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.paper58_benchmark.las_demand import (
    DemandValidationError,
    build_editable_mask,
    remaining_editable_demand,
)


@dataclass(frozen=True)
class LASAllocationResult:
    simulated_map: np.ndarray
    target_demand: dict[int, int]
    achieved_demand: dict[int, int]
    unmet_demand: dict[int, int]
    selected_transitions: list[dict[str, int | float]]
    constraint_violations: list[str]


def _counts(label_map: np.ndarray, class_values: list[int]) -> dict[int, int]:
    arr = np.asarray(label_map)
    return {int(cls): int(np.count_nonzero(arr == int(cls))) for cls in class_values}


def _is_allowed(from_cls: int, to_cls: int, allowed_transitions: set[tuple[int, int]] | None) -> bool:
    if from_cls == to_cls:
        return True
    if allowed_transitions is None:
        return True
    return (int(from_cls), int(to_cls)) in allowed_transitions


def allocate_demand_constrained(
    start_map: np.ndarray,
    suitability: np.ndarray,
    class_values: list[int],
    target_demand: dict[int, int],
    exclusion_mask: np.ndarray | None = None,
    immutable_classes: set[int] | None = None,
    allowed_transitions: set[tuple[int, int]] | None = None,
) -> LASAllocationResult:
    start = np.asarray(start_map)
    scores = np.asarray(suitability, dtype=np.float32)
    if scores.shape != start.shape + (len(class_values),):
        raise DemandValidationError(
            f"suitability shape {scores.shape} does not match start/class shape {start.shape + (len(class_values),)}"
        )
    demand = {int(cls): int(count) for cls, count in target_demand.items()}
    editable = build_editable_mask(start, exclusion_mask=exclusion_mask, immutable_classes=immutable_classes)
    remaining = remaining_editable_demand(start, demand, editable)
    class_to_col = {int(cls): index for index, cls in enumerate(class_values)}
    simulated = start.copy()
    assigned = ~editable
    selected: list[dict[str, int | float]] = []

    candidates: list[tuple[float, int, int, int, int]] = []
    for row, col in np.argwhere(editable):
        from_cls = int(start[row, col])
        for to_cls in class_values:
            to_key = int(to_cls)
            if remaining.get(to_key, 0) <= 0:
                continue
            if not _is_allowed(from_cls, to_key, allowed_transitions):
                continue
            score = float(scores[row, col, class_to_col[to_key]])
            if from_cls == to_key:
                score += 1e-6
            candidates.append((score, int(row), int(col), from_cls, to_key))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[4]))
    for score, row, col, from_cls, to_cls in candidates:
        if assigned[row, col]:
            continue
        if remaining.get(to_cls, 0) <= 0:
            continue
        simulated[row, col] = to_cls
        assigned[row, col] = True
        remaining[to_cls] -= 1
        if from_cls != to_cls:
            selected.append(
                {
                    "row": row,
                    "col": col,
                    "from_class": from_cls,
                    "to_class": to_cls,
                    "score": score,
                }
            )

    violations: list[str] = []
    if np.any(~assigned):
        for row, col in np.argwhere(~assigned):
            from_cls = int(start[row, col])
            assigned_class = None
            for to_cls in class_values:
                to_key = int(to_cls)
                if remaining.get(to_key, 0) > 0 and _is_allowed(from_cls, to_key, allowed_transitions):
                    assigned_class = to_key
                    break
            if assigned_class is None:
                violations.append(f"no feasible class for pixel ({int(row)}, {int(col)}) from class {from_cls}")
                simulated[row, col] = from_cls
            else:
                simulated[row, col] = assigned_class
                remaining[assigned_class] -= 1
            assigned[row, col] = True

    achieved = _counts(simulated, class_values)
    unmet = {int(cls): max(0, int(demand.get(int(cls), 0)) - int(achieved.get(int(cls), 0))) for cls in class_values}
    for cls in class_values:
        target = int(demand.get(int(cls), 0))
        actual = int(achieved.get(int(cls), 0))
        if target != actual:
            violations.append(f"class {int(cls)} demand target {target} achieved {actual}")

    return LASAllocationResult(
        simulated_map=simulated,
        target_demand=demand,
        achieved_demand=achieved,
        unmet_demand=unmet,
        selected_transitions=selected,
        constraint_violations=violations,
    )
```

- [ ] **Step 4: Run allocation tests**

Run:

```bash
python -m pytest tests/test_paper58_las_allocation.py -q
```

Expected: PASS, `3 passed`.

- [ ] **Step 5: Commit allocation**

Run:

```bash
git add scripts/paper58_benchmark/las_allocation.py tests/test_paper58_las_allocation.py
git commit -m "feat: add Paper58 LAS constrained allocation"
```

Expected: commit succeeds.

## Task 4: FLUS-Facing Metrics

**Files:**
- Create: `scripts/paper58_benchmark/las_metrics.py`
- Test: `tests/test_paper58_las_metrics.py`

- [ ] **Step 1: Write failing metrics tests**

Create `tests/test_paper58_las_metrics.py`:

```python
import numpy as np
import pytest

from scripts.paper58_benchmark.las_metrics import (
    allocation_disagreement,
    figure_of_merit,
    method_metric_row,
    quantity_disagreement,
    transition_accuracy,
)


def test_figure_of_merit_uses_change_intersection_over_union():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    true = np.array([[1, 2], [2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [3, 2]], dtype=np.int32)

    assert figure_of_merit(start, true, pred) == pytest.approx(1 / 3)


def test_transition_accuracy_scores_changed_pixels_only():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    true = np.array([[1, 2], [2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int32)

    assert transition_accuracy(start, true, pred) == pytest.approx(0.5)


def test_quantity_and_allocation_disagreement_are_separated():
    true = np.array([[1, 1], [2, 2]], dtype=np.int32)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int32)

    assert quantity_disagreement(true, pred) == pytest.approx(0.25)
    assert allocation_disagreement(true, pred) == pytest.approx(0.0)


def test_method_metric_row_reports_expected_fields():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    true = np.array([[1, 2], [2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int32)

    row = method_metric_row("paper58_las", "external", "tier1", "Wetland", start, true, pred)

    assert row["method"] == "paper58_las"
    assert row["area"] == "external"
    assert row["fom"] == pytest.approx(0.5)
    assert row["transition_accuracy"] == pytest.approx(0.5)
    assert row["true_change_pixels"] == 2
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_paper58_las_metrics.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.paper58_benchmark.las_metrics'`.

- [ ] **Step 3: Implement metrics**

Create `scripts/paper58_benchmark/las_metrics.py`:

```python
from __future__ import annotations

import numpy as np

from scripts.paper58_benchmark.evaluate_benchmark import binary_change_metrics


def figure_of_merit(start_map: np.ndarray, true_map: np.ndarray, pred_map: np.ndarray) -> float:
    start = np.asarray(start_map)
    true = np.asarray(true_map)
    pred = np.asarray(pred_map)
    true_change = true != start
    pred_change = pred != start
    union = true_change | pred_change
    if not np.any(union):
        return 1.0
    intersection = true_change & pred_change
    return float(np.count_nonzero(intersection) / np.count_nonzero(union))


def transition_accuracy(start_map: np.ndarray, true_map: np.ndarray, pred_map: np.ndarray) -> float:
    start = np.asarray(start_map)
    true = np.asarray(true_map)
    pred = np.asarray(pred_map)
    changed = true != start
    if not np.any(changed):
        return 1.0
    correct = (pred == true) & changed
    return float(np.count_nonzero(correct) / np.count_nonzero(changed))


def quantity_disagreement(true_map: np.ndarray, pred_map: np.ndarray) -> float:
    true = np.asarray(true_map)
    pred = np.asarray(pred_map)
    classes = sorted({int(value) for value in np.unique(true)} | {int(value) for value in np.unique(pred)})
    total = int(true.size)
    if total == 0:
        return 0.0
    difference = 0
    for cls in classes:
        difference += abs(int(np.count_nonzero(true == cls)) - int(np.count_nonzero(pred == cls)))
    return float(0.5 * difference / total)


def allocation_disagreement(true_map: np.ndarray, pred_map: np.ndarray) -> float:
    true = np.asarray(true_map)
    pred = np.asarray(pred_map)
    total = int(true.size)
    if total == 0:
        return 0.0
    total_disagreement = 1.0 - float(np.count_nonzero(true == pred) / total)
    return float(max(0.0, total_disagreement - quantity_disagreement(true, pred)))


def method_metric_row(
    method: str,
    area: str,
    tier: str,
    stratum: str,
    start_map: np.ndarray,
    true_map: np.ndarray,
    pred_map: np.ndarray,
) -> dict[str, int | float | str]:
    start = np.asarray(start_map)
    true = np.asarray(true_map)
    pred = np.asarray(pred_map)
    if start.shape != true.shape or start.shape != pred.shape:
        raise ValueError(f"shape mismatch: start={start.shape}, true={true.shape}, pred={pred.shape}")
    change = binary_change_metrics(true != start, pred != start)
    return {
        "method": method,
        "area": area,
        "tier": tier,
        "stratum": stratum,
        "n_pixels": int(start.size),
        "true_change_pixels": int(np.count_nonzero(true != start)),
        "pred_change_pixels": int(np.count_nonzero(pred != start)),
        "change_precision": change["precision"],
        "change_recall": change["recall"],
        "change_f1": change["f1"],
        "fom": figure_of_merit(start, true, pred),
        "transition_accuracy": transition_accuracy(start, true, pred),
        "quantity_disagreement": quantity_disagreement(true, pred),
        "allocation_disagreement": allocation_disagreement(true, pred),
    }
```

- [ ] **Step 4: Run metrics tests**

Run:

```bash
python -m pytest tests/test_paper58_las_metrics.py -q
```

Expected: PASS, `4 passed`.

- [ ] **Step 5: Commit metrics**

Run:

```bash
git add scripts/paper58_benchmark/las_metrics.py tests/test_paper58_las_metrics.py
git commit -m "feat: add Paper58 LAS evaluation metrics"
```

Expected: commit succeeds.

## Task 5: FLUS-Compatible Prediction Ingestion

**Files:**
- Create: `scripts/paper58_benchmark/flus.py`
- Test: `tests/test_paper58_flus_ingestion.py`

- [ ] **Step 1: Write failing FLUS ingestion tests**

Create `tests/test_paper58_flus_ingestion.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from scripts.paper58_benchmark.flus import FLUSIngestionError, load_flus_prediction


def test_load_flus_prediction_accepts_npy(tmp_path: Path):
    path = tmp_path / "flus.npy"
    arr = np.array([[1, 2], [2, 3]], dtype=np.int32)
    np.save(path, arr)

    loaded = load_flus_prediction(path, expected_shape=(2, 2), allowed_classes={1, 2, 3})

    assert np.array_equal(loaded, arr)


def test_load_flus_prediction_accepts_csv(tmp_path: Path):
    path = tmp_path / "flus.csv"
    path.write_text("1,2\n2,3\n", encoding="utf-8")

    loaded = load_flus_prediction(path, expected_shape=(2, 2), allowed_classes={1, 2, 3})

    assert loaded.dtype == np.int32
    assert loaded.tolist() == [[1, 2], [2, 3]]


def test_load_flus_prediction_rejects_shape_mismatch(tmp_path: Path):
    path = tmp_path / "flus.npy"
    np.save(path, np.ones((2, 3), dtype=np.int32))

    with pytest.raises(FLUSIngestionError, match="FLUS prediction shape"):
        load_flus_prediction(path, expected_shape=(2, 2), allowed_classes={1})


def test_load_flus_prediction_rejects_unknown_classes(tmp_path: Path):
    path = tmp_path / "flus.npy"
    np.save(path, np.array([[1, 9]], dtype=np.int32))

    with pytest.raises(FLUSIngestionError, match="unknown FLUS classes: \\[9\\]"):
        load_flus_prediction(path, expected_shape=(1, 2), allowed_classes={1, 2})
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_paper58_flus_ingestion.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.paper58_benchmark.flus'`.

- [ ] **Step 3: Implement FLUS ingestion**

Create `scripts/paper58_benchmark/flus.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np


class FLUSIngestionError(ValueError):
    """Raised when a FLUS-compatible prediction cannot be used in matched evaluation."""


def load_flus_prediction(
    path: Path,
    expected_shape: tuple[int, ...],
    allowed_classes: set[int],
) -> np.ndarray:
    source = Path(path)
    if not source.exists():
        raise FLUSIngestionError(f"FLUS prediction not found: {source}")
    if source.suffix.lower() == ".npy":
        arr = np.load(source)
    elif source.suffix.lower() == ".csv":
        arr = np.loadtxt(source, delimiter=",", dtype=np.int32)
    else:
        raise FLUSIngestionError(f"unsupported FLUS prediction format: {source.suffix}")

    pred = np.asarray(arr, dtype=np.int32)
    if pred.shape != expected_shape:
        raise FLUSIngestionError(f"FLUS prediction shape {pred.shape} does not match expected shape {expected_shape}")
    unknown = sorted({int(value) for value in np.unique(pred)} - {int(value) for value in allowed_classes})
    if unknown:
        raise FLUSIngestionError(f"unknown FLUS classes: {unknown}")
    return pred
```

- [ ] **Step 4: Run FLUS ingestion tests**

Run:

```bash
python -m pytest tests/test_paper58_flus_ingestion.py -q
```

Expected: PASS, `4 passed`.

- [ ] **Step 5: Commit FLUS ingestion**

Run:

```bash
git add scripts/paper58_benchmark/flus.py tests/test_paper58_flus_ingestion.py
git commit -m "feat: add FLUS prediction ingestion"
```

Expected: commit succeeds.

## Task 6: Paper58-LAS Registry Evaluation

**Files:**
- Create: `scripts/paper58_benchmark/evaluate_las.py`
- Test: `tests/test_paper58_las_evaluation.py`

- [ ] **Step 1: Write failing LAS evaluation tests**

Create `tests/test_paper58_las_evaluation.py`:

```python
import json
from pathlib import Path

import numpy as np

from scripts.paper58_benchmark.evaluate_las import evaluate_las


def _provenance_fields() -> dict:
    return {
        "bbox": [120.0, 30.0, 120.1, 30.1],
        "data_source": "test_source",
        "development_contact_status": "none",
        "contact_evidence": "synthetic no-contact evidence",
        "expected_role": "positive_change_candidate",
    }


def test_evaluate_las_writes_method_rows(tmp_path: Path):
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 3]], dtype=np.int32)
    paper58_pred = np.array([[1, 2], [2, 2]], dtype=np.int32)
    flus_pred = np.array([[1, 1], [2, 3]], dtype=np.int32)

    label_start = tmp_path / "start.npy"
    label_end = tmp_path / "end.npy"
    pred_path = tmp_path / "paper58.npy"
    flus_path = tmp_path / "flus.npy"
    np.save(label_start, start)
    np.save(label_end, end)
    np.save(pred_path, paper58_pred)
    np.save(flus_path, flus_pred)

    registry = tmp_path / "benchmark_registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "external",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Wetland",
                        **_provenance_fields(),
                        "label_start_path": str(label_start),
                        "label_end_path": str(label_end),
                        "prediction_path": str(pred_path),
                        "flus_prediction_path": str(flus_path),
                        "qc_status": "include",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "las_out"

    result = evaluate_las(registry_path=registry, output_dir=output_dir)

    assert result["summary"]["n_evaluated_rows"] == 1
    assert result["summary"]["methods"] == ["flus", "paper58_direct", "paper58_las"]
    assert (output_dir / "las_metrics_by_method.csv").exists()
    assert (output_dir / "las_summary.json").exists()
    assert (output_dir / "simulated" / "external_2020_2021_paper58_las.npy").exists()


def test_evaluate_las_keeps_failure_rows_visible(tmp_path: Path):
    registry = tmp_path / "benchmark_registry.json"
    registry.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "excluded",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Urban",
                        **_provenance_fields(),
                        "label_start_path": "missing_start.npy",
                        "label_end_path": "missing_end.npy",
                        "prediction_path": "missing_pred.npy",
                        "qc_status": "class_collapse",
                        "excluded_reason": "synthetic excluded row",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "las_out"

    result = evaluate_las(registry_path=registry, output_dir=output_dir)

    assert result["summary"]["n_evaluated_rows"] == 0
    failures = (output_dir / "las_failures.csv").read_text(encoding="utf-8")
    assert "excluded" in failures
    assert "class_collapse" in failures
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_paper58_las_evaluation.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.paper58_benchmark.evaluate_las'`.

- [ ] **Step 3: Implement LAS evaluation**

Create `scripts/paper58_benchmark/evaluate_las.py`:

```python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper58_benchmark.evaluate_benchmark import _read_registry
from scripts.paper58_benchmark.flus import load_flus_prediction
from scripts.paper58_benchmark.las_allocation import allocate_demand_constrained
from scripts.paper58_benchmark.las_demand import derive_observed_demand
from scripts.paper58_benchmark.las_metrics import method_metric_row
from scripts.paper58_benchmark.las_suitability import (
    build_transition_suitability,
    class_values_from_maps,
    one_hot_probability_cube,
    transition_prior_from_pairs,
)
from scripts.paper58_benchmark.schema import DEFAULT_BENCHMARK_DIR, write_csv, write_json


LAS_METRIC_FIELDS = [
    "method",
    "area",
    "tier",
    "stratum",
    "n_pixels",
    "true_change_pixels",
    "pred_change_pixels",
    "change_precision",
    "change_recall",
    "change_f1",
    "fom",
    "transition_accuracy",
    "quantity_disagreement",
    "allocation_disagreement",
]


def _path(value: object) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _load_array(path_value: object) -> np.ndarray:
    path = _path(path_value)
    if path is None:
        raise FileNotFoundError("missing array path")
    return np.load(path)


def _transition_training_pairs(rows: list[dict[str, Any]], target: dict[str, Any], target_shape: tuple[int, ...]) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for row in rows:
        if row.get("area") == target.get("area"):
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


def _write_failures(path: Path, registry_rows: list[dict[str, Any]]) -> None:
    fields = ["area", "start_year", "end_year", "tier", "qc_status", "excluded_reason"]
    failures = [
        {field: row.get(field, "") for field in fields}
        for row in registry_rows
        if row.get("qc_status") != "include"
    ]
    write_csv(path, failures, fields)


def _write_selected_transitions(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    fields = ["area", "start_year", "end_year", "row", "col", "from_class", "to_class", "score"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _method_names(metric_rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["method"]) for row in metric_rows})


def evaluate_las(
    registry_path: Path = DEFAULT_BENCHMARK_DIR / "benchmark_registry.json",
    output_dir: Path = DEFAULT_BENCHMARK_DIR.parent / "las_results",
) -> dict[str, Any]:
    registry_rows = _read_registry(Path(registry_path))
    included_rows = [row for row in registry_rows if row.get("qc_status") == "include"]
    output = Path(output_dir)
    simulated_dir = output / "simulated"
    output.mkdir(parents=True, exist_ok=True)
    simulated_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, int | float | str]] = []

    for row in included_rows:
        start = _load_array(row.get("label_start_path")).astype(np.int32, copy=False)
        end = _load_array(row.get("label_end_path")).astype(np.int32, copy=False)
        paper58_pred = _load_array(row.get("prediction_path")).astype(np.int32, copy=False)
        if start.shape != end.shape or start.shape != paper58_pred.shape:
            raise ValueError(
                f"shape mismatch for {row.get('area')}: start={start.shape}, end={end.shape}, pred={paper58_pred.shape}"
            )

        class_values = class_values_from_maps(start, end, paper58_pred)
        flus_path = _path(row.get("flus_prediction_path"))
        flus_pred = None
        if flus_path is not None:
            flus_pred = load_flus_prediction(flus_path, expected_shape=start.shape, allowed_classes=set(class_values))
            class_values = class_values_from_maps(start, end, paper58_pred, flus_pred)

        start_probs = one_hot_probability_cube(start, class_values, confidence=0.95, floor=0.01)
        forecast_probs = one_hot_probability_cube(paper58_pred, class_values, confidence=0.95, floor=0.01)
        prior = transition_prior_from_pairs(_transition_training_pairs(included_rows, row, start.shape), class_values)
        suitability = build_transition_suitability(
            start,
            class_values=class_values,
            forecast_probs=forecast_probs,
            start_probs=start_probs,
            transition_prior=prior,
        )
        demand = derive_observed_demand(end)
        allocation = allocate_demand_constrained(
            start,
            suitability,
            class_values=class_values,
            target_demand=demand,
        )

        area = str(row.get("area"))
        start_year = int(row.get("start_year"))
        end_year = int(row.get("end_year"))
        out_name = f"{area}_{start_year}_{end_year}_paper58_las.npy"
        np.save(simulated_dir / out_name, allocation.simulated_map.astype(np.int32, copy=False))

        metric_rows.append(
            method_metric_row(
                "paper58_direct",
                area,
                str(row.get("tier")),
                str(row.get("stratum")),
                start,
                end,
                paper58_pred,
            )
        )
        metric_rows.append(
            method_metric_row(
                "paper58_las",
                area,
                str(row.get("tier")),
                str(row.get("stratum")),
                start,
                end,
                allocation.simulated_map,
            )
        )
        if flus_pred is not None:
            metric_rows.append(
                method_metric_row(
                    "flus",
                    area,
                    str(row.get("tier")),
                    str(row.get("stratum")),
                    start,
                    end,
                    flus_pred,
                )
            )

        for selected in allocation.selected_transitions:
            selected_rows.append(
                {
                    "area": area,
                    "start_year": start_year,
                    "end_year": end_year,
                    **selected,
                }
            )

    summary = {
        "n_registry_rows": len(registry_rows),
        "n_evaluated_rows": len(included_rows),
        "n_metric_rows": len(metric_rows),
        "methods": _method_names(metric_rows),
    }
    result = {"summary": summary, "metrics": metric_rows}
    write_csv(output / "las_metrics_by_method.csv", metric_rows, LAS_METRIC_FIELDS)
    write_json(output / "las_summary.json", result)
    _write_failures(output / "las_failures.csv", registry_rows)
    _write_selected_transitions(output / "las_selected_transitions.csv", selected_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Paper58-LAS against Paper58 direct and FLUS-compatible outputs.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_BENCHMARK_DIR / "benchmark_registry.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_DIR.parent / "las_results")
    args = parser.parse_args()
    result = evaluate_las(registry_path=args.registry, output_dir=args.output_dir)
    print(
        "Paper58-LAS evaluation: "
        f"{result['summary']['n_evaluated_rows']} evaluated row(s), "
        f"methods={','.join(result['summary']['methods'])}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run LAS evaluation tests**

Run:

```bash
python -m pytest tests/test_paper58_las_evaluation.py -q
```

Expected: PASS, `2 passed`.

- [ ] **Step 5: Commit LAS evaluation**

Run:

```bash
git add scripts/paper58_benchmark/evaluate_las.py tests/test_paper58_las_evaluation.py
git commit -m "feat: evaluate Paper58 LAS outputs"
```

Expected: commit succeeds.

## Task 7: Regression Verification And First Synthetic Run

**Files:**
- Modify only if a previous task exposed a concrete test failure in the files created above.

- [ ] **Step 1: Run the new LAS test suite**

Run:

```bash
python -m pytest \
  tests/test_paper58_las_demand.py \
  tests/test_paper58_las_suitability.py \
  tests/test_paper58_las_allocation.py \
  tests/test_paper58_las_metrics.py \
  tests/test_paper58_flus_ingestion.py \
  tests/test_paper58_las_evaluation.py \
  -q
```

Expected: PASS with all new LAS tests passing.

- [ ] **Step 2: Run existing benchmark regression tests**

Run:

```bash
python -m pytest \
  tests/test_paper58_benchmark_baselines.py \
  tests/test_paper58_benchmark_evaluation.py \
  tests/test_paper58_benchmark_statistics.py \
  tests/test_paper58_benchmark_holdouts.py \
  tests/test_paper58_benchmark_registry.py \
  -q
```

Expected: PASS. These tests prove the new simulator path did not break the existing Paper58 benchmark path.

- [ ] **Step 3: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Run LAS on the current benchmark registry in a separate output folder**

Run:

```bash
python -m scripts.paper58_benchmark.evaluate_las \
  --registry paper/rse_submission_paper58/benchmark_results_batch5/benchmark_registry.json \
  --output-dir paper/rse_submission_paper58/las_results_batch5_oracle_demand
```

Expected output includes:

```text
Paper58-LAS evaluation:
```

Expected files:

```text
paper/rse_submission_paper58/las_results_batch5_oracle_demand/las_metrics_by_method.csv
paper/rse_submission_paper58/las_results_batch5_oracle_demand/las_summary.json
paper/rse_submission_paper58/las_results_batch5_oracle_demand/las_failures.csv
paper/rse_submission_paper58/las_results_batch5_oracle_demand/las_selected_transitions.csv
```

- [ ] **Step 5: Inspect LAS summary without changing interpretation**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path("paper/rse_submission_paper58/las_results_batch5_oracle_demand/las_summary.json")
payload = json.loads(path.read_text(encoding="utf-8"))
print(payload["summary"])
PY
```

Expected: the printed summary includes `paper58_direct` and `paper58_las`. It includes `flus` only for rows that provide `flus_prediction_path`.

- [ ] **Step 6: Commit first run artifacts only if the run is intentionally part of the benchmark record**

If the first run is intended as a committed benchmark artifact, run:

```bash
git add paper/rse_submission_paper58/las_results_batch5_oracle_demand
git commit -m "data: run Paper58 LAS Batch 5 oracle-demand evaluation"
```

Expected: commit succeeds.

If the first run is exploratory, do not commit the generated `las_results_batch5_oracle_demand` folder.

## Self-Review Checklist

- Spec coverage:
  - Preserved Paper58 base: Task 6 reads existing registry labels and Paper58 prediction paths without changing existing benchmark code.
  - Latent transition suitability: Task 2 builds a suitability tensor from forecast probability, probability gain, embedding pressure, and transition prior.
  - Demand-constrained allocation: Task 1 and Task 3 implement observed demand and constrained allocation.
  - FLUS-facing evaluation: Task 5 and Task 6 ingest FLUS-compatible outputs and write method-level reports.
  - Metrics: Task 4 adds FOM, transition accuracy, quantity disagreement, and allocation disagreement.
  - Failure visibility: Task 6 writes `las_failures.csv` and does not overwrite Batch 2/3/4/5 reports.
- No existing Paper58 direct benchmark files are modified.
- No LatentDynamicsNet training code is modified.
- No manuscript claim is strengthened by this first build.
