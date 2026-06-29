# Paper58 Claim-Robustness Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible claim-robustness audit layer for Paper58 allocator-v2 candidates so a run can be accepted or rejected against the manuscript gates.

**Architecture:** The allocator-v2 helpers already exist in `apply_paper58_spatial_demand_allocation.py` and `tune_paper58_spatial_demand_ratio.py`. This plan adds a focused audit/report module that reads the same-grid and seeded GeoSOS-FLUS comparison outputs, computes the required gates, writes failure tables, and produces a clear pass/fail report without changing historical result directories.

**Tech Stack:** Python 3.11+, standard-library `csv/json/dataclasses/argparse`, NumPy only where existing metric helpers require it, pytest.

---

## File Structure

- Create `scripts/paper58_benchmark/claim_robustness_report.py`
  - Responsibility: read comparison CSVs, compute mean-metric advantages, evaluate acceptance gates, extract failure rows, and write JSON/CSV/Markdown reports.
- Create `tests/test_paper58_claim_robustness_report.py`
  - Responsibility: unit-test metric direction, gate evaluation, failure extraction, and CLI-style report generation.
- Use existing `scripts/paper58_benchmark/apply_paper58_spatial_demand_allocation.py`
  - No first-pass edits expected. It already contains transition reliability, adaptive ratio cap, multi-scale neighborhoods, and CLI flags.
- Use existing `scripts/paper58_benchmark/tune_paper58_spatial_demand_ratio.py`
  - No first-pass edits expected. It already contains v2 grid support and leave-one-area tuning.
- Use existing `scripts/paper58_benchmark/run_seeded_flus_replicates.py`
  - No first-pass edits expected. It already writes `seeded_delta_summary.csv`, `seeded_overall_delta_summary.csv`, and `seeded_metrics_by_method.csv`.

## Task 1: Core Gate Evaluation

**Files:**
- Create: `scripts/paper58_benchmark/claim_robustness_report.py`
- Create: `tests/test_paper58_claim_robustness_report.py`

- [ ] **Step 1: Write failing tests for metric direction and required gates**

Create `tests/test_paper58_claim_robustness_report.py` with:

```python
from scripts.paper58_benchmark.claim_robustness_report import (
    GateThresholds,
    evaluate_acceptance_gates,
    mean_metric_advantages,
)


def test_mean_metric_advantages_handles_lower_is_better() -> None:
    rows = [
        {
            "method": "paper58_v2",
            "mean_change_f1": "0.30",
            "mean_fom": "0.13",
            "mean_transition_accuracy": "0.29",
            "mean_allocation_disagreement": "0.055",
        },
        {
            "method": "geosos_flus_console",
            "mean_change_f1": "0.26",
            "mean_fom": "0.12",
            "mean_transition_accuracy": "0.28",
            "mean_allocation_disagreement": "0.060",
        },
    ]

    advantages = mean_metric_advantages(rows, challenger="paper58_v2", baseline="geosos_flus_console")

    by_metric = {row["metric"]: row for row in advantages}
    assert by_metric["change_f1"]["better"] is True
    assert by_metric["fom"]["better"] is True
    assert by_metric["transition_accuracy"]["better"] is True
    assert by_metric["allocation_disagreement"]["better"] is True
    assert by_metric["allocation_disagreement"]["delta"] == -0.0049999999999999975


def test_evaluate_acceptance_gates_enforces_phase_a_thresholds() -> None:
    mean_advantages = [
        {"metric": "change_f1", "better": True},
        {"metric": "fom", "better": True},
        {"metric": "transition_accuracy", "better": True},
        {"metric": "allocation_disagreement", "better": True},
    ]
    seeded_summary = [
        {"metric": "change_f1", "n_better": "5", "n": "5"},
        {"metric": "fom", "n_better": "5", "n": "5"},
        {"metric": "transition_accuracy", "n_better": "5", "n": "5"},
        {"metric": "allocation_disagreement", "n_better": "3", "n": "5"},
    ]
    paired_summary = [
        {"metric": "change_f1", "n_better": "67", "n": "120"},
        {"metric": "fom", "n_better": "66", "n": "120"},
        {"metric": "transition_accuracy", "n_better": "58", "n": "120"},
        {"metric": "allocation_disagreement", "n_better": "81", "n": "120"},
    ]

    result = evaluate_acceptance_gates(
        mean_advantages,
        seeded_summary,
        paired_summary,
        thresholds=GateThresholds(min_fom_paired_wins=66),
    )

    assert result["passed"] is True
    assert all(row["passed"] for row in result["gates"])


def test_evaluate_acceptance_gates_fails_when_fom_paired_wins_do_not_improve() -> None:
    mean_advantages = [
        {"metric": "change_f1", "better": True},
        {"metric": "fom", "better": True},
        {"metric": "transition_accuracy", "better": True},
        {"metric": "allocation_disagreement", "better": True},
    ]
    seeded_summary = [
        {"metric": "change_f1", "n_better": "5", "n": "5"},
        {"metric": "fom", "n_better": "5", "n": "5"},
        {"metric": "transition_accuracy", "n_better": "5", "n": "5"},
        {"metric": "allocation_disagreement", "n_better": "3", "n": "5"},
    ]
    paired_summary = [
        {"metric": "fom", "n_better": "65", "n": "120"},
    ]

    result = evaluate_acceptance_gates(
        mean_advantages,
        seeded_summary,
        paired_summary,
        thresholds=GateThresholds(min_fom_paired_wins=66),
    )

    assert result["passed"] is False
    failed = [row for row in result["gates"] if not row["passed"]]
    assert failed == [
        {
            "gate": "paired_fom_wins",
            "metric": "fom",
            "observed": 65,
            "required": 66,
            "passed": False,
            "description": "FoM area-by-seed paired wins must improve beyond the current 61/120 baseline.",
        }
    ]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_paper58_claim_robustness_report.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.paper58_benchmark.claim_robustness_report'`.

- [ ] **Step 3: Implement core gate helpers**

Create `scripts/paper58_benchmark/claim_robustness_report.py` with:

```python
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METRICS = ["change_f1", "fom", "transition_accuracy", "allocation_disagreement"]
LOWER_IS_BETTER = {"allocation_disagreement", "quantity_disagreement"}


@dataclass(frozen=True)
class GateThresholds:
    required_mean_metric_wins: int = 4
    required_seed_wins: int = 5
    required_seed_count: int = 5
    min_allocation_seed_wins: int = 3
    min_fom_paired_wins: int = 66


def _to_float(value: Any) -> float:
    return float(value)


def _to_int(value: Any) -> int:
    return int(float(value))


def metric_is_better(metric: str, challenger_value: float, baseline_value: float) -> bool:
    if metric in LOWER_IS_BETTER:
        return challenger_value < baseline_value
    return challenger_value > baseline_value


def mean_metric_advantages(
    metric_summary_rows: list[dict[str, Any]],
    *,
    challenger: str,
    baseline: str,
) -> list[dict[str, Any]]:
    by_method = {str(row["method"]): row for row in metric_summary_rows}
    if challenger not in by_method:
        raise ValueError(f"missing challenger method in metric summary: {challenger}")
    if baseline not in by_method:
        raise ValueError(f"missing baseline method in metric summary: {baseline}")
    challenger_row = by_method[challenger]
    baseline_row = by_method[baseline]
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        challenger_value = _to_float(challenger_row[f"mean_{metric}"])
        baseline_value = _to_float(baseline_row[f"mean_{metric}"])
        rows.append(
            {
                "metric": metric,
                "challenger": challenger,
                "baseline": baseline,
                "challenger_mean": challenger_value,
                "baseline_mean": baseline_value,
                "delta": challenger_value - baseline_value,
                "higher_is_better": metric not in LOWER_IS_BETTER,
                "better": metric_is_better(metric, challenger_value, baseline_value),
            }
        )
    return rows


def _summary_by_metric(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["metric"]): row for row in rows}


def _gate_row(gate: str, metric: str, observed: int, required: int, passed: bool, description: str) -> dict[str, Any]:
    return {
        "gate": gate,
        "metric": metric,
        "observed": int(observed),
        "required": int(required),
        "passed": bool(passed),
        "description": description,
    }


def evaluate_acceptance_gates(
    mean_advantages: list[dict[str, Any]],
    seeded_overall_summary: list[dict[str, Any]],
    paired_summary: list[dict[str, Any]],
    *,
    thresholds: GateThresholds | None = None,
) -> dict[str, Any]:
    limits = thresholds or GateThresholds()
    gates: list[dict[str, Any]] = []

    mean_wins = sum(1 for row in mean_advantages if bool(row["better"]))
    gates.append(
        _gate_row(
            "mean_4_of_4",
            "all",
            mean_wins,
            limits.required_mean_metric_wins,
            mean_wins >= limits.required_mean_metric_wins,
            "24-township mean metrics must remain 4/4 better than fixed geosos_flus_console.",
        )
    )

    seeded = _summary_by_metric(seeded_overall_summary)
    for metric in ("change_f1", "fom", "transition_accuracy"):
        row = seeded.get(metric, {})
        observed = _to_int(row.get("n_better", 0))
        required = limits.required_seed_wins
        gates.append(
            _gate_row(
                "seed_5_of_5",
                metric,
                observed,
                required,
                observed >= required and _to_int(row.get("n", 0)) >= limits.required_seed_count,
                f"{metric} must keep 5/5 seeded mean wins.",
            )
        )
    allocation_row = seeded.get("allocation_disagreement", {})
    allocation_observed = _to_int(allocation_row.get("n_better", 0))
    gates.append(
        _gate_row(
            "allocation_seed_wins",
            "allocation_disagreement",
            allocation_observed,
            limits.min_allocation_seed_wins,
            allocation_observed >= limits.min_allocation_seed_wins,
            "Allocation disagreement must keep at least 3/5 seeded mean wins.",
        )
    )

    paired = _summary_by_metric(paired_summary)
    fom_row = paired.get("fom", {})
    fom_observed = _to_int(fom_row.get("n_better", 0))
    gates.append(
        _gate_row(
            "paired_fom_wins",
            "fom",
            fom_observed,
            limits.min_fom_paired_wins,
            fom_observed >= limits.min_fom_paired_wins,
            "FoM area-by-seed paired wins must improve beyond the current 61/120 baseline.",
        )
    )

    return {
        "passed": all(bool(row["passed"]) for row in gates),
        "thresholds": asdict(limits),
        "gates": gates,
    }
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
python -m pytest tests/test_paper58_claim_robustness_report.py -q
```

Expected: PASS for the three tests in this task.

- [ ] **Step 5: Commit core gate helpers**

Run:

```bash
git add scripts/paper58_benchmark/claim_robustness_report.py tests/test_paper58_claim_robustness_report.py
git commit -m "feat: add paper58 claim robustness gates"
```

Expected: commit succeeds with only the new module and test file.

## Task 2: Failure Row Extraction

**Files:**
- Modify: `scripts/paper58_benchmark/claim_robustness_report.py`
- Modify: `tests/test_paper58_claim_robustness_report.py`

- [ ] **Step 1: Add failing failure-table test**

Append to `tests/test_paper58_claim_robustness_report.py`:

```python
from scripts.paper58_benchmark.claim_robustness_report import failure_rows_from_seeded_metrics


def test_failure_rows_from_seeded_metrics_reports_fom_losses_and_allocation_degradation() -> None:
    rows = [
        {"seed": "1001", "method": "paper58_v2", "area": "a", "fom": "0.10", "allocation_disagreement": "0.09"},
        {"seed": "1001", "method": "geosos_flus_console", "area": "a", "fom": "0.12", "allocation_disagreement": "0.05"},
        {"seed": "1001", "method": "paper58_v2", "area": "b", "fom": "0.20", "allocation_disagreement": "0.04"},
        {"seed": "1001", "method": "geosos_flus_console", "area": "b", "fom": "0.18", "allocation_disagreement": "0.05"},
    ]

    failures = failure_rows_from_seeded_metrics(
        rows,
        challenger="paper58_v2",
        baseline="geosos_flus_console",
        allocation_degradation_threshold=0.02,
    )

    assert failures == [
        {
            "seed": 1001,
            "area": "a",
            "fom_delta": -0.01999999999999999,
            "allocation_disagreement_delta": 0.039999999999999994,
            "fom_loss": True,
            "large_allocation_degradation": True,
        }
    ]
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
python -m pytest tests/test_paper58_claim_robustness_report.py::test_failure_rows_from_seeded_metrics_reports_fom_losses_and_allocation_degradation -q
```

Expected: FAIL with `ImportError` or `AttributeError` because `failure_rows_from_seeded_metrics` is not defined.

- [ ] **Step 3: Implement failure extraction**

Append to `scripts/paper58_benchmark/claim_robustness_report.py`:

```python
def failure_rows_from_seeded_metrics(
    metric_rows: list[dict[str, Any]],
    *,
    challenger: str,
    baseline: str,
    allocation_degradation_threshold: float = 0.02,
) -> list[dict[str, Any]]:
    keyed: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in metric_rows:
        keyed[(_to_int(row["seed"]), str(row["area"]), str(row["method"]))] = row

    failures: list[dict[str, Any]] = []
    for seed, area, method in sorted(keyed):
        if method != challenger:
            continue
        challenger_row = keyed[(seed, area, challenger)]
        baseline_row = keyed.get((seed, area, baseline))
        if baseline_row is None:
            continue
        fom_delta = _to_float(challenger_row["fom"]) - _to_float(baseline_row["fom"])
        allocation_delta = _to_float(challenger_row["allocation_disagreement"]) - _to_float(
            baseline_row["allocation_disagreement"]
        )
        fom_loss = fom_delta < 0.0
        large_allocation_degradation = allocation_delta > float(allocation_degradation_threshold)
        if fom_loss or large_allocation_degradation:
            failures.append(
                {
                    "seed": int(seed),
                    "area": area,
                    "fom_delta": fom_delta,
                    "allocation_disagreement_delta": allocation_delta,
                    "fom_loss": bool(fom_loss),
                    "large_allocation_degradation": bool(large_allocation_degradation),
                }
            )
    return failures
```

- [ ] **Step 4: Run the focused test**

Run:

```bash
python -m pytest tests/test_paper58_claim_robustness_report.py::test_failure_rows_from_seeded_metrics_reports_fom_losses_and_allocation_degradation -q
```

Expected: PASS.

- [ ] **Step 5: Run all claim robustness tests**

Run:

```bash
python -m pytest tests/test_paper58_claim_robustness_report.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit failure extraction**

Run:

```bash
git add scripts/paper58_benchmark/claim_robustness_report.py tests/test_paper58_claim_robustness_report.py
git commit -m "feat: report paper58 claim failure rows"
```

Expected: commit succeeds.

## Task 3: Report Writer And CLI

**Files:**
- Modify: `scripts/paper58_benchmark/claim_robustness_report.py`
- Modify: `tests/test_paper58_claim_robustness_report.py`

- [ ] **Step 1: Add failing report-generation test**

Append to `tests/test_paper58_claim_robustness_report.py`:

```python
import csv

from scripts.paper58_benchmark.claim_robustness_report import run_claim_robustness_audit


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_run_claim_robustness_audit_writes_outputs(tmp_path) -> None:
    metric_summary = tmp_path / "metric_summary_by_method.csv"
    seeded_overall = tmp_path / "seeded_overall_delta_summary.csv"
    seeded_delta = tmp_path / "seeded_delta_summary.csv"
    seeded_metrics = tmp_path / "seeded_metrics_by_method.csv"
    output = tmp_path / "audit"

    _write_csv(
        metric_summary,
        [
            {
                "method": "paper58_v2",
                "mean_change_f1": 0.30,
                "mean_fom": 0.13,
                "mean_transition_accuracy": 0.29,
                "mean_allocation_disagreement": 0.055,
            },
            {
                "method": "geosos_flus_console",
                "mean_change_f1": 0.26,
                "mean_fom": 0.12,
                "mean_transition_accuracy": 0.28,
                "mean_allocation_disagreement": 0.060,
            },
        ],
        ["method", "mean_change_f1", "mean_fom", "mean_transition_accuracy", "mean_allocation_disagreement"],
    )
    _write_csv(
        seeded_overall,
        [
            {"metric": "change_f1", "n_better": 5, "n": 5},
            {"metric": "fom", "n_better": 5, "n": 5},
            {"metric": "transition_accuracy", "n_better": 5, "n": 5},
            {"metric": "allocation_disagreement", "n_better": 3, "n": 5},
        ],
        ["metric", "n_better", "n"],
    )
    _write_csv(
        seeded_delta,
        [
            {"metric": "change_f1", "n_better": 67, "n": 120},
            {"metric": "fom", "n_better": 66, "n": 120},
            {"metric": "transition_accuracy", "n_better": 58, "n": 120},
            {"metric": "allocation_disagreement", "n_better": 81, "n": 120},
        ],
        ["metric", "n_better", "n"],
    )
    _write_csv(
        seeded_metrics,
        [
            {"seed": 1001, "method": "paper58_v2", "area": "a", "fom": 0.10, "allocation_disagreement": 0.09},
            {"seed": 1001, "method": "geosos_flus_console", "area": "a", "fom": 0.12, "allocation_disagreement": 0.05},
        ],
        ["seed", "method", "area", "fom", "allocation_disagreement"],
    )

    result = run_claim_robustness_audit(
        metric_summary_path=metric_summary,
        seeded_overall_summary_path=seeded_overall,
        seeded_paired_summary_path=seeded_delta,
        seeded_metrics_path=seeded_metrics,
        output_dir=output,
        challenger="paper58_v2",
        baseline="geosos_flus_console",
    )

    assert result["acceptance"]["passed"] is True
    assert (output / "claim_robustness_summary.json").exists()
    assert (output / "mean_metric_advantages.csv").exists()
    assert (output / "acceptance_gates.csv").exists()
    assert (output / "failure_rows.csv").exists()
    assert "Phase-A required gates: PASS" in (output / "README.md").read_text(encoding="utf-8")
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
python -m pytest tests/test_paper58_claim_robustness_report.py::test_run_claim_robustness_audit_writes_outputs -q
```

Expected: FAIL because `run_claim_robustness_audit` is not defined.

- [ ] **Step 3: Implement CSV/JSON/Markdown writer and CLI**

Append to `scripts/paper58_benchmark/claim_robustness_report.py`:

```python
def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_ready(row.get(field, "")) for field in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _format_bool(value: bool) -> str:
    return "PASS" if bool(value) else "FAIL"


def write_markdown_report(
    output_dir: Path,
    *,
    challenger: str,
    baseline: str,
    mean_advantages: list[dict[str, Any]],
    acceptance: dict[str, Any],
    failure_rows: list[dict[str, Any]],
) -> None:
    output = Path(output_dir)
    lines = [
        "# Paper58 Claim-Robustness Audit",
        "",
        f"- Challenger: `{challenger}`",
        f"- Baseline: `{baseline}`",
        f"- Phase-A required gates: {_format_bool(bool(acceptance['passed']))}",
        "",
        "## Mean Metric Advantages",
        "",
        "| Metric | Paper58 | GeoSOS-FLUS | Delta | Better |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in mean_advantages:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["metric"]),
                    f"{float(row['challenger_mean']):.6f}",
                    f"{float(row['baseline_mean']):.6f}",
                    f"{float(row['delta']):.6f}",
                    "yes" if bool(row["better"]) else "no",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Acceptance Gates",
            "",
            "| Gate | Metric | Observed | Required | Result |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in acceptance["gates"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["gate"]),
                    str(row["metric"]),
                    str(row["observed"]),
                    str(row["required"]),
                    _format_bool(bool(row["passed"])),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Failure Rows",
            "",
            "| Seed | Area | FoM Delta | Allocation Disagreement Delta | FoM Loss | Large Allocation Degradation |",
            "| ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in failure_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    f"`{row['area']}`",
                    f"{float(row['fom_delta']):.6f}",
                    f"{float(row['allocation_disagreement_delta']):.6f}",
                    "yes" if bool(row["fom_loss"]) else "no",
                    "yes" if bool(row["large_allocation_degradation"]) else "no",
                ]
            )
            + " |"
        )
    if not failure_rows:
        lines.append("|  | none |  |  |  |  |")
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_claim_robustness_audit(
    *,
    metric_summary_path: Path,
    seeded_overall_summary_path: Path,
    seeded_paired_summary_path: Path,
    seeded_metrics_path: Path,
    output_dir: Path,
    challenger: str,
    baseline: str = "geosos_flus_console",
    thresholds: GateThresholds | None = None,
) -> dict[str, Any]:
    metric_summary_rows = _read_csv(Path(metric_summary_path))
    seeded_overall_summary = _read_csv(Path(seeded_overall_summary_path))
    seeded_paired_summary = _read_csv(Path(seeded_paired_summary_path))
    seeded_metric_rows = _read_csv(Path(seeded_metrics_path))
    mean_advantages = mean_metric_advantages(metric_summary_rows, challenger=challenger, baseline=baseline)
    acceptance = evaluate_acceptance_gates(
        mean_advantages,
        seeded_overall_summary,
        seeded_paired_summary,
        thresholds=thresholds,
    )
    failures = failure_rows_from_seeded_metrics(
        seeded_metric_rows,
        challenger=challenger,
        baseline=baseline,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output / "mean_metric_advantages.csv",
        mean_advantages,
        ["metric", "challenger", "baseline", "challenger_mean", "baseline_mean", "delta", "higher_is_better", "better"],
    )
    _write_csv(
        output / "acceptance_gates.csv",
        acceptance["gates"],
        ["gate", "metric", "observed", "required", "passed", "description"],
    )
    _write_csv(
        output / "failure_rows.csv",
        failures,
        ["seed", "area", "fom_delta", "allocation_disagreement_delta", "fom_loss", "large_allocation_degradation"],
    )
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "challenger": challenger,
        "baseline": baseline,
        "metric_summary_path": Path(metric_summary_path),
        "seeded_overall_summary_path": Path(seeded_overall_summary_path),
        "seeded_paired_summary_path": Path(seeded_paired_summary_path),
        "seeded_metrics_path": Path(seeded_metrics_path),
        "mean_advantages": mean_advantages,
        "acceptance": acceptance,
        "failure_rows": failures,
    }
    _write_json(output / "claim_robustness_summary.json", payload)
    write_markdown_report(
        output,
        challenger=challenger,
        baseline=baseline,
        mean_advantages=mean_advantages,
        acceptance=acceptance,
        failure_rows=failures,
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Paper58 claim robustness against fixed GeoSOS-FLUS outputs.")
    parser.add_argument("--metric-summary", type=Path, required=True)
    parser.add_argument("--seeded-overall-summary", type=Path, required=True)
    parser.add_argument("--seeded-paired-summary", type=Path, required=True)
    parser.add_argument("--seeded-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--baseline", default="geosos_flus_console")
    args = parser.parse_args(argv)
    result = run_claim_robustness_audit(
        metric_summary_path=args.metric_summary,
        seeded_overall_summary_path=args.seeded_overall_summary,
        seeded_paired_summary_path=args.seeded_paired_summary,
        seeded_metrics_path=args.seeded_metrics,
        output_dir=args.output_dir,
        challenger=args.challenger,
        baseline=args.baseline,
    )
    print(
        "Paper58 claim robustness audit complete: "
        f"passed={result['acceptance']['passed']}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run report tests**

Run:

```bash
python -m pytest tests/test_paper58_claim_robustness_report.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit report writer and CLI**

Run:

```bash
git add scripts/paper58_benchmark/claim_robustness_report.py tests/test_paper58_claim_robustness_report.py
git commit -m "feat: add paper58 claim robustness report"
```

Expected: commit succeeds.

## Task 4: Verify Existing Allocator-V2 Tests Still Pass

**Files:**
- No file changes.

- [ ] **Step 1: Run the existing focused allocator tests**

Run:

```bash
python -m pytest tests/test_paper58_spatial_demand_allocation.py tests/test_paper58_spatial_demand_ratio_tuning.py tests/test_paper58_claim_robustness_report.py -q
```

Expected: PASS.

- [ ] **Step 2: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

## Task 5: Baseline Claim Audit On Current `transition_floor030`

**Files:**
- Generated output only under `paper/rse_submission_paper58/paper58_claim_robustness_transition_floor030_audit_2026-06-30/`.

- [ ] **Step 1: Run the audit against the current known benchmark**

Run:

```bash
python -m scripts.paper58_benchmark.claim_robustness_report \
  --metric-summary paper/rse_submission_paper58/paper58_true_geosos_flus_xiangzhen24_fixed_flus_external_loo_transition_floor030_same_grid_2026-06-27/metric_summary_by_method.csv \
  --seeded-overall-summary paper/rse_submission_paper58/paper58_true_geosos_flus_xiangzhen24_fixed_flus_external_loo_transition_floor030_seeded_replicates_2026-06-27/seeded_overall_delta_summary.csv \
  --seeded-paired-summary paper/rse_submission_paper58/paper58_true_geosos_flus_xiangzhen24_fixed_flus_external_loo_transition_floor030_seeded_replicates_2026-06-27/seeded_delta_summary.csv \
  --seeded-metrics paper/rse_submission_paper58/paper58_true_geosos_flus_xiangzhen24_fixed_flus_external_loo_transition_floor030_seeded_replicates_2026-06-27/seeded_metrics_by_method.csv \
  --challenger paper58_spatial_demand_ratio_external_loo_transition_floor030 \
  --baseline geosos_flus_console \
  --output-dir paper/rse_submission_paper58/paper58_claim_robustness_transition_floor030_audit_2026-06-30
```

Expected: command completes and prints `passed=False` because the current baseline has FoM paired wins `61/120`, below the Phase-A threshold `66/120`.

- [ ] **Step 2: Inspect the generated acceptance gates**

Run:

```bash
cat paper/rse_submission_paper58/paper58_claim_robustness_transition_floor030_audit_2026-06-30/acceptance_gates.csv
```

Expected rows include:

```csv
gate,metric,observed,required,passed,description
mean_4_of_4,all,4,4,True,24-township mean metrics must remain 4/4 better than fixed geosos_flus_console.
seed_5_of_5,change_f1,5,5,True,change_f1 must keep 5/5 seeded mean wins.
seed_5_of_5,fom,5,5,True,fom must keep 5/5 seeded mean wins.
seed_5_of_5,transition_accuracy,5,5,True,transition_accuracy must keep 5/5 seeded mean wins.
allocation_seed_wins,allocation_disagreement,3,3,True,Allocation disagreement must keep at least 3/5 seeded mean wins.
paired_fom_wins,fom,61,66,False,FoM area-by-seed paired wins must improve beyond the current 61/120 baseline.
```

- [ ] **Step 3: Leave generated audit output uncommitted**

Run:

```bash
git status --short -- paper/rse_submission_paper58/paper58_claim_robustness_transition_floor030_audit_2026-06-30
```

Expected: the audit directory appears as untracked generated output. Do not commit generated result directories during this source-code implementation pass.

## Task 6: Evaluate A New V2 Candidate Against The Gates

**Files:**
- Generated outputs only under new dated result directories.

- [ ] **Step 1: Run focused tests before generating candidate outputs**

Run:

```bash
python -m pytest tests/test_paper58_spatial_demand_allocation.py tests/test_paper58_spatial_demand_ratio_tuning.py tests/test_paper58_claim_robustness_report.py -q
```

Expected: PASS.

- [ ] **Step 2: Run a constrained v2 tuning sweep**

Run a constrained v2 tuning sweep on the external calibration change-gate cases:

```bash
python -m scripts.paper58_benchmark.tune_paper58_spatial_demand_ratio \
  --change-gate-root paper/rse_submission_paper58/paper58_external_calibration_change_gate_cases_changeaware43_2026-06-29 \
  --output-dir paper/rse_submission_paper58/paper58_claim_robustness_v2_tuning_2026-06-30 \
  --ratio-quantiles 0.25,0.30 \
  --ratio-multipliers 1.0,1.25,1.5 \
  --min-fractions 0.003,0.005 \
  --max-fractions 0.22,0.25 \
  --target-neighborhood-weights 1.0,1.25,1.5 \
  --source-neighborhood-penalties 0.75,1.0,1.25 \
  --min-mean-transition-accuracy 0.30 \
  --transition-reliability-strengths 0.5,1.0 \
  --neighborhood-window-sizes 3,5,9 \
  --enable-adaptive-ratio-cap-values false,true \
  --high-candidate-thresholds 0.5,0.6 \
  --high-candidate-quantiles 0.25,0.5
```

Expected: command completes and writes `manifest.json`, `ratio_tuning_summary.csv`, and `ratio_tuning_case_metrics.csv`.

- [ ] **Step 3: Apply the selected v2 parameters to the 24-township target set**

Run this exact driver from the repository root. It reads the known `transition_floor030` manifest for source paths and reads `best_parameters` from the new tuning manifest:

```bash
python - <<'PY'
import json
from pathlib import Path

from scripts.paper58_benchmark.apply_paper58_spatial_demand_allocation import run_spatial_demand_allocation

baseline_manifest = json.loads(Path(
    "paper/rse_submission_paper58/"
    "paper58_spatial_demand_ratio_external_loo_transition_floor030_p25x15_min005_tw10_sp05_xiangzhen24_2026-06-27/"
    "manifest.json"
).read_text(encoding="utf-8"))
tuning_manifest = json.loads(Path(
    "paper/rse_submission_paper58/paper58_claim_robustness_v2_tuning_2026-06-30/manifest.json"
).read_text(encoding="utf-8"))
params = tuning_manifest["best_parameters"]

run_spatial_demand_allocation(
    source_prediction_dir=Path(baseline_manifest["source_prediction_dir"]),
    change_gate_dirs=[Path(path) for path in baseline_manifest["change_gate_dirs"]],
    calibration_label_dir=Path(baseline_manifest["calibration_label_dir"]),
    calibration_prediction_dir=Path(baseline_manifest["calibration_prediction_dir"]),
    output_dir=Path("paper/rse_submission_paper58/paper58_claim_robustness_v2_candidate_xiangzhen24_2026-06-30"),
    selector_class=int(baseline_manifest["parameters"]["selector_class"]),
    min_fraction=float(params["min_fraction"]),
    max_fraction=float(params["max_fraction"]),
    target_neighborhood_weight=float(params["target_neighborhood_weight"]),
    source_neighborhood_penalty=float(params["source_neighborhood_penalty"]),
    calibration_areas=[str(area) for area in baseline_manifest["calibration_areas"]],
    demand_strategy="ratio",
    ratio_quantile=float(params["ratio_quantile"]),
    ratio_multiplier=float(params["ratio_multiplier"]),
    enable_transition_reliability=bool(params.get("enable_transition_reliability", False)),
    transition_reliability_strength=float(params.get("transition_reliability_strength", 0.0)),
    neighborhood_window_sizes=params.get("neighborhood_window_sizes"),
    enable_adaptive_ratio_cap=bool(params.get("enable_adaptive_ratio_cap", False)),
    high_candidate_threshold=float(params.get("high_candidate_threshold", 0.5)),
    high_candidate_quantile=float(params.get("high_candidate_quantile", 0.5)),
)
PY
```

Expected: `paper/rse_submission_paper58/paper58_claim_robustness_v2_candidate_xiangzhen24_2026-06-30/manifest.json` and a `predictions/` directory are created.

- [ ] **Step 4: Run the 24-township same-grid fixed-FLUS comparison**

Run the existing same-grid comparison harness with the v2 candidate as the challenger and write a new output directory:

```bash
python -m scripts.paper58_benchmark.run_true_paper58_flus_same_grid_comparison \
  --paper58-predictions-dir paper/rse_submission_paper58/paper58_true_repo_weights_subset_2026-06-27 \
  --labels-dir data/independent_change_labels/labels \
  --output-dir paper/rse_submission_paper58/paper58_true_geosos_flus_xiangzhen24_claim_robustness_v2_same_grid_2026-06-30 \
  --flus-executable /Users/zhouning/FLUS_console_crossplatform/build/flus_console \
  --flus-demand-source paper58_prediction \
  --extra-prediction paper58_spatial_demand_ratio_claim_robustness_v2=paper/rse_submission_paper58/paper58_claim_robustness_v2_candidate_xiangzhen24_2026-06-30/predictions
```

Expected: the output directory contains `metric_summary_by_method.csv`, `metrics_by_method.csv`, `summary.json`, and `README.md`.

- [ ] **Step 5: Run five seeded GeoSOS-FLUS replicates for the v2 candidate**

Run `scripts.paper58_benchmark.run_seeded_flus_replicates` using seeds `1001` through `1005`:

```bash
python -m scripts.paper58_benchmark.run_seeded_flus_replicates \
  --paper58-predictions-dir paper/rse_submission_paper58/paper58_true_repo_weights_subset_2026-06-27 \
  --labels-dir data/independent_change_labels/labels \
  --output-dir paper/rse_submission_paper58/paper58_true_geosos_flus_xiangzhen24_claim_robustness_v2_seeded_replicates_2026-06-30 \
  --flus-executable /Users/zhouning/FLUS_console_crossplatform/build/flus_console \
  --flus-demand-source paper58_prediction \
  --extra-prediction paper58_spatial_demand_ratio_claim_robustness_v2=paper/rse_submission_paper58/paper58_claim_robustness_v2_candidate_xiangzhen24_2026-06-30/predictions \
  --seed 1001 \
  --seed 1002 \
  --seed 1003 \
  --seed 1004 \
  --seed 1005 \
  --challenger paper58_spatial_demand_ratio_claim_robustness_v2 \
  --baseline geosos_flus_console
```

Expected: the output directory contains `seeded_delta_summary.csv`, `seeded_overall_delta_summary.csv`, `seeded_metrics_by_method.csv`, and `README.md`.

- [ ] **Step 6: Audit the v2 candidate against Phase-A gates**

Run:

```bash
python -m scripts.paper58_benchmark.claim_robustness_report \
  --metric-summary paper/rse_submission_paper58/paper58_true_geosos_flus_xiangzhen24_claim_robustness_v2_same_grid_2026-06-30/metric_summary_by_method.csv \
  --seeded-overall-summary paper/rse_submission_paper58/paper58_true_geosos_flus_xiangzhen24_claim_robustness_v2_seeded_replicates_2026-06-30/seeded_overall_delta_summary.csv \
  --seeded-paired-summary paper/rse_submission_paper58/paper58_true_geosos_flus_xiangzhen24_claim_robustness_v2_seeded_replicates_2026-06-30/seeded_delta_summary.csv \
  --seeded-metrics paper/rse_submission_paper58/paper58_true_geosos_flus_xiangzhen24_claim_robustness_v2_seeded_replicates_2026-06-30/seeded_metrics_by_method.csv \
  --challenger paper58_spatial_demand_ratio_claim_robustness_v2 \
  --baseline geosos_flus_console \
  --output-dir paper/rse_submission_paper58/paper58_claim_robustness_v2_audit_2026-06-30
```

Expected:

- If `claim_robustness_summary.json` has `"passed": true`, v2 is a manuscript-safe Phase-A candidate.
- If `"passed": false`, keep `transition_floor030` as the manuscript-safe aggregate candidate and report v2 as diagnostic.

## Final Verification

- [ ] **Step 1: Run targeted tests**

Run:

```bash
python -m pytest tests/test_paper58_spatial_demand_allocation.py tests/test_paper58_spatial_demand_ratio_tuning.py tests/test_paper58_claim_robustness_report.py -q
```

Expected: PASS.

- [ ] **Step 2: Check staged changes before final commit**

Run:

```bash
git status --short
git diff --check
```

Expected: `git diff --check` exits 0. `git status --short` should show only intentional source, test, plan, and small audit report files.

- [ ] **Step 3: Commit final implementation**

Run:

```bash
git add scripts/paper58_benchmark/claim_robustness_report.py tests/test_paper58_claim_robustness_report.py docs/superpowers/plans/2026-06-30-paper58-claim-robustness-optimization.md
git commit -m "feat: add paper58 claim robustness audit"
```

Expected: commit succeeds. If Task 5 small audit outputs were intentionally committed earlier, do not re-add them here.
