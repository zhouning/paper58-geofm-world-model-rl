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
        seed_count = _to_int(row.get("n", 0))
        required = limits.required_seed_wins
        gates.append(
            _gate_row(
                "seed_5_of_5",
                metric,
                observed,
                required,
                observed >= required and seed_count == limits.required_seed_count,
                f"{metric} must keep 5/5 seeded mean wins.",
            )
        )
    allocation_row = seeded.get("allocation_disagreement", {})
    allocation_observed = _to_int(allocation_row.get("n_better", 0))
    allocation_seed_count = _to_int(allocation_row.get("n", 0))
    gates.append(
        _gate_row(
            "allocation_seed_wins",
            "allocation_disagreement",
            allocation_observed,
            limits.min_allocation_seed_wins,
            allocation_observed >= limits.min_allocation_seed_wins
            and allocation_seed_count == limits.required_seed_count,
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


def _validate_comparison_rows(
    rows: list[dict[str, Any]],
    *,
    challenger: str,
    baseline: str,
    source_name: str,
) -> None:
    expected = {"challenger": challenger, "baseline": baseline}
    for index, row in enumerate(rows, start=1):
        for field, expected_value in expected.items():
            if field not in row:
                raise ValueError(f"{source_name} row {index} is missing required {field} field")
            actual_value = str(row[field])
            if not actual_value.strip():
                raise ValueError(f"{source_name} row {index} has blank {field} field")
            if actual_value != expected_value:
                raise ValueError(
                    f"{source_name} row {index} has mismatched {field}: "
                    f"{actual_value!r} != {expected_value!r}"
                )


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
        "| Metric | Challenger | Baseline | Delta | Better |",
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
    _validate_comparison_rows(
        seeded_overall_summary,
        challenger=challenger,
        baseline=baseline,
        source_name="seeded_overall_summary",
    )
    _validate_comparison_rows(
        seeded_paired_summary,
        challenger=challenger,
        baseline=baseline,
        source_name="seeded_paired_summary",
    )
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
