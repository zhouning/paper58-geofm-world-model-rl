import csv

import pytest

from scripts.paper58_benchmark.claim_robustness_report import (
    GateThresholds,
    evaluate_acceptance_gates,
    mean_metric_advantages,
    run_claim_robustness_audit,
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


def test_evaluate_acceptance_gates_fails_allocation_when_seed_count_is_short() -> None:
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
        {"metric": "allocation_disagreement", "n_better": "3", "n": "3"},
    ]
    paired_summary = [
        {"metric": "fom", "n_better": "66", "n": "120"},
    ]

    result = evaluate_acceptance_gates(
        mean_advantages,
        seeded_summary,
        paired_summary,
        thresholds=GateThresholds(min_fom_paired_wins=66),
    )

    allocation_gate = [row for row in result["gates"] if row["gate"] == "allocation_seed_wins"]
    assert allocation_gate == [
        {
            "gate": "allocation_seed_wins",
            "metric": "allocation_disagreement",
            "observed": 3,
            "required": 3,
            "passed": False,
            "description": "Allocation disagreement must keep at least 3/5 seeded mean wins.",
        }
    ]
    assert result["passed"] is False


def test_evaluate_acceptance_gates_fails_seed_gate_when_seed_count_is_extra() -> None:
    mean_advantages = [
        {"metric": "change_f1", "better": True},
        {"metric": "fom", "better": True},
        {"metric": "transition_accuracy", "better": True},
        {"metric": "allocation_disagreement", "better": True},
    ]
    seeded_summary = [
        {"metric": "change_f1", "n_better": "5", "n": "6"},
        {"metric": "fom", "n_better": "5", "n": "5"},
        {"metric": "transition_accuracy", "n_better": "5", "n": "5"},
        {"metric": "allocation_disagreement", "n_better": "3", "n": "5"},
    ]
    paired_summary = [
        {"metric": "fom", "n_better": "66", "n": "120"},
    ]

    result = evaluate_acceptance_gates(
        mean_advantages,
        seeded_summary,
        paired_summary,
        thresholds=GateThresholds(min_fom_paired_wins=66),
    )

    seed_gates = [
        row for row in result["gates"] if row["gate"] == "seed_5_of_5" and row["metric"] == "change_f1"
    ]
    assert seed_gates == [
        {
            "gate": "seed_5_of_5",
            "metric": "change_f1",
            "observed": 5,
            "required": 5,
            "passed": False,
            "description": "change_f1 must keep 5/5 seeded mean wins.",
        }
    ]
    assert result["passed"] is False


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


def test_run_claim_robustness_audit_rejects_mismatched_seeded_summary_identity(tmp_path) -> None:
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
            {
                "challenger": "other_candidate",
                "baseline": "geosos_flus_console",
                "metric": "change_f1",
                "n_better": 5,
                "n": 5,
            },
        ],
        ["challenger", "baseline", "metric", "n_better", "n"],
    )
    _write_csv(
        seeded_delta,
        [
            {
                "challenger": "paper58_v2",
                "baseline": "other_baseline",
                "metric": "fom",
                "n_better": 66,
                "n": 120,
            },
        ],
        ["challenger", "baseline", "metric", "n_better", "n"],
    )
    _write_csv(
        seeded_metrics,
        [
            {"seed": 1001, "method": "paper58_v2", "area": "a", "fom": 0.10, "allocation_disagreement": 0.09},
            {"seed": 1001, "method": "geosos_flus_console", "area": "a", "fom": 0.12, "allocation_disagreement": 0.05},
        ],
        ["seed", "method", "area", "fom", "allocation_disagreement"],
    )

    with pytest.raises(
        ValueError,
        match="seeded_overall_summary.*challenger.*other_candidate.*paper58_v2",
    ):
        run_claim_robustness_audit(
            metric_summary_path=metric_summary,
            seeded_overall_summary_path=seeded_overall,
            seeded_paired_summary_path=seeded_delta,
            seeded_metrics_path=seeded_metrics,
            output_dir=output,
            challenger="paper58_v2",
            baseline="geosos_flus_console",
        )

    assert not output.exists()
