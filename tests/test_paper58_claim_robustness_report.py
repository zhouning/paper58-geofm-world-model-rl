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
