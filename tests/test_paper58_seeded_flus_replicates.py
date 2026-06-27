import pytest


def test_summarize_seeded_deltas_counts_metric_direction_correctly() -> None:
    from scripts.paper58_benchmark.run_seeded_flus_replicates import summarize_seeded_deltas

    rows = [
        {"seed": 1, "area": "a", "method": "paper58_change_gate", "change_f1": 0.30, "allocation_disagreement": 0.05},
        {"seed": 1, "area": "a", "method": "geosos_flus_console", "change_f1": 0.25, "allocation_disagreement": 0.04},
        {"seed": 2, "area": "a", "method": "paper58_change_gate", "change_f1": 0.20, "allocation_disagreement": 0.03},
        {"seed": 2, "area": "a", "method": "geosos_flus_console", "change_f1": 0.22, "allocation_disagreement": 0.04},
    ]

    summary = summarize_seeded_deltas(rows, challenger="paper58_change_gate", baseline="geosos_flus_console")
    by_metric = {row["metric"]: row for row in summary}

    assert by_metric["change_f1"]["n"] == 2
    assert by_metric["change_f1"]["n_better"] == 1
    assert by_metric["change_f1"]["mean_delta"] == pytest.approx(0.015)
    assert by_metric["allocation_disagreement"]["n_better"] == 1
    assert by_metric["allocation_disagreement"]["mean_delta"] == pytest.approx(0.0)


def test_summarize_seeded_overall_deltas_aggregates_area_means_by_seed() -> None:
    from scripts.paper58_benchmark.run_seeded_flus_replicates import summarize_seeded_overall_deltas

    rows = [
        {"seed": 1, "area": "a", "method": "paper58_change_gate", "change_f1": 0.40, "allocation_disagreement": 0.04},
        {"seed": 1, "area": "b", "method": "paper58_change_gate", "change_f1": 0.20, "allocation_disagreement": 0.02},
        {"seed": 1, "area": "a", "method": "geosos_flus_console", "change_f1": 0.10, "allocation_disagreement": 0.08},
        {"seed": 1, "area": "b", "method": "geosos_flus_console", "change_f1": 0.20, "allocation_disagreement": 0.06},
        {"seed": 2, "area": "a", "method": "paper58_change_gate", "change_f1": 0.10, "allocation_disagreement": 0.08},
        {"seed": 2, "area": "b", "method": "paper58_change_gate", "change_f1": 0.20, "allocation_disagreement": 0.06},
        {"seed": 2, "area": "a", "method": "geosos_flus_console", "change_f1": 0.30, "allocation_disagreement": 0.04},
        {"seed": 2, "area": "b", "method": "geosos_flus_console", "change_f1": 0.20, "allocation_disagreement": 0.02},
    ]

    by_seed, summary = summarize_seeded_overall_deltas(
        rows,
        challenger="paper58_change_gate",
        baseline="geosos_flus_console",
    )
    summary_by_metric = {row["metric"]: row for row in summary}
    by_seed_metric = {(row["seed"], row["metric"]): row for row in by_seed}

    assert by_seed_metric[(1, "change_f1")]["delta"] == pytest.approx(0.15)
    assert by_seed_metric[(2, "change_f1")]["delta"] == pytest.approx(-0.10)
    assert summary_by_metric["change_f1"]["n"] == 2
    assert summary_by_metric["change_f1"]["n_better"] == 1
    assert summary_by_metric["change_f1"]["mean_delta"] == pytest.approx(0.025)
    assert by_seed_metric[(1, "allocation_disagreement")]["better"] is True
    assert by_seed_metric[(2, "allocation_disagreement")]["better"] is False
