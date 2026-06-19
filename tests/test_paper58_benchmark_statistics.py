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


def test_gate_report_reports_insufficient_tier1_when_no_tier1_rows_exist():
    assert gate_report([])["status"] == "insufficient_tier1"
    assert gate_report(
        [
            {"area": "a", "tier": "tier2", "stratum": "Wetland", "primary_change_advantage": 0.1},
        ]
    )["status"] == "insufficient_tier1"


def test_gate_report_fails_with_single_tier1_cluster_even_if_values_are_positive():
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
            "area": "a",
            "tier": "tier1",
            "stratum": "Forest",
            "primary_change_advantage": 0.15,
            "spatial_change_advantage": 0.08,
            "embedding_advantage": 0.02,
        },
        {
            "area": "a",
            "tier": "tier1",
            "stratum": "Urban",
            "primary_change_advantage": 0.12,
            "spatial_change_advantage": 0.07,
            "embedding_advantage": 0.01,
        },
    ]

    report = gate_report(rows, n_boot=200, seed=9)

    assert report["status"] == "fail"
    assert report["tier1_primary_change"]["ci_low"] is None
    assert report["tier1_spatial_change"]["ci_low"] is None


def test_clustered_bootstrap_ignores_missing_or_blank_cluster_identifiers():
    rows = [
        {"area": "a", "primary_change_advantage": 0.1},
        {"area": "", "primary_change_advantage": 0.2},
        {"area": None, "primary_change_advantage": 0.3},
        {"primary_change_advantage": 0.4},
    ]

    ci = clustered_bootstrap_ci(rows, "primary_change_advantage", cluster_key="area", n_boot=50, seed=1)

    assert ci["n_rows"] == 1
    assert ci["n_clusters"] == 1
    assert ci["mean"] == pytest.approx(0.1)
    assert ci["ci_low"] is None
    assert ci["ci_high"] is None


def test_summarize_by_tier_and_stratum_drops_missing_grouping_fields():
    rows = [
        {"tier": "tier1", "stratum": "Wetland", "primary_change_advantage": 0.2},
        {"tier": "", "stratum": "Forest", "primary_change_advantage": 0.1},
        {"tier": None, "stratum": "Mixed", "primary_change_advantage": -0.1},
        {"tier": "tier2", "stratum": "", "primary_change_advantage": 0.3},
        {"tier": "tier2", "stratum": None, "primary_change_advantage": 0.4},
    ]

    summary = summarize_by_tier_and_stratum(rows, "primary_change_advantage")

    assert set(summary["by_tier"]) == {"tier1", "tier2"}
    assert set(summary["by_stratum"]) == {"Wetland", "Forest", "Mixed"}
    assert "" not in summary["by_tier"]
    assert "None" not in summary["by_tier"]
    assert "" not in summary["by_stratum"]
    assert "None" not in summary["by_stratum"]
    assert summary["by_tier"]["tier1"]["n"] == 1
    assert summary["by_tier"]["tier2"]["n"] == 2
    assert summary["by_stratum"]["Wetland"]["mean"] == pytest.approx(0.2)


def test_clustered_bootstrap_is_deterministic_for_a_seed():
    rows = [
        {"area": "a", "primary_change_advantage": 0.1},
        {"area": "b", "primary_change_advantage": 0.2},
        {"area": "c", "primary_change_advantage": -0.1},
    ]

    first = clustered_bootstrap_ci(rows, "primary_change_advantage", cluster_key="area", n_boot=100, seed=7)
    second = clustered_bootstrap_ci(rows, "primary_change_advantage", cluster_key="area", n_boot=100, seed=7)

    assert first == second
