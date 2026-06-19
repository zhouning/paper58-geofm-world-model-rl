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
