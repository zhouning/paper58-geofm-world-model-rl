from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, median

import numpy as np


def _finite_values(rows: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _summary_from_values(values: list[float]) -> dict:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "n_positive": 0,
            "n_negative": 0,
        }
    return {
        "n": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "n_positive": int(sum(value > 0 for value in values)),
        "n_negative": int(sum(value < 0 for value in values)),
    }


def clustered_bootstrap_ci(
    rows: list[dict],
    value_key: str,
    cluster_key: str = "area",
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    clean_rows = [row for row in rows if isinstance(row.get(value_key), (int, float)) and math.isfinite(float(row[value_key]))]
    if not clean_rows:
        return {
            "n_rows": 0,
            "n_clusters": 0,
            "mean": None,
            "median": None,
            "ci_low": None,
            "ci_high": None,
        }

    clusters: dict[str, list[float]] = defaultdict(list)
    for row in clean_rows:
        clusters[str(row.get(cluster_key, ""))].append(float(row[value_key]))

    cluster_names = sorted(clusters)
    values = [float(row[value_key]) for row in clean_rows]
    rng = np.random.default_rng(seed)
    boot_means: list[float] = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(cluster_names, size=len(cluster_names), replace=True)
        sampled_values: list[float] = []
        for cluster in sampled_clusters:
            sampled_values.extend(clusters[cluster])
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
    clean = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    n_positive = sum(value > 0 for value in clean)
    n_negative = sum(value < 0 for value in clean)
    n_tie = sum(value == 0 for value in clean)
    n_effective = n_positive + n_negative

    if n_effective == 0:
        p_value = 1.0
    else:
        k = min(n_positive, n_negative)
        tail = sum(math.comb(n_effective, i) for i in range(k + 1)) / (2**n_effective)
        p_value = min(1.0, 2.0 * tail)

    return {
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "n_tie": int(n_tie),
        "n_effective": int(n_effective),
        "two_sided_p": float(p_value),
    }


def summarize_by_tier_and_stratum(rows: list[dict], value_key: str) -> dict:
    by_tier: dict[str, list[dict]] = defaultdict(list)
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_tier[str(row.get("tier", "unknown"))].append(row)
        by_stratum[str(row.get("stratum", "Unknown"))].append(row)

    return {
        "by_tier": {key: _summary_from_values(_finite_values(group, value_key)) for key, group in sorted(by_tier.items())},
        "by_stratum": {key: _summary_from_values(_finite_values(group, value_key)) for key, group in sorted(by_stratum.items())},
    }


def gate_report(rows: list[dict], n_boot: int = 5000, seed: int = 42) -> dict:
    tier1 = [row for row in rows if row.get("tier") == "tier1"]

    primary = clustered_bootstrap_ci(tier1, "primary_change_advantage", n_boot=n_boot, seed=seed)
    spatial = clustered_bootstrap_ci(tier1, "spatial_change_advantage", n_boot=n_boot, seed=seed + 1)
    embedding = clustered_bootstrap_ci(tier1, "embedding_advantage", n_boot=n_boot, seed=seed + 2)

    tier1_primary_summary = summarize_by_tier_and_stratum(tier1, "primary_change_advantage")
    positive_tier1_strata = sum(
        1
        for summary in tier1_primary_summary["by_stratum"].values()
        if summary["mean"] is not None and summary["mean"] > 0
    )

    primary_gate_pass = primary["ci_low"] is not None and primary["ci_low"] > 0
    spatial_gate_pass = spatial["ci_low"] is not None and spatial["ci_low"] > 0
    strata_gate_pass = positive_tier1_strata >= 3

    if not tier1:
        status = "insufficient_tier1"
    elif primary_gate_pass and spatial_gate_pass and strata_gate_pass:
        status = "pass"
    else:
        status = "fail"

    return {
        "status": status,
        "tier1_primary_change": primary,
        "tier1_spatial_change": spatial,
        "tier1_embedding": embedding,
        "positive_tier1_strata": int(positive_tier1_strata),
        "required_positive_tier1_strata": 3,
        "primary_gate_pass": bool(primary_gate_pass),
        "spatial_gate_pass": bool(spatial_gate_pass),
        "strata_gate_pass": bool(strata_gate_pass),
    }
