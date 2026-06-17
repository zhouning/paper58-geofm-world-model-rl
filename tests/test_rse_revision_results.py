import json
from pathlib import Path

import pytest

from scripts.rse_revision.build_revision_results import (
    bootstrap_ci,
    load_encoder_ablation_rows,
    paired_sign_test,
)


def test_bootstrap_ci_is_deterministic_and_contains_mean():
    values = [0.01, 0.02, 0.03, 0.04]

    result = bootstrap_ci(values, n_boot=200, seed=7)

    assert result["mean"] == pytest.approx(0.025)
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]
    assert result["n"] == 4


def test_paired_sign_test_counts_positive_negative_and_ties():
    values = [0.1, 0.2, -0.1, 0.0, 0.3]

    result = paired_sign_test(values)

    assert result["n_positive"] == 3
    assert result["n_negative"] == 1
    assert result["n_tie"] == 1
    assert 0.0 <= result["two_sided_p"] <= 1.0


def test_load_encoder_ablation_rows_skips_zero_placeholder_areas(tmp_path: Path):
    payload = {
        "alphaearth": {
            "areas": {
                "valid_area": {
                    "persistence_mean": 0.95,
                    "ldn_mean": 0.97,
                    "advantage": 0.02,
                },
                "placeholder_area": {
                    "persistence_mean": 0.0,
                    "ldn_mean": 0.0,
                    "advantage": 0.0,
                },
            }
        }
    }
    path = tmp_path / "paper8_ablation_encoder.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    rows = load_encoder_ablation_rows(path, encoder="alphaearth")

    assert rows == [
        {
            "area": "valid_area",
            "persistence": 0.95,
            "model": 0.97,
            "advantage": 0.02,
            "change_pixel_advantage": None,
        }
    ]
