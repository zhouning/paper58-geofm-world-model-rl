import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.paper58_benchmark.select_paper58_allocator_candidate import (
    apply_allocator_selector_rule,
    audit_target_candidate_bank,
    build_consensus_candidate,
    select_allocator_conjunctive_candidate_rule,
    select_allocator_candidate_rule,
)


CASE_FIELDS = [
    "parameter_index",
    "area",
    "start_year",
    "end_year",
    "predicted_target_change_fraction",
    "start_unique_class_fraction",
    "change_f1",
    "fom",
    "transition_accuracy",
    "allocation_disagreement",
]


def _write_case_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CASE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _case_row(
    area: str,
    feature: float,
    change_f1: float,
    fom: float,
    transition_accuracy: float,
    allocation_disagreement: float,
    parameter_index: int = 0,
    start_unique_class_fraction: float = 0.0,
) -> dict[str, object]:
    return {
        "parameter_index": parameter_index,
        "area": area,
        "start_year": 2020,
        "end_year": 2021,
        "predicted_target_change_fraction": feature,
        "start_unique_class_fraction": start_unique_class_fraction,
        "change_f1": change_f1,
        "fom": fom,
        "transition_accuracy": transition_accuracy,
        "allocation_disagreement": allocation_disagreement,
    }


def test_select_allocator_candidate_rule_filters_parameter_index_and_keeps_external_constraints(tmp_path: Path):
    current = tmp_path / "current.csv"
    win37 = tmp_path / "win37.csv"
    _write_case_metrics(
        current,
        [
            _case_row("area_a", 0.20, 0.30, 0.10, 0.20, 0.06),
            _case_row("area_b", 0.30, 0.30, 0.10, 0.20, 0.06),
            _case_row("area_c", 0.40, 0.30, 0.10, 0.20, 0.06),
        ],
    )
    _write_case_metrics(
        win37,
        [
            _case_row("area_a", 0.10, 0.50, 0.20, 0.30, 0.05, parameter_index=21),
            _case_row("area_b", 0.20, 0.10, 0.05, 0.10, 0.09, parameter_index=21),
            _case_row("area_c", 0.30, 0.10, 0.05, 0.10, 0.09, parameter_index=21),
            _case_row("area_a", 0.10, 0.05, 0.01, 0.01, 0.20, parameter_index=22),
            _case_row("area_b", 0.20, 0.05, 0.01, 0.01, 0.20, parameter_index=22),
            _case_row("area_c", 0.30, 0.05, 0.01, 0.01, 0.20, parameter_index=22),
        ],
    )

    result = select_allocator_candidate_rule(
        {"current": current, "win37": win37},
        output_dir=tmp_path / "selector",
        baseline_candidate="current",
        candidate_parameter_indices={"win37": 21},
    )

    assert result["rule"]["alternate_candidate"] == "win37"
    assert result["rule"]["operator"] == "le"
    assert result["rule"]["threshold"] == pytest.approx(0.10)
    assert result["external_summary"]["selected_candidate_counts"] == {"current": 2, "win37": 1}
    assert result["external_summary"]["mean_change_f1"] > result["external_summary"]["baseline_mean_change_f1"]
    assert result["external_summary"]["mean_fom"] > result["external_summary"]["baseline_mean_fom"]
    assert result["external_summary"]["mean_transition_accuracy"] > result["external_summary"][
        "baseline_mean_transition_accuracy"
    ]
    assert result["external_summary"]["mean_allocation_disagreement"] < result["external_summary"][
        "baseline_mean_allocation_disagreement"
    ]
    assert (tmp_path / "selector" / "external_selector_rules.csv").exists()
    assert (tmp_path / "selector" / "selector_manifest.json").exists()


def test_select_allocator_candidate_rule_honors_max_alternate_fraction(tmp_path: Path):
    current = tmp_path / "current.csv"
    alt = tmp_path / "alt.csv"
    _write_case_metrics(
        current,
        [
            _case_row("area_a", 0.10, 0.30, 0.10, 0.20, 0.06),
            _case_row("area_b", 0.20, 0.30, 0.10, 0.20, 0.06),
            _case_row("area_c", 0.30, 0.30, 0.10, 0.20, 0.06),
            _case_row("area_d", 0.40, 0.30, 0.10, 0.20, 0.06),
        ],
    )
    _write_case_metrics(
        alt,
        [
            _case_row("area_a", 0.10, 0.50, 0.20, 0.30, 0.05),
            _case_row("area_b", 0.20, 0.50, 0.20, 0.30, 0.05),
            _case_row("area_c", 0.30, 0.50, 0.20, 0.30, 0.05),
            _case_row("area_d", 0.40, 0.50, 0.20, 0.30, 0.05),
        ],
    )

    result = select_allocator_candidate_rule(
        {"current": current, "alt": alt},
        output_dir=tmp_path / "selector",
        baseline_candidate="current",
        max_alternate_fraction=0.5,
    )

    assert result["rule"]["alternate_candidate"] == "alt"
    assert result["external_summary"]["selected_alternate_count"] == 2
    assert result["external_summary"]["selected_alternate_fraction"] == pytest.approx(0.5)
    assert result["external_summary"]["selected_candidate_counts"] == {"alt": 2, "current": 2}


def test_select_allocator_conjunctive_candidate_rule_requires_all_conditions_and_min_support(tmp_path: Path):
    current = tmp_path / "current.csv"
    alt = tmp_path / "alt.csv"
    _write_case_metrics(
        current,
        [
            _case_row("area_a", 0.80, 0.30, 0.10, 0.20, 0.06, start_unique_class_fraction=0.8),
            _case_row("area_b", 0.70, 0.30, 0.10, 0.20, 0.06, start_unique_class_fraction=0.7),
            _case_row("area_c", 0.10, 0.30, 0.10, 0.20, 0.06, start_unique_class_fraction=0.1),
            _case_row("area_d", 0.90, 0.30, 0.10, 0.20, 0.06, start_unique_class_fraction=0.9),
            _case_row("area_e", 0.05, 0.30, 0.10, 0.20, 0.06, start_unique_class_fraction=0.1),
        ],
    )
    _write_case_metrics(
        alt,
        [
            _case_row("area_a", 0.10, 0.60, 0.25, 0.40, 0.04, start_unique_class_fraction=0.8),
            _case_row("area_b", 0.20, 0.60, 0.25, 0.40, 0.04, start_unique_class_fraction=0.7),
            _case_row("area_c", 0.15, 0.00, 0.00, 0.00, 0.30, start_unique_class_fraction=0.1),
            _case_row("area_d", 0.70, 0.00, 0.00, 0.00, 0.30, start_unique_class_fraction=0.9),
            _case_row("area_e", 0.80, 0.00, 0.00, 0.00, 0.30, start_unique_class_fraction=0.1),
        ],
    )

    result = select_allocator_conjunctive_candidate_rule(
        {"current": current, "alt": alt},
        output_dir=tmp_path / "selector",
        baseline_candidate="current",
        feature_fields=["predicted_target_change_fraction", "start_unique_class_fraction"],
        min_alternate_count=2,
        max_conditions=2,
    )

    assert result["rule"]["alternate_candidate"] == "alt"
    assert len(result["rule"]["conditions"]) == 2
    assert result["external_summary"]["selected_candidate_counts"] == {"alt": 2, "current": 3}
    assert result["external_summary"]["selected_alternate_count"] == 2
    assert result["external_summary"]["mean_change_f1"] > result["external_summary"]["baseline_mean_change_f1"]
    assert result["external_summary"]["mean_fom"] > result["external_summary"]["baseline_mean_fom"]
    assert result["external_summary"]["mean_transition_accuracy"] > result["external_summary"][
        "baseline_mean_transition_accuracy"
    ]
    assert result["external_summary"]["mean_allocation_disagreement"] < result["external_summary"][
        "baseline_mean_allocation_disagreement"
    ]
    assert (tmp_path / "selector" / "external_conjunctive_selector_rules.csv").exists()
    assert (tmp_path / "selector" / "conjunctive_selector_manifest.json").exists()


def test_select_allocator_conjunctive_candidate_rule_can_rank_by_allocation_first(tmp_path: Path):
    current = tmp_path / "current.csv"
    alt = tmp_path / "alt.csv"
    _write_case_metrics(
        current,
        [
            _case_row("area_a", 0.10, 0.30, 0.10, 0.20, 0.06),
            _case_row("area_b", 0.20, 0.30, 0.10, 0.20, 0.06),
            _case_row("area_c", 0.30, 0.30, 0.10, 0.20, 0.06),
            _case_row("area_d", 0.40, 0.30, 0.10, 0.20, 0.06),
        ],
    )
    _write_case_metrics(
        alt,
        [
            _case_row("area_a", 0.10, 0.80, 0.50, 0.50, 0.055),
            _case_row("area_b", 0.20, 0.00, 0.00, 0.00, 0.30),
            _case_row("area_c", 0.30, 0.00, 0.00, 0.00, 0.30),
            _case_row("area_d", 0.40, 0.50, 0.20, 0.30, 0.010),
        ],
    )

    default_result = select_allocator_conjunctive_candidate_rule(
        {"current": current, "alt": alt},
        output_dir=tmp_path / "selector_default",
        baseline_candidate="current",
        min_alternate_count=1,
        max_conditions=1,
    )
    allocation_first_result = select_allocator_conjunctive_candidate_rule(
        {"current": current, "alt": alt},
        output_dir=tmp_path / "selector_allocation",
        baseline_candidate="current",
        min_alternate_count=1,
        max_conditions=1,
        rank_metric_order=["allocation_disagreement", "fom", "change_f1", "transition_accuracy"],
    )

    assert default_result["rule"]["conditions"][0]["operator"] == "le"
    assert default_result["rule"]["conditions"][0]["threshold"] == pytest.approx(0.10)
    assert allocation_first_result["rule"]["conditions"][0]["operator"] == "ge"
    assert allocation_first_result["rule"]["conditions"][0]["threshold"] == pytest.approx(0.40)
    assert allocation_first_result["manifest"]["rank_metric_order"] == [
        "allocation_disagreement",
        "fom",
        "change_f1",
        "transition_accuracy",
    ]


def test_select_allocator_conjunctive_candidate_rule_can_constrain_unlabeled_target_switch_count(tmp_path: Path):
    current = tmp_path / "current.csv"
    alt = tmp_path / "alt.csv"
    _write_case_metrics(
        current,
        [
            _case_row("area_a", 0.10, 0.30, 0.10, 0.20, 0.06),
            _case_row("area_b", 0.20, 0.30, 0.10, 0.20, 0.06),
            _case_row("area_c", 0.30, 0.30, 0.10, 0.20, 0.06),
            _case_row("area_d", 0.40, 0.30, 0.10, 0.20, 0.06),
        ],
    )
    _write_case_metrics(
        alt,
        [
            _case_row("area_a", 0.10, 0.80, 0.50, 0.50, 0.055),
            _case_row("area_b", 0.20, 0.00, 0.00, 0.00, 0.30),
            _case_row("area_c", 0.30, 0.00, 0.00, 0.00, 0.30),
            _case_row("area_d", 0.40, 0.50, 0.20, 0.30, 0.010),
        ],
    )

    labels = tmp_path / "labels"
    current_predictions = tmp_path / "current_predictions" / "predictions"
    alt_predictions = tmp_path / "alt_predictions" / "predictions"
    labels.mkdir()
    current_predictions.mkdir(parents=True)
    alt_predictions.mkdir(parents=True)
    start = np.ones((1, 20), dtype=np.int32)
    end = start.copy()
    current_target = start.copy()
    alt_target = start.copy()
    current_target[0, :5] = 5
    alt_target[0, :8] = 5
    np.save(labels / "target_area_lulc_2020.npy", start)
    np.save(labels / "target_area_lulc_2021.npy", end)
    np.save(current_predictions / "target_area_lulc_pred_2020_2021.npy", current_target)
    np.save(alt_predictions / "target_area_lulc_pred_2020_2021.npy", alt_target)

    result = select_allocator_conjunctive_candidate_rule(
        {"current": current, "alt": alt},
        output_dir=tmp_path / "selector_target_constrained",
        baseline_candidate="current",
        min_alternate_count=1,
        max_conditions=1,
        target_prediction_dirs={
            "current": tmp_path / "current_predictions",
            "alt": tmp_path / "alt_predictions",
        },
        target_labels_dir=labels,
        target_min_alternate_count=1,
        target_max_alternate_count=1,
    )

    assert result["rule"]["conditions"][0]["operator"] == "ge"
    assert result["rule"]["conditions"][0]["threshold"] == pytest.approx(0.40)
    assert result["external_summary"]["target_unlabeled_alternate_count"] == 1
    assert result["manifest"]["target_unlabeled_constraints"] == {
        "target_min_alternate_count": 1,
        "target_max_alternate_count": 1,
    }


def test_apply_allocator_selector_rule_writes_composite_predictions_and_metrics(tmp_path: Path):
    labels = tmp_path / "labels"
    current_predictions = tmp_path / "current" / "predictions"
    alt_predictions = tmp_path / "alt" / "predictions"
    labels.mkdir()
    current_predictions.mkdir(parents=True)
    alt_predictions.mkdir(parents=True)

    start_a = np.array([[1, 1, 1, 1]], dtype=np.int32)
    end_a = np.array([[1, 5, 1, 1]], dtype=np.int32)
    current_a = np.array([[5, 5, 1, 1]], dtype=np.int32)
    alt_a = np.array([[1, 5, 1, 1]], dtype=np.int32)

    start_b = np.array([[1, 1, 1, 1]], dtype=np.int32)
    end_b = np.array([[1, 1, 5, 1]], dtype=np.int32)
    current_b = np.array([[1, 1, 5, 1]], dtype=np.int32)
    alt_b = np.array([[5, 5, 1, 1]], dtype=np.int32)

    for area, start, end, current, alt in [
        ("area_a", start_a, end_a, current_a, alt_a),
        ("area_b", start_b, end_b, current_b, alt_b),
    ]:
        np.save(labels / f"{area}_lulc_2020.npy", start)
        np.save(labels / f"{area}_lulc_2021.npy", end)
        np.save(current_predictions / f"{area}_lulc_pred_2020_2021.npy", current)
        np.save(alt_predictions / f"{area}_lulc_pred_2020_2021.npy", alt)

    result = apply_allocator_selector_rule(
        {
            "baseline_candidate": "current",
            "alternate_candidate": "alt",
            "feature_candidate": "alt",
            "feature_field": "predicted_target_change_fraction",
            "operator": "le",
            "threshold": 0.25,
        },
        {"current": tmp_path / "current", "alt": tmp_path / "alt"},
        labels_dir=labels,
        output_dir=tmp_path / "target_selector",
    )

    selected = {row["area"]: row["selected_candidate"] for row in result["selected_rows"]}
    assert selected == {"area_a": "alt", "area_b": "current"}
    assert result["target_summary"][0]["mean_change_f1"] == pytest.approx(1.0)
    assert result["target_summary"][0]["mean_allocation_disagreement"] == pytest.approx(0.0)
    assert (tmp_path / "target_selector" / "predictions" / "area_a_lulc_pred_2020_2021.npy").exists()
    assert (tmp_path / "target_selector" / "target_selection_by_area.csv").exists()
    assert (tmp_path / "target_selector" / "target_metric_summary_by_method.csv").exists()


def test_apply_allocator_selector_rule_supports_conjunctive_conditions(tmp_path: Path):
    labels = tmp_path / "labels"
    current_predictions = tmp_path / "current" / "predictions"
    alt_predictions = tmp_path / "alt" / "predictions"
    labels.mkdir()
    current_predictions.mkdir(parents=True)
    alt_predictions.mkdir(parents=True)

    start_a = np.array([[1, 2, 1, 1]], dtype=np.int32)
    end_a = np.array([[1, 5, 1, 1]], dtype=np.int32)
    current_a = np.array([[1, 2, 1, 1]], dtype=np.int32)
    alt_a = np.array([[1, 5, 1, 1]], dtype=np.int32)

    start_b = np.array([[1, 1, 1, 1]], dtype=np.int32)
    end_b = np.array([[1, 1, 5, 1]], dtype=np.int32)
    current_b = np.array([[1, 1, 5, 1]], dtype=np.int32)
    alt_b = np.array([[1, 5, 1, 1]], dtype=np.int32)

    for area, start, end, current, alt in [
        ("area_a", start_a, end_a, current_a, alt_a),
        ("area_b", start_b, end_b, current_b, alt_b),
    ]:
        np.save(labels / f"{area}_lulc_2020.npy", start)
        np.save(labels / f"{area}_lulc_2021.npy", end)
        np.save(current_predictions / f"{area}_lulc_pred_2020_2021.npy", current)
        np.save(alt_predictions / f"{area}_lulc_pred_2020_2021.npy", alt)

    result = apply_allocator_selector_rule(
        {
            "baseline_candidate": "current",
            "alternate_candidate": "alt",
            "conditions": [
                {
                    "feature_candidate": "alt",
                    "feature_field": "predicted_target_change_fraction",
                    "operator": "le",
                    "threshold": 0.25,
                },
                {
                    "feature_candidate": "current",
                    "feature_field": "start_unique_class_fraction",
                    "operator": "ge",
                    "threshold": 0.2,
                },
            ],
        },
        {"current": tmp_path / "current", "alt": tmp_path / "alt"},
        labels_dir=labels,
        output_dir=tmp_path / "target_selector",
    )

    selected = {row["area"]: row["selected_candidate"] for row in result["selected_rows"]}
    assert selected == {"area_a": "alt", "area_b": "current"}
    assert result["target_summary"][0]["mean_change_f1"] == pytest.approx(1.0)
    assert result["target_summary"][0]["mean_allocation_disagreement"] == pytest.approx(0.0)


def test_audit_target_candidate_bank_writes_oracle_headroom(tmp_path: Path):
    labels = tmp_path / "labels"
    current_predictions = tmp_path / "current" / "predictions"
    alt_predictions = tmp_path / "alt" / "predictions"
    labels.mkdir()
    current_predictions.mkdir(parents=True)
    alt_predictions.mkdir(parents=True)

    for area, end_index in [("area_a", 1), ("area_b", 2)]:
        start = np.array([[1, 1, 1, 1]], dtype=np.int32)
        end = start.copy()
        end[0, end_index] = 5
        current = start.copy()
        alt = start.copy()
        current[0, 2] = 5
        alt[0, 1] = 5
        np.save(labels / f"{area}_lulc_2020.npy", start)
        np.save(labels / f"{area}_lulc_2021.npy", end)
        np.save(current_predictions / f"{area}_lulc_pred_2020_2021.npy", current)
        np.save(alt_predictions / f"{area}_lulc_pred_2020_2021.npy", alt)

    result = audit_target_candidate_bank(
        {"current": tmp_path / "current", "alt": tmp_path / "alt"},
        labels_dir=labels,
        output_dir=tmp_path / "audit",
    )

    oracle = {row["area"]: row["selected_candidate"] for row in result["oracle_rows"]}
    assert oracle == {"area_a": "alt", "area_b": "current"}
    assert result["oracle_summary"][0]["mean_fom"] == pytest.approx(1.0)
    assert result["oracle_summary"][0]["mean_allocation_disagreement"] == pytest.approx(0.0)
    assert (tmp_path / "audit" / "target_candidate_metrics_by_area.csv").exists()
    assert (tmp_path / "audit" / "target_candidate_metric_summary_by_method.csv").exists()
    assert (tmp_path / "audit" / "target_oracle_by_area.csv").exists()
    assert (tmp_path / "audit" / "target_oracle_summary_by_method.csv").exists()


def test_build_consensus_candidate_requires_anchor_and_support(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    start = np.array([[1, 1, 1, 1]], dtype=np.int32)
    np.save(labels / "area_a_lulc_2020.npy", start)

    predictions = {
        "anchor": np.array([[5, 5, 5, 1]], dtype=np.int32),
        "support_a": np.array([[5, 1, 5, 1]], dtype=np.int32),
        "support_b": np.array([[5, 1, 1, 1]], dtype=np.int32),
        "support_c": np.array([[1, 5, 1, 1]], dtype=np.int32),
    }
    candidate_dirs = {}
    for name, prediction in predictions.items():
        root = tmp_path / name / "predictions"
        root.mkdir(parents=True)
        np.save(root / "area_a_lulc_pred_2020_2021.npy", prediction)
        candidate_dirs[name] = tmp_path / name

    result = build_consensus_candidate(
        candidate_dirs,
        labels_dir=labels,
        output_dir=tmp_path / "consensus",
        anchor_candidate="anchor",
        min_support=3,
    )

    output = np.load(tmp_path / "consensus" / "predictions" / "area_a_lulc_pred_2020_2021.npy")
    assert output.tolist() == [[5, 1, 1, 1]]
    assert result["manifest"]["n_cases"] == 1
    assert result["case_rows"][0]["selected_change_pixels"] == 1
