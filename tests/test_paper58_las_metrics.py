import numpy as np
import pytest

from scripts.paper58_benchmark.las_metrics import (
    allocation_disagreement,
    figure_of_merit,
    method_metric_row,
    quantity_disagreement,
    transition_accuracy,
)


def test_figure_of_merit_uses_change_intersection_over_union():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    true = np.array([[1, 2], [2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [3, 2]], dtype=np.int32)

    assert figure_of_merit(start, true, pred) == pytest.approx(1 / 3)


def test_figure_of_merit_requires_correct_changed_class():
    start = np.array([[1]], dtype=np.int32)
    true = np.array([[2]], dtype=np.int32)
    pred = np.array([[3]], dtype=np.int32)

    assert figure_of_merit(start, true, pred) == pytest.approx(0.0)


def test_transition_accuracy_scores_changed_pixels_only():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    true = np.array([[1, 2], [2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int32)

    assert transition_accuracy(start, true, pred) == pytest.approx(0.5)


def test_quantity_and_allocation_disagreement_are_separated():
    true = np.array([[1, 1], [2, 2]], dtype=np.int32)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int32)

    assert quantity_disagreement(true, pred) == pytest.approx(0.25)
    assert allocation_disagreement(true, pred) == pytest.approx(0.0)


def test_method_metric_row_reports_expected_fields():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    true = np.array([[1, 2], [2, 3]], dtype=np.int32)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int32)

    row = method_metric_row("paper58_las", "external", "tier1", "Wetland", start, true, pred)

    assert row["method"] == "paper58_las"
    assert row["area"] == "external"
    assert row["fom"] == pytest.approx(0.5)
    assert row["transition_accuracy"] == pytest.approx(0.5)
    assert row["true_change_pixels"] == 2


def test_method_metric_row_rejects_shape_mismatch():
    start = np.array([[1, 1]], dtype=np.int32)
    true = np.array([[1, 2]], dtype=np.int32)
    pred = np.array([[1], [2]], dtype=np.int32)

    with pytest.raises(ValueError, match="shape mismatch"):
        method_metric_row("paper58_las", "external", "tier1", "Wetland", start, true, pred)
