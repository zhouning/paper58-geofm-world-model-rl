import numpy as np
import pytest

from scripts.paper58_benchmark.las_demand import (
    DemandValidationError,
    build_editable_mask,
    derive_change_budget,
    derive_demand,
    derive_observed_demand,
    minimum_change_budget_from_demand,
    project_transition_prior_demand,
    remaining_editable_demand,
    validate_total_demand,
)


def test_derive_observed_demand_counts_classes():
    end = np.array([[1, 2, 2], [3, 3, 3]], dtype=np.int32)

    demand = derive_observed_demand(end)

    assert demand == {1: 1, 2: 2, 3: 3}


def test_derive_demand_can_use_paper58_prediction_instead_of_observed_end():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[2, 2], [2, 2]], dtype=np.int32)
    prediction = np.array([[1, 1], [1, 2]], dtype=np.int32)

    demand = derive_demand(start, end, prediction, demand_source="paper58_prediction")

    assert demand == {1: 3, 2: 1}


def test_derive_demand_rejects_unknown_source():
    start = np.array([[1]], dtype=np.int32)

    with pytest.raises(DemandValidationError, match="unsupported demand_source"):
        derive_demand(start, start, start, demand_source="unknown")


def test_project_transition_prior_demand_projects_start_counts():
    start = np.array([[1, 1], [1, 2]], dtype=np.int32)
    prior = {
        (1, 1): 0.0,
        (1, 2): 1.0,
        (2, 1): 1.0,
        (2, 2): 0.0,
    }

    demand = project_transition_prior_demand(start, class_values=[1, 2], transition_prior=prior)

    assert demand == {1: 1, 2: 3}


def test_project_transition_prior_demand_uses_persistence_for_missing_prior_rows():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    prior = {
        (1, 2): 1.0,
    }

    demand = project_transition_prior_demand(start, class_values=[1, 2], transition_prior=prior)

    assert demand == {1: 0, 2: 4}


def test_derive_demand_can_use_transition_prior():
    start = np.array([[1, 1], [1, 2]], dtype=np.int32)
    end = np.array([[1, 1], [1, 1]], dtype=np.int32)
    prediction = np.array([[2, 2], [2, 2]], dtype=np.int32)

    demand = derive_demand(
        start,
        end,
        prediction,
        demand_source="transition_prior",
        class_values=[1, 2],
        transition_prior={(1, 2): 1.0, (2, 1): 1.0},
    )

    assert demand == {1: 1, 2: 3}


def test_minimum_change_budget_from_demand_uses_net_class_deficits():
    start = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int32)

    budget = minimum_change_budget_from_demand(start, {1: 1, 2: 4, 3: 1})

    assert budget == 2


def test_derive_change_budget_sources():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 1]], dtype=np.int32)
    prediction = np.array([[2, 2], [2, 1]], dtype=np.int32)
    demand = {1: 1, 2: 3}

    assert derive_change_budget(start, end, prediction, demand, "paper58_prediction") == 3
    assert derive_change_budget(start, end, prediction, demand, "observed_end") == 2
    assert derive_change_budget(start, end, prediction, demand, "demand_delta") == 1
    assert derive_change_budget(start, end, prediction, demand, "none") is None


def test_derive_change_budget_scale_is_bounded_by_demand_delta():
    start = np.array([[1, 1], [1, 2]], dtype=np.int32)
    end = np.array([[1, 2], [2, 2]], dtype=np.int32)
    prediction = np.array([[2, 2], [2, 1]], dtype=np.int32)
    demand = {1: 0, 2: 4}

    assert derive_change_budget(start, end, prediction, demand, "paper58_prediction", 0.25) == 3
    assert derive_change_budget(start, end, prediction, demand, "paper58_prediction", 0.75) == 3
    assert derive_change_budget(start, end, prediction, demand, "paper58_prediction", 1.0) == 4


def test_derive_change_budget_rejects_negative_scale():
    start = np.array([[1]], dtype=np.int32)

    with pytest.raises(DemandValidationError, match="change_budget_scale"):
        derive_change_budget(start, start, start, {1: 1}, "paper58_prediction", -0.1)


def test_validate_total_demand_rejects_wrong_total():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)

    with pytest.raises(DemandValidationError, match="demand total 3 does not match raster cells 4"):
        validate_total_demand(start, {1: 1, 2: 2})


def test_validate_total_demand_rejects_negative_class_demand():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)

    with pytest.raises(DemandValidationError, match="demand for class 2 is negative: -1"):
        validate_total_demand(start, {1: 5, 2: -1})


def test_build_editable_mask_respects_exclusion_and_immutable_classes():
    start = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32)
    exclusion = np.array([[False, False, True], [False, True, False]])

    editable = build_editable_mask(start, exclusion_mask=exclusion, immutable_classes={3})

    expected = np.array([[True, True, False], [True, False, False]])
    assert np.array_equal(editable, expected)


def test_remaining_editable_demand_subtracts_fixed_pixels():
    start = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int32)
    editable = np.array([[True, True, True], [True, False, False]])
    full_demand = {1: 1, 2: 3, 3: 2}

    remaining = remaining_editable_demand(start, full_demand, editable)

    assert remaining == {1: 1, 2: 3, 3: 0}


def test_remaining_editable_demand_fails_when_fixed_pixels_exceed_demand():
    start = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int32)
    editable = np.array([[True, True, True], [True, False, False]])
    full_demand = {1: 3, 2: 2, 3: 1}

    with pytest.raises(DemandValidationError, match="fixed class 3 count 2 exceeds demand 1"):
        remaining_editable_demand(start, full_demand, editable)


def test_remaining_editable_demand_rejects_remaining_total_mismatch():
    start = np.array([[1, 4], [2, 2]], dtype=np.int32)
    editable = np.array([[True, False], [True, True]])
    full_demand = {1: 1, 2: 3}

    with pytest.raises(
        DemandValidationError,
        match="remaining editable demand 4 does not match editable cells 3",
    ):
        remaining_editable_demand(start, full_demand, editable)
