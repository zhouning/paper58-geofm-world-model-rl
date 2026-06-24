import numpy as np
import pytest

from scripts.paper58_benchmark.las_demand import (
    DemandValidationError,
    build_editable_mask,
    derive_observed_demand,
    remaining_editable_demand,
    validate_total_demand,
)


def test_derive_observed_demand_counts_classes():
    end = np.array([[1, 2, 2], [3, 3, 3]], dtype=np.int32)

    demand = derive_observed_demand(end)

    assert demand == {1: 1, 2: 2, 3: 3}


def test_validate_total_demand_rejects_wrong_total():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)

    with pytest.raises(DemandValidationError, match="demand total 3 does not match raster cells 4"):
        validate_total_demand(start, {1: 1, 2: 2})


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
