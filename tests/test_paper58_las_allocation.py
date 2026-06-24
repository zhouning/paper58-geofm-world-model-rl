import numpy as np

from scripts.paper58_benchmark.las_allocation import allocate_demand_constrained


def test_allocate_demand_constrained_meets_target_demand():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    class_values = [1, 2]
    suitability = np.zeros((2, 2, 2), dtype=np.float32)
    suitability[0, 0, 1] = 0.9
    suitability[0, 1, 1] = 0.8
    target_demand = {1: 2, 2: 2}

    result = allocate_demand_constrained(start, suitability, class_values, target_demand)

    assert result.achieved_demand == target_demand
    assert result.unmet_demand == {1: 0, 2: 0}
    assert result.simulated_map.shape == start.shape


def test_allocate_respects_exclusion_mask():
    start = np.array([[1, 1], [2, 2]], dtype=np.int32)
    class_values = [1, 2]
    suitability = np.zeros((2, 2, 2), dtype=np.float32)
    suitability[:, :, 1] = 1.0
    target_demand = {1: 1, 2: 3}
    exclusion = np.array([[True, False], [False, False]])

    result = allocate_demand_constrained(
        start,
        suitability,
        class_values,
        target_demand,
        exclusion_mask=exclusion,
    )

    assert result.simulated_map[0, 0] == 1
    assert result.achieved_demand == target_demand


def test_allocate_respects_forbidden_transition():
    start = np.array([[1, 2], [2, 2]], dtype=np.int32)
    class_values = [1, 2, 3]
    suitability = np.ones((2, 2, 3), dtype=np.float32)
    target_demand = {1: 1, 2: 1, 3: 2}

    result = allocate_demand_constrained(
        start,
        suitability,
        class_values,
        target_demand,
        allowed_transitions={(2, 3)},
    )

    assert result.simulated_map[0, 0] == 1
    assert np.count_nonzero(result.simulated_map == 3) == 2
    assert result.constraint_violations == []


def test_allocate_preserves_feasible_scarce_destination():
    start = np.array([[1, 2]], dtype=np.int32)
    class_values = [1, 2, 3]
    suitability = np.zeros((1, 2, 3), dtype=np.float32)
    suitability[0, 0, 1] = 0.9
    suitability[0, 0, 2] = 0.8
    target_demand = {1: 0, 2: 1, 3: 1}

    result = allocate_demand_constrained(
        start,
        suitability,
        class_values,
        target_demand,
        allowed_transitions={(1, 2), (1, 3)},
    )

    assert result.simulated_map.tolist() == [[3, 2]]
    assert result.achieved_demand == target_demand
    assert result.unmet_demand == {1: 0, 2: 0, 3: 0}
    assert result.constraint_violations == []


def test_allocate_feasibility_guard_does_not_double_count_pixels():
    start = np.array([[2, 1, 3]], dtype=np.int32)
    class_values = [1, 2, 3]
    suitability = np.zeros((1, 3, 3), dtype=np.float32)
    suitability[0, 0, 2] = 0.9
    target_demand = {1: 1, 2: 1, 3: 1}

    result = allocate_demand_constrained(
        start,
        suitability,
        class_values,
        target_demand,
        allowed_transitions={(2, 3), (1, 2)},
    )

    assert result.simulated_map.tolist() == [[2, 1, 3]]
    assert result.achieved_demand == target_demand
    assert result.unmet_demand == {1: 0, 2: 0, 3: 0}
    assert result.constraint_violations == []


def test_allocate_records_fallback_changed_transitions():
    start = np.array([[1, 2]], dtype=np.int32)
    class_values = [1, 2, 3]
    suitability = np.zeros((1, 2, 3), dtype=np.float32)
    target_demand = {1: 0, 2: 0, 3: 2}

    result = allocate_demand_constrained(
        start,
        suitability,
        class_values,
        target_demand,
        allowed_transitions={(1, 3)},
    )

    assert result.simulated_map[0, 0] == 3
    assert {
        "row": 0,
        "col": 0,
        "from_class": 1,
        "to_class": 3,
    }.items() <= result.selected_transitions[0].items()


def test_allocate_large_feasible_demand_avoids_recursive_matching_limit():
    n_pixels = 1200
    start = np.ones((1, n_pixels), dtype=np.int32)
    class_values = [1]
    suitability = np.ones((1, n_pixels, 1), dtype=np.float32)

    result = allocate_demand_constrained(start, suitability, class_values, {1: n_pixels})

    assert result.achieved_demand == {1: n_pixels}
    assert result.constraint_violations == []
