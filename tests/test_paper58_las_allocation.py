import time

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


def test_allocate_preserves_map_when_target_demand_already_satisfied():
    start = np.array([[1, 2]], dtype=np.int32)
    class_values = [1, 2]
    suitability = np.zeros((1, 2, 2), dtype=np.float32)
    suitability[0, 0, 1] = 0.99
    suitability[0, 1, 0] = 0.98
    target_demand = {1: 1, 2: 1}

    result = allocate_demand_constrained(start, suitability, class_values, target_demand)

    assert result.simulated_map.tolist() == [[1, 2]]
    assert result.selected_transitions == []


def test_allocate_can_add_balanced_swaps_to_reach_change_budget():
    start = np.array([[1, 2]], dtype=np.int32)
    class_values = [1, 2]
    suitability = np.zeros((1, 2, 2), dtype=np.float32)
    suitability[0, 0, 1] = 0.99
    suitability[0, 1, 0] = 0.98
    target_demand = {1: 1, 2: 1}

    result = allocate_demand_constrained(
        start,
        suitability,
        class_values,
        target_demand,
        target_change_pixels=2,
    )

    assert result.simulated_map.tolist() == [[2, 1]]
    assert result.achieved_demand == target_demand
    assert len(result.selected_transitions) == 2


def test_allocate_can_prune_low_margin_balanced_swaps():
    start = np.array([[1, 2]], dtype=np.int32)
    class_values = [1, 2]
    suitability = np.zeros((1, 2, 2), dtype=np.float32)
    suitability[0, 0, 1] = 0.04
    suitability[0, 1, 0] = 0.03
    target_demand = {1: 1, 2: 1}

    result = allocate_demand_constrained(
        start,
        suitability,
        class_values,
        target_demand,
        target_change_pixels=2,
        balanced_swap_min_margin=0.1,
    )

    assert result.simulated_map.tolist() == [[1, 2]]
    assert result.achieved_demand == target_demand
    assert result.selected_transitions == []


def test_allocate_keeps_high_margin_balanced_swaps():
    start = np.array([[1, 2]], dtype=np.int32)
    class_values = [1, 2]
    suitability = np.zeros((1, 2, 2), dtype=np.float32)
    suitability[0, 0, 1] = 0.09
    suitability[0, 1, 0] = 0.08
    target_demand = {1: 1, 2: 1}

    result = allocate_demand_constrained(
        start,
        suitability,
        class_values,
        target_demand,
        target_change_pixels=2,
        balanced_swap_min_margin=0.1,
    )

    assert result.simulated_map.tolist() == [[2, 1]]
    assert result.achieved_demand == target_demand
    assert len(result.selected_transitions) == 2


def test_allocate_balanced_swaps_respect_forbidden_transitions():
    start = np.array([[1, 2]], dtype=np.int32)
    class_values = [1, 2]
    suitability = np.zeros((1, 2, 2), dtype=np.float32)
    suitability[0, 0, 1] = 0.99
    suitability[0, 1, 0] = 0.98
    target_demand = {1: 1, 2: 1}

    result = allocate_demand_constrained(
        start,
        suitability,
        class_values,
        target_demand,
        allowed_transitions={(1, 2)},
        target_change_pixels=2,
    )

    assert result.simulated_map.tolist() == [[1, 2]]
    assert result.selected_transitions == []


def test_allocate_uses_minimum_required_changes_for_net_demand():
    start = np.array([[1, 1, 2]], dtype=np.int32)
    class_values = [1, 2]
    suitability = np.zeros((1, 3, 2), dtype=np.float32)
    suitability[0, 0, 1] = 0.9
    suitability[0, 1, 1] = 0.8
    target_demand = {1: 1, 2: 2}

    result = allocate_demand_constrained(start, suitability, class_values, target_demand)

    assert np.count_nonzero(result.simulated_map != start) == 1
    assert len(result.selected_transitions) == 1
    assert result.achieved_demand == target_demand


def test_allocate_neighborhood_weight_prefers_target_class_edges():
    start = np.array(
        [
            [1, 1, 2],
            [1, 1, 2],
            [1, 1, 2],
        ],
        dtype=np.int32,
    )
    class_values = [1, 2]
    suitability = np.zeros((3, 3, 2), dtype=np.float32)
    suitability[1, 0, 1] = 0.9
    suitability[1, 1, 1] = 0.1
    target_demand = {1: 5, 2: 4}

    result = allocate_demand_constrained(
        start,
        suitability,
        class_values,
        target_demand,
        neighborhood_weight=3.0,
    )

    assert result.simulated_map[1, 0] == 1
    changed = np.argwhere(result.simulated_map != start)
    assert changed.shape == (1, 2)
    assert tuple(changed[0]) in {(1, 1), (2, 1)}
    assert result.achieved_demand == target_demand


def test_allocate_latent_neighborhood_weight_prefers_semantic_edges():
    start = np.array([[1, 2, 1], [1, 2, 1]], dtype=np.int32)
    class_values = [1, 2]
    suitability = np.zeros((2, 3, 2), dtype=np.float32)
    embeddings = np.zeros((2, 3, 2), dtype=np.float32)
    embeddings[:, 0, :] = np.array([1.0, 0.0], dtype=np.float32)
    embeddings[:, 1, :] = np.array([1.0, 0.0], dtype=np.float32)
    embeddings[:, 2, :] = np.array([0.0, 1.0], dtype=np.float32)
    target_demand = {1: 3, 2: 3}

    result = allocate_demand_constrained(
        start,
        suitability,
        class_values,
        target_demand,
        embedding_grid=embeddings,
        latent_neighborhood_weight=2.0,
    )

    changed = np.argwhere(result.simulated_map != start)
    assert changed.shape == (1, 2)
    assert int(changed[0, 1]) == 0
    assert result.achieved_demand == target_demand


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


def test_allocate_respects_immutable_classes():
    start = np.array([[1, 2], [2, 3]], dtype=np.int32)
    class_values = [1, 2, 3]
    suitability = np.ones((2, 2, 3), dtype=np.float32)
    target_demand = {1: 0, 2: 3, 3: 1}

    result = allocate_demand_constrained(
        start,
        suitability,
        class_values,
        target_demand,
        immutable_classes={3},
    )

    assert result.simulated_map[1, 1] == 3
    assert result.achieved_demand == target_demand
    assert result.constraint_violations == []


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


def test_allocate_multiclass_medium_grid_avoids_per_candidate_full_scans():
    side = 150
    class_values = [1, 2, 3, 4, 5, 6]
    rng = np.random.default_rng(1)
    start = rng.integers(1, 7, size=(side, side), dtype=np.int32)
    suitability = rng.random((side, side, len(class_values)), dtype=np.float32)
    target_demand = {cls: (side * side) // len(class_values) for cls in class_values}
    target_demand[class_values[-1]] += side * side - sum(target_demand.values())

    started = time.perf_counter()
    result = allocate_demand_constrained(start, suitability, class_values, target_demand)
    elapsed = time.perf_counter() - started

    assert result.constraint_violations == []
    assert elapsed < 1.5
