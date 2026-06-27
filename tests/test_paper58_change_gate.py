import numpy as np


def test_apply_change_gate_keeps_highest_scoring_predicted_changes() -> None:
    from scripts.paper58_benchmark.apply_paper58_change_gate import apply_change_gate

    start = np.array([[1, 1, 1, 2]], dtype=np.int32)
    prediction = np.array([[1, 5, 7, 5]], dtype=np.int32)
    pred_prob = np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    start_prob = np.array([[0.8, 0.2, 0.4, 0.1]], dtype=np.float32)
    target_neighborhood = np.array([[0.0, 0.1, 0.9, 0.2]], dtype=np.float32)
    source_neighborhood = np.array([[1.0, 0.2, 0.1, 0.9]], dtype=np.float32)

    gated, diagnostics = apply_change_gate(
        start,
        prediction,
        pred_prob,
        start_prob,
        keep_fraction=0.5,
        target_neighborhood=target_neighborhood,
        source_neighborhood=source_neighborhood,
        target_neighborhood_weight=0.5,
        source_neighborhood_penalty=0.25,
    )

    assert gated.tolist() == [[1, 5, 7, 2]]
    assert diagnostics["candidate_change_pixels"] == 3
    assert diagnostics["kept_change_pixels"] == 2


def test_apply_change_gate_respects_valid_mask() -> None:
    from scripts.paper58_benchmark.apply_paper58_change_gate import apply_change_gate

    start = np.array([[1, 1, 1]], dtype=np.int32)
    prediction = np.array([[5, 7, 5]], dtype=np.int32)
    pred_prob = np.array([[0.9, 0.9, 0.9]], dtype=np.float32)
    start_prob = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
    valid_mask = np.array([[True, False, True]])

    gated, diagnostics = apply_change_gate(
        start,
        prediction,
        pred_prob,
        start_prob,
        keep_fraction=1.0,
        valid_mask=valid_mask,
    )

    assert gated.tolist() == [[5, 1, 5]]
    assert diagnostics["candidate_change_pixels"] == 2


def test_estimate_transition_reliability_accepts_cross_shape_training_cases() -> None:
    from scripts.paper58_benchmark.apply_paper58_change_gate import estimate_transition_reliability

    train_start = np.array([[1, 1, 2]], dtype=np.int32)
    train_end = np.array([[1, 5, 2]], dtype=np.int32)
    train_prediction = np.array([[1, 5, 5]], dtype=np.int32)
    other_start = np.array([[2], [2]], dtype=np.int32)
    other_end = np.array([[5], [2]], dtype=np.int32)
    other_prediction = np.array([[5], [5]], dtype=np.int32)

    reliability, diagnostics = estimate_transition_reliability(
        [
            (train_start, train_end, train_prediction),
            (other_start, other_end, other_prediction),
        ],
        alpha_exact=1.0,
        smoothing=0.0,
    )

    assert reliability[(1, 5)] == 1.0
    assert reliability[(2, 5)] == 1 / 3
    assert diagnostics["training_candidate_change_pixels"] == 4


def test_transition_reliability_gate_keeps_more_reliable_transition_groups() -> None:
    from scripts.paper58_benchmark.apply_paper58_change_gate import apply_transition_reliability_gate

    start = np.array([[1, 1, 2, 2]], dtype=np.int32)
    prediction = np.array([[5, 5, 7, 7]], dtype=np.int32)
    score = np.array([[0.5, 0.4, 0.9, 0.8]], dtype=np.float32)

    gated, diagnostics = apply_transition_reliability_gate(
        start,
        prediction,
        score,
        reliability_by_transition={(1, 5): 0.8, (2, 7): 0.2},
        global_reliability=0.5,
        base_keep_fraction=0.5,
        reliability_slope=0.5,
        min_keep_fraction=0.0,
        max_keep_fraction=1.0,
    )

    assert gated.tolist() == [[5, 5, 2, 2]]
    assert diagnostics["candidate_change_pixels"] == 4
    assert diagnostics["kept_change_pixels"] == 2
    assert diagnostics["transition_group_count"] == 2


def test_spatial_support_reversion_gate_reverts_low_support_low_score_changes() -> None:
    from scripts.paper58_benchmark.apply_paper58_change_gate import apply_spatial_support_reversion_gate

    start = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 5],
        ],
        dtype=np.int32,
    )
    prediction = np.array(
        [
            [5, 1, 1],
            [1, 5, 1],
            [1, 1, 5],
        ],
        dtype=np.int32,
    )
    score = np.array(
        [
            [0.10, 0.90, 0.90],
            [0.90, 0.10, 0.90],
            [0.90, 0.90, 0.90],
        ],
        dtype=np.float32,
    )

    gated, diagnostics = apply_spatial_support_reversion_gate(
        start,
        prediction,
        score,
        target_neighborhood_threshold=0.25,
        score_quantile=0.75,
        max_revert_fraction=1.0,
    )

    assert gated.tolist() == [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 5],
    ]
    assert diagnostics["candidate_change_pixels"] == 2
    assert diagnostics["eligible_revert_pixels"] == 2
    assert diagnostics["reverted_change_pixels"] == 2


def test_spatial_support_reversion_gate_respects_valid_mask_and_revert_cap() -> None:
    from scripts.paper58_benchmark.apply_paper58_change_gate import apply_spatial_support_reversion_gate

    start = np.array([[1, 1, 1, 1]], dtype=np.int32)
    prediction = np.array([[5, 5, 5, 5]], dtype=np.int32)
    score = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    valid_mask = np.array([[True, True, True, False]])
    target_neighborhood = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    gated, diagnostics = apply_spatial_support_reversion_gate(
        start,
        prediction,
        score,
        valid_mask=valid_mask,
        target_neighborhood=target_neighborhood,
        target_neighborhood_threshold=0.25,
        score_quantile=1.0,
        max_revert_fraction=0.34,
    )

    assert gated.tolist() == [[1, 5, 5, 5]]
    assert diagnostics["candidate_change_pixels"] == 3
    assert diagnostics["eligible_revert_pixels"] == 3
    assert diagnostics["reverted_change_pixels"] == 1


def test_transition_exactness_reversion_gate_reverts_low_reliability_low_score_group() -> None:
    from scripts.paper58_benchmark.apply_paper58_change_gate import apply_transition_exactness_reversion_gate

    start = np.array([[1, 1, 2, 2]], dtype=np.int32)
    prediction = np.array([[5, 5, 7, 7]], dtype=np.int32)
    score = np.array([[0.1, 0.9, 0.2, 0.8]], dtype=np.float32)

    gated, diagnostics = apply_transition_exactness_reversion_gate(
        start,
        prediction,
        score,
        exactness_by_transition={(1, 5): 0.1, (2, 7): 0.5},
        global_exactness=0.5,
        max_revert_fraction=0.5,
        min_group_size=1,
    )

    assert gated.tolist() == [[1, 5, 7, 7]]
    assert diagnostics["candidate_change_pixels"] == 4
    assert diagnostics["reverted_change_pixels"] == 1
    assert diagnostics["transition_group_count"] == 2
