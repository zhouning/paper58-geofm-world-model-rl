import numpy as np
import pytest

from scripts.paper58_benchmark.las_suitability import (
    build_transition_suitability,
    class_values_from_maps,
    one_hot_probability_cube,
    transition_prior_from_pairs,
)


def test_class_values_from_maps_returns_sorted_unique_ints():
    a = np.array([[3, 1], [2, 1]], dtype=np.int32)
    b = np.array([[5, 3], [2, 5]], dtype=np.int32)

    assert class_values_from_maps(a, b) == [1, 2, 3, 5]


def test_one_hot_probability_cube_uses_confidence_and_floor():
    labels = np.array([[1, 2]], dtype=np.int32)
    cube = one_hot_probability_cube(labels, [1, 2, 3], confidence=0.9, floor=0.05)

    assert cube.shape == (1, 2, 3)
    assert cube[0, 0, 0] == 0.9
    assert cube[0, 0, 1] == 0.05
    assert cube[0, 0, 2] == 0.05
    assert cube[0, 1, 1] == 0.9


def test_transition_prior_from_pairs_normalizes_by_start_class():
    start = np.array([[1, 1, 2, 2]], dtype=np.int32)
    end = np.array([[1, 2, 2, 3]], dtype=np.int32)

    prior = transition_prior_from_pairs([(start, end)], class_values=[1, 2, 3])

    assert prior[(1, 1)] == 0.5
    assert prior[(1, 2)] == 0.5
    assert prior[(2, 2)] == 0.5
    assert prior[(2, 3)] == 0.5


def test_transition_prior_from_pairs_ignores_oov_classes_for_normalization():
    start = np.array([[1, 1, 1, 9]], dtype=np.int32)
    end = np.array([[2, 2, 9, 2]], dtype=np.int32)

    prior = transition_prior_from_pairs([(start, end)], class_values=[1, 2])

    assert prior[(1, 2)] == 1.0
    assert sum(prior.get((1, to_cls), 0.0) for to_cls in [1, 2]) == 1.0
    assert (9, 2) not in prior


def test_build_transition_suitability_respects_allowed_transitions():
    start = np.array([[1, 2]], dtype=np.int32)
    forecast_probs = np.array([[[0.1, 0.8], [0.7, 0.2]]], dtype=np.float32)
    start_probs = np.array([[[0.8, 0.1], [0.1, 0.8]]], dtype=np.float32)

    suitability = build_transition_suitability(
        start,
        class_values=[1, 2],
        forecast_probs=forecast_probs,
        start_probs=start_probs,
        allowed_transitions={(1, 2)},
    )

    assert suitability.shape == (1, 2, 2)
    assert suitability[0, 0, 1] > 0.0
    assert suitability[0, 0, 0] == 0.0
    assert suitability[0, 1, 0] == 0.0


def test_build_transition_suitability_rejects_2d_embeddings():
    start = np.array([[1, 2]], dtype=np.int32)
    forecast_probs = np.array([[[0.1, 0.8], [0.7, 0.2]]], dtype=np.float32)
    embedding_start = np.array([[0.0, 0.0]], dtype=np.float32)
    embedding_forecast = np.array([[0.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match=r"embeddings must be 3D feature grids"):
        build_transition_suitability(
            start,
            class_values=[1, 2],
            forecast_probs=forecast_probs,
            embedding_start=embedding_start,
            embedding_forecast=embedding_forecast,
        )


def test_build_transition_suitability_rejects_partial_embedding_inputs():
    start = np.array([[1, 2]], dtype=np.int32)
    forecast_probs = np.array([[[0.1, 0.8], [0.7, 0.2]]], dtype=np.float32)
    embedding_start = np.array([[[0.0], [1.0]]], dtype=np.float32)

    with pytest.raises(ValueError, match="embedding_start and embedding_forecast must be provided together"):
        build_transition_suitability(
            start,
            class_values=[1, 2],
            forecast_probs=forecast_probs,
            embedding_start=embedding_start,
        )


def test_build_transition_suitability_uses_3d_embedding_change_pressure():
    start = np.array([[1]], dtype=np.int32)
    forecast_probs = np.array([[[0.0, 0.0]]], dtype=np.float32)
    embedding_start = np.array([[[0.0, 0.0]]], dtype=np.float32)
    embedding_forecast = np.array([[[3.0, 4.0]]], dtype=np.float32)

    suitability = build_transition_suitability(
        start,
        class_values=[1, 2],
        forecast_probs=forecast_probs,
        embedding_start=embedding_start,
        embedding_forecast=embedding_forecast,
        allowed_transitions={(1, 2)},
        change_pressure_weight=0.25,
    )

    assert suitability[0, 0, 1] == 0.25
