import numpy as np
import pytest

from scripts.paper58_benchmark.baselines import (
    fit_linear_embedding_delta,
    label_only_transition_prior,
    leave_one_region_temporal_prior,
    persistence_prediction,
    spatial_shuffle_prediction,
)


def test_persistence_prediction_returns_start_map_copy():
    start = np.array([[1, 2], [3, 4]], dtype=np.int32)
    pred = persistence_prediction(start)

    assert np.array_equal(pred, start)
    pred[0, 0] = 9
    assert start[0, 0] == 1


def test_spatial_shuffle_preserves_histogram_and_shape():
    pred = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int32)
    shuffled = spatial_shuffle_prediction(pred, seed=7)

    assert shuffled.shape == pred.shape
    assert sorted(shuffled.ravel().tolist()) == sorted(pred.ravel().tolist())
    assert not np.array_equal(shuffled, pred)


def test_label_only_transition_prior_uses_leave_out_distribution():
    target_start = np.ones((2, 4), dtype=np.int32)
    train_start = np.ones((2, 4), dtype=np.int32)
    train_end = np.array([[1, 2, 1, 2], [1, 2, 1, 2]], dtype=np.int32)

    pred = label_only_transition_prior(target_start, [(train_start, train_end)], seed=7)

    assert pred.shape == target_start.shape
    assert np.count_nonzero(pred == 1) == 4
    assert np.count_nonzero(pred == 2) == 4


def test_leave_one_region_temporal_prior_uses_other_regions_change_rate():
    target_start = np.full((2, 5), 1, dtype=np.int32)
    training_rows = [
        {
            "area": "source_a",
            "start": np.full((2, 5), 1, dtype=np.int32),
            "end": np.array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1]], dtype=np.int32),
        }
    ]

    pred = leave_one_region_temporal_prior("target", target_start, training_rows, seed=11)

    assert pred.shape == target_start.shape
    assert np.count_nonzero(pred != target_start) == 4


def test_linear_embedding_delta_predicts_residual_shape():
    train_start = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    train_end = train_start + np.array([[[0.1, 0.0], [0.0, -0.1]]], dtype=np.float32)
    test_start = np.array([[[1.0, 0.0]]], dtype=np.float32)

    pred = fit_linear_embedding_delta(train_start, train_end, test_start, alpha=1e-3)

    assert pred.shape == test_start.shape
    assert np.isfinite(pred).all()
    assert pred[0, 0, 0] == pytest.approx(1.1, abs=0.05)
