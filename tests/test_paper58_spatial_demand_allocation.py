import numpy as np


class _Case:
    def __init__(self, area: str) -> None:
        self.area = area


def test_fit_change_fraction_regression_predicts_and_clamps() -> None:
    from scripts.paper58_benchmark.apply_paper58_spatial_demand_allocation import (
        fit_change_fraction_regression,
        predict_change_fraction,
    )

    features = np.array([[0.2, 0.0], [0.4, 0.0], [0.8, 0.0]], dtype=np.float32)
    targets = np.array([0.1, 0.2, 0.4], dtype=np.float32)

    model = fit_change_fraction_regression(features, targets, ridge=0.0)

    assert np.isclose(predict_change_fraction(model, np.array([0.6, 0.0], dtype=np.float32)), 0.3)
    assert predict_change_fraction(model, np.array([4.0, 0.0], dtype=np.float32), max_fraction=0.75) == 0.75
    assert predict_change_fraction(model, np.array([-1.0, 0.0], dtype=np.float32), min_fraction=0.05) == 0.05


def test_fit_change_ratio_demand_model_predicts_from_candidate_change_fraction() -> None:
    from scripts.paper58_benchmark.apply_paper58_spatial_demand_allocation import (
        fit_change_ratio_demand_model,
        predict_ratio_change_fraction,
    )

    predicted_change = np.array([0.10, 0.20, 0.40], dtype=np.float32)
    target_change = np.array([0.05, 0.10, 0.08], dtype=np.float32)

    model = fit_change_ratio_demand_model(predicted_change, target_change, quantile=0.5, multiplier=1.5)

    assert np.isclose(model["base_ratio"], 0.5)
    assert np.isclose(model["effective_ratio"], 0.75)
    assert np.isclose(predict_ratio_change_fraction(model, 0.20), 0.15)
    assert predict_ratio_change_fraction(model, 0.20, max_fraction=0.12) == 0.12
    assert predict_ratio_change_fraction(model, 0.001, min_fraction=0.05) == 0.05


def test_demand_calibrated_spatial_gate_keeps_highest_ranked_changes_to_budget() -> None:
    from scripts.paper58_benchmark.apply_paper58_spatial_demand_allocation import (
        apply_demand_calibrated_spatial_gate,
    )

    start = np.array([[1, 1, 1, 1]], dtype=np.int32)
    prediction = np.array([[5, 5, 5, 1]], dtype=np.int32)
    score = np.array([[0.10, 0.90, 0.50, 0.0]], dtype=np.float32)

    gated, diagnostics = apply_demand_calibrated_spatial_gate(
        start,
        prediction,
        score,
        target_change_fraction=0.5,
    )

    assert gated.tolist() == [[1, 5, 5, 1]]
    assert diagnostics["candidate_change_pixels"] == 3
    assert diagnostics["target_change_pixels"] == 2
    assert diagnostics["kept_change_pixels"] == 2


def test_change_demand_features_include_predicted_change_and_class5_fraction() -> None:
    from scripts.paper58_benchmark.apply_paper58_spatial_demand_allocation import change_demand_features

    start = np.array([[5, 5, 1, 1]], dtype=np.int32)
    prediction = np.array([[5, 1, 5, 1]], dtype=np.int32)

    features = change_demand_features(start, prediction, selector_class=5)

    assert np.allclose(features[:2], np.array([0.5, 0.5], dtype=np.float32))


def test_filter_calibration_cases_by_area_uses_explicit_holdout_whitelist() -> None:
    from scripts.paper58_benchmark.apply_paper58_spatial_demand_allocation import filter_calibration_cases_by_area

    cases = [_Case("train_area"), _Case("calibration_area"), _Case("extra_area")]

    selected, skipped = filter_calibration_cases_by_area(cases, ["calibration_area"])

    assert [case.area for case in selected] == ["calibration_area"]
    assert skipped == [
        {"area": "train_area", "reason": "not_in_calibration_area_whitelist"},
        {"area": "extra_area", "reason": "not_in_calibration_area_whitelist"},
    ]
