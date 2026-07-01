import numpy as np


class _Case:
    def __init__(self, area: str) -> None:
        self.area = area


class _CalibrationCase:
    def __init__(self, area: str, start_map: np.ndarray, prediction_map: np.ndarray, end_map: np.ndarray) -> None:
        self.area = area
        self.start_map = start_map
        self.prediction_map = prediction_map
        self.end_map = end_map


class _ChangeGateCase:
    def __init__(self, area: str, start_map: np.ndarray, score_map: np.ndarray) -> None:
        self.area = area
        self.start_year = 2020
        self.end_year = 2021
        self.start_map = start_map
        self.score_map = score_map


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


def test_fit_transition_reliability_weights_smooths_precision() -> None:
    from scripts.paper58_benchmark.apply_paper58_spatial_demand_allocation import (
        fit_transition_reliability_weights,
        transition_reliability_score,
    )

    start = np.array([[1, 1, 1, 2]], dtype=np.int32)
    prediction = np.array([[5, 5, 2, 5]], dtype=np.int32)
    end = np.array([[5, 1, 2, 2]], dtype=np.int32)

    model = fit_transition_reliability_weights(
        [(start, prediction, end)],
        smoothing=1.0,
        min_weight=0.2,
    )

    assert model["pair_stats"][(1, 5)]["predicted"] == 2
    assert model["pair_stats"][(1, 5)]["hits"] == 1
    assert 0.2 < transition_reliability_score(model, 1, 5) < 1.0
    assert transition_reliability_score(model, 9, 9) == model["default_weight"]


def test_adaptive_ratio_cap_limits_high_candidate_fraction() -> None:
    from scripts.paper58_benchmark.apply_paper58_spatial_demand_allocation import (
        fit_adaptive_ratio_cap_model,
        predict_ratio_change_fraction_v2,
    )

    predicted = np.array([0.10, 0.20, 0.80], dtype=np.float32)
    targets = np.array([0.05, 0.10, 0.08], dtype=np.float32)
    ratio_model = {"effective_ratio": 0.75}
    cap_model = fit_adaptive_ratio_cap_model(
        predicted,
        targets,
        high_candidate_threshold=0.5,
        high_candidate_quantile=0.5,
    )

    assert np.isclose(predict_ratio_change_fraction_v2(ratio_model, cap_model, 0.20, max_fraction=0.5), 0.15)
    assert np.isclose(predict_ratio_change_fraction_v2(ratio_model, cap_model, 0.80, max_fraction=0.5), 0.08)


def test_multi_scale_class_aligned_neighborhood_combines_windows() -> None:
    from scripts.paper58_benchmark.apply_paper58_spatial_demand_allocation import (
        multi_scale_class_aligned_neighborhood,
    )

    start = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 5, 5, 5, 1],
            [1, 5, 1, 5, 1],
            [1, 5, 5, 5, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    prediction = start.copy()
    prediction[2, 2] = 5

    target_support, source_support = multi_scale_class_aligned_neighborhood(
        start,
        prediction,
        classes=[1, 5],
        window_sizes=[3, 5],
    )

    assert target_support.shape == start.shape
    assert source_support.shape == start.shape
    assert target_support[2, 2] > 0.0
    assert source_support[2, 2] > 0.0


def test_demand_gate_v2_prefers_reliable_transition_over_raw_score() -> None:
    from scripts.paper58_benchmark.apply_paper58_spatial_demand_allocation import (
        apply_demand_calibrated_spatial_gate,
    )

    start = np.array([[1, 1, 1]], dtype=np.int32)
    prediction = np.array([[5, 2, 1]], dtype=np.int32)
    score = np.array([[0.9, 0.4, 0.0]], dtype=np.float32)
    reliability = {
        "weights": {(1, 5): 0.2, (1, 2): 1.0},
        "default_weight": 0.2,
        "pair_stats": {},
        "smoothing": 1.0,
        "min_weight": 0.2,
    }

    gated, diagnostics = apply_demand_calibrated_spatial_gate(
        start,
        prediction,
        score,
        target_change_fraction=1 / 3,
        transition_reliability_model=reliability,
        transition_weight_strength=1.0,
    )

    assert gated.tolist() == [[1, 2, 1]]
    assert diagnostics["transition_weight_strength"] == 1.0


def test_run_spatial_demand_allocation_writes_v2_manifest_options(tmp_path, monkeypatch) -> None:
    from scripts.paper58_benchmark import apply_paper58_spatial_demand_allocation as module

    start = np.array([[1, 1, 1]], dtype=np.int32)
    calibration_prediction = np.array([[5, 2, 1]], dtype=np.int32)
    calibration_end = np.array([[1, 2, 1]], dtype=np.int32)
    source_prediction_dir = tmp_path / "source"
    source_prediction_dir.mkdir()
    np.save(source_prediction_dir / "target_area_lulc_pred_2020_2021.npy", calibration_prediction)

    monkeypatch.setattr(
        module,
        "discover_calibration_cases",
        lambda _labels, _predictions: (
            [_CalibrationCase("cal_area", start, calibration_prediction, calibration_end)],
            [],
        ),
    )
    monkeypatch.setattr(
        module,
        "load_case_from_change_gate_dir",
        lambda _path: _ChangeGateCase("target_area", start, np.array([[0.9, 0.4, 0.0]], dtype=np.float32)),
    )

    manifest = module.run_spatial_demand_allocation(
        source_prediction_dir=source_prediction_dir,
        change_gate_dirs=[tmp_path / "case"],
        calibration_label_dir=tmp_path / "labels",
        calibration_prediction_dir=tmp_path / "predictions",
        output_dir=tmp_path / "out",
        demand_strategy="ratio",
        min_fraction=0.0,
        max_fraction=1.0,
        enable_transition_reliability=True,
        transition_reliability_strength=1.0,
        neighborhood_window_sizes=[3, 5],
        enable_adaptive_ratio_cap=True,
        high_candidate_threshold=0.5,
        high_candidate_quantile=0.5,
    )

    assert manifest["parameters"]["enable_transition_reliability"] is True
    assert manifest["parameters"]["transition_reliability_strength"] == 1.0
    assert manifest["parameters"]["neighborhood_window_sizes"] == [3, 5]
    assert manifest["parameters"]["enable_adaptive_ratio_cap"] is True
    assert manifest["transition_reliability_model"]["total_predicted"] == 2
    assert manifest["adaptive_ratio_cap_model"]["n_rows"] == 1
    assert (tmp_path / "out" / "predictions" / "target_area_lulc_pred_2020_2021.npy").exists()


def test_run_spatial_demand_allocation_uses_large_region_ratio_multiplier(tmp_path, monkeypatch) -> None:
    from scripts.paper58_benchmark import apply_paper58_spatial_demand_allocation as module

    calibration_start = np.array([[1, 1, 1, 1]], dtype=np.int32)
    calibration_prediction = np.array([[5, 5, 1, 1]], dtype=np.int32)
    calibration_end = np.array([[5, 1, 1, 1]], dtype=np.int32)
    small_prediction = np.array([[5, 5, 1, 1]], dtype=np.int32)
    large_prediction = np.array([[5, 5, 5, 1, 1, 1]], dtype=np.int32)
    source_prediction_dir = tmp_path / "source"
    source_prediction_dir.mkdir()
    np.save(source_prediction_dir / "small_area_lulc_pred_2020_2021.npy", small_prediction)
    np.save(source_prediction_dir / "large_area_lulc_pred_2020_2021.npy", large_prediction)

    change_gate_cases = {
        tmp_path / "small_case": _ChangeGateCase(
            "small_area",
            calibration_start,
            np.array([[0.9, 0.8, 0.0, 0.0]], dtype=np.float32),
        ),
        tmp_path / "large_case": _ChangeGateCase(
            "large_area",
            np.array([[1, 1, 1, 1, 1, 1]], dtype=np.int32),
            np.array([[0.9, 0.8, 0.7, 0.0, 0.0, 0.0]], dtype=np.float32),
        ),
    }

    monkeypatch.setattr(
        module,
        "discover_calibration_cases",
        lambda _labels, _predictions: (
            [_CalibrationCase("cal_area", calibration_start, calibration_prediction, calibration_end)],
            [],
        ),
    )
    monkeypatch.setattr(
        module,
        "load_case_from_change_gate_dir",
        lambda path: change_gate_cases[path],
    )

    manifest = module.run_spatial_demand_allocation(
        source_prediction_dir=source_prediction_dir,
        change_gate_dirs=[tmp_path / "small_case", tmp_path / "large_case"],
        calibration_label_dir=tmp_path / "labels",
        calibration_prediction_dir=tmp_path / "predictions",
        output_dir=tmp_path / "out",
        demand_strategy="ratio",
        ratio_quantile=0.5,
        ratio_multiplier=1.0,
        large_region_valid_pixel_threshold=5,
        large_region_ratio_multiplier=2.0,
        min_fraction=0.0,
        max_fraction=1.0,
    )

    cases = {case["area"]: case for case in manifest["cases"]}
    assert cases["small_area"]["ratio_multiplier_for_case"] == 1.0
    assert cases["small_area"]["large_region_ratio_multiplier_applied"] is False
    assert np.isclose(cases["small_area"]["predicted_target_change_fraction"], 0.25)
    assert cases["small_area"]["gate_diagnostics"]["target_change_pixels"] == 1
    assert cases["large_area"]["ratio_multiplier_for_case"] == 2.0
    assert cases["large_area"]["large_region_ratio_multiplier_applied"] is True
    assert np.isclose(cases["large_area"]["predicted_target_change_fraction"], 0.5)
    assert cases["large_area"]["gate_diagnostics"]["target_change_pixels"] == 3
    assert manifest["parameters"]["large_region_valid_pixel_threshold"] == 5
    assert manifest["parameters"]["large_region_ratio_multiplier"] == 2.0
