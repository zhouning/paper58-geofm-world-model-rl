import numpy as np


def test_select_best_parameter_row_prefers_fom_then_lower_allocation_then_f1() -> None:
    from scripts.paper58_benchmark.tune_paper58_spatial_demand_ratio import select_best_parameter_row

    rows = [
        {"ratio_quantile": 0.25, "mean_fom": 0.20, "mean_allocation_disagreement": 0.05, "mean_change_f1": 0.30},
        {"ratio_quantile": 0.50, "mean_fom": 0.25, "mean_allocation_disagreement": 0.20, "mean_change_f1": 0.31},
        {"ratio_quantile": 0.75, "mean_fom": 0.25, "mean_allocation_disagreement": 0.15, "mean_change_f1": 0.29},
        {"ratio_quantile": 0.90, "mean_fom": 0.25, "mean_allocation_disagreement": 0.15, "mean_change_f1": 0.34},
    ]

    best = select_best_parameter_row(rows)

    assert best["ratio_quantile"] == 0.90


def test_select_best_parameter_row_honors_transition_floor_when_reachable() -> None:
    from scripts.paper58_benchmark.tune_paper58_spatial_demand_ratio import select_best_parameter_row

    rows = [
        {
            "ratio_quantile": 0.25,
            "mean_fom": 0.30,
            "mean_allocation_disagreement": 0.05,
            "mean_change_f1": 0.31,
            "mean_transition_accuracy": 0.12,
        },
        {
            "ratio_quantile": 0.50,
            "mean_fom": 0.28,
            "mean_allocation_disagreement": 0.06,
            "mean_change_f1": 0.30,
            "mean_transition_accuracy": 0.24,
        },
        {
            "ratio_quantile": 0.75,
            "mean_fom": 0.26,
            "mean_allocation_disagreement": 0.04,
            "mean_change_f1": 0.29,
            "mean_transition_accuracy": 0.31,
        },
    ]

    best = select_best_parameter_row(rows, min_mean_transition_accuracy=0.20)

    assert best["ratio_quantile"] == 0.50


def test_select_best_parameter_row_falls_back_when_transition_floor_unreachable() -> None:
    from scripts.paper58_benchmark.tune_paper58_spatial_demand_ratio import select_best_parameter_row

    rows = [
        {
            "ratio_quantile": 0.25,
            "mean_fom": 0.30,
            "mean_allocation_disagreement": 0.05,
            "mean_change_f1": 0.31,
            "mean_transition_accuracy": 0.12,
        },
        {
            "ratio_quantile": 0.50,
            "mean_fom": 0.28,
            "mean_allocation_disagreement": 0.06,
            "mean_change_f1": 0.30,
            "mean_transition_accuracy": 0.18,
        },
    ]

    best = select_best_parameter_row(rows, min_mean_transition_accuracy=0.20)

    assert best["ratio_quantile"] == 0.25


def test_ratio_parameter_grid_expands_requested_values() -> None:
    from scripts.paper58_benchmark.tune_paper58_spatial_demand_ratio import ratio_parameter_grid

    rows = ratio_parameter_grid(
        ratio_quantiles=[0.25, 0.5],
        ratio_multipliers=[1.0],
        min_fractions=[0.05],
        max_fractions=[0.25],
        target_neighborhood_weights=[0.5, 1.0],
        source_neighborhood_penalties=[0.25],
    )

    assert rows == [
        {
            "ratio_quantile": 0.25,
            "ratio_multiplier": 1.0,
            "min_fraction": 0.05,
            "max_fraction": 0.25,
            "target_neighborhood_weight": 0.5,
            "source_neighborhood_penalty": 0.25,
        },
        {
            "ratio_quantile": 0.25,
            "ratio_multiplier": 1.0,
            "min_fraction": 0.05,
            "max_fraction": 0.25,
            "target_neighborhood_weight": 1.0,
            "source_neighborhood_penalty": 0.25,
        },
        {
            "ratio_quantile": 0.5,
            "ratio_multiplier": 1.0,
            "min_fraction": 0.05,
            "max_fraction": 0.25,
            "target_neighborhood_weight": 0.5,
            "source_neighborhood_penalty": 0.25,
        },
        {
            "ratio_quantile": 0.5,
            "ratio_multiplier": 1.0,
            "min_fraction": 0.05,
            "max_fraction": 0.25,
            "target_neighborhood_weight": 1.0,
            "source_neighborhood_penalty": 0.25,
        },
    ]


def test_ratio_parameter_grid_expands_v2_values() -> None:
    from scripts.paper58_benchmark.tune_paper58_spatial_demand_ratio import ratio_parameter_grid

    rows = ratio_parameter_grid(
        ratio_quantiles=[0.25],
        ratio_multipliers=[1.5],
        min_fractions=[0.05],
        max_fractions=[0.25],
        target_neighborhood_weights=[1.0],
        source_neighborhood_penalties=[0.5],
        transition_reliability_strengths=[0.0, 1.0],
        neighborhood_window_sizes_options=[None, [3, 5]],
        enable_adaptive_ratio_cap_values=[False, True],
        high_candidate_thresholds=[0.5],
        high_candidate_quantiles=[0.25],
    )

    assert len(rows) == 8
    assert rows[0] == {
        "ratio_quantile": 0.25,
        "ratio_multiplier": 1.5,
        "min_fraction": 0.05,
        "max_fraction": 0.25,
        "target_neighborhood_weight": 1.0,
        "source_neighborhood_penalty": 0.5,
        "enable_transition_reliability": False,
        "transition_reliability_strength": 0.0,
        "neighborhood_window_sizes": None,
        "enable_adaptive_ratio_cap": False,
        "high_candidate_threshold": 0.5,
        "high_candidate_quantile": 0.25,
    }
    assert rows[-1] == {
        "ratio_quantile": 0.25,
        "ratio_multiplier": 1.5,
        "min_fraction": 0.05,
        "max_fraction": 0.25,
        "target_neighborhood_weight": 1.0,
        "source_neighborhood_penalty": 0.5,
        "enable_transition_reliability": True,
        "transition_reliability_strength": 1.0,
        "neighborhood_window_sizes": [3, 5],
        "enable_adaptive_ratio_cap": True,
        "high_candidate_threshold": 0.5,
        "high_candidate_quantile": 0.25,
    }


def test_evaluate_ratio_parameters_leave_one_uses_v2_fold_models(monkeypatch) -> None:
    from pathlib import Path

    from scripts.paper58_benchmark.sweep_paper58_change_gate import GateSweepCase
    from scripts.paper58_benchmark import tune_paper58_spatial_demand_ratio as module

    case_a = GateSweepCase(
        area="area_a",
        start_year=2020,
        end_year=2021,
        start_map=np.array([[1, 1, 1, 1]], dtype=np.int32),
        end_map=np.array([[1, 5, 1, 1]], dtype=np.int32),
        prediction_map=np.array([[5, 5, 1, 1]], dtype=np.int32),
        score_map=np.array([[0.9, 0.8, 0.0, 0.0]], dtype=np.float32),
        source_dir=Path("."),
    )
    case_b = GateSweepCase(
        area="area_b",
        start_year=2020,
        end_year=2021,
        start_map=np.array([[1, 1, 1, 1]], dtype=np.int32),
        end_map=np.array([[1, 1, 5, 1]], dtype=np.int32),
        prediction_map=np.array([[5, 5, 5, 1]], dtype=np.int32),
        score_map=np.array([[0.9, 0.8, 0.7, 0.0]], dtype=np.float32),
        source_dir=Path("."),
    )
    calls = []

    def fake_gate(start_map, prediction_map, score_map, target_change_fraction, **kwargs):
        candidates = (np.asarray(prediction_map) != np.asarray(start_map))
        target_pixels = int(round(np.count_nonzero(candidates) * float(target_change_fraction)))
        calls.append(
            {
                "target_change_fraction": target_change_fraction,
                "transition_reliability_model": kwargs["transition_reliability_model"],
                "transition_weight_strength": kwargs["transition_weight_strength"],
                "neighborhood_window_sizes": kwargs["neighborhood_window_sizes"],
            }
        )
        return prediction_map.copy(), {
            "candidate_change_pixels": int(np.count_nonzero(candidates)),
            "target_change_pixels": target_pixels,
            "kept_change_pixels": target_pixels,
        }

    monkeypatch.setattr(module, "apply_demand_calibrated_spatial_gate", fake_gate)

    result = module.evaluate_ratio_parameters_leave_one(
        [case_a, case_b],
        {
            "ratio_quantile": 0.5,
            "ratio_multiplier": 2.0,
            "min_fraction": 0.0,
            "max_fraction": 1.0,
            "target_neighborhood_weight": 1.0,
            "source_neighborhood_penalty": 0.5,
            "enable_transition_reliability": True,
            "transition_reliability_strength": 1.0,
            "neighborhood_window_sizes": [3, 5],
            "enable_adaptive_ratio_cap": True,
            "high_candidate_threshold": 0.5,
            "high_candidate_quantile": 0.5,
        },
    )

    assert len(calls) == 2
    assert calls[0]["transition_reliability_model"]["total_predicted"] == 3
    assert calls[0]["transition_weight_strength"] == 1.0
    assert calls[0]["neighborhood_window_sizes"] == [3, 5]
    assert np.isclose(calls[0]["target_change_fraction"], 1 / 6)
    assert result["enable_adaptive_ratio_cap"] is True


def test_v2_cli_parsers_and_parameter_fields() -> None:
    from scripts.paper58_benchmark.tune_paper58_spatial_demand_ratio import (
        _bool_list,
        _parameter_fields,
        _window_size_options,
    )

    assert _bool_list("false,true,1,0,yes,no") == [False, True, True, False, True, False]
    assert _window_size_options("none;3,5;9") == [None, [3, 5], [9]]
    assert _parameter_fields([{"ratio_quantile": 0.25}]) == [
        "ratio_quantile",
        "ratio_multiplier",
        "min_fraction",
        "max_fraction",
        "target_neighborhood_weight",
        "source_neighborhood_penalty",
    ]
    assert "enable_adaptive_ratio_cap" in _parameter_fields(
        [{"ratio_quantile": 0.25, "enable_adaptive_ratio_cap": True}]
    )
