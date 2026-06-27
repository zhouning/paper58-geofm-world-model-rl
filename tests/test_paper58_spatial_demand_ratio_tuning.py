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
