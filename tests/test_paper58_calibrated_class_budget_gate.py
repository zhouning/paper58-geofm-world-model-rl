import numpy as np


def test_class_budget_gate_reverts_low_rank_overbudget_pixels() -> None:
    from scripts.paper58_benchmark.apply_paper58_calibrated_class_budget_gate import (
        apply_calibrated_class_budget_gate,
    )

    start = np.array([[4, 4, 1, 1, 5, 5]], dtype=np.int32)
    prediction = np.array([[5, 5, 5, 5, 5, 5]], dtype=np.int32)
    score = np.array([[0.1, 0.1, 0.1, 0.1, 0.0, 0.0]], dtype=np.float32)
    target_support = np.array([[0.0, 1.0, 0.0, 0.5, 0.0, 0.0]], dtype=np.float32)
    source_support = np.array([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    gated, diagnostics = apply_calibrated_class_budget_gate(
        start,
        prediction,
        score,
        class_count_ratios={5: 4.0 / 6.0},
        budget_strength=1.0,
        gated_classes=[5],
        min_overbudget_pixels=1,
        target_support_weight=1.0,
        source_support_penalty=1.0,
        target_neighborhood=target_support,
        source_neighborhood=source_support,
    )

    assert gated.tolist() == [[4, 5, 1, 5, 5, 5]]
    assert diagnostics["reverted_pixels"] == 2
    assert diagnostics["class_rows"][0]["target_class"] == 5
    assert diagnostics["class_rows"][0]["overbudget_pixels"] == 2


def test_class4_collapse_trigger_requires_large_source_class_and_class5_expansion() -> None:
    from scripts.paper58_benchmark.apply_paper58_calibrated_class_budget_gate import (
        class4_collapse_class5_expansion_triggered,
    )

    triggered_start = np.array([[4, 4, 4, 4, 5, 5, 5, 1, 1, 1]], dtype=np.int32)
    triggered_pred = np.array([[5, 5, 5, 4, 5, 5, 5, 5, 5, 1]], dtype=np.int32)
    tiny_class4_start = np.array([[4, 5, 5, 5, 5, 5, 5, 1, 1, 1]], dtype=np.int32)
    tiny_class4_pred = np.array([[5, 5, 5, 5, 5, 5, 5, 5, 5, 1]], dtype=np.int32)

    assert class4_collapse_class5_expansion_triggered(
        triggered_start,
        triggered_pred,
        min_class4_start_fraction=0.30,
        max_class4_pred_to_start_ratio=0.30,
        min_class5_expansion_fraction=0.20,
    )
    assert not class4_collapse_class5_expansion_triggered(
        tiny_class4_start,
        tiny_class4_pred,
        min_class4_start_fraction=0.30,
        max_class4_pred_to_start_ratio=0.30,
        min_class5_expansion_fraction=0.20,
    )
