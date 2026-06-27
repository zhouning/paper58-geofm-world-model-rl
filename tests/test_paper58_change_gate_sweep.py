import numpy as np


def test_apply_scored_change_gate_keeps_top_scored_changes() -> None:
    from scripts.paper58_benchmark.sweep_paper58_change_gate import apply_scored_change_gate

    start = np.array([[1, 1, 1, 1]], dtype=np.int32)
    prediction = np.array([[5, 5, 7, 7]], dtype=np.int32)
    score = np.array([[0.1, 0.9, 0.2, 0.8]], dtype=np.float32)

    gated, diagnostics = apply_scored_change_gate(start, prediction, score, keep_fraction=0.5)

    assert gated.tolist() == [[1, 5, 1, 7]]
    assert diagnostics["candidate_change_pixels"] == 4
    assert diagnostics["kept_change_pixels"] == 2


def test_sweep_scored_change_gate_reports_best_fraction_by_metric() -> None:
    from scripts.paper58_benchmark.sweep_paper58_change_gate import (
        best_fraction_by_metric,
        sweep_scored_change_gate,
    )

    start = np.array([[1, 1, 1, 1]], dtype=np.int32)
    end = np.array([[1, 5, 1, 7]], dtype=np.int32)
    prediction = np.array([[5, 5, 7, 7]], dtype=np.int32)
    score = np.array([[0.1, 0.9, 0.2, 0.8]], dtype=np.float32)

    rows = sweep_scored_change_gate(
        area="synthetic",
        start_map=start,
        end_map=end,
        prediction_map=prediction,
        score_map=score,
        keep_fractions=[0.5, 1.0],
    )
    best = best_fraction_by_metric(rows)

    assert best["change_f1"]["keep_fraction"] == 0.5
    assert best["change_f1"]["change_f1"] == 1.0
    assert {row["pred_change_pixels"] for row in rows} == {2, 4}
