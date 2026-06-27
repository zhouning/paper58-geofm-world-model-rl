import numpy as np


def test_local_semantic_decoder_uses_start_area_prototypes() -> None:
    from scripts.paper58_benchmark.apply_paper58_local_semantic_calibrated_gate import (
        decode_with_local_semantic_calibration,
    )

    start = np.array([[11, 11, 5, 5]], dtype=np.int32)
    start_embedding = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    forecast_embedding = np.array(
        [
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.2, 0.8]],
        ],
        dtype=np.float32,
    )
    global_probabilities = np.array(
        [
            [
                [0.40, 0.60],
                [0.45, 0.55],
                [0.10, 0.90],
                [0.20, 0.80],
            ]
        ],
        dtype=np.float32,
    )

    decoded, diagnostics = decode_with_local_semantic_calibration(
        start_map=start,
        start_embedding=start_embedding,
        forecast_embedding=forecast_embedding,
        global_probabilities=global_probabilities,
        decoder_classes=[11, 5],
        semantic_strength=4.0,
        min_class_pixels=1,
    )

    assert decoded.tolist() == [[11, 11, 5, 5]]
    assert diagnostics["local_prototype_classes"] == [5, 11]
    assert diagnostics["changed_from_global_pixels"] == 2


def test_local_semantic_decoder_requires_matching_shapes() -> None:
    from scripts.paper58_benchmark.apply_paper58_local_semantic_calibrated_gate import (
        decode_with_local_semantic_calibration,
    )

    start = np.array([[11, 5]], dtype=np.int32)
    start_embedding = np.zeros((1, 2, 2), dtype=np.float32)
    forecast_embedding = np.zeros((1, 3, 2), dtype=np.float32)
    probabilities = np.zeros((1, 2, 2), dtype=np.float32)

    try:
        decode_with_local_semantic_calibration(
            start_map=start,
            start_embedding=start_embedding,
            forecast_embedding=forecast_embedding,
            global_probabilities=probabilities,
            decoder_classes=[11, 5],
        )
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("expected shape mismatch")
