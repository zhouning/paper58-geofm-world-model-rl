from __future__ import annotations

from pathlib import Path

import numpy as np


class FLUSIngestionError(ValueError):
    """Raised when a FLUS-compatible prediction cannot be used in matched evaluation."""


def load_flus_prediction(
    path: Path,
    expected_shape: tuple[int, ...],
    allowed_classes: set[int],
) -> np.ndarray:
    source = Path(path)
    if not source.exists():
        raise FLUSIngestionError(f"FLUS prediction not found: {source}")
    if source.suffix.lower() == ".npy":
        arr = np.load(source)
    elif source.suffix.lower() == ".csv":
        arr = np.loadtxt(source, delimiter=",", dtype=np.int32, ndmin=2)
    else:
        raise FLUSIngestionError(f"unsupported FLUS prediction format: {source.suffix}")

    raw = np.asarray(arr)
    if not np.all(np.isfinite(raw)) or not np.array_equal(raw, raw.astype(np.int32)):
        raise FLUSIngestionError("non-integer FLUS labels")
    pred = raw.astype(np.int32)
    if pred.shape != expected_shape:
        raise FLUSIngestionError(f"FLUS prediction shape {pred.shape} does not match expected shape {expected_shape}")
    unknown = sorted({int(value) for value in np.unique(pred)} - {int(value) for value in allowed_classes})
    if unknown:
        raise FLUSIngestionError(f"unknown FLUS classes: {unknown}")
    return pred
