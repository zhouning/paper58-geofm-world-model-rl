from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.rse_revision.generate_change_validation_predictions import (
    _load_decoder,
    _load_model,
    _predict_next_embedding,
)
from adk_world_model.world_model import DECODER_PATH, WEIGHTS_PATH


ROOT = Path(__file__).resolve().parents[2]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def class_neighborhood_fraction(label_map: np.ndarray, class_values: list[int]) -> dict[int, np.ndarray]:
    labels = np.asarray(label_map)
    fractions: dict[int, np.ndarray] = {}
    for cls in class_values:
        mask = (labels == int(cls)).astype(np.float32)
        support = np.zeros(labels.shape, dtype=np.float32)
        counts = np.zeros(labels.shape, dtype=np.float32)
        height, width = labels.shape
        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                if row_offset == 0 and col_offset == 0:
                    continue
                src_row_start = max(0, -row_offset)
                src_row_end = min(height, height - row_offset)
                dst_row_start = max(0, row_offset)
                dst_row_end = min(height, height + row_offset)
                src_col_start = max(0, -col_offset)
                src_col_end = min(width, width - col_offset)
                dst_col_start = max(0, col_offset)
                dst_col_end = min(width, width + col_offset)
                support[dst_row_start:dst_row_end, dst_col_start:dst_col_end] += mask[
                    src_row_start:src_row_end,
                    src_col_start:src_col_end,
                ]
                counts[dst_row_start:dst_row_end, dst_col_start:dst_col_end] += 1.0
        fractions[int(cls)] = np.divide(support, counts, out=np.zeros_like(support), where=counts > 0.0)
    return fractions


def class_aligned_neighborhood(
    start_map: np.ndarray,
    prediction_map: np.ndarray,
    class_values: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    neighborhoods = class_neighborhood_fraction(start_map, class_values)
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    target = np.zeros(start.shape, dtype=np.float32)
    source = np.zeros(start.shape, dtype=np.float32)
    for cls, support in neighborhoods.items():
        target[prediction == int(cls)] = support[prediction == int(cls)]
        source[start == int(cls)] = support[start == int(cls)]
    return target, source


def apply_change_gate(
    start_map: np.ndarray,
    prediction_map: np.ndarray,
    pred_prob: np.ndarray,
    start_prob: np.ndarray,
    keep_fraction: float,
    valid_mask: np.ndarray | None = None,
    target_neighborhood: np.ndarray | None = None,
    source_neighborhood: np.ndarray | None = None,
    target_neighborhood_weight: float = 0.5,
    source_neighborhood_penalty: float = 0.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    pred = np.asarray(pred_prob, dtype=np.float32)
    start_p = np.asarray(start_prob, dtype=np.float32)
    if start.shape != prediction.shape or start.shape != pred.shape or start.shape != start_p.shape:
        raise ValueError(
            f"shape mismatch: start={start.shape}, prediction={prediction.shape}, "
            f"pred_prob={pred.shape}, start_prob={start_p.shape}"
        )
    keep = float(keep_fraction)
    if keep < 0.0 or keep > 1.0:
        raise ValueError(f"keep_fraction must be in [0, 1]: {keep}")
    valid = np.ones(start.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != start.shape:
        raise ValueError(f"valid_mask shape {valid.shape} does not match start shape {start.shape}")
    target = np.zeros(start.shape, dtype=np.float32) if target_neighborhood is None else np.asarray(target_neighborhood, dtype=np.float32)
    source = np.zeros(start.shape, dtype=np.float32) if source_neighborhood is None else np.asarray(source_neighborhood, dtype=np.float32)
    if target.shape != start.shape or source.shape != start.shape:
        raise ValueError(f"neighborhood shape mismatch: target={target.shape}, source={source.shape}, start={start.shape}")

    candidates = valid & (prediction != start)
    candidate_indices = np.flatnonzero(candidates.ravel())
    keep_count = int(round(candidate_indices.size * keep))
    score = pred - start_p + float(target_neighborhood_weight) * target - float(source_neighborhood_penalty) * source
    gated = start.copy()
    kept_indices = np.array([], dtype=np.int64)
    if keep_count > 0 and candidate_indices.size > 0:
        order = candidate_indices[np.argsort(score.ravel()[candidate_indices])[::-1]]
        kept_indices = order[:keep_count]
        gated.ravel()[kept_indices] = prediction.ravel()[kept_indices]
    diagnostics = {
        "candidate_change_pixels": int(candidate_indices.size),
        "kept_change_pixels": int(kept_indices.size),
        "keep_fraction": keep,
        "target_neighborhood_weight": float(target_neighborhood_weight),
        "source_neighborhood_penalty": float(source_neighborhood_penalty),
        "score_min": float(np.min(score.ravel()[candidate_indices])) if candidate_indices.size else None,
        "score_max": float(np.max(score.ravel()[candidate_indices])) if candidate_indices.size else None,
    }
    return gated.astype(prediction.dtype, copy=False), diagnostics


def estimate_transition_reliability(
    training_cases: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    valid_masks: list[np.ndarray] | None = None,
    alpha_exact: float = 0.5,
    smoothing: float = 500.0,
) -> tuple[dict[tuple[int, int], float], dict[str, Any]]:
    alpha = float(alpha_exact)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha_exact must be in [0, 1]: {alpha}")
    smooth = float(smoothing)
    if smooth < 0.0:
        raise ValueError(f"smoothing must be non-negative: {smooth}")
    if valid_masks is not None and len(valid_masks) != len(training_cases):
        raise ValueError("valid_masks length must match training_cases length")

    totals: dict[tuple[int, int], int] = {}
    exact_hits: dict[tuple[int, int], int] = {}
    binary_hits: dict[tuple[int, int], int] = {}
    global_total = 0
    global_exact = 0
    global_binary = 0
    for index, (start_map, end_map, prediction_map) in enumerate(training_cases):
        start = np.asarray(start_map)
        end = np.asarray(end_map)
        prediction = np.asarray(prediction_map)
        if start.shape != end.shape or start.shape != prediction.shape:
            raise ValueError(
                f"training case {index} shape mismatch: "
                f"start={start.shape}, end={end.shape}, prediction={prediction.shape}"
            )
        valid = (
            (start != 0)
            & (end != 0)
            & (prediction != 0)
            if valid_masks is None
            else np.asarray(valid_masks[index], dtype=bool)
        )
        if valid.shape != start.shape:
            raise ValueError(f"valid mask {index} shape {valid.shape} does not match training shape {start.shape}")
        candidates = valid & (prediction != start)
        exact = candidates & (end == prediction) & (end != start)
        binary = candidates & (end != start)
        global_total += int(np.count_nonzero(candidates))
        global_exact += int(np.count_nonzero(exact))
        global_binary += int(np.count_nonzero(binary))
        for from_cls, to_cls in zip(start[candidates].ravel(), prediction[candidates].ravel(), strict=False):
            key = (int(from_cls), int(to_cls))
            totals[key] = totals.get(key, 0) + 1
        for from_cls, to_cls in zip(start[exact].ravel(), prediction[exact].ravel(), strict=False):
            key = (int(from_cls), int(to_cls))
            exact_hits[key] = exact_hits.get(key, 0) + 1
        for from_cls, to_cls in zip(start[binary].ravel(), prediction[binary].ravel(), strict=False):
            key = (int(from_cls), int(to_cls))
            binary_hits[key] = binary_hits.get(key, 0) + 1

    global_exact_rate = float(global_exact / global_total) if global_total else 0.0
    global_binary_rate = float(global_binary / global_total) if global_total else 0.0
    global_reliability = alpha * global_exact_rate + (1.0 - alpha) * global_binary_rate
    reliability: dict[tuple[int, int], float] = {}
    for key, total in totals.items():
        exact_rate = (
            float(exact_hits.get(key, 0) / total)
            if smooth == 0.0
            else float((exact_hits.get(key, 0) + smooth * global_exact_rate) / (total + smooth))
        )
        binary_rate = (
            float(binary_hits.get(key, 0) / total)
            if smooth == 0.0
            else float((binary_hits.get(key, 0) + smooth * global_binary_rate) / (total + smooth))
        )
        reliability[key] = alpha * exact_rate + (1.0 - alpha) * binary_rate

    diagnostics = {
        "training_case_count": len(training_cases),
        "training_candidate_change_pixels": int(global_total),
        "training_exact_transition_hits": int(global_exact),
        "training_binary_change_hits": int(global_binary),
        "global_exact_reliability": global_exact_rate,
        "global_binary_reliability": global_binary_rate,
        "global_reliability": global_reliability,
        "alpha_exact": alpha,
        "smoothing": smooth,
        "transition_group_count": len(reliability),
    }
    return reliability, diagnostics


def apply_transition_reliability_gate(
    start_map: np.ndarray,
    prediction_map: np.ndarray,
    score_map: np.ndarray,
    reliability_by_transition: dict[tuple[int, int], float],
    global_reliability: float,
    base_keep_fraction: float = 0.8,
    reliability_slope: float = 0.55,
    min_keep_fraction: float = 0.45,
    max_keep_fraction: float = 1.0,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    score = np.asarray(score_map, dtype=np.float32)
    if start.shape != prediction.shape or start.shape != score.shape:
        raise ValueError(f"shape mismatch: start={start.shape}, prediction={prediction.shape}, score={score.shape}")
    base_keep = float(base_keep_fraction)
    slope = float(reliability_slope)
    min_keep = float(min_keep_fraction)
    max_keep = float(max_keep_fraction)
    if base_keep < 0.0 or base_keep > 1.0:
        raise ValueError(f"base_keep_fraction must be in [0, 1]: {base_keep}")
    if min_keep < 0.0 or min_keep > 1.0 or max_keep < 0.0 or max_keep > 1.0 or min_keep > max_keep:
        raise ValueError(f"invalid keep fraction bounds: min={min_keep}, max={max_keep}")
    if slope < 0.0:
        raise ValueError(f"reliability_slope must be non-negative: {slope}")
    global_rel = float(global_reliability)
    if global_rel < 0.0 or global_rel > 1.0:
        raise ValueError(f"global_reliability must be in [0, 1]: {global_rel}")
    valid = np.ones(start.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != start.shape:
        raise ValueError(f"valid_mask shape {valid.shape} does not match start shape {start.shape}")

    candidates = valid & (prediction != start)
    gated = start.copy()
    kept_total = 0
    transition_rows: list[dict[str, Any]] = []
    denominator = max(global_rel, 1e-6)
    for from_cls in sorted({int(value) for value in np.unique(start[candidates])}):
        from_mask = candidates & (start == from_cls)
        for to_cls in sorted({int(value) for value in np.unique(prediction[from_mask])}):
            group = from_mask & (prediction == to_cls)
            indices = np.flatnonzero(group.ravel())
            if indices.size == 0:
                continue
            reliability = float(reliability_by_transition.get((from_cls, to_cls), global_rel))
            keep_fraction = base_keep + slope * (reliability - global_rel) / denominator
            keep_fraction = max(min_keep, min(max_keep, keep_fraction))
            keep_count = int(round(indices.size * keep_fraction))
            if keep_count > 0:
                order = indices[np.argsort(score.ravel()[indices])[::-1]][:keep_count]
                gated.ravel()[order] = prediction.ravel()[order]
                kept_total += int(order.size)
            transition_rows.append(
                {
                    "from_class": from_cls,
                    "to_class": to_cls,
                    "candidate_pixels": int(indices.size),
                    "kept_pixels": int(keep_count),
                    "reliability": reliability,
                    "keep_fraction": float(keep_fraction),
                }
            )

    diagnostics = {
        "candidate_change_pixels": int(np.count_nonzero(candidates)),
        "kept_change_pixels": int(kept_total),
        "base_keep_fraction": base_keep,
        "reliability_slope": slope,
        "min_keep_fraction": min_keep,
        "max_keep_fraction": max_keep,
        "global_reliability": global_rel,
        "transition_group_count": len(transition_rows),
        "transition_groups": transition_rows,
    }
    return gated.astype(prediction.dtype, copy=False), diagnostics


def apply_spatial_support_reversion_gate(
    start_map: np.ndarray,
    prediction_map: np.ndarray,
    score_map: np.ndarray,
    target_neighborhood_threshold: float = 0.25,
    score_quantile: float = 0.4,
    max_revert_fraction: float = 0.03,
    valid_mask: np.ndarray | None = None,
    target_neighborhood: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    score = np.asarray(score_map, dtype=np.float32)
    if start.shape != prediction.shape or start.shape != score.shape:
        raise ValueError(f"shape mismatch: start={start.shape}, prediction={prediction.shape}, score={score.shape}")

    support_threshold = float(target_neighborhood_threshold)
    quantile = float(score_quantile)
    max_fraction = float(max_revert_fraction)
    if support_threshold < 0.0 or support_threshold > 1.0:
        raise ValueError(f"target_neighborhood_threshold must be in [0, 1]: {support_threshold}")
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError(f"score_quantile must be in [0, 1]: {quantile}")
    if max_fraction < 0.0 or max_fraction > 1.0:
        raise ValueError(f"max_revert_fraction must be in [0, 1]: {max_fraction}")

    valid = np.ones(start.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != start.shape:
        raise ValueError(f"valid_mask shape {valid.shape} does not match start shape {start.shape}")

    if target_neighborhood is None:
        classes = sorted({int(value) for value in np.unique(start)} | {int(value) for value in np.unique(prediction)})
        target_support, _ = class_aligned_neighborhood(start, prediction, classes)
    else:
        target_support = np.asarray(target_neighborhood, dtype=np.float32)
    if target_support.shape != start.shape:
        raise ValueError(
            f"target_neighborhood shape {target_support.shape} does not match start shape {start.shape}"
        )

    candidates = valid & (prediction != start)
    candidate_indices = np.flatnonzero(candidates.ravel())
    gated = prediction.copy()
    score_threshold = float(np.quantile(score.ravel()[candidate_indices], quantile)) if candidate_indices.size else None
    eligible = candidates & (target_support <= support_threshold)
    if score_threshold is not None:
        eligible &= score <= score_threshold
    eligible_indices = np.flatnonzero(eligible.ravel())
    max_revert_count = int(round(candidate_indices.size * max_fraction))
    reverted_indices = np.array([], dtype=np.int64)
    if max_revert_count > 0 and eligible_indices.size > 0:
        support_values = target_support.ravel()[eligible_indices]
        score_values = score.ravel()[eligible_indices]
        order = eligible_indices[np.lexsort((score_values, support_values))]
        reverted_indices = order[:max_revert_count]
        gated.ravel()[reverted_indices] = start.ravel()[reverted_indices]

    diagnostics = {
        "candidate_change_pixels": int(candidate_indices.size),
        "eligible_revert_pixels": int(eligible_indices.size),
        "reverted_change_pixels": int(reverted_indices.size),
        "target_neighborhood_threshold": support_threshold,
        "score_quantile": quantile,
        "score_threshold": score_threshold,
        "max_revert_fraction": max_fraction,
        "max_revert_pixels": int(max_revert_count),
        "target_support_min": float(np.min(target_support.ravel()[candidate_indices])) if candidate_indices.size else None,
        "target_support_max": float(np.max(target_support.ravel()[candidate_indices])) if candidate_indices.size else None,
    }
    return gated.astype(prediction.dtype, copy=False), diagnostics


def apply_transition_exactness_reversion_gate(
    start_map: np.ndarray,
    prediction_map: np.ndarray,
    score_map: np.ndarray,
    exactness_by_transition: dict[tuple[int, int], float],
    global_exactness: float,
    max_revert_fraction: float = 0.4,
    min_group_size: int = 100,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    score = np.asarray(score_map, dtype=np.float32)
    if start.shape != prediction.shape or start.shape != score.shape:
        raise ValueError(f"shape mismatch: start={start.shape}, prediction={prediction.shape}, score={score.shape}")
    global_exact = float(global_exactness)
    max_revert = float(max_revert_fraction)
    min_group = int(min_group_size)
    if global_exact < 0.0 or global_exact > 1.0:
        raise ValueError(f"global_exactness must be in [0, 1]: {global_exact}")
    if max_revert < 0.0 or max_revert > 1.0:
        raise ValueError(f"max_revert_fraction must be in [0, 1]: {max_revert}")
    if min_group < 1:
        raise ValueError(f"min_group_size must be positive: {min_group}")
    valid = np.ones(start.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != start.shape:
        raise ValueError(f"valid_mask shape {valid.shape} does not match start shape {start.shape}")

    candidates = valid & (prediction != start)
    gated = prediction.copy()
    reverted_total = 0
    transition_rows: list[dict[str, Any]] = []
    denominator = max(global_exact, 1e-6)
    for from_cls in sorted({int(value) for value in np.unique(start[candidates])}):
        from_mask = candidates & (start == from_cls)
        for to_cls in sorted({int(value) for value in np.unique(prediction[from_mask])}):
            group = from_mask & (prediction == to_cls)
            indices = np.flatnonzero(group.ravel())
            if indices.size == 0:
                continue
            exactness = float(exactness_by_transition.get((from_cls, to_cls), global_exact))
            revert_fraction = 0.0
            reverted = 0
            if indices.size >= min_group and exactness < global_exact:
                revert_fraction = max_revert * (global_exact - exactness) / denominator
                revert_count = int(round(indices.size * revert_fraction))
                if revert_count > 0:
                    order = indices[np.argsort(score.ravel()[indices])]
                    reverted_indices = order[:revert_count]
                    gated.ravel()[reverted_indices] = start.ravel()[reverted_indices]
                    reverted = int(reverted_indices.size)
                    reverted_total += reverted
            transition_rows.append(
                {
                    "from_class": from_cls,
                    "to_class": to_cls,
                    "candidate_pixels": int(indices.size),
                    "exactness": exactness,
                    "revert_fraction": float(revert_fraction),
                    "reverted_pixels": reverted,
                }
            )

    diagnostics = {
        "candidate_change_pixels": int(np.count_nonzero(candidates)),
        "reverted_change_pixels": int(reverted_total),
        "global_exactness": global_exact,
        "max_revert_fraction": max_revert,
        "min_group_size": min_group,
        "transition_group_count": len(transition_rows),
        "transition_groups": transition_rows,
    }
    return gated.astype(prediction.dtype, copy=False), diagnostics


def decoder_probability_maps(
    forecast_embedding: np.ndarray,
    decoder: Any,
    start_map: np.ndarray,
    prediction_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(decoder, "predict_proba") or not hasattr(decoder, "classes_"):
        raise ValueError("decoder must expose predict_proba and classes_ for Paper58 change-gate")
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    h, w = start.shape
    probabilities = decoder.predict_proba(np.asarray(forecast_embedding, dtype=np.float32).reshape(-1, 64))
    probabilities = probabilities.reshape(h, w, len(decoder.classes_))
    pred_prob = np.zeros(start.shape, dtype=np.float32)
    start_prob = np.zeros(start.shape, dtype=np.float32)
    for col, cls in enumerate(decoder.classes_):
        key = int(cls)
        pred_prob[prediction == key] = probabilities[..., col][prediction == key]
        start_prob[start == key] = probabilities[..., col][start == key]
    return pred_prob, start_prob


def apply_paper58_change_gate(
    start_path: Path,
    prediction_path: Path,
    embedding_path: Path,
    context_path: Path,
    output_dir: Path,
    area: str,
    start_year: int,
    end_year: int,
    weights_path: Path = Path(WEIGHTS_PATH),
    decoder_path: Path = Path(DECODER_PATH),
    keep_fraction: float = 0.65,
    target_neighborhood_weight: float = 0.5,
    source_neighborhood_penalty: float = 0.25,
) -> dict[str, Any]:
    start = np.load(start_path).astype(np.int32, copy=False)
    prediction = np.load(prediction_path).astype(np.int32, copy=False)
    embedding = np.load(embedding_path).astype(np.float32, copy=False)
    context = np.load(context_path).astype(np.float32, copy=False)
    if start.shape != prediction.shape or start.shape != embedding.shape[:2]:
        raise ValueError(
            f"shape mismatch: start={start.shape}, prediction={prediction.shape}, embedding={embedding.shape}"
        )
    model = _load_model(Path(weights_path))
    decoder = _load_decoder(Path(decoder_path))
    forecast_embedding = _predict_next_embedding(model, embedding, context)
    pred_prob, start_prob = decoder_probability_maps(forecast_embedding, decoder, start, prediction)
    classes = sorted({int(value) for value in np.unique(start)} | {int(value) for value in np.unique(prediction)})
    target_neighborhood, source_neighborhood = class_aligned_neighborhood(start, prediction, classes)
    valid_mask = (start != 0) & (prediction != 0)
    gated, diagnostics = apply_change_gate(
        start,
        prediction,
        pred_prob,
        start_prob,
        keep_fraction=keep_fraction,
        valid_mask=valid_mask,
        target_neighborhood=target_neighborhood,
        source_neighborhood=source_neighborhood,
        target_neighborhood_weight=target_neighborhood_weight,
        source_neighborhood_penalty=source_neighborhood_penalty,
    )

    output = Path(output_dir)
    predictions_dir = output / "predictions"
    diagnostics_dir = output / "diagnostics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    out_path = predictions_dir / f"{area}_lulc_pred_{int(start_year)}_{int(end_year)}.npy"
    np.save(out_path, gated)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paper58_change_gate",
        "area": area,
        "start_year": int(start_year),
        "end_year": int(end_year),
        "inputs": {
            "start": Path(start_path),
            "paper58_prediction": Path(prediction_path),
            "embedding_start": Path(embedding_path),
            "context": Path(context_path),
            "weights": Path(weights_path),
            "decoder": Path(decoder_path),
        },
        "output_prediction": out_path,
        "parameters": {
            "keep_fraction": float(keep_fraction),
            "target_neighborhood_weight": float(target_neighborhood_weight),
            "source_neighborhood_penalty": float(source_neighborhood_penalty),
        },
        "diagnostics": diagnostics,
    }
    _write_json(output / "manifest.json", manifest)
    np.save(diagnostics_dir / f"{area}_change_gate_score.npy", pred_prob - start_prob + target_neighborhood_weight * target_neighborhood - source_neighborhood_penalty * source_neighborhood)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply a non-oracle Paper58 confidence and neighborhood change gate.")
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--area", required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--weights", type=Path, default=Path(WEIGHTS_PATH))
    parser.add_argument("--decoder", type=Path, default=Path(DECODER_PATH))
    parser.add_argument("--keep-fraction", type=float, default=0.65)
    parser.add_argument("--target-neighborhood-weight", type=float, default=0.5)
    parser.add_argument("--source-neighborhood-penalty", type=float, default=0.25)
    args = parser.parse_args(argv)
    manifest = apply_paper58_change_gate(
        start_path=args.start,
        prediction_path=args.prediction,
        embedding_path=args.embedding,
        context_path=args.context,
        output_dir=args.output_dir,
        area=args.area,
        start_year=args.start_year,
        end_year=args.end_year,
        weights_path=args.weights,
        decoder_path=args.decoder,
        keep_fraction=args.keep_fraction,
        target_neighborhood_weight=args.target_neighborhood_weight,
        source_neighborhood_penalty=args.source_neighborhood_penalty,
    )
    print(
        "Paper58 change-gate prediction: "
        f"area={manifest['area']}, "
        f"kept={manifest['diagnostics']['kept_change_pixels']}/"
        f"{manifest['diagnostics']['candidate_change_pixels']}, "
        f"output={manifest['output_prediction']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
