from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adk_world_model.world_model import DECODER_PATH, WEIGHTS_PATH
from scripts.paper58_benchmark.apply_paper58_calibrated_class_budget_gate import (
    apply_calibrated_class_budget_gate,
    estimate_class_count_ratios,
)
from scripts.paper58_benchmark.apply_paper58_calibrated_transition_exactness_gate import (
    _filter_calibration_cases,
    _parse_exclude_terms,
    apply_transition_spatial_exactness_reversion_gate,
    discover_calibration_cases,
)
from scripts.paper58_benchmark.apply_paper58_change_gate import (
    class_aligned_neighborhood,
    estimate_transition_reliability,
)
from scripts.rse_revision.generate_change_validation_predictions import (
    _load_decoder,
    _load_model,
    _predict_next_embedding,
)


EMBEDDING_RE = re.compile(r"^(?P<area>.+)_emb_(?P<year>\d{4})\.npy$")


@dataclass(frozen=True)
class LocalSemanticCase:
    area: str
    start_year: int
    end_year: int
    start_path: Path
    embedding_path: Path
    context_path: Path
    source_dir: Path


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_vectors(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-6)


def _local_class_prototypes(
    start_map: np.ndarray,
    start_embedding: np.ndarray,
    decoder_classes: list[int],
    min_class_pixels: int,
) -> tuple[list[int], np.ndarray]:
    start = np.asarray(start_map)
    embedding = _normalize_vectors(np.asarray(start_embedding, dtype=np.float32))
    if start.shape != embedding.shape[:2]:
        raise ValueError(f"shape mismatch: start={start.shape}, start_embedding={embedding.shape}")
    min_pixels = int(min_class_pixels)
    if min_pixels < 1:
        raise ValueError(f"min_class_pixels must be positive: {min_pixels}")
    prototype_classes: list[int] = []
    prototypes: list[np.ndarray] = []
    for cls in sorted({int(value) for value in decoder_classes}):
        mask = start == int(cls)
        if int(np.count_nonzero(mask)) < min_pixels:
            continue
        prototype = embedding[mask].mean(axis=0)
        prototype = prototype / max(float(np.linalg.norm(prototype)), 1e-6)
        prototype_classes.append(int(cls))
        prototypes.append(prototype.astype(np.float32, copy=False))
    if not prototypes:
        return [], np.empty((0, embedding.shape[-1]), dtype=np.float32)
    return prototype_classes, np.stack(prototypes, axis=0)


def decode_with_local_semantic_calibration(
    start_map: np.ndarray,
    start_embedding: np.ndarray,
    forecast_embedding: np.ndarray,
    global_probabilities: np.ndarray,
    decoder_classes: list[int],
    semantic_strength: float = 24.0,
    min_class_pixels: int = 20,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = np.asarray(start_map)
    start_emb = np.asarray(start_embedding, dtype=np.float32)
    forecast_emb = np.asarray(forecast_embedding, dtype=np.float32)
    probabilities = np.asarray(global_probabilities, dtype=np.float32)
    classes = [int(value) for value in decoder_classes]
    if start.ndim != 2 or start_emb.ndim != 3 or forecast_emb.ndim != 3 or probabilities.ndim != 3:
        raise ValueError(
            "shape mismatch: expected start=(H,W), embeddings=(H,W,D), probabilities=(H,W,C); "
            f"got start={start.shape}, start_embedding={start_emb.shape}, "
            f"forecast_embedding={forecast_emb.shape}, probabilities={probabilities.shape}"
        )
    if start.shape != start_emb.shape[:2] or start.shape != forecast_emb.shape[:2] or start.shape != probabilities.shape[:2]:
        raise ValueError(
            f"shape mismatch: start={start.shape}, start_embedding={start_emb.shape}, "
            f"forecast_embedding={forecast_emb.shape}, probabilities={probabilities.shape}"
        )
    if start_emb.shape[-1] != forecast_emb.shape[-1]:
        raise ValueError(
            f"embedding dimension mismatch: start={start_emb.shape[-1]}, forecast={forecast_emb.shape[-1]}"
        )
    if probabilities.shape[-1] != len(classes):
        raise ValueError(f"probability class count {probabilities.shape[-1]} does not match {len(classes)} classes")
    strength = float(semantic_strength)
    if strength < 0.0:
        raise ValueError(f"semantic_strength must be non-negative: {strength}")

    prototype_classes, prototypes = _local_class_prototypes(
        start,
        start_emb,
        classes,
        min_class_pixels=min_class_pixels,
    )
    scores = np.log(np.maximum(probabilities, 1e-8))
    if prototypes.size:
        forecast_norm = _normalize_vectors(forecast_emb).reshape(-1, forecast_emb.shape[-1])
        similarity = (forecast_norm @ prototypes.T).reshape(start.shape + (len(prototype_classes),))
        class_to_col = {int(cls): index for index, cls in enumerate(classes)}
        for proto_col, cls in enumerate(prototype_classes):
            scores[..., class_to_col[int(cls)]] += strength * similarity[..., proto_col]
    global_decoded = np.asarray(classes, dtype=np.int32)[np.argmax(probabilities, axis=-1)]
    locally_decoded = np.asarray(classes, dtype=np.int32)[np.argmax(scores, axis=-1)]
    diagnostics = {
        "decoder_classes": classes,
        "local_prototype_classes": prototype_classes,
        "semantic_strength": strength,
        "min_class_pixels": int(min_class_pixels),
        "changed_from_global_pixels": int(np.count_nonzero(locally_decoded != global_decoded)),
    }
    return locally_decoded.astype(np.int32, copy=False), diagnostics


def _discover_cases(input_dirs: list[Path], labels_dir: Path, end_year: int) -> list[LocalSemanticCase]:
    label_root = Path(labels_dir)
    cases: list[LocalSemanticCase] = []
    seen: set[tuple[str, int, int]] = set()
    for input_dir in [Path(path) for path in input_dirs]:
        for embedding_path in sorted((input_dir / "embeddings").glob("*_emb_*.npy")):
            match = EMBEDDING_RE.match(embedding_path.name)
            if match is None:
                continue
            area = str(match.group("area"))
            start_year = int(match.group("year"))
            case_end_year = int(end_year)
            key = (area, start_year, case_end_year)
            if key in seen:
                continue
            start_path = label_root / f"{area}_lulc_{start_year}.npy"
            context_path = input_dir / "context" / f"{area}_context.npy"
            if not start_path.exists() or not context_path.exists():
                continue
            cases.append(
                LocalSemanticCase(
                    area=area,
                    start_year=start_year,
                    end_year=case_end_year,
                    start_path=start_path,
                    embedding_path=embedding_path,
                    context_path=context_path,
                    source_dir=input_dir,
                )
            )
            seen.add(key)
    return cases


def _decoder_probability_maps(
    probabilities: np.ndarray,
    decoder_classes: list[int],
    start_map: np.ndarray,
    prediction_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    pred_prob = np.zeros(start.shape, dtype=np.float32)
    start_prob = np.zeros(start.shape, dtype=np.float32)
    for col, cls in enumerate(decoder_classes):
        key = int(cls)
        pred_prob[prediction == key] = probabilities[..., col][prediction == key]
        start_prob[start == key] = probabilities[..., col][start == key]
    return pred_prob, start_prob


def _semantic_change_score(
    start_map: np.ndarray,
    prediction_map: np.ndarray,
    probabilities: np.ndarray,
    decoder_classes: list[int],
    target_support_weight: float,
    source_support_penalty: float,
) -> np.ndarray:
    pred_prob, start_prob = _decoder_probability_maps(probabilities, decoder_classes, start_map, prediction_map)
    classes = sorted({int(value) for value in np.unique(start_map)} | {int(value) for value in np.unique(prediction_map)})
    target_support, source_support = class_aligned_neighborhood(start_map, prediction_map, classes)
    return (
        pred_prob
        - start_prob
        + float(target_support_weight) * target_support
        - float(source_support_penalty) * source_support
    ).astype(np.float32, copy=False)


def run_local_semantic_calibrated_gate(
    input_dirs: list[Path],
    labels_dir: Path,
    calibration_label_dir: Path,
    calibration_prediction_dir: Path,
    output_dir: Path,
    weights_path: Path = Path(WEIGHTS_PATH),
    decoder_path: Path = Path(DECODER_PATH),
    end_year: int = 2021,
    semantic_strength: float = 24.0,
    min_class_pixels: int = 20,
    max_revert_fraction: float = 1.0,
    min_group_size: int = 1000,
    exactness_smoothing: float = 500.0,
    target_support_weight: float = 0.75,
    source_support_penalty: float = 0.75,
    budget_strength: float = 0.0,
    budget_gated_classes: list[int] | None = None,
    budget_min_overbudget_pixels: int = 100,
    class_ratio_smoothing: float = 0.0,
    excluded_name_terms_by_area: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    cases = _discover_cases(input_dirs, labels_dir, end_year=end_year)
    if not cases:
        raise ValueError(f"no local semantic cases found under {input_dirs}")
    calibration_cases, skipped_calibration = discover_calibration_cases(
        Path(calibration_label_dir),
        Path(calibration_prediction_dir),
    )
    if not calibration_cases:
        raise ValueError("no usable calibration cases were discovered")

    model = _load_model(Path(weights_path))
    decoder = _load_decoder(Path(decoder_path))
    decoder_classes = [int(value) for value in decoder.classes_]
    output = Path(output_dir)
    predictions_dir = output / "predictions"
    diagnostics_dir = output / "diagnostics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    class_values = [1, 2, 4, 5, 7, 8, 9, 10, 11]
    case_summaries: list[dict[str, Any]] = []
    for case in cases:
        target_calibration_cases = _filter_calibration_cases(
            case.area,
            calibration_cases,
            excluded_name_terms_by_area,
        )
        if not target_calibration_cases:
            raise ValueError(f"no calibration cases remain after exclusions for {case.area}")
        start = np.load(case.start_path).astype(np.int32, copy=False)
        start_embedding = np.load(case.embedding_path).astype(np.float32, copy=False)
        context = np.load(case.context_path).astype(np.float32, copy=False)
        forecast_embedding = _predict_next_embedding(model, start_embedding, context)
        probabilities = decoder.predict_proba(forecast_embedding.reshape(-1, forecast_embedding.shape[-1])).reshape(
            start.shape + (len(decoder_classes),)
        )
        semantic_prediction, semantic_diagnostics = decode_with_local_semantic_calibration(
            start_map=start,
            start_embedding=start_embedding,
            forecast_embedding=forecast_embedding,
            global_probabilities=probabilities,
            decoder_classes=decoder_classes,
            semantic_strength=semantic_strength,
            min_class_pixels=min_class_pixels,
        )
        score = _semantic_change_score(
            start,
            semantic_prediction,
            probabilities,
            decoder_classes,
            target_support_weight=target_support_weight,
            source_support_penalty=source_support_penalty,
        )
        exactness, exactness_diagnostics = estimate_transition_reliability(
            [(item.start_map, item.end_map, item.prediction_map) for item in target_calibration_cases],
            alpha_exact=1.0,
            smoothing=exactness_smoothing,
        )
        valid_mask = (start != 0) & (semantic_prediction != 0)
        gated, exactness_gate_diagnostics = apply_transition_spatial_exactness_reversion_gate(
            start,
            semantic_prediction,
            score,
            exactness_by_transition=exactness,
            global_exactness=float(exactness_diagnostics["global_reliability"]),
            max_revert_fraction=max_revert_fraction,
            min_group_size=min_group_size,
            valid_mask=valid_mask,
            target_support_weight=target_support_weight,
            source_support_penalty=source_support_penalty,
        )
        class_budget_diagnostics: dict[str, Any] | None = None
        if float(budget_strength) > 0.0:
            class_ratios, ratio_diagnostics = estimate_class_count_ratios(
                target_calibration_cases,
                class_values=class_values,
                smoothing=class_ratio_smoothing,
            )
            gated, class_budget_diagnostics = apply_calibrated_class_budget_gate(
                start,
                gated,
                score,
                class_count_ratios=class_ratios,
                budget_strength=budget_strength,
                gated_classes=budget_gated_classes or [5],
                min_overbudget_pixels=budget_min_overbudget_pixels,
                valid_mask=(start != 0) & (gated != 0),
            )
            class_budget_diagnostics["ratio_diagnostics"] = ratio_diagnostics
        prediction_path = predictions_dir / f"{case.area}_lulc_pred_{case.start_year}_{case.end_year}.npy"
        np.save(prediction_path, gated.astype(np.int32, copy=False))
        diagnostic_payload = {
            "area": case.area,
            "start_year": case.start_year,
            "end_year": case.end_year,
            "source_input_dir": case.source_dir,
            "start_path": case.start_path,
            "embedding_path": case.embedding_path,
            "context_path": case.context_path,
            "calibration_area_count": len(target_calibration_cases),
            "calibration_areas": [item.area for item in target_calibration_cases],
            "excluded_name_terms": (excluded_name_terms_by_area or {}).get(case.area, []),
            "semantic_diagnostics": semantic_diagnostics,
            "exactness_diagnostics": exactness_diagnostics,
            "exactness_gate_diagnostics": exactness_gate_diagnostics,
            "class_budget_diagnostics": class_budget_diagnostics,
            "output_prediction": prediction_path,
        }
        _write_json(diagnostics_dir / f"{case.area}_local_semantic_calibrated_gate.json", diagnostic_payload)
        case_summaries.append(
            {
                "area": case.area,
                "start_year": case.start_year,
                "end_year": case.end_year,
                "output_prediction": prediction_path,
                "calibration_area_count": len(target_calibration_cases),
                "semantic_changed_from_global_pixels": semantic_diagnostics["changed_from_global_pixels"],
                "reverted_change_pixels": exactness_gate_diagnostics["reverted_change_pixels"],
                "class_budget_reverted_pixels": (
                    class_budget_diagnostics["reverted_pixels"] if class_budget_diagnostics else 0
                ),
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paper58_local_semantic_calibrated_gate",
        "n_cases": len(cases),
        "input_dirs": [Path(path) for path in input_dirs],
        "labels_dir": Path(labels_dir),
        "calibration_label_dir": Path(calibration_label_dir),
        "calibration_prediction_dir": Path(calibration_prediction_dir),
        "weights_path": Path(weights_path),
        "decoder_path": Path(decoder_path),
        "usable_calibration_area_count": len(calibration_cases),
        "skipped_calibration_cases": skipped_calibration,
        "excluded_name_terms_by_area": excluded_name_terms_by_area or {},
        "parameters": {
            "end_year": int(end_year),
            "semantic_strength": float(semantic_strength),
            "min_class_pixels": int(min_class_pixels),
            "max_revert_fraction": float(max_revert_fraction),
            "min_group_size": int(min_group_size),
            "exactness_smoothing": float(exactness_smoothing),
            "target_support_weight": float(target_support_weight),
            "source_support_penalty": float(source_support_penalty),
            "budget_strength": float(budget_strength),
            "budget_gated_classes": [int(value) for value in (budget_gated_classes or [5])],
            "budget_min_overbudget_pixels": int(budget_min_overbudget_pixels),
            "class_ratio_smoothing": float(class_ratio_smoothing),
        },
        "cases": case_summaries,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _parse_classes(values: str | None) -> list[int] | None:
    if values is None or values.strip() == "":
        return None
    return [int(value.strip()) for value in values.split(",") if value.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply Paper58 local semantic calibrated gate.")
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--calibration-label-dir", type=Path, required=True)
    parser.add_argument("--calibration-prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path(WEIGHTS_PATH))
    parser.add_argument("--decoder", type=Path, default=Path(DECODER_PATH))
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument("--semantic-strength", type=float, default=24.0)
    parser.add_argument("--min-class-pixels", type=int, default=20)
    parser.add_argument("--max-revert-fraction", type=float, default=1.0)
    parser.add_argument("--min-group-size", type=int, default=1000)
    parser.add_argument("--exactness-smoothing", type=float, default=500.0)
    parser.add_argument("--target-support-weight", type=float, default=0.75)
    parser.add_argument("--source-support-penalty", type=float, default=0.75)
    parser.add_argument("--budget-strength", type=float, default=0.0)
    parser.add_argument("--budget-gated-classes", default="5")
    parser.add_argument("--budget-min-overbudget-pixels", type=int, default=100)
    parser.add_argument("--class-ratio-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--exclude-term",
        action="append",
        default=None,
        help="Optional calibration exclusion formatted as target_area=substring; repeat as needed.",
    )
    args = parser.parse_args(argv)
    manifest = run_local_semantic_calibrated_gate(
        input_dirs=args.input_dir,
        labels_dir=args.labels_dir,
        calibration_label_dir=args.calibration_label_dir,
        calibration_prediction_dir=args.calibration_prediction_dir,
        output_dir=args.output_dir,
        weights_path=args.weights,
        decoder_path=args.decoder,
        end_year=args.end_year,
        semantic_strength=args.semantic_strength,
        min_class_pixels=args.min_class_pixels,
        max_revert_fraction=args.max_revert_fraction,
        min_group_size=args.min_group_size,
        exactness_smoothing=args.exactness_smoothing,
        target_support_weight=args.target_support_weight,
        source_support_penalty=args.source_support_penalty,
        budget_strength=args.budget_strength,
        budget_gated_classes=_parse_classes(args.budget_gated_classes),
        budget_min_overbudget_pixels=args.budget_min_overbudget_pixels,
        class_ratio_smoothing=args.class_ratio_smoothing,
        excluded_name_terms_by_area=_parse_exclude_terms(args.exclude_term),
    )
    print(
        "Paper58 local semantic calibrated gate complete: "
        f"cases={manifest['n_cases']}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
