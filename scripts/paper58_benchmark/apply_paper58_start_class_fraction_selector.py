from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PREDICTION_RE = re.compile(r"^(?P<area>.+)_lulc_pred_(?P<start_year>\d{4})_(?P<end_year>\d{4})\.npy$")


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


def _selector_fraction(start_map: np.ndarray, selector_class: int, valid_mask: np.ndarray | None = None) -> float:
    start = np.asarray(start_map)
    valid = start != 0 if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != start.shape:
        raise ValueError(f"valid_mask shape {valid.shape} does not match start shape {start.shape}")
    total = int(np.count_nonzero(valid))
    if total == 0:
        return 0.0
    return float(np.count_nonzero(valid & (start == int(selector_class))) / total)


def _prediction_files(prediction_dir: Path, start_year: int, end_year: int) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(Path(prediction_dir).glob(f"*_lulc_pred_{int(start_year)}_{int(end_year)}.npy")):
        match = PREDICTION_RE.match(path.name)
        if match is not None:
            files[str(match.group("area"))] = path
    return files


def run_start_class_fraction_selector(
    labels_dir: Path,
    fallback_prediction_dir: Path,
    semantic_prediction_dir: Path,
    output_dir: Path,
    selector_class: int = 5,
    max_semantic_fraction: float = 0.25,
    start_year: int = 2020,
    end_year: int = 2021,
) -> dict[str, Any]:
    threshold = float(max_semantic_fraction)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"max_semantic_fraction must be in [0, 1]: {threshold}")
    label_root = Path(labels_dir)
    fallback_root = Path(fallback_prediction_dir)
    semantic_root = Path(semantic_prediction_dir)
    output = Path(output_dir)
    predictions_dir = output / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    fallback_files = _prediction_files(fallback_root, start_year, end_year)
    semantic_files = _prediction_files(semantic_root, start_year, end_year)
    common_areas = sorted(set(fallback_files) & set(semantic_files))
    if not common_areas:
        raise ValueError("no common prediction areas found")

    cases: list[dict[str, Any]] = []
    for area in common_areas:
        start_path = label_root / f"{area}_lulc_{int(start_year)}.npy"
        if not start_path.exists():
            raise FileNotFoundError(f"missing start map for selector: {start_path}")
        start = np.load(start_path).astype(np.int32, copy=False)
        fallback = np.load(fallback_files[area]).astype(np.int32, copy=False)
        semantic = np.load(semantic_files[area]).astype(np.int32, copy=False)
        if start.shape != fallback.shape or start.shape != semantic.shape:
            raise ValueError(
                f"shape mismatch for {area}: start={start.shape}, "
                f"fallback={fallback.shape}, semantic={semantic.shape}"
            )
        fraction = _selector_fraction(start, selector_class=selector_class)
        selected_branch = "semantic" if fraction <= threshold else "fallback"
        selected = semantic if selected_branch == "semantic" else fallback
        output_path = predictions_dir / f"{area}_lulc_pred_{int(start_year)}_{int(end_year)}.npy"
        np.save(output_path, selected)
        cases.append(
            {
                "area": area,
                "start_year": int(start_year),
                "end_year": int(end_year),
                "selector_class": int(selector_class),
                "selector_fraction": fraction,
                "max_semantic_fraction": threshold,
                "selected_branch": selected_branch,
                "start_map": start_path,
                "fallback_prediction": fallback_files[area],
                "semantic_prediction": semantic_files[area],
                "output_prediction": output_path,
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paper58_start_class_fraction_region_selector",
        "labels_dir": label_root,
        "fallback_prediction_dir": fallback_root,
        "semantic_prediction_dir": semantic_root,
        "output_dir": output,
        "parameters": {
            "selector_class": int(selector_class),
            "max_semantic_fraction": threshold,
            "start_year": int(start_year),
            "end_year": int(end_year),
        },
        "cases": cases,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select between two Paper58 prediction branches using only start-year class fraction."
    )
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--fallback-prediction-dir", type=Path, required=True)
    parser.add_argument("--semantic-prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selector-class", type=int, default=5)
    parser.add_argument("--max-semantic-fraction", type=float, default=0.25)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2021)
    args = parser.parse_args(argv)
    manifest = run_start_class_fraction_selector(
        labels_dir=args.labels_dir,
        fallback_prediction_dir=args.fallback_prediction_dir,
        semantic_prediction_dir=args.semantic_prediction_dir,
        output_dir=args.output_dir,
        selector_class=args.selector_class,
        max_semantic_fraction=args.max_semantic_fraction,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(
        "Paper58 start-class-fraction selector complete: "
        f"cases={len(manifest['cases'])}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
