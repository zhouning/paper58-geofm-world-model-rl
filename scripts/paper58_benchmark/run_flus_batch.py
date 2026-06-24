from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper58_benchmark.evaluate_benchmark import _read_registry
from scripts.paper58_benchmark.evaluate_las import _path
from scripts.paper58_benchmark.flus_case import (
    decode_flus_geotiff,
    find_flus_simulation_result,
    write_flus_case,
)
from scripts.paper58_benchmark.las_demand import derive_observed_demand
from scripts.paper58_benchmark.las_suitability import class_values_from_maps, one_hot_probability_cube


ConsoleRunner = Callable[[Path], None]


def _load_array(path_value: object) -> np.ndarray:
    path = _path(path_value)
    if path is None:
        raise FileNotFoundError("missing array path")
    return np.load(path)


def _default_console_runner(flus_executable: Path) -> ConsoleRunner:
    executable = Path(flus_executable)

    def run(case_dir: Path) -> None:
        subprocess.run([str(executable)], cwd=case_dir, check=True)

    return run


def _failure(row: dict[str, Any], exc: Exception) -> dict[str, str | int]:
    return {
        "area": str(row.get("area", "")),
        "start_year": int(row.get("start_year", 0)),
        "end_year": int(row.get("end_year", 0)),
        "reason": f"{type(exc).__name__}: {exc}",
    }


def run_flus_batch(
    registry_path: Path,
    case_root: Path,
    prediction_dir: Path,
    flus_executable: Path,
    console_runner: ConsoleRunner | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    rows = [row for row in _read_registry(Path(registry_path)) if row.get("qc_status") == "include"]
    cases = Path(case_root)
    predictions = Path(prediction_dir)
    cases.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)
    run_console = console_runner or _default_console_runner(Path(flus_executable))

    n_ran = 0
    failures: list[dict[str, str | int]] = []
    for row in rows:
        try:
            area = str(row.get("area"))
            start_year = int(row.get("start_year"))
            end_year = int(row.get("end_year"))
            start = _load_array(row.get("label_start_path")).astype(np.int32, copy=False)
            end = _load_array(row.get("label_end_path")).astype(np.int32, copy=False)
            paper58_pred = _load_array(row.get("prediction_path")).astype(np.int32, copy=False)
            classes = class_values_from_maps(start, end, paper58_pred)
            probability = one_hot_probability_cube(paper58_pred, classes, confidence=0.95, floor=0.01)
            demand = derive_observed_demand(end)
            case_dir = cases / f"{area}_{start_year}_{end_year}"
            write_flus_case(
                output_dir=case_dir,
                start_map=start,
                probability_cube=probability,
                class_values=classes,
                future_demand=demand,
                end_year=end_year,
            )
            run_console(case_dir)
            encoded_result = find_flus_simulation_result(case_dir, end_year=end_year)
            decoded_result = predictions / f"{area}_{start_year}_{end_year}_flus.tif"
            decode_flus_geotiff(encoded_result, decoded_result, classes)
            n_ran += 1
        except Exception as exc:
            failures.append(_failure(row, exc))
            if strict:
                raise

    return {
        "n_rows": len(rows),
        "n_ran": n_ran,
        "n_failed": len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FLUS console on a Paper58 benchmark registry.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--flus-executable", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    summary = run_flus_batch(
        registry_path=args.registry,
        case_root=args.case_root,
        prediction_dir=args.prediction_dir,
        flus_executable=args.flus_executable,
        strict=args.strict,
    )
    print(
        "FLUS batch run: "
        f"{summary['n_ran']}/{summary['n_rows']} ran, "
        f"{summary['n_failed']} failed"
    )


if __name__ == "__main__":
    main()
