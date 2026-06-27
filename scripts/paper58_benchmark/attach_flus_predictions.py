from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts.paper58_benchmark.evaluate_benchmark import _read_registry
from scripts.paper58_benchmark.schema import write_json


SUPPORTED_FLUS_SUFFIXES = (".tif", ".tiff", ".npy", ".csv")


def _candidate_names(area: str, start_year: int, end_year: int) -> list[str]:
    stems = [
        f"{area}_{start_year}_{end_year}_flus",
        f"{area}_{start_year}_{end_year}",
        f"{area}_flus_{start_year}_{end_year}",
    ]
    return [f"{stem}{suffix}" for stem in stems for suffix in SUPPORTED_FLUS_SUFFIXES]


def _find_prediction(prediction_dir: Path, area: str, start_year: int, end_year: int) -> Path | None:
    for name in _candidate_names(area, start_year, end_year):
        candidate = prediction_dir / name
        if candidate.exists():
            return candidate
    return None


def attach_flus_predictions(
    registry_path: Path,
    prediction_dir: Path,
    output_path: Path,
    strict: bool = False,
) -> dict[str, Any]:
    rows = _read_registry(Path(registry_path))
    source_dir = Path(prediction_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"FLUS prediction directory not found: {source_dir}")

    matched = 0
    missing: list[dict[str, int | str]] = []
    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        area = str(row.get("area"))
        start_year = int(row.get("start_year"))
        end_year = int(row.get("end_year"))
        prediction = _find_prediction(source_dir, area, start_year, end_year)
        if prediction is None:
            missing.append({"area": area, "start_year": start_year, "end_year": end_year})
        else:
            updated["flus_prediction_path"] = str(prediction)
            matched += 1
        updated_rows.append(updated)

    summary = {
        "n_rows": len(rows),
        "n_matched": matched,
        "n_missing": len(missing),
        "missing": missing,
    }
    if strict and missing:
        raise ValueError(f"missing FLUS predictions for {len(missing)} row(s)")
    write_json(Path(output_path), {"rows": updated_rows, "flus_attachment_summary": summary})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach FLUS prediction paths to a Paper58 benchmark registry.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    summary = attach_flus_predictions(
        registry_path=args.registry,
        prediction_dir=args.prediction_dir,
        output_path=args.output,
        strict=args.strict,
    )
    print(
        "FLUS prediction attachment: "
        f"{summary['n_matched']}/{summary['n_rows']} matched, "
        f"{summary['n_missing']} missing"
    )


if __name__ == "__main__":
    main()
