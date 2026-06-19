from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adk_world_model.world_model import DEFAULT_TRAINING_AREAS, extract_embeddings, extract_terrain_context


DEFAULT_OUTPUT_DIR = ROOT / "data" / "independent_change_labels" / "embeddings"
DEFAULT_MANIFEST = ROOT / "data" / "independent_change_labels" / "embedding_manifest.json"
DEFAULT_YEARS = [2020, 2021]


EXTRA_VALIDATION_AREAS = [
    {"name": "bishan", "bbox": [106.02, 29.38, 106.33, 29.68]},
    {"name": "banzhucun", "bbox": [106.1100, 29.5799, 106.1647, 29.6124]},
    {"name": "heping", "bbox": [106.1133, 29.5997, 106.1452, 29.6460]},
]


def _area_lookup() -> dict[str, dict]:
    areas = {}
    for area in [*DEFAULT_TRAINING_AREAS, *EXTRA_VALIDATION_AREAS]:
        areas[area["name"].lower()] = {"name": area["name"].lower(), "bbox": area["bbox"]}
    return areas


def _load_area_manifest(area_manifest_path: Path | None) -> dict[str, dict]:
    if area_manifest_path is None:
        return {}
    path = Path(area_manifest_path)
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    areas = payload.get("areas", []) if isinstance(payload, dict) else []
    loaded: dict[str, dict] = {}
    for area in areas:
        if not isinstance(area, dict):
            continue
        name = area.get("name") or area.get("area")
        bbox = area.get("bbox")
        if not isinstance(name, str) or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        loaded[name.lower()] = {"name": name.lower(), "bbox": bbox}
    return loaded


def _parse_csv_values(raw: str) -> list[str]:
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _parse_years(raw: str) -> list[int]:
    years = []
    for item in _parse_csv_values(raw):
        if "-" in item:
            start, end = item.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(item))
    return sorted(set(years))


def fetch_change_validation_embeddings(
    areas: list[str] | None = None,
    years: list[int] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    area_manifest_path: Path | None = None,
    scale: int = 500,
    overwrite: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    area_map = _area_lookup()
    area_map.update(_load_area_manifest(area_manifest_path))
    selected_names = areas or ["poyang_lake", "wuyi_mountain"]
    selected_years = years or DEFAULT_YEARS

    records = []
    context_records = []
    failures = []
    for area_name in selected_names:
        area = area_map.get(area_name.lower())
        if area is None:
            failures.append({"area": area_name, "reason": "unknown_area"})
            continue

        first_grid_shape: tuple[int, int] | None = None
        for year in selected_years:
            out_path = output_dir / f"{area['name']}_emb_{year}.npy"
            if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
                grid = np.load(out_path)
                first_grid_shape = first_grid_shape or tuple(grid.shape[:2])
                records.append(
                    {
                        "area": area["name"],
                        "year": year,
                        "bbox": area["bbox"],
                        "path": str(out_path),
                        "shape": list(grid.shape),
                        "status": "cached",
                    }
                )
                continue

            grid = extract_embeddings(area["bbox"], year, scale=scale)
            if grid is None:
                failures.append({"area": area["name"], "year": year, "reason": "extract_embeddings_returned_none"})
                continue
            np.save(out_path, grid.astype(np.float32, copy=False))
            first_grid_shape = first_grid_shape or tuple(grid.shape[:2])
            records.append(
                {
                    "area": area["name"],
                    "year": year,
                    "bbox": area["bbox"],
                    "path": str(out_path),
                    "shape": list(grid.shape),
                    "status": "fetched",
                }
            )

        context_path = output_dir / f"{area['name']}_context.npy"
        if first_grid_shape is None:
            continue
        if context_path.exists() and context_path.stat().st_size > 0 and not overwrite:
            context = np.load(context_path)
            context_records.append(
                {
                    "area": area["name"],
                    "bbox": area["bbox"],
                    "path": str(context_path),
                    "shape": list(context.shape),
                    "status": "cached",
                }
            )
            continue
        context = extract_terrain_context(area["bbox"], target_shape=first_grid_shape)
        if context is None:
            failures.append({"area": area["name"], "reason": "extract_terrain_context_returned_none"})
            continue
        np.save(context_path, context.astype(np.float32, copy=False))
        context_records.append(
            {
                "area": area["name"],
                "bbox": area["bbox"],
                "path": str(context_path),
                "shape": list(context.shape),
                "status": "fetched",
            }
        )

    manifest = {
        "status": "complete" if records and not failures else "partial" if records else "failed",
        "source": "AlphaEarth annual satellite embeddings via src.adk_world_model.world_model.extract_embeddings",
        "scale_m": scale,
        "output_dir": str(output_dir),
        "n_records": len(records),
        "n_context_records": len(context_records),
        "n_failures": len(failures),
        "records": records,
        "context_records": context_records,
        "failures": failures,
        "next_step": (
            "Run scripts/rse_revision/generate_change_validation_predictions.py with "
            "--embedding-dir data/independent_change_labels/embeddings, then rerun independent validation."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch AlphaEarth grids for Paper58 independent change validation.")
    parser.add_argument("--areas", default="poyang_lake,wuyi_mountain", help="Comma-separated area names.")
    parser.add_argument("--years", default="2020,2021", help="Comma-separated years or ranges, e.g. 2020,2021.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--area-manifest", type=Path)
    parser.add_argument("--scale", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = fetch_change_validation_embeddings(
        areas=_parse_csv_values(args.areas),
        years=_parse_years(args.years),
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        area_manifest_path=args.area_manifest,
        scale=args.scale,
        overwrite=args.overwrite,
    )
    print(
        "Change-validation embedding fetch: "
        f"{manifest['status']}, "
        f"{manifest['n_records']} grid(s), "
        f"{manifest['n_context_records']} context grid(s), "
        f"{manifest['n_failures']} failure(s)"
    )


if __name__ == "__main__":
    main()
