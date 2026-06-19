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

from adk_world_model.world_model import DEFAULT_TRAINING_AREAS, LULC_COLLECTION, extract_lulc_labels


DEFAULT_OUTPUT_DIR = ROOT / "data" / "independent_change_labels" / "labels"
DEFAULT_MANIFEST = ROOT / "data" / "independent_change_labels" / "label_manifest.json"
DEFAULT_YEARS = list(range(2017, 2025))

EXTRA_VALIDATION_AREAS = [
    {"name": "bishan", "bbox": [106.02, 29.38, 106.33, 29.68]},
    {"name": "banzhucun", "bbox": [106.1100, 29.5799, 106.1647, 29.6124]},
    # Match experiments/paper8/data/heping/metadata.json so independent labels
    # are co-registered with cached Heping prediction grids.
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


def _extract_lulc_labels_fixed_scale(bbox: list[float], year: int, scale: int = 10) -> np.ndarray | None:
    """Extract ESRI LULC labels at the requested scale without auto-scale adjustment."""
    import ee
    from adk_world_model.world_model import _init_gee

    if not _init_gee():
        return None
    region = ee.Geometry.Rectangle(bbox)
    img = (
        ee.ImageCollection(LULC_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(region)
        .select(["b1"])
        .mosaic()
        .clip(region)
    )
    img = img.setDefaultProjection(ee.Projection("EPSG:4326").atScale(scale))
    result = img.sampleRectangle(region=region, defaultValue=0).getInfo()
    band_data = result.get("properties", {}).get("b1")
    if band_data is None:
        return None
    return np.array(band_data, dtype=np.int32)


def fetch_independent_lulc_labels(
    areas: list[str] | None = None,
    years: list[int] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    area_manifest_path: Path | None = None,
    scale: int = 10,
    overwrite: bool = False,
    fixed_scale: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    area_map = _area_lookup()
    area_map.update(_load_area_manifest(area_manifest_path))
    selected_names = areas or ["wuyi_mountain", "poyang_lake", "bishan", "banzhucun", "heping"]
    selected_years = years or DEFAULT_YEARS

    records = []
    failures = []
    for area_name in selected_names:
        area = area_map.get(area_name.lower())
        if area is None:
            failures.append({"area": area_name, "reason": "unknown_area"})
            continue
        for year in selected_years:
            out_path = output_dir / f"{area['name']}_lulc_{year}.npy"
            if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
                arr = np.load(out_path)
                records.append(
                    {
                        "area": area["name"],
                        "year": year,
                        "bbox": area["bbox"],
                        "path": str(out_path),
                        "shape": list(arr.shape),
                        "status": "cached",
                    }
                )
                continue
            extractor = _extract_lulc_labels_fixed_scale if fixed_scale else extract_lulc_labels
            labels = extractor(area["bbox"], year, scale=scale)
            if labels is None:
                failures.append({"area": area["name"], "year": year, "reason": "extract_lulc_labels_returned_none"})
                continue
            np.save(out_path, labels.astype(np.int32, copy=False))
            records.append(
                {
                    "area": area["name"],
                    "year": year,
                    "bbox": area["bbox"],
                    "path": str(out_path),
                    "shape": list(labels.shape),
                    "status": "fetched",
                }
            )

    manifest = {
        "status": "complete" if records and not failures else "partial" if records else "failed",
        "source": "ESRI Global LULC 10m Time Series via src.adk_world_model.world_model.extract_lulc_labels",
        "scale_m": scale,
        "scale_mode": "fixed" if fixed_scale else "auto_adjusted",
        "output_dir": str(output_dir),
        "n_records": len(records),
        "n_failures": len(failures),
        "records": records,
        "failures": failures,
        "next_step": (
            "Generate model-predicted end-year LULC maps in data/independent_change_labels/predicted, "
            "then run scripts/rse_revision/evaluate_independent_change_validation.py."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch independent annual LULC labels for Paper58 change validation.")
    parser.add_argument(
        "--areas",
        default="wuyi_mountain,poyang_lake,bishan,banzhucun,heping",
        help="Comma-separated area names. Use names from DEFAULT_TRAINING_AREAS plus bishan,banzhucun,heping.",
    )
    parser.add_argument("--years", default="2017-2024", help="Comma-separated years or ranges, e.g. 2020,2021 or 2017-2024.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--area-manifest", type=Path)
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fixed-scale",
        action="store_true",
        help="Use the requested GEE projection scale directly instead of the world_model auto-scale safeguard.",
    )
    args = parser.parse_args()
    manifest = fetch_independent_lulc_labels(
        areas=_parse_csv_values(args.areas),
        years=_parse_years(args.years),
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        area_manifest_path=args.area_manifest,
        scale=args.scale,
        overwrite=args.overwrite,
        fixed_scale=args.fixed_scale,
    )
    print(
        "Independent LULC label fetch: "
        f"{manifest['status']}, "
        f"{manifest['n_records']} record(s), "
        f"{manifest['n_failures']} failure(s)"
    )


if __name__ == "__main__":
    main()
