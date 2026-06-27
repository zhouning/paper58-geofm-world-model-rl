from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee
import fiona
import numpy as np
import requests
import rasterio
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from scripts.paper58_benchmark.audit_flus_real_dataset import (
    _save_change_map,
    _save_class_count_chart,
    _save_landuse_map,
    _write_csv,
    transition_count_rows,
    transition_summary,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHP = Path("/Users/zhouning/Downloads/shp/xiangzhen.shp")
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "rse_submission_paper58" / "flus_real_datasets" / "gee_dynamicworld_samples"
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
DW_BANDS = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]
FLUS_CLASSES = [
    ("arable", "Arable Land"),
    ("woodland", "Woodland"),
    ("meadow", "Meadow"),
    ("water", "Water"),
    ("construction", "Construction Land"),
    ("unused", "Unused Land"),
]
ADMIN_FIELDS = {
    "province": "省",
    "city": "市",
    "county": "县",
    "town": "乡",
}


def parse_admin_filter(value: str) -> dict[str, str]:
    parts = [part.strip() for part in value.split("|")]
    keys = ["province", "city", "county", "town"]
    return {key: part for key, part in zip(keys, parts, strict=False) if part}


def matches_admin_filter(properties: dict[str, Any], admin_filter: dict[str, str]) -> bool:
    for key, expected in admin_filter.items():
        field = ADMIN_FIELDS.get(key, key)
        if str(properties.get(field, "")).strip() != str(expected).strip():
            return False
    return True


def band_names_for_years(years: list[int]) -> list[str]:
    ordered = [int(year) for year in years]
    labels = [f"flus_label_{year}" for year in ordered]
    probs = [f"p_{name}_{year}" for year in ordered for name, _ in FLUS_CLASSES]
    return [*labels, *probs]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _safe_area_id(index: int, admin_filter: dict[str, str]) -> str:
    raw = "_".join(str(admin_filter.get(key, "")) for key in ["province", "city", "county", "town"] if admin_filter.get(key))
    ascii_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return f"area_{index:03d}" if not ascii_slug else f"area_{index:03d}_{ascii_slug[:60]}"


def select_admin_geometry(shp_path: Path, admin_filter: dict[str, str], simplify_tolerance: float = 0.001) -> dict[str, Any]:
    geometries = []
    properties = []
    with fiona.open(shp_path) as source:
        crs = source.crs
        for feature in source:
            props = dict(feature["properties"])
            if matches_admin_filter(props, admin_filter):
                geometries.append(shape(feature["geometry"]))
                properties.append(props)
    if not geometries:
        raise ValueError(f"admin filter matched no features: {admin_filter}")
    dissolved = unary_union(geometries)
    if simplify_tolerance > 0:
        dissolved = dissolved.simplify(simplify_tolerance, preserve_topology=True)
    bounds = [float(value) for value in dissolved.bounds]
    return {
        "geometry": dissolved,
        "feature_count": len(geometries),
        "example_properties": properties[:5],
        "bounds": bounds,
        "crs": str(crs),
    }


def _annual_dw_mean(year: int, region: ee.Geometry) -> ee.Image:
    start = f"{int(year)}-01-01"
    end = f"{int(year) + 1}-01-01"
    return ee.ImageCollection(DW_COLLECTION).filterDate(start, end).filterBounds(region).select(DW_BANDS).mean()


def _clamped_sum(*images: ee.Image) -> ee.Image:
    total = ee.Image(images[0])
    for image in images[1:]:
        total = total.add(image)
    return total.min(ee.Image.constant(1.0))


def dynamic_world_flus_probabilities(year: int, region: ee.Geometry) -> ee.Image:
    mean = _annual_dw_mean(year, region)
    bands = [
        mean.select("crops").rename(f"p_arable_{year}"),
        mean.select("trees").rename(f"p_woodland_{year}"),
        _clamped_sum(mean.select("grass"), mean.select("shrub_and_scrub")).rename(f"p_meadow_{year}"),
        _clamped_sum(mean.select("water"), mean.select("flooded_vegetation")).rename(f"p_water_{year}"),
        mean.select("built").rename(f"p_construction_{year}"),
        _clamped_sum(mean.select("bare"), mean.select("snow_and_ice")).rename(f"p_unused_{year}"),
    ]
    return ee.Image.cat(bands).clip(region)


def dynamic_world_flus_image(years: list[int], region: ee.Geometry, probability_scale: int = 10000) -> ee.Image:
    labels = []
    probabilities = []
    for year in [int(value) for value in years]:
        prob = dynamic_world_flus_probabilities(year, region)
        label = prob.toArray().arrayArgmax().arrayGet([0]).add(1).toUint8().rename(f"flus_label_{year}")
        labels.append(label)
        probabilities.append(prob.multiply(probability_scale).round().toUint16())
    return ee.Image.cat([*labels, *probabilities]).rename(band_names_for_years([int(value) for value in years])).clip(region)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _class_count_summary(array: np.ndarray, year: int, class_names: dict[int, str]) -> dict[str, Any]:
    valid = array[array != 0]
    unique, counts = np.unique(valid, return_counts=True)
    total = int(valid.size)
    return {
        "year": int(year),
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "pixel_count": int(array.size),
        "valid_pixel_count": total,
        "outside_pixel_count": int(array.size - total),
        "class_counts": [
            {
                "class_value": int(value),
                "class_name": class_names.get(int(value), f"class_{int(value)}"),
                "pixels": int(count),
                "share": float(count / total) if total else 0.0,
            }
            for value, count in zip(unique, counts, strict=False)
        ],
    }


def _write_sample_report(
    area_dir: Path,
    manifest: dict[str, Any],
    landuse_figures: list[str],
    change_figure: str,
    chart_figure: str,
) -> None:
    lines = [
        "# Dynamic World FLUS-Style Sample",
        "",
        f"- Source: `{manifest['source']}`",
        f"- Admin filter: `{manifest['admin_filter']}`",
        f"- Scale: {manifest['scale_m']} m",
        f"- Years: {manifest['years'][0]}->{manifest['years'][-1]}",
        f"- Bands: {len(manifest['bands'])}",
        f"- GeoTIFF: `{manifest['extracted_files'][0]}`",
        "",
        "## Raw Land-Use Labels",
        "",
    ]
    for figure in landuse_figures:
        lines.append(f"![{figure}](figures/{figure})")
        lines.append("")
    lines.extend(
        [
            "## Observed Change Target",
            "",
            f"![{change_figure}](figures/{change_figure})",
            "",
            "## Class Counts",
            "",
            f"![{chart_figure}](figures/{chart_figure})",
            "",
            "## Output Tables",
            "",
            "- `class_counts.csv`",
            "- `transition_counts.csv`",
            "- `transition_summary.csv`",
            "",
        ]
    )
    (area_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def audit_downloaded_sample(area_dir: Path, tif_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    years = [int(year) for year in manifest["years"]]
    class_names = {index + 1: name for index, (_, name) in enumerate(FLUS_CLASSES)}
    figure_dir = area_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tif_path) as dataset:
        labels = [dataset.read(index + 1) for index, _ in enumerate(years)]
        raster_info = {
            "width": int(dataset.width),
            "height": int(dataset.height),
            "count": int(dataset.count),
            "crs": str(dataset.crs) if dataset.crs else None,
            "bounds": [float(dataset.bounds.left), float(dataset.bounds.bottom), float(dataset.bounds.right), float(dataset.bounds.top)],
            "dtypes": list(dataset.dtypes),
        }

    summaries = [_class_count_summary(array, year, class_names) for array, year in zip(labels, years, strict=False)]
    transition_rows = transition_count_rows(labels[0], labels[-1], years[0], years[-1], class_names=class_names)
    transition_summaries = [transition_summary(labels[0], labels[-1], years[0], years[-1])]
    landuse_figures = [
        _save_landuse_map(array, year, class_names, figure_dir, title_prefix="Dynamic World FLUS-style land use")
        for array, year in zip(labels, years, strict=False)
    ]
    change_figure = _save_change_map(labels[0], labels[-1], years[0], years[-1], figure_dir)
    chart_figure = _save_class_count_chart(summaries, figure_dir)
    class_rows = [{"year": summary["year"], **row} for summary in summaries for row in summary["class_counts"]]
    _write_csv(area_dir / "class_counts.csv", class_rows, ["year", "class_value", "class_name", "pixels", "share"])
    _write_csv(
        area_dir / "transition_counts.csv",
        transition_rows,
        ["period", "from_class", "from_name", "to_class", "to_name", "pixels", "changed"],
    )
    _write_csv(
        area_dir / "transition_summary.csv",
        transition_summaries,
        [
            "period",
            "start_year",
            "end_year",
            "n_pixels",
            "changed_pixels",
            "changed_share",
            "persistent_pixels",
            "to_urban_pixels",
            "from_urban_pixels",
        ],
    )
    audit = {
        "raster": raster_info,
        "label_summaries": summaries,
        "transitions": transition_summaries,
        "figures": {
            "landuse": [f"figures/{figure}" for figure in landuse_figures],
            "change": f"figures/{change_figure}",
            "class_counts": f"figures/{chart_figure}",
        },
    }
    manifest["audit"] = audit
    _write_sample_report(area_dir, manifest, landuse_figures, change_figure, chart_figure)
    return audit


def download_sample(
    shp_path: Path,
    admin_filter: dict[str, str],
    output_dir: Path,
    area_id: str,
    years: list[int],
    scale_m: int = 80,
) -> dict[str, Any]:
    selected = select_admin_geometry(shp_path, admin_filter)
    region_geojson = mapping(selected["geometry"])
    region = ee.Geometry(region_geojson)
    image = dynamic_world_flus_image(years, region)
    area_dir = output_dir / area_id
    area_dir.mkdir(parents=True, exist_ok=True)
    url = image.getDownloadURL(
        {
            "name": area_id,
            "region": region.bounds().getInfo()["coordinates"],
            "scale": int(scale_m),
            "format": "ZIPPED_GEO_TIFF",
            "filePerBand": False,
        }
    )
    zip_path = area_dir / f"{area_id}_dynamicworld_flus_{years[0]}_{years[-1]}_{scale_m}m.zip"
    response = requests.get(url, timeout=300)
    if not response.ok:
        raise RuntimeError(
            f"Earth Engine download failed with HTTP {response.status_code}: {response.text[:1000]}"
        )
    zip_path.write_bytes(response.content)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(area_dir)
        extracted = [area_dir / name for name in zf.namelist()]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": DW_COLLECTION,
        "admin_filter": admin_filter,
        "source_shp": shp_path,
        "feature_count": selected["feature_count"],
        "bounds": selected["bounds"],
        "scale_m": int(scale_m),
        "years": [int(year) for year in years],
        "flus_classes": [
            {"class_value": index + 1, "class_key": key, "class_name": name}
            for index, (key, name) in enumerate(FLUS_CLASSES)
        ],
        "bands": band_names_for_years([int(year) for year in years]),
        "probability_scale": 10000,
        "download_zip": zip_path,
        "extracted_files": extracted,
        "example_admin_properties": selected["example_properties"],
    }
    tif_candidates = [path for path in extracted if path.suffix.lower() in {".tif", ".tiff"}]
    if tif_candidates:
        audit_downloaded_sample(area_dir, tif_candidates[0], manifest)
    _write_manifest(area_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch real Dynamic World FLUS-style samples by admin boundary.")
    parser.add_argument("--shp", type=Path, default=DEFAULT_SHP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--admin-filter", action="append", required=True, help="province|city|county|town")
    parser.add_argument("--years", type=int, nargs="+", default=[2020, 2021])
    parser.add_argument("--scale-m", type=int, default=80)
    args = parser.parse_args(argv)

    ee.Initialize()
    manifests = []
    for index, raw_filter in enumerate(args.admin_filter):
        admin_filter = parse_admin_filter(raw_filter)
        area_id = _safe_area_id(index, admin_filter)
        manifests.append(
            download_sample(
                shp_path=args.shp,
                admin_filter=admin_filter,
                output_dir=args.output_dir,
                area_id=area_id,
                years=args.years,
                scale_m=args.scale_m,
            )
        )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": DW_COLLECTION,
        "n_samples": len(manifests),
        "samples": manifests,
    }
    _write_manifest(args.output_dir / "manifest.json", summary)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
