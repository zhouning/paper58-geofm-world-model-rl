from __future__ import annotations

import json
from pathlib import Path
from typing import Any


AEF_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
AEF_BANDS = [f"A{i:02d}" for i in range(64)]
LULC_COLLECTION = "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
SRTM_DEM = "USGS/SRTMGL1_003"

OPTIONAL_DRIVER_NAMES = [
    "distance_to_highway",
    "distance_to_railway",
    "distance_to_road",
    "distance_to_town",
    "distance_to_water",
]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _task_output(prefix: str, name: str, suffix: str = ".tif") -> str:
    return f"{prefix}/{name}{suffix}"


def _gee_raster_layers(row: dict[str, Any], start_year: int, end_year: int, output_prefix: str) -> list[dict[str, Any]]:
    scale_m = int(row.get("target_scale_m", 100))
    return [
        {
            "name": "lulc_start",
            "role": "start_land_use",
            "source": "gee_image_collection",
            "collection": LULC_COLLECTION,
            "year": int(start_year),
            "bands": ["b1"],
            "scale_m": scale_m,
            "output": _task_output(output_prefix, f"lulc_{start_year}"),
        },
        {
            "name": "lulc_end",
            "role": "reference_end_land_use",
            "source": "gee_image_collection",
            "collection": LULC_COLLECTION,
            "year": int(end_year),
            "bands": ["b1"],
            "scale_m": scale_m,
            "output": _task_output(output_prefix, f"lulc_{end_year}"),
        },
        {
            "name": "alphaearth_start",
            "role": "paper58_start_embedding",
            "source": "gee_image_collection",
            "collection": AEF_COLLECTION,
            "year": int(start_year),
            "bands": AEF_BANDS,
            "scale_m": scale_m,
            "output": _task_output(output_prefix, f"alphaearth_{start_year}"),
        },
        {
            "name": "alphaearth_end",
            "role": "paper58_reference_embedding",
            "source": "gee_image_collection",
            "collection": AEF_COLLECTION,
            "year": int(end_year),
            "bands": AEF_BANDS,
            "scale_m": scale_m,
            "output": _task_output(output_prefix, f"alphaearth_{end_year}"),
        },
        {
            "name": "dem",
            "role": "flus_terrain_driver",
            "source": "gee_image",
            "collection": SRTM_DEM,
            "operation": "select:elevation",
            "bands": ["elevation"],
            "scale_m": max(30, scale_m),
            "output": _task_output(output_prefix, "dem"),
        },
        {
            "name": "slope",
            "role": "flus_terrain_driver",
            "source": "gee_terrain",
            "collection": SRTM_DEM,
            "operation": "ee.Terrain.slope",
            "bands": ["slope"],
            "scale_m": max(30, scale_m),
            "output": _task_output(output_prefix, "slope"),
        },
        {
            "name": "aspect",
            "role": "flus_terrain_driver",
            "source": "gee_terrain",
            "collection": SRTM_DEM,
            "operation": "ee.Terrain.aspect",
            "bands": ["aspect"],
            "scale_m": max(30, scale_m),
            "output": _task_output(output_prefix, "aspect"),
        },
    ]


def _derived_layers(row: dict[str, Any], start_year: int, end_year: int, output_prefix: str) -> list[dict[str, Any]]:
    return [
        {
            "name": "restriction_mask_from_admin_polygon",
            "role": "flus_restriction_or_valid_data_mask",
            "source": "candidate_manifest_geometry",
            "source_shp_path": row.get("source_shp_path", ""),
            "record_index": int(row.get("record_index", -1)),
            "strategy": "mask pixels outside administrative polygon before FLUS export",
            "output": _task_output(output_prefix, "restrictedarea"),
        },
        {
            "name": "probability_of_occurrence",
            "role": "flus_probability_cube",
            "source": "paper58_prediction_or_suitability_model",
            "strategy": "derive one band per FLUS/Paper58 class after LULC and embedding acquisition",
            "output": _task_output(output_prefix, "Probability-of-occurrence"),
        },
        {
            "name": "future_demand",
            "role": "flus_future_pixels",
            "source": "reference_end_for_oracle_diagnostic_or_non_oracle_transition_prior",
            "years": {"start": int(start_year), "end": int(end_year)},
            "output": _task_output(output_prefix, "config_mp", suffix=".log"),
        },
    ]


def _optional_driver_assets(output_prefix: str) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "role": "flus_proximity_driver",
            "status": "requires_gee_asset",
            "asset_id": None,
            "strategy": "distance transform from user-provided road/rail/town/water vector or raster asset",
            "output": _task_output(output_prefix, name),
        }
        for name in OPTIONAL_DRIVER_NAMES
    ]


def build_gee_export_specs(
    candidate_manifest_path: Path,
    start_year: int = 2020,
    end_year: int = 2021,
    limit: int | None = None,
    output_root: str = "data/flus_native_admin",
) -> dict[str, Any]:
    manifest = _read_json(Path(candidate_manifest_path))
    candidate_rows = manifest.get("rows")
    if not isinstance(candidate_rows, list):
        raise ValueError("candidate manifest must contain a 'rows' list")
    selected_rows = candidate_rows[: int(limit)] if limit is not None else candidate_rows

    tasks = []
    for row in selected_rows:
        if not isinstance(row, dict):
            raise ValueError("candidate manifest rows must be objects")
        area_id = str(row.get("area_id"))
        output_prefix = f"{output_root.rstrip('/')}/{area_id}"
        tasks.append(
            {
                "area_id": area_id,
                "admin": {
                    "province": row.get("province", ""),
                    "city": row.get("city", ""),
                    "county": row.get("county", ""),
                    "town": row.get("town", ""),
                },
                "years": {"start": int(start_year), "end": int(end_year)},
                "bbox": [float(value) for value in row.get("bbox", [])],
                "target_scale_m": int(row.get("target_scale_m", 100)),
                "estimated_shape": {
                    "width_px": int(row.get("estimated_width_px", 0)),
                    "height_px": int(row.get("estimated_height_px", 0)),
                },
                "output_prefix": output_prefix,
                "gee_rasters": _gee_raster_layers(row, start_year, end_year, output_prefix),
                "derived_layers": _derived_layers(row, start_year, end_year, output_prefix),
                "optional_driver_assets": _optional_driver_assets(output_prefix),
            }
        )

    return {
        "version": 1,
        "purpose": "Paper58 FLUS-native GEE export task specifications",
        "candidate_manifest_path": str(candidate_manifest_path),
        "gee_sources": {
            "alphaearth": AEF_COLLECTION,
            "lulc": LULC_COLLECTION,
            "terrain_dem": SRTM_DEM,
        },
        "summary": {
            "source_rows": len(candidate_rows),
            "n_tasks": len(tasks),
            "start_year": int(start_year),
            "end_year": int(end_year),
        },
        "tasks": tasks,
    }
