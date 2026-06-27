from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee
import fiona
import numpy as np
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import box, mapping, shape

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
for path in [ROOT, SRC_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adk_world_model.world_model import AEF_BANDS, DECODER_PATH, WEIGHTS_PATH
from scripts.paper58_benchmark.fetch_realistic_same_grid_paper58_inputs import (
    ReferenceGrid,
    _alphaearth_image,
    _download_ee_image,
    _esri_lulc_image,
    _json_ready,
    _normalize_context,
    _terrain_image,
    _write_json,
    read_aligned_bands,
)
from scripts.paper58_benchmark.flus_native_admin import iter_dbf_records, iter_shp_record_bboxes
from scripts.rse_revision.generate_change_validation_predictions import (
    _decode_lulc,
    _load_decoder,
    _load_model,
    _predict_next_embedding,
)


DEFAULT_OUTPUT_DIR = ROOT / "paper" / "rse_submission_paper58" / "flus_native_admin_same_grid_paper58_inputs_2026-06-27"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON manifest must be an object: {path}")
    return payload


def select_manifest_rows(
    manifest: dict[str, Any],
    limit: int | None = None,
    area_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("candidate manifest must contain a 'rows' list")
    selected = [row for row in rows if isinstance(row, dict)]
    if area_ids:
        wanted = {str(area_id) for area_id in area_ids}
        selected = [row for row in selected if str(row.get("area_id", "")) in wanted]
    if limit is not None:
        selected = selected[: int(limit)]
    return selected


def _fallback_admin_geometry(row: dict[str, Any]) -> dict[str, Any]:
    shp_path = Path(str(row["source_shp_path"]))
    dbf_path = shp_path.with_suffix(".dbf")
    record_index = int(row["record_index"])
    bbox_row = None
    for candidate in iter_shp_record_bboxes(shp_path):
        if int(candidate["record_index"]) == record_index:
            bbox_row = candidate
            break
    if bbox_row is None:
        raise IndexError(f"record_index not found in {shp_path}: {record_index}")
    props = {}
    for index, candidate in enumerate(iter_dbf_records(dbf_path)):
        if index == record_index:
            props = {
                "省": candidate.get("province", ""),
                "市": candidate.get("city", ""),
                "县": candidate.get("county", ""),
                "乡": candidate.get("town", ""),
                "市_县": candidate.get("city_county", ""),
                "省_县": candidate.get("province_county", ""),
            }
            break
    geom = box(*[float(value) for value in bbox_row["bbox_xy"]])  # type: ignore[index]
    return {
        "geometry": geom,
        "properties": props,
    }


def read_admin_geometry(row: dict[str, Any], simplify_tolerance: float = 0.0005) -> dict[str, Any]:
    shp_path = Path(str(row["source_shp_path"]))
    record_index = int(row["record_index"])
    try:
        with fiona.open(shp_path) as source:
            feature = None
            for index, candidate in enumerate(source):
                if index == record_index:
                    feature = candidate
                    break
            if feature is None:
                raise IndexError(f"record_index not found in {shp_path}: {record_index}")
            geom = shape(feature["geometry"])
            properties = dict(feature["properties"])
    except Exception:
        fallback = _fallback_admin_geometry(row)
        geom = fallback["geometry"]
        properties = fallback["properties"]

    if simplify_tolerance > 0:
        geom = geom.simplify(float(simplify_tolerance), preserve_topology=True)
    bounds = [float(value) for value in geom.bounds]
    return {
        "geometry": geom,
        "geometry_geojson": mapping(geom),
        "properties": properties,
        "bounds": bounds,
    }


def read_lulc_reference_grid(path: Path) -> ReferenceGrid:
    with rasterio.open(path) as dataset:
        start_label = dataset.read(1)
        return ReferenceGrid(
            width=int(dataset.width),
            height=int(dataset.height),
            count=int(dataset.count),
            crs=dataset.crs,
            transform=dataset.transform,
            bounds=(float(dataset.bounds.left), float(dataset.bounds.bottom), float(dataset.bounds.right), float(dataset.bounds.top)),
            valid_mask=(start_label > 0),
        )


def download_alphaearth_admin_aligned(
    year: int,
    clip_region: ee.Geometry,
    download_region: ee.Geometry,
    scale_m: int,
    downloads_dir: Path,
    area: str,
    reference: ReferenceGrid,
    chunk_size: int = 16,
) -> tuple[np.ndarray, list[Path]]:
    chunks = [AEF_BANDS[index : index + int(chunk_size)] for index in range(0, len(AEF_BANDS), int(chunk_size))]
    arrays = []
    paths = []
    for chunk_index, bands in enumerate(chunks):
        stem = f"{area}_alphaearth_{year}_bands_{chunk_index:02d}"
        image = _alphaearth_image(year, clip_region, bands=bands).unmask(0)
        tif_path = _download_ee_image(image, download_region, scale_m, downloads_dir, stem)
        paths.append(tif_path)
        arrays.append(read_aligned_bands(tif_path, reference, Resampling.bilinear, np.float32))
    stacked = np.concatenate(arrays, axis=0)
    if stacked.shape[0] != len(AEF_BANDS):
        raise RuntimeError(f"expected {len(AEF_BANDS)} AlphaEarth bands, got {stacked.shape[0]}")
    return stacked.transpose(1, 2, 0), paths


def _safe_properties(properties: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_ready(value) for key, value in properties.items() if str(key) != "geom"}


def fetch_one_admin_sample(
    row: dict[str, Any],
    output_dir: Path,
    model: Any,
    decoder: Any,
    start_year: int = 2020,
    end_year: int = 2021,
    scale_m: int = 100,
    alphaearth_chunk_size: int = 16,
    simplify_tolerance: float = 0.0005,
    download_end_embedding: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    area = str(row["area_id"])
    output = Path(output_dir)
    labels_dir = output / "labels"
    predictions_dir = output / "predictions"
    embeddings_dir = output / "embeddings"
    downloads_dir = output / "downloads" / area
    geometries_dir = output / "geometries"
    for directory in [labels_dir, predictions_dir, embeddings_dir, downloads_dir, geometries_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    pred_path = predictions_dir / f"{area}_lulc_pred_{start_year}_{end_year}.npy"
    start_path = labels_dir / f"{area}_lulc_{start_year}.npy"
    end_path = labels_dir / f"{area}_lulc_{end_year}.npy"
    emb_start_path = embeddings_dir / f"{area}_emb_{start_year}.npy"
    emb_end_path = embeddings_dir / f"{area}_emb_{end_year}.npy"
    context_path = embeddings_dir / f"{area}_context.npy"

    if (
        pred_path.exists()
        and start_path.exists()
        and end_path.exists()
        and emb_start_path.exists()
        and context_path.exists()
        and not overwrite
    ):
        return {
            "area": area,
            "status": "cached",
            "arrays": {
                "label_start": start_path,
                "label_end": end_path,
                "embedding_start": emb_start_path,
                "context": context_path,
                "paper58_prediction": pred_path,
            },
        }

    admin = read_admin_geometry(row, simplify_tolerance=simplify_tolerance)
    geometry_path = geometries_dir / f"{area}.geojson"
    _write_json(
        geometry_path,
        {
            "type": "Feature",
            "properties": _safe_properties(admin["properties"]),
            "geometry": admin["geometry_geojson"],
        },
    )
    clip_region = ee.Geometry(admin["geometry_geojson"])
    download_region = clip_region.bounds(maxError=1)

    label_start_tif = _download_ee_image(
        _esri_lulc_image(start_year, clip_region).unmask(0).toInt16(),
        download_region,
        scale_m,
        downloads_dir,
        f"{area}_esri_lulc_{start_year}",
    )
    reference = read_lulc_reference_grid(label_start_tif)
    if int(np.count_nonzero(reference.valid_mask)) == 0:
        raise RuntimeError(f"downloaded start LULC contains no valid pixels: {area}")

    label_end_tif = _download_ee_image(
        _esri_lulc_image(end_year, clip_region).unmask(0).toInt16(),
        download_region,
        scale_m,
        downloads_dir,
        f"{area}_esri_lulc_{end_year}",
    )
    terrain_tif = _download_ee_image(
        _terrain_image(clip_region).unmask(0),
        download_region,
        scale_m,
        downloads_dir,
        f"{area}_terrain",
    )
    emb_start, emb_start_tifs = download_alphaearth_admin_aligned(
        start_year,
        clip_region,
        download_region,
        scale_m,
        downloads_dir,
        area,
        reference,
        chunk_size=alphaearth_chunk_size,
    )
    emb_end_tifs: list[Path] = []
    if download_end_embedding:
        emb_end, emb_end_tifs = download_alphaearth_admin_aligned(
            end_year,
            clip_region,
            download_region,
            scale_m,
            downloads_dir,
            area,
            reference,
            chunk_size=alphaearth_chunk_size,
        )
        emb_end[~reference.valid_mask] = 0.0
        np.save(emb_end_path, emb_end.astype(np.float32, copy=False))

    label_start = read_aligned_bands(label_start_tif, reference, Resampling.nearest, np.int32)[0].astype(np.int32, copy=False)
    label_end = read_aligned_bands(label_end_tif, reference, Resampling.nearest, np.int32)[0].astype(np.int32, copy=False)
    context = _normalize_context(read_aligned_bands(terrain_tif, reference, Resampling.bilinear, np.float32), reference.valid_mask)

    label_start = np.where(reference.valid_mask, label_start, 0).astype(np.int32, copy=False)
    label_end = np.where(reference.valid_mask, label_end, 0).astype(np.int32, copy=False)
    emb_start[~reference.valid_mask] = 0.0
    context[:, ~reference.valid_mask] = 0.0

    pred_embedding = _predict_next_embedding(model, emb_start.astype(np.float32, copy=False), context.astype(np.float32, copy=False))
    pred_lulc = _decode_lulc(pred_embedding, decoder)
    pred_lulc = np.where(reference.valid_mask, pred_lulc, 0).astype(np.int32, copy=False)

    np.save(start_path, label_start)
    np.save(end_path, label_end)
    np.save(emb_start_path, emb_start.astype(np.float32, copy=False))
    np.save(context_path, context.astype(np.float32, copy=False))
    np.save(pred_path, pred_lulc)

    return {
        "area": area,
        "status": "fetched",
        "admin": {
            "province": row.get("province", ""),
            "city": row.get("city", ""),
            "county": row.get("county", ""),
            "town": row.get("town", ""),
            "record_index": int(row.get("record_index", -1)),
            "bounds": admin["bounds"],
            "geometry_geojson": geometry_path,
        },
        "grid": {
            "shape": [int(reference.height), int(reference.width)],
            "valid_pixels": int(np.count_nonzero(reference.valid_mask)),
            "scale_m": int(scale_m),
            "bounds": list(reference.bounds),
        },
        "years": {"start": int(start_year), "end": int(end_year)},
        "downloads": {
            "esri_start": label_start_tif,
            "esri_end": label_end_tif,
            "terrain": terrain_tif,
            "alphaearth_start_chunks": emb_start_tifs,
            "alphaearth_end_chunks": emb_end_tifs,
        },
        "arrays": {
            "label_start": start_path,
            "label_end": end_path,
            "embedding_start": emb_start_path,
            "embedding_end": emb_end_path if download_end_embedding else None,
            "context": context_path,
            "paper58_prediction": pred_path,
        },
        "label_classes_start": [int(value) for value in sorted(np.unique(label_start))],
        "label_classes_end": [int(value) for value in sorted(np.unique(label_end))],
        "prediction_classes": [int(value) for value in sorted(np.unique(pred_lulc))],
    }


def fetch_admin_same_grid_inputs(
    candidate_manifest: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    area_ids: list[str] | None = None,
    start_year: int = 2020,
    end_year: int = 2021,
    scale_m: int = 100,
    alphaearth_chunk_size: int = 16,
    simplify_tolerance: float = 0.0005,
    weights_path: Path = Path(WEIGHTS_PATH),
    decoder_path: Path = Path(DECODER_PATH),
    download_end_embedding: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    ee.Initialize()
    manifest = _read_json(candidate_manifest)
    selected_rows = select_manifest_rows(manifest, limit=limit, area_ids=area_ids)
    model = _load_model(Path(weights_path))
    decoder = _load_decoder(Path(decoder_path))
    records = []
    failures = []
    for row in selected_rows:
        try:
            records.append(
                fetch_one_admin_sample(
                    row=row,
                    output_dir=output_dir,
                    model=model,
                    decoder=decoder,
                    start_year=start_year,
                    end_year=end_year,
                    scale_m=scale_m,
                    alphaearth_chunk_size=alphaearth_chunk_size,
                    simplify_tolerance=simplify_tolerance,
                    download_end_embedding=download_end_embedding,
                    overwrite=overwrite,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "area": str(row.get("area_id", "")),
                    "record_index": int(row.get("record_index", -1)),
                    "reason": type(exc).__name__,
                    "message": str(exc)[:2000],
                }
            )

    output = Path(output_dir)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if records and not failures else "partial" if records else "failed",
        "candidate_manifest": candidate_manifest,
        "output_dir": output,
        "years": {"start": int(start_year), "end": int(end_year)},
        "scale_m": int(scale_m),
        "n_requested": len(selected_rows),
        "n_records": len(records),
        "n_failures": len(failures),
        "records": records,
        "failures": failures,
    }
    _write_json(output / "manifest.json", summary)
    return summary


def _parse_area_ids(raw: str) -> list[str] | None:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch same-grid AlphaEarth/ESRI inputs for stratified FLUS-native admin samples.")
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--area-ids", default="")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument("--scale-m", type=int, default=100)
    parser.add_argument("--alphaearth-chunk-size", type=int, default=16)
    parser.add_argument("--simplify-tolerance", type=float, default=0.0005)
    parser.add_argument("--weights", type=Path, default=Path(WEIGHTS_PATH))
    parser.add_argument("--decoder", type=Path, default=Path(DECODER_PATH))
    parser.add_argument("--download-end-embedding", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    summary = fetch_admin_same_grid_inputs(
        candidate_manifest=args.candidate_manifest,
        output_dir=args.output_dir,
        limit=args.limit,
        area_ids=_parse_area_ids(args.area_ids),
        start_year=args.start_year,
        end_year=args.end_year,
        scale_m=args.scale_m,
        alphaearth_chunk_size=args.alphaearth_chunk_size,
        simplify_tolerance=args.simplify_tolerance,
        weights_path=args.weights,
        decoder_path=args.decoder,
        download_end_embedding=args.download_end_embedding,
        overwrite=args.overwrite,
    )
    print(
        "FLUS-native admin same-grid Paper58 inputs: "
        f"{summary['status']}, "
        f"{summary['n_records']}/{summary['n_requested']} fetched, "
        f"failures={summary['n_failures']}, "
        f"output={args.output_dir}"
    )
    return 0 if summary["n_records"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
