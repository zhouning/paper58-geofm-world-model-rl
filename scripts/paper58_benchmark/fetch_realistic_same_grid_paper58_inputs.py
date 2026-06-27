from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee
import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from scripts.rse_revision.generate_change_validation_predictions import (
    _decode_lulc,
    _load_decoder,
    _load_model,
    _predict_next_embedding,
)
from adk_world_model.world_model import AEF_BANDS, AEF_COLLECTION, DECODER_PATH, LULC_COLLECTION, WEIGHTS_PATH


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_TIF = ROOT / "paper" / "rse_submission_paper58" / "flus_real_datasets" / "gee_dynamicworld_caidian_160m" / "area_000" / "area_000.tif"
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "rse_submission_paper58" / "realistic_same_grid_paper58_inputs_2026-06-27"


@dataclass(frozen=True)
class ReferenceGrid:
    width: int
    height: int
    count: int
    crs: Any
    transform: Any
    bounds: tuple[float, float, float, float]
    valid_mask: np.ndarray


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


def read_reference_grid(sample_tif: Path) -> ReferenceGrid:
    with rasterio.open(sample_tif) as dataset:
        start_label = dataset.read(1)
        return ReferenceGrid(
            width=int(dataset.width),
            height=int(dataset.height),
            count=int(dataset.count),
            crs=dataset.crs,
            transform=dataset.transform,
            bounds=(float(dataset.bounds.left), float(dataset.bounds.bottom), float(dataset.bounds.right), float(dataset.bounds.top)),
            valid_mask=(start_label != 0),
        )


def _region_from_bounds(bounds: tuple[float, float, float, float]) -> ee.Geometry:
    left, bottom, right, top = bounds
    return ee.Geometry.Rectangle([left, bottom, right, top])


def _download_ee_image(image: ee.Image, region: ee.Geometry, scale_m: int, target_dir: Path, stem: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / f"{stem}.zip"
    tif_path = target_dir / f"{stem}.tif"
    if tif_path.exists() and tif_path.stat().st_size > 0:
        return tif_path
    url = image.getDownloadURL(
        {
            "name": stem,
            "region": region.getInfo()["coordinates"],
            "scale": int(scale_m),
            "crs": "EPSG:4326",
            "format": "ZIPPED_GEO_TIFF",
            "filePerBand": False,
        }
    )
    response = requests.get(url, timeout=600)
    if not response.ok:
        raise RuntimeError(f"GEE download failed for {stem}: HTTP {response.status_code}: {response.text[:1000]}")
    zip_path.write_bytes(response.content)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
        tif_members = [name for name in zf.namelist() if name.lower().endswith((".tif", ".tiff"))]
    if not tif_members:
        raise RuntimeError(f"GEE download for {stem} contained no GeoTIFF")
    extracted = target_dir / tif_members[0]
    if extracted != tif_path:
        extracted.replace(tif_path)
    return tif_path


def _alphaearth_image(year: int, region: ee.Geometry, bands: list[str] | None = None) -> ee.Image:
    selected_bands = bands or AEF_BANDS
    return (
        ee.ImageCollection(AEF_COLLECTION)
        .filterDate(f"{int(year)}-01-01", f"{int(year) + 1}-01-01")
        .filterBounds(region)
        .select(selected_bands)
        .mosaic()
        .clip(region)
    )


def _esri_lulc_image(year: int, region: ee.Geometry) -> ee.Image:
    return (
        ee.ImageCollection(LULC_COLLECTION)
        .filterDate(f"{int(year)}-01-01", f"{int(year) + 1}-01-01")
        .filterBounds(region)
        .select(["b1"])
        .mosaic()
        .clip(region)
    )


def _terrain_image(region: ee.Geometry) -> ee.Image:
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(region)
    slope = ee.Terrain.slope(dem).select("slope")
    return dem.rename("elevation").addBands(slope.rename("slope")).clip(region)


def read_aligned_bands(path: Path, reference: ReferenceGrid, resampling: Resampling, dtype: np.dtype) -> np.ndarray:
    with rasterio.open(path) as dataset:
        src_crs = dataset.crs or reference.crs
        destination = np.zeros((dataset.count, reference.height, reference.width), dtype=dtype)
        for band_index in range(dataset.count):
            reproject(
                source=dataset.read(band_index + 1),
                destination=destination[band_index],
                src_transform=dataset.transform,
                src_crs=src_crs,
                dst_transform=reference.transform,
                dst_crs=reference.crs,
                resampling=resampling,
            )
    return destination


def download_alphaearth_aligned(
    year: int,
    region: ee.Geometry,
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
        tif_path = _download_ee_image(_alphaearth_image(year, region, bands=bands), region, scale_m, downloads_dir, stem)
        paths.append(tif_path)
        arrays.append(read_aligned_bands(tif_path, reference, Resampling.bilinear, np.float32))
    stacked = np.concatenate(arrays, axis=0)
    if stacked.shape[0] != len(AEF_BANDS):
        raise RuntimeError(f"expected {len(AEF_BANDS)} AlphaEarth bands, got {stacked.shape[0]}")
    return stacked.transpose(1, 2, 0), paths


def _normalize_context(context_bands: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    elevation = context_bands[0].astype(np.float32, copy=False)
    slope = context_bands[1].astype(np.float32, copy=False)
    valid = np.asarray(valid_mask, dtype=bool)
    if np.any(valid):
        elev_min = float(np.min(elevation[valid]))
        elev_max = float(np.max(elevation[valid]))
    else:
        elev_min = float(np.min(elevation))
        elev_max = float(np.max(elevation))
    if elev_max > elev_min:
        elevation = (elevation - elev_min) / (elev_max - elev_min)
    else:
        elevation = np.zeros_like(elevation, dtype=np.float32)
    slope = np.clip(slope / 45.0, 0.0, 1.0)
    return np.stack([elevation, slope], axis=0).astype(np.float32, copy=False)


def build_realistic_same_grid_inputs(
    sample_tif: Path = DEFAULT_SAMPLE_TIF,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    area: str = "caidian_dynamicworld_160m_realistic",
    start_year: int = 2020,
    end_year: int = 2021,
    scale_m: int = 160,
    weights_path: Path = Path(WEIGHTS_PATH),
    decoder_path: Path = Path(DECODER_PATH),
    alphaearth_chunk_size: int = 16,
) -> dict[str, Any]:
    ee.Initialize()
    output = Path(output_dir)
    downloads = output / "downloads" / area
    labels_dir = output / "labels"
    predictions_dir = output / "predictions"
    embeddings_dir = output / "embeddings"
    context_dir = output / "context"
    for directory in [labels_dir, predictions_dir, embeddings_dir, context_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    reference = read_reference_grid(Path(sample_tif))
    region = _region_from_bounds(reference.bounds)
    emb_start, emb_start_tifs = download_alphaearth_aligned(
        start_year,
        region,
        scale_m,
        downloads,
        area,
        reference,
        chunk_size=alphaearth_chunk_size,
    )
    label_start_tif = _download_ee_image(_esri_lulc_image(start_year, region), region, scale_m, downloads, f"{area}_esri_lulc_{start_year}")
    label_end_tif = _download_ee_image(_esri_lulc_image(end_year, region), region, scale_m, downloads, f"{area}_esri_lulc_{end_year}")
    terrain_tif = _download_ee_image(_terrain_image(region), region, scale_m, downloads, f"{area}_terrain")

    label_start = read_aligned_bands(label_start_tif, reference, Resampling.nearest, np.int32)[0].astype(np.int32, copy=False)
    label_end = read_aligned_bands(label_end_tif, reference, Resampling.nearest, np.int32)[0].astype(np.int32, copy=False)
    context = _normalize_context(read_aligned_bands(terrain_tif, reference, Resampling.bilinear, np.float32), reference.valid_mask)

    label_start = np.where(reference.valid_mask, label_start, 0).astype(np.int32, copy=False)
    label_end = np.where(reference.valid_mask, label_end, 0).astype(np.int32, copy=False)
    emb_start[~reference.valid_mask] = 0.0
    context[:, ~reference.valid_mask] = 0.0

    model = _load_model(Path(weights_path))
    decoder = _load_decoder(Path(decoder_path))
    pred_embedding = _predict_next_embedding(model, emb_start.astype(np.float32, copy=False), context.astype(np.float32, copy=False))
    pred_lulc = _decode_lulc(pred_embedding, decoder)
    pred_lulc = np.where(reference.valid_mask, pred_lulc, 0).astype(np.int32, copy=False)

    emb_path = embeddings_dir / f"{area}_emb_{start_year}.npy"
    context_path = context_dir / f"{area}_context.npy"
    start_path = labels_dir / f"{area}_lulc_{start_year}.npy"
    end_path = labels_dir / f"{area}_lulc_{end_year}.npy"
    pred_path = predictions_dir / f"{area}_lulc_pred_{start_year}_{end_year}.npy"
    np.save(emb_path, emb_start)
    np.save(context_path, context)
    np.save(start_path, label_start)
    np.save(end_path, label_end)
    np.save(pred_path, pred_lulc)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "area": area,
        "sample_tif": Path(sample_tif),
        "source_dynamicworld_grid": {
            "shape": [reference.height, reference.width],
            "bounds": list(reference.bounds),
            "scale_m": int(scale_m),
            "valid_pixels": int(np.count_nonzero(reference.valid_mask)),
        },
        "sources": {
            "alphaearth": AEF_COLLECTION,
            "esri_lulc": LULC_COLLECTION,
            "terrain": "USGS/SRTMGL1_003",
        },
        "years": {"start": int(start_year), "end": int(end_year)},
        "downloads": {
            "alphaearth_start_chunks": emb_start_tifs,
            "alphaearth_chunk_size": int(alphaearth_chunk_size),
            "esri_start": label_start_tif,
            "esri_end": label_end_tif,
            "terrain": terrain_tif,
        },
        "arrays": {
            "embedding_start": emb_path,
            "context": context_path,
            "label_start": start_path,
            "label_end": end_path,
            "paper58_prediction": pred_path,
        },
        "label_classes_start": [int(value) for value in sorted(np.unique(label_start))],
        "label_classes_end": [int(value) for value in sorted(np.unique(label_end))],
        "prediction_classes": [int(value) for value in sorted(np.unique(pred_lulc))],
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch realistic same-grid AlphaEarth/ESRI inputs for full Paper58 comparison.")
    parser.add_argument("--sample-tif", type=Path, default=DEFAULT_SAMPLE_TIF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--area", default="caidian_dynamicworld_160m_realistic")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument("--scale-m", type=int, default=160)
    parser.add_argument("--weights", type=Path, default=Path(WEIGHTS_PATH))
    parser.add_argument("--decoder", type=Path, default=Path(DECODER_PATH))
    parser.add_argument("--alphaearth-chunk-size", type=int, default=16)
    args = parser.parse_args(argv)
    manifest = build_realistic_same_grid_inputs(
        sample_tif=args.sample_tif,
        output_dir=args.output_dir,
        area=args.area,
        start_year=args.start_year,
        end_year=args.end_year,
        scale_m=args.scale_m,
        weights_path=args.weights,
        decoder_path=args.decoder,
        alphaearth_chunk_size=args.alphaearth_chunk_size,
    )
    print(
        "Realistic same-grid Paper58 inputs: "
        f"area={manifest['area']}, "
        f"shape={manifest['source_dynamicworld_grid']['shape']}, "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
