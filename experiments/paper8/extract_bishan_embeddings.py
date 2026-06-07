# -*- coding: utf-8 -*-
"""
Paper 8 Phase 0: Extract Bishan District multi-year embedding grids from GEE.

Extracts AlphaEarth 64-dim embeddings at ~500m resolution for 2017-2024,
plus SRTM terrain context (DEM + slope).

Usage:
    python paper8/extract_bishan_embeddings.py
    python paper8/extract_bishan_embeddings.py --verify  # verify + visualize
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, 'D:/adk/data_agent')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Bishan District bounding box (EPSG:4326)
BISHAN_BBOX = [106.02, 29.38, 106.33, 29.68]
YEARS = list(range(2017, 2025))  # 2017-2024
SCALE = 500  # meters — yields ~60-70 pixel grid


def init_gee():
    """Initialize Google Earth Engine."""
    import ee
    try:
        ee.Initialize()
        print('GEE initialized.')
        return True
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize()
            print('GEE authenticated and initialized.')
            return True
        except Exception as e:
            print(f'GEE init failed: {e}')
            return False


def extract_embedding_grid(bbox, year, scale=SCALE):
    """Extract AlphaEarth embedding grid [H, W, 64] from GEE."""
    import ee

    AEF_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    AEF_BANDS = [f"A{i:02d}" for i in range(64)]

    region = ee.Geometry.Rectangle(bbox)
    img = (
        ee.ImageCollection(AEF_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(region)
        .select(AEF_BANDS)
        .mosaic()
        .clip(region)
    )

    proj = ee.Projection("EPSG:4326").atScale(scale)
    img = img.setDefaultProjection(proj)

    result = img.sampleRectangle(region=region, defaultValue=0).getInfo()
    properties = result.get("properties", {})
    if not properties:
        return None

    arrays = []
    for band in AEF_BANDS:
        band_data = properties.get(band)
        if band_data is None:
            return None
        arrays.append(np.array(band_data, dtype=np.float32))

    return np.stack(arrays, axis=-1)  # [H, W, 64]


def extract_terrain(bbox, target_shape=None, scale=SCALE):
    """Extract DEM elevation + slope [2, H, W] from SRTM via GEE."""
    import ee

    region = ee.Geometry.Rectangle(bbox)
    dem = ee.Image("USGS/SRTMGL1_003").clip(region)
    slope = ee.Terrain.slope(dem)
    combined = dem.select("elevation").addBands(slope.select("slope"))

    proj = ee.Projection("EPSG:4326").atScale(scale)
    combined = combined.setDefaultProjection(proj)

    result = combined.sampleRectangle(region=region, defaultValue=0).getInfo()
    properties = result.get("properties", {})

    elev = np.array(properties["elevation"], dtype=np.float32)
    slp = np.array(properties["slope"], dtype=np.float32)

    # Normalize
    elev_min, elev_max = elev.min(), elev.max()
    if elev_max > elev_min:
        elev = (elev - elev_min) / (elev_max - elev_min)
    else:
        elev = np.zeros_like(elev)
    slp = np.clip(slp / 45.0, 0, 1)

    ctx = np.stack([elev, slp], axis=0)  # [2, H, W]

    if target_shape and (ctx.shape[1] != target_shape[0] or ctx.shape[2] != target_shape[1]):
        from PIL import Image as PILImage
        ctx = np.stack([
            np.array(PILImage.fromarray(ctx[i]).resize(
                (target_shape[1], target_shape[0]), PILImage.BILINEAR
            ), dtype=np.float32)
            for i in range(ctx.shape[0])
        ], axis=0)

    return ctx


def extract_all():
    """Extract embeddings for all years + terrain context."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if not init_gee():
        return False

    ref_shape = None

    for year in YEARS:
        out_path = os.path.join(DATA_DIR, f'bishan_emb_{year}.npy')
        if os.path.exists(out_path):
            arr = np.load(out_path)
            print(f'  {year}: already exists, shape={arr.shape}')
            if ref_shape is None:
                ref_shape = arr.shape[:2]
            continue

        print(f'  Extracting {year}...', end=' ', flush=True)
        t0 = time.time()
        grid = extract_embedding_grid(BISHAN_BBOX, year)
        if grid is None:
            print(f'FAILED')
            continue
        np.save(out_path, grid)
        if ref_shape is None:
            ref_shape = grid.shape[:2]
        print(f'shape={grid.shape}, {time.time()-t0:.1f}s')

    # Terrain context
    ctx_path = os.path.join(DATA_DIR, 'bishan_context.npy')
    if os.path.exists(ctx_path):
        print(f'  Terrain: already exists')
    else:
        print(f'  Extracting terrain...', end=' ', flush=True)
        t0 = time.time()
        ctx = extract_terrain(BISHAN_BBOX, target_shape=ref_shape)
        np.save(ctx_path, ctx)
        print(f'shape={ctx.shape}, {time.time()-t0:.1f}s')

    # Save metadata
    import json
    meta = {
        'bbox': BISHAN_BBOX,
        'years': YEARS,
        'scale_m': SCALE,
        'grid_shape': list(ref_shape) if ref_shape else None,
        'embedding_dim': 64,
        'source': 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL',
    }
    with open(os.path.join(DATA_DIR, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print('Done!')
    return True


def verify():
    """Verify extracted data and decode LULC for 2020."""
    print('\n=== Verification ===')
    for year in YEARS:
        path = os.path.join(DATA_DIR, f'bishan_emb_{year}.npy')
        if os.path.exists(path):
            arr = np.load(path)
            norms = np.linalg.norm(arr, axis=-1)
            print(f'  {year}: shape={arr.shape}, norm range=[{norms.min():.3f}, {norms.max():.3f}]')
        else:
            print(f'  {year}: MISSING')

    ctx_path = os.path.join(DATA_DIR, 'bishan_context.npy')
    if os.path.exists(ctx_path):
        ctx = np.load(ctx_path)
        print(f'  Terrain: shape={ctx.shape}, elev=[{ctx[0].min():.3f},{ctx[0].max():.3f}], slope=[{ctx[1].min():.3f},{ctx[1].max():.3f}]')

    # Decode 2020 LULC
    emb_2020 = os.path.join(DATA_DIR, 'bishan_emb_2020.npy')
    decoder_path = 'D:/adk/data_agent/weights/lulc_decoder_v1.pkl'
    if os.path.exists(emb_2020) and os.path.exists(decoder_path):
        import joblib
        z = np.load(emb_2020)
        H, W = z.shape[:2]
        decoder = joblib.load(decoder_path)
        lulc = decoder.predict(z.reshape(-1, 64)).reshape(H, W)
        unique, counts = np.unique(lulc, return_counts=True)
        LULC_NAMES = {1:"Water",2:"Trees",4:"Grass",5:"Shrub",7:"Cropland",8:"Built",9:"Bare",10:"Snow",11:"Wetland"}
        print(f'\n  2020 LULC decode ({H}x{W}):')
        total = H * W
        for cls, cnt in zip(unique, counts):
            name = LULC_NAMES.get(cls, f'Class{cls}')
            print(f'    {name}: {cnt} pixels ({100*cnt/total:.1f}%)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()

    if args.verify:
        verify()
    else:
        extract_all()
        verify()
