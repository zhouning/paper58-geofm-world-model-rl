# -*- coding: utf-8 -*-
"""
Paper 8: Extract village-level (Banzhucun) AlphaEarth embeddings at 10m resolution.

Uses band-by-band extraction to bypass GEE sampleRectangle 262144 pixel limit.
64 GEE calls, each extracting 1 band at full 10m resolution.

Usage:
    python paper8/extract_village_embeddings.py
    python paper8/extract_village_embeddings.py --verify
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, 'D:/adk/data_agent')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'village')

# Banzhucun village bbox (WGS84) with ~200m buffer
VILLAGE_BBOX = [106.1100, 29.5799, 106.1647, 29.6124]
YEARS = [2020, 2023]  # key comparison years
SCALE = 10  # native AlphaEarth resolution
AEF_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
N_BANDS = 64


def init_gee():
    import ee
    try:
        ee.Initialize()
        print('GEE initialized.')
        return True
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize()
            return True
        except Exception as e:
            print(f'GEE init failed: {e}')
            return False


def extract_single_band(bbox, year, band_idx, scale=SCALE):
    """Extract one band of AlphaEarth embeddings at full resolution."""
    import ee
    band_name = f"A{band_idx:02d}"
    region = ee.Geometry.Rectangle(bbox)
    img = (
        ee.ImageCollection(AEF_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(region)
        .select([band_name])
        .mosaic()
        .clip(region)
    )
    proj = ee.Projection("EPSG:4326").atScale(scale)
    img = img.setDefaultProjection(proj)

    result = img.sampleRectangle(region=region, defaultValue=0).getInfo()
    props = result.get("properties", {})
    data = props.get(band_name)
    if data is None:
        return None
    return np.array(data, dtype=np.float32)


def extract_village_grid(bbox, year, scale=SCALE):
    """Extract full 64-band embedding grid band-by-band."""
    bands = []
    for i in range(N_BANDS):
        arr = extract_single_band(bbox, year, i, scale)
        if arr is None:
            print(f'    Band {i} failed!')
            return None
        bands.append(arr)
        if (i + 1) % 16 == 0:
            print(f'    {i+1}/{N_BANDS} bands', end='', flush=True)
    print()
    return np.stack(bands, axis=-1)  # [H, W, 64]


def extract_terrain(bbox, target_shape, scale=30):
    """Extract DEM + slope at ~30m, resize to match embedding grid."""
    import ee
    region = ee.Geometry.Rectangle(bbox)
    dem = ee.Image("USGS/SRTMGL1_003").clip(region)
    slope = ee.Terrain.slope(dem)
    combined = dem.select("elevation").addBands(slope.select("slope"))
    proj = ee.Projection("EPSG:4326").atScale(scale)
    combined = combined.setDefaultProjection(proj)

    result = combined.sampleRectangle(region=region, defaultValue=0).getInfo()
    props = result.get("properties", {})
    elev = np.array(props["elevation"], dtype=np.float32)
    slp = np.array(props["slope"], dtype=np.float32)

    # Normalize
    e_min, e_max = elev.min(), elev.max()
    if e_max > e_min:
        elev = (elev - e_min) / (e_max - e_min)
    else:
        elev = np.zeros_like(elev)
    slp = np.clip(slp / 45.0, 0, 1)
    ctx = np.stack([elev, slp], axis=0)  # [2, H, W]

    # Resize to match embedding grid
    if ctx.shape[1] != target_shape[0] or ctx.shape[2] != target_shape[1]:
        from PIL import Image as PILImage
        ctx = np.stack([
            np.array(PILImage.fromarray(ctx[i]).resize(
                (target_shape[1], target_shape[0]), PILImage.BILINEAR
            ), dtype=np.float32)
            for i in range(2)
        ], axis=0)
    return ctx


def extract_all():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not init_gee():
        return False

    ref_shape = None
    for year in YEARS:
        out_path = os.path.join(DATA_DIR, f'village_emb_{year}.npy')
        if os.path.exists(out_path):
            arr = np.load(out_path)
            print(f'  {year}: exists, shape={arr.shape}')
            if ref_shape is None:
                ref_shape = arr.shape[:2]
            continue

        print(f'  Extracting {year} (64 bands at 10m)...')
        t0 = time.time()
        grid = extract_village_grid(VILLAGE_BBOX, year)
        if grid is None:
            print(f'  {year}: FAILED')
            continue
        np.save(out_path, grid)
        if ref_shape is None:
            ref_shape = grid.shape[:2]
        print(f'  {year}: shape={grid.shape}, {time.time()-t0:.1f}s')

    # Terrain
    ctx_path = os.path.join(DATA_DIR, 'village_context.npy')
    if os.path.exists(ctx_path):
        print(f'  Terrain: exists')
    else:
        print(f'  Extracting terrain...')
        ctx = extract_terrain(VILLAGE_BBOX, ref_shape)
        np.save(ctx_path, ctx)
        print(f'  Terrain: shape={ctx.shape}')

    # Metadata
    import json
    meta = {
        'bbox': VILLAGE_BBOX,
        'years': YEARS,
        'scale_m': SCALE,
        'grid_shape': list(ref_shape) if ref_shape else None,
        'embedding_dim': 64,
        'village': 'Banzhucun (斑竹村)',
        'source': AEF_COLLECTION,
        'method': 'band_by_band_sampleRectangle',
    }
    with open(os.path.join(DATA_DIR, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('Done!')
    return True


def verify():
    print('\n=== Village Embedding Verification ===')
    for year in YEARS:
        path = os.path.join(DATA_DIR, f'village_emb_{year}.npy')
        if os.path.exists(path):
            arr = np.load(path)
            norms = np.linalg.norm(arr, axis=-1)
            print(f'  {year}: shape={arr.shape}, norm=[{norms.min():.3f}, {norms.max():.3f}]')
        else:
            print(f'  {year}: MISSING')

    ctx_path = os.path.join(DATA_DIR, 'village_context.npy')
    if os.path.exists(ctx_path):
        ctx = np.load(ctx_path)
        print(f'  Terrain: shape={ctx.shape}')

    # Decode LULC
    emb_path = os.path.join(DATA_DIR, 'village_emb_2020.npy')
    decoder_path = 'D:/adk/data_agent/weights/lulc_decoder_v1.pkl'
    if os.path.exists(emb_path) and os.path.exists(decoder_path):
        import joblib
        z = np.load(emb_path)
        H, W = z.shape[:2]
        decoder = joblib.load(decoder_path)
        lulc = decoder.predict(z.reshape(-1, 64)).reshape(H, W)
        unique, counts = np.unique(lulc, return_counts=True)
        NAMES = {1:"Water",2:"Trees",4:"Grass",5:"Shrub",7:"Cropland",8:"Built",9:"Bare"}
        total = H * W
        print(f'\n  2020 LULC ({H}x{W} = {total} pixels, 10m resolution):')
        for cls, cnt in zip(unique, counts):
            print(f'    {NAMES.get(cls, f"Class{cls}")}: {cnt} ({100*cnt/total:.1f}%)')

        # Compare with Paper 1 data
        print(f'\n  Paper 1 comparison:')
        print(f'    Paper 1: 10,653 grid cells (1 mu/cell, ~25m)')
        print(f'    Paper 8: {total} pixels (10m resolution)')
        print(f'    Resolution ratio: ~{total/10653:.1f}x more pixels')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()

    if args.verify:
        verify()
    else:
        extract_all()
        verify()
