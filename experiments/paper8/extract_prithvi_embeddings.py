# -*- coding: utf-8 -*-
"""Paper 5+8 × Paper 12 Ablation — Phase A: Prithvi-100M HLS embedding extraction.

For each (study area, year) pair, downloads a median HLS composite (6 bands:
Blue, Green, Red, NIR-Narrow, SWIR1, SWIR2) over the growing season from GEE,
runs it through a frozen Prithvi-100M encoder, and saves L2-normalized
[H, W, 768] spatial-token features as a drop-in replacement for the
AlphaEarth [H, W, 64] cached arrays.

Output convention mirrors extract_bishan_embeddings.py:
    data/prithvi/{area}_emb_{year}.npy   shape (H, W, 768)
    data/prithvi/metadata.json

Usage:
    # Smoke test (Bishan 2020 only):
    python paper8/extract_prithvi_embeddings.py --areas bishan --years 2020

    # All 17 areas × 8 years:
    python paper8/extract_prithvi_embeddings.py --all

    # Verify existing outputs:
    python paper8/extract_prithvi_embeddings.py --verify
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

PAPER8_ROOT = Path(__file__).resolve().parent
ALPHAEARTH_REPO = Path(os.environ.get("ALPHAEARTH_REPO", "D:/adk/AlphaEarth-System"))
DATA_AGENT_DIR = Path(os.environ.get("DATA_AGENT_DIR", "D:/adk/data_agent"))
sys.path.insert(0, str(ALPHAEARTH_REPO))
if DATA_AGENT_DIR.exists():
    sys.path.insert(0, str(DATA_AGENT_DIR))

DATA_DIR = PAPER8_ROOT / "data" / "prithvi"
PRITHVI_WEIGHTS = Path(os.environ.get(
    "PRITHVI_WEIGHTS",
    str(ALPHAEARTH_REPO / "data" / "weights" / "prithvi" / "Prithvi_100M.pt"),
))

YEARS = list(range(2017, 2025))
HLS_COLLECTION = "NASA/HLS/HLSL30/v002"
# HLSL30 (Landsat-derived) bands. Prithvi-100M was pretrained on these 6.
# Two naming conventions exist in the wild — auto-detect at runtime.
HLS_BANDS_CANDIDATES = [
    ["B2", "B3", "B4", "B5", "B6", "B7"],     # HLSL30 native (Landsat-style, current)
    ["B02", "B03", "B04", "B05", "B06", "B07"],  # HLSS30 / older HLSL30 builds
]
GROWING_SEASON = (5, 9)
PATCH_SIZE = 128
TARGET_SCALE_M = 500


_BANDS_CACHE: list[str] | None = None


def detect_hls_bands() -> list[str]:
    """Probe one image from the collection to determine the band naming convention."""
    global _BANDS_CACHE
    if _BANDS_CACHE is not None:
        return _BANDS_CACHE
    import ee
    first = ee.ImageCollection(HLS_COLLECTION).limit(1).first()
    available = set(first.bandNames().getInfo())
    for candidate in HLS_BANDS_CANDIDATES:
        if all(b in available for b in candidate):
            _BANDS_CACHE = candidate
            print(f"  HLS band convention: {candidate}")
            return candidate
    raise RuntimeError(
        f"None of the band candidates {HLS_BANDS_CANDIDATES} match available "
        f"bands in {HLS_COLLECTION}: {sorted(available)}"
    )

# Replicated inline from data_agent/world_model.py DEFAULT_TRAINING_AREAS so this
# script runs standalone on Colab without the GIS Data Agent project on PYTHONPATH.
_FALLBACK_TRAINING_AREAS = [
    {"name": "yangtze_delta",   "bbox": [121.2, 31.0, 121.3, 31.1]},
    {"name": "jing_jin_ji",     "bbox": [116.3, 39.8, 116.4, 39.9]},
    {"name": "pearl_river",     "bbox": [113.2, 23.0, 113.3, 23.1]},
    {"name": "chengdu_plain",   "bbox": [104.0, 30.6, 104.1, 30.7]},
    {"name": "northeast_plain", "bbox": [126.5, 45.7, 126.6, 45.8]},
    {"name": "north_china_plain","bbox":[115.0, 36.5, 115.1, 36.6]},
    {"name": "jianghan_plain",  "bbox": [113.5, 30.3, 113.6, 30.4]},
    {"name": "hetao",           "bbox": [107.0, 40.7, 107.1, 40.8]},
    {"name": "yunnan_eco",      "bbox": [100.2, 25.0, 100.3, 25.1]},
    {"name": "daxinganling",    "bbox": [124.0, 50.3, 124.1, 50.4]},
    {"name": "qinghai_edge",    "bbox": [101.5, 36.5, 101.6, 36.6]},
    {"name": "wuyi_mountain",   "bbox": [117.6, 27.7, 117.7, 27.8]},
    {"name": "guanzhong",       "bbox": [108.9, 34.2, 109.0, 34.3]},
    {"name": "minnan_coast",    "bbox": [118.0, 24.4, 118.1, 24.5]},
    {"name": "poyang_lake",     "bbox": [116.0, 29.0, 116.1, 29.1]},
]

EXTRA_AREAS = [
    {"name": "bishan",    "bbox": [106.02, 29.38, 106.33, 29.68]},
    {"name": "banzhucun", "bbox": [106.11, 29.5799, 106.1647, 29.6124]},
]


def all_areas() -> list[dict]:
    try:
        from world_model import DEFAULT_TRAINING_AREAS
        return list(DEFAULT_TRAINING_AREAS) + EXTRA_AREAS
    except Exception:
        return list(_FALLBACK_TRAINING_AREAS) + EXTRA_AREAS


def init_gee() -> bool:
    import ee
    project = os.environ.get("EE_PROJECT")
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        return True
    except Exception:
        try:
            ee.Authenticate()
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
            return True
        except Exception as e:
            print(f"GEE init failed: {e}")
            print("Set environment variable EE_PROJECT to your registered Earth Engine project ID.")
            return False


def download_hls_composite(bbox: Sequence[float], year: int) -> np.ndarray | None:
    """Median HLS composite for a year's growing season -> [6, H, W] uint16/float.

    Masks clouds/shadows using the HLSL30 Fmask band (bits 1=cloud, 3=cloud shadow,
    4=snow). Then takes the median over the season to get a clean composite.
    """
    import ee

    bands = detect_hls_bands()
    start = f"{year}-{GROWING_SEASON[0]:02d}-01"
    end = f"{year}-{GROWING_SEASON[1]:02d}-30"
    region = ee.Geometry.Rectangle(bbox)

    def mask_clouds(img):
        fmask = img.select("Fmask")
        cloud = fmask.bitwiseAnd(1 << 1).neq(0)
        shadow = fmask.bitwiseAnd(1 << 3).neq(0)
        snow = fmask.bitwiseAnd(1 << 4).neq(0)
        clear = cloud.Or(shadow).Or(snow).Not()
        return img.updateMask(clear)

    coll = (
        ee.ImageCollection(HLS_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
        .map(mask_clouds)
        .select(bands)
    )
    n = coll.size().getInfo()
    if n == 0:
        return None

    img = coll.median().clip(region).reproject(crs="EPSG:4326", scale=TARGET_SCALE_M)

    result = img.sampleRectangle(region=region, defaultValue=0).getInfo()
    props = result.get("properties", {})
    if not props:
        return None

    arrs = []
    for b in bands:
        data = props.get(b)
        if data is None:
            return None
        arrs.append(np.array(data, dtype=np.float32))
    stack = np.stack(arrs, axis=0)
    return stack


def normalize_hls_bands(stack: np.ndarray) -> np.ndarray:
    """HLS reflectance is scaled by 10000; clip to [0, 1]."""
    out = stack.astype(np.float32) / 10000.0
    return np.clip(out, 0.0, 1.0)


def tile_to_patches(stack: np.ndarray, patch: int = PATCH_SIZE) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """[6, H, W] -> [n_patches, 6, patch, patch] with reflection padding.

    Returns the patch tensor plus (H, W, ph, pw) tile counts so we can untile.
    """
    C, H, W = stack.shape
    ph = (H + patch - 1) // patch
    pw = (W + patch - 1) // patch
    pad_h = ph * patch - H
    pad_w = pw * patch - W
    padded = np.pad(stack, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    tiles = padded.reshape(C, ph, patch, pw, patch).transpose(1, 3, 0, 2, 4).reshape(ph * pw, C, patch, patch)
    return tiles, (H, W, ph, pw)


def untile_features(feat_tokens: np.ndarray, tile_meta: tuple[int, int, int, int]) -> np.ndarray:
    """[n_patches, n_tokens, 768] -> [H, W, 768] pixel grid.

    Prithvi outputs one 768-d token per 16×16 input pixel block (patch_size=16,
    so a 128×128 patch yields 8×8 = 64 tokens). To match AlphaEarth's per-pixel
    embedding grid of shape (H, W, 64), we broadcast each token's feature to
    its 16×16 pixel block via np.repeat. Each block thus shares an identical
    768-d feature vector — this is by design: Prithvi is a patch-level encoder
    and we acknowledge this granularity gap in the paper.
    """
    H, W, ph, pw = tile_meta
    n_patches, n_tokens, D = feat_tokens.shape
    side = int(np.sqrt(n_tokens))
    assert side * side == n_tokens, f"non-square token grid: {n_tokens}"

    # Token grid covers the padded (ph*PATCH_SIZE, pw*PATCH_SIZE) input.
    grid = (
        feat_tokens.reshape(ph, pw, side, side, D)
        .transpose(0, 2, 1, 3, 4)
        .reshape(ph * side, pw * side, D)
    )
    # Broadcast each token to its 16×16 pixel block, then crop to original H×W.
    repeat = PATCH_SIZE // side  # = 16
    pixel_grid = np.repeat(np.repeat(grid, repeat, axis=0), repeat, axis=1)
    return pixel_grid[:H, :W, :].astype(np.float32)


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return arr / norms


_PRITHVI_CACHE = {"model": None, "device": None}


def get_prithvi():
    if _PRITHVI_CACHE["model"] is None:
        import torch
        from geoadapter.models.prithvi import PrithviBackbone

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = str(PRITHVI_WEIGHTS) if PRITHVI_WEIGHTS.exists() else None
        model = PrithviBackbone(pretrained=ckpt is not None, checkpoint_path=ckpt).to(device).eval()
        _PRITHVI_CACHE["model"] = model
        _PRITHVI_CACHE["device"] = device
        print(f"Prithvi loaded on {device}, weights={'pretrained' if ckpt else 'random'}")
    return _PRITHVI_CACHE["model"], _PRITHVI_CACHE["device"]


def encode_with_prithvi(stack: np.ndarray, batch_size: int = 8) -> np.ndarray:
    """[6, H, W] HLS reflectance -> [1, 1, 768] L2-normalized Prithvi CLS feature.

    For Paper 5+8 ablation we mirror AlphaEarth's per-area sampling protocol,
    which produces (1, 1, 64) — a single global embedding per area-year
    (paper 5+8 default 15 study areas all at this resolution; only Bishan and
    Banzhucun get a spatial grid). To make Prithvi an apples-to-apples drop-in,
    we take the global CLS token (1, 1, 768) rather than the spatial token grid.
    """
    import torch

    model, device = get_prithvi()
    tiles, _ = tile_to_patches(stack)
    n = tiles.shape[0]
    cls_feats = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(tiles[i:i + batch_size]).float().to(device)
            cls = model(batch, return_spatial=False)  # [B, 768] global CLS
            cls_feats.append(cls.cpu().numpy())
    arr = np.concatenate(cls_feats, axis=0)
    pooled = arr.mean(axis=0, keepdims=True)  # [1, 768] mean over tiles if multiple
    pooled = pooled.reshape(1, 1, -1)
    return l2_normalize(pooled)


def extract_area_year(area: dict, year: int, force: bool = False) -> bool:
    out_path = DATA_DIR / f"{area['name']}_emb_{year}.npy"
    if out_path.exists() and not force:
        # Existing files are valid since we now always emit (1, 1, 768) CLS features
        # — matches AlphaEarth's (1, 1, 64) per-area sampling protocol from paper 5+8.
        existing = np.load(out_path, mmap_mode="r")
        if existing.shape[-1] == 768:
            return True
        print(f"  {area['name']} {year}: re-extracting (wrong dim {existing.shape})")
        out_path.unlink()

    print(f"  {area['name']} {year}:", end=" ", flush=True)
    t0 = time.time()
    hls = download_hls_composite(area["bbox"], year)
    if hls is None:
        print("NO HLS DATA (cloud / out of range)")
        return False
    if hls.shape[1] < 1 or hls.shape[2] < 1:
        print(f"BBOX TOO SMALL at {TARGET_SCALE_M}m scale (got {hls.shape}); skip area")
        return False
    hls = normalize_hls_bands(hls)
    feats = encode_with_prithvi(hls)
    np.save(out_path, feats)
    print(f"shape={feats.shape}, {time.time() - t0:.1f}s")
    return True


def extract_all(area_filter: list[str] | None, year_filter: list[int] | None, force: bool) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not init_gee():
        return

    areas = all_areas()
    if area_filter:
        areas = [a for a in areas if a["name"] in set(area_filter)]
    years = year_filter or YEARS

    print(f"Extracting Prithvi features: {len(areas)} areas × {len(years)} years")
    successes = 0
    failures = []
    for area in areas:
        for year in years:
            ok = extract_area_year(area, year, force=force)
            if ok:
                successes += 1
            else:
                failures.append(f"{area['name']}/{year}")

    meta = {
        "areas": [a["name"] for a in areas],
        "years": years,
        "scale_m": TARGET_SCALE_M,
        "patch_size": PATCH_SIZE,
        "embed_dim": 768,
        "source": HLS_COLLECTION,
        "season": GROWING_SEASON,
        "bands": _BANDS_CACHE or HLS_BANDS_CANDIDATES[0],
        "prithvi_weights": str(PRITHVI_WEIGHTS),
        "l2_normalized": True,
    }
    (DATA_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone: {successes} succeeded, {len(failures)} failed")
    if failures:
        print("Failed:", ", ".join(failures))


def verify() -> None:
    print("\n=== Verification ===")
    for p in sorted(DATA_DIR.glob("*_emb_*.npy")):
        arr = np.load(p)
        norms = np.linalg.norm(arr, axis=-1)
        print(
            f"  {p.name}: shape={arr.shape}, "
            f"norm range=[{norms.min():.4f}, {norms.max():.4f}], "
            f"L2-normed={'YES' if abs(norms.mean() - 1.0) < 1e-3 else 'NO'}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--areas", nargs="*", help="Subset of area names; default = all")
    p.add_argument("--years", nargs="*", type=int, help="Subset of years; default = 2017-2024")
    p.add_argument("--all", action="store_true", help="Extract everything (equivalent to no --areas/--years)")
    p.add_argument("--force", action="store_true", help="Re-extract even if .npy exists")
    p.add_argument("--verify", action="store_true", help="Only verify existing outputs")
    args = p.parse_args()

    if args.verify:
        verify()
    else:
        extract_all(args.areas, args.years, args.force)
        verify()
