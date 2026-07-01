# -*- coding: utf-8 -*-
"""E1 (RSE reviewer M4): Prithvi-100M **spatial patch/token** extraction.

Rationale — the reviewer specifically flagged that the v2 Prithvi ablation used
global CLS tokens (1, 1, 768) while AlphaEarth used local pixel grids
(H, W, 64). The comparison was apples-to-oranges. This script rebuilds the
Prithvi cache using the *spatial* token grid produced by PrithviBackbone
(return_spatial=True), so per-area-year embeddings are (H_p, W_p, 768) rather
than a single global CLS.

Output convention: `data/prithvi_spatial/{area}_emb_{year}.npy` shape
`(H_p, W_p, 768)`, where H_p ~ H_alphaearth (broadcast from the 16-pixel
patch grid via extract_prithvi_embeddings.untile_features() from the parent
paper8 tree).

Usage:
    # Smoke test (Bishan 2020 only, ~5 min):
    python e1_prithvi_patch_extract.py --areas bishan --years 2020 --smoke

    # Full run (17 areas x 8 years = 136 files, ~4-6 h):
    python e1_prithvi_patch_extract.py --all

    # Verify existing outputs without re-extraction:
    python e1_prithvi_patch_extract.py --verify
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PAPER8_ROOT = REPO_ROOT / "experiments" / "paper8"
sys.path.insert(0, str(PAPER8_ROOT))

# Reuse the parent paper8 extraction utilities (HLS composite, tiling, Prithvi
# forward pass). We only override the CLS path with the spatial path.
from extract_prithvi_embeddings import (  # noqa: E402
    HLS_BANDS_CANDIDATES,
    HLS_COLLECTION,
    GROWING_SEASON,
    PATCH_SIZE,
    TARGET_SCALE_M,
    all_areas,
    init_gee,
    download_hls_composite,
    normalize_hls_bands,
    tile_to_patches,
    untile_features,
    l2_normalize,
    get_prithvi,
)

DATA_DIR = HERE / "data" / "prithvi_spatial"
RESULTS_DIR = HERE / "results" / "e1_prithvi_patch"
LOGS_DIR = HERE / "logs"

YEARS = list(range(2017, 2025))


def encode_with_prithvi_spatial(stack: np.ndarray, batch_size: int = 4) -> np.ndarray:
    """[6, H, W] -> [H_p, W_p, 768] spatial-token L2-normalized grid.

    Unlike the parent paper8 CLS path, this preserves the token grid and
    reshapes it back to a per-pixel-block feature map that mirrors
    AlphaEarth's (H, W, 64) layout.
    """
    import torch

    model, device = get_prithvi()
    tiles, tile_meta = tile_to_patches(stack)
    n = tiles.shape[0]
    feats = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(tiles[i:i + batch_size]).float().to(device)
            # Prithvi returns spatial tokens of shape [B, n_tokens, 768]
            # when return_spatial=True. n_tokens = (patch_side / 16)**2, i.e.
            # for PATCH_SIZE=128 -> 64 tokens per patch.
            tok = model(batch, return_spatial=True)
            feats.append(tok.cpu().numpy())
    feat_tokens = np.concatenate(feats, axis=0)  # [n_patches, n_tokens, 768]
    grid = untile_features(feat_tokens, tile_meta)  # [H, W, 768]
    return l2_normalize(grid)


def extract_area_year(area: dict, year: int, force: bool = False) -> tuple[bool, str]:
    out_path = DATA_DIR / f"{area['name']}_emb_{year}.npy"
    if out_path.exists() and not force:
        existing = np.load(out_path, mmap_mode="r")
        if existing.ndim == 3 and existing.shape[-1] == 768 and existing.shape[0] > 1:
            return True, "cached"
        print(f"  {area['name']} {year}: re-extract (shape {existing.shape})")
        out_path.unlink()

    t0 = time.time()
    hls = download_hls_composite(area["bbox"], year)
    if hls is None:
        return False, "no_hls"
    if hls.shape[1] < 16 or hls.shape[2] < 16:
        return False, "bbox_too_small"
    hls = normalize_hls_bands(hls)
    feats = encode_with_prithvi_spatial(hls)
    np.save(out_path, feats)
    return True, f"shape={feats.shape}, {time.time() - t0:.1f}s"


def cmd_verify():
    files = sorted(DATA_DIR.glob("*_emb_*.npy"))
    per_area = {}
    for f in files:
        area, _, year = f.stem.rpartition("_emb_")
        per_area.setdefault(area, []).append(int(year))
    print(f"prithvi_spatial cache: {len(files)} files across {len(per_area)} areas")
    for area, years in sorted(per_area.items()):
        print(f"  {area}: {sorted(years)}")
    shapes = {}
    for f in files[:5]:
        shapes[f.name] = np.load(f, mmap_mode="r").shape
    print("sample shapes:", shapes)


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--areas", nargs="*", default=None,
                   help="area names; default = all 17 study areas")
    p.add_argument("--years", nargs="*", type=int, default=None,
                   help="years to extract; default = 2017..2024")
    p.add_argument("--all", action="store_true", help="alias for full 17x8 run")
    p.add_argument("--smoke", action="store_true", help="Bishan 2020 only, force re-extract")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--force", action="store_true", help="re-extract even if cached")
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        cmd_verify()
        return

    if args.smoke:
        args.areas = ["bishan"]
        args.years = [2020]
        args.force = True

    areas_all = all_areas()
    if args.areas:
        area_map = {a["name"]: a for a in areas_all}
        try:
            areas = [area_map[a] for a in args.areas]
        except KeyError as e:
            print(f"unknown area: {e!s}")
            print("available areas:", sorted(area_map))
            sys.exit(1)
    else:
        areas = areas_all
    years = args.years or YEARS

    if not init_gee():
        print("GEE init failed; abort")
        sys.exit(1)

    print(f"extracting {len(areas)} areas x {len(years)} years = "
          f"{len(areas) * len(years)} files to {DATA_DIR}")
    t_start = time.time()
    manifest = {"areas": [a["name"] for a in areas], "years": years,
                "target_scale_m": TARGET_SCALE_M, "patch_size": PATCH_SIZE,
                "extraction_mode": "spatial_tokens",
                "hls_collection": HLS_COLLECTION, "results": []}
    n_ok = n_skip = 0
    for area in areas:
        for year in years:
            print(f"  {area['name']} {year}:", end=" ", flush=True)
            ok, msg = extract_area_year(area, year, force=args.force)
            print(msg)
            manifest["results"].append({"area": area["name"], "year": year,
                                        "ok": ok, "msg": msg})
            if ok:
                n_ok += 1
            else:
                n_skip += 1
    wall = time.time() - t_start
    manifest["wall_s"] = wall
    manifest["n_ok"] = n_ok
    manifest["n_skip"] = n_skip

    out_path = RESULTS_DIR / "extraction_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nn_ok={n_ok}  n_skip={n_skip}  wall={wall/60:.1f} min")
    print(f"wrote {out_path}")

    # Mark done if all successful
    if n_skip == 0:
        (RESULTS_DIR / ".done_extract").touch()


if __name__ == "__main__":
    main()
