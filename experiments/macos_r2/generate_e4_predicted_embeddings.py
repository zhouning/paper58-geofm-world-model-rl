# -*- coding: utf-8 -*-
"""Generate predicted embeddings for E4 per-year decoder evaluation.

E4 requires AlphaEarth LDN predicted embeddings for all independent-change pairs.
This script:
1. Loads the AlphaEarth LDN checkpoint
2. For each (area, start_year, end_year) pair, loads the start_year embedding
3. Runs the model to predict end_year embedding
4. Saves to data/independent_change_labels/predicted/{area}_{start}_{end}_embedding.npy

Usage:
    python generate_e4_predicted_embeddings.py
    python generate_e4_predicted_embeddings.py --smoke  # test on 3 pairs only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PAPER8_ROOT = REPO_ROOT / "experiments" / "paper8"
sys.path.insert(0, str(PAPER8_ROOT))

from paper58_runtime import _build_model, SCENARIO_DIM, SCENARIOS

INDEP_LABELS = REPO_ROOT / "data" / "independent_change_labels" / "labels"
INDEP_PRED = REPO_ROOT / "data" / "independent_change_labels" / "predicted"
AE_DIRS = [
    REPO_ROOT / "data" / "independent_change_labels" / "embeddings",
    PAPER8_ROOT / "data",
]


def resolve_ckpt() -> Path:
    """Find AlphaEarth LDN checkpoint."""
    candidates = [
        PAPER8_ROOT / "weights" / "latent_dynamics_v1.pt",
        REPO_ROOT / "weights" / "latent_dynamics_v1.pt",
        REPO_ROOT / "src" / "adk_world_model" / "weights" / "latent_dynamics_v1.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"AlphaEarth LDN checkpoint not found in: {candidates}")


def find_start_embedding(area: str, start_year: int) -> Path | None:
    """Find start_year embedding for area across all AE_DIRS.

    Also checks subdirectory {area}/{area}_emb_{year}.npy for cases like heping/.
    """
    for root in AE_DIRS:
        # Direct path
        p = root / f"{area}_emb_{start_year}.npy"
        if p.exists():
            return p
        # Subdirectory path (e.g., heping/heping_emb_2017.npy)
        p2 = root / area / f"{area}_emb_{start_year}.npy"
        if p2.exists():
            return p2
    return None


def discover_pairs() -> list[dict]:
    """Discover all (area, start_year, end_year) pairs from labels directory."""
    pairs = []
    for label_file in INDEP_LABELS.glob("*_lulc_*.npy"):
        stem = label_file.stem
        # Expected pattern: {area}_lulc_{year}.npy
        if not stem.endswith(tuple(str(y) for y in range(2017, 2025))):
            continue
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        area_lulc, year_str = parts
        area = area_lulc.replace("_lulc", "")
        try:
            year = int(year_str)
        except ValueError:
            continue
        # Look for consecutive year labels to form pairs
        next_year = year + 1
        next_file = INDEP_LABELS / f"{area}_lulc_{next_year}.npy"
        if next_file.exists():
            pairs.append({"area": area, "start_year": year, "end_year": next_year})
    # Deduplicate
    seen = set()
    unique = []
    for p in pairs:
        key = (p["area"], p["start_year"], p["end_year"])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return sorted(unique, key=lambda x: (x["area"], x["start_year"]))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--smoke", action="store_true", help="Test on 3 pairs only")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_ckpt()
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_ctx = int(ckpt.get("n_context", 2))
    z_dim = int(ckpt.get("z_dim", 64))
    model = _build_model(z_dim=z_dim, n_context=n_ctx).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    print(f"Model loaded: z_dim={z_dim}, n_context={n_ctx}, device={device}")

    scenario_np = np.zeros(SCENARIO_DIM, dtype=np.float32)
    scenario_np[SCENARIOS["baseline"].id] = 1.0
    scenario = torch.tensor(scenario_np).unsqueeze(0).to(device)

    pairs = discover_pairs()
    if args.smoke:
        pairs = pairs[:3]
    print(f"Discovered {len(pairs)} pairs")

    INDEP_PRED.mkdir(parents=True, exist_ok=True)
    manifest = {"n_pairs": len(pairs), "results": []}

    for i, pair in enumerate(pairs, 1):
        area = pair["area"]
        start_year = pair["start_year"]
        end_year = pair["end_year"]
        out_file = INDEP_PRED / f"{area}_{start_year}_{end_year}_embedding.npy"

        if out_file.exists():
            print(f"[{i}/{len(pairs)}] SKIP {area} {start_year}→{end_year} (exists)")
            manifest["results"].append({
                "area": area, "start_year": start_year, "end_year": end_year,
                "status": "cached"
            })
            continue

        start_emb_path = find_start_embedding(area, start_year)
        if start_emb_path is None:
            print(f"[{i}/{len(pairs)}] SKIP {area} {start_year}→{end_year} (no start embedding)")
            manifest["results"].append({
                "area": area, "start_year": start_year, "end_year": end_year,
                "status": "missing_start_embedding"
            })
            continue

        # Load start embedding
        z_start = np.load(start_emb_path).astype(np.float32)  # (H, W, 64)
        if z_start.shape[-1] != z_dim:
            print(f"[{i}/{len(pairs)}] SKIP {area} {start_year}→{end_year} (z_dim mismatch: {z_start.shape[-1]} != {z_dim})")
            manifest["results"].append({
                "area": area, "start_year": start_year, "end_year": end_year,
                "status": "z_dim_mismatch"
            })
            continue

        # Predict next year embedding (no terrain context for independent-change pairs)
        z_t = torch.tensor(z_start.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            z_pred = F.normalize(model(z_t, scenario), p=2, dim=1)
        pred_np = z_pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # back to (H, W, 64)

        np.save(out_file, pred_np)
        print(f"[{i}/{len(pairs)}] DONE {area} {start_year}→{end_year} → {out_file.name}")
        manifest["results"].append({
            "area": area, "start_year": start_year, "end_year": end_year,
            "status": "generated", "shape": list(pred_np.shape)
        })

    manifest_file = INDEP_PRED / "generation_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"\nGeneration complete. Manifest: {manifest_file}")
    print(f"Generated: {sum(1 for r in manifest['results'] if r['status'] == 'generated')}")
    print(f"Cached: {sum(1 for r in manifest['results'] if r['status'] == 'cached')}")
    print(f"Missing: {sum(1 for r in manifest['results'] if r['status'] == 'missing_start_embedding')}")


if __name__ == "__main__":
    main()
