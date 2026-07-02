# -*- coding: utf-8 -*-
"""E6 (RSE reviewer M1 main-force): expand AlphaEarth cache to 30 areas.

Reviewer requested at least 30 area caches to reach 80% power on the
paired inference. This script adds 13 new bounding boxes covering
under-represented biomes (arid, coastal, wetland margin, karst, urbanising)
and runs the AlphaEarth pixel-embedding extraction + independent-change
validation pipeline on them.

Output (results/e6_expanded_areas/):
    extraction_manifest.json     — new area bboxes + n_ok
    expanded_paired_tests.json   — paired stats over the enlarged 23-area set
    expanded_per_area.csv        — persistence / model / advantage for all
                                    areas (v2's 10 + E6's 13)

Usage:
    python e6_expand_areas.py                     # full extraction + eval
    python e6_expand_areas.py --extract-only      # skip eval (for GEE-quota days)
    python e6_expand_areas.py --smoke             # 1 new area, 2020 only

Depends on:
  - experiments/paper8/extract_bishan_embeddings.py (reused for AlphaEarth)
  - v2 AlphaEarth LDN checkpoint at weights/latent_dynamics_v1.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PAPER8_ROOT = REPO_ROOT / "experiments" / "paper8"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PAPER8_ROOT))
from extract_bishan_embeddings import extract_embedding_grid, extract_terrain, init_gee  # noqa: E402
from run_markers import mark_done  # noqa: E402

RESULTS_DIR = HERE / "results" / "e6_expanded_areas"
AE_EVAL_ROOTS = [
    PAPER8_ROOT / "data",
    REPO_ROOT / "data" / "independent_change_labels" / "embeddings",
]

# 13 new bounding boxes (WGS84 lon_min, lat_min, lon_max, lat_max), each
# ~0.1° x 0.1°. Chosen to broaden biome coverage under-represented in v1/v2.
NEW_AREAS = [
    # Arid / semi-arid
    ("tarim_oasis",       [80.0, 41.0, 80.1, 41.1]),
    ("ordos_desert_edge", [109.0, 38.5, 109.1, 38.6]),
    ("gobi_margin",       [103.5, 41.5, 103.6, 41.6]),
    # Coastal / delta
    ("bohai_delta",       [118.0, 39.0, 118.1, 39.1]),
    ("hainan_coastal",    [110.3, 18.5, 110.4, 18.6]),
    # Wetland / lake margin
    ("dongting_wetland",  [112.9, 29.0, 113.0, 29.1]),
    ("erhai_lake_margin", [100.2, 25.7, 100.3, 25.8]),
    # Karst / mountain
    ("guizhou_karst",     [107.0, 26.5, 107.1, 26.6]),
    ("wumeng_mountains",  [104.5, 27.0, 104.6, 27.1]),
    # Urbanising fringes
    ("suzhou_fringe",     [120.6, 31.4, 120.7, 31.5]),
    ("wuhan_outer_ring",  [114.4, 30.5, 114.5, 30.6]),
    # Forest re-growth
    ("qinling_south_slope", [107.5, 33.4, 107.6, 33.5]),
    ("changbai_lower_belt", [128.0, 42.0, 128.1, 42.1]),
]


def study_area_bboxes() -> list[tuple[str, list[float]]]:
    from extract_prithvi_embeddings import all_areas
    return [(area["name"], area["bbox"]) for area in all_areas()]


def extraction_areas(smoke: bool) -> list[tuple[str, list[float]]]:
    if smoke:
        return NEW_AREAS[:1]
    merged: dict[str, list[float]] = {}
    for name, bbox in study_area_bboxes() + NEW_AREAS:
        merged.setdefault(name, bbox)
    return list(merged.items())


def cmd_extract(smoke: bool) -> dict:
    """Run AlphaEarth extraction for each new area."""
    manifest = {"areas": [], "wall_s_extract": 0.0}
    output_dir = PAPER8_ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not init_gee():
        raise SystemExit("GEE init failed")

    areas = extraction_areas(smoke)
    years = list(range(2017, 2025)) if not smoke else [2020]
    t0 = time.time()
    for name, bbox in areas:
        ref_shape = None
        for year in years:
            out_path = output_dir / f"{name}_emb_{year}.npy"
            print(f"  extracting {name} {year}: bbox={bbox}")
            if out_path.exists() and out_path.stat().st_size > 0:
                arr = __import__("numpy").load(out_path, mmap_mode="r")
                ref_shape = arr.shape[:2]
                manifest["areas"].append({"area": name, "year": year,
                                           "returncode": 0, "status": "cached",
                                           "shape": list(arr.shape)})
                continue
            grid = extract_embedding_grid(bbox, year)
            if grid is None:
                manifest["areas"].append({"area": name, "year": year,
                                           "returncode": 1, "status": "failed_no_grid"})
                continue
            __import__("numpy").save(out_path, grid)
            ref_shape = grid.shape[:2]
            manifest["areas"].append({"area": name, "year": year,
                                       "returncode": 0, "status": "generated",
                                       "shape": list(grid.shape)})
        ctx_path = output_dir / f"{name}_context.npy"
        if ref_shape is not None and not ctx_path.exists():
            ctx = extract_terrain(bbox, target_shape=ref_shape)
            __import__("numpy").save(ctx_path, ctx)
    manifest["wall_s_extract"] = time.time() - t0
    failures = [r for r in manifest["areas"] if r.get("returncode") != 0]
    if failures:
        (RESULTS_DIR / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2))
        raise SystemExit(f"E6 extraction failed for {len(failures)} area-year tasks")
    return manifest


def resolve_ckpt() -> Path:
    env_ckpt = os.environ.get("AE_CKPT")
    if env_ckpt:
        env_path = Path(env_ckpt).expanduser()
        if env_path.exists():
            return env_path
        raise FileNotFoundError(f"AE_CKPT points to a missing file: {env_path}")
    candidates = [
        PAPER8_ROOT / "weights" / "latent_dynamics_v1.pt",
        REPO_ROOT / "weights" / "latent_dynamics_v1.pt",
        REPO_ROOT / "src" / "adk_world_model" / "weights" / "latent_dynamics_v1.pt",
        Path.home() / "paper58-r2" / "weights" / "latent_dynamics_v1.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "AlphaEarth LDN checkpoint not found; expected one of:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def discover_eval_area_sources(roots: list[Path] | tuple[Path, ...] = AE_EVAL_ROOTS) -> dict[str, dict]:
    """Find evaluation areas across one or more AlphaEarth cache roots."""
    sources: dict[str, dict] = {}
    for root in roots:
        area_years: dict[str, set[int]] = {}
        for path in root.glob("*_emb_*.npy"):
            area, _, year_text = path.stem.rpartition("_emb_")
            if not area:
                continue
            try:
                year = int(year_text)
            except ValueError:
                continue
            area_years.setdefault(area, set()).add(year)
        for area, year_set in area_years.items():
            years = sorted(year_set)
            if len(years) < 2:
                continue
            if years != list(range(years[0], years[-1] + 1)):
                continue
            current = sources.get(area)
            if current is None or len(years) > len(current["years"]):
                sources[area] = {"root": root, "years": years}
    return sources


def summarize_eval_area_sources(area_sources: dict[str, dict]) -> dict:
    roots: dict[str, int] = {}
    for source in area_sources.values():
        root = str(source["root"])
        roots[root] = roots.get(root, 0) + 1
    return {"n_areas": len(area_sources), "roots": roots}


def cmd_eval() -> dict:
    """Run paired inference on the enlarged AlphaEarth set."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy import stats

    sys.path.insert(0, str(PAPER8_ROOT))
    from paper58_runtime import _build_model, SCENARIO_DIM, SCENARIOS

    # discover all areas including paper8 and independent-change holdout caches
    area_sources = discover_eval_area_sources()
    source_manifest = {
        "summary": summarize_eval_area_sources(area_sources),
        "areas": {
            area: {"root": str(source["root"]), "years": source["years"]}
            for area, source in sorted(area_sources.items())
        },
    }
    (RESULTS_DIR / "eval_area_sources.json").write_text(json.dumps(source_manifest, indent=2))

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    ck = resolve_ckpt()
    print(f"loaded {ck} on {device}")
    ckpt = torch.load(ck, map_location=device, weights_only=False)
    n_ctx = int(ckpt.get("n_context", 2))
    model = _build_model(z_dim=64, n_context=n_ctx).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    scenario_np = np.zeros(SCENARIO_DIM, dtype=np.float32)
    scenario_np[SCENARIOS["baseline"].id] = 1.0
    scenario = torch.tensor(scenario_np).unsqueeze(0).to(device)

    per_area = {}
    for area, source in sorted(area_sources.items()):
        root = source["root"]
        years = source["years"]
        # take last (t, t+1) transition
        z_t = np.load(root / f"{area}_emb_{years[-2]}.npy").astype(np.float32)
        z_true = np.load(root / f"{area}_emb_{years[-1]}.npy").astype(np.float32)
        H = min(z_t.shape[0], z_true.shape[0])
        W = min(z_t.shape[1], z_true.shape[1])
        z_t = torch.tensor(z_t[:H, :W].transpose(2, 0, 1)).unsqueeze(0).to(device)
        z_true = torch.tensor(z_true[:H, :W].transpose(2, 0, 1)).unsqueeze(0).to(device)
        ctx_t = None
        ctx_file = root / f"{area}_context.npy"
        if ctx_file.exists() and n_ctx > 0:
            ctx = np.load(ctx_file).astype(np.float32)
            if ctx.ndim == 3 and ctx.shape[-1] == 2 and ctx.shape[0] != 2:
                ctx = ctx.transpose(2, 0, 1)
            ctx = ctx[:n_ctx, :H, :W]
            ctx_t = torch.tensor(ctx).unsqueeze(0).to(device)
        with torch.no_grad():
            z_t_n = F.normalize(z_t, p=2, dim=1)
            z_true_n = F.normalize(z_true, p=2, dim=1)
            persist = F.cosine_similarity(z_t_n.flatten(2), z_true_n.flatten(2), dim=1).mean().item()
            if ctx_t is not None:
                pred = F.normalize(model(z_t, scenario, context=ctx_t), p=2, dim=1)
            else:
                pred = F.normalize(model(z_t, scenario), p=2, dim=1)
            model_c = F.cosine_similarity(pred.flatten(2), z_true_n.flatten(2), dim=1).mean().item()
        per_area[area] = {"persistence": persist, "model": model_c,
                          "advantage": model_c - persist}
        print(f"  {area}: persist={persist:.4f}  model={model_c:.4f}  "
              f"adv={model_c - persist:+.5f}")

    with (RESULTS_DIR / "expanded_per_area.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["area", "persistence", "model", "advantage"])
        w.writeheader()
        for area, r in per_area.items():
            w.writerow({"area": area, **r})

    # paired inference on the enlarged set
    advs = np.array([r["advantage"] for r in per_area.values()])
    if len(advs) >= 2:
        rng = np.random.default_rng(20260702)
        mean = float(advs.mean())
        sd = float(advs.std(ddof=1))
        wil = stats.wilcoxon(advs, zero_method="wilcox", alternative="two-sided")
        t_stat, t_p = stats.ttest_1samp(advs, popmean=0.0)
        signs = rng.choice([-1, 1], size=(100_000, len(advs)))
        perm_p = float(np.mean(np.abs((signs * advs[None, :]).mean(axis=1)) >= abs(mean)))
        idx = rng.integers(0, len(advs), size=(20_000, len(advs)))
        boot = advs[idx].mean(axis=1)
        ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])
        stats_out = {"n": int(len(advs)), "mean": mean, "sd": sd,
                     "wilcoxon_p": float(wil.pvalue), "t_p": float(t_p),
                     "permutation_p": perm_p,
                     "bootstrap_ci_lo": float(ci_lo),
                     "bootstrap_ci_hi": float(ci_hi),
                     "cohen_dz": mean / sd if sd else float("inf")}
    else:
        stats_out = {"n": int(len(advs))}
    stats_out["eval_sources"] = source_manifest["summary"]
    (RESULTS_DIR / "expanded_paired_tests.json").write_text(json.dumps(stats_out, indent=2))
    return stats_out


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        m = cmd_extract(args.smoke)
        (RESULTS_DIR / "extraction_manifest.json").write_text(json.dumps(m, indent=2))
        print(f"[E6 extract] wall={m['wall_s_extract']/60:.1f} min")
    if not args.extract_only:
        stats_out = cmd_eval()
        print(f"[E6 eval] {json.dumps(stats_out, indent=2)}")
    mark_done(RESULTS_DIR, smoke=args.smoke)
    print("[E6 DONE]")


if __name__ == "__main__":
    main()
