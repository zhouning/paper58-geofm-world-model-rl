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
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PAPER8_ROOT = REPO_ROOT / "experiments" / "paper8"
sys.path.insert(0, str(PAPER8_ROOT))

RESULTS_DIR = HERE / "results" / "e6_expanded_areas"

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


def cmd_extract(smoke: bool) -> dict:
    """Run AlphaEarth extraction for each new area."""
    manifest = {"areas": [], "wall_s_extract": 0.0}
    extract_script = PAPER8_ROOT / "extract_bishan_embeddings.py"
    if not extract_script.exists():
        raise SystemExit(f"missing extract script: {extract_script}")

    areas = NEW_AREAS if not smoke else NEW_AREAS[:1]
    years = list(range(2017, 2025)) if not smoke else [2020]
    t0 = time.time()
    for name, bbox in areas:
        for year in years:
            args = [sys.executable, str(extract_script),
                    "--area", name,
                    "--bbox", ",".join(map(str, bbox)),
                    "--year", str(year),
                    "--out-dir", str(PAPER8_ROOT / "data")]
            print(f"  extracting {name} {year}: {' '.join(args[3:])}")
            r = subprocess.run(args, capture_output=True, text=True)
            manifest["areas"].append({"area": name, "year": year,
                                       "returncode": r.returncode,
                                       "stderr_tail": r.stderr[-500:] if r.stderr else ""})
    manifest["wall_s_extract"] = time.time() - t0
    return manifest


def cmd_eval() -> dict:
    """Run paired inference on the enlarged AlphaEarth set."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy import stats

    sys.path.insert(0, str(PAPER8_ROOT))
    from paper58_runtime import _build_model, SCENARIO_DIM, SCENARIOS

    AE_DIR = PAPER8_ROOT / "data"

    # discover all areas including v1's 10 and E6's new 13
    all_areas = sorted({p.stem.rsplit("_emb_", 1)[0]
                        for p in AE_DIR.glob("*_emb_*.npy")
                        if "_context" not in p.name and "prithvi" not in p.parent.name})

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    ck_candidates = [PAPER8_ROOT / "weights" / "latent_dynamics_v1.pt",
                     Path.home() / "paper58-r2" / "weights" / "latent_dynamics_v1.pt"]
    ck = next((c for c in ck_candidates if c.exists()), None)
    if ck is None:
        return {"error": "AlphaEarth LDN checkpoint not found",
                "expected": [str(c) for c in ck_candidates]}
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
    for area in all_areas:
        files = sorted(AE_DIR.glob(f"{area}_emb_*.npy"))
        if len(files) < 2:
            continue
        years = sorted(int(f.stem.rsplit("_", 1)[1]) for f in files)
        if years != list(range(years[0], years[-1] + 1)):
            continue
        # take last (t, t+1) transition
        z_t = np.load(AE_DIR / f"{area}_emb_{years[-2]}.npy").astype(np.float32)
        z_true = np.load(AE_DIR / f"{area}_emb_{years[-1]}.npy").astype(np.float32)
        H = min(z_t.shape[0], z_true.shape[0])
        W = min(z_t.shape[1], z_true.shape[1])
        z_t = torch.tensor(z_t[:H, :W].transpose(2, 0, 1)).unsqueeze(0).to(device)
        z_true = torch.tensor(z_true[:H, :W].transpose(2, 0, 1)).unsqueeze(0).to(device)
        ctx_t = None
        ctx_file = AE_DIR / f"{area}_context.npy"
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
    (RESULTS_DIR / ".done").touch()
    print("[E6 DONE]")


if __name__ == "__main__":
    main()
