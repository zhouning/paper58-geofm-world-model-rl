# -*- coding: utf-8 -*-
"""E3 (RSE reviewer S5): 6-step autoregressive rollout on all valid cached areas.

Reviewer flagged: v2 Table 8 lists 6-step advantage for only 3 areas
(Yangtze Delta, Jing-Jin-Ji, Chengdu Plain). The remaining 7 valid cached
areas were listed as ``1-step only recorded''. This is selective reporting.

Fix: load the existing AlphaEarth-side LDN checkpoint (or refit-and-eval if
missing), unroll 6 autoregressive steps for every area that has at least
7 consecutive annual embeddings (2017-2023 covers 6 transitions), and emit a
uniform per-area x per-step advantage table.

Output: results/e3_multistep/multistep_all_areas.csv columns
    area, step (1..6), persistence_cossim, model_cossim, advantage

Usage:
    python e3_multistep_all_areas.py
    python e3_multistep_all_areas.py --smoke      # Bishan only, step 1..3

Depends on paper8/data/*.npy (AlphaEarth cache).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PAPER8_ROOT = REPO_ROOT / "experiments" / "paper8"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PAPER8_ROOT))

from run_markers import mark_done  # noqa: E402
from paper58_runtime import _build_model, SCENARIO_DIM, SCENARIOS  # noqa: E402

AE_DIR = PAPER8_ROOT / "data"
AE_CKPT_CANDIDATES = [
    PAPER8_ROOT / "weights" / "latent_dynamics_v1.pt",
    REPO_ROOT / "weights" / "latent_dynamics_v1.pt",
    REPO_ROOT / "src" / "adk_world_model" / "weights" / "latent_dynamics_v1.pt",
    Path.home() / "paper58-r2" / "weights" / "latent_dynamics_v1.pt",
]

RESULTS_DIR = HERE / "results" / "e3_multistep"
Z_DIM = 64


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_ckpt() -> Path:
    env_ckpt = os.environ.get("AE_CKPT")
    if env_ckpt:
        env_path = Path(env_ckpt).expanduser()
        if env_path.exists():
            return env_path
        raise FileNotFoundError(f"AE_CKPT points to a missing file: {env_path}")
    for c in AE_CKPT_CANDIDATES:
        if c.exists():
            return c
    raise FileNotFoundError(
        "AlphaEarth LDN checkpoint not found; expected one of:\n  "
        + "\n  ".join(str(c) for c in AE_CKPT_CANDIDATES)
        + "\nSet AE_CKPT to point at a valid latent_dynamics_v1.pt")


def load_area(area: str) -> tuple[list[np.ndarray] | None, np.ndarray | None]:
    files = sorted(AE_DIR.glob(f"{area}_emb_*.npy"))
    if not files:
        return None, None
    years = sorted((int(f.stem.rsplit("_", 1)[1]), f) for f in files)
    seq_years = [y for y, _ in years]
    if seq_years != list(range(seq_years[0], seq_years[-1] + 1)):
        return None, None
    arrs = []
    H_min = W_min = None
    for _, f in years:
        a = np.load(f).astype(np.float32)
        if H_min is None:
            H_min, W_min = a.shape[0], a.shape[1]
        H_min = min(H_min, a.shape[0])
        W_min = min(W_min, a.shape[1])
        arrs.append(a)
    seq = [a[:H_min, :W_min].transpose(2, 0, 1) for a in arrs]
    ctx = None
    ctx_file = AE_DIR / f"{area}_context.npy"
    if ctx_file.exists():
        ctx = np.load(ctx_file).astype(np.float32)
        if ctx.ndim == 3 and ctx.shape[-1] == 2 and ctx.shape[0] != 2:
            ctx = ctx.transpose(2, 0, 1)
        ctx = ctx[:, :H_min, :W_min]
    return seq, ctx


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.flatten(2), b.flatten(2), dim=1).mean()


def list_areas() -> list[str]:
    return sorted({p.stem.rsplit("_emb_", 1)[0] for p in AE_DIR.glob("*_emb_*.npy")})


def rollout_for_area(model, device, seq: list[np.ndarray],
                     ctx: np.ndarray | None, n_ctx_ckpt: int, max_steps: int) -> list[dict]:
    """Return rows: [{step, persistence, model, advantage}, ...] up to max_steps."""
    scenario_np = np.zeros(SCENARIO_DIM, dtype=np.float32)
    scenario_np[SCENARIOS["baseline"].id] = 1.0
    scenario = torch.tensor(scenario_np).unsqueeze(0).to(device)

    z0 = torch.tensor(seq[0]).unsqueeze(0).to(device)
    # zero-pad or trim ctx to match ckpt n_context
    if n_ctx_ckpt == 0:
        ctx_t = None
    else:
        if ctx is None:
            # zero context; keeps shape compatibility
            _, H, W = seq[0].shape
            ctx_t = torch.zeros(1, n_ctx_ckpt, H, W, device=device)
        else:
            ctx_arr = ctx[:n_ctx_ckpt]
            ctx_t = torch.tensor(ctx_arr).unsqueeze(0).to(device)

    z_pred = z0
    rows = []
    with torch.no_grad():
        for step in range(1, max_steps + 1):
            if step >= len(seq):
                break
            if ctx_t is not None:
                z_pred = F.normalize(model(z_pred, scenario, context=ctx_t), p=2, dim=1)
            else:
                z_pred = F.normalize(model(z_pred, scenario), p=2, dim=1)
            z_true = F.normalize(torch.tensor(seq[step]).unsqueeze(0).to(device),
                                 p=2, dim=1)
            persist = cosine(F.normalize(z0, p=2, dim=1), z_true).item()
            model_c = cosine(z_pred, z_true).item()
            rows.append({"step": step, "persistence": persist,
                         "model": model_c, "advantage": model_c - persist})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.max_steps = 3

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    ckpt_path = resolve_ckpt()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_ctx = int(ckpt.get("n_context", 2))
    z_dim = int(ckpt.get("z_dim", Z_DIM))
    model = _build_model(z_dim=z_dim, n_context=n_ctx).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    print(f"loaded ckpt {ckpt_path} (n_context={n_ctx}, z_dim={z_dim}) on {device}")

    areas_all = list_areas()
    if args.smoke:
        areas_all = ["bishan"] if "bishan" in areas_all else areas_all[:1]

    all_rows = []
    t0 = time.time()
    for area in areas_all:
        seq, ctx = load_area(area)
        if seq is None:
            print(f"  SKIP {area} (incomplete)")
            continue
        rows = rollout_for_area(model, device, seq, ctx, n_ctx, args.max_steps)
        for r in rows:
            r["area"] = area
            all_rows.append(r)
        adv_str = ", ".join(f"{r['step']}:{r['advantage']:+.4f}" for r in rows)
        print(f"  {area}: {adv_str}")

    with (RESULTS_DIR / "multistep_all_areas.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["area", "step", "persistence", "model", "advantage"])
        w.writeheader()
        w.writerows(all_rows)

    manifest = {"ckpt": str(ckpt_path), "n_context": n_ctx, "z_dim": z_dim,
                "max_steps": args.max_steps, "n_areas": len(set(r["area"] for r in all_rows)),
                "wall_s": time.time() - t0}
    (RESULTS_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    mark_done(RESULTS_DIR, smoke=args.smoke)
    print(f"\n[E3 DONE] {len(all_rows)} rows across "
          f"{len(set(r['area'] for r in all_rows))} areas")


if __name__ == "__main__":
    main()
