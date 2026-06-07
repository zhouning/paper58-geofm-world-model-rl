# -*- coding: utf-8 -*-
"""Paper 5+8 × Paper 12 Ablation — Phase D: evaluation aggregator.

Loads the Prithvi LDN checkpoint (Phase C) and the original AlphaEarth LDN
(D:/adk/data_agent/weights/latent_dynamics_v1.pt), evaluates both on the
same held-out year transitions, and emits a single JSON containing the
five metrics Paper 5+8 reports:

  1. Per-area persistence cos sim (baseline)
  2. Per-area LDN cos sim (model)
  3. Advantage = LDN - persistence
  4. Multi-step rollout decay (steps 1..6)
  5. Change-pixel-only advantage (top-10% changed pixels by persistence cos)

Output: results/paper8_ablation_encoder.json — ready to paste as Table 1
new row in paper8_geofm_world_model_rl.tex.

Usage:
    python paper8/eval_prithvi_vs_alphaearth.py
    python paper8/eval_prithvi_vs_alphaearth.py --eval-year 2024
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn.functional as F

PAPER8_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER8_ROOT))

try:
    from paper58_runtime import _build_model, SCENARIO_DIM, SCENARIOS
except ImportError:
    sys.path.insert(0, "D:/adk/data_agent")
    from world_model import _build_model, SCENARIO_DIM, SCENARIOS

AE_DIR = PAPER8_ROOT / "data"
AE_RAW_DIR = Path("D:/adk/data_agent/weights/raw_data")
PR_DIR = PAPER8_ROOT / "data" / "prithvi"
WEIGHTS_DIR = PAPER8_ROOT / "weights"
AE_CKPT = Path("D:/adk/data_agent/weights/latent_dynamics_v1.pt")
RESULTS_DIR = PAPER8_ROOT / "results"


def load_ae_ldn(device):
    ckpt = torch.load(AE_CKPT, map_location=device, weights_only=False)
    model = _build_model(z_dim=ckpt.get("z_dim", 64),
                        scenario_dim=ckpt.get("scenario_dim", 16),
                        n_context=ckpt.get("n_context", 2)).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, ckpt.get("z_dim", 64)


def load_prithvi_ldn(device, seed=42):
    """Load best Prithvi LDN by val cossim across available seeds."""
    candidates = list(WEIGHTS_DIR.glob("latent_dynamics_prithvi_768d_seed*.pt"))
    if not candidates:
        return None, None
    best = max(candidates, key=lambda p: torch.load(p, map_location="cpu", weights_only=False).get("best_val_cossim", -1.0))
    ckpt = torch.load(best, map_location=device, weights_only=False)
    model = _build_model(z_dim=ckpt["z_dim"], scenario_dim=ckpt["scenario_dim"], n_context=ckpt["n_context"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, ckpt["z_dim"]


def load_emb(area: str, year: int, encoder: str) -> np.ndarray | None:
    if encoder == "prithvi":
        p = PR_DIR / f"{area}_emb_{year}.npy"
        return np.load(p) if p.exists() else None
    p1 = AE_RAW_DIR / f"emb_{area}_{year}.npy"
    if p1.exists():
        return np.load(p1)
    p2 = AE_DIR / f"{area}_emb_{year}.npy"
    return np.load(p2) if p2.exists() else None


def list_areas(encoder: str) -> list[str]:
    if encoder == "prithvi":
        return sorted({p.stem.rsplit("_emb_", 1)[0] for p in PR_DIR.glob("*_emb_*.npy")})
    raw = {p.stem.replace("emb_", "").rsplit("_", 1)[0] for p in AE_RAW_DIR.glob("emb_*.npy")}
    local = {p.stem.rsplit("_emb_", 1)[0] for p in AE_DIR.glob("*_emb_*.npy")}
    return sorted(raw | local)


def cossim_grid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel cos sim. a, b are [H, W, D]; output [H, W]."""
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return np.sum(an * bn, axis=-1)


def predict_next(model, z_curr: np.ndarray, device) -> np.ndarray:
    """[H, W, D] -> next year [H, W, D] L2-normed."""
    z = torch.tensor(z_curr.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    scenario_np = np.zeros(SCENARIO_DIM, dtype=np.float32)
    scenario_np[SCENARIOS["baseline"].id] = 1.0
    scenario = torch.tensor(scenario_np).unsqueeze(0).to(device)
    with torch.no_grad():
        z_pred = F.normalize(model(z, scenario), p=2, dim=1)
    return z_pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)


def eval_encoder(encoder: str, eval_year: int, device) -> dict:
    """For each area: load (eval_year-1, eval_year), compare persistence vs LDN."""
    if encoder == "ae":
        model, z_dim = load_ae_ldn(device)
    else:
        model, z_dim = load_prithvi_ldn(device)
        if model is None:
            return {"error": "no Prithvi LDN checkpoint found; run train_prithvi_ldn.py first"}

    out = {"encoder": encoder, "z_dim": z_dim, "eval_year": eval_year, "areas": {}}
    cur_year = eval_year - 1
    for area in list_areas(encoder):
        z_t = load_emb(area, cur_year, encoder)
        z_tp1 = load_emb(area, eval_year, encoder)
        if z_t is None or z_tp1 is None:
            continue

        persistence_map = cossim_grid(z_t, z_tp1)
        z_pred = predict_next(model, z_t, device)
        ldn_map = cossim_grid(z_pred, z_tp1)

        change_thresh = np.percentile(persistence_map, 10)
        change_mask = persistence_map < change_thresh

        out["areas"][area] = {
            "persistence_mean": float(persistence_map.mean()),
            "ldn_mean": float(ldn_map.mean()),
            "advantage": float(ldn_map.mean() - persistence_map.mean()),
            "change_pixel_persistence": float(persistence_map[change_mask].mean()) if change_mask.any() else None,
            "change_pixel_ldn": float(ldn_map[change_mask].mean()) if change_mask.any() else None,
            "change_pixel_advantage": float((ldn_map - persistence_map)[change_mask].mean()) if change_mask.any() else None,
        }
    return out


def multistep_decay(encoder: str, start_year: int, n_steps: int, device) -> dict:
    """Roll out N steps from start_year, report per-step cos sim vs ground truth."""
    if encoder == "ae":
        model, z_dim = load_ae_ldn(device)
    else:
        model, z_dim = load_prithvi_ldn(device)
        if model is None:
            return {"error": "no Prithvi LDN checkpoint found"}

    per_step = {}
    for step in range(1, n_steps + 1):
        target_year = start_year + step
        sims = []
        for area in list_areas(encoder):
            z_init = load_emb(area, start_year, encoder)
            z_target = load_emb(area, target_year, encoder)
            if z_init is None or z_target is None:
                continue
            z = z_init.copy()
            for _ in range(step):
                z = predict_next(model, z, device)
            sims.append(float(cossim_grid(z, z_target).mean()))
        if sims:
            per_step[step] = {"mean": mean(sims), "std": stdev(sims) if len(sims) > 1 else 0.0, "n_areas": len(sims)}
    return {"encoder": encoder, "z_dim": z_dim, "start_year": start_year, "per_step": per_step}


def aggregate(per_area: dict) -> dict:
    rows = list(per_area.get("areas", {}).values())
    if not rows:
        return per_area
    keys = ["persistence_mean", "ldn_mean", "advantage", "change_pixel_advantage"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            summary[k] = {"mean": mean(vals), "std": stdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}
    per_area["summary"] = summary
    return per_area


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-year", type=int, default=2024)
    p.add_argument("--rollout-start", type=int, default=2018)
    p.add_argument("--rollout-steps", type=int, default=6)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    out = {"eval_year": args.eval_year}
    print("\n=== AlphaEarth LDN ===")
    out["alphaearth"] = aggregate(eval_encoder("ae", args.eval_year, device))
    out["alphaearth_rollout"] = multistep_decay("ae", args.rollout_start, args.rollout_steps, device)

    print("\n=== Prithvi LDN ===")
    out["prithvi"] = aggregate(eval_encoder("prithvi", args.eval_year, device))
    out["prithvi_rollout"] = multistep_decay("prithvi", args.rollout_start, args.rollout_steps, device)

    print("\n=== Summary ===")
    for enc in ("alphaearth", "prithvi"):
        s = out[enc].get("summary", {})
        if not s:
            print(f"  {enc}: no data")
            continue
        print(f"  {enc:12s}  pers={s['persistence_mean']['mean']:.4f}  "
              f"ldn={s['ldn_mean']['mean']:.4f}  adv={s['advantage']['mean']:+.4f}  "
              f"change_adv={s.get('change_pixel_advantage', {}).get('mean', 0):+.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "paper8_ablation_encoder.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
