# -*- coding: utf-8 -*-
"""Paper 5+8 × Paper 12 Ablation — Phase C: retrain LDN at z_dim=768 on Prithvi features.

Forks data_agent/world_model.py:train_dynamics_model() but reads Prithvi
features from data/prithvi/*.npy and trains a LatentDynamicsNet built at
z_dim=768. Same protocol as the original (3-step unrolled MSE, baseline
scenario, no scene-level leakage) but with explicit train/val/test/OOD
split mirroring Paper 5+8's published partition.

Output: weights/latent_dynamics_prithvi_768d_seed{seed}.pt

Usage:
    # Smoke test (Bishan only, 1 seed, 5 epochs):
    python paper8/train_prithvi_ldn.py --smoke

    # Full run (17 areas, 3 seeds, 50 epochs):
    python paper8/train_prithvi_ldn.py --epochs 50 --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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

PR_DIR = PAPER8_ROOT / "data" / "prithvi"
WEIGHTS_DIR = PAPER8_ROOT / "weights"
RESULTS_DIR = PAPER8_ROOT / "results"

YEARS = list(range(2017, 2025))
UNROLL_STEPS = 3
Z_DIM_PRITHVI = 768

SPLIT_TRAIN = {"yangtze_delta", "jing_jin_ji", "pearl_river", "chengdu_plain",
               "northeast_plain", "north_china_plain", "jianghan_plain", "hetao",
               "yunnan_eco", "daxinganling", "qinghai_edge", "wuyi_mountain"}
SPLIT_VAL = {"guanzhong"}
SPLIT_TEST = {"minnan_coast", "poyang_lake"}
SPLIT_OOD = {"bishan", "banzhucun"}


def split_for(area: str) -> str:
    if area in SPLIT_TRAIN: return "train"
    if area in SPLIT_VAL:   return "val"
    if area in SPLIT_TEST:  return "test"
    if area in SPLIT_OOD:   return "ood"
    return "unknown"


def load_area_sequence(area: str) -> list[np.ndarray] | None:
    """Returns [emb_2017, emb_2018, ...] each as [768, H, W] channel-first, or None.

    Missing years break the sequence — we return None to skip the whole area for
    multi-step unroll training. Phase B can still evaluate partial sequences.
    """
    files = sorted(PR_DIR.glob(f"{area}_emb_*.npy"))
    if not files:
        return None
    years = []
    for f in files:
        y = int(f.stem.rsplit("_", 1)[1])
        years.append((y, f))
    years.sort()
    sorted_ys = [y for y, _ in years]
    if sorted_ys != list(range(sorted_ys[0], sorted_ys[-1] + 1)):
        return None
    arrs = []
    H_min, W_min = None, None
    for _, f in years:
        a = np.load(f).astype(np.float32)
        if H_min is None:
            H_min, W_min = a.shape[0], a.shape[1]
        H_min = min(H_min, a.shape[0])
        W_min = min(W_min, a.shape[1])
        arrs.append(a)
    arrs = [a[:H_min, :W_min].transpose(2, 0, 1) for a in arrs]
    return arrs


def list_areas_with_features() -> list[str]:
    return sorted({p.stem.rsplit("_emb_", 1)[0] for p in PR_DIR.glob("*_emb_*.npy")})


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.flatten(2), b.flatten(2), dim=1).mean()


def train_one_seed(seed: int, epochs: int, lr: float, smoke: bool) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== seed={seed} on {device} ===")

    areas = list_areas_with_features()
    if smoke:
        areas = ["bishan"] if "bishan" in areas else areas[:1]

    sequences: dict[str, list[np.ndarray]] = {}
    for area in areas:
        seq = load_area_sequence(area)
        if seq is None:
            print(f"  SKIP {area} (incomplete sequence)")
            continue
        sequences[area] = seq

    train_areas = [a for a in sequences if split_for(a) == "train"] or list(sequences)
    val_areas = [a for a in sequences if split_for(a) == "val"]
    print(f"train areas: {len(train_areas)}, val: {len(val_areas)}, smoke={smoke}")

    model = _build_model(z_dim=Z_DIM_PRITHVI).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    scenario_np = np.zeros(SCENARIO_DIM, dtype=np.float32)
    scenario_np[SCENARIOS["baseline"].id] = 1.0
    scenario = torch.tensor(scenario_np).unsqueeze(0).to(device)

    history = {"train_loss": [], "val_cossim": []}
    best_val = -1.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for area in train_areas:
            seq = sequences[area]
            for start in range(len(seq) - 1):
                steps = min(UNROLL_STEPS, len(seq) - 1 - start)
                z = torch.tensor(seq[start]).unsqueeze(0).to(device)
                loss_sum = torch.tensor(0.0, device=device)
                for s in range(steps):
                    z_pred = model(z, scenario)
                    z_pred = F.normalize(z_pred, p=2, dim=1)
                    z_true = torch.tensor(seq[start + s + 1]).unsqueeze(0).to(device)
                    z_true = F.normalize(z_true, p=2, dim=1)
                    weight = 1.0 / (2 ** s)
                    loss_sum = loss_sum + weight * mse(z_pred, z_true)
                    z = z_pred
                optim.zero_grad()
                loss_sum.backward()
                optim.step()
                total_loss += loss_sum.item()
                n += 1
        avg = total_loss / max(n, 1)
        history["train_loss"].append(avg)

        if val_areas:
            model.eval()
            val_sims = []
            with torch.no_grad():
                for area in val_areas:
                    seq = sequences[area]
                    for i in range(len(seq) - 1):
                        z = torch.tensor(seq[i]).unsqueeze(0).to(device)
                        z_true = torch.tensor(seq[i + 1]).unsqueeze(0).to(device)
                        z_pred = F.normalize(model(z, scenario), p=2, dim=1)
                        z_true = F.normalize(z_true, p=2, dim=1)
                        val_sims.append(cosine(z_pred, z_true).item())
            val_cs = float(np.mean(val_sims)) if val_sims else -1.0
            history["val_cossim"].append(val_cs)
            best_val = max(best_val, val_cs)
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            vc = history["val_cossim"][-1] if history["val_cossim"] else "n/a"
            print(f"  epoch {epoch+1}/{epochs}  train_loss={avg:.6f}  val_cossim={vc}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "z_dim": Z_DIM_PRITHVI,
        "scenario_dim": SCENARIO_DIM,
        "n_context": 2,
        "seed": seed,
        "epochs": epochs,
        "training_areas": sorted(train_areas),
        "val_areas": sorted(val_areas),
        "best_val_cossim": best_val,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "history": history,
    }
    out = WEIGHTS_DIR / f"latent_dynamics_prithvi_768d_seed{seed}.pt"
    torch.save(ckpt, out)
    print(f"  wrote {out}  best_val_cossim={best_val:.4f}")
    return {"seed": seed, "best_val_cossim": best_val, "ckpt": str(out)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--smoke", action="store_true", help="Bishan only, 1 seed, 5 epochs")
    args = p.parse_args()

    if args.smoke:
        args.seeds = [42]
        args.epochs = 5

    runs = []
    t0 = time.time()
    for seed in args.seeds:
        runs.append(train_one_seed(seed, args.epochs, args.lr, args.smoke))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "z_dim": Z_DIM_PRITHVI,
        "epochs": args.epochs,
        "lr": args.lr,
        "smoke": args.smoke,
        "runs": runs,
        "wall_s": time.time() - t0,
    }
    out_path = RESULTS_DIR / "prithvi_ldn_training.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}, wall={summary['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
