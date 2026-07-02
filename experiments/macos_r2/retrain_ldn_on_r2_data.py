# -*- coding: utf-8 -*-
"""Retrain LatentDynamicsNet on the R2 re-extracted embeddings.

The v4 negative-result finding was caused by a train/test distribution mismatch:
- latent_dynamics_v1.pt was trained on pre-commit (2026-06-07) embeddings
- E6 evaluated it on freshly re-extracted (2026-07-02) embeddings from GEE

Fix: retrain the same architecture (459K params, dilated CNN, 3-step unrolled MSE)
on the R2 embeddings, then re-evaluate.

Output:
    weights/latent_dynamics_v2.pt   — new checkpoint
    results/retrain_v2/train_summary.json
    results/retrain_v2/eval_per_area.csv
    results/retrain_v2/eval_paired_tests.json

Usage:
    python retrain_ldn_on_r2_data.py --epochs 100 --seeds 42 123 456
    python retrain_ldn_on_r2_data.py --smoke   # 1 seed, 10 epochs
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PAPER8_ROOT = REPO_ROOT / "experiments" / "paper8"
sys.path.insert(0, str(PAPER8_ROOT))

from paper58_runtime import _build_model, SCENARIO_DIM, SCENARIOS

AE_DIR = PAPER8_ROOT / "data"
WEIGHTS_DIR = HERE / "weights" / "retrain_v2"
RESULTS_DIR = HERE / "results" / "retrain_v2"

YEARS = list(range(2017, 2025))
UNROLL_STEPS = 3
Z_DIM = 64

# Same split as v1 training
SPLIT_TRAIN = {"yangtze_delta", "jing_jin_ji", "pearl_river", "chengdu_plain",
               "northeast_plain", "north_china_plain", "jianghan_plain", "hetao",
               "yunnan_eco", "daxinganling", "qinghai_edge", "wuyi_mountain"}
SPLIT_VAL = {"guanzhong", "minnan_coast"}
SPLIT_TEST = {"poyang_lake"}
SPLIT_OOD = {"bishan", "banzhucun"}


def repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def make_csv_writer(handle, fieldnames):
    return csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_area(area: str):
    files = sorted(AE_DIR.glob(f"{area}_emb_*.npy"))
    if not files:
        return None
    years = sorted((int(f.stem.rsplit("_", 1)[1]), f) for f in files)
    seq_years = [y for y, _ in years]
    if seq_years != list(range(seq_years[0], seq_years[-1] + 1)):
        return None
    arrs = []
    H_min = W_min = None
    for _, f in years:
        a = np.load(f).astype(np.float32)
        if H_min is None:
            H_min, W_min = a.shape[0], a.shape[1]
        H_min = min(H_min, a.shape[0])
        W_min = min(W_min, a.shape[1])
        arrs.append(a)
    return [a[:H_min, :W_min].transpose(2, 0, 1) for a in arrs]


def list_areas():
    return sorted({p.stem.rsplit("_emb_", 1)[0] for p in AE_DIR.glob("*_emb_*.npy")
                   if "_context" not in p.stem and "prithvi" not in str(p.parent)})


def cosine(a, b):
    return F.cosine_similarity(a.flatten(2), b.flatten(2), dim=1).mean()


def cossim_grid(a_np, b_np):
    """Per-pixel cosine similarity, same as eval_prithvi_vs_alphaearth.py."""
    if a_np.shape != b_np.shape:
        h = min(a_np.shape[0], b_np.shape[0])
        w = min(a_np.shape[1], b_np.shape[1])
        a_np = a_np[:h, :w]
        b_np = b_np[:h, :w]
    an = a_np / (np.linalg.norm(a_np, axis=-1, keepdims=True) + 1e-9)
    bn = b_np / (np.linalg.norm(b_np, axis=-1, keepdims=True) + 1e-9)
    return np.sum(an * bn, axis=-1)


def paired_stats_from_advantages(advs, best_ckpt: str, n_perm: int = 100_000,
                                 n_boot: int = 20_000, seed: int = 20260702):
    advs = np.asarray(advs, dtype=np.float64)
    n_pos = int((advs > 0).sum())
    n_neg = int((advs < 0).sum())
    results = {
        "n": int(len(advs)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "best_ckpt": best_ckpt,
    }
    if len(advs) < 2:
        return results

    rng = np.random.default_rng(seed)
    mean = float(advs.mean())
    sd = float(advs.std(ddof=1))
    wil = stats.wilcoxon(advs, zero_method="wilcox", alternative="two-sided")
    _, t_p = stats.ttest_1samp(advs, 0.0)
    signs = rng.choice([-1, 1], size=(n_perm, len(advs)))
    perm_p = float(np.mean(np.abs((signs * advs[None, :]).mean(axis=1)) >= abs(mean)))
    idx = rng.integers(0, len(advs), size=(n_boot, len(advs)))
    boot = advs[idx].mean(axis=1)
    ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])
    results.update({
        "mean": mean,
        "sd": sd,
        "wilcoxon_p": float(wil.pvalue),
        "t_p": float(t_p),
        "permutation_p": perm_p,
        "bootstrap_ci_lo": float(ci_lo),
        "bootstrap_ci_hi": float(ci_hi),
        "cohen_dz": mean / sd if sd else float("inf"),
    })
    return results


def train_one_seed(seed, epochs, lr):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = choose_device()
    print(f"\n=== seed={seed} on {device} ===")

    sequences = {}
    for area in list_areas():
        seq = load_area(area)
        if seq is None:
            continue
        sequences[area] = seq

    train_areas = [a for a in sequences if a in SPLIT_TRAIN]
    val_areas = [a for a in sequences if a in SPLIT_VAL]
    print(f"  train={len(train_areas)} val={len(val_areas)} total={len(sequences)}")

    model = _build_model(z_dim=Z_DIM, n_context=0).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    scenario_np = np.zeros(SCENARIO_DIM, dtype=np.float32)
    scenario_np[SCENARIOS["baseline"].id] = 1.0
    scenario = torch.tensor(scenario_np).unsqueeze(0).to(device)

    best_val = -1.0
    history = {"train_loss": [], "val_cossim": []}
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for area in train_areas:
            seq = sequences[area]
            for start in range(len(seq) - 1):
                steps = min(UNROLL_STEPS, len(seq) - 1 - start)
                z = torch.tensor(seq[start]).unsqueeze(0).to(device)
                loss_sum = torch.tensor(0.0, device=device)
                for s in range(steps):
                    z_pred = F.normalize(model(z, scenario), p=2, dim=1)
                    z_true = F.normalize(
                        torch.tensor(seq[start + s + 1]).unsqueeze(0).to(device), p=2, dim=1)
                    loss_sum = loss_sum + (1.0 / (2 ** s)) * mse(z_pred, z_true)
                    z = z_pred
                optim.zero_grad()
                loss_sum.backward()
                optim.step()
                total += loss_sum.item()
                n += 1
        history["train_loss"].append(total / max(n, 1))

        if val_areas:
            model.eval()
            sims = []
            with torch.no_grad():
                for area in val_areas:
                    seq = sequences[area]
                    z = torch.tensor(seq[-2]).unsqueeze(0).to(device)
                    z_pred = F.normalize(model(z, scenario), p=2, dim=1)
                    z_true = F.normalize(torch.tensor(seq[-1]).unsqueeze(0).to(device), p=2, dim=1)
                    sims.append(cosine(z_pred, z_true).item())
            val_cos = float(np.mean(sims))
            history["val_cossim"].append(val_cos)
            if val_cos > best_val:
                best_val = val_cos
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}: loss={history['train_loss'][-1]:.6f} val_cos={history['val_cossim'][-1] if history['val_cossim'] else 'N/A':.6f}")

    # Save checkpoint
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": best_state if val_areas else model.state_dict(),
        "z_dim": Z_DIM,
        "scenario_dim": SCENARIO_DIM,
        "n_context": 0,
        "training_areas": train_areas,
        "training_years": YEARS,
        "epochs": epochs,
        "final_loss": history["train_loss"][-1],
        "best_val_cossim": best_val,
        "seed": seed,
        "version": "2.0_r2_retrained",
    }
    out = WEIGHTS_DIR / f"latent_dynamics_v2_seed{seed}.pt"
    torch.save(ckpt, out)
    return {"seed": seed, "best_val": best_val, "final_loss": history["train_loss"][-1], "ckpt": repo_rel(out), "history": history}


def eval_best_checkpoint():
    """Evaluate the best v2 checkpoint on all areas (same protocol as v1)."""
    device = choose_device()
    ckpts = sorted(WEIGHTS_DIR.glob("latent_dynamics_v2_seed*.pt"))
    if not ckpts:
        return {"error": "no v2 checkpoints found"}

    # Pick best by val cossim
    best_ck = max(ckpts, key=lambda p: torch.load(p, map_location="cpu", weights_only=False).get("best_val_cossim", -1))
    ckpt = torch.load(best_ck, map_location=device, weights_only=False)
    model = _build_model(z_dim=Z_DIM, n_context=0).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Eval using {best_ck.name} (val_cos={ckpt['best_val_cossim']:.6f})")

    scenario_np = np.zeros(SCENARIO_DIM, dtype=np.float32)
    scenario_np[SCENARIOS["baseline"].id] = 1.0
    scenario = torch.tensor(scenario_np).unsqueeze(0).to(device)

    per_area = {}
    for area in list_areas():
        seq = load_area(area)
        if seq is None or len(seq) < 2:
            continue
        # Last transition
        z_t = np.load(AE_DIR / f"{area}_emb_{YEARS[-2]}.npy").astype(np.float32)
        z_tp1 = np.load(AE_DIR / f"{area}_emb_{YEARS[-1]}.npy").astype(np.float32)

        persist_map = cossim_grid(z_t, z_tp1)
        persist = float(persist_map.mean())

        z_torch = torch.tensor(z_t.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred = F.normalize(model(z_torch, scenario), p=2, dim=1)
        pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        model_map = cossim_grid(pred_np, z_tp1)
        model_cos = float(model_map.mean())

        per_area[area] = {"persistence": persist, "model": model_cos, "advantage": model_cos - persist}

    # Write results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "eval_per_area.csv").open("w", newline="") as f:
        w = make_csv_writer(f, fieldnames=["area", "persistence", "model", "advantage"])
        w.writeheader()
        for area in sorted(per_area):
            w.writerow({"area": area, **per_area[area]})

    advs = np.array([v["advantage"] for v in per_area.values()])
    results = paired_stats_from_advantages(advs, best_ckpt=best_ck.name)
    (RESULTS_DIR / "eval_paired_tests.json").write_text(json.dumps(results, indent=2))
    print(f"Eval: n={len(advs)} mean={results.get('mean', float('nan')):.6f} "
          f"n_pos={results['n_pos']} wilcoxon_p={results.get('wilcoxon_p')}")
    return results


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.seeds = [42]
        args.epochs = 10

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        t0 = time.time()
        runs = [train_one_seed(s, args.epochs, args.lr) for s in args.seeds]
        summary = {"seeds": args.seeds, "epochs": args.epochs, "lr": args.lr,
                   "runs": runs, "wall_s": time.time() - t0}
        (RESULTS_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nTraining done. wall={summary['wall_s']:.1f}s")

    eval_best_checkpoint()


if __name__ == "__main__":
    main()
