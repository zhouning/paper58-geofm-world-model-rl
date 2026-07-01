# -*- coding: utf-8 -*-
"""E1 (RSE reviewer M4): Train LDN on Prithvi *spatial* tokens + eval vs.\ AlphaEarth.

Depends on `e1_prithvi_patch_extract.py` having populated
`experiments/macos_r2/data/prithvi_spatial/*.npy` with shape (H, W, 768).

Training protocol mirrors experiments/paper8/train_prithvi_ldn.py exactly
(3-step unrolled MSE, baseline scenario, GroupNorm CNN with dilated conv 1/2/4)
but at z_dim=768 with terrain context passed through the same conv.

Evaluation:
    1. Per-area persistence cosine similarity in Prithvi space
    2. Per-area LDN cosine similarity in Prithvi space
    3. Advantage = LDN - persistence, per area
    4. Wilcoxon / paired-t / permutation vs zero (same tests as v2 Table 5)
    5. Match against AlphaEarth advantage numbers from v1 revision_results

Outputs (under results/e1_prithvi_patch/):
    train_summary.json     — per-seed training curves + best val cossim
    eval_per_area.csv      — per-area persistence, LDN, advantage
    eval_paired_tests.csv  — Wilcoxon/t/permutation/sign on Prithvi advantage
    encoder_head_to_head.csv — AlphaEarth vs Prithvi(spatial) side by side

Usage:
    python e1_prithvi_patch_train_eval.py --seeds 42 123 456 --epochs 50
    python e1_prithvi_patch_train_eval.py --smoke     # 1 seed, 5 epochs
"""
from __future__ import annotations

import argparse
import json
import math
import os
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

try:
    from paper58_runtime import _build_model, SCENARIO_DIM, SCENARIOS
except ImportError as exc:
    raise SystemExit(f"cannot import paper58_runtime: {exc}")

PR_SPATIAL_DIR = HERE / "data" / "prithvi_spatial"
WEIGHTS_DIR = HERE / "weights" / "e1"
RESULTS_DIR = HERE / "results" / "e1_prithvi_patch"
AE_METRICS = REPO_ROOT / "paper" / "rse_submission_paper58" / "revision_results" / "alphaearth_area_metrics.csv"

YEARS = list(range(2017, 2025))
UNROLL_STEPS = 3
Z_DIM_PRITHVI = 768

# same split as the parent train_prithvi_ldn.py
SPLIT_TRAIN = {"yangtze_delta", "jing_jin_ji", "pearl_river", "chengdu_plain",
               "northeast_plain", "north_china_plain", "jianghan_plain", "hetao",
               "yunnan_eco", "daxinganling", "qinghai_edge", "wuyi_mountain"}
SPLIT_VAL = {"guanzhong"}
SPLIT_TEST = {"minnan_coast", "poyang_lake"}
SPLIT_OOD = {"bishan", "banzhucun"}

RNG_SEED_STATS = 20260702


def load_area_sequence(area: str) -> list[np.ndarray] | None:
    files = sorted(PR_SPATIAL_DIR.glob(f"{area}_emb_*.npy"))
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


def list_areas_with_features() -> list[str]:
    return sorted({p.stem.rsplit("_emb_", 1)[0] for p in PR_SPATIAL_DIR.glob("*_emb_*.npy")})


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.flatten(2), b.flatten(2), dim=1).mean()


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_seed(seed: int, epochs: int, lr: float, smoke: bool) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = choose_device()
    print(f"\n=== seed={seed} on {device} ===")

    areas_all = list_areas_with_features()
    if smoke:
        areas_all = ["bishan"] if "bishan" in areas_all else areas_all[:1]

    sequences = {}
    for area in areas_all:
        seq = load_area_sequence(area)
        if seq is None:
            print(f"  SKIP {area} (incomplete)")
            continue
        sequences[area] = seq

    train_areas = [a for a in sequences if a in SPLIT_TRAIN] or list(sequences)
    val_areas = [a for a in sequences if a in SPLIT_VAL]
    print(f"  train={len(train_areas)}  val={len(val_areas)}  smoke={smoke}")

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
        total = 0.0
        n = 0
        for area in train_areas:
            seq = sequences[area]
            for start in range(len(seq) - 1):
                steps = min(UNROLL_STEPS, len(seq) - 1 - start)
                z = torch.tensor(seq[start]).unsqueeze(0).to(device)
                loss_sum = torch.tensor(0.0, device=device)
                for s in range(steps):
                    z_pred = F.normalize(model(z, scenario), p=2, dim=1)
                    z_true = F.normalize(
                        torch.tensor(seq[start + s + 1]).unsqueeze(0).to(device),
                        p=2, dim=1)
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
                    for i in range(len(seq) - 1):
                        z = torch.tensor(seq[i]).unsqueeze(0).to(device)
                        pred = F.normalize(model(z, scenario), p=2, dim=1)
                        true = F.normalize(
                            torch.tensor(seq[i + 1]).unsqueeze(0).to(device),
                            p=2, dim=1)
                        sims.append(cosine(pred, true).item())
            vc = float(np.mean(sims)) if sims else -1.0
            history["val_cossim"].append(vc)
            best_val = max(best_val, vc)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WEIGHTS_DIR / f"prithvi_spatial_ldn_seed{seed}.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "z_dim": Z_DIM_PRITHVI, "scenario_dim": SCENARIO_DIM,
                "n_context": 2, "seed": seed, "epochs": epochs,
                "history": history, "best_val_cossim": best_val}, out_path)
    print(f"  wrote {out_path}  best_val={best_val:.4f}")
    return {"seed": seed, "best_val_cossim": best_val, "ckpt": str(out_path),
            "history": history}


def eval_area_advantage(seed: int, sequences: dict[str, list[np.ndarray]]) -> dict[str, float]:
    """Persistence vs LDN cosine similarity per area, evaluated on last (t, t+1) transition."""
    device = choose_device()
    ckpt_path = WEIGHTS_DIR / f"prithvi_spatial_ldn_seed{seed}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = _build_model(z_dim=Z_DIM_PRITHVI).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scenario_np = np.zeros(SCENARIO_DIM, dtype=np.float32)
    scenario_np[SCENARIOS["baseline"].id] = 1.0
    scenario = torch.tensor(scenario_np).unsqueeze(0).to(device)

    per_area = {}
    with torch.no_grad():
        for area, seq in sequences.items():
            # evaluate on the last available annual transition (t -> t+1)
            z_t = torch.tensor(seq[-2]).unsqueeze(0).to(device)
            z_true = torch.tensor(seq[-1]).unsqueeze(0).to(device)
            z_t_n = F.normalize(z_t, p=2, dim=1)
            z_true_n = F.normalize(z_true, p=2, dim=1)
            persist = cosine(z_t_n, z_true_n).item()
            z_pred = F.normalize(model(z_t, scenario), p=2, dim=1)
            model_c = cosine(z_pred, z_true_n).item()
            per_area[area] = {
                "persistence": persist,
                "model": model_c,
                "advantage": model_c - persist,
            }
    return per_area


def paired_significance(advantages: np.ndarray, seed: int = RNG_SEED_STATS) -> dict:
    n = len(advantages)
    if n < 2:
        return {"n": n}
    rng = np.random.default_rng(seed)
    mean = float(np.mean(advantages))
    sd = float(np.std(advantages, ddof=1))
    dz = mean / sd if sd else float("inf")

    t_stat, t_p = stats.ttest_1samp(advantages, popmean=0.0)
    wil = stats.wilcoxon(advantages, zero_method="wilcox", alternative="two-sided")
    n_pos = int(np.sum(advantages > 0))
    sign_p = float(stats.binomtest(n_pos, n, 0.5).pvalue)

    # sign-flip permutation
    signs = rng.choice([-1, 1], size=(100_000, n))
    perm_means = (signs * advantages[None, :]).mean(axis=1)
    perm_p = float(np.mean(np.abs(perm_means) >= abs(mean)))

    idx = rng.integers(0, n, size=(20_000, n))
    boot_means = advantages[idx].mean(axis=1)
    ci_lo, ci_hi = np.quantile(boot_means, [0.025, 0.975])
    return {"n": n, "mean": mean, "sd": sd, "cohen_dz": dz,
            "t_stat": float(t_stat), "t_p": float(t_p),
            "wilcoxon_p": float(wil.pvalue), "permutation_p": perm_p,
            "sign_p": sign_p, "n_pos": n_pos,
            "bootstrap_ci_lo": float(ci_lo), "bootstrap_ci_hi": float(ci_hi)}


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--eval-only", action="store_true",
                   help="reuse existing checkpoints; only regenerate eval CSVs")
    args = p.parse_args()

    if args.smoke:
        args.seeds = [42]
        args.epochs = 5

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    train_summary = {"z_dim": Z_DIM_PRITHVI, "epochs": args.epochs,
                     "lr": args.lr, "smoke": args.smoke, "seeds": args.seeds,
                     "runs": []}
    if not args.eval_only:
        t0 = time.time()
        for seed in args.seeds:
            train_summary["runs"].append(train_one_seed(seed, args.epochs, args.lr, args.smoke))
        train_summary["wall_s"] = time.time() - t0
        (RESULTS_DIR / "train_summary.json").write_text(json.dumps(train_summary, indent=2))

    # Aggregate evaluation: for every seed, get per-area advantage, then
    # average per area across seeds
    sequences = {}
    for area in list_areas_with_features():
        seq = load_area_sequence(area)
        if seq is not None:
            sequences[area] = seq
    print(f"\neval sequences loaded: {len(sequences)}")

    seed_results = {}
    for seed in args.seeds:
        seed_results[seed] = eval_area_advantage(seed, sequences)

    # per-area mean across seeds
    areas = sorted(sequences.keys())
    rows = []
    for area in areas:
        persist = np.mean([seed_results[s][area]["persistence"] for s in args.seeds])
        model = np.mean([seed_results[s][area]["model"] for s in args.seeds])
        adv_by_seed = [seed_results[s][area]["advantage"] for s in args.seeds]
        rows.append({"area": area, "persistence": persist, "model": model,
                     "advantage_mean": float(np.mean(adv_by_seed)),
                     "advantage_std": float(np.std(adv_by_seed, ddof=1)) if len(adv_by_seed) > 1 else 0.0,
                     "n_seeds": len(args.seeds)})

    import csv
    with (RESULTS_DIR / "eval_per_area.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Paired significance
    advs = np.array([r["advantage_mean"] for r in rows], dtype=float)
    tests = paired_significance(advs)
    (RESULTS_DIR / "eval_paired_tests.json").write_text(json.dumps(tests, indent=2))

    # Head-to-head against AlphaEarth
    if AE_METRICS.exists():
        import csv as _csv
        ae_advantage = {}
        with AE_METRICS.open() as f:
            reader = _csv.DictReader(f)
            for row in reader:
                ae_advantage[row["area"]] = float(row["advantage"])
        head_rows = []
        for r in rows:
            head_rows.append({"area": r["area"],
                              "prithvi_spatial_advantage": r["advantage_mean"],
                              "alphaearth_advantage": ae_advantage.get(r["area"])})
        with (RESULTS_DIR / "encoder_head_to_head.csv").open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(head_rows[0].keys()))
            w.writeheader()
            w.writerows(head_rows)

    (RESULTS_DIR / ".done").touch()
    print(f"\n[E1 DONE] Prithvi spatial paired stats: n={tests.get('n')} "
          f"mean={tests.get('mean'):.5f} wilcoxon_p={tests.get('wilcoxon_p'):.3f}")
    print(f"  results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
