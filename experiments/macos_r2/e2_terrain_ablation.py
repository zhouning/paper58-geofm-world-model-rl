# -*- coding: utf-8 -*-
"""E2 (RSE reviewer S4): terrain-context ablation.

Reviewer flagged: v2 architecture uses DEM+slope as static context but the
ablation table only reports L2-norm / dilated-conv / unrolled-loss removal.
Terrain removal was never tested.

Retrain LDN on the AlphaEarth cache in three configurations x 3 seeds:
  - full          : elevation + slope (current v2 default)
  - no_slope      : elevation only (n_context = 1)
  - no_context    : both dropped (n_context = 0)

Evaluation: per-area cosine advantage and paired-inference vs zero for each
config, plus config-vs-full paired difference tests.

Outputs (results/e2_terrain_ablation/):
    train_summary.json
    eval_per_area.csv        — area x config -> persistence, model, advantage
    config_paired_tests.csv  — full vs no_slope, full vs no_context

Usage:
    python e2_terrain_ablation.py --seeds 42 123 456 --epochs 50
    python e2_terrain_ablation.py --smoke        # 1 seed x 5 epochs
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PAPER8_ROOT))

from paper58_runtime import _build_model, SCENARIO_DIM, SCENARIOS  # noqa: E402
from run_markers import mark_done  # noqa: E402

AE_DIR = PAPER8_ROOT / "data"
WEIGHTS_DIR = HERE / "weights" / "e2"
RESULTS_DIR = HERE / "results" / "e2_terrain_ablation"
UNROLL_STEPS = 3
Z_DIM = 64

SPLIT_TRAIN = {"yangtze_delta", "jing_jin_ji", "pearl_river", "chengdu_plain",
               "northeast_plain", "north_china_plain", "jianghan_plain", "hetao",
               "yunnan_eco", "daxinganling", "qinghai_edge", "wuyi_mountain"}
SPLIT_VAL = {"guanzhong"}


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_area(area: str) -> tuple[list[np.ndarray] | None, np.ndarray | None]:
    """Returns (embedding_sequence, terrain_context) both channel-first, or (None, None)."""
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

    ctx_file = AE_DIR / f"{area}_context.npy"
    if ctx_file.exists():
        ctx = np.load(ctx_file).astype(np.float32)
        # expected shape (2, H, W); if (H, W, 2) transpose
        if ctx.ndim == 3 and ctx.shape[-1] == 2 and ctx.shape[0] != 2:
            ctx = ctx.transpose(2, 0, 1)
        ctx = ctx[:, :H_min, :W_min]
    else:
        ctx = None
    return seq, ctx


def list_areas_with_features() -> list[str]:
    return sorted({p.stem.rsplit("_emb_", 1)[0] for p in AE_DIR.glob("*_emb_*.npy")
                   if not p.parent.name.startswith("prithvi")})


def build_context(mode: str, ctx: np.ndarray | None, H: int, W: int) -> np.ndarray | None:
    if mode == "no_context":
        return None
    if ctx is None:
        return None
    if mode == "full":
        return ctx
    if mode == "no_slope":
        return ctx[:1]
    raise ValueError(mode)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.flatten(2), b.flatten(2), dim=1).mean()


def train_config(mode: str, seed: int, epochs: int, smoke: bool) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = choose_device()

    n_ctx = {"full": 2, "no_slope": 1, "no_context": 0}[mode]
    print(f"\n=== [{mode}] seed={seed} on {device} (n_context={n_ctx}) ===")

    areas_all = list_areas_with_features()
    if smoke:
        areas_all = ["bishan"] if "bishan" in areas_all else areas_all[:1]

    sequences = {}
    contexts = {}
    for area in areas_all:
        seq, ctx = load_area(area)
        if seq is None:
            continue
        sequences[area] = seq
        contexts[area] = ctx

    train_areas = [a for a in sequences if a in SPLIT_TRAIN] or list(sequences)
    val_areas = [a for a in sequences if a in SPLIT_VAL]
    print(f"  train={len(train_areas)}  val={len(val_areas)}")

    model = _build_model(z_dim=Z_DIM, n_context=n_ctx).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = torch.nn.MSELoss()
    scenario_np = np.zeros(SCENARIO_DIM, dtype=np.float32)
    scenario_np[SCENARIOS["baseline"].id] = 1.0
    scenario = torch.tensor(scenario_np).unsqueeze(0).to(device)

    history = {"train_loss": [], "val_cossim": []}
    for epoch in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for area in train_areas:
            seq = sequences[area]
            _, H, W = seq[0].shape
            ctx_full = build_context(mode, contexts[area], H, W)
            if ctx_full is not None:
                ctx_t = torch.tensor(ctx_full).unsqueeze(0).to(device)
            else:
                ctx_t = None
            for start in range(len(seq) - 1):
                steps = min(UNROLL_STEPS, len(seq) - 1 - start)
                z = torch.tensor(seq[start]).unsqueeze(0).to(device)
                loss_sum = torch.tensor(0.0, device=device)
                for s in range(steps):
                    if ctx_t is not None:
                        z_pred = F.normalize(model(z, scenario, context=ctx_t), p=2, dim=1)
                    else:
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
                    _, H, W = seq[0].shape
                    ctx_full = build_context(mode, contexts[area], H, W)
                    ctx_t = torch.tensor(ctx_full).unsqueeze(0).to(device) if ctx_full is not None else None
                    for i in range(len(seq) - 1):
                        z = torch.tensor(seq[i]).unsqueeze(0).to(device)
                        z_true = F.normalize(
                            torch.tensor(seq[i + 1]).unsqueeze(0).to(device),
                            p=2, dim=1)
                        if ctx_t is not None:
                            pred = F.normalize(model(z, scenario, context=ctx_t), p=2, dim=1)
                        else:
                            pred = F.normalize(model(z, scenario), p=2, dim=1)
                        sims.append(cosine(pred, z_true).item())
            history["val_cossim"].append(float(np.mean(sims)) if sims else -1)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = {"model_state_dict": model.state_dict(), "n_context": n_ctx,
            "mode": mode, "seed": seed, "history": history}
    out = WEIGHTS_DIR / f"terrain_{mode}_seed{seed}.pt"
    torch.save(ckpt, out)
    print(f"  wrote {out}  final_train_loss={history['train_loss'][-1]:.6f}")

    # eval per area
    model.eval()
    per_area = {}
    scenario_np = np.zeros(SCENARIO_DIM, dtype=np.float32)
    scenario_np[SCENARIOS["baseline"].id] = 1.0
    scenario = torch.tensor(scenario_np).unsqueeze(0).to(device)
    with torch.no_grad():
        for area, seq in sequences.items():
            _, H, W = seq[0].shape
            ctx_full = build_context(mode, contexts[area], H, W)
            ctx_t = torch.tensor(ctx_full).unsqueeze(0).to(device) if ctx_full is not None else None
            z_t = torch.tensor(seq[-2]).unsqueeze(0).to(device)
            z_true = torch.tensor(seq[-1]).unsqueeze(0).to(device)
            persist = cosine(F.normalize(z_t, p=2, dim=1), F.normalize(z_true, p=2, dim=1)).item()
            if ctx_t is not None:
                pred = F.normalize(model(z_t, scenario, context=ctx_t), p=2, dim=1)
            else:
                pred = F.normalize(model(z_t, scenario), p=2, dim=1)
            model_c = cosine(pred, F.normalize(z_true, p=2, dim=1)).item()
            per_area[area] = {"persistence": persist, "model": model_c,
                              "advantage": model_c - persist}
    return {"mode": mode, "seed": seed, "history": history,
            "per_area": per_area, "ckpt": str(out)}


def paired_test(diffs: np.ndarray) -> dict:
    if len(diffs) < 2:
        return {"n": len(diffs)}
    wil = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
    t_stat, t_p = stats.ttest_1samp(diffs, popmean=0.0)
    return {"n": int(len(diffs)), "mean": float(np.mean(diffs)),
            "wilcoxon_p": float(wil.pvalue), "t_p": float(t_p),
            "n_pos": int((diffs > 0).sum()),
            "n_neg": int((diffs < 0).sum())}


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--modes", nargs="*", default=["full", "no_slope", "no_context"])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.seeds = [42]
        args.epochs = 5
        args.modes = ["full", "no_context"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_runs = []
    t0 = time.time()
    for mode in args.modes:
        for seed in args.seeds:
            all_runs.append(train_config(mode, seed, args.epochs, args.smoke))
    wall = time.time() - t0

    # write per-area CSV averaged across seeds per mode
    rows = []
    areas = sorted({a for run in all_runs for a in run["per_area"]})
    for mode in args.modes:
        for area in areas:
            adv_seeds = [run["per_area"][area]["advantage"]
                         for run in all_runs
                         if run["mode"] == mode and area in run["per_area"]]
            if not adv_seeds:
                continue
            persist_seeds = [run["per_area"][area]["persistence"]
                             for run in all_runs
                             if run["mode"] == mode and area in run["per_area"]]
            model_seeds = [run["per_area"][area]["model"]
                           for run in all_runs
                           if run["mode"] == mode and area in run["per_area"]]
            rows.append({"mode": mode, "area": area,
                         "n_seeds": len(adv_seeds),
                         "persistence_mean": float(np.mean(persist_seeds)),
                         "model_mean": float(np.mean(model_seeds)),
                         "advantage_mean": float(np.mean(adv_seeds)),
                         "advantage_std": float(np.std(adv_seeds, ddof=1)) if len(adv_seeds) > 1 else 0.0})
    with (RESULTS_DIR / "eval_per_area.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # paired tests: full - {no_slope, no_context}
    def area_mean(mode: str) -> dict[str, float]:
        return {r["area"]: r["advantage_mean"] for r in rows if r["mode"] == mode}
    pair_rows = []
    if "full" in args.modes:
        full_map = area_mean("full")
        for other in args.modes:
            if other == "full":
                continue
            other_map = area_mean(other)
            common = sorted(set(full_map) & set(other_map))
            diffs = np.array([full_map[a] - other_map[a] for a in common])
            t = paired_test(diffs)
            t["compare"] = f"full_minus_{other}"
            t["areas"] = common
            pair_rows.append(t)
    (RESULTS_DIR / "config_paired_tests.json").write_text(json.dumps(pair_rows, indent=2))

    summary = {"seeds": args.seeds, "epochs": args.epochs, "modes": args.modes,
               "wall_s": wall, "smoke": args.smoke,
               "n_runs": len(all_runs),
               "runs": [{"mode": r["mode"], "seed": r["seed"], "ckpt": r["ckpt"]}
                        for r in all_runs]}
    (RESULTS_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2))
    mark_done(RESULTS_DIR, smoke=args.smoke)
    print(f"\n[E2 DONE] wall={wall/60:.1f} min, "
          f"{len(rows)} per-area rows written")


if __name__ == "__main__":
    main()
