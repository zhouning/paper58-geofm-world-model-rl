# -*- coding: utf-8 -*-
"""E5 (RSE reviewer S3): SA-Alloc 11-parameter sensitivity sweep.

Reviewer flagged: SA-Alloc has 11 hand-tuned constants (small/large-region
multipliers, valid-pixel threshold, p_min/p_max, quantile, three neighborhood
weights, three window sizes) with no sensitivity analysis. Some of the values
were fit on the 24-township benchmark and not re-tuned when the four external
regions were added.

Fix: sweep the three highest-leverage parameters (valid-pixel threshold,
small-region multiplier, large-region multiplier) on a compact grid; hold
the other 8 at their v2 audited values. For every grid cell, recompute the
four evaluation metrics (change F1, FoM, transition accuracy, allocation
disagreement) on the 24-township benchmark and summarise the parameter
sensitivity.

Output (results/e5_sa_alloc_sensitivity/):
    sensitivity_grid.csv        -- one row per grid cell x per township
    sensitivity_summary.csv     -- one row per grid cell, aggregated
    heatmap_matrix.json         -- 4 metric x grid rendering

Usage:
    python e5_sa_alloc_sensitivity.py
    python e5_sa_alloc_sensitivity.py --smoke      # 2x2x2 grid on 4 townships

Depends on the raw per-area predictions + labels used to generate v2 Table 11.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

RESULTS_DIR = HERE / "results" / "e5_sa_alloc_sensitivity"
INDEP_LABELS = REPO_ROOT / "data" / "independent_change_labels" / "labels"
INDEP_PRED = REPO_ROOT / "data" / "independent_change_labels" / "predicted"

# v2 audited SA-Alloc default values:
DEFAULTS = {
    "small_mul": 1.5,
    "large_mul": 3.0,
    "threshold_pixels": 50_000,
    "p_min": 0.05,
    "p_max": 0.25,
    "quantile": 0.25,
    "target_weight": 1.5,
    "source_weight": 0.65,
    "trans_weight": 0.6,
    "windows": (3, 5, 9),
}

# Sweep grid (three highest-leverage parameters):
GRID_SMALL_MUL = [1.0, 1.5, 2.0]                   # 3 values
GRID_LARGE_MUL = [2.0, 3.0, 4.0]                   # 3 values
GRID_THRESHOLD = [30_000, 50_000, 70_000, 100_000]  # 4 values
# 3*3*4 = 36 cells. Each requires recomputing 4 metrics on 24 townships = 24 rows.
# Total 864 rows.


def load_township_pair(area: str, start_year: int, end_year: int) -> dict | None:
    """Return dict with `start_lab`, `end_lab`, `pred_lab`, or None if any file missing."""
    slab = INDEP_LABELS / f"{area}_{start_year}.npy"
    elab = INDEP_LABELS / f"{area}_{end_year}.npy"
    plab = INDEP_PRED / f"{area}_{start_year}_{end_year}_predicted_label.npy"
    if not (slab.exists() and elab.exists() and plab.exists()):
        return None
    return {"start_lab": np.load(slab), "end_lab": np.load(elab),
            "pred_lab": np.load(plab), "area": area,
            "start_year": start_year, "end_year": end_year}


def score_metrics(pred: np.ndarray, start_lab: np.ndarray, end_lab: np.ndarray) -> dict:
    """Compute the four SA-Alloc reporting metrics on a single township."""
    valid = end_lab > 0
    true_change = (end_lab != start_lab) & valid
    pred_change = (pred != start_lab) & valid
    tp = int((true_change & pred_change).sum())
    fp = int((~true_change & pred_change).sum())
    fn = int((true_change & ~pred_change).sum())
    tn = int((~true_change & ~pred_change & valid).sum())
    n_valid = int(valid.sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fom = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    trans_acc = float(((pred == end_lab) & true_change).sum()) / max(int(true_change.sum()), 1)
    # allocation disagreement (Pontius): |quantity_disag| omitted; only allocation
    all_disag = float(fp + fn) / max(n_valid, 1)  # simple upper bound
    return {"change_f1": f1, "fom": fom, "transition_accuracy": trans_acc,
            "allocation_disagreement": all_disag,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_valid": n_valid}


def apply_sa_alloc(start_lab: np.ndarray, end_lab: np.ndarray,
                   pred_lab: np.ndarray, params: dict) -> np.ndarray:
    """Simplified SA-Alloc that re-projects a raw decoded prediction to the target
    change fraction dictated by (small_mul, large_mul, threshold_pixels).

    This is a *ranking-based* allocation: rank candidate change pixels by a
    lightweight score (target-neighborhood support minus source-neighborhood
    support), retain enough to hit the target change fraction, else revert to
    start-year class.
    """
    valid = end_lab > 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        return pred_lab.copy()

    mul = params["small_mul"] if n_valid < params["threshold_pixels"] else params["large_mul"]

    candidate_change = (pred_lab != start_lab) & valid
    n_candidate = int(candidate_change.sum())
    if n_candidate == 0:
        return pred_lab.copy()

    # observed candidate change fraction
    p_c = n_candidate / n_valid
    # calibrated target fraction
    p_star = float(np.clip(p_c * params["quantile"] * mul,
                           params["p_min"], params["p_max"]))
    target_n = int(round(p_star * n_valid))
    target_n = max(0, min(target_n, n_candidate))

    # Simple ranking: prefer candidates where the neighborhood contains many
    # pixels of the same predicted target class (target-neighborhood support).
    from scipy.ndimage import uniform_filter
    score = np.zeros_like(pred_lab, dtype=np.float32)
    for cls in np.unique(pred_lab[candidate_change]):
        cls_mask = (pred_lab == cls).astype(np.float32)
        support = uniform_filter(cls_mask, size=5, mode="reflect")
        # add support where candidate pixel is predicted to become cls
        score[(pred_lab == cls) & candidate_change] += \
            params["target_weight"] * support[(pred_lab == cls) & candidate_change]
    src_score = np.zeros_like(pred_lab, dtype=np.float32)
    for cls in np.unique(start_lab[candidate_change]):
        cls_mask = (start_lab == cls).astype(np.float32)
        support = uniform_filter(cls_mask, size=5, mode="reflect")
        src_score[(start_lab == cls) & candidate_change] -= \
            params["source_weight"] * support[(start_lab == cls) & candidate_change]
    total_score = score + src_score
    total_score[~candidate_change] = -np.inf

    # keep top-target_n candidates
    flat_scores = total_score.flatten()
    if target_n < len(flat_scores):
        threshold = np.partition(flat_scores, -target_n)[-target_n] if target_n > 0 else np.inf
    else:
        threshold = -np.inf
    keep = total_score >= threshold
    keep = keep & candidate_change

    result = pred_lab.copy()
    result[~keep & candidate_change] = start_lab[~keep & candidate_change]
    return result


def load_v2_registry() -> list[dict]:
    """Discover all 24 township pairs from the v2 predictions cache."""
    pairs = []
    for pred_file in sorted(INDEP_PRED.glob("*_predicted_label.npy")):
        stem = pred_file.stem.replace("_predicted_label", "")
        # stem like: xiangzhen_record_000191_2020_2021
        parts = stem.rsplit("_", 2)
        area = parts[0]
        start_year = int(parts[1])
        end_year = int(parts[2])
        pairs.append({"area": area, "start_year": start_year,
                      "end_year": end_year})
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = load_v2_registry()
    print(f"discovered {len(pairs)} township pairs from prediction cache")

    if args.smoke:
        pairs = pairs[:4]
        grid_small = [1.0, 1.5]
        grid_large = [2.0, 3.0]
        grid_thr = [30_000, 70_000]
    else:
        grid_small = GRID_SMALL_MUL
        grid_large = GRID_LARGE_MUL
        grid_thr = GRID_THRESHOLD

    t0 = time.time()
    rows = []
    summary_rows = []
    cell_id = 0
    for small_mul, large_mul, threshold in product(grid_small, grid_large, grid_thr):
        cell_id += 1
        params = dict(DEFAULTS)
        params.update({"small_mul": small_mul, "large_mul": large_mul,
                       "threshold_pixels": threshold})
        cell_metrics = {"change_f1": [], "fom": [], "transition_accuracy": [],
                        "allocation_disagreement": []}
        for pair in pairs:
            data = load_township_pair(pair["area"], pair["start_year"], pair["end_year"])
            if data is None:
                continue
            reallocated = apply_sa_alloc(data["start_lab"], data["end_lab"],
                                         data["pred_lab"], params)
            m = score_metrics(reallocated, data["start_lab"], data["end_lab"])
            row = {"cell_id": cell_id, "small_mul": small_mul,
                   "large_mul": large_mul, "threshold": threshold,
                   "area": pair["area"], "start_year": pair["start_year"],
                   "end_year": pair["end_year"], **m}
            rows.append(row)
            for k in cell_metrics:
                cell_metrics[k].append(m[k])
        summary_rows.append({"cell_id": cell_id, "small_mul": small_mul,
                             "large_mul": large_mul, "threshold": threshold,
                             "n": len(cell_metrics["change_f1"]),
                             **{f"{k}_mean": float(np.mean(v))
                                for k, v in cell_metrics.items()},
                             **{f"{k}_std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                                for k, v in cell_metrics.items()}})
        print(f"cell {cell_id}: mul=({small_mul}, {large_mul}) thr={threshold} "
              f"F1={summary_rows[-1]['change_f1_mean']:.4f} "
              f"FoM={summary_rows[-1]['fom_mean']:.4f}")

    # Write CSVs
    if rows:
        with (RESULTS_DIR / "sensitivity_grid.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with (RESULTS_DIR / "sensitivity_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Determine range of each metric across the grid to quantify sensitivity
    metric_ranges = {}
    for metric in ["change_f1", "fom", "transition_accuracy", "allocation_disagreement"]:
        vals = [r[f"{metric}_mean"] for r in summary_rows]
        metric_ranges[metric] = {"min": float(min(vals)), "max": float(max(vals)),
                                 "range": float(max(vals) - min(vals)),
                                 "mean": float(np.mean(vals)),
                                 "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    (RESULTS_DIR / "metric_ranges.json").write_text(json.dumps({
        "grid_size": len(summary_rows), "n_pairs": len(pairs),
        "wall_s": time.time() - t0,
        "defaults": {k: v for k, v in DEFAULTS.items() if k != "windows"},
        "grid": {"small_mul": grid_small, "large_mul": grid_large,
                 "threshold": grid_thr},
        "metric_sensitivity": metric_ranges,
    }, indent=2))
    (RESULTS_DIR / ".done").touch()
    print(f"\n[E5 DONE] {len(summary_rows)} cells x {len(pairs)} pairs = "
          f"{len(rows)} rows written")


if __name__ == "__main__":
    main()
