# -*- coding: utf-8 -*-
"""E4 (RSE reviewer S6): per-year decoder retraining + independent-change re-run.

Reviewer flagged: the v2 linear-probe decoder was trained only on 2020
AlphaEarth embeddings but applied to decode predictions for 2017-2023. This
conflates dynamics error with decoder year-drift error.

Fix: for each year in {2017, 2018, 2019, 2020, 2021, 2022, 2023}, train a
per-year logistic-regression decoder on (AlphaEarth embedding, ESRI LULC)
paired pixels, then re-decode each independent-change validation pair using
the *end-year* decoder (rather than a single 2020-trained decoder).

Output (results/e4_per_year_decoder/):
    decoder_by_year.csv        — year, n_samples, mean_cv_acc, macro_f1
    per_pair_end_accuracy_delta.csv - pair_id, v2_end_acc, retrained_end_acc, delta

Usage:
    python e4_per_year_decoder.py
    python e4_per_year_decoder.py --smoke      # 2020 only

Depends on:
  - data/independent_change_labels/{labels,predicted}/*.npy
  - AlphaEarth per-pixel embeddings cached under paper8/data/*_emb_*.npy
  - scripts/rse_revision/evaluate_independent_change_validation.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PAPER8_ROOT = REPO_ROOT / "experiments" / "paper8"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAPER8_ROOT))
from run_markers import mark_done  # noqa: E402

RESULTS_DIR = HERE / "results" / "e4_per_year_decoder"
INDEP_LABELS = REPO_ROOT / "data" / "independent_change_labels" / "labels"
INDEP_PRED = REPO_ROOT / "data" / "independent_change_labels" / "predicted"
AE_DIR = REPO_ROOT / "data" / "independent_change_labels" / "embeddings"
AE_DIRS = [AE_DIR, PAPER8_ROOT / "data"]

# Same 6-class merge rule used in v2 §4.6:
# raw ESRI: 1=water, 2=trees, 4=flooded_veg, 5=crops, 7=built, 8=bare, 9=snow, 10=clouds, 11=rangeland
ESRI_MERGE = {
    1: 1,   # water
    4: 1,   # flooded_veg -> water
    2: 2,   # trees
    5: 3,   # crops
    7: 4,   # built
    8: 5,   # bare
    9: 5,   # snow -> bare/snow
    10: 0,  # clouds -> drop (mark as 0 = unlabeled)
    11: 6,  # rangeland
}
CLASS_NAMES = {1: "water", 2: "trees", 3: "crops", 4: "built",
               5: "bare/snow", 6: "rangeland"}


def merge_labels(arr: np.ndarray) -> np.ndarray:
    """Apply the 6-class merge rule; 0 = drop-mask."""
    out = np.zeros_like(arr, dtype=np.int32)
    for raw, merged in ESRI_MERGE.items():
        out[arr == raw] = merged
    return out


def find_label_path(area: str, year: int) -> Path | None:
    for candidate in (
        INDEP_LABELS / f"{area}_lulc_{year}.npy",
        INDEP_LABELS / f"{area}_{year}.npy",
    ):
        if candidate.exists():
            return candidate
    return None


def embedding_files_for_year(year: int) -> list[tuple[str, Path]]:
    area_to_path: dict[str, Path] = {}
    for root in AE_DIRS:
        for emb_file in sorted(root.glob(f"*_emb_{year}.npy")):
            area = emb_file.stem.rsplit("_emb_", 1)[0]
            area_to_path.setdefault(area, emb_file)
    return sorted(area_to_path.items())


def train_year_decoder(year: int) -> dict:
    """Sample (embedding, label) pixel pairs for every area available at this year;
    fit 5-fold LogReg; return CV mean accuracy + macro F1 + sample count."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import KFold

    X_all, y_all = [], []
    per_area_seen = {}
    for area, emb_file in embedding_files_for_year(year):
        emb = np.load(emb_file)              # (H, W, 64)
        label_file = find_label_path(area, year)
        if label_file is None:
            continue
        lab = np.load(label_file)
        if lab.shape[:2] != emb.shape[:2]:
            continue
        lab_m = merge_labels(lab)
        mask = lab_m > 0
        n_local = int(mask.sum())
        if n_local == 0:
            continue
        # subsample to avoid dominance from any single area
        idx = np.argwhere(mask)
        if len(idx) > 800:
            sel = np.random.default_rng(20260702 + year).choice(len(idx), 800, replace=False)
            idx = idx[sel]
        X_local = emb[idx[:, 0], idx[:, 1]]
        y_local = lab_m[idx[:, 0], idx[:, 1]]
        X_all.append(X_local)
        y_all.append(y_local)
        per_area_seen[area] = int(len(idx))

    if not X_all:
        return {"year": year, "n_samples": 0, "reason": "no_paired_data"}
    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    for tr, te in kf.split(X):
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X[tr], y[tr])
        preds = clf.predict(X[te])
        accs.append((preds == y[te]).mean())
        f1s.append(f1_score(y[te], preds, average="macro"))

    # fit final model on all data for downstream use
    final = LogisticRegression(max_iter=1000, C=1.0)
    final.fit(X, y)

    ck_dir = HERE / "weights" / "e4"
    ck_dir.mkdir(parents=True, exist_ok=True)
    import pickle
    ck = ck_dir / f"decoder_year_{year}.pkl"
    with ck.open("wb") as f:
        pickle.dump({"clf": final, "class_labels": sorted(set(y.tolist())),
                     "class_names": CLASS_NAMES, "year": year,
                     "cv_acc": float(np.mean(accs)),
                     "cv_macro_f1": float(np.mean(f1s))}, f)
    return {"year": year, "n_samples": int(len(X)),
            "n_areas": len(per_area_seen),
            "per_area_samples": per_area_seen,
            "cv_acc_mean": float(np.mean(accs)), "cv_acc_std": float(np.std(accs)),
            "cv_macro_f1_mean": float(np.mean(f1s)),
            "cv_macro_f1_std": float(np.std(f1s)),
            "checkpoint": str(ck)}


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--years", nargs="*", type=int,
                   default=[2017, 2018, 2019, 2020, 2021, 2022, 2023])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.years = [2020]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    for year in args.years:
        print(f"training decoder for year {year} ...", end=" ", flush=True)
        r = train_year_decoder(year)
        print(f"n={r.get('n_samples')} acc={r.get('cv_acc_mean', float('nan')):.4f}")
        rows.append(r)

    with (RESULTS_DIR / "decoder_by_year.csv").open("w", newline="") as f:
        keys = ["year", "n_samples", "n_areas", "cv_acc_mean", "cv_acc_std",
                "cv_macro_f1_mean", "cv_macro_f1_std", "checkpoint"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    (RESULTS_DIR / "training_manifest.json").write_text(json.dumps({
        "years": args.years, "wall_s": time.time() - t0,
        "class_merge_rule": ESRI_MERGE,
        "class_names": {int(k): v for k, v in CLASS_NAMES.items()},
    }, indent=2))

    # Downstream: redecode the 11 independent-change pairs (Wuyi 0% excluded)
    # using the *end-year* decoder, and compare end_accuracy against the
    # v2 revision_results CSV.
    v2_csv = REPO_ROOT / "paper" / "rse_submission_paper58" / "revision_results" / "independent_change_validation_by_area.csv"
    delta_rows = []
    if v2_csv.exists():
        import pandas as pd
        v2 = pd.read_csv(v2_csv)
        for _, r in v2.iterrows():
            end_year = int(r["end_year"])
            if r["true_change_pixels"] == 0:
                continue  # Wuyi 2020-2021 excluded
            ck = HERE / "weights" / "e4" / f"decoder_year_{end_year}.pkl"
            if not ck.exists():
                delta_rows.append({"pair_id": f"{r['area']}_{r['start_year']}-{end_year}",
                                   "end_year": end_year,
                                   "v2_end_accuracy": float(r["model_end_accuracy"]),
                                   "retrained_end_accuracy": None,
                                   "delta": None,
                                   "status": "decoder_missing"})
                continue
            # Load retrained decoder + apply to predicted embedding grid
            import pickle
            with ck.open("rb") as f:
                pkg = pickle.load(f)
            clf = pkg["clf"]
            pred_emb_file = INDEP_PRED / f"{r['area']}_{r['start_year']}_{end_year}_embedding.npy"
            end_label_file = find_label_path(str(r["area"]), end_year)
            start_label_file = find_label_path(str(r["area"]), int(r["start_year"]))
            if not (pred_emb_file.exists() and end_label_file is not None and start_label_file is not None):
                delta_rows.append({"pair_id": f"{r['area']}_{r['start_year']}-{end_year}",
                                   "end_year": end_year,
                                   "v2_end_accuracy": float(r["model_end_accuracy"]),
                                   "retrained_end_accuracy": None,
                                   "delta": None,
                                   "status": "prediction_missing"})
                continue
            emb = np.load(pred_emb_file)  # (H, W, 64)
            end_lab = merge_labels(np.load(end_label_file))
            preds = clf.predict(emb.reshape(-1, emb.shape[-1])).reshape(emb.shape[:2])
            valid = end_lab > 0
            acc = float((preds[valid] == end_lab[valid]).mean()) if valid.any() else float("nan")
            delta_rows.append({"pair_id": f"{r['area']}_{r['start_year']}-{end_year}",
                               "end_year": end_year,
                               "v2_end_accuracy": float(r["model_end_accuracy"]),
                               "retrained_end_accuracy": acc,
                               "delta": acc - float(r["model_end_accuracy"]),
                               "status": "ok"})
    with (RESULTS_DIR / "per_pair_end_accuracy_delta.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()) if delta_rows
                          else ["pair_id"])
        w.writeheader()
        w.writerows(delta_rows)

    mark_done(RESULTS_DIR, smoke=args.smoke)
    print(f"\n[E4 DONE] {len(rows)} decoders trained, "
          f"{len(delta_rows)} independent-change pairs re-evaluated")


if __name__ == "__main__":
    main()
