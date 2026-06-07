# -*- coding: utf-8 -*-
"""Paper 5+8 × Paper 12 Ablation — Phase B: static encoder comparison.

Compares AlphaEarth (64-d, L2-normed) vs Prithvi-100M (768-d, L2-normed at
extraction time) without any LDN training, on the same study areas.

Two tables produced:
    Table A1 — per-area persistence cosine similarity by encoder
    Table A2 — LULC linear-probe accuracy by encoder

Both tables use *each encoder's own embedding space*. Cross-encoder cosine
isn't meaningful so we don't report it. The apples-to-apples metric is
"how much does next year resemble this year, in this encoder's space?"

Usage:
    python paper8/compare_encoders.py
    python paper8/compare_encoders.py --areas bishan hetao
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np

PAPER8_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, "D:/adk/data_agent")

AE_DIR = PAPER8_ROOT / "data"  # local Bishan-style AE cache
AE_RAW_DIR = Path("D:/adk/data_agent/weights/raw_data")  # paper 5+8 default 15-area cache
PR_DIR = PAPER8_ROOT / "data" / "prithvi"
LULC_DECODER_AE = Path("D:/adk/data_agent/weights/lulc_decoder_v1.pkl")
RESULTS = PAPER8_ROOT / "results" / "encoder_comparison.json"


def ae_path_for(area: str, year: int) -> Path | None:
    """paper 5+8 raw cache first (emb_{area}_{year}.npy), then local data dir."""
    p1 = AE_RAW_DIR / f"emb_{area}_{year}.npy"
    if p1.exists():
        return p1
    p2 = AE_DIR / f"{area}_emb_{year}.npy"
    if p2.exists():
        return p2
    return None


def list_areas(area_filter: list[str] | None) -> list[str]:
    """Areas with at least one Prithvi .npy. Optionally restrict to a subset."""
    names = sorted({p.stem.rsplit("_emb_", 1)[0] for p in PR_DIR.glob("*_emb_*.npy")})
    if area_filter:
        names = [n for n in names if n in set(area_filter)]
    return names


def load_year_pair(area: str, encoder: str, ya: int, yb: int) -> tuple[np.ndarray, np.ndarray] | None:
    if encoder == "prithvi":
        a = PR_DIR / f"{area}_emb_{ya}.npy"
        b = PR_DIR / f"{area}_emb_{yb}.npy"
        if not (a.exists() and b.exists()):
            return None
        return np.load(a), np.load(b)
    a = ae_path_for(area, ya)
    b = ae_path_for(area, yb)
    if a is None or b is None:
        return None
    return np.load(a), np.load(b)


def cos_sim_pixelwise(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cos(z_ya[r,c], z_yb[r,c]) over all pixels.

    Inputs may already be L2-normed; compute defensively anyway.
    Resamples b to a's shape if mismatched (trivial padding/cropping diff
    between encoders sometimes occurs at the patch edge).
    """
    if a.shape[:2] != b.shape[:2]:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return float(np.sum(an * bn, axis=-1).mean())


def persistence_cossim_per_area(area: str, encoder: str) -> list[float]:
    """Cos sim across all consecutive year pairs that exist."""
    if encoder == "prithvi":
        base = PR_DIR
        years = sorted({int(p.stem.rsplit("_", 1)[1]) for p in base.glob(f"{area}_emb_*.npy")})
    else:
        # AE: collect years from both raw cache and local data dir
        years_raw = {int(p.stem.split("_")[-1]) for p in AE_RAW_DIR.glob(f"emb_{area}_*.npy")}
        years_local = {int(p.stem.rsplit("_", 1)[1]) for p in AE_DIR.glob(f"{area}_emb_*.npy")}
        years = sorted(years_raw | years_local)
    sims = []
    for ya, yb in zip(years[:-1], years[1:]):
        if yb - ya != 1:
            continue
        pair = load_year_pair(area, encoder, ya, yb)
        if pair is None:
            continue
        sims.append(cos_sim_pixelwise(pair[0], pair[1]))
    return sims


def fit_lulc_probe(encoder: str, fit_year: int = 2020) -> dict:
    """Fit a fresh LogisticRegression LULC probe on this encoder's space.

    For AlphaEarth we already have a pretrained decoder; we still refit on
    the same in-domain pixels so the comparison is symmetric.

    Labels come from decoding AlphaEarth via lulc_decoder_v1.pkl, which is
    the closest thing to ground truth we have available without re-pulling
    ESA WorldCover. Both encoders are evaluated against these AE-derived
    pseudo-labels: this is biased *toward* AE but gives a relative number.
    """
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    decoder_ae = joblib.load(LULC_DECODER_AE)

    # Enumerate areas with features for the requested year
    if encoder == "prithvi":
        feat_files = sorted(PR_DIR.glob(f"*_emb_{fit_year}.npy"))
        if not feat_files:
            return {"error": f"no prithvi files for year {fit_year}"}
        areas = [(f, f.stem.rsplit("_emb_", 1)[0]) for f in feat_files]
    else:
        # Pull AE files from both caches
        from_raw = [(p, p.stem.replace(f"emb_", "").rsplit("_", 1)[0])
                    for p in AE_RAW_DIR.glob(f"emb_*_{fit_year}.npy")]
        from_local = [(p, p.stem.rsplit("_emb_", 1)[0])
                      for p in AE_DIR.glob(f"*_emb_{fit_year}.npy")]
        areas = from_raw + from_local
        if not areas:
            return {"error": f"no AE files for year {fit_year}"}

    Xs, ys = [], []
    for f, area in areas:
        ae_p = ae_path_for(area, fit_year)
        if ae_p is None:
            continue
        ae_arr = np.load(ae_p).reshape(-1, 64)
        labels = decoder_ae.predict(ae_arr)
        feat_arr_full = np.load(f)
        feat_arr = feat_arr_full.reshape(-1, feat_arr_full.shape[-1])
        h_ae = ae_arr.shape[0]
        h_pr = feat_arr.shape[0]
        if h_ae != h_pr:
            n = min(h_ae, h_pr)
            ae_arr = ae_arr[:n]
            feat_arr = feat_arr[:n]
            labels = labels[:n]
        Xs.append(feat_arr)
        ys.append(labels)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    if X.shape[0] < 4:
        return {"error": f"too few samples: {X.shape[0]}"}
    # Drop classes with <2 samples (too rare for stratified split)
    from collections import Counter
    counts = Counter(y.tolist())
    keep_mask = np.array([counts[v] >= 2 for v in y])
    X, y = X[keep_mask], y[keep_mask]
    stratify = y if min(Counter(y.tolist()).values()) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=stratify)
    clf = LogisticRegression(max_iter=200, n_jobs=-1)
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xte, yte))
    return {"accuracy": acc, "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]), "n_features": int(X.shape[1])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--areas", nargs="*", help="Area subset; default = all areas with Prithvi outputs")
    p.add_argument("--probe-year", type=int, default=2020)
    args = p.parse_args()

    areas = list_areas(args.areas)
    if not areas:
        sys.exit("No Prithvi outputs found in data/prithvi/. Run extract_prithvi_embeddings.py first.")

    print(f"=== Table A1: persistence cos sim per area ===")
    table_a1 = {}
    for area in areas:
        ae = persistence_cossim_per_area(area, "ae")
        pr = persistence_cossim_per_area(area, "prithvi")
        table_a1[area] = {
            "alphaearth": {"mean": mean(ae) if ae else None, "std": stdev(ae) if len(ae) > 1 else 0.0, "n": len(ae)},
            "prithvi": {"mean": mean(pr) if pr else None, "std": stdev(pr) if len(pr) > 1 else 0.0, "n": len(pr)},
        }
        ae_str = f"{table_a1[area]['alphaearth']['mean']:.4f}" if ae else "  --   "
        pr_str = f"{table_a1[area]['prithvi']['mean']:.4f}" if pr else "  --   "
        print(f"  {area:20s}  AE={ae_str}  Prithvi={pr_str}")

    print(f"\n=== Table A2: LULC linear-probe accuracy (year {args.probe_year}) ===")
    table_a2 = {
        "alphaearth": fit_lulc_probe("ae", args.probe_year),
        "prithvi": fit_lulc_probe("prithvi", args.probe_year),
    }
    for k, v in table_a2.items():
        if "error" in v:
            print(f"  {k:12s}  ERROR: {v['error']}")
        else:
            print(f"  {k:12s}  acc={v['accuracy']:.4f}  ({v['n_features']}-d, n_train={v['n_train']})")

    out = {"table_a1_persistence_cossim": table_a1, "table_a2_lulc_probe": table_a2}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
