# -*- coding: utf-8 -*-
"""Paper 5+8 × Paper 12 Ablation — Colab driver for Prithvi feature extraction.

Convention mirrors colab_paper8.py: upload zip to Drive, unzip in /content,
run this script with --areas/--years, sync results back to Drive.

Setup in Colab (paste in cell 0):
    from google.colab import drive
    drive.mount('/content/drive')
    !mkdir -p /content/paper58_ablation && cd /content/paper58_ablation
    !cp /content/drive/MyDrive/paper58_ablation.zip .
    !unzip -qo paper58_ablation.zip
    !pip install -q earthengine-api torch torchvision rasterio

The zip should contain (run package step locally first):
    extract_prithvi_embeddings.py
    geoadapter/                   # symlink or copy of D:/adk/AlphaEarth-System/geoadapter
    Prithvi_100M.pt               # 350 MB Prithvi checkpoint
    world_model.py                # for DEFAULT_TRAINING_AREAS

Run (in another Colab cell):
    !python colab_extract_prithvi.py --smoke         # bishan 2020 only (~2 min)
    !python colab_extract_prithvi.py --all           # full 17 × 8 run (~4 h on L4)

Output: /content/paper58_ablation/data/prithvi/*.npy, synced to
        /content/drive/MyDrive/prithvi_paper58/
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/content/paper58_ablation")
DRIVE_OUT = Path("/content/drive/MyDrive/prithvi_paper58")
LOCAL_OUT = WORK / "data" / "prithvi"

sys.path.insert(0, str(WORK))


def ee_auth():
    """Authenticate Earth Engine — opens browser flow first time."""
    import ee
    try:
        ee.Initialize()
        print("GEE already authenticated")
    except Exception:
        ee.Authenticate()
        ee.Initialize()
        print("GEE authenticated")


def sync_to_drive():
    DRIVE_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in LOCAL_OUT.glob("*.npy"):
        dst = DRIVE_OUT / src.name
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy(src, dst)
            n += 1
    meta = LOCAL_OUT / "metadata.json"
    if meta.exists():
        shutil.copy(meta, DRIVE_OUT / "metadata.json")
    print(f"Synced {n} new files to {DRIVE_OUT}")


def restore_from_drive():
    """If Drive has previous extractions, copy them down so --force isn't needed."""
    if not DRIVE_OUT.exists():
        return
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in DRIVE_OUT.glob("*.npy"):
        dst = LOCAL_OUT / src.name
        if not dst.exists():
            shutil.copy(src, dst)
            n += 1
    if n:
        print(f"Restored {n} files from Drive — these will be skipped")


def gpu_check():
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(out.stdout.split("\n")[0] if out.stdout else "no GPU")


def run_extract(args_list):
    cmd = [sys.executable, str(WORK / "extract_prithvi_embeddings.py")] + args_list
    print(">>", " ".join(cmd))
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(WORK)).returncode
    print(f"=== exit={rc}, wall={time.time() - t0:.1f}s ===")
    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Bishan 2020 only (~2 min)")
    p.add_argument("--all", action="store_true", help="All 17 areas × 8 years (~4 h on L4)")
    p.add_argument("--areas", nargs="*")
    p.add_argument("--years", nargs="*", type=int)
    p.add_argument("--skip-restore", action="store_true")
    p.add_argument("--no-sync", action="store_true")
    args = p.parse_args()

    if not (WORK / "extract_prithvi_embeddings.py").exists():
        sys.exit(f"missing {WORK}/extract_prithvi_embeddings.py — unzip paper58_ablation.zip first")

    gpu_check()
    ee_auth()
    if not args.skip_restore:
        restore_from_drive()

    if args.smoke:
        rc = run_extract(["--areas", "bishan", "--years", "2020"])
    elif args.all:
        rc = run_extract([])
    else:
        cli = []
        if args.areas:
            cli += ["--areas"] + args.areas
        if args.years:
            cli += ["--years"] + [str(y) for y in args.years]
        rc = run_extract(cli)

    if not args.no_sync:
        sync_to_drive()
    sys.exit(rc)


if __name__ == "__main__":
    main()
