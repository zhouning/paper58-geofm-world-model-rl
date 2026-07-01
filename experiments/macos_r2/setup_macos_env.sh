#!/usr/bin/env bash
# Paper 58 RSE R2 — macOS environment self-check + missing-dep top-up.
# Assumes farmland-mpc-pure conda env is already active.
set -euo pipefail

echo "=== Paper 58 R2 macOS env self-check ==="
python_bin="$(command -v python)"
echo "python: $python_bin"
python -c "import sys; print('python version:', sys.version.split()[0])"

echo "--- checking required Python packages ---"
python <<'PY'
import sys
required = {
    "numpy": None,
    "pandas": None,
    "scipy": None,
    "sklearn": None,
    "torch": None,
    "tqdm": None,
    "ee": "earthengine-api",
}
missing = []
for mod, pip_name in required.items():
    try:
        __import__(mod)
        print(f"  OK  {mod}")
    except ImportError:
        pkg = pip_name or mod
        missing.append(pkg)
        print(f"  MISS {mod} (pip install {pkg})")
if missing:
    print("\nMissing packages:")
    print("  pip install " + " ".join(missing))
    sys.exit(1)
PY

echo "--- checking torch device ---"
python <<'PY'
import torch
if torch.backends.mps.is_available():
    print("  device: mps (Apple Silicon GPU OK)")
elif torch.cuda.is_available():
    print("  device: cuda")
else:
    print("  device: cpu (all experiments run on CPU fallback)")
PY

echo "--- checking Prithvi weights ---"
default_pri="${PRITHVI_WEIGHTS:-$HOME/paper58-r2/weights/Prithvi_100M.pt}"
if [ -f "$default_pri" ]; then
    echo "  OK  $default_pri"
else
    echo "  MISS  $default_pri  — set PRITHVI_WEIGHTS or download from"
    echo "        https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M"
    exit 1
fi

echo "--- checking GEE auth ---"
python <<'PY'
try:
    import ee
    ee.Initialize()
    _ = ee.Number(1).getInfo()
    print("  OK  GEE auth active")
except Exception as e:
    print(f"  MISS  GEE not authenticated: {e!s}")
    print("        run: earthengine authenticate")
    raise SystemExit(1)
PY

echo "--- checking geoadapter (PrithviBackbone) ---"
python <<'PY'
try:
    from geoadapter.models.prithvi import PrithviBackbone
    print("  OK  geoadapter PrithviBackbone importable")
except ImportError as e:
    print(f"  MISS  geoadapter not installed: {e!s}")
    print("        cd external/alphaearth_system && pip install -e .")
    raise SystemExit(1)
PY

echo ""
echo "All checks passed. You can now run: bash run_all_macos.sh --all"
