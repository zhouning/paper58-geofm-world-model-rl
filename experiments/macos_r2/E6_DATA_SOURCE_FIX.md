# E6 Data Source Fix — Filter to Complete 8-Year Time Series

**Date**: 2026-07-02  
**Status**: ✅ **Fixed — now defaults to 8-year minimum**  
**File**: `experiments/macos_r2/e6_expand_areas.py`

---

## Problem

E6 originally mixed two data sources:
1. **30 areas** from `experiments/paper8/data` — complete 2017-2024 (8 years)
2. **43 areas** from `data/independent_change_labels/embeddings` — only 2020-2021 (2 years)

This produced a **73-area baseline** with inconsistent temporal coverage, making it unclear whether differences were due to:
- Genuine dynamics patterns
- Artifact of 2-year vs 8-year time series

---

## Fix

Added `--min-years` parameter to filter areas by minimum consecutive years:

```python
def discover_eval_area_sources(roots, min_years=2):
    """
    Args:
        min_years: Minimum consecutive years required (default 2)
                   Set to 8 to include only complete 2017-2024 time series
    """
    # ... filters areas with < min_years ...
```

**Default behavior** (changed):
- Old: `min_years=2` (includes both 8-year and 2-year areas → 73 total)
- **New**: `min_years=8` (includes only complete time series → 30 total)

---

## Usage (macOS)

### Option 1: Use only complete 8-year areas (recommended, now default)
```bash
cd ~/paper58-geofm-world-model-rl/experiments/macos_r2
python e6_expand_areas.py --eval-only --min-years 8
```
**Output**: 30 areas (paper8/data only)

### Option 2: Include 2-year areas (original behavior, now requires explicit flag)
```bash
python e6_expand_areas.py --eval-only --min-years 2
```
**Output**: 73 areas (paper8/data + independent_change_labels)

### Option 3: Custom threshold
```bash
python e6_expand_areas.py --eval-only --min-years 4
```
**Output**: Areas with ≥4 consecutive years

---

## Expected Results

### 30-area baseline (8-year minimum, recommended)
- All areas from `experiments/paper8/data`
- Complete 2017-2024 time series
- Comparable temporal coverage across all areas
- Uses 2023→2024 transition (last year pair)

### 73-area baseline (2-year minimum, original)
- 30 from `experiments/paper8/data` (8 years)
- 43 from `independent_change_labels` (2 years)
- **Mixed temporal coverage** (not recommended for paired inference)
- 2-year areas only have 2020→2021 transition

---

## Recommendation for v4 Manuscript

**Use the 30-area baseline (8-year minimum)** for the following reasons:

1. **Consistent temporal coverage**: All areas have complete 2017-2024 time series
2. **Same evaluation protocol**: All use 2023→2024 transition (last year pair)
3. **Cleaner interpretation**: Differences reflect dynamics patterns, not data artifacts
4. **Reviewer-requested expansion**: From 10 → 30 areas (3× increase, satisfies M1)

The 43 two-year areas can be used for **independent validation** (separate section) but should **not** be mixed with the 30-area baseline in paired inference.

---

## Next Steps (macOS)

```bash
cd ~/paper58-geofm-world-model-rl
git pull origin main  # Get the fixed code (commit TBD)
cd experiments/macos_r2
python e6_expand_areas.py --eval-only --min-years 8  # Run with 30-area filter
```

**Expected output**:
```json
{
  "n": 30,
  "mean": -0.0055,
  "wilcoxon_p": 0.XX,
  "eval_sources": {
    "n_areas": 30,
    "roots": {
      ".../experiments/paper8/data": 30
    }
  }
}
```

---

## Documentation

Updated:
- `discover_eval_area_sources()` — added `min_years` parameter with filtering logic
- `cmd_eval()` — added `min_years` parameter, passed to discovery function
- `main()` — added `--min-years` CLI argument (default 8)

All changes maintain backward compatibility via the `--min-years` flag.
