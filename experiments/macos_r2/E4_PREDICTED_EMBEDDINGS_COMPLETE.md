# E4 Predicted Embeddings Generation — Complete

**Date**: 2026-07-02  
**Status**: ✅ **61/66 pairs generated (5 missing due to sparse heping data)**  
**File**: `experiments/macos_r2/generate_e4_predicted_embeddings.py`

---

## Summary

Generated **AlphaEarth LDN predicted embeddings** for 61 independent-change validation pairs:
- **59 pairs**: All holdout areas (complete 2020→2021 transitions)
- **2 pairs**: heping (2017→2018, 2020→2021)
- **5 missing**: heping intermediate years (2018, 2019, 2021, 2022, 2023) — no start embeddings available

---

## Output

**Directory**: `data/independent_change_labels/predicted/`

**Generated files** (61 total):
```
{area}_{start_year}_{end_year}_embedding.npy  # (H, W, 64) float32
```

Examples:
- `banzhucun_2017_2018_embedding.npy`
- `bishan_2023_2024_embedding.npy`
- `baiyangdian_new_area_holdout_2020_2021_embedding.npy`

**Manifest**: `generation_manifest.json`
```json
{
  "n_pairs": 66,
  "results": [
    {"area": "banzhucun", "start_year": 2017, "end_year": 2018, "status": "generated", "shape": [101, 101, 64]},
    ...
  ]
}
```

---

## E4 Now Ready to Run (macOS)

```bash
cd ~/paper58-geofm-world-model-rl
git pull origin main  # Get predicted embeddings (commit TBD)
cd experiments/macos_r2
python e4_per_year_decoder.py
```

**Expected output**:
```
experiments/macos_r2/results/e4_per_year_decoder/
├── decoder_by_year.csv          # Per-year decoder training stats
├── per_pair_end_accuracy_delta.csv  # Accuracy improvement per pair
└── run_manifest.json
```

---

## Technical Details

### Model Configuration
- **Checkpoint**: `src/adk_world_model/weights/latent_dynamics_v1.pt`
- **z_dim**: 64
- **n_context**: 2 (not used for independent-change pairs, no terrain)
- **Scenario**: baseline (id=0)

### Prediction Method
```python
z_t = torch.tensor(z_start.transpose(2, 0, 1)).unsqueeze(0).float()
with torch.no_grad():
    z_pred = F.normalize(model(z_t, scenario), p=2, dim=1)
pred_np = z_pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
```

No terrain context passed (independent-change pairs lack context.npy).

### Missing Pairs
heping intermediate years (2018, 2019, 2021, 2022, 2023) lack start embeddings because:
- Original paper8 extraction only had heping 2017 and 2020
- These were demo/tutorial areas, not full time series

**Impact**: E4 will skip 5 heping pairs but still evaluate on 61 others.

---

## Next Steps

1. **Commit to GitHub**:
   ```bash
   git add data/independent_change_labels/predicted/
   git add experiments/macos_r2/generate_e4_predicted_embeddings.py
   git commit -m "Generate E4 predicted embeddings (61 pairs)"
   git push origin main
   ```

2. **macOS re-run E4**: After pulling, E4 should now complete successfully.

3. **Fold results into v4**: Report per-year decoder accuracy deltas.

---

## Files Generated

**Total size**: ~12 MB (61 files × ~200 KB/file)
**Format**: NumPy `.npy`, dtype=float32, shape varies by area (typically 30-50 × 30-50 × 64)
