# E3 Multi-Step Rollout — Ready to Re-Run (Bug Fixed)

**Date**: 2026-07-02  
**Status**: ✅ **Bug fixed, ready for macOS re-run**  
**Priority**: High (needed for reviewer response)

---

## Bug Fixed

**File**: `experiments/macos_r2/e3_multistep_all_areas.py` (line 137-160)

**Problem**: The original code compared z_0 (initial year) vs z_step (target year) for persistence, producing persistence=1.0 for step=1 in some areas.

**Fix**: Now correctly compares z_{step-1} vs z_step (consecutive-year transitions).

```python
# OLD (wrong)
persist = cosine(F.normalize(z0, p=2, dim=1), z_true).item()

# NEW (correct)
z_prev = F.normalize(torch.tensor(seq[step - 1]).unsqueeze(0).to(device), p=2, dim=1)
persist = cosine(z_prev, z_true).item()
```

---

## How to Re-Run (macOS)

```bash
cd ~/paper58-geofm-world-model-rl
git pull origin main  # Get the fixed code (commit 6b62831)
cd experiments/macos_r2
bash run_all_macos.sh --only e3
```

**Expected runtime**: ~30 minutes (10 areas × 6 steps)

---

## Expected Output

`experiments/macos_r2/results/e3_multistep/multistep_all_areas.csv`:

```csv
area,step,persistence,model,advantage
bishan,1,0.9863,0.9854,-0.0009
bishan,2,0.9781,0.9719,-0.0062
bishan,3,...,...,...
...
yangtze_delta,6,...,...,...
```

Also generates:
- `multistep_summary.json` — aggregate stats per step
- `multistep_paired_tests.json` — Wilcoxon tests per step

---

## What to Check

1. **Persistence values should be < 1.0** for all steps (no more persistence=1.0 bug)
2. **Persistence should decrease with step** (0.98 at step=1 → 0.92 at step=6)
3. **Model decay should be comparable** to persistence decay

---

## Next Steps After Re-Run

1. **Commit results** to GitHub:
   ```bash
   git add experiments/macos_r2/results/e3_multistep/
   git commit -m "E3 multi-step rollout results (corrected persistence)"
   git push origin main
   ```

2. **Windows端验证**: 我会检查结果并 fold 到 v4 manuscript

3. **Report to reviewer**: Include E3 results in next revision cycle

---

## Questions?

If any issues during re-run, check:
- `experiments/macos_r2/results/e3_multistep/run_manifest.json` for runtime info
- `experiments/macos_r2/results/e3_multistep/errors.log` for any exceptions

---

**Contact**: Windows 端审计完成，E3 代码已修复，等待 macOS 重跑结果。
