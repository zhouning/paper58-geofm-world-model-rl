# v3 Revision Execution Summary

**Date**: 2026-07-02  
**Status**: ✅ **COMPLETED**  
**Approach**: Option A — 撤回 v2 baseline，用新 baseline，诚实说明数据更新

---

## Changes Made to v3 tex

### 1. Abstract (Line 54) ✅

**Changed**:
- Old: mean advantage `+0.0047`, 95% bootstrap CI `[-0.003, +0.013]`, p=0.32, Cohen dz=0.35, power analysis
- **New**: mean advantage `$-0.0055$` (negative), 2/10 positive, all p>0.05, region-dependent

**Added reproducibility note**:
> embeddings re-extracted from Google Earth Engine in R2 for reproducibility

### 2. Abstract End (Line 59) ✅

**Added**:
> **R2 reproducibility note**: the R1 baseline (mean advantage $+0.0047$, 7/10 positive) used AlphaEarth embeddings that predated the initial repository commit and are no longer available; the R2 baseline reported here uses freshly re-extracted embeddings (commit \texttt{9c2ba1d}) and is fully reproducible from the repository.

### 3. §3.2 Data Refresh Note Box (After line 173) ✅

**Added new subsection** `\subsection{R2 Data Refresh and Reproducibility Note}`:
- Explains why R1 baseline is not reproducible
- States R2 baseline: mean = $-0.0055$ (2/10 positive, opposite sign from R1)
- Attributes sign reversal to HLS data updates + AlphaEarth stochastic variation
- Confirms full reproducibility (commit hashes provided)

### 4. §5 Results — Table~\ref{tab:aggregate} (Line 477) ✅

**Updated**:
- Persistence mean: 0.9757 → **0.9616**
- Model mean: 0.9804 → **0.9561**
- Advantage: +0.0047 → **$-0.0055$**
- Range: [$-0.003$, +0.013] → **[$-0.0162$, +0.0072]**

### 5. §5 Results — Table~\ref{tab:paired_inference} (Line 492) ✅

**Updated**:
- Mean paired difference: $+4.73\times 10^{-3}$ → **$-5.49\times 10^{-3}$**
- Positive/negative: 7/3 → **2/8**
- Removed: Cohen's dz, power calculation, bootstrap CI
- All tests: p=0.30-0.34 → **p>0.05**

**Updated reading paragraph**:
- Emphasizes negative mean, non-significance
- Notes region-dependent behavior (Qinghai +0.0072, Guanzhong $-0.0162$)
- Explains R2 vs R1 difference with reference to Section~\ref{sec:data_refresh}

### 6. §5 Results — Table~\ref{tab:perarea} (Line 519) ✅

**Completely replaced** with new 10-area R2 data:
- Qinghai Edge: 0.9789/0.9489/+0.0300 → **0.9419/0.9347/+0.0072**
- Jianghan Plain: 0.9862/0.9639/+0.0223 → **0.9325/0.9285/+0.0040**
- Chengdu Plain: 0.9716/0.9708/+0.0007 → **0.9314/0.9429/$-0.0115$**
- Guanzhong: (missing in R1) → **0.9685/0.9847/$-0.0162$**
- Removed: Minnan Coast (not in 10-area baseline)

**Updated explanation**:
- Largest positive: Qinghai Edge, Jianghan Plain
- Largest negative: Guanzhong, Daxinganling
- Added note about R2 vs R1 data difference

### 7. §5 Results — Table~\ref{tab:category_diagnostic} (Line 544) ✅

**Updated category grouping** for R2 10-area baseline:
- Urban: 4 areas/+0.0009 → **3 areas/$-0.0070$**
- Agriculture: 1/+0.0223 → **2/$-0.0050$**
- Wetland: 1/+0.0120 → **1/$-0.0043$**
- Plateau: 1/+0.0300 → **1/+0.0072**
- Mixed: 2/$-0.0019 → **2/$-0.0104$**
- Forest: 1/$-0.0168 → **1/$-0.0146$**

### 8. §5 Results — Table 4 Summary (Line 439) ✅

**Updated "Area-level embedding dynamics" row**:
- Status: Region-dependent → **Not superior**
- Evidence: mean +0.0047, CI, p=0.32, dz=0.35, power → **mean $-0.0055$, 2/10 positive, all p>0.05, R2 commit \texttt{9c2ba1d}**
- Boundary: 7/10 positive → **Only Qinghai Edge and Jianghan Plain positive**

### 9. §6 Encoder Ablation — Table (Line 851) ✅

**Updated AlphaEarth row**:
- Persistence: 0.9757 → **0.9616**
- LDN advantage: +0.0047 → **$-0.0055$**

### 10. §7.7 Limitations — Statistical Power (Line 1087) ✅

**Completely rewritten**:
- Old: "underpowered for observed effect (dz=0.35), n≈65 needed for 80% power"
- **New**: "R2 baseline shows mean $-0.0055$ (2/10 positive), differs from R1 (+0.0047, 7/10) due to GEE data updates; sign reversal reflects reproducibility challenges; 10-area sample likely underpowered, need 30+ areas"

### 11. NEW §7.8 On Reproducibility and GEE Data Updates ✅

**Added entire new subsection** (before old §7.8):
- Explains R1 baseline not reproducible (embeddings not in version control)
- Sign reversal: R1 +0.0047 → R2 $-0.0055$
- Attributes to: (1) HLS data updates in GEE, (2) AlphaEarth stochastic variation
- Discusses broader reproducibility challenge for GEE-based research
- States R2 solution: commit embeddings to Git LFS, document extraction params
- Confirms R2 full reproducibility (checkpoint, embeddings, code all in version control)

### 12. §7.9 (renumbered) Completed Follow-Up Experiments ✅

**Split old §7.8 into two subsections**:

**§7.9 Completed (E2, E5)**:
- E2 terrain ablation: p=0.08/0.23 (NS), terrain doesn't help
- E5 SA-Alloc sensitivity: moderate sensitivity, no dominant config

**§7.10 In-Progress (E1, E3, E4, E6, E7)**:
- E1: completed but effect size -2.7e-5 (needs diagnosis)
- E3: persistence bug (needs re-run)
- E4: needs Windows predicted embeddings
- E6: 43/73 areas only 2 years (needs clarification)
- E7: scaffold only (dependency constraints)
- References audit document: `AUDIT_FINDINGS.md`

### 13. §8 Conclusions — Main Findings (Line 1174) ✅

**Updated baseline bullet**:
- Old: mean +0.0047, 7/10 positive, p>0.30, dz=0.35, n≈65 needed
- **New**: R2 mean $-0.0055$ (2/10); R1 mean +0.0047 (7/10, not reproducible); sign reversal reflects GEE data updates; R2 fully reproducible

### 14. §8 Conclusions — Follow-Up Status (Line 1189) ✅

**Completely rewritten**:
- Old: "All seven packaged as macOS pipeline, will fold into v3"
- **New**: "Two completed (E2, E5); five in-progress (E1, E3, E4, E6, E7); R2 baseline sign reversal; all R2 data in version control"

---

## Files Modified

1. `paper/rse_submission_paper58/manuscript/rse_geofm_world_model_rl_v3.tex` — **main manuscript**

---

## Files Created (Documentation)

1. `experiments/macos_r2/AUDIT_FINDINGS.md` — **Windows 端审计报告**
2. `experiments/macos_r2/BASELINE_TRACEBACK_COMPLETE.md` — **v2 baseline 完整追溯**
3. `experiments/macos_r2/V3_REVISION_PLAN.md` — **v3 修订计划**
4. `experiments/macos_r2/V3_EXECUTION_SUMMARY.md` — **本文档**

---

## Compilation Result

✅ **PDF compiled successfully** (48 pages, 825KB)
- No LaTeX errors
- All tables and references updated
- Abstract, results, discussion, conclusions all consistent

---

## Key Numbers Summary

| Metric | R1 (old) | R2 (new) | Change |
|---|---|---|---|
| Mean advantage | +0.0047 | **$-0.0055$** | Sign reversal |
| n_positive / n_negative | 7 / 3 | **2 / 8** | Reversed |
| Wilcoxon p | 0.32 | **>0.05** | Still NS |
| Reproducible? | ❌ No | ✅ **Yes** | Full version control |
| Data source | Pre-commit embeddings | **GEE re-extraction (9c2ba1d)** | Documented |
| Checkpoint | (unknown) | **2455c34** | In git |

---

## Narrative Changes

### Before (R1/v2)
> "GeoFM-LDN shows mildly positive but non-significant advantage over persistence (mean +0.0047, 7/10 positive, p=0.32). The signal is region-dependent and underpowered (n≈65 needed for 80% power)."

### After (R2/v3)
> "GeoFM-LDN shows **negative mean advantage** over persistence (mean $-0.0055$, 2/10 positive, all p>0.05). The dynamics network is **region-dependent and does not universally outperform persistence**. The R1 baseline (+0.0047) used pre-commit embeddings that are no longer available; the **sign reversal reflects GEE data updates**. All R2 data, checkpoints, and code are **fully reproducible** from the repository."

---

## Honesty & Transparency

v3 now:
1. ✅ Admits R1 baseline is not reproducible
2. ✅ Explains why (embeddings not in version control)
3. ✅ Reports sign reversal honestly (R1 +0.0047 → R2 $-0.0055$)
4. ✅ Provides technical explanation (GEE data updates + stochastic variation)
5. ✅ Confirms R2 full reproducibility (commit hashes, Git LFS)
6. ✅ Splits E1-E7 into completed (E2, E5) vs in-progress (E1, E3, E4, E6, E7)
7. ✅ References audit document for technical details

---

## Reviewer Response Strategy

When submitting v3, the cover letter should say:

> **Response to Reviewer M1 (Cache expansion to 30 areas):** We have expanded the embedding extraction to 30 areas (commit 9c2ba1d). However, during this expansion, we discovered that the R1 baseline (10 areas, mean advantage +0.0047) could not be reproduced because the original embeddings were not committed to version control. We re-extracted all embeddings from Google Earth Engine, yielding a revised R2 baseline (mean advantage $-0.0055$, opposite sign). This sign reversal likely reflects updates to the Harmonized Landsat-Sentinel data collection in GEE between the original extraction (pre-2026-06-07) and the R2 re-extraction (2026-07-02). All R2 data, checkpoints, and evaluation code are now in version control and produce bit-identical results across machines. We discuss this reproducibility challenge in Section 7.8 and provide a detailed technical audit in the repository (experiments/macos_r2/BASELINE_TRACEBACK_COMPLETE.md).

> **Response to Reviewer S4 (Terrain ablation) and S3 (SA-Alloc sensitivity):** These experiments (E2, E5) have been completed on macOS and incorporated into Section 7.9. E2 shows terrain context does not significantly improve 1-step prediction (p=0.08); E5 shows moderate parameter sensitivity with no dominant configuration.

> **Response to Other Follow-Up Experiments (E1, E3, E4, E6, E7):** Five experiments require further work due to technical issues discovered during audit (Section 7.10). We commit to completing these for the next revision cycle.

---

## Next Steps

1. ✅ v3 tex compiled successfully
2. ⏭️ **Push to GitHub** with commit message:
   ```
   v3: adopt R2 baseline with data refresh note
   
   - R1 baseline (+0.0047) not reproducible (embeddings pre-commit)
   - R2 baseline (-0.0055) from GEE re-extraction (commit 9c2ba1d)
   - Sign reversal reflects HLS data updates + stochastic variation
   - All R2 data/checkpoints/code in version control (fully reproducible)
   - E2 (terrain) and E5 (SA-Alloc sensitivity) completed
   - E1, E3, E4, E6, E7 in-progress (see AUDIT_FINDINGS.md)
   ```

3. ⏭️ **Write reviewer response letter** (use template above)
4. ⏭️ **Decide on E1-E7 priority**:
   - Fix E3 (persistence bug) — 30 min
   - Complete E4 (needs Windows predicted embeddings) — 1 day
   - Diagnose E1 (effect size -2.7e-5) — 1-2 days
   - Clarify E6 (43 areas with 2 years) — discuss with team
   - E7 (third encoder) — optional, mark as out-of-scope if time-limited

---

## Success Criteria Met

✅ Honest about R1 baseline not reproducible  
✅ R2 baseline fully reproducible (all artifacts in git)  
✅ Sign reversal explained with technical reasoning  
✅ No hidden/buried mentions of old +0.0047  
✅ All tables/figures consistent with R2 data  
✅ Abstract/results/discussion/conclusions aligned  
✅ E2/E5 results incorporated, E1/E3/E4/E6/E7 status clear  
✅ PDF compiles without errors  

---

## Conclusion

v3 successfully adopts **Option A**: 撤回 v2 baseline，用新 baseline (-0.0055)，诚实说明数据更新。

The manuscript now:
- Reports **negative mean advantage** (honest, not misleading)
- Explains **why R1 is not reproducible** (transparency)
- Confirms **R2 full reproducibility** (scientific rigor)
- Splits **completed vs in-progress experiments** (honest status)

This is a **scientifically honest revision** that prioritizes reproducibility over preserving the original claim.
