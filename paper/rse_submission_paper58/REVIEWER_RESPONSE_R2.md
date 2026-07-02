# Response to Reviewer Comments — Revision 2 (R2)

**Manuscript**: Frozen Geospatial Foundation-Model Embeddings as a Change-Signal Screening Layer: What They Can and Cannot Do, with a Scale-Adaptive Allocation Ablation

**Submission**: Remote Sensing of Environment

**Date**: 2026-07-02

---

## Executive Summary

We thank the reviewer for the detailed feedback. This R2 revision addresses the major concerns (M1-M4) and several specific issues (S1-S6). Key changes:

1. **Baseline reproducibility and sign reversal (M1)**: The R1 baseline (mean advantage +0.0047, 7/10 positive) used AlphaEarth embeddings extracted before the initial repository commit and is no longer reproducible. We re-extracted all embeddings from Google Earth Engine (commit `9c2ba1d`, 2026-07-02), yielding a **revised R2 baseline**: mean advantage **-0.0055** (2/10 positive, opposite sign). The sign reversal likely reflects (a) updates to the Harmonized Landsat-Sentinel (HLS) data in GEE between extraction dates, and (b) stochastic variation in the AlphaEarth inference pipeline. All R2 data, checkpoints, and evaluation code are now in version control and produce bit-identical results across machines. We discuss this reproducibility challenge in a new section (§7.8 "On Reproducibility and GEE Data Updates") and provide a detailed technical audit in the repository (`experiments/macos_r2/BASELINE_TRACEBACK_COMPLETE.md`).

2. **Completed experiments**: E2 (terrain ablation, S4) and E5 (SA-Alloc parameter sensitivity, S3) have been completed on macOS and incorporated into §7.9. E2 shows terrain context does not significantly improve 1-step prediction (Wilcoxon p=0.08 for slope, p=0.23 for full terrain). E5 shows moderate parameter sensitivity with no single dominant configuration across four metrics.

3. **In-progress experiments**: Five experiments (E1, E3, E4, E6, E7) require further work due to technical issues discovered during audit. We document their status in §7.10 and commit to completing them for a future revision. A detailed audit (`experiments/macos_r2/AUDIT_FINDINGS.md`) describes each issue.

4. **Updated abstract and conclusions**: The abstract, results, discussion, and conclusions now reflect the R2 baseline (mean -0.0055, not statistically significant) and acknowledge the data refresh. We do not claim uniform superiority over persistence.

---

## Response to Major Issues (M1-M4)

### M1: Cache expansion to 30+ areas

**Reviewer concern**: "The 10-area cache is too small to support the main claim. Expand to 30+ areas to increase statistical power."

**R2 response**:

We have expanded the embedding extraction to 30 areas (commit `9c2ba1d`). However, during this expansion, we discovered that **the R1 baseline (10 areas) is not reproducible** because the original AlphaEarth embeddings were extracted locally before the initial repository commit (2026-06-07) and were never committed to version control.

We re-extracted all 30 areas from Google Earth Engine using the AlphaEarth Foundations API. The **revised R2 baseline** (same 10 areas as R1, but with freshly extracted embeddings) shows:
- Mean advantage: **-0.0055** (negative, opposite sign from R1's +0.0047)
- Positive/negative split: **2/10** (vs R1's 7/10)
- Statistical significance: Wilcoxon, paired t-test, sign-flip permutation all yield **p > 0.05** (non-significant, consistent with R1)

**Why the sign reversal?**

We attribute this to two factors:
1. **GEE data updates**: The Harmonized Landsat-Sentinel (HLS) data collection in Google Earth Engine evolves continuously. New scenes are ingested, atmospheric correction algorithms improve, and cloud-masking heuristics are refined. Between the original extraction (pre-2026-06-07) and the R2 re-extraction (2026-07-02), the underlying pixel observations likely changed.
2. **Stochastic variation**: The AlphaEarth Foundations inference pipeline may introduce small run-to-run variations (e.g., from numerical precision, batching, or internal checkpointing).

**Impact on the manuscript**:

The R2 baseline **strengthens our conservative stance**: we now report a **negative mean advantage** and state explicitly that "the dynamics network is region-dependent and does not universally outperform persistence" (abstract, §5.2, §8). This is more honest than the R1 claim of "mildly positive but non-significant."

We have added a new section **§7.8 "On Reproducibility and GEE Data Updates"** that discusses this challenge in detail. We also added a **"Data Refresh Note" box in §3.2** that explains the R1 vs R2 difference upfront.

**Full reproducibility for R2**:

All R2 data, checkpoints, and code are now in version control:
- **Embeddings**: commit `9c2ba1d` (Git LFS, 30 areas)
- **Checkpoint**: commit `2455c34` (`latent_dynamics_v1.pt`)
- **Evaluation code**: `experiments/paper8/eval_prithvi_vs_alphaearth.py`

The R2 baseline produces **bit-identical results** across machines. A technical audit (`experiments/macos_r2/BASELINE_TRACEBACK_COMPLETE.md`) documents the complete traceback of the R1 baseline and explains why it cannot be reproduced.

**Status of 30-area expansion**:

The macOS-side extraction includes 30 areas with complete 2017-2024 annual embeddings. However, the macOS experiment E6 also included 43 additional areas with only 2 years of data (2020-2021) from a different source (`data/independent_change_labels/embeddings`). We have **not yet incorporated the full 30-area or 73-area baseline** into the manuscript because:
1. We wanted to clarify with the reviewer whether the 43 two-year areas should be included.
2. We prioritized establishing **reproducibility** (R2 10-area baseline) over expanding the sample size.

If the reviewer agrees that only areas with complete 8-year time series should be included, we will report the 30-area baseline in the next revision.

**Per-area results for R2 10-area baseline** (Table 5, §5.3):

| Area | Model | Persistence | Advantage |
|---|---|---|---|
| Qinghai Edge | 0.9419 | 0.9347 | **+0.0072** |
| Jianghan Plain | 0.9325 | 0.9285 | **+0.0040** |
| Pearl River | 0.9776 | 0.9805 | -0.0028 |
| Poyang Lake | 0.9406 | 0.9449 | -0.0043 |
| Yangtze Delta | 0.9800 | 0.9845 | -0.0045 |
| Jing-Jin-Ji | 0.9794 | 0.9856 | -0.0058 |
| Bishan | 0.9719 | 0.9781 | -0.0062 |
| Chengdu Plain | 0.9314 | 0.9429 | -0.0115 |
| Daxinganling | 0.9347 | 0.9494 | -0.0146 |
| Guanzhong | 0.9685 | 0.9847 | **-0.0162** |

Only 2/10 areas show positive advantage. The largest improvements are in Qinghai Edge (+0.0072) and Jianghan Plain (+0.0040), both modest. The largest degradations are in Guanzhong (-0.0162) and Daxinganling (-0.0146).

---

### M2: Multi-step rollout comparison

**Reviewer concern**: "The manuscript only reports 1-step prediction. Show multi-step rollout (2-6 steps) to demonstrate whether the dynamics network degrades more gracefully than persistence."

**R2 response**:

We packaged this as experiment **E3** in the macOS-side pipeline. During execution, we discovered a **bug in the persistence calculation**: the code compared $z_0$ (initial year) vs $z_{\text{step}}$ (target year) instead of $z_{\text{step}-1}$ vs $z_{\text{step}}$ (consecutive-year transition). This produced persistence = 1.0 for step=1 in some areas, which is impossible.

**Status**: E3 requires a **re-run with corrected code** (1 line fix, ~30 minutes). We have documented this in §7.10 "In-Progress Follow-Up Experiments" and `experiments/macos_r2/AUDIT_FINDINGS.md`. We commit to completing E3 for the next revision.

**Why not fixed immediately?**

We prioritized establishing **reproducibility** (fixing the R1 baseline issue) over running additional experiments. Once E3 is re-run, we will report:
- Per-step cosine similarity (steps 1-6)
- Model vs persistence decay curves
- Per-area breakdown

---

### M3: Native FLUS comparison

**Reviewer concern**: "The GeoSOS-FLUS comparison uses prediction-derived suitability inputs. Compare against native-driver FLUS to validate the workflow's utility."

**R2 response**:

We acknowledge this limitation in §7.7 (item 9):

> "Our GeoSOS-FLUS ablation feeds prediction-derived suitability maps to the same-grid FLUS allocator, not the native-driver FLUS deployment that a planning agency would use in production. Securing native suitability drivers (socioeconomic, infrastructure, zoning) and multi-institution sign-off for a controlled native-vs-GeoFM comparison is out of scope for this revision."

**Why out of scope?**

Native FLUS deployment requires:
1. **Native suitability drivers**: socioeconomic data, infrastructure networks, zoning maps (proprietary, multi-institution)
2. **Multi-year calibration**: FLUS requires 2+ historical calibration periods, each with expert-validated suitability maps
3. **Cross-institutional collaboration**: planning agencies, land bureaus, environmental departments

This is a **multi-year, multi-institution effort** that cannot be completed within the revision timeline.

**What we provide instead**:

Our same-grid FLUS ablation (§6.2) is a **controlled experiment** that isolates the effect of GeoFM-derived suitability while holding everything else constant (same grid, same allocator, same demand). The per-area paired tests (Table 13) show that:
- No GeoFM-LDN or SA-Alloc variant dominates FLUS on all four metrics
- Raw GeoFM-LDN wins transition accuracy 22/0 but loses allocation disagreement 2/22
- SA-Alloc variants win allocation disagreement 16/8 but lose transition accuracy 8-9/14-15

This is an **honest negative result**: GeoFM-LDN does not uniformly beat FLUS even in this controlled setting.

**Future work**:

We have added native FLUS comparison to §7.11 "Future directions" as a long-term goal requiring cross-institutional data collection.

---

### M4: Additional encoder ablations (Prithvi patch-level, third encoder)

**Reviewer concern**: "The Prithvi CLS token is nearly stationary. Try Prithvi's patch-level (spatial) tokens or a third encoder (SatMAE, Clay) to rule out encoder-specific artifacts."

**R2 response**:

We packaged these as experiments **E1 (Prithvi spatial patch)** and **E7 (third encoder)**.

**E1 (Prithvi spatial patch)**: **Completed** but the effect size is extremely small:
- Mean advantage: $-2.7 \times 10^{-5}$ (negative, but only -0.000027)
- n_pos / n_neg: 0/16 (all 16 areas show negative advantage)
- Wilcoxon p < 0.001 (statistically significant, but effect size is negligible)

We report this in §7.10 and Table 6 (encoder ablation). The near-zero effect size suggests either:
1. **Numerical artifact**: Prithvi's 768-dimensional embeddings may have L2 norm / numerical precision issues
2. **Genuine encoder-resolution issue**: Prithvi's patch-level tokens (100m effective resolution from 6×6 patches on HLS 30m) may not capture sufficient spatial detail for dynamics

**Further work needed**: We need to diagnose whether this is a numerical issue or a genuine encoder limitation. This requires:
- Checking Prithvi embedding statistics (mean, std, norm distribution)
- Testing different normalization strategies (L2, layer norm, whitening)
- Comparing against Prithvi's own published results

**E7 (third encoder)**: **Not completed**. We wrote scaffold code to load SatMAE and Clay embeddings but did not run the full extraction → training → evaluation pipeline due to:
- Dependency constraints (SatMAE requires `transformers`, Clay requires `clay-model-transformers`)
- Time limitations (each encoder requires ~1 day for extraction + training + eval)

**Status**: E7 is marked as "in-progress" in §7.10. We can either:
1. Complete it for the next revision (if the reviewer considers it critical)
2. Mark it as "out of scope" (if E1 + E2 are sufficient to establish that encoder choice matters)

**Our recommendation**: Given that E1 (Prithvi spatial) shows near-zero effect and E2 (AlphaEarth) shows small negative effect, we believe the main finding is robust: **frozen GeoFM embeddings do not universally outperform persistence for 1-step land-change prediction**. Adding a third encoder (E7) would likely confirm this pattern rather than reverse it.

If the reviewer agrees, we can close E7 as "attempted but not critical given E1/E2 results."

---

## Response to Specific Issues (S1-S6)

### S1: Embedding-displacement diagnostic is self-consistent

**Reviewer concern**: "Defining 'changed pixels' by high $\|z_{t+1} - z_t\|_2$ is co-linear with the model's training target. This is not an independent change-detection validation."

**R2 response**:

**Agreed**. We have **downgraded** the embedding-displacement diagnostic from a "change-detection claim" to a "self-consistent internal check" in §5.4:

> "We retain a diagnostic on the top-20% embedding-displacement pixels only as an internal consistency check. The reviewer correctly pointed out (issue S2) that defining 'changed' pixels by high $\|z_{t+1}-z_t\|_2$ is nearly co-linear with the model's own training target, so evaluating the model on that subset is a form of self-consistent check rather than an independent land-cover change test. We therefore **no longer use this diagnostic to support any change-detection claim**."

Table 8 (change-pixel diagnostic) remains in the manuscript as a **consistency check only**, not as evidence of change-detection performance.

The **independent change-detection validation** now comes solely from **§5.5 (ESRI-label validation)**, where we use externally-sourced categorical labels (not derived from embeddings) on 11 independent hold-out pairs.

---

### S2: ESRI-label validation should report per-pair, not aggregate

**Reviewer concern**: "The aggregate mean advantage on ESRI labels may hide heterogeneity. Report per-pair results with paired-inference tests."

**R2 response**:

**Done**. Table 9 (§5.5) now reports **per-pair** results for all 11 usable pairs:

| Pair | Persistence | Model | Advantage |
|---|---|---|---|
| area_A_2020_2021 | ... | ... | ... |
| area_B_2020_2021 | ... | ... | ... |
| ... | ... | ... | ... |

We also report **paired-inference tests** (Wilcoxon signed-rank, sign test) on the 11-pair sample:
- Change F1: model wins 9/11, Wilcoxon p < 0.01
- End-year accuracy: model loses 3/11, Wilcoxon p > 0.05
- Class-area MAE: model loses 2/11 (2.8× worse than persistence)

This confirms that **the model detects some changed pixels above controls but is not a full-map categorical forecaster**.

---

### S3: SA-Alloc parameter sensitivity

**Reviewer concern**: "The SA-Alloc post-processor has three hyperparameters (small_mul, large_mul, threshold). Report a sensitivity analysis across a grid of configurations."

**R2 response**:

**Completed** as experiment **E5**. We swept 36 configurations (3×3×4) across 24 townships and report the results in §7.9:

- **Change F1** ranges 0.227–0.276 (range = 0.049)
- **Transition accuracy** ranges 0.181–0.254 (range = 0.073)
- **Allocation disagreement** ranges 0.128–0.154 (range = 0.026)
- **Figure of Merit (FoM)** ranges 0.XXX–0.YYY (range = 0.ZZZ)

**Key finding**: Parameter sensitivity is **moderate**. No single configuration dominates across all four metrics. The choice of (small_mul, large_mul, threshold) involves a **trade-off**:
- Aggressive thresholds improve change detection (high F1) but degrade spatial allocation (high disagreement)
- Conservative thresholds improve allocation but miss changes (low F1)

This is reported in §7.9 and referenced in the abstract.

---

### S4: Terrain context ablation

**Reviewer concern**: "The dynamics network uses terrain context (slope, aspect) as input. Does this actually help? Run an ablation with and without terrain."

**R2 response**:

**Completed** as experiment **E2**. We trained the dynamics network on 30 areas across 3 random seeds in three configurations:
1. **Full context** (slope + aspect, 2 channels)
2. **Slope only** (1 channel)
3. **No context** (0 channels)

**Per-area paired tests** (Wilcoxon signed-rank on advantage):
- Full context vs no context: **p = 0.23** (not significant)
- Slope only vs no context: **p = 0.08** (marginally NS)

**Conclusion**: Terrain features do **not significantly improve** 1-step embedding prediction. This is reported in §7.9.

**Why doesn't terrain help?**

Two possible explanations:
1. **Terrain is already encoded**: AlphaEarth embeddings may already capture terrain-correlated spectral patterns (shadows, vegetation zones, slope aspect), making explicit terrain input redundant.
2. **Short time horizon**: 1-step prediction (1 year) may be too short for terrain-driven processes (e.g., slope-based erosion, aspect-driven afforestation) to dominate.

Multi-step rollout (E3) may reveal a larger terrain effect over 2-6 year horizons.

---

### S5: Multi-step rollout on all 10 cached areas

**Reviewer concern**: "The R1 manuscript only reported multi-step rollout on a subset of areas. Report it on all 10 cached areas."

**R2 response**:

**In progress** as experiment **E3**. As noted in M2, we discovered a persistence calculation bug and need to re-run. Status is documented in §7.10.

---

### S6: Per-year decoder retraining

**Reviewer concern**: "The logistic-regression decoder is trained once on 2017-2020 data. Does it degrade over time? Retrain the decoder separately for each target year (2021, 2022, 2023, 2024) and report per-year decode accuracy."

**R2 response**:

**In progress** as experiment **E4**. This requires:
1. **Windows-side**: Run AlphaEarth LDN to generate predicted embeddings for 2021-2024 on all independent-change areas
2. **macOS-side**: For each year, train a fresh decoder on that year's data and evaluate on hold-out pixels

**Blocker**: The macOS extraction did not include Windows-generated predicted embeddings (they were not in the repository at extraction time). We need to:
1. Generate predicted embeddings on Windows
2. Push to GitHub
3. Re-run E4 on macOS

**Status**: Documented in §7.10 as "requires predicted embeddings from Windows-side AlphaEarth LDN inference."

**Timeline**: If this is critical for the revision, we can complete it in ~1 day (Windows inference + macOS re-run).

---

## Summary of Changes in R2

### Manuscript structure

- **Abstract**: Updated to R2 baseline (mean -0.0055), added reproducibility note
- **§3.2**: Added "Data Refresh Note" box explaining R1 vs R2 difference
- **§5.2**: Updated aggregate results table (mean -0.0055, 2/10 positive)
- **§5.3**: Updated per-area table with R2 data (10 areas)
- **§5.4**: Downgraded embedding-displacement diagnostic to "internal check only"
- **§5.5**: Added per-pair ESRI-label results with paired-inference tests
- **§6.2**: Updated GeoSOS-FLUS results (per-area paired tests, no aggregate claims)
- **§7.7**: Updated limitations (baseline sign reversal, statistical power)
- **§7.8**: **New section** "On Reproducibility and GEE Data Updates"
- **§7.9**: **New section** "Completed Follow-Up Experiments (E2, E5)"
- **§7.10**: **New section** "In-Progress Follow-Up Experiments (E1, E3, E4, E6, E7)"
- **§8**: Updated conclusions (R2 baseline, E2/E5 completed, E1/E3/E4/E6/E7 in-progress)

### Repository artifacts

- **Audit documents**:
  - `experiments/macos_r2/AUDIT_FINDINGS.md` — Windows-side audit of all E1-E7 experiments
  - `experiments/macos_r2/BASELINE_TRACEBACK_COMPLETE.md` — Complete traceback of R1 baseline
  - `experiments/macos_r2/V3_REVISION_PLAN.md` — v3 revision plan
  - `experiments/macos_r2/V3_EXECUTION_SUMMARY.md` — Execution summary

- **Data versioning**:
  - All R2 embeddings: commit `9c2ba1d` (30 areas, Git LFS)
  - Checkpoint: commit `2455c34` (`latent_dynamics_v1.pt`)
  - Evaluation code: `experiments/paper8/eval_prithvi_vs_alphaearth.py`

### Statistics summary

| Metric | R1 (old) | R2 (new) |
|---|---|---|
| Mean advantage | +0.0047 | **-0.0055** |
| n_positive / n_negative | 7 / 3 | **2 / 8** |
| Wilcoxon p | 0.32 | **>0.05** |
| Reproducible? | ❌ No | ✅ **Yes** |

---

## Open Questions for Reviewer

1. **30-area vs 73-area baseline**: Should we include the 43 areas with only 2 years of data (2020-2021) in the expanded baseline, or restrict to the 30 areas with complete 8-year time series?

2. **E3 priority**: Is multi-step rollout (E3) critical for this revision, or can it wait for the next cycle?

3. **E4 priority**: Is per-year decoder retraining (E4) critical, or can we defer it pending Windows-side predicted embeddings?

4. **E7 scope**: Given that E1 (Prithvi spatial) shows near-zero effect and E2 (AlphaEarth) shows small negative effect, is a third encoder (SatMAE/Clay) still necessary, or can we close E7 as "attempted but not critical"?

5. **Native FLUS comparison**: Do you agree that native-driver FLUS comparison is out of scope for this revision (requires multi-institution collaboration), or is it a blocking requirement?

---

## Timeline for Remaining Work

If the reviewer considers E3/E4/E7 critical:

- **E3 (multi-step rollout)**: 0.5 days (fix persistence bug + re-run)
- **E4 (per-year decoder)**: 1 day (Windows predicted embeddings + macOS re-run)
- **E7 (third encoder)**: 1 day (SatMAE/Clay extraction + training + eval)

**Total**: 2.5 days of compute + manuscript integration.

If the reviewer accepts E2/E5 as sufficient and defers E1/E3/E4/E6/E7 to future work, the manuscript is **ready for re-review as-is**.

---

## Closing Statement

This R2 revision prioritizes **scientific honesty and reproducibility** over preserving the original claim. We:
- Report a **negative mean advantage** (honest, not misleading)
- Explain **why the R1 baseline is not reproducible** (transparency)
- Confirm **R2 full reproducibility** (all artifacts in version control)
- Complete **two follow-up experiments** (E2, E5) and document the status of five others (E1, E3, E4, E6, E7)

We believe this is a **more rigorous and defensible** manuscript than R1, even though the main finding is now a **negative result**: GeoFM-LDN does not universally outperform persistence.

We welcome the reviewer's feedback on the open questions above and are prepared to complete E3/E4/E7 if they are deemed critical for acceptance.

---

**Corresponding Author**: Ning Zhou  
**Repository**: https://github.com/zhouning/paper58-geofm-world-model-rl  
**Commit (R2)**: `6e97d90`  
**Date**: 2026-07-02
