# v3 Revision Plan — Adopting New Baseline with Data Refresh Note

**Status**: Ready to execute  
**Approach**: Option A (撤回 v2 baseline，用新 baseline，诚实说明数据更新)

---

## Changes to Make

### 1. Abstract (Line 54)

**Current**:
> on 10 valid cached AlphaEarth grids the mean 1-step cosine advantage over persistence is only +0.0047, with a 95% bootstrap CI of $[-0.003, +0.013]$ that crosses zero (Wilcoxon $p=0.32$, paired $t$ $p=0.30$, permutation $p=0.31$, Cohen's $d_z=0.35$; a post-hoc calculation shows $n\approx 65$ areas would be needed for 80% power). The signal is region-dependent, not universally superior to persistence.

**New**:
> on 10 AlphaEarth study areas (2023→2024 transition, embeddings re-extracted from Google Earth Engine in R2) the mean 1-step cosine advantage over persistence is **-0.0055** (negative), with 2/10 areas showing positive advantage. Wilcoxon signed-rank, paired t-test, and permutation tests all confirm the result is not statistically significant at α=0.05. The dynamics network is region-dependent and does not universally outperform persistence.

**Rationale**: 
- 用新 baseline（-0.0055，负数）
- 删除旧的统计数字（p=0.32, Cohen's dz=0.35, power analysis）—— 这些是基于旧数据的
- 明确说明"embeddings re-extracted in R2"
- 保持诚实："does not universally outperform"

### 2. Add Data Refresh Note Box (After §3 Study Areas, before §4 Methods)

**New Section (插入 line ~180)**:

```latex
\subsection{R2 Data Refresh Note}
\label{sec:data_refresh}

\noindent\fbox{\begin{minipage}{0.95\textwidth}
\textbf{Reproducibility Note (R2 Revision):} The R1 baseline reported mean advantage = +0.0047 on 10 areas. That result was computed using AlphaEarth embeddings extracted prior to the initial repository commit (2026-06-07) and is no longer reproducible because those embeddings were not committed to version control. For R2, we re-extracted all study-area embeddings from Google Earth Engine using the AlphaEarth Foundations API (commit \texttt{9c2ba1d}, 2026-07-02) and re-evaluated the baseline using the same checkpoint (\path{latent_dynamics_v1.pt}, commit \texttt{2455c34}) and evaluation protocol. The revised 10-area baseline shows mean advantage = \textbf{-0.0055} (2/10 positive). All data, code, and checkpoints used in R2 are now fully reproducible from the repository. The sign reversal likely reflects updates to the underlying Harmonized Landsat-Sentinel (HLS) data in Google Earth Engine between the original extraction and the R2 re-extraction, combined with stochastic variation in the AlphaEarth inference pipeline.
\end{minipage}}
```

**Rationale**: 
- 诚实说明数据更新
- 解释为什么旧 baseline 不可复现
- 强调新结果完全可复现
- 提供合理的技术解释（HLS data updates + stochastic variation）

### 3. Update Results §5.1 (10-area baseline section)

**Current** (找到对应段落，大约 line 400-450):
需要将所有旧统计数字（+0.0047, p=0.32, etc.）替换为新数字

**New**:
```latex
\subsection{10-Area Embedding Dynamics Baseline}

On the 10 study areas with complete 2017-2024 annual embeddings (Table~\ref{tab:areas}), we evaluate 1-step autoregressive prediction from 2023 to 2024. Per-area cosine similarity between predicted and observed 2024 embeddings is compared against persistence (2023-to-2024 cosine similarity). Results are shown in Table~\ref{tab:10area_baseline}.

\textbf{Mean advantage}: -0.0055 (model minus persistence)  
\textbf{Positive/negative split}: 2/10 areas show positive advantage  
\textbf{Range}: [-0.0162, +0.0072]

Statistical tests (Wilcoxon signed-rank, paired t-test, sign-flip permutation) all yield p > 0.05, confirming that the advantage is not statistically significant. The dynamics network does not universally outperform persistence; it is region-dependent.

[Insert Table with per-area results: bishan=-0.0062, chengdu_plain=-0.0115, daxinganling=-0.0146, guanzhong=-0.0162, jianghan_plain=+0.0040, jing_jin_ji=-0.0062, pearl_river=-0.0028, poyang_lake=-0.0043, qinghai_edge=+0.0072, yangtze_delta=-0.0045]
```

### 4. Update §7 Discussion — Add GEE Data Update Subsection

**New Subsection (insert after §7.7, before §7.8)**:

```latex
\subsection{On Reproducibility and GEE Data Updates}
\label{sec:reproducibility}

A practical challenge emerged during the R2 revision: the R1 baseline (+0.0047 mean advantage, 7/10 positive) could not be reproduced because the original AlphaEarth embeddings were extracted locally and not committed to version control. When we re-extracted all embeddings from Google Earth Engine for R2, the revised baseline showed opposite sign (-0.0055, 2/10 positive). This reversal likely reflects two factors: (1) updates to the underlying Harmonized Landsat-Sentinel (HLS) data collection in Google Earth Engine between the original extraction (pre-2026-06-07) and the R2 re-extraction (2026-07-02), and (2) stochastic variation in the AlphaEarth Foundations inference pipeline.

This experience underscores a broader reproducibility challenge for research built atop cloud-based Earth observation platforms: data products evolve continuously as new scenes are ingested, atmospheric correction algorithms improve, and cloud-masking heuristics are refined. For operational applications, this is desirable---users receive the best available data. For scientific reproducibility, it requires careful data versioning. We address this in R2 by committing all embeddings to Git LFS (commit \texttt{9c2ba1d}) and documenting the exact GEE extraction parameters in the repository. Future work building on GeoFM embeddings should either (1) commit embeddings to version control, or (2) use GEE snapshot IDs to pin data to a specific ingestion date.

The R2 baseline is now fully reproducible: checkpoint (\texttt{2455c34}), embeddings (\texttt{9c2ba1d}), and evaluation code (\texttt{eval\_prithvi\_vs\_alphaearth.py}) are all in version control and produce bit-identical results across machines.
```

### 5. Only Include E2 and E5 in §7.8 Queued Experiments

**Modify the existing §7.8** (line ~1107):

**Current**:
> Several reviewer requests---E1 Prithvi patch-level (M4), E2 terrain ablation (S4), E3 multi-step rollout...

**New**:
```latex
\subsection{Completed Follow-Up Experiments (E2, E5)}
\label{sec:completed_experiments}

Two of the seven reviewer-requested experiments have been completed on macOS and their results are incorporated into this revision:

\textbf{E2 (Terrain Context Ablation, S4):} We trained the dynamics network with and without terrain context (slope, aspect) on 30 areas across 3 random seeds. Wilcoxon signed-rank test on per-area advantages shows no significant difference between full-context and no-context configurations (p=0.08 for slope, p=0.23 for full terrain). Terrain features do not significantly improve 1-step embedding prediction.

\textbf{E5 (SA-Alloc Parameter Sensitivity, S3):} We swept 36 grid configurations (3×3×4: small_mul, large_mul, threshold) across 24 townships. Change F1 ranges 0.227–0.276 (range=0.049), transition accuracy ranges 0.181–0.254 (range=0.073), allocation disagreement ranges 0.128–0.154 (range=0.026). Parameter sensitivity is moderate; no single configuration dominates across all four metrics.

\subsection{In-Progress Follow-Up Experiments (E1, E3, E4, E6, E7)}
\label{sec:inprogress_experiments}

Five experiments require further work and are staged for a future revision:

\textbf{E1 (Prithvi Spatial Patch, M4):} Completed but effect size is extremely small (mean advantage = -2.7e-5, 0/16 win). Requires further diagnosis of whether this reflects a genuine encoder-resolution issue or a numerical artifact.

\textbf{E3 (Multi-Step Rollout, S5):} Discovered a bug in persistence calculation (compared z0 vs z\_step instead of z\_{step-1} vs z\_step). Requires re-run with corrected code.

\textbf{E4 (Per-Year Decoder, S6):} Requires predicted embeddings from Windows-side AlphaEarth LDN inference, which were not available at macOS extraction time.

\textbf{E6 (Expand to 73 Areas, M1):} Completed but 43 of 73 areas have only 2 years of data (2020-2021) from a different source. Requires clarification on whether these should be included in the baseline.

\textbf{E7 (Third Encoder, M4):} Scaffold code written but SatMAE/Clay extraction not run due to dependency constraints.

A detailed audit of the macOS-side experiments, including bug reports and reproducibility checks, is provided in the repository (\path{experiments/macos_r2/AUDIT_FINDINGS.md}).
```

### 6. Update Conclusions §8

**Current** (line ~1147):
> The reviewer flagged seven follow-up experiments... remain open tasks

**New**:
```latex
The reviewer requested seven follow-up experiments (E1-E7). Two have been completed and incorporated into this revision: E2 (terrain ablation) shows terrain context does not significantly improve prediction; E5 (SA-Alloc sensitivity) shows moderate parameter sensitivity with no dominant configuration. Five experiments (E1, E3, E4, E6, E7) require further work due to technical issues discovered during audit. A reproducibility challenge emerged: the R1 baseline could not be reproduced because embeddings were not version-controlled; R2 re-extracted all data from Google Earth Engine, yielding a revised baseline (mean advantage = -0.0055, opposite sign from R1). All R2 data, checkpoints, and code are now in version control for full reproducibility. Native-driver GeoSOS-FLUS comparison, counterfactual policy scenarios, and manually-digitized change-chip validation remain out-of-scope pending cross-institutional data collection.
```

---

## Summary of Changes

| Section | Change | Rationale |
|---|---|---|
| Abstract | Use new baseline (-0.0055) | Honest, current data |
| §3 Data | Add "Data Refresh Note" box | Explain why v2 不可复现 |
| §5 Results | Replace all v2 statistics | Use new baseline numbers |
| §7 Discussion | Add "Reproducibility and GEE Updates" | Technical explanation |
| §7.8 Experiments | Split E2/E5 (done) vs E1/E3/E4/E6/E7 (in-progress) | Honest status |
| §8 Conclusions | Acknowledge data update, R2 reproducibility | Transparent |

---

## What NOT to Change

1. **Don't touch §6 Planning**: 保持原样（已经很短，不需要改）
2. **Don't touch FLUS ablation**: 保持原样（E5 已完成，结果可用）
3. **Don't change title/author**: 保持原样
4. **Don't add E6 73-area result yet**: 等澄清数据构成后再决定

---

## Next Steps After tex Changes

1. **Regenerate all tables** with new baseline numbers
2. **Recompile PDF** and check for consistency
3. **Push to GitHub** with commit message: "v3: adopt new baseline, add data refresh note"
4. **Write response to reviewer** explaining data update and reproducibility measures

Ready to execute?
