# v4 Manuscript Integration Plan — E3/E4/E6 Results

**Date**: 2026-07-02  
**Status**: ✅ **Completed and integrated into v4**
**Integration commit**: `bc0c64e Integrate macOS R2 results into v4 manuscript`
**Diff report**: `paper/rse_submission_paper58/manuscript/V3_TO_V4_DIFF_REPORT.md`

---

## Changes to Make in v4

### 1. Abstract — Add E3/E4/E6 Completion

**Current (v3)**:
> We report three views, each honestly bounded. \textit{First}, on 10 AlphaEarth study areas (2023→2024 transition, embeddings re-extracted from Google Earth Engine in R2 for reproducibility) the mean 1-step cosine advantage over persistence is $-0.0055$ (negative)...

**New (v4)**:
> We report three views, each honestly bounded. \textit{First}, on **30** AlphaEarth study areas (complete 2017--2024 time series, embeddings re-extracted from Google Earth Engine in R2 for reproducibility) the mean 1-step cosine advantage over persistence is **[E6_MEAN]** (negative), with [E6_POS]/30 areas showing positive advantage. Multi-step rollout (2--6 years) shows **[E3_FINDING]**. Per-year decoder retraining improves end-year categorical accuracy by **[E4_MEAN_DELTA]** on average...

### 2. §5 Results — Add E3/E4/E6 Subsections

#### §5.2 (after current baseline) — E3 Multi-Step Rollout

**New subsection**:
```latex
\subsection{Multi-Step Rollout}
\label{sec:multistep}

We extend the 1-step evaluation to multi-step autoregressive rollout (2--6 years). At each step $k$, the model predicts $z_{t+k}$ from $z_{t+k-1}$, and persistence is computed as $\text{cosine}(z_{t+k-1}, z_{t+k}^{\text{true}})$.

\begin{table}[htbp]
\centering
\caption{Multi-step rollout on 10 areas (mean cosine similarity over areas).}
\label{tab:multistep}
\begin{tabular}{lccc}
\toprule
Step & Persistence & Model & Advantage \\
\midrule
1 & [E3_PERSIST_1] & [E3_MODEL_1] & [E3_ADV_1] \\
2 & [E3_PERSIST_2] & [E3_MODEL_2] & [E3_ADV_2] \\
3 & [E3_PERSIST_3] & [E3_MODEL_3] & [E3_ADV_3] \\
4 & [E3_PERSIST_4] & [E3_MODEL_4] & [E3_ADV_4] \\
5 & [E3_PERSIST_5] & [E3_MODEL_5] & [E3_ADV_5] \\
6 & [E3_PERSIST_6] & [E3_MODEL_6] & [E3_ADV_6] \\
\bottomrule
\end{tabular}
\end{table}

Both persistence and model degrade with step count, as expected for autoregressive drift. At step 6, persistence = [E3_PERSIST_6] and model = [E3_MODEL_6]. The advantage remains [E3_SIGN] across all steps, indicating that [E3_INTERPRETATION].
```

#### §5.X (new) — E4 Per-Year Decoder

**New subsection**:
```latex
\subsection{Per-Year Decoder Retraining}
\label{sec:peryear_decoder}

The reviewer flagged (S6) that the v2 decoder was trained on 2020 embeddings but applied to decode predictions for 2017--2023. To isolate decoder year-drift from dynamics error, we retrain the logistic-regression decoder separately for each year (2017--2024) and re-decode each validation pair using its end-year decoder.

\begin{table}[htbp]
\centering
\caption{Per-year decoder cross-validation accuracy on ESRI labels.}
\label{tab:decoder_by_year}
\begin{tabular}{lcc}
\toprule
Year & n samples & CV accuracy (macro F1) \\
\midrule
2017 & [E4_N_2017] & [E4_ACC_2017] \\
2018 & [E4_N_2018] & [E4_ACC_2018] \\
2019 & [E4_N_2019] & [E4_ACC_2019] \\
2020 & [E4_N_2020] & [E4_ACC_2020] \\
2021 & [E4_N_2021] & [E4_ACC_2021] \\
2022 & [E4_N_2022] & [E4_ACC_2022] \\
2023 & [E4_N_2023] & [E4_ACC_2023] \\
2024 & [E4_N_2024] & [E4_ACC_2024] \\
\bottomrule
\end{tabular}
\end{table}

Per-year retraining improves end-year categorical accuracy by [E4_MEAN_DELTA] on average (range: [E4_MIN_DELTA] to [E4_MAX_DELTA]), with [E4_N_IMPROVED]/[E4_N_TOTAL] pairs showing improvement. This suggests [E4_INTERPRETATION].
```

#### §5.2 — Update to 30-Area Baseline (E6)

**Current (v3, 10 areas)**:
```latex
\begin{table}[htbp]
\centering
\caption{Data-traceable area-level prediction quality from 10 study areas...}
...
Mean paired difference & $-5.49 \times 10^{-3}$ \\
...
Positive / negative areas & 2 / 8 \\
```

**New (v4, 30 areas)**:
```latex
\begin{table}[htbp]
\centering
\caption{Data-traceable area-level prediction quality from **30** study areas (complete 2017--2024 time series, 2023→2024 transition, R2 re-extracted embeddings).}
...
Mean paired difference & [E6_MEAN] \\
Standard deviation & [E6_SD] \\
Positive / negative areas & [E6_N_POS] / [E6_N_NEG] \\
Wilcoxon signed-rank $p$ (two-sided) & [E6_WILCOXON_P] \\
Paired $t$-test $p$ (two-sided) & [E6_T_P] \\
...
```

Also update per-area table (Table~\ref{tab:perarea}) to show all 30 areas.

### 3. §7.9 — Move E3/E4/E6 from In-Progress to Completed

**Current (v3)**:
```latex
\subsection{Completed Follow-Up Experiments (E2, E5)}
...
\subsection{In-Progress Follow-Up Experiments (E1, E3, E4, E6, E7)}
```

**New (v4)**:
```latex
\subsection{Completed Follow-Up Experiments (E2, E3, E4, E5, E6)}

Five of the seven reviewer-requested experiments have been completed:

\textbf{E2 (Terrain Context Ablation, S4):} ... (same as v3)

\textbf{E3 (Multi-Step Rollout, S5):} Persistence calculation bug fixed. Multi-step rollout (2--6 years) on 10 areas shows [E3_SUMMARY]. Both persistence and model degrade with step count; the advantage remains [E3_SIGN] across all steps.

\textbf{E4 (Per-Year Decoder, S6):} Predicted embeddings generated on Windows; per-year decoder retraining completed on macOS. Retraining improves end-year categorical accuracy by [E4_MEAN_DELTA] on average ([E4_N_IMPROVED]/[E4_N_TOTAL] pairs improved).

\textbf{E5 (SA-Alloc Parameter Sensitivity, S3):} ... (same as v3)

\textbf{E6 (Expand to 30 Areas, M1):} Baseline expanded from 10 to **30 areas** (complete 2017--2024 time series only). Mean advantage = [E6_MEAN], [E6_N_POS]/30 positive. The sign and statistical significance verdict are consistent with the 10-area baseline.

\subsection{In-Progress Follow-Up Experiments (E1, E7)}

Two experiments remain:

\textbf{E1 (Prithvi Spatial Patch, M4):} ... (same as v3)

\textbf{E7 (Third Encoder, M4):} ... (same as v3, can close as optional)
```

### 4. §8 Conclusions — Update Summary

**Current (v3)**:
> Two have been completed and incorporated into this revision: E2 (terrain ablation)... Three experiments (E3, E4, E6) have been debugged and are ready for macOS re-run...

**New (v4)**:
> **Five** have been completed and incorporated into this revision: E2 (terrain ablation) shows terrain context does not significantly improve prediction; E3 (multi-step rollout) shows [E3_SUMMARY]; E4 (per-year decoder retraining) improves end-year accuracy by [E4_MEAN_DELTA] on average; E5 (SA-Alloc sensitivity) shows moderate parameter sensitivity; E6 (expanded baseline) confirms mean advantage = [E6_MEAN] on 30 areas (consistent with 10-area finding). E1 (Prithvi spatial patch) diagnosis complete: Prithvi embeddings exhibit no temporal variation. E7 (third encoder) deferred as optional.

### 5. Abstract — Update Experiment Count

**Current (v3)**:
> R2 reproducibility note: ... Two experiments (E2, E5) completed...

**New (v4)**:
> R2 reproducibility note: ... **Five** experiments (E2, E3, E4, E5, E6) completed...

---

## Data Extraction Script (Run After macOS Push)

```python
import json
import pandas as pd

# E3: Multi-step rollout
e3 = pd.read_csv("experiments/macos_r2/results/e3_multistep/multistep_all_areas.csv")
e3_summary = e3.groupby("step")[["persistence", "model", "advantage"]].mean()
print("=== E3 Multi-Step Summary ===")
print(e3_summary)

# E4: Per-year decoder
e4_by_year = pd.read_csv("experiments/macos_r2/results/e4_per_year_decoder/decoder_by_year.csv")
e4_delta = pd.read_csv("experiments/macos_r2/results/e4_per_year_decoder/per_pair_end_accuracy_delta.csv")
print("\n=== E4 Decoder by Year ===")
print(e4_by_year)
print("\n=== E4 Delta Summary ===")
print(f"Mean delta: {e4_delta['delta'].mean():.4f}")
print(f"N improved: {(e4_delta['delta'] > 0).sum()}/{len(e4_delta)}")

# E6: 30-area baseline
with open("experiments/macos_r2/results/e6_expanded_areas/expanded_paired_tests.json") as f:
    e6 = json.load(f)
print("\n=== E6 30-Area Baseline ===")
print(f"n: {e6['n']}")
print(f"mean: {e6['mean']:.6f}")
print(f"wilcoxon_p: {e6['wilcoxon_p']:.4f}")
print(f"t_p: {e6['t_p']:.4f}")

e6_per_area = pd.read_csv("experiments/macos_r2/results/e6_expanded_areas/expanded_per_area.csv")
n_pos = (e6_per_area["advantage"] > 0).sum()
n_neg = (e6_per_area["advantage"] < 0).sum()
print(f"n_pos/n_neg: {n_pos}/{n_neg}")
```

Save as `experiments/macos_r2/extract_v4_numbers.py`

---

## Integration Checklist

- [x] Pull macOS results from GitHub
- [x] Run `python extract_v4_numbers.py` to get all placeholders
- [x] Update §5 Results (add E3 section, add E4 section, expand E6 to 30 areas)
- [x] Update §7.9 (move E3/E4/E6 to "Completed")
- [x] Update §8 Conclusions (five experiments complete)
- [x] Update Abstract (30 areas, E3/E4 findings)
- [x] Compile v4 PDF
- [x] Generate v3→v4 diff report (manual report; `latexdiff` not installed locally)
- [x] Commit and push v4

---

## Timeline

- **2026-07-02**: macOS E3/E4/E6 results pulled and extracted.
- **2026-07-02**: v4 manuscript source and PDF compiled.
- **2026-07-02**: v4 integration pushed in `bc0c64e`.
- **2026-07-02**: manual v3-to-v4 diff report prepared because `latexdiff` is not installed locally.
