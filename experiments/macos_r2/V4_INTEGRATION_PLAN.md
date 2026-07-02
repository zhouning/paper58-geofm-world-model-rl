# v4 Manuscript Integration Plan — E3/E4/E6 Results

**Date**: 2026-07-02  
**Status**: ✅ **Completed and integrated into v4; retrain_v2 critical fix and E3 v2 rerun applied**
**Integration commit**: `bc0c64e Integrate macOS R2 results into v4 manuscript`
**Critical fix source commit**: `23f1111 CRITICAL FIX: retrain LDN on R2 embeddings (train/test mismatch found)`
**Diff report**: `paper/rse_submission_paper58/manuscript/V3_TO_V4_DIFF_REPORT.md`

---

## Changes to Make in v4

### 1. Abstract — Add E3/E4/E6 Completion

**Current (v3)**:
> We report three views, each honestly bounded. \textit{First}, on 10 AlphaEarth study areas (2023→2024 transition, embeddings re-extracted from Google Earth Engine in R2 for reproducibility) the mean 1-step cosine advantage over persistence is $-0.0055$ (negative)...

**New (v4)**:
> We report three views, each honestly bounded. \textit{First}, on **30** AlphaEarth study areas (complete 2017--2024 time series, embeddings re-extracted from Google Earth Engine in R2 for reproducibility), retraining the dynamics model on the same R2 embeddings changes the mean 1-step cosine advantage over persistence to **-0.0030**, with **16/30** areas showing positive advantage and Wilcoxon **p=0.57**. The pre-R2 checkpoint result is retained only as a mismatch audit. The R2-retrained multi-step rollout is near zero at step 1 (**-0.0039**, p=0.49) but negative from step 2 onward, reaching **-0.0629** at step 6. Per-year decoder retraining improves end-year categorical accuracy by **0.0778** on average...

### 2. §5 Results — Add E3/E4/E6 Subsections

#### §5.2 (after current baseline) — E3 Multi-Step Rollout

**New subsection**:
```latex
\subsection{R2-retrained multi-step degradation diagnostic}
\label{sec:multistep}

We extend the 1-step evaluation to multi-step autoregressive rollout (1--6 annual steps). At each step $k$, the model predicts $z_{t+k}$ from $z_{t+k-1}$, and persistence is computed as $\text{cosine}(z_{t+k-1}, z_{t+k}^{\text{true}})$. The final run uses the R2-retrained `latent_dynamics_v2_seed456.pt` checkpoint.

\begin{table}[htbp]
\centering
\caption{R2-retrained corrected multi-step rollout on 30 complete-time-series areas (mean cosine similarity over areas).}
\label{tab:multistep}
\begin{tabular}{lcccc}
\toprule
Step & Persistence & Model & Advantage & Positive / negative \\
\midrule
1 & 0.9681 & 0.9642 & $-$0.0039 & 13 / 17 \\
2 & 0.9676 & 0.9509 & $-$0.0167 & 8 / 22 \\
3 & 0.9647 & 0.9322 & $-$0.0326 & 2 / 28 \\
4 & 0.9625 & 0.9190 & $-$0.0436 & 2 / 28 \\
5 & 0.9616 & 0.9089 & $-$0.0527 & 0 / 30 \\
6 & 0.9657 & 0.9028 & $-$0.0629 & 1 / 29 \\
\bottomrule
\end{tabular}
\end{table}

Step 1 is near zero and non-significant (Wilcoxon p=0.49). From step 2 onward, the rollout advantage is significantly negative and worsens gradually, reaching -0.0629 at step 6.
```

#### §5.X (new) — E4 Per-Year Decoder

**New subsection**:
```latex
\subsection{Per-Year Decoder Retraining}
\label{sec:peryear_decoder}

The reviewer flagged (S6) that the v2 decoder was trained on 2020 embeddings but applied to decode predictions for 2017--2023. To isolate decoder year-drift from dynamics error, we retrain the logistic-regression decoder separately for each year (2017--2024) and re-decode each validation pair using its end-year decoder.

\begin{table}[htbp]
\centering
\caption{Per-year decoder cross-validation on ESRI labels.}
\label{tab:decoder_by_year}
\begin{tabular}{lccc}
\toprule
Year & $n$ samples & CV accuracy & Macro-F1 \\
\midrule
2017 & 800 & 0.6675 & 0.4935 \\
2018 & 800 & 0.7488 & 0.5139 \\
2019 & 800 & 0.7225 & 0.4881 \\
2020 & 24,967 & 0.8349 & 0.7352 \\
2021 & 24,969 & 0.8287 & 0.7056 \\
2022 & 800 & 0.7300 & 0.3856 \\
2023 & 800 & 0.7425 & 0.3999 \\
2024 & 800 & 0.7475 & 0.4617 \\
\bottomrule
\end{tabular}
\end{table}

Per-year retraining improves end-year categorical accuracy by 0.0778 on average (range: -0.0260 to +0.1646), with 8/9 pairs showing improvement. This confirms decoder year-drift, but does not overturn the full-map forecasting verdict.
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
Mean paired difference & $-0.003004$ \\
Standard deviation & 0.022736 \\
Positive / negative areas & 16 / 14 \\
Wilcoxon signed-rank $p$ (two-sided) & 0.5699 \\
Paired $t$-test $p$ (two-sided) & 0.4751 \\
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

\textbf{E3 (Multi-Step Rollout, S5):} Persistence calculation bug fixed. Multi-step rollout on 30 complete-time-series areas was rerun with the R2-retrained checkpoint. Step 1 is near zero/non-significant (mean -0.0039, 13/30 positive, Wilcoxon p=0.49); steps 2--6 are significantly negative and reach -0.0629 at step 6 (1/30 positive).

\textbf{E4 (Per-Year Decoder, S6):} Predicted embeddings generated on Windows; per-year decoder retraining completed on macOS. Retraining improves end-year categorical accuracy by 0.0778 on average (8/9 pairs improved).

\textbf{E5 (SA-Alloc Parameter Sensitivity, S3):} ... (same as v3)

\textbf{E6 (Expand to 30 Areas, M1):} Baseline expanded from 10 to **30 areas** (complete 2017--2024 time series only). After R2 retraining, mean advantage = -0.003004, 16/30 positive, Wilcoxon p=0.5699. The corrected verdict is near-zero/non-significant rather than significantly negative.

\subsection{In-Progress Follow-Up Experiments (E1, E7)}

Two experiments remain:

\textbf{E1 (Prithvi Spatial Patch, M4):} ... (same as v3)

\textbf{E7 (Third Encoder, M4):} ... (same as v3, can close as optional)
```

### 4. §8 Conclusions — Update Summary

**Current (v3)**:
> Two have been completed and incorporated into this revision: E2 (terrain ablation)... Three experiments (E3, E4, E6) have been debugged and are ready for macOS re-run...

**New (v4)**:
> **Five** have been completed and incorporated into this revision: E2 (terrain ablation) shows terrain context does not significantly improve prediction; E3 (multi-step rollout) shows near-zero/non-significant step-1 performance but negative autoregressive rollout from steps 2--6; E4 (per-year decoder retraining) improves end-year accuracy by 0.0778 on average; E5 (SA-Alloc sensitivity) shows moderate parameter sensitivity; E6 (expanded baseline plus R2 retraining) reports mean advantage = -0.003004 on 30 areas with no significant paired effect. E1 (Prithvi spatial patch) diagnosis complete: Prithvi embeddings exhibit no temporal variation. E7 (third encoder) deferred as optional.

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
- [x] Pull critical fix commit `23f1111`
- [x] Run `python retrain_ldn_on_r2_data.py --epochs 100 --seeds 42 123 456`
- [x] Regenerate retrain_v2 paired tests with `--eval-only`
- [x] Update extractor/tests to include retrain_v2 results
- [x] Update v4 manuscript interpretation from "significantly negative" to "near-zero/non-significant after R2 retraining"
- [x] Rerun E3 multi-step rollout with `experiments/macos_r2/weights/retrain_v2/latent_dynamics_v2_seed456.pt`
- [x] Update v4 manuscript E3 text/table from legacy pre-R2 diagnostic to R2-retrained results

---

## Timeline

- **2026-07-02**: macOS E3/E4/E6 results pulled and extracted.
- **2026-07-02**: v4 manuscript source and PDF compiled.
- **2026-07-02**: v4 integration pushed in `bc0c64e`.
- **2026-07-02**: manual v3-to-v4 diff report prepared because `latexdiff` is not installed locally.
- **2026-07-02**: critical fix `23f1111` pulled; LDN retrained on R2 embeddings (3 seeds, 100 epochs); best checkpoint `latent_dynamics_v2_seed456.pt`.
- **2026-07-02**: R2-retrained E6 result: mean advantage -0.003004, 16/30 positive, Wilcoxon p=0.5699, bootstrap 95% CI [-0.0121, +0.0037].
- **2026-07-02**: E3 multi-step rerun completed with the R2-retrained checkpoint; step 1 mean advantage -0.0039 (Wilcoxon p=0.49), step 6 mean advantage -0.0629 (1/30 positive).
