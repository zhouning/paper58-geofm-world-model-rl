# Current Work Progress: 2026-06-17

## Repository

Local repository:

```text
D:\test\paper58-geofm-world-model-rl
```

GitHub repository:

```text
https://github.com/zhouning/paper58-geofm-world-model-rl
```

Main manuscript:

```text
paper/rse_submission_paper58/manuscript/rse_geofm_world_model_rl.tex
paper/rse_submission_paper58/manuscript/rse_geofm_world_model_rl.pdf
```

## Current RSE Manuscript Position

The manuscript remains targeted to Remote Sensing of Environment, but the claims have been narrowed after reviewer-style assessment.

The current defensible claim is:

```text
Frozen AlphaEarth embeddings can act as a low-cost state space for baseline-trend land-surface change-signal modelling and representation-level forecasting.
```

The manuscript now explicitly does not claim:

```text
operational categorical LULC forecasting
validated counterfactual scenario simulation
complete deployment-ready planning transfer
```

## Author Metadata

The author block has been changed to:

```text
Ning Zhou
SuperMap Software Co., Ltd., Beijing, China
Correspondence: Ning Zhou
```

## New Experiment Added This Session

A new label-only transition-prior baseline was added to the independent categorical change validation.

Implemented in:

```text
scripts/rse_revision/evaluate_independent_change_validation.py
```

The baseline estimates start-class to end-class transition frequencies from other ESRI label pairs with the same grid shape. It uses no AlphaEarth embeddings and no LatentDynamicsNet predictions.

Mean validation results over 12 area-year pairs:

```text
Model change F1:                 0.335
Spatially shuffled model F1:      0.269
Label-only transition-prior F1:   0.166
Persistence change F1:            0.000

Model end-year accuracy:          0.666
Transition-prior end accuracy:    0.764
Persistence end-year accuracy:    0.846
```

Interpretation:

```text
The model has stronger change-detection signal than shuffled and label-only transition-frequency controls, but it is still weaker than simple baselines for full-map categorical accuracy and area-bias metrics.
```

## New Figure Added This Session

A spatial validation/failure-mode figure was generated:

```text
paper/rse_submission_paper58/figures/fig_rse_revision_spatial_change_validation.pdf
paper/rse_submission_paper58/figures/fig_rse_revision_spatial_change_validation.png
```

It visualizes Banzhucun 2020-2023:

```text
reference start map
reference end map
decoded model prediction
reference binary change
predicted binary change
hit / miss / false alarm map
```

The figure reports:

```text
true change pixels: 59,588
change hits:        36,605
misses:             22,983
false alarms:       55,763
```

This figure is intentionally framed as spatial validation plus failure-mode evidence, not a polished success-only example.

## Manuscript Changes Made This Session

Updated:

```text
paper/rse_submission_paper58/manuscript/rse_geofm_world_model_rl.tex
```

Key changes:

- Updated author metadata.
- Rewrote the abstract around bounded change-signal modelling.
- Added label-only transition prior to baseline methods.
- Expanded the independent categorical change-validation protocol.
- Updated the empirical evidence-chain table.
- Updated the independent validation table with prior F1 and prior end-year accuracy.
- Inserted the new spatial validation/failure-mode figure.
- Reframed the Discussion around what Paper58 actually solves.
- Revised limitations to state that full categorical forecasting remains weaker than simple baselines.
- Revised the conclusion to avoid operational overclaiming.
- Fixed `\graphicspath` so manuscript compilation from the `manuscript` directory can find revision figures.

## Scripts And Tests Added/Updated

Updated:

```text
scripts/rse_revision/evaluate_independent_change_validation.py
scripts/rse_revision/make_revision_figures.py
scripts/rse_revision/audit_empirical_pipeline.py
tests/test_rse_revision_change_validation.py
tests/test_rse_revision_figures.py
tests/test_rse_revision_enhancements.py
```

Generated/updated results:

```text
paper/rse_submission_paper58/revision_results/independent_change_validation_summary.json
paper/rse_submission_paper58/revision_results/independent_change_validation_by_area.csv
paper/rse_submission_paper58/revision_results/independent_change_validation_transitions.csv
paper/rse_submission_paper58/revision_results/empirical_pipeline_audit.json
paper/rse_submission_paper58/revision_results/empirical_pipeline_audit.csv
```

## Verification Completed

Tests:

```powershell
python -m pytest tests -q --basetemp .pytest_run\final-tests
```

Result:

```text
20 passed
```

Result rebuilding:

```powershell
python scripts\rse_revision\build_revision_results.py
python scripts\rse_revision\evaluate_independent_change_validation.py
python scripts\rse_revision\audit_empirical_pipeline.py
python scripts\rse_revision\make_revision_figures.py
```

Results:

```text
AlphaEarth advantage: n=10, mean=0.004729, 95% CI=[-0.002801, 0.012986]
Independent change validation: complete, 12 evaluated pair(s), 0 skipped pair(s)
Empirical pipeline audit: 3 complete, 3 diagnostic, 0 missing
Revision figures written successfully
```

LaTeX:

```powershell
pdflatex -interaction=nonstopmode rse_geofm_world_model_rl.tex
pdflatex -interaction=nonstopmode rse_geofm_world_model_rl.tex
```

Status:

```text
PDF compiled successfully.
No missing figures.
No undefined references.
No undefined citations detected in the log scan.
Remaining LaTeX warnings are minor overfull/underfull boxes.
```

Git check:

```powershell
git -C D:\test\paper58-geofm-world-model-rl diff --check
```

Status:

```text
Passed, with only CRLF conversion warnings from Git.
```

## Local Scratch Notes

`.pytest_run/` and `.pytest_tmp/` were added to `.gitignore` because pytest temp directories caused Windows permission warnings during `git status`.

## Recommended Next Step

If continuing Paper58 before RSE submission, the next highest-value tasks are:

1. Clean and verify the reference list, especially arXiv / future-dated / unpublished AlphaEarth-related citations.
2. Decide whether to shorten or move the downstream planning probe to supplementary material.
3. Add manually interpreted image chips or higher-quality reference labels if available.
4. Review the final PDF visually for figure placement and page economy.
