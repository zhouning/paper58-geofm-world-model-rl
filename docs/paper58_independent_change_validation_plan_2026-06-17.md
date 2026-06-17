# Paper58 Independent Change Validation Plan

Date: 2026-06-17

## Goal

Strengthen the RSE submission with an auditable land-cover transition validation workflow. The key question is no longer only whether embedding-space cosine similarity improves over persistence, but whether decoded embedding dynamics correspond to independently labelled categorical LULC change.

## Implemented Workflow

- Independent ESRI annual LULC label extraction:
  - `scripts/rse_revision/fetch_independent_lulc_labels.py`
- AlphaEarth grid extraction for validation-only missing regions:
  - `scripts/rse_revision/fetch_change_validation_embeddings.py`
- Decoded prediction generation:
  - `scripts/rse_revision/generate_change_validation_predictions.py`
- Independent categorical change evaluation:
  - `scripts/rse_revision/evaluate_independent_change_validation.py`
- Evidence-chain audit:
  - `scripts/rse_revision/audit_empirical_pipeline.py`

## Current Result

The current validation contains 12 evaluated area-year pairs and 0 skipped pairs:

- Banzhucun: 2017-2020 and 2020-2023
- Bishan: annual pairs from 2017-2018 through 2023-2024
- Heping: 2017-2020
- Poyang Lake: 2020-2021
- Wuyi Mountain: 2020-2021

Mean metrics over 12 pairs:

- Model change F1: 0.335
- Spatially shuffled model change F1: 0.269
- Persistence change F1: 0.000
- Model changed-pixel accuracy: 0.279
- Spatially shuffled changed-pixel accuracy: 0.264
- Model end-year categorical accuracy: 0.666
- Persistence end-year categorical accuracy: 0.846

Interpretation: the decoded dynamics contain non-zero and partially spatially localized change signal, but the model is not yet a superior full-map categorical LULC forecaster.

## Data Contract

Independent labels:

```text
data/independent_change_labels/labels/{area}_lulc_{year}.npy
```

Model predictions:

```text
data/independent_change_labels/predicted/{area}_lulc_pred_{start_year}_{end_year}.npy
```

Validation AlphaEarth grids:

```text
data/independent_change_labels/embeddings/{area}_emb_{year}.npy
data/independent_change_labels/embeddings/{area}_context.npy
```

Outputs:

```text
paper/rse_submission_paper58/revision_results/independent_change_validation_summary.json
paper/rse_submission_paper58/revision_results/independent_change_validation_by_area.csv
paper/rse_submission_paper58/revision_results/independent_change_validation_transitions.csv
paper/rse_submission_paper58/revision_results/empirical_pipeline_audit.json
paper/rse_submission_paper58/revision_results/empirical_pipeline_audit.csv
```

## Reproduction Commands

Fetch validation embeddings for Poyang Lake and Wuyi Mountain:

```powershell
python scripts\rse_revision\fetch_change_validation_embeddings.py --areas poyang_lake,wuyi_mountain --years 2020,2021 --scale 500
```

Generate decoded prediction maps:

```powershell
python scripts\rse_revision\generate_change_validation_predictions.py --embedding-dir experiments\paper8\data --embedding-dir experiments\paper8\data\village --embedding-dir experiments\paper8\data\heping --embedding-dir data\independent_change_labels\embeddings
```

Evaluate independent change validation:

```powershell
python scripts\rse_revision\evaluate_independent_change_validation.py
```

Update the evidence-chain audit:

```powershell
python scripts\rse_revision\audit_empirical_pipeline.py
```

## Boundary for the Manuscript

Supported:

- The model detects categorical change with non-zero F1 where persistence predicts no change.
- Mean change F1 is higher than a spatially shuffled model control, supporting partial spatial localization.
- Results are region-dependent and weaker in Poyang Lake and no-change Wuyi Mountain.

Not supported:

- The model is operationally ready for categorical LULC transition forecasting.
- The model is superior to persistence for full end-year categorical maps.
- The current scenario-conditioned interface is a validated counterfactual simulator.
