# Frozen Geospatial Foundation-Model Embeddings for Bounded Land-Cover Change Screening and Allocation

This repository contains the code, processed caches, trained checkpoints, result tables, figures, and manuscript materials for a Remote Sensing of Environment submission on frozen geospatial foundation-model embeddings for land-cover change screening, categorical allocation, and downstream planning probes.

The manuscript evaluates whether annual AlphaEarth Foundations embeddings can serve as a frozen representation state for lightweight latent dynamics. The central conclusion is deliberately bounded: the embedding channel carries useful change-screening and allocation signal, but the current dynamics model does not outperform persistence as a standalone land-cover forecaster and is not a replacement for native driver-calibrated cellular-automata modelling.

## Current Manuscript

Main LaTeX source:

`paper/rse_submission_paper58/manuscript/rse_geofm_world_model_rl_v5.tex`

Compiled PDF:

`paper/rse_submission_paper58/manuscript/rse_geofm_world_model_rl_v5.pdf`

Submission support files:

- `paper/rse_submission_paper58/submission_docs/highlights.md`
- `paper/rse_submission_paper58/submission_docs/declarations.md`
- `paper/rse_submission_paper58/submission_docs/data_code_availability.md`
- `paper/rse_submission_paper58/submission_docs/author_checklist.md`
- `paper/rse_submission_paper58/submission_docs/compile_verification.md`

## What Is Included

- `paper/rse_submission_paper58/`
  - RSE submission package, manuscript source, compiled PDF, figures, declarations, highlights, and result tables used by the manuscript.
- `experiments/paper8/`
  - Main experimental code for embedding-space dynamics, dual-representation planning, encoder comparison, cached outputs, and trained checkpoints.
- `experiments/macos_r2/`
  - Cache-aligned retraining, expanded-area evaluation, multi-step rollout diagnostics, and per-year decoder outputs.
- `data/independent_change_labels/`
  - Independent ESRI-label validation inputs, predicted maps, embedding caches, manifests, and readiness reports.
- `scripts/rse_revision/`
  - Analysis and consistency scripts used to regenerate or verify manuscript tables and submission constraints.
- `src/legacy_runtime/`
  - Runtime files imported by the experiment scripts from the original workspace, including county environment and policy code.
- `src/adk_world_model/`
  - World-model source files and figure-generation scripts used for the geospatial world-model component.
- `external/alphaearth_system/`
  - Minimal AlphaEarth-System/GeoAdapter subset needed for the Prithvi encoder diagnostic.
- `reproducibility/`
  - Reproduction guide, manifest, and verification records.

## Public Data Sources

The experiments use public Earth-observation data products:

- AlphaEarth annual embeddings from Google Earth Engine: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- ESRI Global LULC 10 m Time Series: `projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS`
- NASA HLS inputs for the Prithvi diagnostic, accessed through Google Earth Engine

No human-participant, animal-subject, or access-restricted data are used.

## Reproducibility Checks

The manuscript consistency check verifies the abstract length, total word count, highlights, required source snippets, table consistency against CSV outputs, citation coverage, and submission-context guardrails:

```powershell
python scripts\rse_revision\check_manuscript_v5_consistency.py
```

The current submission package passes this check with:

- Abstract: 217 words
- Manuscript total: 13,879 words
- Highlights: 4 bullets, maximum 72 characters
- Independent validation table rows checked: 10
- Citation keys and bibliography items: 21 / 21

Related tests can be run with:

```powershell
python -m pytest tests/test_manuscript_v5_consistency.py tests/test_rse_revision_results.py tests/test_rse_revision_enhancements.py tests/test_rse_revision_change_validation.py -q
```

The latest verified run passed 30 tests. The only observed warning was a local pytest cache warning on Windows.

## Compiling the Manuscript

From the repository root, run twice:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper/rse_submission_paper58/manuscript paper/rse_submission_paper58/manuscript/rse_geofm_world_model_rl_v5.tex
```

The verified PDF has 42 pages. The final LaTeX log contains no undefined citations, no undefined references, no `Float too large` warning, and no cross-reference rerun warning.

## Key Reproduction Entry Points

- Embedding-space paired inference and manuscript table checks:
  - `scripts/rse_revision/check_manuscript_v5_consistency.py`
- Independent categorical change validation:
  - `scripts/rse_revision/evaluate_independent_change_validation.py`
  - `data/independent_change_labels/`
- Dual-representation planning experiment:
  - `experiments/paper8/train_dual_rep.py`
  - `experiments/paper8/results/dual_rep/dropout_statistical_tests.json`
- Prithvi vs AlphaEarth encoder diagnostic:
  - `experiments/paper8/compare_encoders.py`
  - `experiments/paper8/train_prithvi_ldn.py`
  - `experiments/paper8/eval_prithvi_vs_alphaearth.py`
- Geospatial world-model figures:
  - `src/adk_world_model/experiments/fig_world_model.py`

## Interpretation Boundary

This repository supports a bounded scientific claim. The released experiments show that frozen GeoFM embeddings can provide useful change-screening and allocation information under controlled diagnostics. They do not establish a general forecasting advantage over persistence, an operational categorical land-cover forecaster, or a scenario-conditioned counterfactual simulator.

## Citation

A DOI-archived snapshot will be added if required by the journal or before final publication. Until then, cite the public GitHub repository URL when referring to the code and reproducibility package:

`https://github.com/zhouning/paper58-geofm-world-model-rl`
