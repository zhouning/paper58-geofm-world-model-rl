# Remote-Sensing Foundation-Model Embeddings for Land-Cover Change Screening and Allocation Diagnostics

This repository contains the code, processed caches, trained checkpoints, result tables, figures, and manuscript materials for a Science of Remote Sensing submission on frozen AlphaEarth Foundations embeddings for bounded land-use and land-cover (LULC) change screening and categorical allocation diagnostics.

The manuscript evaluates whether annual AlphaEarth embeddings can serve as a frozen remote-sensing representation state for lightweight latent dynamics. The central conclusion is deliberately bounded: frozen embeddings carry a useful product-label change-screening signal and can support controlled allocation diagnostics, but the current dynamics model does not outperform persistence as a standalone embedding forecaster and is not a replacement for native driver-calibrated cellular-automata modelling.

## Current Manuscript

Target journal: Science of Remote Sensing (Elsevier)

Article title:

`Remote-Sensing Foundation-Model Embeddings for Land-Cover Change Screening and Allocation Diagnostics`

Main LaTeX source:

`paper/srs_submission_paper58/manuscript/srs_geofm_embedding_change_screening.tex`

Compiled PDF:

`paper/srs_submission_paper58/manuscript/srs_geofm_embedding_change_screening.pdf`

Submission support files:

- `paper/srs_submission_paper58/submission_docs/highlights_srs.txt`
- `paper/srs_submission_paper58/submission_docs/cover_letter_srs.md`
- `paper/srs_submission_paper58/submission_docs/title_page_metadata_srs.md`
- `paper/srs_submission_paper58/submission_docs/data_code_availability_srs.md`
- `paper/srs_submission_paper58/submission_docs/journal_requirements_audit.md`
- `paper/srs_submission_paper58/submission_docs/compile_verification_srs.md`
- `paper/srs_submission_paper58/submission_docs/reframing_from_rse_to_srs.md`

The SRS package uses single-anonymized review conventions, so the manuscript title page retains author, affiliation, postal address, and corresponding-author email.

## What Is Included

- `paper/srs_submission_paper58/`
  - Current SRS manuscript source, compiled PDF, figures, cover letter, highlights, declarations, availability text, and verification records.
- `paper/rse_submission_paper58/`
  - Earlier RSE submission package retained for provenance and transfer context. The SRS manuscript should be used for any new submission.
- `experiments/paper8/`
  - Main experimental code for embedding-space dynamics, dual-representation planning diagnostics, encoder comparison, cached outputs, and trained checkpoints.
- `experiments/macos_r2/`
  - Cache-aligned retraining, expanded-area evaluation, multi-step rollout diagnostics, per-year decoder outputs, and SA-Alloc sensitivity outputs.
- `data/independent_change_labels/`
  - ESRI product-label validation inputs, predicted maps, embedding caches, manifests, and readiness reports. The directory name is legacy; the manuscript treats these labels as product-label comparison targets, not manually interpreted reference truth.
- `scripts/rse_revision/`
  - Analysis and consistency scripts used to regenerate or verify manuscript tables and submission constraints.
- `src/legacy_runtime/`
  - Runtime files imported by experiment scripts from the original workspace, including county environment and policy code.
- `src/adk_world_model/`
  - World-model source files and figure-generation scripts used by earlier experiment branches and diagnostic figures.
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

## Main Evidence Boundary

The SRS manuscript separates three evidence layers:

1. Embedding-space dynamics: GeoFM-LDN does not significantly outperform persistence over 30 complete 2017-2024 areas (mean cosine advantage `-0.0030`, Wilcoxon `p=0.57`).
2. ESRI product-label change screening: decoded predictions improve binary change F1 over shuffled, transition-prior, and persistence controls on the 11 positive-change pairs, but lose to persistence and the transition prior on full-map end-year accuracy and class-area bias.
3. Same-grid allocation diagnostics: SA-Alloc can produce competitive low-input allocation surfaces, but no GeoFM-LDN or SA-Alloc variant dominates the stripped-down GeoSOS-FLUS console across per-township metrics.

These results support frozen remote-sensing embeddings as reproducible change-screening and allocation diagnostic layers. They do not establish an operational categorical LULC forecaster, a scenario-conditioned world model, or a replacement for locally calibrated GeoSOS-FLUS or related cellular-automata workflows.

## Verified SRS Package

The SRS manuscript was compiled from `paper/srs_submission_paper58/manuscript` with:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error srs_geofm_embedding_change_screening.tex
pdflatex -interaction=nonstopmode -halt-on-error srs_geofm_embedding_change_screening.tex
```

Latest verified output:

- PDF: 39 pages, 697,249 bytes
- Abstract: 215 words
- Keywords: 7
- Highlights: 5 bullets, maximum 75 characters
- Citation consistency: 20 cited keys, 20 bibliography items, no missing or uncited bibliography entries
- Cross-references: no missing labels
- LaTeX log: no fatal errors, no undefined citations or references, no overfull hbox warnings

## Key Reproduction Entry Points

- SRS manuscript and submission package:
  - `paper/srs_submission_paper58/`
- Embedding-space paired inference and manuscript table checks:
  - `scripts/rse_revision/check_manuscript_v5_consistency.py`
- ESRI product-label categorical change validation:
  - `scripts/rse_revision/evaluate_independent_change_validation.py`
  - `data/independent_change_labels/`
- SA-Alloc sensitivity results:
  - `experiments/macos_r2/results/e5_sa_alloc_sensitivity/metric_ranges.json`
- Dual-representation planning diagnostic:
  - `experiments/paper8/train_dual_rep.py`
  - `experiments/paper8/results/dual_rep/dropout_statistical_tests.json`
- Prithvi vs AlphaEarth encoder diagnostic:
  - `experiments/paper8/compare_encoders.py`
  - `experiments/paper8/train_prithvi_ldn.py`
  - `experiments/paper8/eval_prithvi_vs_alphaearth.py`

## Citation

A DOI-archived snapshot should be added before final upload or publication when available. Until then, cite the public GitHub repository URL when referring to the code and reproducibility package:

`https://github.com/zhouning/paper58-geofm-world-model-rl`
