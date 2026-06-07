# Reproduction Guide

This guide maps manuscript claims to repository files.

## 1. Manuscript Build

Current RSE source:

`paper/rse_submission_paper58/manuscript/rse_geofm_world_model_rl.tex`

Compile from the repository root:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper/rse_submission_paper58/manuscript paper/rse_submission_paper58/manuscript/rse_geofm_world_model_rl.tex
```

Run two or three times for cross-reference stability.

## 2. World Model Results

Relevant files:

- `src/adk_world_model/world_model.py`
- `src/adk_world_model/experiments/run_world_model.py`
- `src/adk_world_model/experiments/fig_world_model.py`
- `src/adk_world_model/experiments/output/world_model_17areas.json`
- `src/adk_world_model/experiments/output/world_model_rollout.json`
- `src/adk_world_model/experiments/output/world_model_lulc_decode.json`
- `src/adk_world_model/experiments/output/fig_wm_*.pdf`

These support the latent dynamics, rollout, decoder, and figure results in the manuscript.

## 3. Downstream Planning / Feature Dropout

Relevant files:

- `experiments/paper8/train_dual_rep.py`
- `experiments/paper8/dual_rep_env.py`
- `experiments/paper8/intervention_dynamics.py`
- `experiments/paper8/results/dual_rep/dropout_statistical_tests.json`
- `experiments/paper8/results/dual_rep/*_eval_seed*.json`
- `experiments/paper8/results/dual_rep/*_model_seed*.zip`

The manuscript table reporting full, 0.3 dropout, and 1.0 dropout settings is derived from `dropout_statistical_tests.json`.

## 4. Encoder Ablation

Relevant files:

- `experiments/paper8/README_PRITHVI_ABLATION.md`
- `experiments/paper8/extract_prithvi_embeddings.py`
- `experiments/paper8/compare_encoders.py`
- `experiments/paper8/train_prithvi_ldn.py`
- `experiments/paper8/eval_prithvi_vs_alphaearth.py`
- `experiments/paper8/paper58_runtime.py`
- `experiments/paper8/results/encoder_comparison.json`
- `experiments/paper8/results/paper8_ablation_encoder.json`
- `external/alphaearth_system/geoadapter/`
- `external/alphaearth_system/colab/train_paper58_ablation.ipynb`

The Prithvi ablation depends on HLS extraction and Prithvi model assets. If pretrained Prithvi weights are not bundled due to size or licensing, follow the extraction notes in `experiments/paper8/README_PRITHVI_ABLATION.md`.

## 5. Runtime Dependencies

The original experiment scripts imported several files from the broader workspace. They are copied into:

`src/legacy_runtime/`

If running scripts directly from `experiments/paper8`, add both `src/legacy_runtime` and `src/adk_world_model` to `PYTHONPATH`, or copy the needed files into the script directory.

Example PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src\legacy_runtime;$PWD\src\adk_world_model;$PWD\external\alphaearth_system;$env:PYTHONPATH"
python experiments\paper8\compare_encoders.py
```

## 6. Data Provenance

Cached embeddings in `experiments/paper8/data/` were generated from public geospatial sources and preserved for review reproducibility. See:

- `paper/wm_v2_dataset_provenance.md`
- `experiments/paper8/data/metadata.json`
- `experiments/paper8/data/prithvi/*.npy`

## 7. Known Caveats

- Original scripts contain some absolute path assumptions from the development environment. The included runtime files and cached outputs make the results inspectable; path normalization may be needed for a clean rerun on a different machine.
- Scenario-conditioned counterfactuals are an architectural placeholder in the manuscript and are not claimed as fully trained policy scenarios.
- Some large binary artifacts are included for reproducibility. If using GitHub, Git LFS may be required.
