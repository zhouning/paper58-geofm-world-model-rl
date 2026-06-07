# Paper58: GeoFM World Model and Embedding-Space Planning

This repository collects the code, data caches, trained models, experiment outputs, figures, and manuscript materials for Paper58:

**Geospatial World Modeling via Frozen Foundation Model Embeddings: From Latent Dynamics Prediction to Embedding-Space Planning**

The purpose is to give reviewers a single place to inspect and reproduce the Paper58 pipeline.

## Repository Contents

- `experiments/paper8/`
  - Main Paper58 experimental code, cached embeddings, trained policy/model checkpoints, and result JSON files.
  - Includes the dual-representation RL planning experiments, encoder ablation, Prithvi comparison, and embedding-space environments.
- `paper/rse_submission_paper58/`
  - RSE-oriented LaTeX submission package, figures, compiled PDF, cover letter draft, declarations, and provenance.
- `paper/`
  - Additional manuscript versions, design notes, presentation exports, and integration memos.
- `src/legacy_runtime/`
  - Runtime files imported by the Paper58 scripts from the original experiment workspace, including county environment and policy code.
- `src/adk_world_model/`
  - ADK world model source files and figure-generation scripts used for the geospatial world-model component.
- `external/alphaearth_system/`
  - Minimal AlphaEarth-System/GeoAdapter subset needed for the Prithvi encoder ablation.
- `archives/`
  - Original Colab/archive bundles used during the Paper58 experiments.
- `reproducibility/`
  - Reproduction notes, manifest, and verification records.

## Main Manuscript

The current RSE submission source is:

`paper/rse_submission_paper58/manuscript/rse_geofm_world_model_rl.tex`

The compiled PDF is:

`paper/rse_submission_paper58/manuscript/rse_geofm_world_model_rl.pdf`

## Quick Start

Create an environment and install the Python dependencies:

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

For Linux/macOS, use the corresponding virtual environment activation path.

Most scripts were developed on Windows with paths rooted at `D:/test` and `D:/adk`. The repository preserves the original files and also includes runtime dependencies so reviewers can inspect and adapt the scripts locally.

## Key Reproduction Entry Points

- Dual-representation feature-dropout experiment:
  - `experiments/paper8/train_dual_rep.py`
  - result summary: `experiments/paper8/results/dual_rep/dropout_statistical_tests.json`
- Prithvi vs AlphaEarth encoder ablation:
  - `experiments/paper8/compare_encoders.py`
  - `experiments/paper8/train_prithvi_ldn.py`
  - `experiments/paper8/eval_prithvi_vs_alphaearth.py`
  - result summary: `experiments/paper8/results/paper8_ablation_encoder.json`
- Geospatial world-model figures:
  - `src/adk_world_model/experiments/fig_world_model.py`
  - outputs: `src/adk_world_model/experiments/output/fig_wm_*.pdf`

## Data and Large Files

This repository intentionally includes cached `.npy`, `.npz`, `.pt`, and `.zip` files needed to inspect or rerun the reported experiments without regenerating every embedding from Google Earth Engine.

Some data are derived from public services:

- AlphaEarth annual embeddings from Google Earth Engine: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- Prithvi/HLS extraction scripts in `experiments/paper8/extract_prithvi_embeddings.py`

If a GitHub push rejects large files, use Git LFS for `.npy`, `.npz`, `.pt`, `.zip`, `.pdf`, `.pptx`, and `.docx` artifacts, or archive the same repository snapshot on Zenodo/OSF.

## Verification

See:

- `reproducibility/MANIFEST.tsv`
- `reproducibility/REPRODUCTION_GUIDE.md`
- `paper/rse_submission_paper58/submission_docs/compile_verification.md`

## Notes for Reviewers

The manuscript frames scenario-conditioned counterfactual simulation as future work unless scenario-labeled training data are added. The implemented and reported results focus on baseline-trend latent dynamics, changed-pixel prediction, encoder ablation, and downstream embedding-space planning.
