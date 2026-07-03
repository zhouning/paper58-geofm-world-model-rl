# RSE Submission Package: Paper58

Current target journal: Remote Sensing of Environment (RSE).

## Source of Truth

Main LaTeX manuscript:

`manuscript/rse_geofm_world_model_rl_v5.tex`

Compiled PDF:

`manuscript/rse_geofm_world_model_rl_v5.pdf`

## Package Layout

- `manuscript/`
  - `rse_geofm_world_model_rl_v5.tex`: current submission LaTeX source.
  - `rse_geofm_world_model_rl_v5.pdf`: compiled PDF generated from the current source.
- `figures/`
  - Required figure files referenced by the manuscript.
- `submission_docs/`
  - Cover letter draft, declarations, highlights, data/code availability, and author checklist.
- Result and experiment-output directories
  - CSV/JSON files used by the manuscript consistency checks and result tables.

## Compile Command

From `paper/rse_submission_paper58` or the repository root with adjusted paths, run twice:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=manuscript manuscript/rse_geofm_world_model_rl_v5.tex
```

Repository-root command:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper/rse_submission_paper58/manuscript paper/rse_submission_paper58/manuscript/rse_geofm_world_model_rl_v5.tex
```

## RSE Positioning

Primary framing:

- Frozen geospatial foundation-model embeddings as a diagnostic state space for land-cover change screening.
- Lightweight latent dynamics prediction in AlphaEarth embedding space, bounded by persistence comparisons.
- Independent ESRI-label validation separating binary change screening from full-map categorical forecasting.
- Same-grid GeoSOS-FLUS comparison treated as a controlled allocation ablation, not a native-driver competition.
- Downstream RL planning retained as an auxiliary-channel probe, not as an operational planning system.

## Pre-Submission Notes

- Confirm the funding statement before upload.
- Confirm the generative-AI disclosure wording in `submission_docs/declarations.md` if the Elsevier system requests it.
- Add a repository DOI snapshot if required before acceptance.
- Keep scenario-conditioned counterfactual simulation as future work unless scenario-labeled training data are added.

