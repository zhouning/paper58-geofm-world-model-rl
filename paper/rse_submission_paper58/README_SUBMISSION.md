# RSE Submission Package: Paper 5+8

Date organized: 2026-06-07

## Source of Truth

Main LaTeX manuscript:

`manuscript/rse_geofm_world_model_rl.tex`

This file is copied from `D:/test/paper8_geofm_world_model_rl.tex` and adjusted only to make figure paths work inside this submission package.

## Package Layout

- `manuscript/`
  - `rse_geofm_world_model_rl.tex`: submission LaTeX source.
  - `rse_geofm_world_model_rl.pdf`: compiled PDF generated from this package, after verification.
  - `paper8_geofm_world_model_rl_original_compile.pdf`: previous PDF copied from the original working directory.
  - `paper8_geofm_world_model_rl_original_compile.log`: previous compile log copied for provenance.
- `figures/`
  - Required `fig_wm_*.pdf` figure files referenced by the manuscript.
- `submission_docs/`
  - Cover letter draft, declarations, data/code availability, and author checklist.
- `provenance/`
  - Original Paper 5 source, original Paper 5+8 source, merge design, and integration memo.

## Compile Command

From this package root:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=manuscript manuscript/rse_geofm_world_model_rl.tex
```

Run twice if cross-references need a second pass.

## RSE Positioning

Recommended target journal: Remote Sensing of Environment.

Primary framing:

- Frozen geospatial foundation model embeddings as a state space for land-cover change modeling.
- Lightweight latent dynamics prediction in AlphaEarth embedding space.
- Change-pixel evaluation and ablations showing when the model improves over persistence.
- Downstream embedding-space planning as an application probe, not the dominant contribution.

## Pre-Submission Notes

- Replace anonymized code URL before final submission if journal policy requires a public repository.
- Confirm all author names, affiliations, funding statements, and ethics/declarations.
- Treat scenario-conditioned counterfactual simulation as future work unless scenario-labeled training data are added.
- Keep the RL planning section framed as downstream decision-support evidence rather than a full cross-region planning validation.
