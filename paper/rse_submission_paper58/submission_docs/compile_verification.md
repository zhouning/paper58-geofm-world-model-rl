# Compile Verification

Date: 2026-06-07

Command run from package root:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=manuscript manuscript/rse_geofm_world_model_rl.tex
```

The command was run three times to stabilize references.

## Result

- Final PDF: `manuscript/rse_geofm_world_model_rl.pdf`
- Final log: `manuscript/rse_geofm_world_model_rl.log`
- Output size: 522,389 bytes
- Page count: 26 pages
- Fatal errors: none
- Missing figures: none
- Undefined citations: none after rerun
- Undefined references: none after rerun

## Remaining Warnings

The log still contains overfull and underfull hbox warnings. The most important severe overfull warnings occur around:

- Abstract/keywords area near line 48.
- ESRI Global LULC dataset path near lines 160-163.
- Equation/table blocks near lines 251-295.
- Encoder ablation table/text near lines 542-550.
- Feature-dropout table near lines 633-642.

These are layout polish issues rather than package-completeness failures. They should be reviewed before final RSE submission.
