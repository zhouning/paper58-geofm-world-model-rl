# Compile verification

Date: 2026-07-05

Command run from `paper/srs_submission_paper58/manuscript`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error srs_geofm_embedding_change_screening.tex
pdflatex -interaction=nonstopmode -halt-on-error srs_geofm_embedding_change_screening.tex
```

Result:

- PDF generated: `manuscript/srs_geofm_embedding_change_screening.pdf`
- PDF size: 656,664 bytes
- Pages: 38
- Fatal LaTeX errors: none
- Undefined citations/references: none in final log scan
- Figure/table reference compliance: 22 figure/table labels, all cited in text, no missing label refs
- Citation consistency: 20 cited keys, 20 bibitems, no cited-without-bibitem keys, no uncited bibitems
- Remaining warnings: 9 underfull hbox warnings only; no overfull hbox warnings in the final log

Evidence-chain checks:

- Main 30-area aligned-cache result is traced to `experiments/macos_r2/results/retrain_v2/eval_paired_tests.json` and the consolidated `experiments/macos_r2/v4_manuscript_numbers.json`: n=30, mean advantage -0.0030039748, 16/30 positive, Wilcoxon p=0.5698576.
- Corrected multi-step rollout is traced to `experiments/macos_r2/results/e3_multistep/multistep_summary.json` and `multistep_paired_tests.json`.
- Per-year decoder retraining is traced to `experiments/macos_r2/results/e4_per_year_decoder/decoder_by_year.csv` and `per_pair_end_accuracy_delta.csv`.
- ESRI product-label validation is traced to `paper/rse_submission_paper58/revision_results/independent_change_validation_summary.json` and `independent_change_validation_by_area.csv`.
- Decoder confusion figure is traced to `src/adk_world_model/experiments/output/world_model_lulc_decode.json`.

Notes:

- Current referenced figure files are present both in `figures/` for separate upload and in `manuscript/figures/` for local compilation because the source uses `figures/...` include paths. Historical RSE/Paper8 figures that are not referenced by the SRS manuscript have been removed from the SRS package.
- The manuscript now frames ESRI labels as product labels, not manually interpreted CEOS-style reference data.
- The planning/RL material has been reduced to an ancillary diagnostic; the main evidence chain is remote-sensing validation, ESRI product-label change screening, and same-grid allocation diagnostics.
- Corresponding-author email and postal address have been inserted into the manuscript title page and submission metadata.
- CRediT author statement and arXiv preprint DOI metadata have been added for SRS/Elsevier compliance.
- Submission-time risk: create a DOI-archived Zenodo or institutional repository snapshot for the exact submitted GitHub state before final upload if the SRS/Elsevier workflow requires repository-deposited research data rather than a mutable GitHub URL.