# Compile verification

Date: 2026-07-04

Command run from `paper/srs_submission_paper58/manuscript`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error srs_geofm_embedding_change_screening.tex
pdflatex -interaction=nonstopmode -halt-on-error srs_geofm_embedding_change_screening.tex
```

Result:

- PDF generated: `manuscript/srs_geofm_embedding_change_screening.pdf`
- PDF size: 697,249 bytes
- Pages: 39
- Fatal LaTeX errors: none
- Undefined citations/references: none in final log scan
- Citation consistency: 20 cited keys, 20 bibitems, no cited-without-bibitem keys, no uncited bibitems
- Remaining warnings: 9 underfull hbox warnings only; no overfull hbox warnings in the final log

Notes:

- Figure files are present both in `figures/` for separate upload and in `manuscript/figures/` for local compilation because the source uses `figures/...` include paths.
- The manuscript now frames ESRI labels as product labels, not manually interpreted CEOS-style reference data.
- The planning/RL material has been reduced to an ancillary diagnostic; the main evidence chain is remote-sensing validation, ESRI product-label change screening, and same-grid allocation diagnostics.
- Corresponding-author email and postal address have been inserted into the manuscript title page and submission metadata.
- CRediT author statement and arXiv preprint DOI metadata have been added for SRS/Elsevier compliance.
