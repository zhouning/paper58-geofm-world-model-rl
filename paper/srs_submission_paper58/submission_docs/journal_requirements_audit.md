# Science of Remote Sensing / Elsevier requirements audit

Official pages checked on 2026-07-05:

- Journal page: https://www.sciencedirect.com/journal/science-of-remote-sensing
- Guide for Authors: https://www.sciencedirect.com/journal/science-of-remote-sensing/publish/guide-for-authors

The checklist below follows the SRS/Elsevier author-facing requirements visible on 2026-07-05. Recheck the live Editorial Manager upload flow because Elsevier pages and transfer forms are dynamic.

## Prepared items

- Main manuscript source: `manuscript/srs_geofm_embedding_change_screening.tex`
- Main manuscript PDF: compiled from the source after verification
- Highlights: `submission_docs/highlights_srs.txt`, 5 bullets, each under 85 characters
- Keywords: 7 keywords in the manuscript
- Abstract: unstructured abstract, 216 words, under the 250-word SRS limit
- Cover letter: `submission_docs/cover_letter_srs.md` and plain-text upload copy `submission_docs/cover_letter_srs.txt`
- Competing interest declaration: included in manuscript and separate upload file `submission_docs/declaration_of_competing_interests_srs.txt`
- Funding statement: included in manuscript
- Data and code availability: included in manuscript and separate text file; DOI repository snapshot is still an upload-time action
- Generative AI declaration: included in manuscript following Elsevier-style disclosure
- CRediT author statement: included in manuscript for the single author
- Figures: 2 referenced external figures copied into `figures/` and `manuscript/figures/`; PNG companions are available for upload if requested
- Figure/table citation rule: 22 figure/table labels, all cited in text, no missing label refs
- References: numeric square-bracket natbib style for Elsevier compatibility; cited keys and bibitems are consistent
- Peer review model: SRS uses single anonymized review, so author-identifying title-page information is retained

## SRS-specific reframing checks

- The title now leads with remote-sensing foundation-model embeddings and LULC screening, not a generic world-model claim.
- The abstract reports the negative/non-significant persistence result and the bounded change-screening result.
- The Introduction starts from LULC change assessment and validation practice.
- Related Work includes land-cover product validation and accuracy assessment.
- The Discussion frames the contribution as foundation-model embedding validation for remote sensing.
- The manuscript explicitly rejects operational forecasting, scenario-conditioned world-model, and native-driver replacement claims.

## Evidence-chain checks

- Current 30-area aligned-cache result: `experiments/macos_r2/results/retrain_v2/eval_paired_tests.json` and `experiments/macos_r2/v4_manuscript_numbers.json`.
- Mismatch/old-cache result: retained only as a reproducibility warning, not as the current SRS evidence source.
- ESRI product-label validation: `paper/rse_submission_paper58/revision_results/independent_change_validation_summary.json` and `independent_change_validation_by_area.csv`.
- Decoder confusion figure: `src/adk_world_model/experiments/output/world_model_lulc_decode.json`.
- Spatial validation figure: Banzhucun 2020-2023 labels/prediction arrays under `data/independent_change_labels/`.
- Old SRS figure files not referenced by the current manuscript have been removed from the SRS package.

## Remaining upload-time checks

- Create a DOI-archived Zenodo or institutional repository snapshot for the exact submitted GitHub state before final upload if the SRS/Elsevier workflow enforces repository-deposited research data. Treat this as high priority because SRS follows a research-data deposit policy, not merely a mutable code-link preference.
- Add the repository DOI to the manuscript Data and Code Availability section and submission metadata if available before upload.
- Add a phone number manually if Editorial Manager requires one; none was provided in the author details received for this package.
- Check whether the live upload system asks for a graphical abstract. None is currently prepared.
- If Editorial Manager requires a form-field declaration of interest, paste the prepared no-competing-interests text from `submission_docs/declaration_of_competing_interests_srs.txt`; CRediT remains included in the manuscript for the single author.
- Confirm whether the transfer workflow imports old RSE metadata; if so, overwrite the title, abstract, highlights, and cover letter with the SRS versions.
