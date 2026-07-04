# Science of Remote Sensing / Elsevier requirements audit

Official pages checked on 2026-07-04:

- Journal page: https://www.sciencedirect.com/journal/science-of-remote-sensing
- Guide for Authors: https://www.sciencedirect.com/journal/science-of-remote-sensing/publish/guide-for-authors

The checklist below follows the SRS/Elsevier author-facing requirements visible on 2026-07-04. Recheck the live Editorial Manager upload flow because Elsevier pages and transfer forms are dynamic.

## Prepared items

- Main manuscript source: `manuscript/srs_geofm_embedding_change_screening.tex`
- Main manuscript PDF: compiled from the source after verification
- Highlights: `submission_docs/highlights_srs.txt`, 5 bullets, each under 85 characters
- Keywords: 7 keywords in the manuscript
- Abstract: unstructured abstract, currently under 250 words
- Cover letter: `submission_docs/cover_letter_srs.md`
- Competing interest declaration: included in manuscript
- Funding statement: included in manuscript
- Data and code availability: included in manuscript and separate text file; persistent DOI snapshot remains an upload-time author action
- Generative AI declaration: included in manuscript following Elsevier-style disclosure
- CRediT author statement: included in manuscript for the single author
- Figures: copied into `figures/` for separate upload if requested
- References: switched to numeric square-bracket natbib style for Elsevier compatibility
- Peer review model: SRS uses single anonymized review, so author-identifying title-page information is retained

## SRS-specific reframing checks

- The title now leads with remote-sensing foundation-model embeddings and LULC screening, not a generic world-model claim.
- The abstract reports the negative persistence result and the bounded change-screening result.
- The Introduction starts from LULC change assessment and validation practice.
- Related Work includes land-cover product validation and accuracy assessment.
- The Discussion frames the contribution as foundation-model embedding validation for remote sensing.
- The manuscript explicitly rejects operational forecasting and native-driver replacement claims.

## Remaining upload-time checks

- Corresponding-author email and postal address have been inserted from the author's provided information.
- Create a DOI-archived repository snapshot before final upload, then add the DOI to the submission metadata if available.
- Check whether the live upload system asks for a graphical abstract. None is currently prepared.
- Check whether SRS requires separate declaration-of-interest and CRediT entry forms in addition to manuscript text.
- Add a phone number manually if Editorial Manager requires one; none was provided in the author details received for this package.
- Confirm whether the transfer workflow imports old RSE metadata; if so, overwrite the title, abstract, highlights, and cover letter with the SRS versions.
