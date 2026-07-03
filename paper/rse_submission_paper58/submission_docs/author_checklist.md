# RSE Author Checklist

## Journal Requirements Verified

Official target journal: Remote Sensing of Environment (RSE), ScienceDirect Guide for Authors.

- [x] Article type checked: Original Research Article.
- [x] RSE word limit checked: 15,000 words including references and figure captions.
- [x] Manuscript word count checked by the consistency script: 13,879 words.
- [x] RSE abstract limit checked: 250 words.
- [x] Current abstract word count: 217 words.
- [x] RSE highlights requirement checked: 3-5 bullets, each no more than 85 characters including spaces.
- [x] Current highlights file: `submission_docs/highlights.md` with 4 bullets, maximum 72 characters.
- [x] RSE validation expectation addressed through paired tests, independent ESRI-label validation, and same-grid allocation ablation.

## Manuscript Package

- [x] Compile `manuscript/rse_geofm_world_model_rl_v5.tex` inside this package without missing figures.
- [x] Confirm the final PDF is placed at `manuscript/rse_geofm_world_model_rl_v5.pdf` after the final compile pass.
- [x] Confirm all figure files referenced by the manuscript compile successfully from `figures/`.
- [x] Review layout warnings. Severe issues from the v5 audit were fixed: no `Float too large`, no undefined citations, and no undefined references remain after two `pdflatex` runs. Minor overfull hbox warnings may remain from long terms and paths.

## Scientific Framing

- [x] Keep RSE framing centered on remote-sensing foundation-model embeddings and land-surface dynamics.
- [x] Present the RL planning experiment as downstream decision-support evidence, not as an operational deployment.
- [x] Avoid claiming completed scenario-conditioned counterfactual simulation unless scenario-labeled training data are added.
- [x] Clearly explain that AlphaEarth/Prithvi encoder-ablation statistics are diagnostic and reported on valid cached areas.
- [x] Treat the GeoSOS-FLUS comparison as a stripped-down same-grid allocation ablation, not a native-driver competition.

## Submission Metadata

- [x] Author name and affiliation are present in the manuscript source.
- [x] Corresponding author information is present in the manuscript source.
- [x] Complete funding statement: no specific grant funding received.
- [x] Complete conflict-of-interest declaration.
- [x] Complete data and code availability with the public GitHub repository URL.
- [x] Complete generative-AI disclosure wording in `declarations.md` and the manuscript source.
- [ ] Add an archived DOI snapshot for the public repository if the journal or editor requests permanent archival storage before acceptance.

## Final Checks

- [x] Verify references, citations, word count, highlights, and result-table consistency with the manuscript consistency script.
- [ ] Confirm figure resolution and readability by final human visual PDF inspection before upload. Raster render QA on sampled pages passed, but this item remains an author-side upload check.
- [x] Confirm primary tables do not overflow the page after resizing/shortening.
- [x] Confirm title, abstract, keywords, declarations, and availability statements are consistent with the bounded claims.

