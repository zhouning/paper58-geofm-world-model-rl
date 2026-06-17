# Paper58 RSE Revision Design

Date: 2026-06-17

## Goal

Upgrade the RSE submission package from a concept-forward draft into a data-traceable pre-submission manuscript. The revised paper should make a narrower but defensible claim: frozen geospatial foundation-model embeddings can support lightweight baseline-trend land-surface dynamics prediction, especially for change-prone areas, while counterfactual scenario simulation remains future work.

## Reviewer Risks To Resolve

1. Author and submission metadata are incomplete or stale.
2. Some main figures are generated from mock values rather than traceable result files.
3. Scenario-conditioned projections are presented despite the scenario branch being trained only on the baseline scenario.
4. The manuscript overstates decision-support and counterfactual planning readiness.
5. Existing visual evidence is mostly generic charts, with insufficient remote-sensing spatial evidence.
6. Metrics need statistical uncertainty and clearer support for change-pixel claims.
7. The data structure must be clarified: random point sampling supports vector evaluation, while convolutional dynamics requires gridded embeddings.

## Scientific Positioning

The retained innovation claim is:

- A frozen GeoFM embedding field is treated as a predictive state space rather than only a static feature source.
- A small residual latent dynamics model is trained on top of frozen embeddings instead of retraining the foundation encoder or generating pixel-level imagery.
- L2 manifold preservation, multi-step unrolling, and change-focused evaluation are presented as methodological requirements for embedding-space forecasting.

The downgraded claims are:

- Scenario-conditioned counterfactual simulation: architectural placeholder only.
- Planning-oriented decision support: downstream application probe only, not an operational policy simulator.
- Broad generalization: supported only within the cached Chinese study-area evidence; extreme biome and unlabeled-policy scenarios remain limitations.

## Evidence Plan

### Keep And Strengthen

- AlphaEarth vs persistence comparisons where local cached arrays and/or recorded result files support them.
- Prithvi encoder comparison, but present it as an encoder-representation diagnostic rather than a full fair model-vs-model benchmark.
- Feature-dropout planning results, but shorten the main text and avoid claiming completed cross-region planning transfer.

### Add

- A reproducible revision results script that loads local JSON/CSV/NPY files and writes derived tables with confidence intervals, paired tests, and provenance metadata.
- A revised figure script with no mock-data path. It must fail if required results are absent unless explicitly run in a diagnostic mode.
- Spatial evidence panels from cached embedding grids where available: embedding-change intensity, persistence error, model error when a trained model is locally available, and LULC decoder outputs only when the decoder/provenance is available.
- A provenance table documenting which figures/tables are generated from which files.

### Remove Or Move

- The current scenario projection figure must be removed from main results. If retained, it should become a schematic or supplementary caveat figure, not evidence of scenario performance.
- Claims that other scenario weights learned counterfactual behavior must be removed.
- Unsupported "first" and "operational planning" language should be softened.

## Figure Plan

Use Python/matplotlib because the repository already uses that backend.

- Figure 1: architecture schematic revised to show baseline-trained latent dynamics and explicitly mark scenario conditioning as unactivated.
- Figure 2: data-traceable per-area performance with confidence intervals or split-level uncertainty.
- Figure 3: spatial case-study evidence from available cached grids.
- Figure 4: change-focused evaluation and statistical uncertainty.
- Figure 5: ablations/encoder diagnostics, with captions matching computed values.
- Supplementary: planning feature-dropout box/point plot and metadata/provenance table.

## Manuscript Edits

- Rewrite abstract to focus on tested baseline-trend dynamics.
- Rework introduction contributions into narrower evidence-backed claims.
- Split methodology into point-sampled evaluation and gridded dynamics training/evaluation.
- Replace scenario-conditioning section with a short "architectural extension not evaluated here" paragraph.
- Rewrite results around data-traceable figures and statistics.
- Compress downstream planning section.
- Expand limitations with scenario grounding, missing operational labels, sampling constraints, and transfer limits.
- Update declarations, data/code availability, author metadata placeholders, and compile notes.

## Verification

- Run the revision result-building script.
- Run the figure script and inspect generated PDFs/PNGs.
- Compile the RSE manuscript with `pdflatex` at least twice.
- Check for missing figures, undefined references, inconsistent figure captions, and severe overfull boxes.
- Run `git diff --check` before final summary.
