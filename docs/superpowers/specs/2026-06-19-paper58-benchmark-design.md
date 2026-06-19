# Paper58 External Benchmark Design

Date: 2026-06-19

## Goal

Turn Paper58 from a submission-polishing task into a strict evidence-first benchmark effort. The manuscript should advance only if new external validation shows strong, reproducible support for the core claim that frozen GeoFM embeddings can support baseline-trend change-signal modelling. If the evidence does not pass the benchmark gates, the RSE submission must be downgraded, reframed, or paused.

## Current Evidence Context

The current RSE revision package already contains a useful but bounded evidence chain:

- Area-level AlphaEarth embedding dynamics show a positive mean model-minus-persistence cosine advantage of +0.0047 across 10 valid cached grids, but the area-level bootstrap interval crosses zero.
- Independent ESRI-label validation currently covers 12 area-year pairs and shows non-zero categorical change signal: model change F1 0.335, spatial shuffle 0.269, label-only transition prior 0.166, and persistence 0.000.
- Full-map end-year categorical accuracy remains below persistence, so the model is not currently supported as an operational LULC forecaster.
- Planning results are useful diagnostics, but they do not yet prove cross-region operational planning transfer.
- Existing manifest files are not complete facts of record. For example, `data/independent_change_labels/label_manifest.json` records only a subset of label files that exist locally.

These facts make the current paper scientifically promising but not strong enough for the user's selected standard: extreme evidence before submission progress.

## Scientific Position

The benchmark must test a narrow and defensible claim:

> Frozen GeoFM embeddings can provide a low-cost state space for baseline-trend land-surface change-signal modelling when evaluated against independent labels and strong non-neural controls.

The benchmark must not try to prove these unsupported claims:

- Operational categorical LULC forecasting.
- Validated counterfactual scenario simulation.
- Broad region-independent superiority over persistence.
- Cross-region land-use planning deployment.

Negative results are part of the target output. If the benchmark shows that the model fails in some land-cover regimes or loses to strong baselines, those failures must be reported.

## Benchmark Tiers

### Tier 1: External Holdout Evidence

Tier 1 is the only evidence allowed to support the main manuscript claim. It consists of area-year pairs that were not used for dynamics training or tuning. Tier 1 pairs must have traceable labels, embeddings, predictions, terrain context where needed, and quality-control status.

Tier 1 must be stratified where data allow by land-use or ecological type:

- Urban.
- Agriculture.
- Wetland.
- Forest.
- Plateau.
- Mixed.

### Tier 2: Existing Study-Area Expansion

Tier 2 covers existing cached study areas, additional year pairs in areas already used for development, and any pairs with possible training or tuning contact. Tier 2 supports robustness analysis, failure analysis, and supplementary evidence, but it must not be mixed into the Tier 1 main endpoint.

### Tier 3: Diagnostic and Visual Evidence

Tier 3 covers interpretability and explanation outputs:

- Decoded map chips.
- Spatial error panels.
- Transition confusion tables.
- Embedding residual maps.
- Embedding residual PCA or clustering.
- Failure-case panels.

Tier 3 cannot independently support the core claim. It explains Tier 1 and Tier 2 behavior.

## Evidence Gates

The benchmark must evaluate these endpoints at area-year-pair or region-cluster level, not by treating pixels as independent samples.

### Primary Change Endpoint

Endpoint:

```text
model_change_f1 - best_non_neural_baseline_change_f1
```

Gate:

- The 95 percent region-clustered bootstrap confidence interval must be strictly positive in Tier 1.

### Spatial Localization Endpoint

Endpoint:

```text
model_change_f1 - spatial_shuffle_change_f1
```

Gate:

- The 95 percent region-clustered bootstrap confidence interval must be strictly positive in Tier 1.

### Embedding Dynamics Endpoint

Endpoint:

```text
model_cosine_similarity - persistence_cosine_similarity
```

Gate:

- If the 95 percent confidence interval is strictly positive, the embedding-space result can be a main result.
- If the interval crosses zero, the embedding-space result can only be secondary or diagnostic.

### Stratified Robustness Endpoint

Endpoint:

```text
direction of model advantage by land-use or ecological type
```

Gate:

- At least three strata must show directionally positive Tier 1 evidence before any broad robustness language is allowed.
- Failed strata must appear in the main results or supplement with a concrete failure explanation.

## Baselines

The benchmark must compare against these baselines:

1. Persistence: predict the end-year categorical map as the start-year map.
2. Spatial shuffle model control: preserve the model-predicted class histogram and destroy spatial correspondence.
3. Label-only transition prior: use leave-out empirical class-transition frequencies without embeddings or dynamics predictions.
4. Leave-one-region temporal prior: use region-held-out historical change tendency to predict a change mask or transition distribution.
5. Linear embedding delta baseline: fit a linear or ridge model for embedding residuals to test whether LatentDynamicsNet adds nonlinear or local-context value.
6. Dynamics ablation: include at least one implementation-relevant ablation such as no dilated convolution, no L2 re-normalization, or no terrain context.

The strongest non-neural baseline in each endpoint is the comparison target for the primary gate.

## Dataset Registry

A new benchmark registry must become the source of truth. It must be generated by scanning actual files rather than trusting stale manifest files.

Required scan sources:

- `data/independent_change_labels/labels/*.npy`
- `data/independent_change_labels/predicted/*.npy`
- `data/independent_change_labels/embeddings/*.npy`
- `experiments/paper8/data/**`
- Existing revision result files under `paper/rse_submission_paper58/revision_results/`

Each registry row must describe one candidate area-year pair and include:

- Area name.
- Start year and end year.
- Tier assignment.
- Land-use or ecological stratum.
- Label start path.
- Label end path.
- Prediction path.
- Embedding start path.
- Embedding end path.
- Terrain context path if used.
- Bounding box if known.
- Scale.
- Array shapes.
- Data source.
- Training or development contact status.
- QC status.
- Exclusion reason when excluded.

The registry must be written to:

```text
paper/rse_submission_paper58/benchmark_results/benchmark_registry.json
paper/rse_submission_paper58/benchmark_results/benchmark_registry.csv
```

## Quality Control

Each candidate pair must pass quality control before evaluation:

- Start labels, end labels, predictions, and embeddings must refer to the same area and year interval.
- Label and prediction shapes must match exactly.
- Embedding shapes must match the prediction grid or have an explicit registered alignment rule.
- Arrays must be non-empty.
- Predictions must not be constant maps unless the pair is explicitly marked as a no-change negative control.
- Change prevalence must be recorded. Zero-change pairs can be retained only as negative controls, not as positive change evidence.
- Class distributions must be recorded for labels and predictions.
- Missing files, shape mismatches, class collapse, untraceable bounding boxes, and stale manifests must produce explicit exclusion reasons.

Excluded samples must remain in the registry and be reported in summary outputs.

## Statistics

The statistical unit is the area-year pair, clustered by region where repeated yearly pairs exist.

Required statistics:

- Area-year-pair means and medians.
- Region-clustered bootstrap confidence intervals.
- Area-level bootstrap confidence intervals where region clusters are unavailable.
- Paired sign tests for directional consistency.
- Stratified summaries by land-use or ecological type.
- Separate Tier 1 and Tier 2 summaries.

Pixel-level metrics can be reported descriptively, but pixel count cannot be used as the sample size for significance claims.

## Outputs

The benchmark pipeline must write results under:

```text
paper/rse_submission_paper58/benchmark_results/
```

Required outputs:

- `benchmark_registry.json`
- `benchmark_registry.csv`
- `benchmark_metrics_by_pair.csv`
- `benchmark_summary.json`
- `benchmark_summary_by_tier.csv`
- `benchmark_summary_by_stratum.csv`
- `benchmark_gate_report.json`
- `benchmark_failures.csv`
- Rebuildable figures from benchmark outputs only.

Existing `revision_results/` files must not be overwritten by the new benchmark pipeline.

## Implementation Architecture

Create a separate benchmark package under:

```text
scripts/paper58_benchmark/
```

Planned modules:

- `build_registry.py`: scan files, infer candidate pairs, assign tiers, run QC, and write registry files.
- `baselines.py`: implement persistence, spatial shuffle, label-only transition prior, leave-one-region temporal prior, and linear embedding delta baseline helpers.
- `evaluate_benchmark.py`: read the registry, evaluate only QC-passing pairs, and write metric tables.
- `statistics.py`: implement pair-level, area-level, and region-clustered summaries and gate checks.
- `make_benchmark_figures.py`: generate paper-candidate and supplement-candidate figures from benchmark outputs.

The new benchmark modules may reuse code from `scripts/rse_revision/`, but benchmark outputs must remain separate until the evidence gate passes.

## Testing

Tests must be added before or with implementation:

- Registry tests for discovered pairs, missing predictions, shape mismatch, class collapse, zero-change negative controls, and Tier 1/Tier 2 separation.
- Baseline tests using small toy arrays for persistence, spatial shuffle, label-only transition prior, and leave-one-region temporal prior.
- Statistics tests proving confidence intervals use pair or region-cluster units rather than pixels.
- Evaluation tests proving Tier 1 and Tier 2 are not pooled for the main gate.
- Figure tests proving missing benchmark inputs fail explicitly.

## Manuscript Gate

Before the benchmark gate passes, manuscript edits are limited to:

- Removing or downgrading unsupported claims.
- Fixing references and data availability language.
- Clarifying limitations and current evidence status.
- Recording that stronger evidence is being built.

After the benchmark gate passes:

- Tier 1 becomes the main result.
- Tier 2 becomes robustness and failure analysis.
- Tier 3 becomes explanatory figures and supplement.
- The manuscript can be revised for RSE only if all main claims map directly to benchmark outputs.

If the benchmark gate fails:

- Do not force an RSE submission under the current forecasting framing.
- Reframe the paper as an audit of frozen GeoFM embeddings for change-signal modelling, or pause submission until stronger data exist.
- Keep negative results visible.

## Submission Readiness Criteria

RSE submission work can resume only when all criteria below are satisfied:

- Tier 1 primary change endpoint confidence interval is strictly positive.
- Tier 1 spatial localization endpoint confidence interval is strictly positive.
- At least three land-use or ecological strata show directionally positive Tier 1 evidence.
- Failure regions and excluded pairs are documented.
- Every figure and table used in the manuscript can be rebuilt from scripts.
- The manuscript avoids operational, counterfactual, and broadly superior language unless directly supported by benchmark evidence.

## Risks

- New external areas may require network or GEE access and may fail because of authentication, quota, or data availability.
- ESRI annual labels can be noisy or internally inconsistent, especially for small patches and wetland transitions.
- A larger benchmark may weaken the current positive story. This is acceptable under the evidence-first principle.
- Linear or temporal-prior baselines may outperform LatentDynamicsNet in some regimes. Those outcomes must be reported rather than hidden.

## Approved Direction

The selected direction is route B: build an external validation benchmark first, then decide whether the manuscript deserves RSE submission work. Route A, simple strengthening of existing cached evidence, is too weak for the selected evidence threshold. Route C, full scenario and operational planning validation, is stronger but too large for the current Paper58 submission cycle.
