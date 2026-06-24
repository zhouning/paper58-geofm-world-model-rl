# Paper58 GeoSOS-FLUS Surpass Design

## Goal

Advance Paper58 from a GeoFM embedding change predictor into a land-use simulation model that can be evaluated against, and ultimately outperform, GeoSOS-FLUS under matched experimental conditions.

The target system is provisionally named **Paper58-LAS: GeoFM Latent Allocation Simulator**.

## Hard Requirements

1. Paper58 remains the technical foundation. The AlphaEarth embedding state space, LatentDynamicsNet forecast, LULC decoder, strict holdout registry, and existing benchmark evidence are preserved as the base system.
2. The new system must include real architectural innovation beyond a direct FLUS imitation.
3. The claim "surpasses GeoSOS-FLUS" is allowed only after matched-condition experiments show stable advantage over GeoSOS-FLUS or a verified FLUS-compatible implementation.
4. Existing contradictory evidence remains visible. Batch 2 failure, `xiong_an_fringe_holdout`, and Batch 5 `liaohe_delta_wetland_holdout` cannot be hidden by pooled results.

## Current Evidence Boundary

Paper58 currently supports the claim that frozen GeoFM embeddings plus LatentDynamicsNet can provide useful land-cover change signals under strict external holdout evaluation. It does not yet support the stronger claim that Paper58 is a full land-use simulation model comparable to GeoSOS-FLUS.

The strongest current constraints are:

- Batch 2 only failed the spatial gate because spatial CI low was below zero.
- `xiong_an_fringe_holdout` remains the decisive Batch 2 spatial failure.
- Batch 3 independently passed and is supportive evidence.
- Batch 5 mixed gate passed, but `liaohe_delta_wetland_holdout` is negative on both primary and spatial change advantage.
- Batch 5 pass does not erase Batch 2 failure or row-level Batch 5 risks.

Therefore the next phase must be designed as a new simulation-and-allocation extension, not as manuscript strengthening from existing pooled benchmark evidence.

## Relationship To GeoSOS-FLUS

GeoSOS-FLUS is treated as the operational benchmark class: land-use suitability, demand constraints, cellular/neighborhood effects, conversion restrictions, and competitive allocation. Paper58-LAS should match this behavioral surface while replacing FLUS's hand-engineered or ANN-style suitability layer with a GeoFM latent world-model layer.

Paper58-LAS should not be described as "FLUS with AlphaEarth" because the intended contribution is stronger:

- suitability is derived from forecasted AlphaEarth latent dynamics;
- neighborhood pressure can be computed in both class space and embedding space;
- transition confidence can be diagnosed at embedding, decoder, and allocation levels;
- cross-region generalization can use globally consistent AlphaEarth representations rather than locally tuned driver variables alone.

## Proposed Architecture

### Layer 1: Preserved Paper58 Base

Inputs:

- start-year AlphaEarth embedding grid;
- start-year LULC label grid;
- optional context embedding grids already supported by Paper58;
- trained LatentDynamicsNet forecast;
- trained LULC decoder or decoder probability interface;
- existing holdout registry and strict provenance metadata.

Outputs:

- forecast embedding grid;
- decoder probability cube;
- direct Paper58 predicted LULC map;
- change pressure map derived from start-vs-forecast embedding displacement and decoder probability shift.

This layer must remain separately evaluable so that Paper58 direct prediction and Paper58-LAS allocation can be compared.

### Layer 2: Latent Transition Suitability

Create a transition-suitability tensor:

```text
S[pixel, from_class, to_class]
```

The first implementation should combine:

- decoder probability of the target class on the forecast embedding;
- probability gain from start embedding to forecast embedding;
- embedding residual magnitude as a change-pressure term;
- transition priors learned from leave-one-region training rows;
- optional calibration terms for known weak transitions such as class-11 wetland/new urban classes.

The output is a normalized, class-aware suitability surface for each allowed transition.

### Layer 3: Demand-Constrained Allocation

Add a demand allocator that accepts target class demands:

```text
D[to_class] = target number of pixels or target area
```

Demand can come from:

- observed end-year map for benchmark oracle-demand experiments;
- Markov or historical transition projection for realistic forecast experiments;
- external scenario table for policy/scenario simulation;
- GeoSOS-FLUS output demand when reproducing published or user-provided FLUS runs.

The allocator should satisfy demand within a configurable tolerance and must report quantity disagreement separately from allocation disagreement.

### Layer 4: Latent Neighborhood Competition

For each candidate transition, compute a neighborhood score that combines:

- local class-neighborhood composition;
- neighboring decoder probability mass for the target class;
- latent embedding similarity or residual coherence with nearby candidate pixels;
- optional edge/contiguity penalties to avoid salt-and-pepper artifacts.

This layer is the main architectural innovation beyond simply adding a FLUS-style postprocessor. FLUS-style CA competition is extended from discrete land-use classes into the GeoFM latent semantic space.

### Layer 5: Constraints And Conversion Rules

Support hard constraints:

- immutable classes;
- forbidden transitions;
- protected-area or exclusion mask;
- water/building/farmland preservation rules where supplied;
- class-specific maximum or minimum conversion budgets.

Constraints must be applied before allocation and rechecked after allocation. Violations should fail closed in benchmark mode.

### Layer 6: Simulation Output

The simulator writes:

- simulated LULC map;
- per-class demand satisfaction table;
- transition matrix;
- change mask;
- selected transition score table;
- allocation trace or iteration summary;
- diagnostic layers for suitability, neighborhood pressure, and constraint masking.

## Experimental Design

### Compared Methods

At minimum:

1. Persistence.
2. Existing non-neural baselines: transition prior and temporal prior.
3. Current Paper58 direct prediction.
4. FLUS-compatible baseline.
5. GeoSOS-FLUS real output when available.
6. Paper58-LAS.

If real GeoSOS-FLUS output is unavailable at first, a FLUS-compatible open implementation can be used for development, but final surpass claims require either real GeoSOS-FLUS output or a documented compatible reproduction using the same demand, mask, and neighborhood assumptions.

### Matched Conditions

Every compared method must share:

- same start map;
- same end-year target for benchmark evaluation;
- same class definitions;
- same region extent and resolution;
- same demand source;
- same exclusion mask and conversion rules;
- same holdout membership and provenance rules.

### Metrics

Required metrics:

- change F1;
- Figure of Merit (FOM);
- transition accuracy on changed pixels;
- per-transition recall for dominant transitions;
- quantity disagreement;
- allocation disagreement;
- spatial change advantage over spatial shuffle;
- per-class and per-stratum summaries;
- batch-level confidence intervals with cluster bootstrap, consistent with existing Paper58 benchmark practice.

### Gates

The claim ladder is:

1. **Mechanism complete**: Paper58-LAS produces demand-constrained maps without constraint violations.
2. **Competitive**: Paper58-LAS beats current Paper58 direct prediction and FLUS-compatible baseline on most independent holdouts.
3. **GeoSOS-FLUS surpass candidate**: Paper58-LAS beats GeoSOS-FLUS under matched demand, mask, and metric settings across multiple independent batches.
4. **Strong surpass claim**: Paper58-LAS passes independent batch gates with positive lower confidence bounds and no hidden failure batch.

Initial surpass gate:

- positive mean advantage over GeoSOS-FLUS in FOM or change F1;
- positive lower confidence bound for Tier 1 rows;
- positive advantage in at least Urban, Wetland, Agriculture, and one Forest/Grassland stratum;
- no reliance on pooled pass to hide Batch 2 or Batch 5 risk cases;
- explicit row-level reporting for `xiong_an_fringe_holdout` and `liaohe_delta_wetland_holdout`.

## Implementation Scope For First Build

The first build should be a simulator extension and benchmark harness, not a retraining effort.

Add modules for:

- demand derivation and validation;
- transition-suitability tensor construction;
- constrained allocation;
- latent-neighborhood scoring;
- FLUS-compatible result ingestion;
- expanded metric computation;
- report writing.

Do not initially change:

- LatentDynamicsNet architecture;
- existing benchmark thresholds;
- existing Batch 2/3/4/5 evidence;
- manuscript claims.

This keeps the first phase testable and reversible. If the allocation layer exposes systematic failures, later work can add training-level changes such as allocation-aware loss, spatial-localization loss, wetland/class-11 calibration, or latent-neighborhood contrastive regularization.

## Data Flow

```text
holdout registry row
  -> load start LULC, end LULC, Paper58 prediction, embeddings
  -> Paper58 base forecast / existing prediction artifacts
  -> decoder probability cube
  -> transition suitability tensor
  -> demand table
  -> constraint mask and conversion rules
  -> latent-neighborhood competition
  -> demand-constrained allocation
  -> simulated LULC map
  -> FLUS-facing evaluation
  -> batch reports and row diagnostics
```

## Failure Handling

Benchmark mode should fail closed when:

- demand totals do not match the raster size or allowed editable area;
- class definitions differ across methods;
- FLUS output shape differs from the target row;
- constraints cannot be satisfied within tolerance;
- a Tier 1 row lacks required provenance or comparable baseline artifacts.

Diagnostic mode may continue with warnings, but its outputs must be labeled as non-gate evidence.

## Testing Strategy

Unit tests:

- demand derivation preserves totals;
- forbidden transitions are never selected;
- allocation meets class demand within tolerance;
- neighborhood scoring is deterministic for a fixed seed;
- FLUS-ingestion rejects shape and class mismatches.

Integration tests:

- run a tiny synthetic holdout through the full Paper58-LAS path;
- compare Paper58 direct prediction, FLUS-compatible baseline, and Paper58-LAS in one report;
- verify failure rows are written rather than silently skipped.

Regression tests:

- existing Paper58 benchmark tests must continue to pass;
- Batch 2 and Batch 5 historical reports must not be overwritten by the new simulator.

## Expected Outcome

If successful, Paper58-LAS converts Paper58 from a latent change predictor into a defensible land-use simulation framework:

- it preserves the original GeoFM world-model contribution;
- it adds a new latent allocation and neighborhood competition architecture;
- it becomes directly comparable with GeoSOS-FLUS;
- it creates an evidence path for claiming superiority only after matched-condition experiments support that claim.

