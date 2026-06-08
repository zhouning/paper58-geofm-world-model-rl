# Paper11 Experiment Plan

## 1. Main Hypotheses

### H1: Representation

GeoFM-enhanced state features improve farmland layout optimization compared with explicit GIS features alone.

### H2: Suitability Reward

A weakly supervised latent suitability score improves the realism of optimized farmland layouts beyond slope and contiguity rewards.

### H3: Transfer

GeoFM-enhanced policies transfer better across regions because AlphaEarth embeddings provide a globally consistent environmental representation.

## 2. Experimental Conditions

Use at least four main conditions:

| ID | State | Reward | Purpose |
|---|---|---|---|
| B0 | explicit planning features | base reward | Paper1/Paper3/Paper4-style baseline |
| B1 | explicit + AlphaEarth 64d | base reward | tests representation-only gain |
| B2 | explicit + suitability score | base + suitability reward | tests distilled suitability channel |
| B3 | explicit + AlphaEarth 64d + suitability score | base + suitability reward | full Paper11 model |

Add two diagnostic conditions if time permits:

| ID | State | Reward | Purpose |
|---|---|---|---|
| D1 | AlphaEarth 64d only | base reward | tests whether explicit planning features are still necessary |
| D2 | explicit + random 64d | base reward | controls for dimensionality rather than semantic information |

Expected relationship:

```text
B3 >= B2 or B1 > B0 > D1
B1 > D2
```

The exact order of B1 and B2 is empirical. If B2 outperforms B1, the paper can argue that suitability distillation makes foundation-model information more decision-aligned.

## 3. Metrics

### Planning Quality

Use existing planning metrics:

- average farmland slope change;
- contiguity improvement;
- baimu-fang count change;
- baimu-fang area change;
- total reward;
- valid action rate;
- budget efficiency.

### Suitability Quality

Add Paper11-specific metrics:

- mean latent suitability of final farmland;
- suitability-weighted farmland area;
- low-suitability farmland area;
- high-suitability farmland preservation rate;
- action concentration in high-suitability candidate blocks.

### Spatial Realism

Use map-level and block-level diagnostics:

- fragmentation index;
- compactness;
- adjacency of newly consolidated farmland;
- conflict with high-slope or ecological blocks;
- distribution of actions across suitability quantiles.

### Transfer

For train-region to held-out-region evaluation:

- zero-shot reward;
- zero-shot planning metrics;
- zero-shot suitability metrics;
- drop from in-region performance;
- action ranking stability by suitability quantile.

## 4. Transfer Design

The most important Paper11 evidence is cross-region transfer.

Suggested splits:

```text
train: multiple townships or county regions
validation: held-out township within the same county
test: different county or physiographic region
OOD: region with clearly different terrain, urban pressure, or ecological background
```

Run:

```text
train on region A -> test on region B
train on multiple regions -> zero-shot test on held-out region
feature-only policy vs feature+GeoFM policy
GeoFM-only fallback policy
```

The main claim is stronger if GeoFM improves the drop ratio rather than only improving in-region reward:

```text
transfer_drop = in_region_score - held_out_score
```

## 5. Ablation Matrix

### GeoFM Feature Ablations

| Variant | Meaning |
|---|---|
| raw 64d mean | main AlphaEarth block representation |
| PCA-8/PCA-16 | dimensionality-reduced version |
| suitability scalar only | decision-aligned distilled proxy |
| random 64d | dimensionality control |
| shuffled GeoFM among blocks | spatial semantic control |

### Reward Ablations

| Variant | Meaning |
|---|---|
| base reward | slope, contiguity, baimu-fang only |
| base + suitability gain | rewards improvement in farmland suitability |
| base + unsuitable penalty | penalizes poor suitability conversions |
| base + both | full Paper11 reward |

### Constraint Ablations

These are diagnostic and should not be framed as recommended models:

| Variant | Meaning |
|---|---|
| no slope term | tests whether GeoFM can replace DEM slope |
| no contiguity term | tests whether GeoFM can replace spatial structure |
| no action mask | tests whether planning constraints are necessary |

Expected result:

```text
Removing explicit constraints should degrade planning validity.
```

This supports the key boundary claim: GeoFM enriches planning logic but does not replace it.

## 6. Suitability Model Validation

Before using suitability in reward, validate the weakly supervised model.

Minimum checks:

- AUC or F1 for stable farmland weak labels;
- calibration curve for suitability score;
- suitability distribution by land-use class;
- suitability distribution by slope quantile;
- map inspection of high- and low-suitability blocks;
- correlation with available external proxies, if any.

If no external high-standard farmland or soil labels are available, report this honestly and frame suitability as weakly supervised proxy suitability.

## 7. Figure Plan

### Figure 1: Concept

Show the shift from explicit GIS-only state to GeoFM-enhanced suitability state.

```text
explicit constraints + GeoFM latent semantics -> suitability-aware DRL -> optimized layout
```

### Figure 2: System

Show data flow:

```text
parcels/DEM/blocks + AlphaEarth embeddings -> block features -> policy -> action -> reward
```

### Figure 3: Main Performance

Compare B0, B1, B2, B3 on planning metrics and suitability metrics.

### Figure 4: Transfer

Show in-region and held-out-region performance, with transfer drop.

### Figure 5: Spatial Case Study

Map initial layout, baseline optimized layout, and Paper11 optimized layout for one representative region.

### Figure 6: Ablations

Show raw GeoFM, PCA, suitability scalar, random features, and shuffled features.

## 8. Evidence Needed for Manuscript Claims

| Claim | Required evidence |
|---|---|
| GeoFM improves representation | B1 outperforms B0 and D2 |
| Suitability reward improves realism | B2/B3 improve suitability metrics without harming slope or contiguity |
| GeoFM improves transfer | B1/B3 have smaller transfer drop than B0 |
| Explicit constraints remain necessary | D1 and no-slope/no-contiguity variants fail key planning metrics |
| AlphaEarth is a proxy, not a measurement | careful wording plus weak-label validation |

