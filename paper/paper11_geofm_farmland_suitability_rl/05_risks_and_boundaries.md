# Paper11 Risks and Boundaries

## 1. Main Scientific Risk

The largest risk is over-interpreting AlphaEarth embeddings. The paper must not imply that a 64-dimensional remote-sensing embedding directly measures soil fertility, irrigation access, or legal farmland quality.

Safe claim:

```text
GeoFM embeddings provide latent remote-sensing proxies for land-surface and environmental conditions related to farmland suitability.
```

Unsafe claim:

```text
GeoFM embeddings measure irrigation and soil quality.
```

## 2. Methodological Risks

### Risk: Dimensionality Masquerading as Semantics

Adding 64 dimensions may improve learning simply because the network has more input capacity.

Control:

```text
explicit + random 64d
explicit + shuffled GeoFM 64d
```

### Risk: Suitability Score Learns Land-Use Labels Only

The suitability model may only identify current cropland rather than suitability for future or retained farmland.

Control:

```text
stable farmland labels
low-slope persistent cropland labels
held-out spatial validation
distribution checks by land-use and slope class
```

### Risk: Reward Gaming

The policy may chase suitability reward while harming slope, contiguity, or baimu-fang formation.

Control:

```text
hard slope penalty
explicit contiguity term
multi-metric reporting
Pareto plots of suitability vs planning metrics
```

### Risk: Transfer Failure

GeoFM may help in-region optimization but not held-out regions.

Control:

```text
report transfer drop
include multiple region types
analyze where and why transfer fails
```

## 3. Implementation Risks

### Pixel-to-Block Alignment

AlphaEarth pixels and parcel/block geometries must be aligned carefully. Misalignment can corrupt block embeddings.

Required checks:

- CRS consistency;
- block geometry validity;
- pixel count per block;
- missing embedding rate;
- distribution of embedding norms;
- visual inspection for selected blocks.

### Temporal Mismatch

If AlphaEarth year and land-use planning base year differ, suitability may be biased.

Required record:

```text
planning_base_year
AlphaEarth_embedding_year
land_use_data_year
DEM_year_or_source
```

### Weak Label Leakage

If weak labels are generated directly from final evaluation metrics, the suitability reward may leak evaluation logic.

Mitigation:

Use separate weak-label construction and final evaluation metrics where possible. Report the overlap honestly when unavoidable.

## 4. Recommended First Version

The first version should be conservative:

```text
state:
  existing 17 block features
  + AlphaEarth 64d mean
  + 1 suitability score

reward:
  existing slope/contiguity/baimu reward
  + bounded suitability gain
  - low-suitability conversion penalty

policy:
  existing shared per-block scorer

evaluation:
  B0/B1/B2/B3 comparison
  random/shuffled embedding control
  held-out-region transfer
```

This version is enough to support a coherent paper. More complex aggregation, attention, temporal dynamics, and scenario actions should be left for later papers unless the simple version fails.

## 5. Next Decisions Before Implementation

The next implementation step needs five decisions:

1. Which regions are available for Paper11 beyond Bishan?
2. Which year should be the planning base year?
3. Are high-standard farmland, irrigation, soil, or productivity labels available for validation?
4. Should AlphaEarth be used as raw 64d, PCA-compressed, or both?
5. Should the first experiment modify the real `CountyLevelEnv` or build a separate `Paper11GeoFMEnv` wrapper?

Recommended answer for decision 5:

```text
Build a separate Paper11GeoFMEnv wrapper first.
```

Reason: it avoids destabilizing existing legacy runtime code and lets Paper11 preserve a clean experimental boundary.

