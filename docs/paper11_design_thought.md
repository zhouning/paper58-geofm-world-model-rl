# Paper11 Design Thought: GeoFM-Enhanced Farmland Suitability and Spatial Layout Optimization

## Working Title

GeoFM-enhanced farmland suitability representation for reinforcement-learning-based spatial layout optimization.

## Starting Point

Paper1--Paper4 mainly optimize farmland spatial layout using explicit GIS and planning features. The core information includes land-use type, DEM-derived slope, parcel or block area, adjacency, and spatial contiguity. This design is interpretable and operational, but it represents farmland suitability only through a limited set of observable variables.

In farmland suitability theory, suitability is also affected by irrigation convenience, soil quality, water conditions, crop growth stability, surrounding land-use pressure, ecological background, and other environmental factors. These variables were not fully available in the earlier papers, so the DRL state and reward functions were necessarily simplified.

Paper11 starts from this gap: can AlphaEarth 64-dimensional embeddings enrich the environmental representation used by farmland-layout DRL when explicit irrigation, soil, and productivity data are missing?

## Core Hypothesis

AlphaEarth embeddings should not be treated as direct measurements of irrigation, soil quality, or fertility. Instead, they can be used as a remote-sensing foundation-model proxy for latent farmland suitability.

The 64-dimensional AlphaEarth vector may encode land-cover semantics, vegetation condition, water or wetness signals, radar structure, seasonal land-surface behavior, local spatial context, and environmental background. These signals can provide information related to farmland suitability that was absent from the original 14-dimensional Paper1-style state.

Therefore, Paper11 should treat AlphaEarth as an environmental semantic channel, not as a replacement for explicit planning constraints.

## Proposed State Representation

The block-level DRL state can be expanded from:

```text
state_old = [
  land-use type,
  slope,
  parcel/block geometry,
  adjacency,
  contiguity,
  fragmentation,
  other Paper1 planning features
]
```

to:

```text
state_new = [
  explicit planning features,
  DEM-derived terrain features,
  AlphaEarth 64-dimensional block embedding,
  optional public environmental features
]
```

The AlphaEarth embedding should be aggregated from 10 m pixels to parcels or management blocks. The first implementation should use the mean of the 64-dimensional embedding within each block:

```text
block_embedding = mean(pixel_embeddings_within_block)
```

More complex aggregation can be tested later, such as standard deviation, temporal change, quantiles, or attention-based aggregation. The first version should avoid unnecessary dimensional explosion.

## Interpretation of AlphaEarth for Paper11

AlphaEarth can plausibly enrich the state with:

- Land-cover and land-use semantics.
- Vegetation vigor and seasonal crop-growth signals.
- Water, wetness, or irrigation-related proxy signals.
- Surrounding land-use pressure, such as urban edge or construction disturbance.
- Spatial context and landscape texture.
- Environmental background related to terrain, climate, and long-term surface condition.

AlphaEarth should not be claimed to explicitly contain:

- Measured soil organic matter.
- Soil pH or nutrient contents.
- Irrigation canal infrastructure.
- Legal farmland protection status.
- Exact slope or elevation constraints.

Those variables should be kept explicit when available. Slope in particular should remain DEM-derived because it is a hard planning constraint and a direct reward component.

## Reward Function Direction

Paper11 can move beyond the earlier reward structure centered on slope reduction and contiguity improvement. A more realistic reward can combine spatial-layout quality with latent suitability:

```text
reward =
  contiguity_improvement
+ farmland_suitability_gain
+ stable_high-quality_farmland_preservation
- high_slope_farmland_penalty
- fragmentation_penalty
- conversion_of_unsuitable_land_penalty
- ecological_or_construction_conflict_penalty
+ irrigation_or_wetness_proxy_bonus
+ soil_or_productivity_proxy_bonus
```

The suitability terms should be carefully defined. A practical route is to learn a latent suitability score from weak labels:

```text
latent_suitability = g(AlphaEarth_64d, DEM, land_use)
```

Possible weak labels include stable farmland, low-slope persistent cropland, high-standard farmland if available, or areas historically retained as productive cropland. This score can then be used either as an input feature, a reward term, or both.

## Experimental Design

Paper11 should compare at least four state and reward settings:

1. Baseline Paper1-style DRL

```text
explicit planning features only
```

2. GeoFM-enhanced state

```text
explicit planning features + AlphaEarth 64d
```

3. GeoFM + DEM enhanced state

```text
explicit planning features + AlphaEarth 64d + richer DEM features
```

4. Suitability-aware DRL

```text
explicit planning features + AlphaEarth 64d + latent suitability reward
```

The key evaluation is not only whether the final reward is higher, but whether the optimized layout is more realistic under farmland suitability theory.

## Cross-Region Generalization

Cross-region transfer should be a major Paper11 contribution. Earlier DRL policies may overfit to local engineered features and local reward distributions. AlphaEarth provides a globally consistent remote-sensing representation, so it may improve policy transfer.

Suggested transfer tests:

```text
train on region A -> test on region B
train on multiple regions -> zero-shot test on held-out region
feature-only policy vs feature+AlphaEarth policy
AlphaEarth-only fallback policy
```

Expected result:

```text
feature + AlphaEarth > feature only > AlphaEarth only
```

If AlphaEarth improves transfer stability, Paper11 can argue that remote-sensing foundation-model embeddings help bridge the gap between local planning features and transferable spatial decision policies.

## Scientific Contribution

Paper11 can be framed as moving from:

```text
GIS-feature-based farmland layout optimization
```

to:

```text
remote-sensing-foundation-model-enhanced farmland suitability cognition and spatial decision optimization
```

The contribution is not merely adding 64 more input dimensions. The contribution is to use a foundation-model embedding as a latent environmental suitability representation that compensates for missing irrigation, soil, and productivity data, and to test whether this representation improves both decision quality and cross-region generalization in DRL-based farmland layout optimization.

## Key Caution

The manuscript must not overclaim that AlphaEarth directly measures irrigation or soil quality. The safer and more defensible wording is:

```text
AlphaEarth provides latent remote-sensing proxies for environmental and land-surface conditions related to farmland suitability.
```

Explicit DEM, slope, adjacency, contiguity, and planning constraints should remain in the model. AlphaEarth should enrich the representation rather than replace the planning logic.

