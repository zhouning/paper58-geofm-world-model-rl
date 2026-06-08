# Paper11 System Design

## 1. Design Goal

Build a GeoFM-enhanced DRL planning system that keeps the existing farmland layout MDP interpretable, while adding a latent suitability representation derived from AlphaEarth embeddings.

The system should answer three questions:

1. Does GeoFM information improve action selection beyond explicit slope, area, and contiguity features?
2. Does a learned suitability score provide a better reward signal than raw embeddings alone?
3. Does GeoFM improve cross-region transfer of the policy?

## 2. Recommended Architecture

The recommended approach is an incremental extension of the current county/block environment.

```text
DLTB parcels + DEM slope + block definitions
        -> explicit block features

AlphaEarth 10 m annual embeddings
        -> block-level embedding aggregation
        -> optional latent suitability model

explicit block features + GeoFM embedding + suitability score
        -> MaskablePPO with per-block scorer
        -> block selection actions
        -> greedy parcel swap inside selected block
        -> suitability-aware reward and planning metrics
```

This preserves the current action semantics:

```text
action = select one block for investment
```

It does not introduce a future-state simulator or scenario action.

## 3. Data Units

### Parcel

Parcel data keep the existing fields:

- land-use category;
- DLBM code;
- slope;
- area;
- geometry;
- township and county identifiers;
- swappability status.

### Block

Blocks remain the management unit for action selection. Each block receives:

- existing 17-dimensional explicit feature vector;
- 64-dimensional AlphaEarth block embedding;
- optional embedding dispersion features;
- optional latent suitability score;
- optional suitability uncertainty score.

### Global State

Global state keeps the existing county-level planning features:

- budget remaining;
- global slope;
- global contiguity;
- step fraction;
- slope improvement;
- contiguity improvement;
- baimu-fang count and area;
- investment distribution signals.

Do not mix GeoFM into the global vector in the first version. Keep GeoFM as a per-block property so the scorer can learn which blocks are suitable for action.

## 4. GeoFM Aggregation

The first implementation should use mean aggregation from pixels to blocks:

```text
block_embedding_b = mean(z_p for p inside block b)
```

Use the same year as the planning base year. If the planning base year is unavailable, use the nearest available AlphaEarth annual embedding and record the mismatch.

Recommended optional features for later ablation:

```text
embedding_std_b = std(z_p for p inside block b)
embedding_crop_centroid_distance_b = cosine_distance(block_embedding_b, stable_farmland_centroid)
embedding_water_centroid_distance_b = cosine_distance(block_embedding_b, wetland_or_water_centroid)
```

The first paper version should avoid high-dimensional quantile or attention aggregation unless mean aggregation fails.

## 5. Latent Suitability Model

Paper11 should not claim that AlphaEarth directly measures farmland suitability. Instead, it can train a weakly supervised suitability model:

```text
suitability_b = g(
    AlphaEarth_64d_b,
    DEM/slope features_b,
    land-use features_b
)
```

Weak positive labels can include:

- stable cropland over multiple years;
- low-slope persistent cropland;
- high-standard farmland polygons, if available;
- historically retained productive cropland;
- farmland retained near water or irrigation proxies, if defensible from data.

Weak negative labels can include:

- steep cropland converted away from farmland;
- construction-disturbed edge parcels;
- fragmented cropland repeatedly targeted for conversion;
- forest or ecological land that is unsuitable for farmland expansion under planning rules.

The output should be interpreted as a planning suitability proxy:

```text
suitability_b in [0, 1]
```

## 6. State Design

Recommended first state:

```text
block_features_p11 = [
  existing_17_block_features,
  AlphaEarth_64d_mean,
  latent_suitability_score
]
```

This gives:

```text
K_BLOCK_P11 = 17 + 64 + 1 = 82
```

The policy can reuse the existing per-block scorer pattern:

```text
Input to scorer for each block:
  block_features_p11 + global_features

Output:
  one logit per block
```

A lower-dimensional variant can be added if 64 raw dimensions are unstable:

```text
block_features_p11_compressed = [
  existing_17_block_features,
  PCA_8_or_16(AlphaEarth_64d_mean),
  latent_suitability_score
]
```

The raw 64-dimensional version should be the main version because the paper's scientific claim depends on foundation-model semantics, not only a hand-designed scalar.

## 7. Reward Design

Keep the existing planning reward components:

```text
reward_base =
  slope_weight * slope_delta
+ cont_weight * contiguity_delta
+ baimu_weight * baimu_area_delta
+ baimu_bonus * baimu_new_count
- invalid_action_penalty
```

Add a bounded suitability term:

```text
reward_p11 =
  reward_base
+ suit_weight * suitability_gain
- conflict_weight * unsuitable_conversion_penalty
```

Where:

```text
suitability_gain =
  mean_suitability(new_farmland_after_action)
- mean_suitability(old_farmland_before_action)
```

The unsuitable conversion penalty should discourage gaining farmland area in blocks whose latent suitability is low or whose explicit constraints are poor:

```text
unsuitable_conversion_penalty =
  converted_area_low_suitability
  * max(0, suitability_threshold - suitability_b)
```

Slope must remain explicit and dominant. A block with high AlphaEarth suitability but high DEM slope should not be rewarded as high-quality farmland.

## 8. Policy Design

Use the existing shared scorer strategy:

```text
score_b = MLP([block_features_p11_b, global_features])
action = sample_or_argmax(masked_softmax(score))
```

This design has three advantages:

- it supports variable numbers of blocks across regions;
- it directly tests whether block-level GeoFM features improve action ranking;
- it can transfer to regions with different block counts.

Possible variants:

```text
Policy A: explicit features only
Policy B: explicit + raw AlphaEarth 64d
Policy C: explicit + suitability score
Policy D: explicit + raw AlphaEarth 64d + suitability score
```

Policy D is recommended as the full Paper11 model.

## 9. Implementation Boundary

The first Paper11 implementation should not:

- train a temporal GeoFM dynamics model;
- predict future land-use maps;
- use scenario actions;
- optimize directly in embedding-grid space;
- remove explicit planning constraints;
- interpret AlphaEarth channels as named soil or irrigation measurements.

