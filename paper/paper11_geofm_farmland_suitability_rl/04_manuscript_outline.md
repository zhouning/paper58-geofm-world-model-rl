# Paper11 Manuscript Outline

## 1. Working Titles

Recommended:

```text
GeoFM-enhanced farmland suitability representation for reinforcement-learning-based spatial layout optimization
```

Alternative:

```text
Remote-sensing foundation-model embeddings improve transferable farmland layout optimization
```

More planning-oriented:

```text
Latent environmental suitability from GeoFM embeddings for farmland consolidation planning
```

## 2. One-Sentence Argument

In farmland spatial layout optimization, we show that frozen GeoFM embeddings can serve as latent environmental suitability proxies for DRL policies, improving decision quality and cross-region transfer when combined with explicit DEM, adjacency, contiguity, and planning constraints.

## 3. Contribution Framing

Paper11 should claim three contributions:

1. A GeoFM-enhanced state representation for block-level farmland spatial layout optimization.
2. A weakly supervised latent suitability score that converts remote-sensing foundation-model semantics into a planning-relevant reward term.
3. A transfer-oriented evaluation showing whether globally consistent GeoFM embeddings improve cross-region DRL policy generalization.

Do not frame the contribution as:

```text
AlphaEarth directly measures soil and irrigation quality.
```

Frame it as:

```text
AlphaEarth provides latent remote-sensing proxies for environmental and land-surface conditions related to farmland suitability.
```

## 4. Abstract Logic

A future abstract should follow this chain:

1. Farmland layout optimization needs suitability information, but explicit soil, irrigation, and productivity data are often unavailable.
2. Existing DRL planning environments rely on explicit GIS features and therefore underrepresent latent environmental suitability.
3. Paper11 augments block-level DRL states with AlphaEarth 64-dimensional embeddings and a weakly supervised suitability score.
4. The method retains explicit slope, contiguity, baimu-fang, and planning constraints.
5. Experiments compare explicit-only, GeoFM-enhanced, suitability-reward, and full models across in-region and transfer settings.
6. The expected contribution is a more transferable and environmentally informed planning policy, bounded to proxy suitability rather than direct soil or irrigation measurement.

## 5. Introduction Structure

### Paragraph 1: Planning Problem

Farmland spatial layout optimization is not only a geometric consolidation problem. It also requires deciding whether farmland should be retained, exchanged, or consolidated in places that are environmentally suitable for stable agricultural production.

### Paragraph 2: Limitation of Existing DRL Planning

Earlier DRL approaches can encode slope, adjacency, parcel area, contiguity, and fragmentation, but they often lack explicit data on irrigation, soil, water conditions, and long-term productivity. This makes the state representation operational but incomplete.

### Paragraph 3: Opportunity From GeoFM

GeoFM embeddings such as AlphaEarth provide globally available, multi-sensor, 64-dimensional land-surface representations. These embeddings may encode vegetation, wetness, land-cover context, urban pressure, and environmental background relevant to suitability.

### Paragraph 4: Key Caution

These embeddings are not direct measurements of soil or irrigation. The scientific question is whether they are useful latent proxies for decision-making when direct variables are unavailable.

### Paragraph 5: Paper Contribution

Introduce GeoFM-enhanced state representation, weak suitability reward, and cross-region transfer evaluation.

## 6. Method Structure

### 6.1 Problem Formulation

Define farmland layout optimization as a block-level MDP:

```text
state: block features + global planning features
action: select a block for investment
transition: execute greedy parcel swaps within selected block
reward: slope, contiguity, baimu-fang, suitability terms
```

### 6.2 GeoFM Block Representation

Describe pixel-to-block aggregation:

```text
block_embedding = mean(pixel_embeddings within block)
```

Explain why mean aggregation is the first version: simple, stable, and avoids overfitting.

### 6.3 Latent Suitability Learning

Describe weak labels and suitability model:

```text
suitability = g(GeoFM embedding, DEM, land-use features)
```

Make clear that this is weakly supervised proxy suitability.

### 6.4 Suitability-Aware MDP

Describe expanded state and reward:

```text
state = explicit features + GeoFM embedding + suitability score
reward = base planning reward + suitability gain - unsuitable conversion penalty
```

### 6.5 Policy

Describe the shared per-block scorer policy, emphasizing variable block counts and transfer.

## 7. Results Structure

### Result 1: Suitability Proxy Is Meaningful

Show suitability score validation using weak labels and distributional diagnostics.

### Result 2: GeoFM State Improves Planning

Compare explicit-only and explicit+GeoFM models.

### Result 3: Suitability Reward Improves Spatial Realism

Show that suitability-aware reward improves suitability-weighted farmland metrics while preserving slope and contiguity.

### Result 4: GeoFM Improves Transfer

Show reduced performance drop in held-out regions.

### Result 5: Ablations Confirm Boundaries

Show that random/shuffled embeddings do not perform like real GeoFM embeddings and that explicit constraints remain necessary.

## 8. Discussion Structure

The discussion should make four points:

1. GeoFM embeddings can help bridge missing environmental data in planning-scale DRL.
2. Suitability proxy learning is useful because raw embeddings alone may not be decision-aligned.
3. Cross-region transfer is the strongest reason to use globally consistent foundation-model representations.
4. GeoFM should enrich, not replace, explicit planning constraints and domain rules.

## 9. Claim Boundaries

Use bounded verbs:

- `suggests` for latent environmental interpretation;
- `indicates` for transfer trends;
- `shows` only for measured performance comparisons;
- `enables` for system capability.

Avoid:

- `measures soil quality`;
- `detects irrigation infrastructure`;
- `solves farmland suitability`;
- `universal transfer`;
- `foundation model replaces GIS planning`.

