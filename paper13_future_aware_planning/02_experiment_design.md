# Paper13 Experiment Design

## Research Question

Does coupling land-state prediction with DRL planning produce farmland layouts that perform better under future land-use conditions than layouts optimized only for the current state?

Operational version:

```text
Given data up to year t, predict land state at t+k.
Optimize farmland layout using current-only, future-only, or current+future state.
Evaluate each plan under target-year land conditions.
```

## Study Units

Two feasible spatial units:

1. Parcel/block-level planning units.
2. Region clusters derived from AlphaEarth embedding space.

The preferred Paper13 unit is parcel/block. It aligns with farmland spatial layout optimization and earlier DRL work. Pixel or cluster-level experiments can be used as a fast pilot, but the final manuscript should return to planning units.

## Data Layers

Minimum required:

- Current land-use / land-cover labels.
- Historical AlphaEarth embeddings, ideally annual 2017-2024.
- DEM-derived elevation and slope.
- Parcel or management-block geometry.
- Adjacency / contiguity graph.
- Current farmland mask and swappable candidate mask.

Recommended:

- Target-year land-use labels for holdout evaluation.
- Construction land or urban expansion labels.
- Ecological redline / protected area mask if available.
- High-standard farmland or stable cropland weak labels.
- Existing Paper7/Paper8 trajectory data for pilot training.

## Prediction Layer

The prediction layer can output either latent or decoded future variables.

### Latent Output

```text
z_pred_{t+k}: predicted AlphaEarth embedding
delta_z = z_pred_{t+k} - z_t
prediction_uncertainty
```

Advantages:

- Preserves richer semantics than hard land-use labels.
- Compatible with AlphaEarth-based suitability scoring.
- Avoids forcing every planning feature through a decoder.

Risk:

- Harder to explain to planning readers unless translated into interpretable variables.

### Decoded Output

```text
lulc_pred_{t+k}
p(farmland stable)
p(construction expansion)
p(ecological conversion)
future_suitability_score
future_risk_score
```

Advantages:

- Easier to interpret and evaluate.
- Better for reward design and manuscript figures.

Risk:

- Decoder errors can propagate.
- Class probabilities may look more certain than the latent model actually is.

Recommended: use both. Keep `z_pred` in the DRL state and derive decoded risk/suitability terms for reward and interpretation.

## DRL Settings

### Setting A: Current-State DRL

State:

```text
current planning features + current land-use + DEM + adjacency + z_t
```

Reward:

```text
current contiguity / slope / suitability / fragmentation
```

Purpose:

Baseline equivalent to Paper11-style planning.

### Setting B: Future-State DRL

State:

```text
predicted future land-use + predicted future suitability + predicted risk + z_pred_{t+k}
```

Reward:

```text
future contiguity / future risk avoidance / future suitability
```

Purpose:

Tests whether planning only from the predicted target-year background works.

Expected issue:

It may ignore present feasibility. This should not be the recommended final model, but it is an important ablation.

### Setting C: Current + Future Joint-State DRL

State:

```text
current state + predicted future state + delta change features
```

Reward:

```text
current feasibility + future outcome quality
```

Purpose:

Main proposed model.

Expected result:

This should outperform current-only and future-only because it balances today's constraints with future risk.

### Setting D: Rule-Based Future-Aware Planning

Rules:

```text
avoid high predicted construction risk
prioritize stable high-suitability farmland
prefer adjacent candidate blocks
penalize high slope and ecological conflict
```

Purpose:

Shows whether DRL adds value beyond simple use of future prediction.

### Setting E: No-Planning / Persistence

Keep current farmland layout unchanged.

Purpose:

Lower bound and policy-relevant reference.

## Evaluation Metrics

### Future Outcome Metrics

- Target-year farmland contiguity.
- Target-year fragmentation.
- Target-year average slope of retained or planned farmland.
- Share of planned farmland overlapping predicted or observed future construction expansion.
- Share of planned farmland in stable high-suitability areas.
- Farmland-loss risk exposure.
- Ecological conflict rate.

### Robustness Metrics

- Performance across multiple future horizons: `t+1`, `t+3`, `t+5`.
- Performance across model uncertainty samples or ensemble predictions.
- Worst-case and mean-case future reward.
- Sensitivity to prediction error.

### Transfer Metrics

- Train on one region, test on another.
- Train on multiple regions, hold out one region.
- Compare current-only vs AlphaEarth-enhanced vs future-aware policies.

### Prediction Metrics

These validate the future-state estimator before using it for planning:

- Embedding cosine similarity.
- Changed-pixel or changed-block advantage over persistence.
- Decoded land-use accuracy / F1.
- Farmland-to-construction and farmland-to-nonfarmland transition accuracy.
- Calibration of future risk scores if probabilities are used.

## Experimental Matrix

| Experiment | Purpose | Compare |
| --- | --- | --- |
| E1 prediction validation | prove future layer is usable | world model vs persistence vs linear extrapolation |
| E2 current/future DRL | core Paper13 question | current-only vs future-only vs current+future |
| E3 rule baseline | show DRL is needed | joint-state DRL vs rule-based future-aware planning |
| E4 horizon sensitivity | test future horizon | 1-year vs 3-year vs 5-year target |
| E5 prediction-error robustness | avoid overclaiming | clean prediction vs noisy prediction vs ensemble |
| E6 transfer | test generalization | within-region vs cross-region |
| E7 component ablation | isolate design choices | no `z_pred`, no decoded risk, no delta features, no uncertainty |

## Suggested Tables

### Table 1: Future Prediction Performance

Columns:

```text
region, split, horizon, cos_sim, persistence_cos, changed_block_F1, farmland_loss_AUC
```

### Table 2: Planning Performance Under Target-Year Conditions

Columns:

```text
method, future_reward, contiguity, fragmentation, slope, construction_conflict, stable_farmland_preservation
```

### Table 3: Cross-Region Transfer

Columns:

```text
train_region, test_region, current-only, AlphaEarth-current, future-aware, improvement
```

### Table 4: Ablation

Columns:

```text
variant, future_reward, risk_exposure, contiguity, transfer_score
```

## Suggested Figures

1. Conceptual pipeline: prediction to planning.
2. Map comparison: current-only plan vs future-aware plan.
3. Future-risk heatmap with selected farmland blocks.
4. Target-year metric bar chart.
5. Horizon sensitivity curve.
6. Transfer performance plot.

## Validity Checks

Before reporting planning results, run these checks:

1. Confirm future predictions outperform persistence on changed farmland-relevant blocks.
2. Confirm decoded future risk is not simply current land-use copied forward.
3. Confirm current+future policy is evaluated on data not used to generate reward labels when possible.
4. Confirm action masks enforce hard planning constraints.
5. Confirm no claim of causal action effect is made unless intervention data support it.

## Primary Risk

The biggest threat is circular evaluation: optimizing on a predicted future and then evaluating on the same prediction.

Preferred evaluation order:

```text
train prediction model on years <= t
predict t+k
optimize plan using predicted t+k
evaluate plan using observed t+k or independently generated held-out labels
```

If observed target-year data are unavailable, the manuscript must describe the evaluation as scenario-based or simulation-based rather than real future validation.

