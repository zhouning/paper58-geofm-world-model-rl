# Paper13 Design Synthesis

## One-Sentence Idea

Paper13 should be framed as a future-aware farmland spatial planning framework that couples remote-sensing foundation-model-based land-state prediction with reinforcement-learning-based spatial layout optimization.

The key novelty is not "AlphaEarth plus DRL." It is the decision coupling:

```text
land-use change prediction becomes an operational input to farmland layout optimization
```

Most land-use prediction studies stop at monitoring or forecasting. Most farmland DRL studies optimize against the present planning state. Paper13 bridges this gap by asking whether planning against a predicted future state produces layouts that are more robust to expected land-use change.

## What Paper13 Should Not Be

Paper13 should not be presented as:

- Paper11 with more input dimensions.
- A purely methodological world-model paper.
- A causal intervention simulator unless suitable intervention data exist.
- A claim that AlphaEarth directly measures soil, irrigation, or legal protection status.

The safe scientific boundary is:

```text
The model estimates baseline future land states from observed historical dynamics.
The DRL planner uses those predicted states to optimize future-facing layout decisions.
```

The risky boundary is:

```text
The model predicts how any planning action causally changes future land use.
```

That second claim requires action-conditioned training data, credible counterfactual constraints, or a separate causal identification design.

## Relationship to Existing Papers

### Paper11

Paper11 focuses on current-state representation:

```text
explicit GIS features + DEM + AlphaEarth embedding -> DRL optimization
```

Its research question is whether AlphaEarth's 64-dimensional remote-sensing embedding can serve as a latent environmental suitability proxy when soil, irrigation, productivity, and local ecological data are incomplete.

Paper13 should inherit this representation, but add a temporal prediction layer:

```text
current suitability representation
  -> future land-state prediction
  -> future-aware DRL optimization
```

Thus Paper13 is more ambitious and more decision-oriented. Paper11 improves how the current state is represented; Paper13 changes the time horizon of the decision problem.

### Paper10

Paper10 is methodological. It asks whether a world model trained for embedding reconstruction preserves decision-relevant ordering, and whether discriminative or ranking losses are needed.

Paper13 is application-driven. It can cite Paper10's logic if the world-model prediction needs stronger decision relevance, but it should not merge the two papers too early.

Useful distinction:

```text
Paper10: Is the learned latent prediction objective decision-relevant?
Paper13: Can predicted future land states improve farmland layout planning?
```

### Paper58

Paper58 provides the technical substrate:

- AlphaEarth annual 64-dimensional embeddings as a frozen GeoFM state space.
- LatentDynamicsNet as a lightweight residual dynamics model.
- L2 re-normalization to keep predictions on the unit hypersphere.
- Dilated convolution to include spatial context.
- Multi-step rollout loss to reduce exposure bias.
- Baseline-trend prediction, not validated intervention simulation.
- Existing RL-related prototypes in embedding space.

For Paper13, Paper58's world model should be used as a future-state estimator first. The most defensible design is:

```text
AlphaEarth / LatentDynamicsNet predicts z_{t+k}
decoder or suitability head derives future planning variables
DRL optimizes layout using z_t and predicted z_{t+k}
```

## Recommended Technical Level

### Level 1: Passive Future-Aware Optimization

This should be the main Paper13 version.

The world model predicts the future land-state background under a baseline trend. DRL actions are optimized against that predicted future background, but they do not modify the world model's trajectory.

Pipeline:

```text
current parcel/block state
  + AlphaEarth embedding z_t
  + DEM and planning constraints
    -> baseline future prediction z_{t+k}
    -> decoded future land use / suitability / risk
    -> future-aware DRL state and reward
    -> optimized farmland layout for target year
```

This supports a clear question:

```text
Under expected future land-use change, does future-aware optimization preserve better farmland layout quality than current-state optimization?
```

### Level 2: Active Intervention-Aware Optimization

This is a future extension, not the first manuscript claim.

In Level 2, actions enter the dynamics model:

```text
planning action a_t
  -> action-conditioned future state prediction z_{t+1}
  -> long-horizon reward
  -> model-based RL
```

The repository already contains intervention-style prototypes, including `InterventionEnv`, `DualRepEnv`, and `InterventionDynamicsNet`. These are useful design references, but Paper13 should only use them as primary evidence if the training transitions genuinely represent intervention effects rather than learned proxies from existing optimization trajectories.

## Core State Design

At the block or parcel level:

```text
state_current = [
  explicit planning features,
  land-use class,
  DEM / slope,
  geometry / area,
  adjacency / contiguity,
  AlphaEarth embedding z_t
]
```

Future prediction layer:

```text
future_state = [
  predicted embedding z_{t+k},
  decoded future land-use class,
  future farmland-loss risk,
  future construction-conflict risk,
  future ecological or low-suitability risk,
  optional uncertainty score
]
```

DRL input:

```text
state_paper13 = [
  state_current,
  future_state,
  delta_state = future_state - current_state,
  planning mask and constraints
]
```

The delta is important because it tells the planner what is changing, not only what the future state looks like.

## Core Reward Design

The reward should evaluate the planning decision under future conditions, not just under the current map:

```text
reward =
  future_contiguity_gain
+ stable_high-suitability_farmland_preservation
+ low-risk_farmland_allocation
- high-slope_farmland_penalty
- future_farmland_loss_risk
- predicted_urban_expansion_conflict
- future_fragmentation_penalty
+ optional robustness_across_future_scenarios
```

The current-state reward should remain available as a baseline. Paper13's central comparison is whether future-aware reward terms improve target-year outcomes.

## Main Hypotheses

H1. Future-aware DRL avoids allocating farmland to areas with high predicted conversion or degradation risk better than current-state DRL.

H2. Joint current-plus-future state performs better than future-only state because planning also needs present constraints, legal feasibility, and current spatial structure.

H3. AlphaEarth-derived future embeddings improve cross-region transfer by providing a consistent latent land-surface representation across regions.

H4. Prediction uncertainty matters: layouts optimized on a single predicted future may overfit, while uncertainty-aware or multi-scenario optimization should produce more robust plans.

## Scientific Contribution

The contribution should be written as:

```text
We introduce a future-aware prediction-optimization coupling framework for farmland spatial planning, in which GeoFM-based latent land-state prediction informs DRL-based spatial layout optimization under expected future land-use change.
```

More specific contributions:

1. Formulate future-aware farmland planning as a two-stage prediction-optimization problem.
2. Convert AlphaEarth/world-model predictions into parcel/block-level future planning variables.
3. Design a DRL state and reward that jointly encode present feasibility and predicted future risk.
4. Evaluate whether future-aware optimization improves target-year farmland contiguity, suitability, risk avoidance, and transfer robustness.

## Key Design Principle

The planner should be judged by target-year outcomes.

If the model predicts 2030 conditions, the plan must be evaluated under observed 2030 data where available, a held-out future year proxy, or a carefully documented simulated future. Evaluating only against the same predicted state used for optimization would create a circular result.

