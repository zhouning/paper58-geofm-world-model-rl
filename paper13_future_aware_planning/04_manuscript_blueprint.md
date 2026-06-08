# Paper13 Manuscript Blueprint

## Candidate Title

Future-Aware Farmland Spatial Planning by Coupling Foundation-Model Land-State Prediction with Deep Reinforcement Learning

Alternative:

Prediction-Optimization Coupled Farmland Layout Planning Under Future Land-Use Change

## Abstract Logic

1. Farmland spatial planning is usually optimized against current land-use conditions.
2. Current-state optimization can be fragile when land-use change is expected.
3. GeoFM embeddings and lightweight latent dynamics make future land-state prediction feasible.
4. We couple predicted future land states with DRL-based farmland layout optimization.
5. Experiments compare current-only, future-only, joint current+future, rule-based, and persistence baselines.
6. The contribution is future-aware decision coupling, not merely adding remote-sensing features.

## Introduction Argument

### Problem

Farmland layout optimization needs to preserve productive, contiguous, and feasible farmland. However, the land-use background is not static. Urban expansion, ecological restoration, land degradation, and local land-cover transitions can change which areas are suitable for future farmland protection or consolidation.

### Gap

Existing land-use change models usually predict future maps but do not optimize planning decisions. Existing DRL-based farmland planning usually optimizes on the current state and ignores predicted future change.

### Insight

If a model can predict future land-state risk, then farmland planning can be optimized for target-year conditions rather than only present conditions.

### Contribution

The paper introduces a prediction-optimization coupling framework:

```text
GeoFM-based future land-state prediction -> future-aware DRL planning
```

## Methods Structure

### 1. Problem Formulation

Define farmland layout planning under future land-use change:

```text
Given state S_t and predicted future state S_hat_{t+k},
find a planning policy pi that maximizes expected target-year planning utility.
```

Separate three spaces:

- current observation space
- predicted future land-state space
- planning action and reward space

### 2. Future Land-State Prediction

Describe AlphaEarth/LatentDynamicsNet at the level needed for Paper13:

- frozen AlphaEarth embeddings
- residual latent dynamics
- L2 manifold preservation
- decoded future land-use and risk variables
- limitation: baseline trajectory, not action-conditioned causality

### 3. Future-Aware Planning State

Define:

```text
S_current
S_future
Delta S
S_joint = [S_current, S_future, Delta S]
```

Explain why each is needed:

- current state: feasibility and hard constraints
- future state: target-year risk and suitability
- delta: direction and magnitude of expected change

### 4. DRL Optimization

Describe action space, masks, policy, and reward.

Reward should emphasize:

- future contiguity
- stable high-suitability farmland preservation
- risk avoidance
- slope and ecological conflict penalties
- robustness if multiple futures are used

### 5. Experimental Design

Compare:

- no planning / persistence
- current-state DRL
- future-state DRL
- current+future DRL
- rule-based future-aware planning

Include transfer and ablation if space allows.

## Results Story

Expected result structure:

1. Future prediction is good enough on changed farmland-relevant blocks.
2. Current+future DRL improves target-year planning utility.
3. Future-only DRL is useful but weaker than joint-state DRL because it loses present feasibility.
4. Rule-based future-aware planning helps but underperforms DRL on contiguity and multi-objective tradeoffs.
5. AlphaEarth future features improve cross-region robustness.
6. Prediction uncertainty and horizon length affect planning quality.

## Safe Claim Language

Use:

```text
baseline future land-state prediction
future-aware optimization
prediction-informed planning
target-year evaluation
expected future risk
```

Avoid unless new data support it:

```text
causal effect of planning actions on land-use change
arbitrary counterfactual simulation
intervention-aware world model
policy action changes the predicted future trajectory
```

## Discussion Points

### Why Coupling Matters

Prediction alone does not produce a planning decision. Optimization alone may choose layouts that look good now but become fragile under future land-use change. Coupling the two gives the planner a way to act on forecast information.

### Why AlphaEarth Helps

AlphaEarth provides a globally consistent latent representation of land-surface condition. It can encode signals that explicit planning features miss, while DEM, slope, adjacency, and legal constraints remain explicit.

### Why Not Full Model-Based RL Yet

The current evidence supports baseline future-state estimation. Full model-based RL would require validated action-conditioned dynamics. This boundary makes the paper more defensible.

### Planning Relevance

The output is not only a prediction map. It is a spatial layout decision evaluated under future conditions, which is closer to planning practice.

## Limitations

- Future prediction errors propagate into planning.
- AlphaEarth embeddings are proxies, not direct measurements of soil, irrigation, or legal status.
- Baseline future prediction cannot identify arbitrary intervention effects.
- Target-year validation depends on available future observations.
- Reward weights may reflect planning priorities and require sensitivity analysis.

## Target Contribution Statement

Final manuscript wording can be:

```text
This study shifts farmland spatial layout optimization from current-state decision-making to future-aware decision-making by coupling GeoFM-based latent land-state prediction with DRL-based spatial optimization. The framework uses predicted baseline future land states to guide farmland preservation and consolidation decisions, while preserving explicit planning constraints and evaluating outcomes under target-year conditions.
```

