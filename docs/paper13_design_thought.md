# Paper13 Design Thought: Future-Aware Prediction-Optimization Coupled Farmland Planning

## Working Title

Future-aware prediction-optimization coupled planning for farmland spatial layout under land-use change.

## Core Idea

Paper13 is defined as the idea of "prediction-optimization coupled planning for future land states."

The key shift is from optimizing farmland layout under the current land-use state to optimizing farmland layout under a predicted future land-use state. Instead of asking only:

```text
Given the current land-use pattern, what is the best farmland layout?
```

Paper13 asks:

```text
Given the likely future land-use pattern, what farmland layout should be planned now or for a target future year?
```

This turns the problem into a coupled pipeline:

```text
Current observations and planning data
        ↓
Future land-state prediction
        ↓
Future-aware DRL optimization
        ↓
Planning solution robust to expected future change
```

## Relationship to Paper11

Paper11 focuses on enriching the DRL state representation for farmland suitability and spatial layout optimization. It uses:

```text
explicit planning features + DEM + AlphaEarth 64-dimensional embeddings
```

to compensate for missing suitability-related variables such as irrigation convenience, soil quality, water conditions, vegetation status, and environmental background.

Paper13 builds on Paper11 but goes beyond it. Paper11 is mainly about richer representation of the current planning state. Paper13 adds a temporal prediction layer:

```text
Paper11:
current suitability representation -> DRL optimization

Paper13:
current suitability representation -> future land-state prediction -> DRL optimization
```

Thus, Paper13 is not merely "Paper11 with more features." It is a planning framework that explicitly accounts for expected future land-use change before optimization.

## Relationship to Paper10

Paper10 is conceptually different. Based on existing design notes, Paper10 is closer to:

```text
Decomposing the World Model: Representation vs Objective
```

Its core concern is methodological and diagnostic: whether a world model trained only for embedding reconstruction preserves decision-relevant ordering signals, and whether discriminative or ranking losses are needed to make representation learning useful for planning.

Paper13 is application-driven:

```text
How can a land-use prediction model and a DRL planner be coupled to obtain future-aware farmland layout decisions?
```

The relationship is complementary:

- Paper10 can provide theoretical or methodological justification for decision-aware world-model learning.
- Paper13 uses world-model prediction as a component in a land-planning decision system.

They should not be merged too early. Paper10 is an AI/model-learning paper; Paper13 is a remote-sensing, land-planning, and decision-support paper.

## Two Technical Levels

### Level 1: Passive Future-Aware Optimization

This is the first feasible version of Paper13.

The world model predicts future land status under a baseline trend. DRL then optimizes farmland layout using the predicted future state as the planning background.

```text
AlphaEarth embeddings + DEM + current land-use data
        ↓
Predict future land status or future suitability state
        ↓
Run DRL optimization on predicted future state
        ↓
Obtain future-aware planning layout
```

In this version, DRL actions do not change the predicted future trajectory. The prediction model is used as a future-state estimator, not as an intervention simulator.

This version is scientifically defensible because it answers:

```text
If current land-use trends continue, how should farmland spatial layout be optimized for a future target year?
```

Example target years:

```text
2028, 2030, 2035
```

### Level 2: Active Intervention-Aware Optimization

This is the stronger but much harder version.

Here, DRL actions are fed into the world model, and the world model predicts how those actions change future land states.

```text
DRL selects planning action
        ↓
World model predicts action-conditioned future land state
        ↓
DRL receives long-term reward
        ↓
Policy learns to optimize future outcomes
```

This becomes true model-based reinforcement learning or planning with a world model.

This version requires intervention-labeled data or credible counterfactual constraints. Without data showing how planning actions affect future land-use change, it is unsafe to claim that the model can predict arbitrary intervention outcomes.

Therefore, Level 2 should be treated as a later-stage extension unless suitable intervention data can be collected.

## Proposed Paper13 Pipeline

### Step 1: Build the Current State

For each parcel or management block:

```text
state_t = [
  planning features,
  DEM terrain features,
  AlphaEarth 64-dimensional embedding,
  land-use label,
  spatial adjacency and contiguity features
]
```

### Step 2: Predict Future Land State

Use an AlphaEarth-based world model or similar temporal prediction model to infer:

```text
future_embedding_{t+k}
future_land_use_{t+k}
future_suitability_{t+k}
future_risk_{t+k}
```

The prediction can be decoded into land-use classes or kept in embedding space for planning.

### Step 3: Construct Future-Aware DRL State

The DRL agent receives both current and predicted future information:

```text
DRL_state = [
  current planning features,
  current AlphaEarth embedding,
  predicted future embedding,
  predicted future land-use class,
  predicted future suitability score,
  predicted risk of farmland loss or degradation
]
```

### Step 4: Optimize Future-Aware Reward

The reward should consider the quality of the final planning layout under future conditions:

```text
reward =
  future_contiguity_improvement
+ future_high-suitability_farmland_preservation
- future_high-slope_or_low-suitability_farmland_penalty
- future_farmland_loss_risk
- conflict_with_predicted_urban_expansion
- fragmentation_under_future_state
+ robustness_across_future_scenarios
```

### Step 5: Compare Against Current-State Planning

The main empirical question:

```text
Does future-aware optimization produce better planning outcomes than optimization based only on the current state?
```

## Core Baselines

Paper13 should compare:

1. Current-state DRL

```text
Optimize using only current land-use and suitability state.
```

2. Future-state DRL

```text
Optimize using predicted future land-use and suitability state.
```

3. Current + future joint-state DRL

```text
Optimize using both current state and predicted future state.
```

4. Rule-based future-aware planning

```text
Use handcrafted rules based on predicted future land-use risk.
```

5. No-planning or persistence baseline

```text
Keep current layout unchanged.
```

## Evaluation Questions

Paper13 should evaluate:

- Does future-aware planning improve expected future farmland contiguity?
- Does it reduce the risk of allocating farmland to areas likely to be urbanized, degraded, or ecologically unsuitable?
- Does it preserve stable, high-suitability farmland better than current-state optimization?
- Does it improve robustness across different predicted future scenarios?
- Does it generalize across regions better when AlphaEarth embeddings are included?

## Key Scientific Contribution

Paper13 can be framed as:

```text
A future-aware farmland planning framework that couples remote-sensing foundation-model-based land-state prediction with reinforcement-learning-based spatial optimization.
```

Its novelty lies in coupling two components that are usually separate:

```text
land-use change prediction
farmland spatial layout optimization
```

Instead of predicting land-use change only for monitoring, the prediction becomes an input to spatial decision-making.

## Key Risks and Boundaries

The most important boundary is counterfactual validity.

If the model only learns historical baseline transitions, then it can support passive future-aware optimization, not arbitrary intervention simulation.

Safe wording:

```text
We use predicted baseline future land states to inform robust farmland layout optimization.
```

Risky wording to avoid unless intervention data exist:

```text
The model predicts how each planning action will causally change future land use.
```

Other risks:

- Future land-use prediction error may propagate into DRL optimization.
- DRL may overfit to biased future predictions.
- Predicted future states should be evaluated against held-out years before being trusted for planning.
- Multiple future scenarios should be used where possible to avoid single-trajectory planning.

## Suggested Positioning

Paper13 is more ambitious than Paper11.

Paper11:

```text
GeoFM-enhanced current-state farmland suitability and DRL optimization.
```

Paper13:

```text
GeoFM-based future land-state prediction + future-aware DRL optimization.
```

Paper10:

```text
World-model representation/objective diagnosis for decision relevance.
```

The three papers are connected but have distinct roles:

- Paper10 asks whether the learned representation is decision-relevant.
- Paper11 uses AlphaEarth to enrich current-state farmland suitability representation.
- Paper13 uses predicted future land states to optimize farmland layout for future conditions.

