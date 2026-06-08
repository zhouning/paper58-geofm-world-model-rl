# Paper13 Implementation Roadmap

## Goal

Build a reproducible Paper13 pipeline that turns current parcel/block data and AlphaEarth predictions into future-aware farmland planning experiments.

Recommended folder for code if implemented later:

```text
experiments/paper13/
```

This roadmap is a design artifact only. It does not yet modify the runtime code.

## Stage 0: Freeze the Claim Boundary

Decision:

```text
Paper13 v1 = passive future-aware optimization
```

Do not implement or report action-conditioned world-model planning as the main result unless intervention-labeled data are introduced.

Deliverable:

- A one-page claim boundary note in the manuscript or supplement.

## Stage 1: Build Parcel/Block Future-State Dataset

Inputs:

- Current parcel/block table.
- AlphaEarth pixel embeddings for available years.
- Parcel/block geometry.
- DEM and slope.
- Current and target-year land-use labels if available.

Processing:

1. Aggregate pixel embeddings to block-level mean embedding:

```text
block_z_t = mean(pixel_z_t within block)
```

2. Aggregate predicted future embeddings:

```text
block_z_pred_{t+k} = mean(pixel_z_pred_{t+k} within block)
```

3. Build delta features:

```text
block_delta_z = block_z_pred_{t+k} - block_z_t
```

4. Decode or score future planning variables:

```text
future_lulc
future_farmland_stability
future_construction_risk
future_suitability
future_uncertainty
```

Expected files:

```text
experiments/paper13/data/block_current.parquet
experiments/paper13/data/block_future_pred.parquet
experiments/paper13/data/block_future_eval.parquet
experiments/paper13/data/adjacency.npz
```

## Stage 2: Prediction Validation

Before DRL, validate whether the future layer is meaningful.

Minimum scripts:

```text
experiments/paper13/validate_future_prediction.py
experiments/paper13/summarize_prediction_metrics.py
```

Metrics:

- Cosine similarity against observed future embedding.
- Improvement over persistence.
- Changed-block accuracy.
- Farmland-loss or construction-risk AUC.

Exit condition:

Proceed to DRL only if the future predictor provides non-trivial changed-block signal.

## Stage 3: Implement Future-Aware Environment

Recommended environment:

```text
FutureAwareFarmlandEnv
```

State modes:

```text
current
future
current_future
```

Observation layout:

```text
per_block = [
  current_features,
  current_embedding,
  predicted_future_embedding,
  delta_embedding,
  decoded_future_risk,
  decoded_future_suitability
]
global = [
  budget,
  step_fraction,
  target_horizon,
  region_summary_features
]
```

Action:

```text
select block or parcel to convert / preserve / optimize
```

The exact action semantics should match the existing farmland layout optimization code, so Paper13 remains comparable with earlier papers.

Hard masks:

- Non-candidate blocks cannot be selected.
- Protected or legally forbidden conversions cannot be selected.
- Already processed blocks can be masked if the action is single-use.

Reward modes:

```text
current_reward
future_reward
joint_reward
```

## Stage 4: Train Baselines

Methods:

1. Persistence / no planning.
2. Rule-based current planning.
3. Rule-based future-aware planning.
4. Current-state DRL.
5. Future-state DRL.
6. Current+future joint-state DRL.

Seeds:

Use at least 10 seeds for DRL if runtime is manageable; 15 seeds matches the style of existing Paper8 results.

Outputs:

```text
experiments/paper13/results/<method>/eval_seed*.json
experiments/paper13/results/summary.csv
experiments/paper13/results/statistical_tests.json
```

## Stage 5: Evaluate Target-Year Outcomes

Evaluation must use a target-year layer distinct from the policy's current-state input.

Preferred:

```text
observed target-year land-use / embedding
```

Fallback:

```text
held-out prediction target or scenario simulation
```

For each trained policy, compute:

- Future reward.
- Contiguity.
- Fragmentation.
- Slope exposure.
- Future construction conflict.
- Stable farmland preservation.
- High-risk farmland allocation.

## Stage 6: Robustness and Ablation

Ablations:

- Remove predicted future embedding.
- Remove decoded future risk.
- Remove delta features.
- Remove current features and use future-only.
- Remove AlphaEarth and use explicit GIS features only.
- Replace world-model prediction with persistence.

Robustness:

- Vary horizon.
- Add noise to future prediction.
- Use ensemble or Monte Carlo future states if available.
- Cross-region transfer.

## Stage 7: Manuscript Assets

Figures to generate:

```text
fig_pipeline.pdf
fig_future_risk_map.pdf
fig_plan_comparison_map.pdf
fig_metric_bars.pdf
fig_horizon_sensitivity.pdf
fig_transfer.pdf
```

Tables:

```text
table_prediction_validation.csv
table_planning_metrics.csv
table_transfer.csv
table_ablation.csv
```

## Minimal Feasible Version

If time is limited, implement only this:

1. One region with current and target-year land-use labels.
2. Block-level AlphaEarth current and predicted future embeddings.
3. Three methods:

```text
current-only DRL
current+future DRL
rule-based future-aware baseline
```

4. Four outcome metrics:

```text
future contiguity
future construction/risk conflict
future stable farmland preservation
average slope
```

This is enough to test the central Paper13 hypothesis without overbuilding the whole platform.

## Implementation Cautions

- Keep hard planning constraints explicit; do not hide them in embeddings.
- Normalize embedding features consistently.
- Do not evaluate against the same predicted future used to optimize unless clearly labeled as simulation.
- Keep `z_t`, `z_pred`, and `delta_z` separate in logs for interpretation.
- Store action sequences so maps can be reconstructed.
- Report uncertainty and prediction error propagation as limitations, not as afterthoughts.

