# Paper13 Future-Aware Farmland Planning

This folder separates the Paper13 research design from the Paper58 world-model repository.

Paper13 is not a direct continuation of Paper58 experiments. It uses Paper58's AlphaEarth latent land-state prediction as one component in a downstream planning framework:

```text
current planning state
  -> baseline future land-state prediction
  -> future-aware farmland spatial optimization
  -> evaluation under held-out or simulated future conditions
```

## Core Position

The defensible first version of Paper13 is passive future-aware optimization:

```text
If the baseline land-use trajectory continues, which farmland layout is better for the target future year?
```

It should not claim arbitrary action-conditioned causal simulation unless intervention-labeled data are added.

## Files

- `01_design_synthesis.md`: deep synthesis of the Paper13 idea, including its relationship to Paper11, Paper10, and Paper58.
- `02_experiment_design.md`: proposed datasets, baselines, metrics, ablations, robustness tests, and expected tables.
- `03_implementation_roadmap.md`: staged execution plan for turning the design into code and experiments.
- `04_manuscript_blueprint.md`: paper narrative, section outline, contribution claims, and safe wording.

## Source Material Read

The design here is derived from:

- `docs/paper13_design_thought.md`
- `docs/paper11_design_thought.md`
- `docs/world_model_paper_cn.md`
- `docs/paper_review_epi_world_model_vs_alphaearth.md`
- `docs/leworldmodel_analysis_for_adk.md`
- `experiments/paper8/embedding_space_env.py`
- `experiments/paper8/dual_rep_env.py`
- `experiments/paper8/intervention_env.py`
- `experiments/paper8/intervention_dynamics.py`
- `src/adk_world_model/world_model.py`

