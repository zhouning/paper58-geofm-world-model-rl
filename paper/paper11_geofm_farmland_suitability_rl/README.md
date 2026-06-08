# Paper11: GeoFM-Enhanced Farmland Suitability RL

This folder is an independent analysis and design package for Paper11.

Paper11's core idea is to use AlphaEarth or other GeoFM embeddings as a latent environmental semantic channel for farmland suitability representation, while retaining explicit GIS, DEM, adjacency, contiguity, and planning constraints in the reinforcement-learning environment.

It is distinct from Paper13. Paper11 optimizes farmland layout under the current planning state. Paper13 adds future land-state prediction before optimization.

## Files

- `01_design_synthesis.md`: distilled understanding of the Paper11 design thought from `docs/paper11_design_thought.md`.
- `02_system_design.md`: proposed state, reward, policy, and data-flow design for a realizable Paper11 system.
- `03_experiment_plan.md`: baseline matrix, ablations, transfer tests, metrics, and expected evidence.
- `04_manuscript_outline.md`: paper-level argument, contribution framing, figure plan, and section outline.
- `05_risks_and_boundaries.md`: claim boundaries, failure modes, and next-step decisions.

## Central Claim

AlphaEarth 64-dimensional embeddings should not be interpreted as direct measurements of soil, irrigation, or fertility. They should be treated as latent remote-sensing proxies for land-surface and environmental conditions related to farmland suitability. Paper11 tests whether this proxy improves both planning decision quality and cross-region generalization in DRL-based farmland spatial layout optimization.

