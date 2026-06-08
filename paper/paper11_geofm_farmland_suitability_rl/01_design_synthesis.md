# Paper11 Design Synthesis

## 1. Source Understanding

The source design note is `docs/paper11_design_thought.md`. Its central insight is that the earlier farmland-layout DRL papers use explicit GIS and planning variables, but represent farmland suitability through a narrow observable feature set: land-use type, slope, parcel or block geometry, adjacency, contiguity, fragmentation, and related block statistics.

That design is operational and interpretable, but it misses many suitability drivers that are difficult to obtain at planning scale, including irrigation convenience, soil condition, water or wetness regime, crop-growth stability, surrounding land-use pressure, and broader ecological background.

Paper11 starts from this gap:

```text
Can GeoFM embeddings enrich the environmental representation used by farmland-layout DRL when explicit irrigation, soil, and productivity data are missing?
```

The answer should be tested, not assumed.

## 2. Boundary Against Other Papers

Paper11 should be kept separate from the world-model and future-prediction ideas.

```text
Paper11:
current suitability representation -> DRL spatial layout optimization

Paper13:
current suitability representation -> future land-state prediction -> future-aware DRL optimization

Paper10:
diagnosis of representation/objective alignment and decision relevance
```

Therefore, Paper11 should not claim to predict future land-use states, train a temporal world model, or solve counterfactual scenario simulation. It should ask whether a frozen foundation-model representation improves current-state suitability cognition and decision policy transfer.

## 3. Core Design Thought

The design is not simply:

```text
add 64 more features to the state
```

The design is:

```text
explicit planning constraints
+ latent environmental semantics from GeoFM
+ suitability-aware reward and evaluation
-> more realistic and transferable farmland layout optimization
```

The novelty depends on showing that GeoFM embeddings carry decision-useful environmental signals that are missing from the original hand-engineered state, while also proving that hard planning constraints remain necessary.

## 4. Interpretation of AlphaEarth

AlphaEarth can plausibly provide proxy information about:

- land-cover and land-use semantics;
- vegetation vigor and seasonal crop-growth signals;
- water, wetness, and irrigation-adjacent surface signals;
- urban-edge pressure and construction disturbance;
- landscape texture and surrounding land-use context;
- broader environmental background learned from multi-sensor observations.

AlphaEarth should not be claimed to directly provide:

- measured soil organic matter;
- soil pH or nutrient content;
- irrigation canal infrastructure;
- legal farmland protection status;
- exact slope or elevation constraints.

This distinction is the scientific safety line of Paper11.

## 5. Practical Design Implication

The right system design is a conservative extension of the existing block-level and county-level MDP:

```text
old block state:
  17 explicit block features + 12 global features

paper11 block state:
  17 explicit block features
  + 64-dimensional block-level GeoFM embedding
  + optional low-dimensional suitability score
  + 12 global features
```

The current `ParcelScoringPolicy` pattern is useful because it scores each block independently with a shared network. This makes the policy dimension-agnostic over the number of blocks and supports cross-region transfer. Paper11 should keep this policy pattern and expand the per-block feature vector rather than replacing the action interface.

## 6. One-Sentence Paper Argument

In farmland spatial layout optimization, Paper11 shows that frozen GeoFM embeddings can act as latent environmental suitability proxies for DRL policies, supported by state/reward ablations and cross-region transfer tests, with claims bounded to remote-sensing proxy signals rather than direct soil or irrigation measurement.

