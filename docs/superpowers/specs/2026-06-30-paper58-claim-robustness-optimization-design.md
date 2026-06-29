# Paper58 Claim-Robustness Optimization Design

## Goal

Improve the evidence strength for the current Paper58 vs. GeoSOS-FLUS claim
without changing the base latent-dynamics model. The first optimization phase
targets manuscript robustness: keep the existing 24-township aggregate advantage
over the fixed `geosos_flus_console` baseline while improving area-level paired
wins, especially FoM.

The implementation should remain a deterministic allocator/post-processing
extension around the current best Paper58 path:

```text
paper58_spatial_demand_ratio_external_loo_transition_floor030
```

## Current Evidence Boundary

The current best Paper58 variant beats the fixed GeoSOS-FLUS console baseline on
24-township mean metrics:

- change F1: `0.2883` vs. `0.2635`;
- FoM: `0.1248` vs. `0.1196`;
- transition accuracy: `0.2862` vs. `0.2808`;
- allocation disagreement: `0.0584` vs. `0.0601`, where lower is better.

Seeded mean evidence is strong for the first three metrics:

- change F1: `5/5` seed wins;
- FoM: `5/5` seed wins;
- transition accuracy: `5/5` seed wins;
- allocation disagreement: `3/5` seed wins.

The remaining weakness is local robustness across area-by-seed pairs:

- change F1: `65/120` wins;
- FoM: `61/120` wins;
- transition accuracy: `54/120` wins;
- allocation disagreement: `81/120` wins.

This phase should support a stronger version of the existing safe claim, not a
new claim that Paper58 wins every region or fully replaces native GeoSOS-FLUS.

## Claim Ladder

This work keeps three claim levels separate:

1. **Current supported claim**: the transition-aware Paper58 spatial-demand ratio
   variant surpasses the fixed GeoSOS-FLUS console baseline in aggregate mean
   metrics under strict non-target tuning on the 24-township same-grid target set.
2. **Phase-A target claim**: the same aggregate advantage is preserved and the
   area-level FoM paired-win evidence improves materially under the same
   non-target tuning rule.
3. **Future claims**: Paper58 performance against more native GeoSOS-FLUS GUI or
   traditional-driver workflows. Those are out of scope for this phase.

## Scope

In scope:

- add an optional allocator-v2 mode beside the existing `transition_floor030`
  path;
- learn transition-pair reliability only from external calibration areas;
- add low-change safeguards to reduce over-allocation in low-change targets;
- add multi-scale source and target neighborhood support to reduce spatial
  spread;
- run the same 24-township fixed-FLUS benchmark and the same five seed
  replicates;
- write explicit winner, loser, and failure tables for manuscript audit.

Out of scope:

- retraining `LatentDynamicsNet`;
- tuning on target township end-year labels;
- changing the fixed GeoSOS-FLUS console execution;
- overwriting previous reports or changing published historical summaries;
- claiming full replacement of native GeoSOS-FLUS.

## Architecture

The optimized path is a strict extension of the current allocator:

```text
start LULC
+ Paper58 source prediction
+ Paper58 change score
+ external calibration rows
-> transition reliability model
-> adaptive low-change cap
-> multi-scale spatial support maps
-> v2 candidate ranking
-> demand-limited LULC output
-> fixed-FLUS comparison report
```

The existing CLI defaults must remain unchanged. V2 behavior is enabled only
through explicit arguments so the previous `transition_floor030` result is still
reproducible.

## Components

### Transition Reliability

Estimate reliability for each predicted transition `(from_class, to_class)` from
external calibration rows:

```text
raw_precision = exact_transition_hits / predicted_transition_pixels
smoothed_precision = (exact_transition_hits + smoothing) /
                     (predicted_transition_pixels + 2 * smoothing)
weight = min_weight + (1 - min_weight) * smoothed_precision
```

The reliability model must report support, exact hits, raw precision, smoothed
precision, and final weight for every transition pair used in target rows.

### Low-Change Safeguard

Some target rows have high candidate change fractions but low true change. V2
adds a calibration-derived cap for this pattern:

```text
candidate_fraction = mean(source_prediction != start)
ratio_fraction = current_ratio_model(candidate_fraction)
cap_fraction = calibrated_low_change_cap(candidate_fraction)
target_fraction = min(ratio_fraction, cap_fraction)
```

The cap must be learned from external rows only. It should activate only for
high candidate fractions so high-change rows are not globally suppressed.

### Multi-Scale Spatial Support

Compute source and target class-neighborhood support at multiple window sizes,
initially `3`, `5`, and `9`. The v2 candidate score combines:

```text
score_v2 =
    base_score
  + target_weight * target_support_multi_scale
  - source_penalty * source_support_multi_scale
  + transition_weight_strength * transition_reliability
```

The scoring should affect candidate changes only. Stable pixels remain stable
unless selected by the existing demand-limited gate.

### Reporting

Each run must write:

- metric summary by method;
- metrics by method and area;
- seed-level win table;
- area-by-seed paired-win table;
- failure table for rows where Paper58 loses FoM or has large allocation
  degradation;
- run manifest with all v2 parameters and calibration-row provenance.

## Acceptance Gates

The phase is successful only if all required gates pass:

- 24-township mean metrics remain `4/4` better than fixed `geosos_flus_console`;
- change F1, FoM, and transition accuracy remain `5/5` seed wins;
- allocation disagreement remains at least `3/5` seed wins;
- FoM area-by-seed paired wins improve from `61/120` to at least `66/120`;
- no new report hides area-level failures behind pooled mean metrics;
- all v2 parameters are selected without target township end-year truth.

Stretch goal:

- FoM paired wins reach `70/120` or better without reducing allocation
  disagreement wins below `81/120`.

If the required gates fail, keep the old `transition_floor030` path as the
manuscript-safe candidate and report v2 as a negative or diagnostic result.

## Testing

Focused tests should cover:

- transition-reliability smoothing and fallback weights for rare transitions;
- low-change cap behavior on high-candidate, low-target calibration examples;
- multi-scale support shape, determinism, and source/target contrast;
- v2 candidate ranking when reliability and spatial support disagree;
- default CLI behavior matching the current non-v2 path.

Evaluation verification should include:

```bash
python -m pytest tests/test_paper58_spatial_demand_allocation.py tests/test_paper58_spatial_demand_ratio_tuning.py -q
```

Then run the 24-township fixed-FLUS comparison and five seeded replicate
comparison using a new output directory. Do not overwrite existing reports.

## Risks And Controls

Transition reliability can suppress rare but real transitions. Use smoothing,
minimum weights, and diagnostics before accepting a v2 run.

Low-change caps can under-allocate genuine high-change rows. Activate the cap
only for high candidate fractions and report capped rows explicitly.

Multi-scale support can increase persistence and reduce true novel changes.
Track transition accuracy and FoM together so apparent spatial smoothing does not
hide class-transition losses.

Selector overfitting is the main scientific risk. The phase must use only
external calibration rows for parameter selection and must keep target-row truth
strictly evaluation-only.

## Future Phases

After this phase:

1. **Failure-mode optimization**: address specific weak regions and transition
   pairs such as low-change over-allocation and weak `5->2`, `5->11`, `2->11`,
   and `7->2` behavior.
2. **Stronger baseline validation**: compare against a more native GeoSOS-FLUS
   GUI or traditional-driver workflow when suitable inputs are available.
3. **Training-level changes**: consider allocation-aware loss or transition
   calibration only after allocator-level robustness stops improving.
