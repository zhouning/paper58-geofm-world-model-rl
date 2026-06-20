# Paper58 Batch 5 Mixed Gate Design

## Purpose

Batch 5 is a mixed external Tier 1 experiment designed to restore the full benchmark gate's cross-stratum logic while preserving a small targeted probe of the Batch 2/Batch 4 urban class-11 weakness.

Batch 5 is not a manuscript-strengthening shortcut. It is the next experiment-first readout after:

- Batch 2 failed the spatial gate,
- Batch 3 independently passed the full gate,
- Batch 4 passed primary and spatial confidence checks but failed the full gate because it was intentionally all Urban.

## Design

Batch 5 should contain `6-8` new 2020-2021 strict Tier 1 candidates with no area-name overlap with Batch 1, Batch 2, Batch 3, or Batch 4.

Target composition:

- `2` `xiong_an_like` Urban stress candidates focused on class-11 urban-fringe transitions.
- `1` `suzhou_like` Urban localization candidate focused on dense-edge spatial robustness.
- `2` Wetland or Agriculture candidates to restore non-Urban gate support.
- `1-2` Forest or Grassland candidates to add a third positive Tier 1 stratum and reduce dependence on any single landscape type.

The batch should cover at least `3` strata by manifest design, and it should keep enough reserve diversity that one QC exclusion does not collapse the full gate to fewer than `3` evaluated strata.

## Selection Priorities

Urban rows should be guided by Batch 4 diagnostics:

- Use `xiongxian_river_corridor_holdout` as the closest observed class-11 representation-bottleneck analogue.
- Use `baiyangdian_new_area_holdout` as a possible observed-versus-forecast separation analogue.
- Do not over-sample weak low-change rows like `xinxiang_floodplain_newtown_holdout`; low true-change counts can make interpretation noisy.

Non-Urban rows should favor candidate landscapes that previously produced stable positive evidence in independent batches:

- Agriculture is supported in Batch 2 and Batch 3.
- Wetland is supported in Batch 3, while Batch 2 wetland was positive but modest.
- Forest and Grassland are useful third-stratum candidates, but single-row strata should be treated cautiously.

## Stop/Go Rule

Batch 5-only is the next full-gate readout after manifest design, acquisition, prediction, registry build, provenance audit, and benchmark evaluation.

Interpretation rules:

- If Batch 5 fails, keep the failure visible and diagnose whether it comes from primary, spatial, QC, or stratum coverage.
- If Batch 5 passes, compare it against Batch 2, Batch 3, and Batch 4 before changing manuscript claims.
- Do not pool Batch 5 with prior batches to hide a Batch 5-only failure.

## Non-Goals

- Do not change benchmark thresholds.
- Do not change the dynamics model.
- Do not edit the manuscript.
- Do not add another all-Urban batch for the full-gate readout.
- Do not treat a full-gate pass as sufficient for manuscript strengthening until the contradictory Batch 2 and diagnostic Batch 4 evidence are explicitly retained.
