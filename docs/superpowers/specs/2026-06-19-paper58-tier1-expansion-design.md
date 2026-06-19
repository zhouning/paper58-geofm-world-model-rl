# Paper58 Tier 1 Expansion Design

Date: 2026-06-19

## Goal

Build a strict next-phase experiment package for Paper58 that expands genuinely external Tier 1 evidence before any further manuscript strengthening or submission polishing. The only purpose of this phase is to determine whether the current Paper58 claim survives a stronger external holdout test.

## Starting Facts

The current local benchmark already established three hard facts:

- The benchmark gate failed.
- Only one evaluated row currently counts as Tier 1 in the benchmark outputs: `poyang_lake 2020-2021`.
- The only other nominal Tier 1 row, `wuyi_mountain 2020-2021`, is a zero-change negative control and cannot support positive change evidence.

There is also a stricter provenance problem:

- `poyang_lake` and `wuyi_mountain` both appear in `src/adk_world_model/world_model.py` under `DEFAULT_TRAINING_AREAS`.
- That means they cannot automatically be treated as strict external holdouts unless a provenance audit proves they were not used for model fitting, decoder training, hyperparameter selection, or benchmark design decisions that matter for Paper58.

Local file discovery found no hidden reserve of already-generated external pairs that would solve this problem without new data acquisition.

## Scientific Position

The narrow claim under test remains:

> Frozen GeoFM embeddings can support baseline-trend land-surface change-signal modelling under independent external evaluation against strong non-neural controls.

This phase is not allowed to claim:

- operational land-cover forecasting,
- broad cross-region superiority,
- policy or planning deployment validity,
- counterfactual intervention simulation.

If the expanded Tier 1 benchmark still fails, the paper must not advance under a stronger narrative.

## Scope

This design covers only the Tier 1 evidence expansion subproject.

Included:

- provenance-first external holdout definition,
- external area manifest,
- new label and embedding acquisition for new holdout areas,
- prediction generation for those holdouts,
- benchmark registry and gate rerun,
- stop/go decision for the manuscript.

Excluded:

- optimistic manuscript rewriting,
- new planning experiments,
- intervention-aware simulations,
- architectural model changes intended to rescue the benchmark after seeing the new holdout results.

## Why The Next Step Must Be New Holdouts

Three plausible routes exist:

1. Reuse only local cached assets and try to reframe them as stronger evidence.
2. Acquire new external holdout areas and rerun the benchmark under stricter provenance control.
3. Stop experimentation and move directly to a downgraded negative-results manuscript path.

The selected route is 2.

Route 1 is scientifically too weak because the local audit found only one positive external pair and because the current nominal Tier 1 areas may not be true holdouts. Route 3 remains a valid fallback, but it violates the user's stated principle unless route 2 has been tried and still fails.

## Tier 1 Eligibility Rules

An area-year pair can count as Tier 1 only if all conditions below are satisfied:

1. The area is not in `DEFAULT_TRAINING_AREAS`.
2. The area is not one of the development areas already used in Paper58 workflow construction: `bishan`, `banzhucun`, `heping`.
3. The area was not used for decoder training, dynamics training, hyperparameter tuning, baseline design, figure selection, or error-driven benchmark editing.
4. The area has explicit provenance metadata recorded in a holdout manifest.
5. Start-year labels, end-year labels, predictions, embeddings, and context all exist and align.
6. The reference labels contain non-zero change if the pair is to support the positive gate.

Any pair failing provenance certainty must not be counted as Tier 1. It can be recorded as `tier2` or `review_required`, but not used for the main gate.

## Evidence Target

The expansion phase should not stop after adding one or two more pairs. The minimum target is:

- at least 4 independent Tier 1 region clusters,
- at least 3 Tier 1 strata represented by positive-change evaluated rows,
- at least 1 positive-change evaluated pair per retained Tier 1 region cluster,
- at least 1 reserve holdout pair beyond the minimum in case one candidate becomes a zero-change control or QC exclusion.

As a working target, the first acquisition batch should aim for 6 to 8 candidate Tier 1 pairs across 4 to 6 regions.

## Strata Strategy

The first expansion batch should deliberately span multiple land-surface regimes instead of adding near-duplicates of the same regime.

Priority coverage:

- Urban
- Agriculture
- Forest or Plateau
- Wetland or Mixed

The point is not balance for its own sake. The point is to test whether any positive result survives heterogeneity instead of depending on a single wetland pair.

## External Holdout Manifest

Create a new source-of-truth manifest for candidate Tier 1 areas. It must be machine-readable and versioned.

Required fields per area:

- `area`
- `bbox`
- `stratum`
- `years`
- `data_source`
- `selection_reason`
- `development_contact_status`
- `contact_evidence`
- `expected_role`
- `notes`

Required `development_contact_status` values:

- `none`
- `known_contact`
- `uncertain`

Only `none` can map to Tier 1.

## Data Sources And Reuse

This phase should reuse the existing Paper58 acquisition and prediction stack rather than introducing a parallel pipeline.

Reuse these scripts:

- `scripts/rse_revision/fetch_independent_lulc_labels.py`
- `scripts/rse_revision/fetch_change_validation_embeddings.py`
- `scripts/rse_revision/generate_change_validation_predictions.py`
- `scripts/paper58_benchmark/build_registry.py`
- `scripts/paper58_benchmark/evaluate_benchmark.py`

The current scripts already cover the right operations. The problem is not missing machinery. The problem is that the area definitions and Tier assignment are too permissive for strict external evidence.

## Required Code Changes

The next implementation phase should make the smallest changes that close the scientific gap:

1. Add a holdout area manifest consumed by the acquisition scripts.
2. Decouple candidate external areas from `DEFAULT_TRAINING_AREAS`.
3. Extend benchmark schema and registry rows with provenance metadata:
   - `bbox`
   - `data_source`
   - `development_contact_status`
   - `contact_evidence`
   - `expected_role`
4. Change tier assignment so that Tier 1 vs Tier 2 comes from manifest provenance, not just area name heuristics.
5. Add explicit handling for `review_required` or equivalent provenance uncertainty.
6. Preserve negative controls and zero-change areas in the registry without letting them inflate positive evidence.

## Execution Phases

### Phase 1: Provenance Audit

Audit all currently benchmarked areas and classify them as:

- strict external holdout,
- development contact,
- training contact,
- uncertain provenance.

This phase decides whether `poyang_lake` and `wuyi_mountain` remain eligible for Tier 1. The conservative default is that they do not.

### Phase 2: Holdout Selection

Populate the holdout manifest with a first batch of new candidate areas. The batch should satisfy the target stratum coverage and reserve capacity.

### Phase 3: Data Acquisition

For each selected holdout:

- fetch ESRI annual labels for the requested years,
- fetch AlphaEarth annual embeddings,
- fetch terrain context aligned to the embedding grid.

### Phase 4: Prediction Generation

Run the existing baseline-scenario world-model prediction path to generate end-year LULC maps for the new holdouts.

### Phase 5: Registry And Gate Rerun

Rebuild the benchmark registry, recompute metrics, and rerun the gate.

### Phase 6: Decision

Only after the rerun may the project decide between:

- manuscript strengthening,
- claim downgrading,
- negative-results framing,
- pause.

## Quality Control Additions

The expanded benchmark must add provenance QC to the existing array-shape QC.

New required QC checks:

- reject Tier 1 assignment when `development_contact_status != none`,
- reject Tier 1 assignment when `contact_evidence` is absent,
- reject Tier 1 assignment when the area appears in known Paper58 training or development registries,
- record zero-change areas as negative controls,
- record acquisition failures explicitly rather than silently dropping areas.

## Testing

Implementation must add or update tests that prove:

- holdout manifests are parsed correctly,
- acquisition scripts read candidate areas from the manifest,
- benchmark tier assignment follows provenance metadata,
- uncertain-contact rows cannot pass into Tier 1,
- zero-change rows remain visible as negative controls,
- gate summaries use only provenance-cleared Tier 1 rows.

## Stop Conditions

This phase should stop and return a negative or paused decision if any of the following occurs:

- fewer than 4 provenance-cleared Tier 1 region clusters can be assembled,
- fewer than 3 positive-change Tier 1 strata remain after QC,
- GEE or data availability prevents assembling a minimally credible holdout set,
- the expanded gate still fails.

If any stop condition is met, the project should not keep broadening the sample opportunistically until a positive story appears.

## Manuscript Gate

Before this phase passes, manuscript work remains limited to:

- claim downgrading,
- limitations,
- data and reference cleanup,
- transparent recording of failed or insufficient external evidence.

Only after a successful expanded Tier 1 gate may the RSE manuscript move back toward stronger claims.

## Risks

- Newly acquired holdouts may show weaker results than the current local benchmark.
- Some areas may end up as zero-change controls, reducing usable Tier 1 count.
- Provenance audit may force currently nominal Tier 1 rows out of Tier 1 entirely.
- GEE access, quota, or extraction failures may block part of the acquisition batch.

These are acceptable risks. The purpose of this phase is to learn whether the claim survives external scrutiny, not to preserve the current narrative.

## Approved Direction

The approved direction is:

> Build a provenance-first Tier 1 expansion workflow, acquire a new batch of genuinely external holdout areas, rerun the benchmark gate, and let that result decide whether Paper58 deserves stronger submission work.
