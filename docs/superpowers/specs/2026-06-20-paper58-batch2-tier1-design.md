# Paper58 Batch 2 External Tier 1 Stability Check Design

Date: 2026-06-20

## Goal

Run a second, independently recorded external Tier 1 holdout batch for Paper58 before doing any manuscript strengthening. The purpose is to test whether the Batch 1 expanded-gate pass is reproducible under new external areas rather than only preserved by pooling.

## User Principle

The governing principle is:

> First make the experiments strong and sufficient, then consider the paper. There is no submission urgency.

This design therefore excludes manuscript rewriting. All work in this phase must create, evaluate, or document additional experimental evidence.

## Starting Facts

The first strict Tier 1 expansion completed successfully on branch `paper58-benchmark`.

Current committed results show:

- `benchmark_registry.json`: 20 candidate rows, 18 evaluated rows.
- `benchmark_provenance_audit.json`: 20 rows, 8 Tier 1 rows, 12 Tier 2 rows, 0 invalid Tier 1 rows.
- `benchmark_gate_report.json`: `status = pass`.
- Evaluated strict Tier 1 evidence: 7 rows across 7 region clusters.
- Positive Tier 1 strata: 4, covering Agriculture, Forest, Urban, and Wetland.
- Tier 1 primary change advantage CI low: 0.10190708381580636.
- Tier 1 spatial change advantage CI low: 0.04810621853180953.
- `haibei_plateau_holdout` is visible as Tier 1 provenance but excluded by QC with `class_collapse`.
- `poyang_lake` and `wuyi_mountain` remain Tier 2 because of known training contact.

These results are strong enough to justify more experimentation, but not strong enough to stop. Batch 2 must be treated as an independent stability check.

## Core Question

The experiment asks:

> Does a second batch of genuinely external Tier 1 holdouts independently support the same bounded claim that frozen GeoFM embeddings can provide a low-cost baseline-trend land-surface change-signal state space?

The experiment must not be used to claim:

- operational categorical LULC forecasting,
- validated counterfactual scenario simulation,
- policy or planning deployment validity,
- universal superiority across all regions or land-cover regimes.

## Decision Rule

Batch 2 must be evaluated in two views:

1. `Batch 2 only`: the second batch alone.
2. `Batch 1 + Batch 2 combined`: pooled strict Tier 1 evidence after Batch 2 is added.

The Batch 2 only result is the primary stability check. The combined result is secondary.

Interpretation rules:

- If Batch 2 only passes or remains clearly positive and the combined result also passes, the external evidence is stronger.
- If Batch 2 only fails, remains insufficient, or is substantially weaker while the combined result passes only because Batch 1 dominates, the evidence is not strengthened.
- If Batch 2 only is insufficient because too many candidates fail acquisition or QC, the correct response is more experimental design, not manuscript strengthening.
- Failed, zero-change, and excluded rows must remain visible in manifests, registries, audits, and reports.

## Batch 2 Composition

Batch 2 should add 8 candidate external holdout areas. The target is to retain at least 6 evaluable strict Tier 1 rows after acquisition and QC, with a hard minimum of 4.

Recommended composition:

- Urban: 2 candidates, one coastal or peri-urban and one inland urban fringe.
- Agriculture: 2 candidates, one dryland or oasis agriculture and one plains agriculture.
- Forest: 1 or 2 candidates outside the Batch 1 northeast and Qinling context where possible.
- Wetland or water-margin: 1 candidate outside Poyang and Dongting hydrologic contexts.
- Mixed, plateau, or grassland: 1 or 2 candidates to test ecological heterogeneity and class-collapse risk.

All candidates must use the same first comparison window, 2020-2021, unless a separate approved temporal protocol explicitly changes that window.

Each bounding box should remain approximately 0.1 by 0.1 degrees. The goal is comparability with Batch 1 and low extraction cost, not exhaustive regional mapping.

## Candidate Eligibility

Each Batch 2 candidate must satisfy all Tier 1 provenance rules:

1. The area name and bounding box are absent from `DEFAULT_TRAINING_AREAS`.
2. The area is absent from known development-contact areas: `bishan`, `banzhucun`, and `heping`.
3. The area does not overlap or closely neighbor Batch 1 candidate boxes.
4. The area has a clear `selection_reason`.
5. The manifest records `development_contact_status = none`.
6. The `contact_evidence` field states why no training or development contact is known.
7. The area is not selected after inspecting model predictions.

If any candidate cannot satisfy these conditions, it must not be counted as Tier 1.

## File Boundaries

Batch 2 must not overwrite the current Batch 1 result directory.

Source manifests:

- Existing Batch 1 and provenance manifest:
  - `data/independent_change_labels/paper58_holdout_areas.json`
- New Batch 2 manifest:
  - `data/independent_change_labels/paper58_holdout_areas_batch2.json`
- Generated combined manifest:
  - `data/independent_change_labels/paper58_holdout_areas_combined.json`

Benchmark output directories:

- Existing Batch 1 expanded result:
  - `paper/rse_submission_paper58/benchmark_results/`
- Batch 2 only result:
  - `paper/rse_submission_paper58/benchmark_results_batch2/`
- Combined Batch 1 + Batch 2 result:
  - `paper/rse_submission_paper58/benchmark_results_combined/`

Acquisition and prediction arrays may remain under the existing shared directories because file names are area-specific:

- `data/independent_change_labels/labels/`
- `data/independent_change_labels/embeddings/`
- `data/independent_change_labels/predicted/`

The acquisition and prediction manifests may be updated in place, but they must preserve explicit records for all fetched or failed Batch 2 areas.

## Minimal Implementation Surface

Do not rewrite the benchmark schema, tier logic, model, or gate definition.

Allowed implementation work:

- Add the Batch 2 manifest.
- Add a small manifest-combining utility that reads two validated holdout manifests, rejects duplicate areas, and writes a combined manifest.
- Reuse existing acquisition scripts with `--area-manifest`.
- Reuse existing prediction generation with `--area-manifest`.
- Reuse existing benchmark scripts with explicit `--output-dir` values.
- Add only minimal tests needed to prove manifest combination and output isolation.

Avoid:

- changing model weights,
- changing prediction logic,
- changing gate thresholds,
- changing tier assignment rules,
- editing the RSE manuscript.

## Evaluation Protocol

### Batch 2 Only

Run the full strict benchmark on `paper58_holdout_areas_batch2.json` and write outputs to `benchmark_results_batch2/`.

The Batch 2 only report must include:

- registry row counts,
- number of evaluated Tier 1 rows,
- number of Tier 1 region clusters,
- positive Tier 1 strata,
- primary change advantage bootstrap interval,
- spatial change advantage bootstrap interval,
- invalid Tier 1 provenance rows,
- failures and QC exclusions.

### Combined

Generate `paper58_holdout_areas_combined.json` from Batch 1 plus Batch 2, then rerun the benchmark to `benchmark_results_combined/`.

The combined result must be interpreted only after Batch 2 only has been examined.

## Pass And Stop Criteria

Batch 2 only is considered supportive only if all conditions below hold:

- `n_evaluated_tier1 >= 4`.
- `n_tier1_clusters >= 4`.
- `positive_tier1_strata >= 3`.
- `tier1_primary_change.ci_low > 0`.
- `tier1_spatial_change.ci_low > 0`.
- `invalid_tier1_rows = []`.

The combined result is supportive only if:

- gate status remains `pass`,
- strict Tier 1 clusters increase relative to Batch 1,
- positive Tier 1 strata do not decrease,
- Batch 2 only is not a failure hidden by pooled results.

Stop and record an insufficient or unstable result if:

- GEE or data availability leaves fewer than 4 Batch 2 strict Tier 1 evaluated rows,
- fewer than 3 positive Batch 2 strata remain after QC,
- Batch 2 only primary or spatial CI lower bound is not positive,
- provenance audit finds any invalid Tier 1 row,
- class-collapse or zero-change outcomes dominate a stratum.

## Scientific Interpretation

Supportive Batch 2 results would strengthen only the bounded claim:

> Frozen GeoFM embeddings can support baseline-trend change-signal modelling under repeated independent external holdout evaluation.

They would not support operational forecasting, counterfactual scenario simulation, or planning deployment claims.

Non-supportive Batch 2 results are scientifically useful. They should be documented as limits on external stability and used to guide further candidate selection, diagnostic chips, or failure analysis.

## Deliverables

This phase should produce:

- Batch 2 holdout manifest.
- Combined manifest.
- Batch 2 acquisition manifests and arrays.
- Batch 2 prediction readiness report and prediction arrays.
- Batch 2 benchmark registry, audit, metrics, gate report, and figures.
- Combined benchmark registry, audit, metrics, gate report, and figures.
- A handoff note recording whether Batch 2 only and combined evidence strengthen, weaken, or leave unchanged the Paper58 experimental basis.

## Out Of Scope

The following are intentionally out of scope for this design:

- RSE manuscript edits.
- New planning experiments.
- Manual image-chip interpretation.
- New model architecture.
- Hyperparameter tuning after seeing Batch 2 results.
- Any change to gate thresholds designed to rescue a weak result.
