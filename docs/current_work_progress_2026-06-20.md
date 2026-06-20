# Current Work Progress: 2026-06-20

## Paper58 Resume State

- Repository: `D:\test\paper58-geofm-world-model-rl`
- Active worktree: `D:\test\paper58-geofm-world-model-rl\.worktrees\paper58-benchmark`
- Active branch: `paper58-benchmark`
- Resume from commit: `36e9db3` (`data: run expanded Paper58 Tier 1 benchmark`)

## Governing Rule

Do not strengthen the RSE manuscript unless the expanded strict Tier 1 benchmark passes with genuinely external evidence.

If the expanded gate remains `fail` or `insufficient_tier1`, manuscript work must stay limited to:

- claim downgrading,
- limitations,
- data/reference cleanup,
- negative or insufficient-evidence reporting.

## Design And Plan

Approved design/spec:

```text
docs/superpowers/specs/2026-06-19-paper58-tier1-expansion-design.md
```

Approved implementation plan:

```text
docs/superpowers/plans/2026-06-19-paper58-tier1-expansion.md
```

## Completed Before External Acquisition

Completed Task 4-7 from the Tier 1 expansion plan.

Key commits:

```text
454a2b0 feat: drive Paper58 acquisition from holdout manifest
07c2910 fix: fail closed on invalid Paper58 holdout manifests
54f09fe fix: enforce Paper58 holdout manifest area filters
97e4ffb fix: reject invalid Tier 1 provenance rows
8d5731a feat: add Paper58 provenance audit output
7bd306c data: record strict Paper58 provenance audit
```

## What Was Added

Manifest-driven strict provenance support now exists across the Paper58 benchmark path:

- acquisition scripts can read validated holdout manifests,
- prediction generation is fail-closed when manifest eligibility is missing,
- evaluator rejects invalid Tier 1 provenance rows,
- provenance audit outputs are written before interpreting benchmark results.

## Strict Pre-Acquisition Local Results

Full local verification run:

```text
python -m pytest tests/test_paper58_benchmark_holdouts.py tests/test_paper58_benchmark_registry.py tests/test_paper58_benchmark_evaluation.py tests/test_paper58_benchmark_statistics.py tests/test_paper58_benchmark_figures.py tests/test_paper58_benchmark_provenance_audit.py tests/test_rse_revision_change_validation.py -q
```

Result:

```text
71 passed
```

Whitespace check:

```text
git diff --check
```

Result:

```text
clean
```

Strict local benchmark pipeline was rerun sequentially:

```text
python -m scripts.paper58_benchmark.build_registry --holdout-manifest data\independent_change_labels\paper58_holdout_areas.json
python -m scripts.paper58_benchmark.audit_provenance
python -m scripts.paper58_benchmark.evaluate_benchmark --n-boot 1000
```

Verified outputs:

- Registry: `12 candidate pair(s), 11 included pair(s)`
- Provenance audit: `12 row(s), 0 invalid Tier 1 row(s)`
- Gate: `status = insufficient_tier1`

Strict pre-acquisition provenance summary from:

```text
paper/rse_submission_paper58/benchmark_results/benchmark_provenance_audit.json
```

```json
{
  "n_rows": 12,
  "tier_counts": {
    "tier2": 12
  },
  "invalid_tier1_rows": []
}
```

Strict pre-acquisition gate summary from:

```text
paper/rse_submission_paper58/benchmark_results/benchmark_gate_report.json
```

```json
{
  "status": "insufficient_tier1",
  "positive_tier1_strata": 0,
  "required_positive_tier1_strata": 3,
  "primary_gate_pass": false,
  "spatial_gate_pass": false,
  "strata_gate_pass": false
}
```

## Scientific Interpretation

The strict provenance correction worked:

- `poyang_lake` and `wuyi_mountain` are no longer treated as strict Tier 1 evidence.
- Current local benchmark evidence has `tier1 = 0`.
- Therefore the manuscript must not move toward stronger claims yet.

This is the intended stop/go outcome before new external holdout acquisition.

## Completed External Tier 1 Expansion

Completed Task 8-10 from:

```text
docs/superpowers/plans/2026-06-19-paper58-tier1-expansion.md
```

Key commits:

```text
098ecf2 data: fetch Paper58 Tier 1 holdout labels and embeddings
d253c5b data: generate Paper58 Tier 1 holdout predictions
36e9db3 data: run expanded Paper58 Tier 1 benchmark
```

Task 8 acquisition completed with no failures:

```text
Independent LULC label fetch: complete, 16 record(s), 0 failure(s)
Change-validation embedding fetch: complete, 16 grid(s), 8 context grid(s), 0 failure(s)
```

Task 9 prediction generation completed:

```text
Change-validation prediction generation: complete, 8 prediction(s)
```

Task 10 expanded benchmark was rerun sequentially:

```text
python -m scripts.paper58_benchmark.build_registry --holdout-manifest data\independent_change_labels\paper58_holdout_areas.json
python -m scripts.paper58_benchmark.audit_provenance
python -m scripts.paper58_benchmark.evaluate_benchmark --n-boot 5000
python -m scripts.paper58_benchmark.make_benchmark_figures
```

Verified outputs:

- Registry: `20 candidate pair(s), 18 included pair(s)`
- Provenance audit: `20 row(s), 0 invalid Tier 1 row(s)`
- Evaluation: `18 evaluated pair(s), gate status=pass`
- Figure size: `182492` bytes, so full benchmark figures were committed.

Current expanded gate summary from:

```text
paper/rse_submission_paper58/benchmark_results/benchmark_gate_report.json
```

```json
{
  "status": "pass",
  "tier1_primary_change": {
    "n_rows": 7,
    "n_clusters": 7,
    "mean": 0.25162800142943026,
    "ci_low": 0.10190708381580636,
    "ci_high": 0.4099235447784561
  },
  "tier1_spatial_change": {
    "n_rows": 7,
    "n_clusters": 7,
    "mean": 0.11020757535580673,
    "ci_low": 0.04810621853180953,
    "ci_high": 0.16711471734787858
  },
  "positive_tier1_strata": 4,
  "required_positive_tier1_strata": 3,
  "primary_gate_pass": true,
  "spatial_gate_pass": true,
  "strata_gate_pass": true
}
```

Current provenance summary from:

```text
paper/rse_submission_paper58/benchmark_results/benchmark_provenance_audit.json
```

```json
{
  "n_rows": 20,
  "tier_counts": {
    "tier1": 8,
    "tier2": 12
  },
  "invalid_tier1_rows": []
}
```

Expanded Tier 1 interpretation:

- Seven strict Tier 1 rows were evaluated after QC.
- The evaluated Tier 1 rows cover seven region clusters.
- Positive Tier 1 evidence spans four strata: Agriculture, Forest, Urban, and Wetland.
- `haibei_plateau_holdout` remains visible in the registry as Tier 1 provenance but was excluded by QC with `class_collapse`.
- `poyang_lake` and `wuyi_mountain` remain Tier 2 because of known training contact.

## Batch 2 Tier 1 Stability Check

Design:

```text
docs/superpowers/specs/2026-06-20-paper58-batch2-tier1-design.md
```

Plan:

```text
docs/superpowers/plans/2026-06-20-paper58-batch2-tier1-stability.md
```

Batch 2 only outputs:

```text
paper/rse_submission_paper58/benchmark_results_batch2
```

Combined outputs:

```text
paper/rse_submission_paper58/benchmark_results_combined
```

Decision rule:

- Batch 2 only is the primary stability check.
- Combined pass does not strengthen the evidence if Batch 2 only fails or remains insufficient.

Batch 2 only gate summary from:

```text
paper/rse_submission_paper58/benchmark_results_batch2/benchmark_gate_report.json
```

```json
{
  "status": "fail",
  "tier1_primary_change": {
    "n_rows": 8,
    "n_clusters": 8,
    "ci_low": 0.07977047981666133
  },
  "tier1_spatial_change": {
    "n_rows": 8,
    "n_clusters": 8,
    "ci_low": -0.01096989826772667
  },
  "positive_tier1_strata": 6
}
```

Combined gate summary from:

```text
paper/rse_submission_paper58/benchmark_results_combined/benchmark_gate_report.json
```

```json
{
  "status": "pass",
  "tier1_primary_change": {
    "n_rows": 15,
    "n_clusters": 15,
    "ci_low": 0.11119339244025246
  },
  "tier1_spatial_change": {
    "n_rows": 15,
    "n_clusters": 15,
    "ci_low": 0.025691451755083954
  },
  "positive_tier1_strata": 6
}
```

Stability-check interpretation:

- Batch 2 only fails because the spatial Tier 1 lower confidence bound is below zero.
- Combined pooling passes, but that pooled pass cannot be used as stronger evidence because Batch 2 only failed.
- The practical stop/go outcome is still "do more experiments before strengthening manuscript claims."

## Batch 2 Failure Diagnosis

Diagnosis inputs:

```text
paper/rse_submission_paper58/benchmark_results_batch2/benchmark_metrics_by_pair.csv
paper/rse_submission_paper58/benchmark_results_batch2/benchmark_summary.json
paper/rse_submission_paper58/benchmark_results_batch2/benchmark_failures.csv
```

Diagnostic script and outputs:

```text
scripts/paper58_benchmark/make_batch2_diagnostics.py
paper/rse_submission_paper58/diagnostics_batch2
```

Generated outputs:

```text
paper/rse_submission_paper58/diagnostics_batch2/fig_batch2_xiongan_spatial_failure.png
paper/rse_submission_paper58/diagnostics_batch2/fig_batch2_xiongan_spatial_failure.pdf
paper/rse_submission_paper58/diagnostics_batch2/batch2_spatial_advantage_ranked.csv
paper/rse_submission_paper58/diagnostics_batch2/batch2_spatial_leave_one_out.csv
paper/rse_submission_paper58/diagnostics_batch2/batch2_spatial_alignment_shift.csv
paper/rse_submission_paper58/diagnostics_batch2/batch2_embedding_decoder_audit.csv
paper/rse_submission_paper58/diagnostics_batch2/xiong_an_fringe_holdout_transition_counts.csv
paper/rse_submission_paper58/diagnostics_batch2/xiong_an_fringe_holdout_transition_fate.csv
paper/rse_submission_paper58/diagnostics_batch2/batch2_diagnostic_summary.txt
```

Key findings:

- `benchmark_failures.csv` is empty, so the Batch 2 gate failure is not caused by QC exclusion or provenance invalidation.
- Batch 2 primary advantage is consistently positive across all 8 evaluated Tier 1 rows.
- The failure comes from weak spatial advantage, not from the primary model-versus-baseline comparison.

Spatial-advantage ordering within Batch 2:

- `xiong_an_fringe_holdout`: `-0.07471971561389112`
- `hexi_irrigation_holdout`: `0.0`
- `songnen_plain_holdout`: `0.019263289464010014`
- `changbai_margin_holdout`: `0.02834008097165991`
- `ordos_grassland_holdout`: `0.030841542721009474`
- `erlong_lake_margin_holdout`: `0.031426775612822117`
- `west_sichuan_plateau_holdout`: `0.05736981465136798`
- `beibu_gulf_urban_holdout`: `0.10072847799500165`

Leave-one-out spatial sensitivity:

- All 8 rows: spatial `ci_low = -0.01096989826772667`
- Drop `xiong_an_fringe_holdout`: spatial `ci_low = 0.01871684745598648`
- Dropping any other single row leaves spatial `ci_low <= 0`

Interpretation:

- `xiong_an_fringe_holdout` is the decisive failure point for the Batch 2 spatial gate.
- The rest of Batch 2 still has only a thin positive spatial margin, so the issue is not purely a one-row artifact.

`xiong_an_fringe_holdout` local diagnosis:

- True change pixels: `125`
- Model change pixels: `151`
- Model change `F1 = 0.23188405797101452`
- Spatial-shuffle change `F1 = 0.30660377358490565`
- Multi-seed shuffle sanity check: shuffle mean `F1 = 0.318678`, `2.5% = 0.283688`, `97.5% = 0.355972`
- The saved spatial diagnostic figure shows reference start, reference end, model prediction, shuffled prediction, reference change, predicted change, model error, and shuffle error panels.
- Best whole-mask shift diagnostic: shifting the model change mask by `dy = 3`, `dx = 3` increases change F1 from `0.23188405797101452` to `0.35398230088495575`.

This means the negative spatial result for `xiong_an_fringe_holdout` is not just a fixed-seed shuffle artifact.

Decoder/localization audit for `xiong_an_fringe_holdout`:

- 2020 original embedding decoded against 2020 ESRI label: accuracy `0.7361111111111112`
- 2021 original embedding decoded against 2021 ESRI label: accuracy `0.6527777777777778`
- Change mask from decoded 2020/2021 original embeddings: `F1 = 0.30409356725146197`
- Best shift for decoded-observed change remains `dy = 0`, `dx = 0`, with `F1 = 0.30409356725146197`
- Model-predicted change against the ESRI start label: `F1 = 0.23188405797101452`
- Model-predicted change against the decoded start label: `F1 = 0.0`

Interpretation of the decoder/localization audit:

- The ESRI label and original AlphaEarth embedding grids are not showing the same `dy = 3`, `dx = 3` shift seen in the future model prediction.
- Therefore the leading root-cause hypothesis is not a simple label/embedding grid registration error.
- The current evidence points more narrowly to the learned future-prediction output or its transition typing/localization under this urban-fringe case.

Observed transition mismatch for `xiong_an_fringe_holdout`:

- true dominant transitions: `5->11` (`38` pixels), `5->7` (`30`), `7->11` (`14`), `7->5` (`12`)
- predicted changed pixels: `4->1` (`49`), `7->5` (`26`), `0->11` (`24`), `4->5` (`18`)
- decoded-observed changed pixels: `7->5` (`27`), `5->1` (`11`), `5->7` (`7`)

Transition-fate audit for top true transitions in `xiong_an_fringe_holdout`:

- true `5->11` (`38` pixels): decoded 2020 start is `5:38`; decoded 2021 observed end is still `5:38`; model-predicted end is also `5:38`
- true `5->7` (`30`): decoded 2021 observed end is `5:25;7:5`; model-predicted end is `5:26;7:4`
- true `7->11` (`14`): decoded 2021 observed end is `5:12;7:2`; model-predicted end is `7:11;5:3`
- true `7->5` (`12`): decoded 2021 observed end is `5:12`; model-predicted end is `7:9;5:3`

Interpretation of the transition-fate audit:

- The largest missed transition (`5->11`) is already absent in the decoded 2021 observed embedding, not only in the model forecast.
- Therefore the current evidence does not support a simple story of "the dynamics model alone misplaced an otherwise separable wetland transition."
- The stronger hypothesis is that some critical urban-fringe transitions are weakly separated in the current embedding-plus-decoder semantic view, and the future model inherits that limitation.

Practical reading:

- The model is detecting change mass in `xiong_an_fringe_holdout`, but the spatial placement and/or transition typing is misaligned enough that shuffling the same predicted map performs better.
- The best-shift result indicates a local spatial alignment problem: a small translation of the predicted change mask recovers more signal than the raw prediction.
- The new transition-fate audit shows that semantic transition failure and spatial-localization failure are both involved; `5->11` in particular looks like a decoder/representation bottleneck rather than a pure forecast-only error.
- `hexi_irrigation_holdout` is not negative, but it has only `5` true change pixels and contributes almost no spatial margin.

## Next Valid Step

Continue experiment-first work on `paper58-benchmark`.

Recommended next steps:

- prioritize a new external Batch 3 that strengthens urban spatial robustness rather than pooling Batch 2 away,
- treat `xiong_an_fringe_holdout` as a diagnostic area for spatial-localization failure, not as manuscript-strengthening evidence,
- run a targeted future-prediction localization robustness audit around `xiong_an_fringe_holdout` if model debugging is prioritized before Batch 3,
- if model debugging continues, separate forecast-stage error from decoder-stage error before spending time on architecture changes,
- consider Batch 3 urban replacement holdouts if the next priority is stronger external evidence rather than model debugging,
- keep manuscript work limited to transparent negative or mixed-evidence reporting until an independent new batch passes on its own.

## Resume Instruction

In a new window, continue from branch `paper58-benchmark`.

Batch 2 stability check is complete.

Resume from the Batch 2 decision rule above:

- treat `benchmark_results_batch2` as the primary readout,
- do not treat the combined pooled pass as permission to strengthen the manuscript,
- use `xiong_an_fringe_holdout` as the first diagnostic target when planning the next experiment,
- remember that the decoder/localization audit points away from simple label/embedding registration as the main cause,
- remember that the new transition-fate audit makes `5->11` and part of `5->7` look representation/decoder-limited, not just forecast-limited,
- continue with stronger and more diverse experiments first.
