# Current Work Progress: 2026-06-20

## Paper58 Resume State

- Repository: `D:\test\paper58-geofm-world-model-rl`
- Active worktree: `D:\test\paper58-geofm-world-model-rl\.worktrees\paper58-benchmark`
- Active branch: `paper58-benchmark`
- Resume from current pushed `paper58-benchmark` branch head.

## Governing Rule

Do not strengthen the RSE manuscript from pooled evidence alone. Independent external batches remain the primary evidence standard.

Current experiment-first rule:

- Batch 2 only failed and must remain visible as contradictory evidence.
- Batch 3 only passed and can be treated as independent supportive evidence.
- Because the current user priority is "stronger and more complete experiments first", keep manuscript work secondary unless explicitly resumed later.

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
paper/rse_submission_paper58/diagnostics_batch2/batch2_decoder_true_end_confidence_by_area.csv
paper/rse_submission_paper58/diagnostics_batch2/batch2_forecast_true_end_confidence_by_area.csv
paper/rse_submission_paper58/diagnostics_batch2/xiong_an_fringe_holdout_transition_counts.csv
paper/rse_submission_paper58/diagnostics_batch2/xiong_an_fringe_holdout_transition_fate.csv
paper/rse_submission_paper58/diagnostics_batch2/xiong_an_fringe_holdout_shifted_transition_fate.csv
paper/rse_submission_paper58/diagnostics_batch2/xiong_an_fringe_holdout_forecast_transition_fate.csv
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

Decoder-confidence audit inside the same transition-fate table:

- true `5->11`: mean probability assigned to the true end class `11` is only `0.00162` (median `0.00139`); mean top class is `5` with probability `0.923946`
- true `7->11`: mean probability assigned to class `11` is only `0.002345` (median `0.000939`); mean top class is `5` with probability `0.819499`
- true `5->7`: class `7` is not impossible, but still secondary on average: mean `0.213349` versus class `5` mean `0.784187`
- true `7->5`: class `5` is correctly dominant on average with mean `0.919697`

Cross-area decoder-confidence audit for true end class `11`:

- `xiong_an_fringe_holdout`: `52` changed pixels ending as class `11`, mean true-end probability `0.001815`, median `0.001362`, top decoded class `5` (`50` pixels)
- `beibu_gulf_urban_holdout`: mean `0.103621`
- `erlong_lake_margin_holdout`: mean `0.105357`
- `songnen_plain_holdout`: mean `0.306869`
- `ordos_grassland_holdout`: mean `0.370788`
- `west_sichuan_plateau_holdout`: mean `0.51839`
- `hexi_irrigation_holdout`: mean `0.813417`, but only `1` changed pixel ending as class `11`

Forecast-embedding confidence audit for `xiong_an_fringe_holdout`:

- The audit recomputes the 2020->2021 forecast embedding and compares decoder probabilities on the observed 2021 embedding versus the forecast embedding.
- Top true transition `5->11` (`38` pixels): observed 2021 embedding gives class `11` mean probability `0.00162`; forecast embedding gives `0.001977`; both decode all pixels as class `5`.
- True `7->11` (`14` pixels): observed class `11` mean probability `0.002345`; forecast class `11` mean probability `0.001308`.
- Aggregated true end class `11` in xiong'an (`52` changed pixels): observed mean `0.001815`, forecast mean `0.001797`, delta `-0.000018`; top decoded class remains `5`.
- Cross-area observed-vs-forecast class `11` confidence:
  - `xiong_an_fringe_holdout`: observed `0.001815`, forecast `0.001797`
  - `beibu_gulf_urban_holdout`: observed `0.103621`, forecast `0.064316`
  - `erlong_lake_margin_holdout`: observed `0.105357`, forecast `0.213454`
  - `songnen_plain_holdout`: observed `0.306869`, forecast `0.427367`
  - `ordos_grassland_holdout`: observed `0.370788`, forecast `0.662011`
  - `west_sichuan_plateau_holdout`: observed `0.51839`, forecast `0.576219`
  - `hexi_irrigation_holdout`: observed `0.813417`, forecast `0.839603`, but only `1` pixel

Shifted-transition fate audit for `xiong_an_fringe_holdout`:

- The audit applies the best whole-mask shift (`dy = 3`, `dx = 3`) to the model changed pixels and then rechecks the true transition pixels.
- The shift improves coarse change-mask F1 from `0.23188405797101452` to `0.35398230088495575`, but it does not recover class `11` transition typing.
- True `5->11` (`38` pixels): raw model end `5:38`; shifted model end `5:38`; raw class-11 matches `0`; shifted class-11 matches `0`.
- True `7->11` (`14` pixels): raw model end `7:11;5:3`; shifted model end `7:7;1:6;5:1`; raw class-11 matches `0`; shifted class-11 matches `0`.
- True `5->7` (`30` pixels): raw class-7 matches `4`; shifted class-7 matches `1`, so the best coarse shift does not improve this semantic transition either.

Interpretation of the transition-fate audit:

- The largest missed transition (`5->11`) is already absent in the decoded 2021 observed embedding, not only in the model forecast.
- Therefore the current evidence does not support a simple story of "the dynamics model alone misplaced an otherwise separable wetland transition."
- The stronger hypothesis is that some critical urban-fringe transitions are weakly separated in the current embedding-plus-decoder semantic view, and the future model inherits that limitation.
- The new probability readout strengthens that diagnosis: class `11` is not merely losing the argmax by a small margin in the true 2021 embedding; it is receiving near-zero average probability on the two most important missed wetland transitions.
- The cross-area table shows this is unusually severe in `xiong_an_fringe_holdout`, not a uniform class-`11` failure across all Batch 2 holdouts.
- The forecast-embedding probability audit further narrows the failure: forecast dynamics do not erase a strong class `11` signal in xiong'an, because the observed 2021 embedding already assigns near-zero probability to class `11`. Forecasting leaves that near-zero signal essentially unchanged for the aggregated class `11` pixels.
- The shifted-transition fate audit separates localization from semantics: the best spatial shift improves binary change overlap but still gives zero class-`11` matches for `5->11` and `7->11`, so the class-`11` failure is not solved by the spatial shift.

Practical reading:

- The model is detecting change mass in `xiong_an_fringe_holdout`, but the spatial placement and/or transition typing is misaligned enough that shuffling the same predicted map performs better.
- The best-shift result indicates a local spatial alignment problem: a small translation of the predicted change mask recovers more signal than the raw prediction.
- The new transition-fate, decoder-confidence, forecast-confidence, and shifted-transition audits show that semantic transition failure and spatial-localization failure are both involved; `5->11` in particular now looks like a decoder/representation bottleneck that is not repaired by the best spatial shift.
- The forecast-confidence audit points away from "forecast stage erased a separable wetland signal" as the primary explanation for class `11`; the observed AlphaEarth embedding plus decoder already misses that semantic transition in xiong'an.
- `hexi_irrigation_holdout` is not negative, but it has only `5` true change pixels and contributes almost no spatial margin.

## Next Valid Step

Continue experiment-first work on `paper58-benchmark`.

Recommended next steps:

- keep Batch 2 and Batch 3 separated in all interpretations rather than pooling away the Batch 2 failure,
- treat `xiong_an_fringe_holdout` as a diagnostic area for spatial-localization failure, not as manuscript-strengthening evidence,
- use the forecast-confidence and shifted-transition audits as the current forecast-vs-decoder separation for class `11` in xiong'an,
- compare the Batch 2 urban failure cases against the Batch 3 urban pass cases before changing the dynamics model,
- if model debugging continues, focus on spatial localization robustness and alternate decoder/representation probes before expanding manuscript claims,
- prioritize another external batch or a targeted robustness experiment before moving serious effort back to the paper.

## Batch 3 External Tier 1 Check

Batch 3 manifest:

```text
data/independent_change_labels/paper58_holdout_areas_batch3.json
```

Batch 3 design intent:

- keep Batch 3 fully separate from Batch 1 and Batch 2 area names,
- bias the candidate set toward urban-fringe and wetland-edge robustness,
- preserve some ecological diversity so Batch 3 does not collapse to a single transition theme,
- treat Batch 3 only as the next primary readout if acquisition and prediction complete.

Batch 3 manifest shape:

- `10` candidate areas,
- `10` strict Tier 1 candidates by current provenance rules,
- at least `4` Urban candidates,
- at least `3` Wetland candidates,
- all candidate year pairs are `2020-2021`.

Batch 3 acquisition status:

- Labels: `Independent LULC label fetch: complete, 20 record(s), 0 failure(s)`
- Embeddings: `Change-validation embedding fetch: complete, 20 grid(s), 10 context grid(s), 0 failure(s)`
- Predictions: `Change-validation prediction generation: complete, 10 prediction(s)`
- Registry: `10` candidate pair(s), `9` included pair(s)
- Provenance audit: `10` row(s), `0` invalid Tier 1 row(s)

Batch 3 outputs:

```text
paper/rse_submission_paper58/benchmark_results_batch3
```

Batch 3 gate summary from:

```text
paper/rse_submission_paper58/benchmark_results_batch3/benchmark_gate_report.json
```

```json
{
  "status": "pass",
  "tier1_primary_change": {
    "n_rows": 9,
    "n_clusters": 9,
    "ci_low": 0.10797167654040589
  },
  "tier1_spatial_change": {
    "n_rows": 9,
    "n_clusters": 9,
    "ci_low": 0.06050222512559061
  },
  "positive_tier1_strata": 5
}
```

Batch 3 QC note from:

```text
paper/rse_submission_paper58/benchmark_results_batch3/benchmark_failures.csv
```

- `taihu_marsh_edge_holdout` was excluded as `negative_control` with `zero_reference_change`.

Batch 3 interpretation:

- Batch 3 only independently passes the Tier 1 gate.
- This is the first new external batch after the Batch 2 failure that passes on its own rather than through pooling.
- Positive Tier 1 evidence spans `5` strata: Agriculture, Forest, Grassland, Urban, and Wetland.
- Urban coverage is materially better than a single replacement row: `4` urban Tier 1 rows were evaluated and all had positive primary advantage.
- Spatial behavior is supportive overall, but not uniformly perfect: `suzhou_fringe_holdout` has a slight negative spatial advantage (`-0.0009090909090908872`) even though the batch-level spatial gate passes.
- Therefore the evidence is stronger than the Batch 2-only readout, but the Batch 2 failure still needs to stay visible in any later narrative.

## Batch 2 vs Batch 3 Comparison Diagnosis

Diagnostic script:

```text
scripts/paper58_benchmark/make_batch23_comparison_diagnostics.py
```

Diagnostic outputs:

```text
paper/rse_submission_paper58/diagnostics_batch23
```

Generated files:

```text
paper/rse_submission_paper58/diagnostics_batch23/batch23_spatial_advantage_ranked.csv
paper/rse_submission_paper58/diagnostics_batch23/batch23_comparison_summary.json
paper/rse_submission_paper58/diagnostics_batch23/batch23_comparison_summary.txt
```

Batch-level contrast:

```json
{
  "batch2_status": "fail",
  "batch2_spatial_ci_low": -0.01096989826772667,
  "batch3_status": "pass",
  "batch3_spatial_ci_low": 0.06050222512559061
}
```

Urban-only contrast:

```json
{
  "batch2": {
    "n": 2,
    "mean_primary_change_advantage": 0.100659,
    "mean_spatial_change_advantage": 0.013004,
    "n_negative_spatial": 1,
    "spatial_ci_low": -0.07472
  },
  "batch3": {
    "n": 4,
    "mean_primary_change_advantage": 0.24143,
    "mean_spatial_change_advantage": 0.158805,
    "n_negative_spatial": 1,
    "spatial_ci_low": 0.047863
  }
}
```

Comparison interpretation:

- `xiong_an_fringe_holdout` remains the lowest spatial row across Batch 2 and Batch 3 combined: `spatial_change_advantage = -0.07471971561389112`.
- `suzhou_fringe_holdout` is a useful Batch 3 caution case: it is slightly negative spatially (`-0.0009090909090908872`) but not severe enough to dominate the batch.
- Batch 3 urban support is broad enough to pass the urban-only spatial bootstrap readout: `spatial_ci_low = 0.047863`.
- Batch 2 urban support is not broad enough because the two-row urban set splits into one positive (`beibu_gulf_urban_holdout`) and one strongly negative (`xiong_an_fringe_holdout`).
- The next experiment should not simply add more random urban holdouts. It should contrast xiong'an-like urban-fringe cases against Batch 3's better urban cases and test whether the failure is driven by local transition semantics, spatial localization, or both.

## Batch 2/3 Urban Contrast Diagnosis

Diagnostic script:

```text
scripts/paper58_benchmark/make_batch23_urban_contrast.py
```

Diagnostic outputs:

```text
paper/rse_submission_paper58/diagnostics_batch23_urban
```

Compared urban areas:

- Batch 2: `xiong_an_fringe_holdout`, `beibu_gulf_urban_holdout`
- Batch 3: `fuzhou_delta_urban_holdout`, `nanning_fringe_holdout`, `suzhou_fringe_holdout`, `wuhan_outer_ring_holdout`

Spatial alignment contrast from:

```text
paper/rse_submission_paper58/diagnostics_batch23_urban/batch2_spatial_alignment_shift.csv
```

- `xiong_an_fringe_holdout`: raw change F1 `0.23188405797101452`, best-shift F1 `0.35398230088495575`, best shift `dy=3`, `dx=3`.
- `suzhou_fringe_holdout`: raw change F1 `0.15909090909090912`, best-shift F1 `0.25301204819277107`, best shift `dy=0`, `dx=1`.
- `beibu_gulf_urban_holdout`, `fuzhou_delta_urban_holdout`, `nanning_fringe_holdout`, and `wuhan_outer_ring_holdout` all have best shift `dy=0`, `dx=0`.

Decoder/representation contrast from:

```text
paper/rse_submission_paper58/diagnostics_batch23_urban/batch2_decoder_true_end_confidence_by_area.csv
paper/rse_submission_paper58/diagnostics_batch23_urban/batch2_forecast_true_end_confidence_by_area.csv
```

Class `11` true-end confidence among urban holdouts:

- `xiong_an_fringe_holdout`: `52` changed pixels ending as class `11`, observed mean true-end probability `0.001815`, forecast mean `0.001797`, delta `-0.000018`.
- `nanning_fringe_holdout`: `3` pixels, observed mean `0.005612`, forecast mean `0.001659`, delta `-0.003953`.
- `suzhou_fringe_holdout`: `7` pixels, observed mean `0.057749`, forecast mean `0.008375`, delta `-0.049374`.
- `wuhan_outer_ring_holdout`: `4` pixels, observed mean `0.089355`, forecast mean `0.019453`, delta `-0.069902`.
- `beibu_gulf_urban_holdout`: `3` pixels, observed mean `0.103621`, forecast mean `0.064316`, delta `-0.039305`.

Top-transition contrast from:

```text
paper/rse_submission_paper58/diagnostics_batch23_urban/urban_transition_fate_all.csv
paper/rse_submission_paper58/diagnostics_batch23_urban/urban_forecast_transition_fate_all.csv
```

Key urban class-`11` findings:

- `xiong_an_fringe_holdout` is not merely forecast-erased: the observed 2021 embedding already assigns near-zero probability to class `11` on `5->11` and `7->11`, and the forecast leaves that near-zero signal essentially unchanged.
- `suzhou_fringe_holdout`, `wuhan_outer_ring_holdout`, and `beibu_gulf_urban_holdout` show a different pattern: class-`11` pixels are much fewer and observed probabilities are higher than xiong'an, but the forecast can reduce the class-`11` probability substantially.
- Therefore xiong'an remains the primary representation/decoder bottleneck case, while suzhou is the nearest spatial-localization caution case in Batch 3.

Practical next experiment:

- Do not treat "urban" as one homogeneous stratum.
- Add or mine xiong'an-like urban-fringe cases with substantial `5/7 -> 11` transitions to test whether the class-`11` bottleneck repeats.
- In parallel, use suzhou-like cases to test small-offset localization sensitivity separately from the class-`11` representation failure.

## Batch 4 Urban Failure-Mode Manifest

Batch 4 design spec:

```text
docs/superpowers/specs/2026-06-20-paper58-batch4-urban-failure-modes-design.md
```

Batch 4 manifest:

```text
data/independent_change_labels/paper58_holdout_areas_batch4.json
```

Manifest-only status:

- Batch 4 currently exists only as a candidate manifest. No acquisition, prediction generation, registry build, provenance audit, or benchmark evaluation has been run yet for this batch.
- The batch contains `10` strict Tier 1 urban candidates, all with year pair `2020-2021`.
- The intended mix is `7` `xiong_an_like` urban-fringe candidates plus `3` `suzhou_like` urban-fringe candidates.
- Batch 4 area names are intentionally disjoint from Batch 1, Batch 2, and Batch 3.
- All rows keep `development_contact_status = none` and explicit no-contact evidence in the manifest.

Interpretation rule:

- Batch 4-only should be treated as the next primary external readout after acquisition and prediction complete.
- If Batch 4 fails, keep that failure visible rather than pooling it away.
- If Batch 4 passes, compare it against Batch 2 and Batch 3 before making any stronger manuscript claim.

## Batch 4 External Urban Failure-Mode Check

Batch 4 acquisition and prediction status:

- Labels: `Independent LULC label fetch: complete, 20 record(s), 0 failure(s)`
- Embeddings: `Change-validation embedding fetch: complete, 20 grid(s), 10 context grid(s), 0 failure(s)`
- Predictions: `Change-validation prediction generation: complete, 10 prediction(s)`
- Registry: `10` candidate pair(s), `10` included pair(s)
- Provenance audit: `10` row(s), `0` invalid Tier 1 row(s)

Batch 4 outputs:

```text
paper/rse_submission_paper58/benchmark_results_batch4
```

Batch 4 gate summary from:

```text
paper/rse_submission_paper58/benchmark_results_batch4/benchmark_gate_report.json
```

```json
{
  "status": "fail",
  "tier1_primary_change": {
    "n_rows": 10,
    "n_clusters": 10,
    "ci_low": 0.09074926988339427
  },
  "tier1_spatial_change": {
    "n_rows": 10,
    "n_clusters": 10,
    "ci_low": 0.06973994206162697
  },
  "positive_tier1_strata": 1
}
```

Batch 4 interpretation:

- Batch 4-only fails the full benchmark gate.
- That failure is not a rerun of the Batch 2 spatial collapse. Batch 4 primary and spatial gates both pass, but the batch is intentionally all-Urban, so it cannot satisfy the benchmark's cross-stratum requirement of `3` positive Tier 1 strata.
- Therefore the correct reading is "useful targeted urban evidence, but not a substitute for the multi-stratum external gate."
- No QC exclusions occurred in Batch 4; all `10` Tier 1 urban rows were evaluated.

Batch 4 within-batch urban split:

- `xiong_an_like` subset (`7` rows): mean primary advantage `0.11029778521218196`, mean spatial advantage `0.07609815979578147`, `1` negative primary row, `0` negative spatial rows.
- `suzhou_like` subset (`3` rows): mean primary advantage `0.21521442869783067`, mean spatial advantage `0.1710894196229192`, `0` negative primary rows, `0` negative spatial rows.
- Worst `xiong_an_like` primary row: `xinxiang_floodplain_newtown_holdout` with primary advantage `-0.004336043360433604` and only `7` true change pixels.
- Weakest `xiong_an_like` spatial row: `xiongxian_river_corridor_holdout` with spatial advantage `0.020533766015777985`.
- All `3` `suzhou_like` rows remain positive on both primary and spatial advantage.

Practical reading:

- Batch 4 adds independent urban-only evidence that the Batch 2 `xiong_an` failure mode does not automatically recur across every similar urban-fringe candidate.
- The `suzhou_like` slice looks stronger than the `xiong_an_like` slice in this batch, which is directionally consistent with the earlier diagnosis that `xiong_an` is the harder representation/decoder bottleneck case.
- Batch 4 does not erase Batch 2. It narrows the problem: the severe Batch 2 failure is not reproduced as a batch-level urban spatial collapse here, but the harder `xiong_an_like` subset is still weaker than the `suzhou_like` subset.
- Because Batch 4 is all Urban by design, it should be treated as a targeted diagnostic batch rather than as the next manuscript-strengthening benchmark gate.

## Batch 4 Urban Deep Diagnosis

Diagnostic outputs:

```text
paper/rse_submission_paper58/diagnostics_batch4_urban
```

Generated files include:

```text
paper/rse_submission_paper58/diagnostics_batch4_urban/batch2_spatial_alignment_shift.csv
paper/rse_submission_paper58/diagnostics_batch4_urban/batch2_embedding_decoder_audit.csv
paper/rse_submission_paper58/diagnostics_batch4_urban/batch2_decoder_true_end_confidence_by_area.csv
paper/rse_submission_paper58/diagnostics_batch4_urban/batch2_forecast_true_end_confidence_by_area.csv
paper/rse_submission_paper58/diagnostics_batch4_urban/batch4_transition_fate_all.csv
paper/rse_submission_paper58/diagnostics_batch4_urban/batch4_shifted_transition_fate_all.csv
paper/rse_submission_paper58/diagnostics_batch4_urban/batch4_forecast_transition_fate_all.csv
paper/rse_submission_paper58/diagnostics_batch4_urban/batch4_urban_summary.json
paper/rse_submission_paper58/diagnostics_batch4_urban/batch4_urban_summary.txt
```

Batch 4 alignment summary:

- `xiong_an_like` mean raw change F1 is `0.18969518718242584`; mean best-shift change F1 is `0.20850384060590588`.
- `xiong_an_like` has `2` non-zero best-shift rows: `xiongxian_river_corridor_holdout` (`dy=2`, `dx=0`) and `xinxiang_floodplain_newtown_holdout` (`dy=0`, `dx=-3`).
- `suzhou_like` mean raw change F1 is `0.30944500731734775`; mean best-shift change F1 is the same, and `0` rows need a non-zero best shift.
- Therefore Batch 4 does not reproduce a broad urban spatial-offset failure. The localization concern is concentrated in the weaker `xiong_an_like` rows.

Batch 4 class-`11` diagnosis:

- Among `xiong_an_like` rows with class-`11` true-end pixels, mean observed class-`11` probability is `0.04760283333333333`; mean forecast class-`11` probability is `0.0226665`, with mean delta `-0.024936333333333335`.
- `xiongxian_river_corridor_holdout` is the strongest new analogue to the Batch 2 `xiong_an` semantic issue: `14` true class-`11` end pixels, observed mean class-`11` probability `0.002548`, and forecast mean class-`11` probability `0.005619`.
- `hengshui_lake_urban_edge_holdout` also has very low observed class-`11` probability (`0.002017`) on `4` pixels.
- `xinxiang_floodplain_newtown_holdout` has the lowest class-`11` probability (`0.000407`), but only `2` class-`11` end pixels and only `7` total true-change pixels, so it is a weak primary-error row but not a large class-`11` stress case.
- `baiyangdian_new_area_holdout` has higher observed class-`11` probability (`0.214879` on `8` pixels) but the forecast reduces it to `0.11491`, so it looks more like forecast suppression than the near-zero observed representation failure seen in `xiong_an`.
- `suzhou_like` class-`11` rows are few and also low confidence on average, but this slice remains much stronger on raw change F1 and does not show non-zero best-shift localization needs.

Transition-fate interpretation:

- Class-`11` semantic mismatches remain visible in Batch 4: for example, `xiongxian_river_corridor_holdout` true `5->11` has `14` pixels, but observed embeddings and model predictions are dominated by class `5`, with class `11` receiving near-zero probability.
- Whole-mask shifts do not recover class-`11` transition typing on the class-`11` rows; this matches the Batch 2 lesson that spatial shifting and semantic decoding are separable problems.
- The evidence now supports a narrower statement: class-`11` representation/decoder weakness recurs in some `xiong_an_like` urban-fringe cases, but Batch 4 does not show a batch-level urban spatial collapse.

Next experimental implication:

- The next external batch should not be another all-Urban batch if the goal is to pass the full benchmark gate; it needs at least three positive Tier 1 strata.
- For targeted diagnosis, prioritize `xiongxian_river_corridor_holdout` and `baiyangdian_new_area_holdout`: the former is the clearer observed class-`11` bottleneck analogue, while the latter may separate observed representation from forecast suppression.
- If adding a Batch 5, use a mixed design: keep a small number of class-`11` stress urban-fringe rows, but add non-Urban strata to restore the full gate's cross-stratum logic.

## Batch 5 Mixed-Gate Design Direction

Batch 5 design spec:

```text
docs/superpowers/specs/2026-06-20-paper58-batch5-mixed-gate-design.md
```

Approved design direction:

- Use a mixed `6-8` row Tier 1 candidate batch rather than another all-Urban batch.
- Include `2` `xiong_an_like` Urban stress rows, `1` `suzhou_like` Urban localization row, `2` Wetland or Agriculture rows, and `1-2` Forest or Grassland rows.
- Require no area-name overlap with Batch 1, Batch 2, Batch 3, or Batch 4.
- Keep the batch manifest-only first; do not run acquisition until the candidate rows and provenance are recorded.

Design rationale:

- Batch 2 failed the spatial gate.
- Batch 3 passed as a mixed external batch.
- Batch 4 passed primary and spatial confidence checks but failed the full gate because it was all Urban.
- Therefore Batch 5 should restore the full gate's cross-stratum logic while retaining a small urban stress probe.

## Batch 5 Mixed-Gate Manifest

Batch 5 manifest:

```text
data/independent_change_labels/paper58_holdout_areas_batch5.json
```

Manifest-only status:

- Batch 5 currently exists only as a candidate manifest. No acquisition, prediction generation, registry build, provenance audit, or benchmark evaluation has been run yet for this batch.
- The manifest contains `7` strict Tier 1 `2020-2021` candidates with no area-name overlap against Batch 1, Batch 2, Batch 3, or Batch 4.
- The candidate mix is `2` `xiong_an_like` Urban stress rows, `1` `suzhou_like` Urban localization row, `1` Wetland row, `1` Agriculture row, `1` Forest row, and `1` Grassland row.
- The manifest therefore restores multi-stratum full-gate coverage by design while keeping the targeted urban stress probe limited to three rows.

Interpretation rule:

- Batch 5-only should be treated as the next primary full-gate readout once acquisition, prediction generation, registry build, provenance audit, and benchmark evaluation are completed.
- Do not pool Batch 5 with earlier batches to hide any Batch 5-only failure.
- Do not strengthen manuscript claims from Batch 5 until its standalone readout is compared against the contradictory Batch 2 result and the targeted Batch 4 urban diagnostic result.

Next step:

- Run Batch 5 acquisition only after accepting this manifest-only design as the next experiment step.

## Batch 5 Mixed-Gate External Readout

Batch 5 acquisition and prediction status:

- Labels: `Independent LULC label fetch: complete, 14 record(s), 0 failure(s)`
- Embeddings: `Change-validation embedding fetch: complete, 14 grid(s), 7 context grid(s), 0 failure(s)`
- Predictions: `Change-validation prediction generation: complete, 7 prediction(s)`
- Registry: `7` candidate pair(s), `7` included pair(s)
- Provenance audit: `7` row(s), `0` invalid Tier 1 row(s)

Batch 5 outputs:

```text
paper/rse_submission_paper58/benchmark_results_batch5
```

Batch 5 gate summary from:

```text
paper/rse_submission_paper58/benchmark_results_batch5/benchmark_gate_report.json
```

```json
{
  "status": "pass",
  "tier1_primary_change": {
    "n_rows": 7,
    "n_clusters": 7,
    "ci_low": 0.03874008044819379
  },
  "tier1_spatial_change": {
    "n_rows": 7,
    "n_clusters": 7,
    "ci_low": 0.026682096295538787
  },
  "positive_tier1_strata": 4
}
```

Batch 5 interpretation:

- Batch 5-only independently passes the full gate.
- The batch restores cross-stratum support after the intentionally all-Urban Batch 4 design, with positive Tier 1 evidence across `4` strata.
- All `7` evaluated rows remain strict Tier 1 by provenance, so this readout is not being rescued by Tier mixing or pooled reinterpretation.

Risk still kept visible:

- `liaohe_delta_wetland_holdout` is a real negative row inside a passing batch: primary advantage `-0.009467775748968649`, spatial advantage `-0.036697621375040734`, with `62` true change pixels.
- `wenan_lakeplain_newtown_holdout` is also weak on primary advantage: `-0.03290246768507639`, though its spatial advantage remains slightly positive at `0.010098010098010098`.
- Therefore Batch 5 is better read as a successful mixed-gate experiment with retained within-batch heterogeneity, not as permission to erase Batch 2 or claim uniformly strong behavior.

Practical reading:

- The mixed design worked as intended for the benchmark gate: it preserved the urban stress probe while restoring enough non-Urban support to satisfy the cross-stratum rule.
- `renqiu_baiyangdian_edge_holdout`, `wuxi_taihu_dense_edge_holdout`, `dabie_forest_edge_holdout`, and `xilingol_grassland_margin_holdout` are supportive rows.
- `liaohe_delta_wetland_holdout` is the first Batch 5 follow-up target if another diagnostic round is needed, because it is a negative wetland row with non-trivial reference change rather than a tiny-count artifact.
- Batch 5 does not justify strengthening the manuscript by itself. Any later narrative still has to retain the contradictory Batch 2 failure, the supportive-but-not-uniform Batch 3 pass, and the targeted Batch 4 urban diagnostic result.

## Batch 5 Liaohe Wetland Diagnosis

Diagnostic script:

```text
scripts/paper58_benchmark/make_batch5_liaohe_diagnostics.py
```

Diagnostic outputs:

```text
paper/rse_submission_paper58/diagnostics_batch5_liaohe
```

Generated files include:

```text
paper/rse_submission_paper58/diagnostics_batch5_liaohe/batch5_liaohe_spatial_alignment_shift.csv
paper/rse_submission_paper58/diagnostics_batch5_liaohe/batch5_liaohe_embedding_decoder_audit.csv
paper/rse_submission_paper58/diagnostics_batch5_liaohe/batch5_liaohe_transition_counts_all.csv
paper/rse_submission_paper58/diagnostics_batch5_liaohe/batch5_liaohe_transition_fate_all.csv
paper/rse_submission_paper58/diagnostics_batch5_liaohe/batch5_liaohe_shifted_transition_fate_all.csv
paper/rse_submission_paper58/diagnostics_batch5_liaohe/batch5_liaohe_forecast_transition_fate_all.csv
paper/rse_submission_paper58/diagnostics_batch5_liaohe/batch5_liaohe_decoder_true_end_confidence_by_area.csv
paper/rse_submission_paper58/diagnostics_batch5_liaohe/batch5_liaohe_forecast_true_end_confidence_by_area.csv
paper/rse_submission_paper58/diagnostics_batch5_liaohe/batch5_liaohe_summary.json
paper/rse_submission_paper58/diagnostics_batch5_liaohe/batch5_liaohe_summary.txt
```

Comparison rows:

- Focus: `liaohe_delta_wetland_holdout`
- Prior wetland comparison rows: `erlong_lake_margin_holdout`, `honghu_lake_margin_holdout`, `zhalong_wetland_edge_holdout`
- Secondary Batch 5 weak-primary risk: `wenan_lakeplain_newtown_holdout`

Liaohe headline metrics:

- Primary advantage: `-0.009467775748968649`
- Spatial advantage: `-0.036697621375040734`
- True change pixels: `62`
- Raw model change F1: `0.1792114695340502`
- Best-shift change F1: `0.3942307692307692`
- Best shift: `dy = 2`, `dx = -4`

Liaohe interpretation:

- The best-shift result is large: shifting the model change mask improves change F1 from `0.1792114695340502` to `0.3942307692307692`, so spatial localization is a real part of the Liaohe failure.
- The dominant true transition is `5->11` with `29` pixels. The decoded observed 2021 embedding is dominated by `5:26;1:3`, and the model prediction is dominated by `5:28;1:1`.
- Among all Liaohe true class-`11` end pixels (`46` pixels), the observed embedding assigns class `11` mean probability `0.052933` and decodes mostly as class `5` (`35` pixels).
- The forecast embedding raises mean class-`11` probability to `0.104908`, but the forecast top decoded class still remains `5` (`38` pixels). This is not a simple forecast-erasure case; the observed representation is already weak for class `11`, and the forecast does not recover the argmax.
- The best spatial shift improves coarse change overlap but does not solve the dominant semantic transition: for true `5->11`, raw model matches are `0`, shifted matches are only `1`.
- Compared with prior wetland rows, Liaohe has lower class-`11` observed confidence than `erlong_lake_margin_holdout` (`0.105357`) and `zhalong_wetland_edge_holdout` (`0.236527`), while `honghu_lake_margin_holdout` has no class-`11` true-end row in this comparison.

Practical reading:

- `liaohe_delta_wetland_holdout` is a combined spatial-localization and class-`11` representation/decoder weakness case.
- It is more similar to the Batch 2/Batch 4 class-`11` bottleneck family than to a pure localization-only miss, but its best-shift improvement is strong enough that localization should remain part of the diagnosis.
- `wenan_lakeplain_newtown_holdout` remains a secondary risk because its primary advantage is negative (`-0.03290246768507639`) with only `10` true change pixels; it should not displace Liaohe as the main Batch 5 follow-up target.
- This diagnosis does not change the Batch 5 pass and does not erase Batch 2. It preserves the correct reading: Batch 5 is a mixed-gate pass with meaningful within-batch heterogeneity.

## Resume Instruction

In a new window, continue from branch `paper58-benchmark`.

Batch 2 and Batch 3 checks are complete, Batch 4 external urban-only readout is recorded, and Batch 5 mixed-gate external readout is complete and pushed.

Resume from the current decision rule:

- treat `benchmark_results_batch2` as the primary readout,
- treat `benchmark_results_batch3` as the primary new independent supportive readout,
- treat `benchmark_results_batch5` as the latest independent mixed-gate pass, but not as permission to erase Batch 2 or directly strengthen the manuscript,
- do not treat the combined pooled pass as permission to strengthen the manuscript,
- use `xiong_an_fringe_holdout` as the first diagnostic target when planning the next experiment,
- remember that the decoder/localization audit points away from simple label/embedding registration as the main cause,
- remember that the new transition-fate audit makes `5->11` and part of `5->7` look representation/decoder-limited, not just forecast-limited,
- remember that the decoder-confidence audit shows class `11` gets near-zero probability on the main missed wetland transitions,
- remember that the cross-area confidence table shows xiong'an is the lowest-confidence class-`11` case in Batch 2,
- remember that the forecast-confidence audit shows forecast embeddings do not materially reduce xiong'an class `11` probability beyond the already near-zero observed 2021 embedding,
- remember that the shifted-transition fate audit shows xiong'an's best spatial shift does not recover `5->11` or `7->11` semantic matches,
- remember that Batch 3-only now passes with `primary ci_low = 0.10797167654040589`, `spatial ci_low = 0.06050222512559061`, and `positive_tier1_strata = 5`,
- remember that `taihu_marsh_edge_holdout` was excluded only because it had `zero_reference_change`,
- remember that the Batch 2 vs Batch 3 comparison diagnostic identifies xiong'an as the lowest spatial row across both batches and Batch 3 urban as broader but still not uniformly perfect,
- remember that the urban contrast diagnostic separates xiong'an's near-zero observed class-`11` representation from suzhou/wuhan/beibu's smaller forecast-suppressed class-`11` cases,
- remember that Batch 4-only now fails the full benchmark gate because it is all-Urban and therefore misses the `3`-strata requirement, even though its primary and spatial Tier 1 confidence bounds are both positive,
- remember that the Batch 4 `suzhou_like` subset is stronger than the Batch 4 `xiong_an_like` subset, and that the weakest new primary row is `xinxiang_floodplain_newtown_holdout`,
- remember that Batch 4 deep diagnostics identify `xiongxian_river_corridor_holdout` as the clearest new class-`11` representation bottleneck analogue and `baiyangdian_new_area_holdout` as a possible forecast-suppression case,
- remember that Batch 5-only passes with `primary ci_low = 0.03874008044819379`, `spatial ci_low = 0.026682096295538787`, and `positive_tier1_strata = 4`,
- remember that Batch 5 still contains important within-batch risk: `liaohe_delta_wetland_holdout` is negative on primary and spatial advantage, and `wenan_lakeplain_newtown_holdout` is negative on primary advantage,
- remember that the Batch 5 Liaohe diagnosis finds both spatial-localization weakness and class-`11` representation/decoder weakness: best shift improves F1 to `0.3942307692307692`, but dominant `5->11` shifted semantic matches remain only `1`,
- use `liaohe_delta_wetland_holdout` as the first Batch 5 experimental follow-up target if continuing experiments, because it is a negative wetland row with `62` true change pixels and a real class-`11` bottleneck,
- continue with stronger and more diverse experiments first rather than shifting attention to the manuscript.
