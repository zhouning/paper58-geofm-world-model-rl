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

## Next Valid Step

## Resume Instruction

In a new window, continue from branch `paper58-benchmark`.

The expanded strict Tier 1 gate has passed, so the next manuscript-facing step may be a separate revision plan that maps every proposed claim to the benchmark outputs. Do not directly strengthen the manuscript prose without that claim-to-evidence mapping.
