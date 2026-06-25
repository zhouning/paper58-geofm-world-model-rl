# Current Work Progress: 2026-06-25

## Paper58 Resume State

- Repository: `/Users/zhouning/paper58-geofm-world-model-rl`
- Active worktree: `/Users/zhouning/paper58-geofm-world-model-rl/.worktrees/paper58-benchmark`
- Active branch: `paper58-benchmark`
- Resume code state from commit: `71364c0 feat: add adaptive demand gate for LAS`
- Working tree currently contains only untracked experiment output directories under `paper/rse_submission_paper58/`; do not add them to git.

## Latest Completed Work

The Paper58-LAS route was extended with a new demand arbitration layer:

- added `transition_prior_adaptive_blend` in `scripts/paper58_benchmark/las_demand.py`
- wired the new demand gate through `scripts/paper58_benchmark/evaluate_las.py`
- exposed the same option in `scripts/paper58_benchmark/run_flus_batch.py`
- added unit tests in:
  - `tests/test_paper58_las_demand.py`
  - `tests/test_paper58_las_evaluation.py`
  - `tests/test_paper58_flus_batch.py`
- updated the RSE comparison report in:
  - `paper/rse_submission_paper58/paper58_geosos_flus_comparison_report_2026-06-24.md`

## Current Verified Results

Most recent verification:

```text
54 passed
git diff --check clean
```

Main Batch 5 results with the adaptive demand gate:

- F1 advantage: `+0.1986`
- F1 CI low: `+0.0856`
- FoM advantage: `+0.0745`
- FoM CI low: `+0.0330`
- recall advantage: `+0.3908`
- transition accuracy advantage: `+0.2425`
- allocation disagreement advantage: `-0.0432`

FoM-first frontier:

- FoM advantage: `+0.0805`
- FoM CI low: `+0.0387`
- transition accuracy advantage: `+0.2690`
- allocation disagreement advantage: `-0.0634`

Leave-one-area-out candidate selection audit:

- candidate pool included fixed neighborhood settings, adaptive neighborhood settings, and the new adaptive demand candidates
- all 7 held-out areas selected the adaptive demand F1 frontier
- LOAO held-out F1 advantage remained `+0.1986`
- LOAO held-out FoM advantage remained `+0.0745`

## What To Keep Visible

- Do not overclaim full GeoSOS-FLUS native workflow victory.
- The strongest proven claim is against the matched official FLUS console baseline.
- `huaibei_irrigation_plain_holdout` is still the main weak case.
- Allocation disagreement is still the main gap to close.

## Recommended Next Step

Continue from the current adaptive-demand frontier and decide whether the next effort should be:

1. further reduce allocation disagreement and diagnose `huaibei_irrigation_plain_holdout`, or
2. start the stronger native GeoSOS-FLUS comparison track if the user wants the broader claim.

## Resume Hint

If you reopen the project in a new window, start from:

```text
git -C /Users/zhouning/paper58-geofm-world-model-rl/.worktrees/paper58-benchmark status --short --branch
git -C /Users/zhouning/paper58-geofm-world-model-rl/.worktrees/paper58-benchmark log -5 --oneline --decorate
```
