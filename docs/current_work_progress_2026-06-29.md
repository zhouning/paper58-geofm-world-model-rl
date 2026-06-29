# Current Work Progress - 2026-06-29

## Repository state

- Synced `main` with `origin/main` at `0985e182a73c96503b29dbe5667ac54d3da1b727`.
- Set local Git config `core.longpaths=true` so the Windows checkout can handle the long Paper58 result paths added upstream.
- Worktree was clean before this progress note was written.
- Pre-sync local edits were preserved in local stash `stash@{0}` with message `codex-sync-paper58-before-origin-main-2026-06-29`.
- The stash touches:
  - `scripts/rse_revision/fetch_change_validation_embeddings.py`
  - `scripts/rse_revision/fetch_independent_lulc_labels.py`
  - `scripts/rse_revision/generate_change_validation_predictions.py`
  - `tests/test_rse_revision_change_validation.py`
- The stash was not reapplied after sync because it conflicts with the updated `origin/main`. Keep it for later review instead of silently overwriting the new upstream implementation.

## Paper58 vs GeoSOS-FLUS status

Latest evidence is from the 2026-06-27 same-grid benchmark with 24 real township samples and the fixed `geosos_flus_console` baseline.

Overall mean metrics:

| Method | Change F1 | FoM | Transition accuracy | Allocation disagreement |
| --- | ---: | ---: | ---: | ---: |
| `geosos_flus_console` | 0.2635 | 0.1196 | 0.2808 | 0.0601 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | 0.2883 | 0.1248 | 0.2862 | 0.0584 |

Seeded GeoSOS-FLUS repeat evidence:

- Paper58 wins `5/5` seeds on Change F1.
- Paper58 wins `5/5` seeds on FoM.
- Paper58 wins `5/5` seeds on transition accuracy.
- Paper58 has lower mean allocation disagreement, but wins only `3/5` seeds on that metric.

Area-by-seed paired evidence is mixed:

- Change F1: `65/120` wins.
- FoM: `61/120` wins.
- Transition accuracy: `54/120` wins.
- Allocation disagreement: `81/120` wins.

Safe manuscript claim:

> Under strict non-target tuning on the 24 real-township same-grid target set, the transition-aware Paper58 spatial-demand ratio variant surpasses the fixed GeoSOS-FLUS console baseline in aggregate mean metrics.

Do not claim yet:

> Paper58 fully surpasses GeoSOS-FLUS in every area, every local transition case, or the complete native GeoSOS-FLUS GUI/traditional-driver workflow.

## Verification

- `python -m pytest tests/test_rse_revision_change_validation.py -q`
- Result: `19 passed`, with one non-fatal pytest cache warning about an existing `.pytest_cache` path.
