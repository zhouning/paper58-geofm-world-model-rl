# Paper58 Batch 5 Liaohe Diagnostics Design

## Purpose

Batch 5 passed the mixed external full gate, but `liaohe_delta_wetland_holdout` is negative on both primary and spatial change advantage. This diagnostic follows the experiment-first rule: it explains the negative wetland row without using the Batch 5 pass to erase the Batch 2 failure.

## Scope

The diagnostic targets `liaohe_delta_wetland_holdout` first, keeps `wenan_lakeplain_newtown_holdout` as a secondary weak-primary risk, and compares Liaohe against prior evaluated wetland rows:

- `erlong_lake_margin_holdout`
- `honghu_lake_margin_holdout`
- `zhalong_wetland_edge_holdout`

## Outputs

Write outputs under:

```text
paper/rse_submission_paper58/diagnostics_batch5_liaohe
```

Required outputs:

- spatial best-shift table for the focus and comparison rows,
- embedding-decoder audit table,
- transition count and transition-fate aggregate tables,
- shifted-transition fate aggregate table using each row's best shift,
- observed and forecast true-end confidence tables,
- JSON and text summary with Liaohe headline metrics and WEnan risk retained.

## Interpretation Rules

- Do not change benchmark thresholds.
- Do not change the dynamics model.
- Do not edit the manuscript.
- Do not pool Batch 5 with earlier batches to hide Batch 2 or Batch 5 row-level failures.
- Treat any Liaohe diagnosis as a follow-up experiment result, not as manuscript-strengthening evidence.
