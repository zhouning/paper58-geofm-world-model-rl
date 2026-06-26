# Current Work Progress: 2026-06-25

## Paper58 Resume State

- Repository: `/Users/zhouning/paper58-geofm-world-model-rl`
- Active worktree: `/Users/zhouning/paper58-geofm-world-model-rl/.worktrees/paper58-benchmark`
- Active branch: `paper58-benchmark`
- Resume code state from commit: `71364c0 feat: add adaptive demand gate for LAS`
- Working tree contains tracked benchmark-code/test/doc changes plus untracked experiment output directories under `paper/rse_submission_paper58/`; do not add the output directories to git.

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
target LAS tests: 83 passed
full test suite: 236 passed
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

New budget sweep around the adaptive-demand frontier:

- best weak-case-stabilized candidate: `las_results_batch5_adaptive_demand_grid_w0.75_l1_0.14_change_0.22_neigh_adapt_budget_scale_0.82`
- mean F1 advantage: `+0.1893`
- mean F1 bootstrap CI: `[+0.0854, +0.3052]`
- mean FoM advantage: `+0.0979`
- mean FoM bootstrap CI: `[+0.0380, +0.1633]`
- paired sign test for mean F1: `7/7` positive
- `huaibei_irrigation_plain_holdout` F1 advantage: `+0.0011`
- `huaibei_irrigation_plain_holdout` predicted changes: `67` vs FLUS `54`
- note: this candidate is not the pure LOAO F1 frontier; it trades some mean F1 for removing the `huaibei` F1 loss and increasing mean FoM.

Adaptive change-budget gate:

- implemented `select_change_budget_scale` in `scripts/paper58_benchmark/las_demand.py`
- wired `--adaptive-change-budget-scale`, `--adaptive-change-budget-fraction-low`, and `--adaptive-change-budget-fraction-high` through `scripts/paper58_benchmark/evaluate_las.py`
- current tested setting: base budget scale `0.82`, adaptive scale `0.85`, low/high raw change-fraction thresholds `0.13/0.30`
- matched FLUS comparison output: `las_results_batch5_adaptive_demand_grid_w0.75_l1_0.14_change_0.22_neigh_adapt_budget_gate_low0.13_high0.30_scale_0.82_0.85_with_flus`
- mean F1 advantage: `+0.2049`
- mean F1 bootstrap CI: `[+0.0902, +0.3285]`
- paired sign test for F1: `7/7` positive, two-sided `p=0.015625`
- mean FoM advantage: `+0.0802`
- mean FoM bootstrap CI: `[+0.0355, +0.1315]`
- mean recall advantage: `+0.3954`
- mean transition accuracy advantage: `+0.2495`
- allocation disagreement advantage: `-0.0392`
- same-registry comparison against the `0.82` rerun shows the gate improves `wuxi_taihu_dense_edge_holdout` F1 (`+0.0405`) without changing FoM; all other matched rows remain unchanged under this gate.

Leave-one-area-out candidate selection audit:

- candidate pool included fixed neighborhood settings, adaptive neighborhood settings, and the new adaptive demand candidates
- all 7 held-out areas selected the adaptive demand F1 frontier
- LOAO held-out F1 advantage remained `+0.1986`
- LOAO held-out FoM advantage remained `+0.0745`
- adding the new `0.82` budget-scale candidate to a two-candidate LOAO audit (`adaptive085` vs `adaptive082`, F1 primary with FoM tie-break) still selected `adaptive085` for all 7 held-out areas
- adding the new adaptive change-budget gate to a three-candidate same-registry LOAO audit (`adaptive082`, `adaptive085`, `adaptive_budget_gate`, F1 primary with FoM tie-break) selected `adaptive_budget_gate` for all 7 held-out areas
- same-registry LOAO held-out F1 advantage: `+0.2049`
- same-registry LOAO held-out FoM advantage: `+0.0802`

Methodology hardening:

- added explicit `candidate_priority` support to `scripts/paper58_benchmark/select_las_candidate_loo.py`
- reason: exact training-score ties must not be resolved by candidate name ordering, because that makes LOAO selection sensitive to arbitrary labels rather than a pre-registered priority
- new explicit-priority audit over `adaptive_budget_gate`, fresh `fixed082`, and stricter balanced-swap base-score gates selected `adaptive_budget_gate` for all 7 held-out areas
- explicit-priority LOAO held-out F1 advantage: `+0.2049`
- explicit-priority LOAO held-out FoM advantage: `+0.0802`
- strict balanced-swap base-score gates were tested but rejected:
  - base-score floor `1.0`: no change relative to current gate
  - base-score floors `1.5`, `2.0`, and `2.5`: mean F1 fell to `+0.1146`, mean FoM fell to `+0.0511`
- current evidence does not support tightening balanced swaps as a global algorithm change.

Balanced-swap per-side base-score gate audit:

- added an optional experimental `balanced_swap_min_side_base_score` argument/CLI flag so a balanced swap can require both directions to have a minimum raw Paper58/GeoFM suitability before neighborhood terms are added.
- tested side-score floors `0.02`, `0.05`, `0.10`, and `0.20` under the current adaptive-budget frontier settings.
- side-score floor `0.02` remained above FLUS but weakened the frontier:
  - mean F1 advantage: `+0.1357`, bootstrap CI low `+0.0508`
  - mean FoM advantage: `+0.0520`, bootstrap CI low `+0.0120`
  - allocation disagreement advantage improved from `-0.0392` to `-0.0268`, but this came from lost true-change recall/F1 rather than better spatial placement.
- side-score floors `0.05`, `0.10`, and `0.20` collapsed to the same weaker result:
  - mean F1 advantage: `+0.1146`, bootstrap CI low `+0.0329`
  - mean FoM advantage: `+0.0511`, bootstrap CI low `+0.0107`
  - allocation disagreement advantage improved to `-0.0232`, again by pruning changes rather than improving placement.
- region-level deltas versus the current adaptive-budget gate show why this is not a real model improvement:
  - `liaohe_delta_wetland_holdout`: F1 advantage dropped by `-0.4435`, FoM advantage by `-0.1091`
  - `wuxi_taihu_dense_edge_holdout`: floor `0.05` dropped F1 advantage by `-0.1296`
  - `huaibei_irrigation_plain_holdout`: no F1/FoM improvement
  - `wenan_lakeplain_newtown_holdout`: allocation disagreement improved, but F1/FoM both decreased
- explicit-priority LOAO over `adaptive_budget_gate`, `side_base_002`, `side_base_005`, `side_base_010`, and `side_base_020` selected `adaptive_budget_gate` for all 7 held-out areas.
- conclusion: keep the per-side base-score floor only as an experimental diagnostic switch, not as the selected Paper58-LAS model. The current best scientifically defensible candidate remains the adaptive change-budget gate.

Spatial evidence gating audit:

- measured reciprocal churn in the Paper58 prediction relative to start for each Batch 5 tier-1 holdout.
- high churn areas are the same ones where LAS still over-samples reciprocal swaps, especially `huaibei_irrigation_plain_holdout`, `wenan_lakeplain_newtown_holdout`, and `renqiu_baiyangdian_edge_holdout`.
- added an optional `adaptive_churn_budget_scale` / `adaptive_churn_fraction_high` gate that lowers the gross change budget when the Paper58 prediction has a high reciprocal swap fraction.
- strongest tested churn setting:
  - `adaptive_churn_budget_scale=0.65`
  - `adaptive_churn_fraction_high=0.70`
- this setting slightly exceeded the current adaptive-budget gate on mean F1:
  - mean F1 advantage: `+0.2050`
  - F1 bootstrap CI low: `+0.0905`
- but it did not improve the broader frontier:
  - mean FoM advantage: `+0.0799`, slightly below the current `+0.0802`
  - LOAO F1-primary selected it only on 5/7 folds, not all 7
  - LOAO FoM-primary still did not favor it as the overall winner
- another latent-neighborhood candidate (`latent010`) remained competitive on F1/FoM and `latent010_pressure005` improved some regions, but none displaced `adaptive_budget_gate` under both F1-primary and FoM-primary LOAO audits.
- conclusion: churn-aware gating and latent-neighborhood evidence are useful diagnostics, but neither has yet beaten the current adaptive change-budget gate under the full scientific selection protocol.

Suitability-weight audit:

- exposed optional suitability score weights in `evaluate_las.py`:
  - `suitability_forecast_prob_weight`
  - `suitability_probability_gain_weight`
  - `suitability_transition_prior_weight`
  - `suitability_change_pressure_weight`
- reason: the current LAS suitability uses near-hard Paper58 one-hot probabilities (`0.95/0.01`), which can amplify reciprocal churn when Paper58 directly predicts many `5<->7` swaps.
- tested reduced hard-label weighting with stronger transition-prior evidence:
  - `forecast_prob_weight=0.50`
  - `probability_gain_weight=0.25`
  - `transition_prior_weight=0.75`
- best whole-sample setting with churn gate reached:
  - mean F1 advantage: `+0.2059`
  - mean FoM advantage: `+0.0793`
- same setting without churn gate reached:
  - mean F1 advantage: `+0.2057`
  - mean FoM advantage: `+0.0796`
- however, explicit-priority LOAO over the current gate, hard-label-reduced suitability candidates, churn candidate, and latent candidate fell to:
  - held-out F1 advantage: `+0.1944`
  - held-out FoM advantage: `+0.0742`
- conclusion: hard-label de-emphasis can improve the all-sample F1 mean, but it is not yet a scientifically defensible replacement because it fails LOAO generalization.

Fresh diagnostic findings:

- same current-code rerun with and without FLUS paths generated identical LAS maps for all 7 areas, so the current pipeline does not use baseline outputs to generate LAS predictions.
- spatial error decomposition shows `huaibei_irrigation_plain_holdout` and `wenan_lakeplain_newtown_holdout` are dominated by many `5<->7` balanced swaps:
  - `huaibei`: LAS predicted `67` changes for `14` true changes; exact transition TP `5`, FP `61`
  - `wenan`: LAS predicted `82` changes for `10` true changes; exact transition TP `4`, FP `78`
- simple lower-budget settings reduce allocation disagreement but hurt `huaibei` F1/FoM and lower the overall F1/FoM frontier, so the next useful model work should target spatial placement quality rather than only reducing gross change budget.

## What To Keep Visible

- Do not overclaim full GeoSOS-FLUS native workflow victory.
- The strongest proven claim is against the matched official FLUS console baseline.
- `huaibei_irrigation_plain_holdout` is no longer an F1 weak case under the `0.82` budget-stabilized candidate and the new adaptive budget gate, but its FoM is still below FLUS.
- Allocation disagreement is still the main gap to close.
- Avoid post-hoc per-area tuning. New candidates should be evaluated through same-registry matched comparison plus LOAO selection with explicit tie-breaking.
- Do not present scalar balanced-swap tightening as a model advance unless it improves F1/FoM under the same LOAO protocol. The latest side-base gate improved allocation disagreement only by suppressing changes and was rejected.
- Do not promote all-sample F1 gains from churn or suitability-weight sweeps unless LOAO also improves. The current all-sample best candidates are not stable enough to replace `adaptive_budget_gate`.

## Recommended Next Step

Continue from the current adaptive-demand frontier and decide whether the next effort should be:

1. continue targeting reciprocal-swap churn and spatial placement jointly, especially `5<->7` swaps in `huaibei` and `wenan`, using stronger Paper58 suitability/transition-prior/embedding evidence and LOAO validation rather than scalar pruning thresholds, or
2. start the stronger native GeoSOS-FLUS comparison track if the user wants the broader claim.

## Resume Hint

If you reopen the project in a new window, start from:

```text
git -C /Users/zhouning/paper58-geofm-world-model-rl/.worktrees/paper58-benchmark status --short --branch
git -C /Users/zhouning/paper58-geofm-world-model-rl/.worktrees/paper58-benchmark log -5 --oneline --decorate
```
