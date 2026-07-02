# Paper 58 RSE v3-to-v4 Diff Report

Date: 2026-07-02
Base manuscript: `rse_geofm_world_model_rl_v3.tex`
Updated manuscript: `rse_geofm_world_model_rl_v4.tex`
Compiled PDF: `rse_geofm_world_model_rl_v4.pdf`
Integration commit: `bc0c64e Integrate macOS R2 results into v4 manuscript`

`latexdiff` is not installed on this macOS machine, so this report records the manual v3-to-v4 content diff and the exact result files used for the integration.

## Summary

v4 incorporates the completed macOS R2 E3, E4, and E6 results. The revision strengthens the manuscript boundary: GeoFM-LDN is not presented as outperforming persistence in embedding-space forecasting. Instead, v4 reports a statistically stable negative 30-area baseline, a negative corrected 6-step rollout, and a per-year decoder follow-up that improves decoded end-year accuracy but does not overturn the full-map forecasting verdict.

## Main Text Changes

1. Abstract
   - Replaced the v3 10-area non-significant baseline with the v4 30-area complete-time-series baseline.
   - Added E3 corrected multi-step rollout result.
   - Added E4 per-year decoder retraining result.
   - Updated follow-up status from two completed experiments to five completed experiments.

2. Methodology: LULC decoder
   - Replaced "decoder time stability (open issue)" with completed E4 follow-up wording.
   - Clarified that annual decoders were trained for 2017--2024 and only grid-matched validation pairs were re-decoded.

3. Results: area-level embedding dynamics
   - Expanded Table `tab:aggregate` from 10 to 30 areas.
   - Updated Table `tab:paired_inference` with the 30-area negative paired tests.
   - Expanded Table `tab:perarea` to all 30 complete-time-series areas.
   - Removed the legacy 10-area area-performance figure from v4 to avoid conflicting with the 30-area table.

4. Results: E4 per-year decoder
   - Added subsection `Per-year decoder retraining`.
   - Added Table `tab:decoder_by_year`.
   - Reported the paired end-year accuracy delta summary and the two excluded Banzhucun grid-mismatch cases.

5. Results: E3 multi-step rollout
   - Replaced the v3 "only three areas have full rollout" disclosure with the corrected 30-area 6-step rollout table.
   - Reframed E3 as a failure-mode result rather than a long-horizon advantage.

6. Discussion and limitations
   - Replaced the v3 "10-area underpowered/non-significant" interpretation with the v4 30-area negative interpretation.
   - Updated the decoder limitation from "not established" to "partially addressed, not eliminated".
   - Updated the terrain/receptive-field limitation to reflect the completed terrain-context ablation.

7. Follow-up experiment status
   - Moved E3, E4, and E6 into completed follow-up experiments.
   - Kept E1 as diagnosed and E7 as optional/deferred.

8. Conclusions and Data Availability
   - Updated conclusions to state five completed reviewer follow-ups.
   - Added macOS R2 E3/E4/E6 output files and `v4_manuscript_numbers.json` to Data and Code Availability.

## Key Numerical Changes

| Item | v3 | v4 |
|---|---:|---:|
| Area-level baseline sample | 10 areas | 30 complete-time-series areas |
| Mean Model - Persistence | -0.0055 | -0.014934 |
| Positive / negative areas | 2 / 8 | 3 / 27 |
| Wilcoxon p | >0.05 | 2.55e-7 |
| Paired t-test p | >0.05 | 1.19e-4 |
| E3 step-1 advantage | incomplete disclosure | -0.0125 over 30 areas |
| E3 step-6 advantage | incomplete disclosure | -0.1853 over 30 areas |
| E4 per-year decoder delta | not run | +0.0778 mean over 9 re-evaluable pairs |
| E4 improved pairs | not run | 8 / 9 |

## Source Files Used

- `experiments/macos_r2/results/e3_multistep/multistep_all_areas.csv`
- `experiments/macos_r2/results/e3_multistep/multistep_summary.json`
- `experiments/macos_r2/results/e3_multistep/multistep_paired_tests.json`
- `experiments/macos_r2/results/e4_per_year_decoder/decoder_by_year.csv`
- `experiments/macos_r2/results/e4_per_year_decoder/per_pair_end_accuracy_delta.csv`
- `experiments/macos_r2/results/e6_expanded_areas/expanded_per_area.csv`
- `experiments/macos_r2/results/e6_expanded_areas/expanded_paired_tests.json`
- `experiments/macos_r2/v4_manuscript_numbers.json`

## Verification

- `python -m pytest tests/test_macos_r2_extract_v4_numbers.py tests/test_macos_r2_label_prediction_paths.py tests/test_macos_r2_orchestrator.py tests/test_macos_r2_e3_checkpoint.py tests/test_macos_r2_e1_e6.py -q`
  - Result: 27 passed.
- `python experiments/macos_r2/extract_v4_numbers.py`
  - Result: regenerated `v4_manuscript_numbers.json`.
- `git diff --check`
  - Result: clean.
- `pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper/rse_submission_paper58/manuscript paper/rse_submission_paper58/manuscript/rse_geofm_world_model_rl_v4.tex`
  - Run twice.
  - Result: compiled `rse_geofm_world_model_rl_v4.pdf`, 49 pages.
