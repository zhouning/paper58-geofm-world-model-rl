# macOS Rerun Results — 2026-07-02

## Summary

Pulled Windows-side fixes through commit `94f3080` and reran the macOS-side experiments requested in `E3_READY_TO_RERUN.md`, `E4_PREDICTED_EMBEDDINGS_COMPLETE.md`, and `E6_DATA_SOURCE_FIX.md`.

## E3 Multi-Step Rollout

- Code path: `experiments/macos_r2/e3_multistep_all_areas.py`
- Result files:
  - `results/e3_multistep/multistep_all_areas.csv`
  - `results/e3_multistep/multistep_summary.json`
  - `results/e3_multistep/multistep_paired_tests.json`
- Coverage: 30 complete 2017-2024 AlphaEarth areas, 6 rollout steps, 180 rows.
- Mean advantage by step:
  - step 1: -0.01248 (3/30 positive)
  - step 2: -0.03599 (0/30 positive)
  - step 3: -0.06860 (0/30 positive)
  - step 4: -0.10320 (0/30 positive)
  - step 5: -0.14685 (0/30 positive)
  - step 6: -0.18534 (0/30 positive)

## E4 Per-Year Decoder

- Code path: `experiments/macos_r2/e4_per_year_decoder.py`
- Result files:
  - `results/e4_per_year_decoder/decoder_by_year.csv`
  - `results/e4_per_year_decoder/per_pair_end_accuracy_delta.csv`
  - `results/e4_per_year_decoder/training_manifest.json`
- Trained decoders: 2017-2024 inclusive.
- Independent-change pairs: 11 total.
- Re-evaluable pairs: 9/11.
- Excluded pairs:
  - `banzhucun_2017-2020`: shape mismatch, predicted embedding 8x13 vs label 363x610.
  - `banzhucun_2020-2023`: shape mismatch, predicted embedding 8x13 vs label 363x610.
- Valid pair deltas: 8 positive, 1 negative.
- New generated valid artifact: `data/independent_change_labels/predicted/heping_2017_2020_embedding.npy`.

## E6 Expanded Areas

- Command: `python e6_expand_areas.py --eval-only --min-years 8`
- Result files:
  - `results/e6_expanded_areas/expanded_per_area.csv`
  - `results/e6_expanded_areas/expanded_paired_tests.json`
  - `results/e6_expanded_areas/eval_area_sources.json`
- Coverage: 30 areas from `experiments/paper8/data`, all with complete 2017-2024 sequences.
- Mean advantage: -0.01493.
- Positive/negative split: 3/27.
- Wilcoxon p: 2.5518e-07.

## Verification

- `git diff --check`
- `/Users/zhouning/miniconda3/envs/farmland-mpc/bin/python -m pytest tests/test_macos_r2_label_prediction_paths.py tests/test_macos_r2_orchestrator.py tests/test_macos_r2_e3_checkpoint.py tests/test_macos_r2_e1_e6.py -q`
- Result: 25 passed.
