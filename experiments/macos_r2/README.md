# Paper 58 RSE Round 2 — macOS Experiment Package

This directory contains **all the experiments the RSE reviewer flagged as
Major (M4) / Substantive (S3, S4, S5, S6) issues** that the Windows machine
could not run in-scope. The package is designed for one-shot execution on
the macOS workstation, with each experiment producing self-describing JSON
+ CSV outputs that Windows-side revision v3 can consume directly.

## Contents

| Experiment | Reviewer issue | What it produces | Est.\ wall-clock |
|---|---|---|---|
| **E1** — Prithvi patch/token ablation | M4 | `results/e1_prithvi_patch/*.{json,csv}` — 17-area × 3-seed LDN trained on Prithvi spatial tokens `(H, W, 768)`; paired-inference vs.\ AlphaEarth LDN | 4–6h (GEE) + 20–40 min (train) |
| **E2** — Terrain-context ablation | S4 | `results/e2_terrain_ablation/*.{json,csv}` — three configs (with-terrain, no-slope, no-context) × 3 seeds on the AlphaEarth cache | ~3h |
| **E3** — Multi-step degradation on all valid areas | S5 | `results/e3_multistep/multistep_all_areas.csv` — 6-step rollout for every cached area (fills the 7 blanks in v2 Table 8) | ~30 min |
| **E4** — Per-year decoder retraining | S6 | `results/e4_per_year_decoder/decoder_by_year.csv` — decoder trained per year (2017–2023), re-runs 11 independent-change validation pairs | ~45 min |
| **E5** — SA-Alloc sensitivity sweep | S3 | `results/e5_sa_alloc_sensitivity/*.csv` — 4×3×3 grid over threshold / small-mul / large-mul, reports 4 metrics × 24 townships | 1–2h |
| **E6** — Expand to 30 areas | M1 (main) | `results/e6_expanded_areas/*.csv` — 13 new AlphaEarth cached grids, re-run paired inference at $n=23$ | ~5h (GEE) |
| **E7** — Third encoder (SatMAE / Clay) | M4 addendum | `results/e7_third_encoder/*.csv` — SatMAE branch trained + evaluated the same way as AlphaEarth and Prithvi | ~half-day |

## One-shot execution

Assuming `farmland-mpc-pure` conda env is active, GEE is authenticated,
and Prithvi-100M weights are at `~/paper58-r2/weights/Prithvi_100M.pt`:

```bash
cd ~/paper58-geofm-world-model-rl                       # git pull recent v2
git pull origin main
cd experiments/macos_r2
bash setup_macos_env.sh                                  # env self-check + missing deps
bash run_all_macos.sh --all                              # runs E1 -> E7 sequentially
```

Individual experiments (any order, each idempotent):

```bash
bash run_all_macos.sh --only e1                          # Prithvi patch ablation
bash run_all_macos.sh --only e2                          # terrain ablation
bash run_all_macos.sh --only e2,e3,e4,e5                 # no-GEE fast path
bash run_all_macos.sh --resume                           # resume after Ctrl-C or crash
```

## Output convention

Every experiment writes into `results/{eN_name}/` with three files:

- `summary.json` — headline numbers Windows-side v3 needs (means, per-area breakdowns, $p$-values, wall-clock, git SHA).
- `raw.csv` — every trial-level row (for reviewer audit).
- `run_manifest.json` — configuration + environment fingerprint (Python + torch + platform + args + start/end timestamps).

## Result upload

Once done, from macOS:

```bash
git add experiments/macos_r2/results/
git commit -m "Paper 58 RSE R2 — macOS-side experiments E1-E7 complete"
git push origin main
```

Then on Windows side I pull, replace the placeholder tables in v3 tex, and
recompile.

## Failure mode / abort-safe

Each experiment writes a `.done` sentinel file when successful. `run_all_macos.sh --resume`
re-scans and only runs experiments without a `.done` file. Partial JSON
files are safe to overwrite.

## Dependency versions we assume

Same as `farmland-mpc-pure`:
- Python 3.11
- PyTorch >= 2.2 (Apple Silicon MPS OK; scripts auto-fall-back to CPU)
- earthengine-api (already authenticated)
- geoadapter (for PrithviBackbone) — installed from `external/geoadapter/`
- numpy, pandas, scipy, scikit-learn, torch, tqdm

Any missing deps are captured by `setup_macos_env.sh`.
