# Paper 5+8 × Paper 12 Encoder Ablation

Plan file: `C:/Users/zn198/.claude/plans/sparkling-pondering-pretzel.md`

## What this is

An ablation that replaces Google AlphaEarth (480M params, 64-dim L2-normed, multi-modal) with Prithvi-100M (100M params, 768-dim RGB+NIR+SWIR) as the frozen encoder in Paper 5+8's JEPA-style geospatial world model, then retrains the LatentDynamicsNet at the new dim. Goal: a single new row in Paper 5+8 Table 1 demonstrating that AlphaEarth's advantage isn't just parameter count.

**Not** a replacement of AlphaEarth in the main paper. See plan §Out of scope.

## Files

| File | Phase | Purpose |
|---|---|---|
| `extract_prithvi_embeddings.py` | A | GEE HLS → Prithvi forward → L2-norm → `[H,W,768]` npy |
| `colab_extract_prithvi.py` | A | Colab driver (mirrors `colab_paper8.py` convention) |
| `compare_encoders.py` | B | Static comparison (persistence cossim + LULC linear probe), no LDN |
| `train_prithvi_ldn.py` | C | Retrain LDN at `z_dim=768`, 17 areas × 3 seeds |
| `eval_prithvi_vs_alphaearth.py` | D | Aggregate metrics → `results/paper8_ablation_encoder.json` |

## Run order

```bash
# Phase A (Colab, ~4h on L4):
#   1. Local: zip extract_prithvi_embeddings.py + geoadapter/ + Prithvi_100M.pt
#   2. Upload to Drive as paper58_ablation.zip
#   3. In Colab cell: !python colab_extract_prithvi.py --smoke   (sanity: Bishan 2020)
#   4. Then:           !python colab_extract_prithvi.py --all   (full 17 × 8)
#   5. Sync Drive output back to D:/test/paper8/data/prithvi/

# Phase B (local, ~5 min):
python paper8/compare_encoders.py

# Phase C (local CPU, ~25h) — start before Phase D:
python paper8/train_prithvi_ldn.py --smoke              # 5-epoch smoke first
python paper8/train_prithvi_ldn.py --epochs 50          # full

# Phase D (local, ~5 min):
python paper8/eval_prithvi_vs_alphaearth.py
```

## Risks

- **HLS cloud cover** in some China areas → may drop area-years; `metadata.json` records coverage
- **Cross-encoder cossim absolute values not comparable** (768-d ≠ 64-d statistics); use *advantage over persistence* as the apples-to-apples metric
- **L2-normalizing Prithvi at extraction** is a design choice — acknowledged in paper text
- **Banzhucun is 10m**, Prithvi was trained at 30m → resampling loses info; included with caveat

## After Paper 12 demo (2026-06-09)

This ablation is **scheduled to start after** the Paper 12 demo. Don't compete with Paper 12 Colab budget. ETA 1.5-2 weeks once started.
