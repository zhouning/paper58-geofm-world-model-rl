# macOS 端 R2 剩余实验执行清单

**目标**: 完成 E3、E4、E6 三个实验，将结果 push 到 GitHub，Windows 端整合到 v4  
**预计时间**: ~1 小时（大部分是 E6 的 30-area GEE 提取时间）  
**日期**: 2026-07-02

---

## Step 0: 拉取最新代码

```bash
cd ~/paper58-geofm-world-model-rl
git pull origin main
```

**确认**:
- 应该看到最新 commit `94f3080` "Add E4 predicted embeddings"
- 应该有以下新文件：
  - `experiments/macos_r2/generate_e4_predicted_embeddings.py`
  - `experiments/macos_r2/E1_DIAGNOSIS_COMPLETE.md`
  - `experiments/macos_r2/E3_READY_TO_RERUN.md`
  - `experiments/macos_r2/E4_PREDICTED_EMBEDDINGS_COMPLETE.md`
  - `experiments/macos_r2/E6_DATA_SOURCE_FIX.md`
  - `data/independent_change_labels/predicted/*_embedding.npy` (61 files)

---

## Step 1: 运行 E3（Multi-Step Rollout）

**Bug 已修复**：现在正确使用 z_{step-1} vs z_step 计算 persistence。

```bash
cd ~/paper58-geofm-world-model-rl/experiments/macos_r2
python e3_multistep_all_areas.py
```

**预计时间**: ~30 分钟  
**预期输出**: `experiments/macos_r2/results/e3_multistep/multistep_all_areas.csv`

**如何验证 bug 已修复**:
```bash
head -20 results/e3_multistep/multistep_all_areas.csv
```

预期 step=1 时 persistence 应该在 0.95-0.99 之间（不是之前的 1.000）：
```csv
area,step,persistence,model,advantage
bishan,1,0.9863,0.9854,-0.0009
bishan,2,0.9781,0.9719,-0.0062
...
```

---

## Step 2: 运行 E4（Per-Year Decoder）

**数据已就绪**：Windows 端已生成 61 pairs 的 predicted embeddings。

```bash
cd ~/paper58-geofm-world-model-rl/experiments/macos_r2
python e4_per_year_decoder.py
```

**预计时间**: ~15 分钟  
**预期输出**:
- `experiments/macos_r2/results/e4_per_year_decoder/decoder_by_year.csv` — 每年 decoder CV accuracy
- `experiments/macos_r2/results/e4_per_year_decoder/per_pair_end_accuracy_delta.csv` — 每 pair 准确率提升

**如何验证成功**:
```bash
cat results/e4_per_year_decoder/per_pair_end_accuracy_delta.csv | head -10
```

应该看到实际数值（而不是全部 "prediction_missing"）：
```csv
pair_id,end_year,v2_end_accuracy,retrained_end_accuracy,delta,status
bishan_2017_2018,2018,0.6605,0.6812,+0.0207,ok
bishan_2018_2019,2019,0.6906,0.7103,+0.0197,ok
...
```

---

## Step 3: 运行 E6（30-Area Baseline，Complete Time Series Only）

**数据源过滤已实现**：只用 8 年完整时间序列的 area。

```bash
cd ~/paper58-geofm-world-model-rl/experiments/macos_r2
python e6_expand_areas.py --eval-only --min-years 8
```

**预计时间**: ~5 分钟（只 eval，不 extract）  
**预期输出**:
- `experiments/macos_r2/results/e6_expanded_areas/expanded_paired_tests.json`
- `experiments/macos_r2/results/e6_expanded_areas/expanded_per_area.csv`
- `experiments/macos_r2/results/e6_expanded_areas/eval_area_sources.json`

**如何验证过滤生效**:
```bash
cat results/e6_expanded_areas/expanded_paired_tests.json
```

应该看到 `n: 30`（不是之前的 73）：
```json
{
  "n": 30,
  "mean": -XXX,
  "eval_sources": {
    "n_areas": 30,
    "roots": {
      ".../experiments/paper8/data": 30
    }
  }
}
```

---

## Step 4: 提交结果到 GitHub

```bash
cd ~/paper58-geofm-world-model-rl
git add experiments/macos_r2/results/e3_multistep/
git add experiments/macos_r2/results/e4_per_year_decoder/
git add experiments/macos_r2/results/e6_expanded_areas/
git commit -m "R2 macOS: complete E3/E4/E6 with fixes

- E3: Multi-step rollout with corrected persistence calculation
- E4: Per-year decoder evaluation with Windows-generated predictions
- E6: 30-area baseline (min_years=8, complete time series only)"
git push origin main
```

---

## 完成后通知 Windows

Push 完成后，告诉我"macOS E3/E4/E6 完成"，我这边会：
1. Pull 结果到 Windows
2. 整合到 v4 manuscript
3. 更新 abstract、results tables、discussion
4. 编译 v4 PDF
5. 生成 v4 diff report

---

## 故障排除

### 如果 E3 报错

**症状**: `AttributeError` 或 `RuntimeError` 关于 tensor 形状

**解决**:
```bash
git log --oneline experiments/macos_r2/e3_multistep_all_areas.py | head -3
```
应该看到 `6b62831 Fix E3 persistence bug` 是最新的。如果没有，重新 pull：
```bash
git pull origin main --rebase
```

### 如果 E4 找不到 predicted embeddings

**症状**: `prediction_missing` 状态出现在所有 pair 中

**解决**:
```bash
ls data/independent_change_labels/predicted/*_embedding.npy | wc -l
```
应该看到 `61`。如果少于 61，确认 LFS 已下载：
```bash
git lfs pull
```

### 如果 E6 只看到 10 area（不是 30）

**症状**: `n: 10` 而不是 `n: 30`

**原因**: 缺少 E6 之前提取的 20 新 area

**解决**: 检查 `experiments/paper8/data/` 是否有以下额外 area：
- bohai_delta, changbai_lower_belt, dongting_wetland
- erhai_lake_margin, gobi_margin, guizhou_karst
- hainan_coastal, hetao, minnan_coast
- north_china_plain, northeast_plain, ordos_desert_edge
- qinling_south_slope, suzhou_fringe, tarim_oasis
- wuhan_outer_ring, wumeng_mountains, wuyi_mountain
- yunnan_eco, （其他）

如果这些不存在，需要先运行 `python e6_expand_areas.py --extract-only`（会 GEE 提取，需 ~5 小时）。

---

## 总结

| 步骤 | 命令 | 时间 |
|---|---|---|
| Step 0 | `git pull origin main` | 30秒 |
| Step 1 (E3) | `python e3_multistep_all_areas.py` | 30分钟 |
| Step 2 (E4) | `python e4_per_year_decoder.py` | 15分钟 |
| Step 3 (E6) | `python e6_expand_areas.py --eval-only --min-years 8` | 5分钟 |
| Step 4 | `git add ... && git commit && git push` | 2分钟 |

**总计**: ~55 分钟

完成后 ping 我，我立即开始 v4 整合。
