# Paper 58 RSE R2 macOS Experiment Audit — Windows端发现的严重问题

**Auditor**: Windows 端手工验证（2026-07-02）  
**Status**: **HOLD — 需要补充实验和澄清数据来源**

## Executive Summary

macOS 端提交的 E1-E7 实验中，**E3 有根本性 bug**，**E4 未完成**，**E6 数据构成存疑**，**v2 原始基线数据无法复现**。在彻底解决这些问题之前，**不能将 macOS 结果直接用于 v3 论文**。

---

## 发现 1: E3 multi-step persistence 定义错误

**文件**: `experiments/macos_r2/e3_multistep_all_areas.py:149`

**Bug**: 
```python
persist = cosine(F.normalize(z0, p=2, dim=1), z_true).item()
```

这将 persistence 定义为 **z0（初始年）** 与 z_step 的相似度，而不是 z_{step-1} 与 z_step 的相似度。导致：
- step=1: 比较 z0 vs z1（正确）
- step=2: 比较 z0 vs z2（**错误**！应该是 z1 vs z2）
- step=6: 比较 z0 vs z6（完全错误）

**影响**: E3 的所有 multi-step persistence 值都不可信。

**修复方案**: 改为 `persist = cosine(F.normalize(z_prev, p=2, dim=1), z_true)`，并在每步更新 `z_prev = z_true`。

---

## 发现 2: E6 混入了 43 个只有 2 年数据的 area

**文件**: `experiments/macos_r2/results/e6_expanded_areas/eval_area_sources.json`

**问题**:
- 30 个 area 来自 `experiments/paper8/data`（完整 2017-2024，8 年）
- 43 个 area 来自 `data/independent_change_labels/embeddings`（只有 2020-2021，2 年）

**数据构成**:
```
v2 原始 10 区（8年数据）: mean advantage = -0.00549，2/10 正
新增 63 区（混合 8年 + 2年）: mean advantage = -0.01382，11/63 正
全部 73 区: mean advantage = -0.01267，13/73 正
```

**疑问**:
1. 43 个 2 年 area 是否适合用于 dynamics 评估？（dynamics 需要多年时序）
2. 它们是 holdout 数据还是独立验证数据？
3. 评估协议是否对 2 年 vs 8 年数据一视同仁？

**需要澄清**: 这 43 个 area 的来源、用途、是否应该纳入 paired inference。

---

## 发现 3: v2 原始基线数据无法复现

**文件**: `paper/rse_submission_paper58/revision_results/alphaearth_area_metrics.csv`

**问题**: v2 报告的 10-area 数据来自 `experiments/paper8/results/paper8_ablation_encoder.json`，但这些数字**无法用当前代码复现**。

**对比**（chengdu_plain 为例）:

| 来源 | persistence | model | advantage |
|---|---|---|---|
| v2 JSON | 0.9708 | 0.9716 | +0.0007 |
| Windows 手算（2023→2024） | 0.9429 | 0.9331 | -0.0098 |
| macOS E6（2023→2024） | 0.9429 | 0.9331 | -0.0098 |

**差异**: v2 的 model 值比手算高 **0.04**（4%），差异巨大。

**可能原因**:
1. v2 用了不同的评估年份（不是 2023→2024）
2. v2 用了多年平均而不是单一 transition
3. v2 用了不同的 checkpoint
4. v2 的生成脚本 `eval_prithvi_vs_alphaearth.py` 已被修改

**严重性**: v2 论文里的 `mean advantage = +0.00473, p=0.32` **无法复现**。手算和 E6 都得到 `mean = -0.00549`（负数，2/10 正）。

**需要做**: 追溯 v2 原始评估的确切协议，或者承认 v2 基线有误，用 E6 重新评估。

---

## 发现 4: E4 per-year decoder 完全未完成

**文件**: `experiments/macos_r2/results/e4_per_year_decoder/per_pair_end_accuracy_delta.csv`

**状态**: 所有 11 个 independent-change pair 都标记为 `prediction_missing` 或 `decoder_missing`。

**原因**: macOS 端缺少 `data/independent_change_labels/predicted/*_embedding.npy` 文件（这些文件需要 Windows 端先用 AlphaEarth LDN 生成预测）。

**影响**: **E4 实验无结论**，无法评估 per-year decoder retraining 的影响。

**修复方案**: 
1. Windows 端生成所有 predicted embedding（需跑 `generate_change_validation_predictions.py`）
2. 推到 GitHub
3. macOS 端重跑 E4

---

## 发现 5: E1 Prithvi spatial advantage 过小

**文件**: `experiments/macos_r2/results/e1_prithvi_patch/eval_paired_tests.json`

**结果**:
- n=16，mean advantage = **-2.7e-5**（-0.000027）
- 所有 16 个 area 都是负 advantage（0/16 win）
- Wilcoxon p=0.0004（显著），Cohen dz=-3.15（大效应）

**问题**: 虽然统计显著，但**效应量极小**（只有 -0.000027）。这可能是：
1. 数值精度问题（Prithvi 768 维 embedding 的 L2 norm 问题）
2. Prithvi LDN 训练未真正收敛（虽然 val_cossim=0.99999）
3. Prithvi spatial token 确实不适合做 dynamics

**需要检查**:
1. Prithvi LDN 的训练曲线（是否过拟合？）
2. Prithvi spatial token 的数值范围（是否需要不同的 normalization？）
3. 重跑 E1 训练，用更多 epochs 或不同 learning rate

---

## 发现 6: E7 third encoder 只是 scaffold

**文件**: `experiments/macos_r2/results/e7_third_encoder/run_manifest.json`

**状态**: 只写了 SatMAE/Clay 的加载代码，**没有实际跑 extraction + training + eval**。

**原因**: macOS 端没有安装 transformers（SatMAE）或 clay-model-transformers（Clay）。

**影响**: **E7 无结论**，无法回应 reviewer M4 的"第三 encoder ablation"要求。

---

## 建议的补充实验清单

### Priority 1 — 必须修复才能用于 v3

1. **修复 E3 persistence bug 并重跑**  
   预计 30 分钟（只需改 1 行代码 + 重跑）

2. **澄清 E6 的 43 个 2 年 area 是否应该纳入**  
   - 如果不该纳入：重算 paired inference，只用 30 个完整 8 年 area
   - 如果该纳入：在论文里明确说明数据构成差异

3. **追溯 v2 原始基线或承认其有误**  
   - 要么找到原始评估协议并文档化
   - 要么承认 v2 的 `+0.00473` 是错误的，用手算的 `-0.00549` 替代

4. **完成 E4 per-year decoder**  
   - Windows 端生成 predicted embeddings → push → macOS 重跑

### Priority 2 — 加强 encoder ablation（M4 核心）

5. **深入诊断 E1 Prithvi spatial 的 -2.7e-5 效应量**  
   - 检查训练收敛性
   - 尝试不同 normalization
   - 或者承认 Prithvi spatial 确实失败，如实报告

6. **完成 E7 third encoder（SatMAE 或 Clay）**  
   - 安装依赖 → extraction → training → eval
   - 或者明确标记为"未完成，资源限制"

### Priority 3 — 可选加强

7. **E2 terrain ablation 增加样本量**  
   当前 n=30（v2 的 10 + E6 的 20），如果扩到 73 会更有说服力

8. **E5 SA-Alloc sensitivity 加入 per-area variance 分析**  
   当前只报告了 grid-level 聚合，缺少 per-township 的稳定性分析

---

## 当前状态总结

| 实验 | 状态 | 可用性 | 需要修复 |
|---|---|---|---|
| E1 Prithvi spatial | ✅ 完成 | ⚠️ 效应量过小，需诊断 | 深入分析 -2.7e-5 的原因 |
| E2 Terrain | ✅ 完成 | ✅ 可用 | 无（但可扩展到 73 area） |
| E3 Multi-step | ❌ Bug | ❌ 不可用 | **必须重跑** |
| E4 Per-year decoder | ❌ 未完成 | ❌ 不可用 | **必须完成** |
| E5 SA-Alloc sensitivity | ✅ 完成 | ✅ 可用 | 可选加强 per-area 分析 |
| E6 Expand to 73 | ⚠️ 数据构成存疑 | ⚠️ 需澄清 | **必须澄清 43 个 2 年 area** |
| E7 Third encoder | ❌ 只是 scaffold | ❌ 不可用 | 可选完成或标记为未完成 |

**总体评估**: 7 个实验中，**只有 E2 和 E5 可以直接用于 v3**。E1/E3/E4/E6/E7 都需要进一步工作。

---

## 对 v3 论文的影响

基于当前审计结果，**v3 不能声称"已完成 reviewer 要求的所有实验"**。建议的处理方案：

### Option A: 诚实报告部分完成
在 v3 §7.8 Queued experiments 里明确写：
- E2/E5 已完成，结果已纳入 v3
- E3 发现 bug 正在修复
- E4/E7 因依赖问题未完成，承诺在 final revision 完成
- E6 数据构成需与 reviewer 澄清
- E1 完成但效应量过小，需进一步诊断

### Option B: 暂停 v3 提交，先完成所有实验
预计需要额外 3-5 天：
- 修复 E3 并重跑（0.5 天）
- 完成 E4（1 天）
- 诊断 E1 并决定是否重训（1-2 天）
- 澄清 E6 数据构成或重算（0.5 天）
- 完成 E7 或明确标记为 out-of-scope（1 天）

### Option C: 重新评估 v2 基线 + 只用可信实验
承认 v2 的 `+0.00473` 无法复现，用手算的 `-0.00549` 作为新基线，然后：
- 只纳入 E2（terrain）和 E5（SA-Alloc sensitivity）
- E1/E3/E4/E6/E7 标记为"attempted but inconclusive"
- 重新定位论文为 negative result（n=10 时模型显著差于 persistence）

---

## 我的建议

你说得对——在看到这些问题之前，我不应该武断地说"v2 claim 被推翻"。现在需要的是：

1. **先修复 E3 和完成 E4**（这两个是硬性 bug/未完成）
2. **澄清 E6 的 43 个 2 年 area**（与 reviewer 确认是否应该纳入）
3. **追溯 v2 基线**（找到原始评估协议，或承认需要重新评估）
4. **诊断 E1 的 -2.7e-5**（决定是重训还是如实报告）

只有这 4 项完成后，才能决定 v3 的叙事方向。

你希望我：
- (A) 先修复 E3 + 完成 E4，其他实验暂缓
- (B) 深入追溯 v2 原始数据来源，确保 baseline 可信
- (C) 写一个"实验修复计划"文档，交给 macOS 端执行

你的选择？
