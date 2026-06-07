# Paper 5+8 × Paper 9 结合方案

**创建日期**：2026-05-07
**状态**：规划文档，待 Paper 9 投稿完成后再推进
**背景**：讨论把 Paper 5+8 (GeoFM World Model) 与 Paper 9 (Contrastive MPC Breakthrough) 结合的可能性

---

## 两篇论文的各自定位

| Paper | 核心贡献 | encoder | dynamics 预测 | planning |
|-------|---------|---------|--------------|---------|
| **Paper 5+8** | 冻结地理 FM 作 state representation + 学 latent dynamics | ✅ 冻结 GeoFM | ✅ DualRep transition | 下游应用 (RSE 目标) |
| **Paper 9** | contrastive ranking loss 修复 MPC discriminative gap | ❌ 无 (手工 block features) | ✅ ensemble transition | MPC H=5 K=50 |

两篇都在做 world model，但走了两条互不相交的路线：
- **Paper 5+8 优化 "state 怎么表示"**
- **Paper 9 优化 "transition model 怎么训"**

这两个维度本质上是正交的 —— 一个回答 *representation*，一个回答 *learning objective*。

---

## 关键参考数据

| 方法 | Slope % (5-seed) | Contiguity | 备注 |
|------|---------|--------------|------|
| Paper 5+8 DualRep (full) | -0.742% ± 0.151% | +0.0152 ± 0.0042 | 冻结 FM embedding + MSE |
| Paper 9 Contrastive MPC | -1.286% ± 0.079% | +0.0150 ± ?? | 手工特征 + ranking loss |
| Paper 9 baseline v4 MPC | -0.209% ± 0.008% | (near zero) | 手工特征 + pure MSE |

**观察**：Paper 9 > Paper 5+8 在 slope 上。这个任务里，*training objective* 比 *state representation* 更关键。但这个结论仅对 slope 成立，contiguity 两者接近。

---

## 三层次结合方案

### Level 1 — 故事层（零代码成本，立即可做）

投稿时把两篇放进同一个叙事：

> "World model 的质量由两个正交环节决定：
> (a) **state representation**（Paper 5+8：frozen geospatial FM embeddings 取代 hand-crafted features）
> (b) **training objective**（Paper 9：contrastive ranking loss 取代 pure MSE）
> 这两个方向独立有效，且概念可组合。"

**做法**：
- Paper 5+8 投 RSE 时 cite Paper 9 作为 "orthogonal improvement on training objective"
- Paper 9 投 TMLR / NeurIPS MBRL workshop 时 cite Paper 5+8 作为 "orthogonal improvement on state representation"
- 两篇 cover letter 互引，审稿人看到 roadmap 而非 one-off trick

**好处**：互相加权，故事层拔高到 MBRL 路线图。
**代价**：无。
**推荐时机**：Paper 9 最终定稿前加一段 "Relationship to representation learning" 段落。

---

### Level 2 — 架构拼装（中等代价，1–2 周）

真正把 DualRep (Paper 5+8) 的 state 换进 Paper 9 的 contrastive pipeline：

```
GeoFM encoder (frozen, from Paper 5+8)
    ↓ produces embedded block states
Transition model (Paper 9 architecture)
    ↓ trained with MSE + λ · ranking loss (Paper 9 recipe)
Contrastive-trained ensemble
    ↓
MPC H=5 K=50 greedy (Paper 9 planner)
```

**2×2 Ablation 设计（核心实验）**：

| | hand-crafted features | GeoFM embedding |
|--|--|--|
| MSE only | -0.209% (Paper 9 baseline) | -0.742% (Paper 5+8 DualRep) |
| MSE + ranking (λ=5.0) | **-1.286%** (Paper 9) | **?? (待实验)** |

右下角是唯一未知的一格。这个实验可以直接决定：

- 如果右下格 > -1.5%：**两个维度可叠加**，证明 representation × objective 都重要，论文可以写成 "full stack improvement"
- 如果右下格 ≈ -1.29%：**ceiling bound by objective**，representation 在高 λ 下已无额外收益 → 支持 "discriminative gap is dominant bottleneck" 论点
- 如果右下格 < -1.29%：**存在 incompatibility**，FM embedding 与 ranking loss 可能干扰，需要深入分析（也是有趣发现）

**潜在风险与假设**：
1. GeoFM embedding 空间的 pairwise reward discrimination 是否也塌陷？(Paper 9 诊断需要在新空间重做)
2. `pairwise_data.npz` 需要重新生成（在 FM embedding 空间采样 1000 states × 50 actions）
3. Transition model 的输入维度需要适配 FM embedding 维度

**产出**：
- 单独一篇短 paper (Paper 10: "Decomposing the World Model: Representation vs Objective") 投 MBRL workshop
- 或作为 Paper 9 的 journal extension 投 TMLR long version

---

### Level 3 — 概念合一（大代价，JEPA 级故事）

完全对齐 LeCun JEPA / H-JEPA 蓝图：

- **Encoder**：Paper 5+8 的 frozen GeoFM（甚至在 GeoFM embedding 上再学一个 task encoder）
- **Predictor**：Paper 9 的 transition model
- **Loss**：embedding 空间 prediction (JEPA style) + discriminative ranking (Paper 9 style)
- **Planning**：H-JEPA style hierarchical MPC (county → township → block)
- **Regularization**：VICReg / BYOL 防 representation collapse

**JEPA framing 论点**：

> "JEPA 主张预测空间从像素移到 embedding（I-JEPA, V-JEPA）；我们证明即使在 embedding 空间里，纯 reconstruction objective 仍然丢失了 decision-relevant 的 ordering signal。补救方案不是换架构，而是加一个 discriminative auxiliary loss。这是对 LeCun JEPA 蓝图的 *补充诊断*，不是替代。"

**这是独立的大工程**，不是小修小补。预计 3–6 月，需要：
- 跨区域（3–5 县）验证通用性
- 与 Dreamer V3 / TD-MPC2 原版 baseline 正面对比
- 经济/生态影响量化（需要领域合作者）
- 10+ seeds × 多 episode 的完整统计

**目标**：Nature Machine Intelligence 主投，ICML / NeurIPS 作为备选。

---

## 路线图建议

### 短期（2–4 周）— Level 1
- Paper 9 定稿加一段 "Relationship to representation learning" 讨论（~150 字）
- Paper 5+8 投稿时 cite Paper 9，反之亦然
- 两篇 cover letter 互引

### 中期（1–2 月）— Level 2
- 写 2×2 ablation pipeline（主要工程量在 FM encoder 接入 contrastive trainer）
- 重新生成 FM embedding 空间的 pairwise data
- 跑右下角格子的实验
- 根据结果决定是写独立短 paper 还是 Paper 9 journal extension

### 长期（3–6 月）— Level 3
- 跨区域数据工程（至少 2 个新县）
- SOTA baseline 实现（Dreamer V3, TD-MPC2 discrete variant）
- H-JEPA style hierarchy 实现
- 经济/生态影响建模（联系合作者）

---

## 当前优先级

✅ **先确保 Paper 9 OK** —— 正在进行的 5-seed multi-objective eval 完成后，更新论文并投稿
⏸ **Level 1** —— Paper 9 定稿时顺手加一段即可
⏸ **Level 2** —— Paper 9 投稿后启动，不阻塞主线
⏸ **Level 3** —— 远期规划，看前两篇论文反响再决定

---

## 关键开放问题（留给 Level 2 实验回答）

1. FM embedding 空间下，baseline 的 pairwise ranking accuracy 是多少？如果已经 > 0.8，说明 representation 本身足够 discriminative，contrastive loss 可能不必要。
2. contrastive loss 在 FM embedding 上的最优 λ 是多少？可能与手工特征下的 λ=5.0 不同。
3. 如果 2×2 表右下格收益有限，是 representation 已饱和，还是 contrastive 有上限？
4. 跨区域时，FM embedding + contrastive 是否比 hand-crafted + contrastive 更 transferable？这可能是 Level 3 故事的关键。

---

## 参考文件

- Paper 5+8 manuscript: `D:/test/geofm_world_model_rl.tex`
- Paper 5+8 原 Paper 5: `D:/adk/docs/paper5_world_model_paper.tex`
- Paper 5+8 merge spec: `docs/superpowers/specs/2026-04-21-geofm-world-model-rl-merge-design.md`
- Paper 9 manuscript: `D:/test/paper9_v5.tex` (24 pages, compiled)
- Paper 9 breakthrough memo: `C:/Users/zn198/.claude/projects/D--test/memory/paper9_contrastive_breakthrough.md`
- Paper 9 contrastive code: `D:/test/paper9_contrastive/`
