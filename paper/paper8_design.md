# Paper 8 设计方案：真正的 "Dreaming in Embedding Space"

> 基于 Paper 7 审稿反馈和六篇论文积累的战略设计
> 日期：2026-04-07

---

## 确定的设计决策

- **动作空间**：先方案 A（空间选择 + scenario，离散），论文中讨论方案 C 作为 future work
- **评估方式**：嵌入空间评估（主要）+ 真实环境验证（补充）

---

## Paper 8 详细技术架构

### 整体流程

```
┌─────────────────────────────────────────────────────────┐
│                    EmbeddingSpaceEnv                      │
│                                                           │
│  State: z_t ∈ R^{64×H×W}  (AlphaEarth embedding grid)   │
│                                                           │
│  Action: (region_id, scenario_id)                         │
│    region_id ∈ {0, ..., K-1}  (K spatial clusters)       │
│    scenario_id ∈ {0, ..., 4}  (5 scenarios)              │
│                                                           │
│  Transition: LatentDynamicsNet (frozen or fine-tuned)      │
│    z_{t+1}[r,c] = LDN(z_t[r,c], s_blended[r,c], ctx)    │
│    where s_blended = scenario if (r,c) ∈ region           │
│                    = baseline otherwise                    │
│                                                           │
│  Reward: R(z_t, z_{t+1}) =                                │
│    w1 · cropland_gain(z_{t+1})     (LULC decoder)        │
│    + w2 · slope_proxy(z_{t+1})     (embedding direction)  │
│    + w3 · contiguity_proxy(z_{t+1}) (spatial coherence)   │
│    calibrated by ATT from causal inference                │
│                                                           │
│  Done: after T steps (T = planning horizon in years)      │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│             SpatialInterventionPolicy                     │
│                                                           │
│  Input: z_t flattened or pooled                           │
│  Architecture: GNN on region adjacency graph              │
│    OR per-region scorer (like ParcelScoringPolicy)        │
│  Output: (region_id, scenario_id) with masking            │
└─────────────────────────────────────────────────────────┘
           │
           ▼  (evaluation)
┌─────────────────────────────────────────────────────────┐
│  Evaluation Path 1: Embedding-space metrics               │
│    - LULC class distribution change (via decoder)         │
│    - Cropland area %, Forest area %                       │
│    - Spatial coherence (neighbor cosine similarity)       │
│                                                           │
│  Evaluation Path 2: Real-env translation                  │
│    - Map region interventions → block selections          │
│    - Execute on CountyLevelEnv                            │
│    - Report slope%, contiguity, baimu fang                │
└─────────────────────────────────────────────────────────┘
```

### 1. EmbeddingSpaceEnv 详细设计

**State**: AlphaEarth embedding grid z_t ∈ R^{64×H×W}
- 初始状态：从 GEE 提取某区域某年的嵌入
- H, W 取决于区域大小和分辨率（璧山区 ~60×70 at 500m）
- 每步是一年的时间推进

**Action**: MultiDiscrete([K, 5])
- K = 空间区域数（与 Paper 3 的 block 数不同——这里是嵌入网格上的聚类）
- 5 = scenario 数量
- 区域划分方案：
  - 方案 1：将 H×W 网格等分为 K 个 patch（如 8×8 = 64 个区域）
  - 方案 2：基于初始嵌入做 K-means 聚类（语义相似的像素归为一组）
  - 方案 3：复用 Paper 3-4 的 block 定义（需要空间映射）
  - 推荐方案 2：K-means 聚类，K=50-100

**Transition**: LatentDynamicsNet.forward(z_t, s_blended, context)
- s_blended 的构建：
  ```python
  s_blended = torch.zeros(1, 16)
  s_blended[0, 4] = 1.0  # baseline for all
  # Override for selected region:
  for (r, c) in selected_region_pixels:
      s_region = encode_scenario(scenario_names[action_scenario])
      # 在 forward 中需要 per-pixel scenario — 当前 LDN broadcast globally
      # 解决：分两次 forward，baseline + intervention，然后 blend
  ```
  实现方式：调用 intervention_predict() 的 blending 逻辑（已在 causal_world_model.py 中实现）

**Reward**: 多层设计
```python
def compute_reward(z_t, z_tp1, decoder):
    # Layer 1: LULC change (via linear decoder)
    lulc_t = decoder.predict(z_t.reshape(-1, 64))   # (H*W,)
    lulc_tp1 = decoder.predict(z_tp1.reshape(-1, 64))
    
    cropland_change = (lulc_tp1 == 7).sum() - (lulc_t == 7).sum()  # class 7 = cropland
    forest_change = (lulc_tp1 == 2).sum() - (lulc_t == 2).sum()    # class 2 = trees
    
    # Layer 2: Embedding-space slope proxy
    # 耕地像素的嵌入 vs 非耕地像素的嵌入的分离度
    crop_mask_tp1 = (lulc_tp1 == 7)
    if crop_mask_tp1.sum() > 0:
        crop_emb_mean = z_tp1[:, crop_mask_tp1].mean(dim=1)  # (64,)
        # 与"理想耕地"方向的对齐度可以用 decoder 权重定义
    
    # Layer 3: Spatial coherence (contiguity proxy)
    # 相邻像素嵌入的 cosine similarity
    coherence = compute_neighbor_cosine(z_tp1)
    
    reward = w1 * cropland_change + w2 * forest_penalty + w3 * coherence
    return reward * calibration_factor  # ATT 校准
```

**Episode 结构**:
- 每步 = 1 年推进（World Model 是年度分辨率）
- 总步数 = 5-10 年规划期
- 每步选择一个区域 + scenario
- 总预算 = T 次干预（每年只能干预一个区域）

### 2. 嵌入→真实环境翻译器

将嵌入空间策略的决策"翻译"为 Paper 4 的 block-level 动作：

```python
class EmbeddingToBlockTranslator:
    """将 (region_id, scenario_id) 映射为 CountyLevelEnv 的 block_id。"""
    
    def __init__(self, block_geofm_embeddings, region_assignments):
        # block_geofm_embeddings: (2600, 64) — 已在 E4 中提取
        # region_assignments: (2600,) — 每个 block 属于哪个 embedding region
        self.block_emb = block_geofm_embeddings
        self.region_map = region_assignments
    
    def translate(self, region_id, scenario_id):
        """返回该 region 中最适合该 scenario 的 block_id。"""
        blocks_in_region = np.where(self.region_map == region_id)[0]
        
        if scenario_id == 2:  # agricultural_intensification
            # 选择该 region 中坡度最高的耕地 block（最需要优化的）
            ...
        elif scenario_id == 1:  # ecological_restoration
            # 选择该 region 中最适合退耕还林的 block
            ...
        
        return best_block_id
```

### 3. 跨区域泛化实验

这是 Paper 8 相比 Paper 1-7 的最大优势：

```
训练：在璧山区 (Bishan) 的嵌入上训练策略
测试：在 World Model 的 17 个研究区中选 3-5 个做 zero-shot 测试

可行性：
- AlphaEarth 嵌入全球覆盖，任何区域都有
- EmbeddingSpaceEnv 不依赖地块数据，只需要嵌入
- LatentDynamicsNet 已在 17 个区域上训练过
- 策略网络（如果用 per-region scorer）是区域无关的
```

### 4. 代码实现路线图

```
paper8/
├── embedding_space_env.py      — EmbeddingSpaceEnv (Gymnasium)
│   ├── __init__: 加载嵌入 + 划分区域 + 加载 LDN + decoder
│   ├── reset: 设置初始嵌入（某区域某年）
│   ├── step: LDN forward with blended scenario → reward
│   └── action_masks: 区域可用性
│
├── spatial_intervention_policy.py — 策略网络
│   ├── RegionScorer: per-region embedding → logit (复用 ParcelScoringPolicy 模式)
│   └── ScenarioSelector: global state → scenario distribution
│
├── embedding_reward.py         — 嵌入空间奖励函数
│   ├── lulc_change_reward: decoder-based
│   ├── coherence_reward: spatial cosine
│   └── causal_calibration: ATT scaling
│
├── block_translator.py         — 嵌入策略 → block 动作翻译器
│
├── train_embedding_rl.py       — 训练脚本
├── eval_embedding_rl.py        — 嵌入空间 + 真实环境评估
└── cross_region_transfer.py    — 跨区域 zero-shot 泛化实验
```

### 5. 实验矩阵

| 实验 | 目的 | 指标 |
|------|------|------|
| E1 | 嵌入空间策略 vs random vs greedy | LULC change, coherence |
| E2 | 翻译到真实环境的效果 | slope%, cont, baimu |
| E3 | vs Paper 7 model-based | slope% on real env |
| E4 | vs Paper 4 model-free | slope% on real env |
| E5 | 跨区域 zero-shot (3-5 区域) | LULC change on unseen regions |
| E6 | 因果校准消融 | with/without ATT calibration |
| E7 | 区域划分粒度 (K=25,50,100) | 策略质量 vs 动作空间大小 |
| E8 | 规划时长 (T=3,5,10 年) | 长期 vs 短期策略差异 |

| | Paper 7 (当前) | Paper 8 (规划) |
|---|---|---|
| **状态空间** | MDP 原始观测 (2600×17+12) | **64-dim GeoFM 嵌入空间** |
| **转移模型** | 新训练的 TransitionModel (237K) | **LatentDynamicsNet (459K, 已有)** |
| **动作定义** | 选择一个 block (离散) | **在嵌入空间中施加 scenario 干预** |
| **与 World Model 的关系** | 仅用 GeoFM 做辅助特征（E4 失败） | **World Model 就是环境** |
| **与 Causal 论文的关系** | 用 ATT 做 reward 校准 | **ATT 校准 scenario 编码（已有代码）** |
| **理论基础** | Learned dynamics + causal calibration | **JEPA + World Model + Causal** |
| **目标期刊** | Nature Machine Intelligence | **Nature / Science 子刊** |

---

## Paper 8 的核心思想

**将耕地整治优化重新定义为在 GeoFM 嵌入空间中的干预规划问题。**

传统 MDP：`选择 block → 贪心引擎执行交换 → 计算指标变化`
Paper 8 MDP：`选择空间子区域 + scenario → World Model 预测嵌入变化 → 在嵌入空间评估效果`

这意味着：
- **不需要地块级模拟**——整个优化在 64 维嵌入空间完成
- **不需要特定数据集**——任何有 AlphaEarth 嵌入的区域都可以直接优化
- **World Model 论文直接成为基础设施**——不是辅助特征，而是核心环境

---

## 关键技术挑战与解决方案

### 挑战 1：如何在连续嵌入空间定义"动作"？

传统的 block-level MDP 动作是离散的（选择第 k 个 block）。在嵌入空间中没有"block"的概念。

**解决方案：Spatial Intervention Action**

```
动作 = (sub_region_mask, scenario_id, intensity)

- sub_region_mask: 在 H×W 嵌入网格上选择干预区域
  → 可以用注意力机制（attention over spatial locations）生成
  → 或离散化为 K 个预定义子区域（如 K-means 聚类）

- scenario_id: 5 种已定义的 scenario 之一
  (urban_sprawl, ecological_restoration, agricultural_intensification, 
   climate_adaptation, baseline)

- intensity: 干预强度 ∈ [0, 1]
  → 控制 scenario encoding 的缩放幅度
```

**最简方案（推荐先做）**：
- 将县域划分为 N 个空间区块（复用 Paper 3 的 block 定义或 K-means）
- 动作 = 选择哪个区块 + 施加哪种 scenario
- 动作空间 = Discrete(N × 5) 或 MultiDiscrete([N, 5])

### 挑战 2：如何在嵌入空间定义"奖励"？

传统奖励基于坡度、连片度、百亩方等需要地块级信息的指标。嵌入空间中没有这些。

**解决方案：Embedding-Space Reward**

三层奖励设计：

```
Layer 1: 嵌入空间度量（无需解码）
  - cosine_shift: 干预前后嵌入的余弦距离（越大=变化越大）
  - cropland_direction: 嵌入变化方向与"耕地增加"方向的对齐度
    → 用 LULC decoder 的权重向量定义"耕地方向"
  - neighborhood_coherence: 相邻像素嵌入的一致性变化

Layer 2: 解码后度量（轻量级）
  - 通过 linear probe 将嵌入解码为 LULC 类别
  - 计算耕地面积变化、连片度变化
  - 这仅需矩阵乘法，无需地块级模拟

Layer 3: 因果校准（来自 Causal 论文）
  - 用 ATT 估计校准 Layer 2 的奖励（复用 integrate_statistical_prior）
```

### 挑战 3：World Model 的 Scenario 调节还没训练

LatentDynamicsNet 目前只在 baseline scenario 上训练过。其他 4 个 scenario 的效果未经验证。

**解决方案：分两步走**

```
Step A（可立即做）：仅用 baseline scenario + intensity 缩放
  - 动作 = 选择区域 + intensity ∈ {0.5, 1.0, 1.5, 2.0}
  - 等价于"在该区域施加不同强度的历史趋势延续"
  - 这已经足够产生差异化的嵌入变化

Step B（需要额外数据）：训练 scenario-specific dynamics
  - 收集有明确政策标签的历史数据（如退耕还林区域标记为 ecological_restoration）
  - 在标签数据上微调 LatentDynamicsNet
  - 这是 World Model 论文自己识别的 Future Work
```

Paper 8 可以先用 Step A 发表，Step B 作为其自己的 future work。

---

## Paper 8 与现有代码的关系

### 直接复用（无需修改）
- `D:/adk/data_agent/world_model.py` — LatentDynamicsNet, predict_sequence(), encode_scenario()
- `D:/adk/data_agent/weights/latent_dynamics_v1.pt` — 训练好的模型权重
- `D:/adk/data_agent/causal_world_model.py` — intervention_predict(), integrate_statistical_prior()
- `D:/adk/data_agent/dreamer_env.py` — ParcelEmbeddingMapper, ActionToScenarioEncoder (参考)

### 需要新建
- `EmbeddingSpaceEnv` — 在 64-dim 嵌入空间中运行的 Gymnasium 环境
- `SpatialInterventionPolicy` — 空间干预策略网络（选择 where + what + how much）
- `EmbeddingReward` — 嵌入空间奖励函数
- 训练脚本 + 评估脚本

### 评估方案
- **嵌入空间评估**：cosine similarity, LULC 类别变化（通过 decoder）
- **真实环境对照**：将嵌入空间策略的决策"翻译"回 block-level 动作，在真实 CountyLevelEnv 上执行
  → 这需要一个 embedding→block action 的映射器

---

## Paper 8 是否依赖 Paper 7 修改后的结果？

**不依赖。** 原因：

1. Paper 8 的架构完全不同——在嵌入空间而非观测空间运行
2. Paper 7 的贡献（learned dynamics + causal calibration）和 Paper 8 的贡献（embedding-space MDP + JEPA realization）是正交的
3. Paper 7 的结果（无论修改后数字是否变化）不影响 Paper 8 的设计逻辑

但 Paper 7 的经验教训会指导 Paper 8：
- E4 的负面结果（GeoFM 嵌入作为辅助特征不好用）→ Paper 8 应该让嵌入成为**唯一**的状态表示，而非辅助
- 因果校准的成功（+17.8%）→ Paper 8 应该从一开始就内置 ATT 校准
- 分布偏移问题 → Paper 8 可以用 World Model 的 17 个训练区域做真正的跨区域泛化

---

## Paper 8 的独特卖点（vs Paper 7）

1. **真正的 JEPA 实现**：frozen encoder (AlphaEarth) + learned predictor (LatentDynamicsNet) + RL policy，完整的 JEPA 范式在地理空间领域的首次实现

2. **区域无关性**：不需要地块级数据，只要有 AlphaEarth 嵌入（全球 10m 覆盖）就能优化。这解决了 Paper 1-4 的"单一研究区"弱点

3. **与 World Model 论文的深度整合**：World Model 不再是辅助工具，而是 MDP 的核心环境

4. **与 Causal 论文的深度整合**：ATT 校准不再是后处理，而是 scenario encoding 的内在组成部分

---

## 时间线建议

| 阶段 | Paper 7 修改 | Paper 8 设计 |
|------|:---:|:---:|
| Week 1-2 | 增加 seeds + α 网格搜索 | 设计 EmbeddingSpaceEnv |
| Week 3-4 | 修改论文 + 补充分析 | 实现核心代码 + 初步实验 |
| Week 5-6 | 提交 Paper 7 | 完整实验 |
| Week 7-8 | — | 论文撰写 |

**两条路可以并行推进**：Paper 7 修改主要是增加计算（多 seeds），Paper 8 主要是新代码开发，不冲突。

---

## 结论

Paper 8 的设计**不需要等 Paper 7 修改结果**，可以立即开始。但 Paper 7 的修改应该优先（因为审稿意见明确、修改可行、投稿在即），Paper 8 的代码开发可以在等 Paper 7 seeds 训练的间隙进行。

Paper 7 = "learned dynamics 在观测空间" → 方法论扎实、结果强
Paper 8 = "JEPA world model 在嵌入空间" → 创新性更高、通用性更强

两篇互补，不冲突。
