# 论文对比分析：Toward World Models for Epidemiology vs. AlphaEarth 地理空间世界模型

> **论文**: Zeeshan Memon, Yiqi Su, et al. *Toward World Models for Epidemiology*. arXiv:2604.09519v1, 2026.
>
> **对比对象**: 本项目 AlphaEarth/Dreamer 地理空间世界模型 (`world_model.py` / `dreamer_env.py`)
>
> **分析日期**: 2026-04-14

---

## 1. 论文概要

### 1.1 核心论点

计算流行病学是世界模型（World Model）的天然应用场景，但目前严重不足。论文提出将疫情建模为**受控部分可观测动力系统 (Controlled POMDP)**：

| 要素 | 流行病学映射 |
|------|-------------|
| 隐状态 $x_t$ | 真实感染数、免疫水平、行为响应 |
| 动作 $a_t$ | 干预措施 (NPI、疫苗分配) |
| 观测 $o_t$ | 报告病例、住院数、污水信号（有噪声 + 受政策反馈影响） |
| 动力学 $P_\theta$ | $x_{t+1} \sim P_\theta(x_{t+1} \mid x_t, a_t)$ |
| 观测模型 $\Omega_\theta$ | $o_t \sim \Omega_\theta(o_t \mid x_t, a_{t-1})$，关键: **内生性** |

### 1.2 三个 Case Study

论文通过三个案例论证传统预测模型的不足：

1. **策略性误报 (User Deception)** — 行为监测中的策略性误报使观测过程本身被行为者扭曲，观测不再是隐状态的无偏代理。
2. **时滞信号 (Moving Target)** — 住院/死亡等信号滞后真实感染 1-2 周，且滞后长度本身随政策变化。
3. **反事实分析 (Counterfactual)** — 相同的历史轨迹在不同干预序列下产生分叉未来，需要模型具备 rollout 而非回归能力。

### 1.3 架构设计

- **编码器 $\phi$**: 学习的循环/记忆表示（GRU），隐式编码不确定性
- **动力学模型 $P_\theta$**: 建模隐状态随时间的演化（以动作为条件）
- **观测模型 $\Omega_\theta$**: 将隐状态映射回实际可测量信号，捕获噪声和偏差
- **策略模型**: LLM (CovidLLM / GPT-4o-mini / Qwen-7B) 将内部信息状态 $h_t$ 映射为候选干预

### 1.4 训练目标

序列变分目标（类 Dreamer ELBO）：

$$\mathcal{L}_{WM} = \sum_{t=1}^{T} \mathbb{E}_{q_\phi(x_t)} [\log \Omega_\theta(o_t \mid x_t)] - \beta \cdot \text{KL}(q_\phi(x_{1:T}) \| p_\theta(x_{1:T} \mid a_{1:T}))$$

策略优化使用 GRPO (Group Relative Policy Optimization) 迭代。

### 1.5 关键引用

- Ha & Schmidhuber (2018): World Models 原始概念
- Hafner et al. (2020): Dreamer — 隐空间想象力学习行为
- Schrittwieser et al. (2020): MuZero — 学习模型规划
- Silver et al. (2021): Reward is Enough 范式

---

## 2. 本项目 AlphaEarth 世界模型概要

### 2.1 架构

基于 **JEPA (Joint Embedding Predictive Architecture)** 的地理空间世界模型，采用 Dreamer 风格架构：

- **AlphaEarth**: 冻结的遥感基础模型，提供 64 维年度地理空间嵌入
- **LatentDynamicsNet**: 残差 CNN（膨胀卷积，dilation 1/2/4，~170m 感受野），在嵌入空间预测年际变化
- **DreamerEnv**: Gym 风格封装，集成世界模型到 RL 循环，提供前瞻 K 步辅助奖励
- **ParcelEmbeddingMapper**: 向量 GIS 数据到嵌入空间的桥接（地块几何 → 区域均值聚合）
- **ActionToScenarioEncoder**: 离散 RL 动作 → 16D 场景向量 (5 个预定义场景)

### 2.2 核心公式

残差动力学：
$$z_{t+1} = \text{Normalize}(z_t + f(z_t, s, c))$$

其中 $f$ 为 LatentDynamicsNet，$s$ 为场景向量，$c$ 为空间上下文（DEM 高程/坡度）。L2 归一化将预测嵌入约束在单位超球面上，防止多年自回归 rollout 中的流形漂移。

### 2.3 领域集成

- **数据源**: Google Earth Engine — 年度 AlphaEarth 嵌入 + ESRI LULC 标签
- **优化角色**: 长期评估器 — 基础 RL 评估即时约束（坡度适宜性），世界模型评估 3-10 年生态/农业可持续性
- **场景规划**: 5 个地理空间场景（城市扩张、生态恢复等），DreamPlanner 评分候选土地利用变更

---

## 3. 逐维度对比

| 维度 | AlphaEarth/Dreamer (本项目) | EpiWM (论文) |
|------|---------------------------|--------------|
| **应用域** | 地表覆盖/土地利用演化 | 疫情传播动力学 |
| **隐状态表示** | 64D AlphaEarth embedding（地块年度嵌入） | 真实感染/免疫/行为复合状态 |
| **动力学模型** | LatentDynamicsNet（残差 CNN，膨胀卷积） | GRU encoder + learned $P_\theta$ |
| **残差/转移建模** | 确定性残差: $z + f(z,s,c)$ | 概率转移: $x_{t+1} \sim P_\theta$ |
| **流形约束** | L2 归一化 → 单位超球面 | KL 散度 → 后验正则化 |
| **动作/场景编码** | 5 场景 → 16D 场景向量 (ActionToScenarioEncoder) | NPI 干预措施（封锁、疫苗、口罩令） |
| **反事实推理** | DreamPlanner: 模拟 3-10 年嵌入空间位移 | 相同历史 → 不同干预序列 → 分叉轨迹 |
| **RL 集成方式** | DreamerEnv: 世界模型提供辅助奖励（前瞻 K 步） | 世界模型内 imagination rollout → 策略优化 |
| **基础模型** | AlphaEarth（冻结，遥感 foundation model） | CovidLLM（LLaMA-2-7B，LoRA 微调） |
| **策略模型** | MaskablePPO (Stable Baselines 3) | LLM as policy (GPT-4o-mini / Qwen) + GRPO |
| **空间建模** | 膨胀卷积 ~170m 感受野 + DEM 地形上下文 | 不涉及显式空间结构 |
| **观测内生性** | 不涉及（遥感观测相对客观） | **核心贡献**: 观测本身受政策扭曲 |
| **不确定性建模** | 确定性（L2 流形约束） | 概率性（变分推断 + KL 正则化） |
| **训练损失** | MSE 残差损失 | ELBO 变分目标 |
| **因果推断** | 显式: PSM/DiD/Granger/GCCM/Causal Forest | 隐式: rollout 反事实 |
| **时间粒度** | 年度（年际土地利用变化） | 周级（疫情周报数据） |

---

## 4. 关键共鸣点

### 4.1 共同范式: 隐空间动力学 + 想象力规划

两个系统都遵循 Dreamer/JEPA 的核心思想：**不在原始观测空间预测，而在学习的隐空间中做多步 rollout**。

- 本项目: LatentDynamicsNet 在 64D 嵌入空间做多年 rollout，评估土地变更的长期可持续性
- 论文: 学习的 $P_\theta$ 在隐疫情状态空间做 rollout，评估干预措施的长期效果

这一共性说明 World Model 范式正在从游戏/机器人向领域科学全面扩散。

### 4.2 反事实推理

- 本项目: CausalWorldModelToolset 提供干预预测、反事实对比、嵌入效应分析（显式因果推断 + 世界模型 rollout）
- 论文: Case Study 3 直接论证反事实分析是 World Model 的核心价值

两者方法互补 — 本项目的显式因果推断（PSM/DiD/Granger）提供统计可检验性，论文的 rollout 反事实提供灵活的 what-if 模拟。

### 4.3 场景条件生成

- 本项目: ActionToScenarioEncoder 将离散动作映射为 16D 场景向量，作为动力学模型的条件输入
- 论文: 将粗粒度干预措施映射到动力学模型的 action 条件输入

两者都解决了同一个问题：如何将人类可理解的高层决策编码为动力学模型可消费的条件向量。

---

## 5. 论文对本项目的启发

### 5.1 观测内生性 (Observation Endogeneity)

**论文最独特的贡献。** 论文指出观测并非隐状态的被动读数，而是受过去动作影响的内生信号。

**对本项目的启发**: 本项目当前假设遥感观测是客观的，但实际地理信息数据同样存在内生性：
- 地方政府的土地利用上报数据可能受政策导向影响（虚报耕地保有量、低报建设用地面积）
- 统计年报中的指标受考核体系扭曲
- 甚至遥感影像的获取频率和分辨率也受预算/政策影响

如果未来接入非遥感数据源（统计年报、上报数据、实地调查），可以引入内生性观测模型 $\Omega_\theta$。

### 5.2 LLM as Policy

论文用 LLM 直接作为策略模型（而非传统 RL），生成人类可理解的干预方案和理由。

**对本项目的启发**: 当前使用 MaskablePPO 作为策略模型，输出离散动作。对于规划场景：
- LLM 策略可以生成人类可解释的用地变更理由（"将该地块从耕地转为生态林，因为坡度>25度且位于水源保护区"）
- 与现有 LlmAgent 架构天然兼容
- 可作为 DreamPlanner 的增强方向

### 5.3 概率性世界模型

论文的 ELBO 变分目标比当前的 MSE 残差损失更 principled：
- 显式建模预测不确定性（而非点估计）
- KL 正则化防止后验坍缩
- 不确定性可传播到策略优化（风险敏感决策）

**对本项目的启发**: 当前 L2 归一化是确定性流形约束，可以考虑升级为：
- 变分自编码器 (VAE) 风格: 预测均值 + 方差
- 集成方法: 多个 LatentDynamicsNet 实例
- 概率性 rollout: 蒙特卡洛采样多条轨迹

### 5.4 多信号融合

论文处理多种异质观测信号（病例报告、住院、污水、移动性轨迹），需要学习不同信号的噪声特性和时滞。

**对本项目的启发**: 本项目的多模态融合框架 (`fusion/`) 已有 10 种策略和 22 个模块，但世界模型目前仅接入 AlphaEarth 嵌入单一信号源。可以考虑：
- 融合多源遥感（Sentinel-2 + Landsat + SAR）的嵌入
- 引入非遥感信号（社会经济统计、交通流量、用电量）作为辅助观测
- 学习每种信号的噪声特性和时滞

---

## 6. 总结

这篇论文本质上是**在流行病学领域重新推导了本项目已经在地理空间领域实现的核心架构** — 隐空间动力学 + 场景条件 rollout + RL 策略优化。

**本项目的优势**:
- 空间建模更具体（残差 CNN、膨胀卷积、170m 感受野、DEM 上下文）
- 因果推断更显式（6 种统计因果方法 + 世界模型因果）
- 已有完整的 RL 集成（DreamerEnv + MaskablePPO + 5 DRL 场景）

**论文的优势**:
- POMDP 形式化更完整（隐状态/观测/动作的数学框架）
- 观测内生性建模是独特贡献
- 概率性训练目标（ELBO）比确定性 MSE 更 principled
- LLM as Policy 是新颖方向

**共同验证**: World Model 范式正在从游戏/机器人向领域科学全面扩散，地理空间和流行病学是两个最自然的应用领域 — 都涉及隐状态演化、长时间尺度规划、和反事实推理。
