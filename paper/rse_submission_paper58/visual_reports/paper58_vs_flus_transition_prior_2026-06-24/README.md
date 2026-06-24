# Paper58-LAS vs GeoSOS-FLUS 路线直观对比图文报告

日期：2026-06-24

> 说明：这里的 GeoSOS-FLUS 路线指当前可复现实验里的 **official FLUS console baseline**，即使用官方 FLUS console 源码编译出的命令行程序和同一 Batch 5 数据构造的对比基线。它使用官方源码，但还不是 GeoSOS-FLUS native GUI/完整传统驱动因子工作流。

## 现在能不能算全部完成？

不能把所有工作都看作完成。更准确的状态是：

- **已完成**：Paper58-LAS 与 official FLUS console baseline 的可复现控制对比，包含 oracle demand、Paper58-prediction demand、transition-prior demand 三种设置。
- **已形成强阶段性结论**：在 oracle demand 和 transition-prior 非 oracle demand 下，Paper58-LAS 均超过 official FLUS console baseline。
- **还未完成**：GeoSOS-FLUS native workflow/完整传统驱动因子输入包对比、zero-local-user-data operational 验证、以及更强 demand forecast 模块。

## 一眼结论

在最有解释力的 `transition_prior` 非 oracle demand 设置下，Paper58-LAS 不读取目标区域 2021 年真实标签来生成需求，仍然显著优于 official FLUS console baseline：

- Change F1：official FLUS console `0.101` -> Paper58-LAS `0.251`，平均优势 `+0.150`。
- FoM：official FLUS console `0.030` -> Paper58-LAS `0.104`，平均优势 `+0.074`。
- Recall：official FLUS console `0.123` -> Paper58-LAS `0.519`。
- Transition accuracy：official FLUS console `0.084` -> Paper58-LAS `0.377`。

代价也很清楚：Paper58-LAS 的 allocation disagreement 更高，说明它更敢抓变化、抓到更多真实变化，但空间落点还不够干净。

## 图 1：证据阶梯

![Evidence ladder](figures/fig1_evidence_ladder.png)

看图说话：

- oracle demand 下，Paper58-LAS 的 F1 和 FoM 都明显赢。
- 直接用 Paper58 预测图计数做 demand 时，优势明显收缩，FoM 略输，这是必须保留的负证据。
- transition-prior demand 下，F1 和 FoM 又重新变成稳定正优势，说明“非 oracle 条件下也能赢”已经有了更强证据。

## 图 2：主设置下的指标对比

![Transition-prior metric means](figures/fig2_transition_prior_metric_means.png)

看图说话：

- Paper58-LAS 在 Change F1、FoM、Recall、Transition accuracy 上全部高于 official FLUS console baseline。
- 最明显的提升来自 Recall 和 Transition accuracy：它能抓住更多真实变化像元，也更常把变化方向判对。
- 右侧两个 disagreement 是越低越好。Paper58-LAS 的 allocation disagreement 更高，这是当前弱点。

## 图 3：逐区胜负

![Area advantages](figures/fig3_area_advantages.png)

看图说话：

- 7 个 holdout 区域里，6 个区域 Paper58-LAS 获得正优势。
- `liaohe_delta_wetland_holdout` 和 `xilingol_grassland_margin_holdout` 是最强正例。
- `huaibei_irrigation_plain_holdout` 是主要负例，说明农业灌溉平原场景还需要专项诊断。

逐区核心数值：

| area | stratum | change_f1_advantage | fom_advantage |
| --- | --- | --- | --- |
| dabie_forest_edge_holdout | Forest | +0.119 | +0.113 |
| huaibei_irrigation_plain_holdout | Agriculture | -0.085 | -0.047 |
| liaohe_delta_wetland_holdout | Wetland | +0.334 | +0.088 |
| renqiu_baiyangdian_edge_holdout | Urban | +0.100 | +0.046 |
| wenan_lakeplain_newtown_holdout | Urban | +0.073 | +0.038 |
| wuxi_taihu_dense_edge_holdout | Urban | +0.123 | +0.073 |
| xilingol_grassland_margin_holdout | Grassland | +0.389 | +0.208 |

## 图 4：邻域权重不是偶然调参

![Weight scan](figures/fig4_weight_scan.png)

看图说话：

- `neighborhood_weight=0.5` 到 `3.0` 区间内，F1 和 FoM 优势都保持为正。
- 这说明 transition-prior 的胜出不是某一个单点参数碰巧得到的。
- `w=0` 时 FoM 的 CI low 贴近或略低于 0，说明邻域约束仍然是 LAS 当前有效性的组成部分。

## 图 5：空间图怎么理解

![Spatial examples](figures/fig5_spatial_examples.png)

看图说话：

- 绿色是正确抓到的真实变化，红色是真实变化但没抓对，黄色是稳定区域上的误报变化。
- Liaohe 和 Wuxi 中，Paper58-LAS 明显有更多绿色，说明它比 official FLUS console baseline 更能找到真实变化。
- Huaibei 中，Paper58-LAS 出现更多黄色或没把真实变化抓准，这就是逐区指标里它输掉的直观原因。

## 最终判断

如果问题是“Paper58 路线是否已经在当前可复现 official FLUS console baseline 上形成超越证据”，答案是：**是，阶段性证据已经很强**。

如果问题是“是否已经正式完成对 GeoSOS-FLUS native workflow 的全面超越证明”，答案是：**还没有**。下一步必须做 GeoSOS-FLUS native 输入包/完整传统驱动因子工作流对比，并验证 Paper58 的 zero-local-user-data 流程。
