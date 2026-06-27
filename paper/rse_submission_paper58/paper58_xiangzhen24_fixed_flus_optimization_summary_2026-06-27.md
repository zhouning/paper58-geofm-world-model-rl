# Paper58 与 GeoSOS-FLUS 真实乡镇样本固定基线对比总结

生成日期：2026-06-27

## 报告路径

- 最新综合图文报告 README：`/Users/zhouning/paper58-geofm-world-model-rl/paper/rse_submission_paper58/paper58_true_geosos_flus_xiangzhen24_fixed_flus_external_loo_transition_floor030_same_grid_2026-06-27/README.md`
- 最新综合图文报告 HTML：`/Users/zhouning/paper58-geofm-world-model-rl/paper/rse_submission_paper58/paper58_true_geosos_flus_xiangzhen24_fixed_flus_external_loo_transition_floor030_same_grid_2026-06-27/report.html`
- 最新指标汇总 CSV：`/Users/zhouning/paper58-geofm-world-model-rl/paper/rse_submission_paper58/paper58_true_geosos_flus_xiangzhen24_fixed_flus_external_loo_transition_floor030_same_grid_2026-06-27/metric_summary_by_method.csv`
- 逐区域指标 CSV：`/Users/zhouning/paper58-geofm-world-model-rl/paper/rse_submission_paper58/paper58_true_geosos_flus_xiangzhen24_fixed_flus_external_loo_transition_floor030_same_grid_2026-06-27/metrics_by_method.csv`
- transition-aware 外部 LOO 参数选择结果：`/Users/zhouning/paper58-geofm-world-model-rl/paper/rse_submission_paper58/paper58_spatial_demand_ratio_external_loo_transition_floor030_tuning_2026-06-27`
- transition-aware 外部 LOO ratio-demand 预测集：`/Users/zhouning/paper58-geofm-world-model-rl/paper/rse_submission_paper58/paper58_spatial_demand_ratio_external_loo_transition_floor030_p25x15_min005_tw10_sp05_xiangzhen24_2026-06-27`
- 旧版外部 LOO ratio-demand 预测集：`/Users/zhouning/paper58-geofm-world-model-rl/paper/rse_submission_paper58/paper58_spatial_demand_ratio_external_loo_p25x15_min000_tw10_sp05_xiangzhen24_2026-06-27`
- 目标集诊断 ratio-demand 预测集：`/Users/zhouning/paper58-geofm-world-model-rl/paper/rse_submission_paper58/paper58_spatial_demand_ratio_p25x15_min005_tw10_sp05_xiangzhen24_2026-06-27`
- 5 次固定随机种子 GeoSOS-FLUS 复现实验：`/Users/zhouning/paper58-geofm-world-model-rl/paper/rse_submission_paper58/paper58_true_geosos_flus_xiangzhen24_fixed_flus_external_loo_transition_floor030_seeded_replicates_2026-06-27`
- 本总结：`/Users/zhouning/paper58-geofm-world-model-rl/paper/rse_submission_paper58/paper58_xiangzhen24_fixed_flus_optimization_summary_2026-06-27.md`

## 本轮数据

本轮仍使用真实行政区数据，不是模拟小样例。样本来自 `/Users/zhouning/Downloads/shp/xiangzhen.shp` 中分层抽取的 24 个真实乡镇边界，并通过 GEE 构建同网格输入：

- 起始/结束标签：ESRI LULC 2020/2021。
- Paper58 输入：AlphaEarth 2020 embedding、上下文栅格与同网格标签。
- 空间尺度：100 m。
- 有效像元：每个乡镇 3,176 到 37,126 个，均值 16,308 个。
- 真实变化比例：1.05% 到 22.67%，均值 7.64%。
- GeoSOS-FLUS：使用 `/Users/zhouning/FLUS_console_crossplatform/build/flus_console`，即用户确认的 GUI 同源 CLI 源码构建版本。

## 固定基线设计

为避免比较对象漂移，本轮继续采用固定 GeoSOS-FLUS 基线：用 raw Paper58 预测驱动一次 GeoSOS-FLUS，得到同一个 `geosos_flus_console` 参照；然后把多个 Paper58 变体作为额外方法加入同一份报告。

这一设计的含义是：GeoSOS-FLUS、Paper58 raw、change-aware、spatial-demand regression、旧版外部 LOO ratio-demand、transition-aware 外部 LOO ratio-demand 都在同一批标签和同一空间网格下比较。GeoSOS-FLUS 控制台使用随机分配项，因此本轮额外做了 5 次固定随机种子重复运行，用“每个种子的 24 区域总体均值”判断结论是否依赖单次随机结果。

本轮新增的关键改进是 transition-aware 外部 LOO 选择：参数仍只由 11 个外部 calibration 区 leave-one-area 结果选择，未使用 24 个目标乡镇结束年份真值调参；选择规则是在可达到的情况下先约束外部 LOO 平均转换准确率不低于 0.30，再最大化 FoM、降低 allocation disagreement、提高 F1。选中参数为 `ratio_quantile=0.25`、`ratio_multiplier=1.5`、`min_fraction=0.05`、`max_fraction=0.25`、`target_neighborhood_weight=1.0`、`source_neighborhood_penalty=0.5`。

## 最新主要结果

| 方法 | n | 变化 F1 | FoM | 转换准确率 | 分配分歧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GeoSOS-FLUS 固定基线 | 24 | 0.2635 | 0.1196 | 0.2808 | 0.0601 |
| Paper58 raw latent-dynamics | 24 | 0.2767 | 0.1141 | 0.3711 | 0.0873 |
| Paper58 change-aware + allocation loss | 24 | 0.2804 | 0.1133 | 0.3669 | 0.0831 |
| Paper58 spatial-demand regression holdout-cal | 24 | 0.2470 | 0.1056 | 0.2490 | 0.0650 |
| Paper58 spatial-demand ratio external LOO | 24 | 0.2871 | 0.1245 | 0.2702 | 0.0582 |
| Paper58 transition-aware external LOO | 24 | 0.2883 | 0.1248 | 0.2862 | 0.0584 |

最新结果说明：

- Paper58 raw 和 change-aware 分支继续表现出强语义优势：变化 F1 与转换准确率高于 GeoSOS-FLUS，但空间分配分歧偏高。
- spatial-demand regression 失败，主要原因是线性回归在目标乡镇上外推出过低变化预算，部分区域甚至被压到 0 变化，导致 F1、FoM 和转换准确率同时下降。
- 旧版 external LOO 已经证明非目标调参方向有效：F1、FoM、allocation 优于 GeoSOS-FLUS，但转换准确率仍低。
- transition-aware external LOO 进一步修正了这个短板。单次同网格报告中，它相对 GeoSOS-FLUS 的变化 F1 差值为 +0.0248，FoM 差值为 +0.0052，转换准确率差值为 +0.0054，分配分歧差值为 -0.0018；四项指标均优于 GeoSOS-FLUS。

必须明确：`paper58_spatial_demand_ratio_external_loo_transition_floor030` 已经把“外部非目标调参 + 同网格目标评估”的总体均值推进到 4/4 指标超过 GeoSOS-FLUS；这比上一版 external LOO 更强。但区域级配对胜率仍不是 100%，所以论文不能写成“所有区域、所有局部场景都已经全面超过 GeoSOS-FLUS”。更准确的表述是：在 24 个真实乡镇目标集上，Paper58 transition-aware spatial-demand 在总体均值层面已经给出严格非目标调参的全面胜出证据，同时仍存在区域异质性和失败案例。

## 5 次 GeoSOS-FLUS 随机复现

| 方法 | 复现次数 | 变化 F1 均值 | FoM 均值 | 转换准确率均值 | 分配分歧均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GeoSOS-FLUS 固定基线 | 5 | 0.2663 ± 0.0006 | 0.1206 ± 0.0007 | 0.2802 ± 0.0013 | 0.0587 ± 0.0007 |
| Paper58 raw latent-dynamics | 5 | 0.2767 ± 0.0000 | 0.1141 ± 0.0000 | 0.3711 ± 0.0000 | 0.0873 ± 0.0000 |
| Paper58 transition-aware external LOO | 5 | 0.2883 ± 0.0000 | 0.1248 ± 0.0000 | 0.2862 ± 0.0000 | 0.0584 ± 0.0000 |

复现结论：

- 按每个随机种子的 24 区域总体均值，transition-aware external LOO 在变化 F1、FoM、转换准确率 3 个指标上均为 5/5 种子胜出；分配分歧均值更低，但只在 3/5 个种子胜出。
- 按区域×种子配对，胜率仍混合：变化 F1 为 65/120，FoM 为 61/120，转换准确率为 54/120，分配分歧为 81/120。
- 因此，当前可以写成“Paper58 在严格非目标调参条件下，于总体均值层面稳定超过 GeoSOS-FLUS 的主要精度指标，并显著缩小空间分配短板”；不应写成“每个区域、每次局部配对都超过 GeoSOS-FLUS”。

## 图像示例

下图第一行展示起始年份真值、结束年份真值、Paper58 及其优化分支、GeoSOS-FLUS 输出；第二行是变化误差图。蓝色代表真实变化，绿色代表命中，红色代表漏判，金色代表误报。图上出现的 “error” 是误差图含义，不是程序报错。

### Paper58 ratio-demand 优势样本：`xiangzhen_record_000191`

![xiangzhen_record_000191](paper58_true_geosos_flus_xiangzhen24_fixed_flus_external_loo_transition_floor030_same_grid_2026-06-27/figures/same_grid_xiangzhen_record_000191.png)

该样本用于观察 ratio-demand 如何减少 raw Paper58 的过度变化，同时保留较高置信度变化像元。它比简单 keep gate 更有意义，因为排序同时考虑 Paper58 change score 与邻域支持。

### 高变化区域样本：`xiangzhen_record_002058`

![xiangzhen_record_002058](paper58_true_geosos_flus_xiangzhen24_fixed_flus_external_loo_transition_floor030_same_grid_2026-06-27/figures/same_grid_xiangzhen_record_002058.png)

高变化区域可以检验模型是否过度保守。regression-demand 在部分样本中过度压低变化预算，而 ratio-demand 通过候选变化比例保留机制避免把真实变化大量过滤掉。

### 低变化比例样本：`xiangzhen_record_021584`

![xiangzhen_record_021584](paper58_true_geosos_flus_xiangzhen24_fixed_flus_external_loo_transition_floor030_same_grid_2026-06-27/figures/same_grid_xiangzhen_record_021584.png)

低变化比例区域对误报特别敏感。ratio-demand 的实际意义是给 Paper58 加上一个需求预算和空间排序层，减少斑块发散与局部误报，使结果图更接近 GeoSOS-FLUS 的稳定成图风格。

## 算法原理差异

Paper58 当前路线是表征动力学模型：用 AlphaEarth embedding 表示起始地表状态，LatentDynamicsNet 预测下一时相 embedding，再通过 LULC decoder 输出类别图。新增 ratio-demand 模块不是调用 GeoSOS-FLUS，而是在 Paper58 输出上加入 Paper58-native 后处理：先用独立 calibration 区估计候选变化应保留的比例，再用 Paper58 change score、目标类别邻域支持、源类别邻域惩罚进行空间排序和预算裁剪。

GeoSOS-FLUS 是 CA 分配模型：核心优势在于从土地需求、适宜性概率、邻域影响、惯性系数和随机竞争中逐步分配未来土地类型。它不一定更懂语义转移，但很擅长把变化放在空间上更紧凑、更符合邻域扩张的位置。

当前最新结果说明，Paper58 的语义预测能力已经足够强；真正的突破点在于把 GeoSOS-FLUS 擅长的空间分配思想转化为 Paper58-native 的可学习、可校准模块。ratio-demand 是这个方向的第一条有效证据。

## 对实际用户的影响

如果用户关心“是否能识别真实变化，以及变化类别是否合理”，Paper58 raw/change-aware 已经有优势。

如果用户关心“地图成图是否空间稳定、斑块是否紧凑、误报是否少”，原始 Paper58 仍有短板；但 ratio-demand 结果显示，这个短板可以通过 Paper58-native 空间需求分配显著缓解。

换成实际使用语言：在没有 ratio-demand 之前，Paper58 更适合作为高语义质量预测核心；加入 transition-aware ratio-demand 后，Paper58 开始具备同时追求语义准确、变化检出和空间成图稳定的能力。当前结果对用户的直接含义是：如果用户看总体项目均值，Paper58 优化版已经比 GeoSOS-FLUS 更有竞争力；如果用户要求每个乡镇都稳胜，仍需要继续做区域级鲁棒性优化和失败区域诊断。

## 当前论文表述建议

现在可以在论文中新增一个方法小节：`Paper58 spatial demand allocation`。其中应包含：

1. calibration 区学习候选变化保留比例；
2. 目标区不读取结束年份真值；
3. 按 Paper58 change score 与邻域支持排序；
4. 按需求预算裁剪变化像元；
5. 与 GeoSOS-FLUS 固定基线同网格比较。

当前可以写的结论是：在 24 个真实乡镇的严格同网格实验中，Paper58 transition-aware spatial-demand ratio 使用 11 个外部 calibration 区完成非目标调参，并在目标集单次报告中实现 F1、FoM、transition accuracy、allocation disagreement 四项总体均值同时优于 GeoSOS-FLUS；在 5 次固定随机种子复现实验中，F1、FoM、transition accuracy 均为 5/5 种子胜出，allocation disagreement 平均更低但只在 3/5 个种子胜出。

当前还不应写的结论是：Paper58 已经在每个区域、每个随机种子、每个局部转移类型上最终全面超过 GeoSOS-FLUS。区域×种子配对胜率仍混合，论文需要保留失败区域分析。

## 下一步

1. 继续优化区域级鲁棒性，重点分析区域×种子中 F1、FoM、transition accuracy 输给 GeoSOS-FLUS 的样本。
2. 在外部 calibration score cases 上增加类别转移分层约束，避免总体 transition accuracy 提升但少数转移类型退化。
3. 继续扩展新增行政区样本，目标是不少于 48 个真实乡镇/县域，并保持 GeoSOS-FLUS 多随机种子复现。
4. 当 external LOO 或新增行政区验证中 4 项指标在总体均值和区域配对胜率上都稳定超过 GeoSOS-FLUS 后，再把“全面超越”写成论文主结论。
