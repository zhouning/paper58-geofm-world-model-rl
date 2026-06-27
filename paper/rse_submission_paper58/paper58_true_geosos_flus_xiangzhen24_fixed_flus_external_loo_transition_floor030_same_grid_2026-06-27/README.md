# 完整 Paper58 与 GeoSOS-FLUS 同网格严格对比报告

本报告使用完整 Paper58 latent-dynamics 预测图作为 Paper58 端结果：起始年份 AlphaEarth embedding 经 LatentDynamicsNet 预测下一年 embedding，再由 decoder 解码为 LULC 类别。评估时，Paper58、GeoSOS-FLUS、起始真值和结束真值都落在同一 ESRI/AlphaEarth 标签网格上。

证据边界：这是比 `paper58_proxy` 更严格的同网格严格对比，但这些 AlphaEarth/ESRI 区域没有 GeoSOS-FLUS 原生 ANN 驱动因子与训练样本。因此 GeoSOS-FLUS 的适宜性层由完整 Paper58 预测图转换为概率，需求量使用非真值来源推导；本实验主要比较完整 Paper58 预测与 GeoSOS-FLUS 控制台 CA 分配器在同网格条件下的结果。

## 运行说明

- Paper58 端使用 src/adk_world_model/weights/latent_dynamics_v1.pt 和 lulc_decoder_v1.pkl 生成的完整 latent-dynamics LULC 预测。
- GeoSOS-FLUS 控制台在同一 ESRI/AlphaEarth 标签网格上运行；适宜性层由完整 Paper58 预测图转换为 one-hot 概率，需求量来源为 `paper58_prediction`，不使用目标结束年份真值需求。
- 额外方法 `paper58_changeaware_alloc` 从 `paper/rse_submission_paper58/paper58_semantic_changeaware_alloc0020_xiangzhen_stratified24_s0005_chg0040_cw11x2_2026-06-27/predictions` 读取，用于评估 Paper58 优化后输出。
- 额外方法 `paper58_spatial_demand_ratio_external_loo` 从 `paper/rse_submission_paper58/paper58_spatial_demand_ratio_external_loo_p25x15_min000_tw10_sp05_xiangzhen24_2026-06-27/predictions` 读取，用于评估 Paper58 优化后输出。
- 额外方法 `paper58_spatial_demand_ratio_external_loo_transition_floor030` 从 `paper/rse_submission_paper58/paper58_spatial_demand_ratio_external_loo_transition_floor030_p25x15_min005_tw10_sp05_xiangzhen24_2026-06-27/predictions` 读取，用于评估 Paper58 优化后输出。
- 额外方法 `paper58_spatial_demand_regression_holdoutcal` 从 `paper/rse_submission_paper58/paper58_spatial_demand_allocation_holdoutcal_changeaware_xiangzhen24_2026-06-27/predictions` 读取，用于评估 Paper58 优化后输出。

## 如何阅读误差图

- 第一行展示起始年份真值、结束年份真值、完整 Paper58 结果、额外 Paper58 优化结果和 GeoSOS-FLUS 结果。
- 第二行展示变化误差：蓝色是真实变化，绿色是命中，红色是漏判，金色是误报。
- 绿色只代表变化位置命中；类别是否正确需要结合 `FoM` 和 `转换准确率`。

## 核心判读与证据边界

本报告中最关键的 Paper58 优化分支是 `paper58_spatial_demand_ratio_external_loo_transition_floor030`。它的参数不是从 24 个目标乡镇真值中挑出来的，而是只使用 11 个外部 calibration 区做 leave-one-area 参数选择。选择规则是：在外部 LOO 平均转换准确率能够达到 0.30 的前提下，先筛掉转换准确率不足的参数，再最大化 FoM、降低分配分歧、提高变化 F1。

单次同网格目标集结果显示，`paper58_spatial_demand_ratio_external_loo_transition_floor030` 相对 `geosos_flus_console` 在 4 项总体均值上全部占优：变化 F1 +0.0248，FoM +0.0052，转换准确率 +0.0054，分配分歧 -0.0018（该指标越低越好）。

5 次固定随机种子复现实验进一步说明，这个结论不是单次 GeoSOS-FLUS 随机结果造成的偶然现象。按每个随机种子的 24 区域总体均值，Paper58 在变化 F1、FoM、转换准确率上均为 5/5 种子胜出；分配分歧平均更低，但只在 3/5 个种子胜出。

需要保留的科学边界是：区域级表现仍有异质性。按区域×种子配对，Paper58 的胜出次数分别为变化 F1 65/120、FoM 61/120、转换准确率 54/120、分配分歧 81/120。因此，本报告支持“Paper58 优化后在严格非目标调参条件下，于总体均值层面已经稳定超过 GeoSOS-FLUS 的主要精度指标”；但还不能表述为“每个区域、每个局部场景都已经超过 GeoSOS-FLUS”。

## 总体对比结论

- `变化 F1` 衡量模型是否找到了真实变化像元。`paper58_latent_dynamics` 均值=0.2767，`geosos_flus_console` 均值=0.2635，差值=+0.0132 (更优)。
- `FoM` 要求变化位置和目标类别同时正确。`paper58_latent_dynamics` 均值=0.1141，`geosos_flus_console` 均值=0.1196，差值=-0.0055 (更差)。
- `转换准确率` 只在真实变化像元上检查目标类别是否命中。`paper58_latent_dynamics` 均值=0.3711，`geosos_flus_console` 均值=0.2808，差值=+0.0902 (更优)。
- `分配分歧` 越低越好，表示空间位置和类别分配错配。`paper58_latent_dynamics` 均值=0.0873，`geosos_flus_console` 均值=0.0601，差值=+0.0272 (更差)。
- `paper58_changeaware_alloc` 是 Paper58 优化后输出；相对 GeoSOS-FLUS：变化 F1 差值=+0.0169 (更优)，FoM 差值=-0.0063 (更差)，转换准确率差值=+0.0861 (更优)，分配分歧差值=+0.0230 (更差)。四项指标中有 2/4 项优于 GeoSOS-FLUS。
- `paper58_spatial_demand_ratio_external_loo` 是 Paper58 优化后输出；相对 GeoSOS-FLUS：变化 F1 差值=+0.0236 (更优)，FoM 差值=+0.0049 (更优)，转换准确率差值=-0.0106 (更差)，分配分歧差值=-0.0020 (更优)。四项指标中有 3/4 项优于 GeoSOS-FLUS。
- `paper58_spatial_demand_ratio_external_loo_transition_floor030` 是 Paper58 优化后输出；相对 GeoSOS-FLUS：变化 F1 差值=+0.0248 (更优)，FoM 差值=+0.0052 (更优)，转换准确率差值=+0.0054 (更优)，分配分歧差值=-0.0018 (更优)。四项指标中有 4/4 项优于 GeoSOS-FLUS。
- `paper58_spatial_demand_regression_holdoutcal` 是 Paper58 优化后输出；相对 GeoSOS-FLUS：变化 F1 差值=-0.0165 (更差)，FoM 差值=-0.0140 (更差)，转换准确率差值=-0.0318 (更差)，分配分歧差值=+0.0049 (更差)。四项指标中有 0/4 项优于 GeoSOS-FLUS。

## 指标表

| 方法 | 区域 | 变化 F1 | FoM | 转换准确率 | 分配分歧 |
| --- | --- | ---: | ---: | ---: | ---: |
| `paper58_latent_dynamics` | `xiangzhen_record_000191` | 0.3760 | 0.1502 | 0.3740 | 0.0974 |
| `paper58_changeaware_alloc` | `xiangzhen_record_000191` | 0.4486 | 0.1928 | 0.4987 | 0.0507 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_000191` | 0.3583 | 0.1641 | 0.2987 | 0.0518 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_000191` | 0.3583 | 0.1641 | 0.2987 | 0.0518 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_000191` | 0.2961 | 0.1380 | 0.2104 | 0.0465 |
| `geosos_flus_console` | `xiangzhen_record_000191` | 0.1927 | 0.0717 | 0.1065 | 0.0481 |
| `paper58_latent_dynamics` | `xiangzhen_record_002058` | 0.2265 | 0.1037 | 0.3378 | 0.1143 |
| `paper58_changeaware_alloc` | `xiangzhen_record_002058` | 0.3741 | 0.2131 | 0.4440 | 0.1901 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_002058` | 0.3399 | 0.1893 | 0.2854 | 0.1254 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_002058` | 0.3399 | 0.1893 | 0.2854 | 0.1254 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_002058` | 0.3451 | 0.1929 | 0.3181 | 0.1526 |
| `geosos_flus_console` | `xiangzhen_record_002058` | 0.2944 | 0.1577 | 0.2873 | 0.0814 |
| `paper58_latent_dynamics` | `xiangzhen_record_002815` | 0.3584 | 0.0907 | 0.2861 | 0.1140 |
| `paper58_changeaware_alloc` | `xiangzhen_record_002815` | 0.2738 | 0.0679 | 0.2112 | 0.1183 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_002815` | 0.2765 | 0.0874 | 0.1778 | 0.0716 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_002815` | 0.2765 | 0.0874 | 0.1778 | 0.0716 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_002815` | 0.2711 | 0.0824 | 0.1791 | 0.0774 |
| `geosos_flus_console` | `xiangzhen_record_002815` | 0.3211 | 0.1209 | 0.2620 | 0.1087 |
| `paper58_latent_dynamics` | `xiangzhen_record_003053` | 0.1441 | 0.0776 | 0.4146 | 0.0391 |
| `paper58_changeaware_alloc` | `xiangzhen_record_003053` | 0.1119 | 0.0593 | 0.4878 | 0.0104 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_003053` | 0.1453 | 0.0783 | 0.3659 | 0.0069 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_003053` | 0.1402 | 0.0754 | 0.3659 | 0.0069 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_003053` | 0.0000 | 0.0000 | 0.0000 | 0.0119 |
| `geosos_flus_console` | `xiangzhen_record_003053` | 0.1538 | 0.0833 | 0.4146 | 0.0347 |
| `paper58_latent_dynamics` | `xiangzhen_record_003882` | 0.3261 | 0.1487 | 0.4064 | 0.0807 |
| `paper58_changeaware_alloc` | `xiangzhen_record_003882` | 0.3029 | 0.1071 | 0.3207 | 0.0623 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_003882` | 0.3064 | 0.1246 | 0.2449 | 0.0455 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_003882` | 0.3064 | 0.1246 | 0.2449 | 0.0455 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_003882` | 0.2490 | 0.1114 | 0.1676 | 0.0394 |
| `geosos_flus_console` | `xiangzhen_record_003882` | 0.3353 | 0.1599 | 0.3677 | 0.0637 |
| `paper58_latent_dynamics` | `xiangzhen_record_010308` | 0.2954 | 0.1101 | 0.2378 | 0.0917 |
| `paper58_changeaware_alloc` | `xiangzhen_record_010308` | 0.3091 | 0.1264 | 0.3862 | 0.1675 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_010308` | 0.2953 | 0.1213 | 0.2445 | 0.1165 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_010308` | 0.2953 | 0.1213 | 0.2445 | 0.1165 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_010308` | 0.3078 | 0.1276 | 0.3373 | 0.1446 |
| `geosos_flus_console` | `xiangzhen_record_010308` | 0.3003 | 0.1219 | 0.2327 | 0.0767 |
| `paper58_latent_dynamics` | `xiangzhen_record_015068` | 0.4304 | 0.2036 | 0.4838 | 0.1102 |
| `paper58_changeaware_alloc` | `xiangzhen_record_015068` | 0.3564 | 0.1547 | 0.3576 | 0.1341 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_015068` | 0.2759 | 0.1297 | 0.2176 | 0.0651 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_015068` | 0.2759 | 0.1297 | 0.2176 | 0.0651 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_015068` | 0.2817 | 0.1323 | 0.2269 | 0.0701 |
| `geosos_flus_console` | `xiangzhen_record_015068` | 0.4268 | 0.2207 | 0.3831 | 0.0767 |
| `paper58_latent_dynamics` | `xiangzhen_record_019130` | 0.2404 | 0.0525 | 0.2439 | 0.0591 |
| `paper58_changeaware_alloc` | `xiangzhen_record_019130` | 0.2242 | 0.0471 | 0.2276 | 0.0791 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_019130` | 0.2647 | 0.0650 | 0.1870 | 0.0524 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_019130` | 0.2647 | 0.0650 | 0.1870 | 0.0524 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_019130` | 0.2717 | 0.0596 | 0.1789 | 0.0491 |
| `geosos_flus_console` | `xiangzhen_record_019130` | 0.2371 | 0.0673 | 0.1870 | 0.0454 |
| `paper58_latent_dynamics` | `xiangzhen_record_019254` | 0.2220 | 0.0728 | 0.1308 | 0.0862 |
| `paper58_changeaware_alloc` | `xiangzhen_record_019254` | 0.1976 | 0.0500 | 0.0939 | 0.0609 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_019254` | 0.1291 | 0.0392 | 0.0575 | 0.0357 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_019254` | 0.1291 | 0.0392 | 0.0575 | 0.0357 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_019254` | 0.0448 | 0.0180 | 0.0202 | 0.0352 |
| `geosos_flus_console` | `xiangzhen_record_019254` | 0.1433 | 0.0499 | 0.0728 | 0.0460 |
| `paper58_latent_dynamics` | `xiangzhen_record_020866` | 0.3589 | 0.1658 | 0.4521 | 0.1014 |
| `paper58_changeaware_alloc` | `xiangzhen_record_020866` | 0.3379 | 0.1149 | 0.3653 | 0.1370 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_020866` | 0.3535 | 0.1433 | 0.2920 | 0.1044 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_020866` | 0.3535 | 0.1433 | 0.2920 | 0.1044 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_020866` | 0.3644 | 0.1338 | 0.3371 | 0.1204 |
| `geosos_flus_console` | `xiangzhen_record_020866` | 0.3620 | 0.1742 | 0.4318 | 0.0893 |
| `paper58_latent_dynamics` | `xiangzhen_record_021584` | 0.0559 | 0.0245 | 0.3303 | 0.0208 |
| `paper58_changeaware_alloc` | `xiangzhen_record_021584` | 0.3013 | 0.1700 | 0.4924 | 0.0083 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_021584` | 0.2710 | 0.1520 | 0.2966 | 0.0057 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_021584` | 0.3013 | 0.1700 | 0.4924 | 0.0083 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_021584` | 0.3013 | 0.1700 | 0.4924 | 0.0083 |
| `geosos_flus_console` | `xiangzhen_record_021584` | 0.2790 | 0.1509 | 0.2875 | 0.0135 |
| `paper58_latent_dynamics` | `xiangzhen_record_021766` | 0.2737 | 0.1424 | 0.4410 | 0.2159 |
| `paper58_changeaware_alloc` | `xiangzhen_record_021766` | 0.3118 | 0.1662 | 0.4610 | 0.2496 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_021766` | 0.3299 | 0.1769 | 0.3241 | 0.1501 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_021766` | 0.3299 | 0.1769 | 0.3241 | 0.1501 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_021766` | 0.3175 | 0.1680 | 0.3608 | 0.1811 |
| `geosos_flus_console` | `xiangzhen_record_021766` | 0.1622 | 0.0760 | 0.1102 | 0.0553 |
| `paper58_latent_dynamics` | `xiangzhen_record_022880` | 0.1753 | 0.0873 | 0.4171 | 0.1125 |
| `paper58_changeaware_alloc` | `xiangzhen_record_022880` | 0.2243 | 0.1204 | 0.3775 | 0.0115 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_022880` | 0.2580 | 0.1481 | 0.2995 | 0.0107 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_022880` | 0.2591 | 0.1467 | 0.3490 | 0.0081 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_022880` | 0.2696 | 0.1541 | 0.3416 | 0.0061 |
| `geosos_flus_console` | `xiangzhen_record_022880` | 0.1371 | 0.0728 | 0.1139 | 0.0355 |
| `paper58_latent_dynamics` | `xiangzhen_record_025932` | 0.4943 | 0.2162 | 0.3727 | 0.2028 |
| `paper58_changeaware_alloc` | `xiangzhen_record_025932` | 0.5379 | 0.2295 | 0.4555 | 0.0907 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_025932` | 0.4757 | 0.1996 | 0.2883 | 0.1640 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_025932` | 0.4757 | 0.1996 | 0.2883 | 0.1640 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_025932` | 0.5304 | 0.2333 | 0.4233 | 0.1108 |
| `geosos_flus_console` | `xiangzhen_record_025932` | 0.4936 | 0.2224 | 0.3436 | 0.1892 |
| `paper58_latent_dynamics` | `xiangzhen_record_027267` | 0.2837 | 0.0773 | 0.2116 | 0.0313 |
| `paper58_changeaware_alloc` | `xiangzhen_record_027267` | 0.3087 | 0.0998 | 0.3149 | 0.0311 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_027267` | 0.3250 | 0.1153 | 0.2343 | 0.0303 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_027267` | 0.3383 | 0.1160 | 0.2519 | 0.0301 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_027267` | 0.3232 | 0.1058 | 0.3123 | 0.0303 |
| `geosos_flus_console` | `xiangzhen_record_027267` | 0.2486 | 0.0737 | 0.1549 | 0.0250 |
| `paper58_latent_dynamics` | `xiangzhen_record_028657` | 0.2950 | 0.1148 | 0.3953 | 0.0383 |
| `paper58_changeaware_alloc` | `xiangzhen_record_028657` | 0.2289 | 0.0757 | 0.3290 | 0.0240 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_028657` | 0.2763 | 0.1092 | 0.2861 | 0.0219 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_028657` | 0.2720 | 0.1032 | 0.3043 | 0.0225 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_028657` | 0.2289 | 0.0757 | 0.3290 | 0.0240 |
| `geosos_flus_console` | `xiangzhen_record_028657` | 0.2792 | 0.1170 | 0.3160 | 0.0317 |
| `paper58_latent_dynamics` | `xiangzhen_record_030462` | 0.4608 | 0.1805 | 0.4763 | 0.0230 |
| `paper58_changeaware_alloc` | `xiangzhen_record_030462` | 0.0941 | 0.0090 | 0.1580 | 0.0167 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_030462` | 0.1676 | 0.0169 | 0.1558 | 0.0152 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_030462` | 0.1676 | 0.0169 | 0.1558 | 0.0152 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_030462` | 0.1231 | 0.0119 | 0.1558 | 0.0159 |
| `geosos_flus_console` | `xiangzhen_record_030462` | 0.2961 | 0.1350 | 0.2280 | 0.0249 |
| `paper58_latent_dynamics` | `xiangzhen_record_031108` | 0.3186 | 0.1513 | 0.5070 | 0.0891 |
| `paper58_changeaware_alloc` | `xiangzhen_record_031108` | 0.2893 | 0.1127 | 0.4140 | 0.2214 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_031108` | 0.3353 | 0.1408 | 0.3186 | 0.1067 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_031108` | 0.3353 | 0.1408 | 0.3186 | 0.1067 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_031108` | 0.3079 | 0.1229 | 0.3674 | 0.1591 |
| `geosos_flus_console` | `xiangzhen_record_031108` | 0.3413 | 0.1730 | 0.4907 | 0.0811 |
| `paper58_latent_dynamics` | `xiangzhen_record_031426` | 0.2576 | 0.1057 | 0.3608 | 0.2593 |
| `paper58_changeaware_alloc` | `xiangzhen_record_031426` | 0.2866 | 0.1064 | 0.3736 | 0.1987 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_031426` | 0.3179 | 0.1325 | 0.2917 | 0.1172 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_031426` | 0.3179 | 0.1325 | 0.2917 | 0.1172 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_031426` | 0.3082 | 0.1228 | 0.3369 | 0.1500 |
| `geosos_flus_console` | `xiangzhen_record_031426` | 0.2811 | 0.1364 | 0.2939 | 0.1495 |
| `paper58_latent_dynamics` | `xiangzhen_record_031513` | 0.3040 | 0.1321 | 0.3590 | 0.0297 |
| `paper58_changeaware_alloc` | `xiangzhen_record_031513` | 0.3492 | 0.1346 | 0.3590 | 0.0165 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_031513` | 0.3953 | 0.1823 | 0.3162 | 0.0171 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_031513` | 0.3803 | 0.1739 | 0.3419 | 0.0168 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_031513` | 0.0000 | 0.0000 | 0.0000 | 0.0261 |
| `geosos_flus_console` | `xiangzhen_record_031513` | 0.2680 | 0.1094 | 0.2479 | 0.0243 |
| `paper58_latent_dynamics` | `xiangzhen_record_033687` | 0.1743 | 0.0740 | 0.4301 | 0.0501 |
| `paper58_changeaware_alloc` | `xiangzhen_record_033687` | 0.1733 | 0.0784 | 0.4191 | 0.0332 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_033687` | 0.1871 | 0.0826 | 0.2647 | 0.0132 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_033687` | 0.1952 | 0.0890 | 0.3419 | 0.0177 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_033687` | 0.1918 | 0.0895 | 0.3971 | 0.0207 |
| `geosos_flus_console` | `xiangzhen_record_033687` | 0.1585 | 0.0748 | 0.3162 | 0.0372 |
| `paper58_latent_dynamics` | `xiangzhen_record_038533` | 0.1448 | 0.0606 | 0.2941 | 0.0635 |
| `paper58_changeaware_alloc` | `xiangzhen_record_038533` | 0.1138 | 0.0479 | 0.3529 | 0.0272 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_038533` | 0.1406 | 0.0664 | 0.2810 | 0.0171 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_038533` | 0.1406 | 0.0664 | 0.2810 | 0.0171 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_038533` | 0.1246 | 0.0601 | 0.1242 | 0.0164 |
| `geosos_flus_console` | `xiangzhen_record_038533` | 0.1362 | 0.0615 | 0.2092 | 0.0419 |
| `paper58_latent_dynamics` | `xiangzhen_record_038743` | 0.1735 | 0.0876 | 0.5026 | 0.0201 |
| `paper58_changeaware_alloc` | `xiangzhen_record_038743` | 0.1671 | 0.0833 | 0.5026 | 0.0209 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_038743` | 0.2108 | 0.1148 | 0.4021 | 0.0154 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_038743` | 0.2108 | 0.1148 | 0.4021 | 0.0154 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_038743` | 0.0296 | 0.0150 | 0.0159 | 0.0278 |
| `geosos_flus_console` | `xiangzhen_record_038743` | 0.1729 | 0.0890 | 0.5026 | 0.0173 |
| `paper58_latent_dynamics` | `xiangzhen_record_042889` | 0.2519 | 0.1079 | 0.4404 | 0.0447 |
| `paper58_changeaware_alloc` | `xiangzhen_record_042889` | 0.4059 | 0.1526 | 0.4025 | 0.0347 |
| `paper58_spatial_demand_ratio_external_loo` | `xiangzhen_record_042889` | 0.4549 | 0.2087 | 0.3557 | 0.0363 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_042889` | 0.4549 | 0.2087 | 0.3557 | 0.0363 |
| `paper58_spatial_demand_regression_holdoutcal` | `xiangzhen_record_042889` | 0.4411 | 0.2093 | 0.3435 | 0.0375 |
| `geosos_flus_console` | `xiangzhen_record_042889` | 0.3031 | 0.1508 | 0.3797 | 0.0459 |

## 原始输入与结果图

### `xiangzhen_record_000191`

![xiangzhen_record_000191](figures/same_grid_xiangzhen_record_000191.png)

#### 单区域判读

真实变化像元 385 个。完整 Paper58 预测变化像元 796 个，GeoSOS-FLUS 预测变化像元 248 个。完整 Paper58 的变化 F1 差值 +0.1832，FoM 差值 +0.0785；分配分歧差值 +0.0492（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_changeaware_alloc` 预测变化像元 899 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.2559，FoM 差值 +0.1211，转换准确率差值 +0.3922，分配分歧差值 +0.0026（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 469 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1656，FoM 差值 +0.0924，转换准确率差值 +0.1922，分配分歧差值 +0.0037（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 469 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1656，FoM 差值 +0.0924，转换准确率差值 +0.1922，分配分歧差值 +0.0037（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 304 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1033，FoM 差值 +0.0663，转换准确率差值 +0.1039，分配分歧差值 -0.0016（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+242 像元），最大的低估是类别 7（-140 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 7（+527 像元），最大的低估是类别 2（-266 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 7（+221 像元），最大的低估是类别 2（-157 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+221 像元），最大的低估是类别 2（-157 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 5（+140 像元），最大的低估是类别 2（-134 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+227 像元），最大的低估是类别 7（-149 像元）。

### `xiangzhen_record_002058`

![xiangzhen_record_002058](figures/same_grid_xiangzhen_record_002058.png)

#### 单区域判读

真实变化像元 4,678 个。完整 Paper58 预测变化像元 12,499 个，GeoSOS-FLUS 预测变化像元 5,314 个。完整 Paper58 的变化 F1 差值 -0.0680，FoM 差值 -0.0540；分配分歧差值 +0.0329（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_changeaware_alloc` 预测变化像元 7,313 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0797，FoM 差值 +0.0553，转换准确率差值 +0.1567，分配分歧差值 +0.1087（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 3,819 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0454，FoM 差值 +0.0316，转换准确率差值 -0.0019，分配分歧差值 +0.0440（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 3,819 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0454，FoM 差值 +0.0316，转换准确率差值 -0.0019，分配分歧差值 +0.0440（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 4,642 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0506，FoM 差值 +0.0352，转换准确率差值 +0.0308，分配分歧差值 +0.0712（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+7,942 像元），最大的低估是类别 11（-9,263 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 11（+1,314 像元），最大的低估是类别 7（-1,377 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 2（+1,739 像元），最大的低估是类别 7（-880 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 2（+1,739 像元），最大的低估是类别 7（-880 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 2（+1,382 像元），最大的低估是类别 7（-1,007 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 2（+2,861 像元），最大的低估是类别 11（-4,076 像元）。

### `xiangzhen_record_002815`

![xiangzhen_record_002815](figures/same_grid_xiangzhen_record_002815.png)

#### 单区域判读

真实变化像元 748 个。完整 Paper58 预测变化像元 2,126 个，GeoSOS-FLUS 预测变化像元 1,183 个。完整 Paper58 的变化 F1 差值 +0.0373，FoM 差值 -0.0302；分配分歧差值 +0.0054（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_changeaware_alloc` 预测变化像元 1,947 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0472，FoM 差值 -0.0530，转换准确率差值 -0.0508，分配分歧差值 +0.0096（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 1,017 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0446，FoM 差值 -0.0335，转换准确率差值 -0.0842，分配分歧差值 -0.0371（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,017 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0446，FoM 差值 -0.0335，转换准确率差值 -0.0842，分配分歧差值 -0.0371（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 1,133 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0499，FoM 差值 -0.0385，转换准确率差值 -0.0829，分配分歧差值 -0.0312（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+1,105 像元），最大的低估是类别 2（-685 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 2（+1,301 像元），最大的低估是类别 11（-705 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 2（+901 像元），最大的低估是类别 11（-625 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 2（+901 像元），最大的低估是类别 11（-625 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 2（+978 像元），最大的低估是类别 11（-674 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+572 像元），最大的低估是类别 11（-414 像元）。

### `xiangzhen_record_003053`

![xiangzhen_record_003053](figures/same_grid_xiangzhen_record_003053.png)

#### 单区域判读

真实变化像元 82 个。完整 Paper58 预测变化像元 390 个，GeoSOS-FLUS 预测变化像元 360 个。完整 Paper58 的变化 F1 差值 -0.0098，FoM 差值 -0.0057；分配分歧差值 +0.0043（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_changeaware_alloc` 预测变化像元 633 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0420，FoM 差值 -0.0241，转换准确率差值 +0.0732，分配分歧差值 -0.0243（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 331 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0086，FoM 差值 -0.0050，转换准确率差值 -0.0488，分配分歧差值 -0.0278（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 346 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0137，FoM 差值 -0.0080，转换准确率差值 -0.0488，分配分歧差值 -0.0278（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 0 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.1538，FoM 差值 -0.0833，转换准确率差值 -0.4146，分配分歧差值 -0.0229（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+134 像元），最大的低估是类别 7（-134 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 5（+563 像元），最大的低估是类别 7（-563 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 5（+305 像元），最大的低估是类别 7（-305 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+320 像元），最大的低估是类别 7（-320 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 5（+0 像元），最大的低估是类别 5（+0 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+134 像元），最大的低估是类别 7（-134 像元）。

### `xiangzhen_record_003882`

![xiangzhen_record_003882](figures/same_grid_xiangzhen_record_003882.png)

#### 单区域判读

真实变化像元 1,319 个。完整 Paper58 预测变化像元 2,987 个，GeoSOS-FLUS 预测变化像元 2,325 个。完整 Paper58 的变化 F1 差值 -0.0093，FoM 差值 -0.0112；分配分歧差值 +0.0170（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_changeaware_alloc` 预测变化像元 3,336 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0324，FoM 差值 -0.0528，转换准确率差值 -0.0470，分配分歧差值 -0.0014（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 1,742 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0289，FoM 差值 -0.0353，转换准确率差值 -0.1228，分配分歧差值 -0.0182（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,742 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0289，FoM 差值 -0.0353，转换准确率差值 -0.1228，分配分歧差值 -0.0182（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 946 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0863，FoM 差值 -0.0485，转换准确率差值 -0.2002，分配分歧差值 -0.0243（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 2（+877 像元），最大的低估是类别 11（-590 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 7（+1,835 像元），最大的低估是类别 5（-973 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 7（+768 像元），最大的低估是类别 11（-577 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+768 像元），最大的低估是类别 11（-577 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 7（+391 像元），最大的低估是类别 11（-462 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 2（+819 像元），最大的低估是类别 11（-562 像元）。

### `xiangzhen_record_010308`

![xiangzhen_record_010308](figures/same_grid_xiangzhen_record_010308.png)

#### 单区域判读

真实变化像元 593 个。完整 Paper58 预测变化像元 910 个，GeoSOS-FLUS 预测变化像元 739 个。完整 Paper58 的变化 F1 差值 -0.0049，FoM 差值 -0.0118；分配分歧差值 +0.0149（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_changeaware_alloc` 预测变化像元 1,549 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0088，FoM 差值 +0.0045，转换准确率差值 +0.1535，分配分歧差值 +0.0908（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 809 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0050，FoM 差值 -0.0006，转换准确率差值 +0.0118，分配分歧差值 +0.0397（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 809 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0050，FoM 差值 -0.0006，转换准确率差值 +0.0118，分配分歧差值 +0.0397（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 1,259 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0075，FoM 差值 +0.0057，转换准确率差值 +0.1046，分配分歧差值 +0.0679（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+634 像元），最大的低估是类别 11（-379 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 11（+483 像元），最大的低估是类别 7（-743 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 5（+452 像元），最大的低估是类别 7（-460 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+452 像元），最大的低估是类别 7（-460 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 5（+361 像元），最大的低估是类别 7（-638 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+610 像元），最大的低估是类别 11（-363 像元）。

### `xiangzhen_record_015068`

![xiangzhen_record_015068](figures/same_grid_xiangzhen_record_015068.png)

#### 单区域判读

真实变化像元 864 个。完整 Paper58 预测变化像元 1,752 个，GeoSOS-FLUS 预测变化像元 1,043 个。完整 Paper58 的变化 F1 差值 +0.0036，FoM 差值 -0.0171；分配分歧差值 +0.0335（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_changeaware_alloc` 预测变化像元 1,566 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0705，FoM 差值 -0.0659，转换准确率差值 -0.0255，分配分歧差值 +0.0573（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 818 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.1510，FoM 差值 -0.0910，转换准确率差值 -0.1655，分配分歧差值 -0.0116（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 818 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.1510，FoM 差值 -0.0910，转换准确率差值 -0.1655，分配分歧差值 -0.0116（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 861 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.1451，FoM 差值 -0.0884，转换准确率差值 -0.1562，分配分歧差值 -0.0066（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 2（+412 像元），最大的低估是类别 7（-667 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 5（+469 像元），最大的低估是类别 2（-532 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 5（+719 像元），最大的低估是类别 2（-432 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+719 像元），最大的低估是类别 2（-432 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 5（+701 像元），最大的低估是类别 2（-460 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+288 像元），最大的低估是类别 7（-473 像元）。

### `xiangzhen_record_019130`

![xiangzhen_record_019130](figures/same_grid_xiangzhen_record_019130.png)

#### 单区域判读

真实变化像元 123 个。完整 Paper58 预测变化像元 526 个，GeoSOS-FLUS 预测变化像元 265 个。完整 Paper58 的变化 F1 差值 +0.0033，FoM 差值 -0.0147；分配分歧差值 +0.0136（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_changeaware_alloc` 预测变化像元 546 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0129，FoM 差值 -0.0201，转换准确率差值 +0.0407，分配分歧差值 +0.0336（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 285 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0276，FoM 差值 -0.0023，转换准确率差值 +0.0000，分配分歧差值 +0.0070（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 285 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0276，FoM 差值 -0.0023，转换准确率差值 +0.0000，分配分歧差值 +0.0070（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 304 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0345，FoM 差值 -0.0076，转换准确率差值 -0.0081，分配分歧差值 +0.0036（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+149 像元），最大的低估是类别 11（-342 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 5（+135 像元），最大的低估是类别 11（-301 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 7（+77 像元），最大的低估是类别 11（-154 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+77 像元），最大的低估是类别 11（-154 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 5（+78 像元），最大的低估是类别 11（-181 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 7（+85 像元），最大的低估是类别 11（-165 像元）。

### `xiangzhen_record_019254`

![xiangzhen_record_019254](figures/same_grid_xiangzhen_record_019254.png)

#### 单区域判读

真实变化像元 8,037 个。完整 Paper58 预测变化像元 8,208 个，GeoSOS-FLUS 预测变化像元 4,594 个。完整 Paper58 的变化 F1 差值 +0.0787，FoM 差值 +0.0229；分配分歧差值 +0.0402（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_changeaware_alloc` 预测变化像元 8,724 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0543，FoM 差值 +0.0001，转换准确率差值 +0.0212，分配分歧差值 +0.0149（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 4,556 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0142，FoM 差值 -0.0107，转换准确率差值 -0.0153，分配分歧差值 -0.0102（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 4,556 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0142，FoM 差值 -0.0107，转换准确率差值 -0.0153，分配分歧差值 -0.0102（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 1,160 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0985，FoM 差值 -0.0319，转换准确率差值 -0.0526，分配分歧差值 -0.0108（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 2（+5,769 像元），最大的低估是类别 11（-9,528 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 2（+6,306 像元），最大的低估是类别 11（-10,805 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 2（+6,733 像元），最大的低估是类别 11（-9,219 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 2（+6,733 像元），最大的低估是类别 11（-9,219 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 2（+6,311 像元），最大的低估是类别 11（-7,449 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 2（+5,769 像元），最大的低估是类别 11（-8,753 像元）。

### `xiangzhen_record_020866`

![xiangzhen_record_020866](figures/same_grid_xiangzhen_record_020866.png)

#### 单区域判读

真实变化像元 887 个。完整 Paper58 预测变化像元 2,061 个，GeoSOS-FLUS 预测变化像元 1,798 个。完整 Paper58 的变化 F1 差值 -0.0031，FoM 差值 -0.0084；分配分歧差值 +0.0121（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_changeaware_alloc` 预测变化像元 2,505 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0242，FoM 差值 -0.0592，转换准确率差值 -0.0665，分配分歧差值 +0.0476（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 1,308 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0085，FoM 差值 -0.0308，转换准确率差值 -0.1398，分配分歧差值 +0.0151（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,308 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0085，FoM 差值 -0.0308，转换准确率差值 -0.1398，分配分歧差值 +0.0151（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 1,846 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0024，FoM 差值 -0.0404，转换准确率差值 -0.0947，分配分歧差值 +0.0311（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 2（+839 像元），最大的低估是类别 7（-423 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 7（+842 像元），最大的低估是类别 2（-615 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 5（+205 像元），最大的低估是类别 11（-213 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+205 像元），最大的低估是类别 11（-213 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 7（+420 像元），最大的低估是类别 2（-323 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 2（+773 像元），最大的低估是类别 7（-421 像元）。

### `xiangzhen_record_021584`

![xiangzhen_record_021584](figures/same_grid_xiangzhen_record_021584.png)

#### 单区域判读

真实变化像元 327 个。完整 Paper58 预测变化像元 4,217 个，GeoSOS-FLUS 预测变化像元 397 个。完整 Paper58 的变化 F1 差值 -0.2231，FoM 差值 -0.1264；分配分歧差值 +0.0072（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_changeaware_alloc` 预测变化像元 788 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0223，FoM 差值 +0.0191，转换准确率差值 +0.2049，分配分歧差值 -0.0052（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 411 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0080，FoM 差值 +0.0012，转换准确率差值 +0.0092，分配分歧差值 -0.0078（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 788 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0223，FoM 差值 +0.0191，转换准确率差值 +0.2049，分配分歧差值 -0.0052（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 788 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0223，FoM 差值 +0.0191，转换准确率差值 +0.2049，分配分歧差值 -0.0052（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+3,660 像元），最大的低估是类别 11（-3,481 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 11（+526 像元），最大的低估是类别 7（-317 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 11（+363 像元），最大的低估是类别 7（-165 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 11（+526 像元），最大的低估是类别 7（-317 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 11（+526 像元），最大的低估是类别 7（-317 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 11（+84 像元），最大的低估是类别 7（-59 像元）。

### `xiangzhen_record_021766`

![xiangzhen_record_021766](figures/same_grid_xiangzhen_record_021766.png)

#### 单区域判读

真实变化像元 898 个。完整 Paper58 预测变化像元 2,324 个，GeoSOS-FLUS 预测变化像元 520 个。完整 Paper58 的变化 F1 差值 +0.1115，FoM 差值 +0.0664；分配分歧差值 +0.1606（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_changeaware_alloc` 预测变化像元 2,053 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1496，FoM 差值 +0.0902，转换准确率差值 +0.3508，分配分歧差值 +0.1944（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 1,072 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1677，FoM 差值 +0.1009，转换准确率差值 +0.2138，分配分歧差值 +0.0948（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,072 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1677，FoM 差值 +0.1009，转换准确率差值 +0.2138，分配分歧差值 +0.0948（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 1,395 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1553，FoM 差值 +0.0920，转换准确率差值 +0.2506，分配分歧差值 +0.1259（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+1,342 像元），最大的低估是类别 11（-714 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 5（+871 像元），最大的低估是类别 7（-565 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 5（+629 像元），最大的低估是类别 7（-377 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+629 像元），最大的低估是类别 7（-377 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 5（+730 像元），最大的低估是类别 7（-447 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+934 像元），最大的低估是类别 11（-714 像元）。

### `xiangzhen_record_022880`

![xiangzhen_record_022880](figures/same_grid_xiangzhen_record_022880.png)

#### 单区域判读

真实变化像元 808 个。完整 Paper58 预测变化像元 3,425 个，GeoSOS-FLUS 预测变化像元 549 个。完整 Paper58 的变化 F1 差值 +0.0382，FoM 差值 +0.0145；分配分歧差值 +0.0770（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_changeaware_alloc` 预测变化像元 2,045 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0873，FoM 差值 +0.0476，转换准确率差值 +0.2636，分配分歧差值 -0.0239（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 1,068 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1209，FoM 差值 +0.0753，转换准确率差值 +0.1856，分配分歧差值 -0.0247（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,400 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1220，FoM 差值 +0.0739，转换准确率差值 +0.2351，分配分歧差值 -0.0274（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 1,262 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1325，FoM 差值 +0.0813，转换准确率差值 +0.2277，分配分歧差值 -0.0294（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 11（+210 像元），最大的低估是类别 7（-241 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 5（+1,896 像元），最大的低估是类别 11（-1,461 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 5（+1,080 像元），最大的低估是类别 11（-776 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+1,403 像元），最大的低估是类别 11（-1,083 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 5（+1,329 像元），最大的低估是类别 11（-1,089 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+145 像元），最大的低估是类别 11（-151 像元）。

### `xiangzhen_record_025932`

![xiangzhen_record_025932](figures/same_grid_xiangzhen_record_025932.png)

#### 单区域判读

真实变化像元 652 个。完整 Paper58 预测变化像元 841 个，GeoSOS-FLUS 预测变化像元 685 个。完整 Paper58 的变化 F1 差值 +0.0007，FoM 差值 -0.0063；分配分歧差值 +0.0135（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_changeaware_alloc` 预测变化像元 1,118 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0442，FoM 差值 +0.0071，转换准确率差值 +0.1120，分配分歧差值 -0.0986（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 584 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0179，FoM 差值 -0.0229，转换准确率差值 -0.0552，分配分歧差值 -0.0252（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 584 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0179，FoM 差值 -0.0229，转换准确率差值 -0.0552，分配分歧差值 -0.0252（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 958 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0368，FoM 差值 +0.0109，转换准确率差值 +0.0798，分配分歧差值 -0.0784（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 7（+219 像元），最大的低估是类别 2（-97 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 7（+709 像元），最大的低估是类别 5（-512 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 7（+216 像元），最大的低估是类别 5（-183 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+216 像元），最大的低估是类别 5（-183 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 7（+555 像元），最大的低估是类别 5（-436 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 7（+164 像元），最大的低估是类别 5（-75 像元）。

### `xiangzhen_record_027267`

![xiangzhen_record_027267](figures/same_grid_xiangzhen_record_027267.png)

#### 单区域判读

真实变化像元 794 个。完整 Paper58 预测变化像元 1,737 个，GeoSOS-FLUS 预测变化像元 1,113 个。完整 Paper58 的变化 F1 差值 +0.0351，FoM 差值 +0.0037；分配分歧差值 +0.0064（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_changeaware_alloc` 预测变化像元 2,167 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0601，FoM 差值 +0.0262，转换准确率差值 +0.1599，分配分歧差值 +0.0061（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 1,132 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0765，FoM 差值 +0.0417，转换准确率差值 +0.0793，分配分歧差值 +0.0053（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,281 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0898，FoM 差值 +0.0424，转换准确率差值 +0.0970，分配分歧差值 +0.0051（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 2,003 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0746，FoM 差值 +0.0321，转换准确率差值 +0.1574，分配分歧差值 +0.0053（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+1,178 像元），最大的低估是类别 7（-642 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 7（+1,011 像元），最大的低估是类别 2（-934 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 5（+328 像元），最大的低估是类别 11（-404 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+411 像元），最大的低估是类别 11（-418 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 7（+885 像元），最大的低估是类别 2（-794 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+884 像元），最大的低估是类别 7（-502 像元）。

### `xiangzhen_record_028657`

![xiangzhen_record_028657](figures/same_grid_xiangzhen_record_028657.png)

#### 单区域判读

真实变化像元 769 个。完整 Paper58 预测变化像元 2,336 个，GeoSOS-FLUS 预测变化像元 1,645 个。完整 Paper58 的变化 F1 差值 +0.0158，FoM 差值 -0.0021；分配分歧差值 +0.0067（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_changeaware_alloc` 预测变化像元 3,005 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0503，FoM 差值 -0.0413，转换准确率差值 +0.0130，分配分歧差值 -0.0077（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 1,569 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0029，FoM 差值 -0.0078，转换准确率差值 -0.0299，分配分歧差值 -0.0098（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,856 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0072，FoM 差值 -0.0138，转换准确率差值 -0.0117，分配分歧差值 -0.0092（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 3,005 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0503，FoM 差值 -0.0413，转换准确率差值 +0.0130，分配分歧差值 -0.0077（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+681 像元），最大的低估是类别 7（-465 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 7（+2,198 像元），最大的低估是类别 2（-1,529 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 7（+981 像元），最大的低估是类别 2（-401 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+1,198 像元），最大的低估是类别 2（-588 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 7（+2,198 像元），最大的低估是类别 2（-1,529 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+419 像元），最大的低估是类别 7（-342 像元）。

### `xiangzhen_record_030462`

![xiangzhen_record_030462](figures/same_grid_xiangzhen_record_030462.png)

#### 单区域判读

真实变化像元 443 个。完整 Paper58 预测变化像元 1,076 个，GeoSOS-FLUS 预测变化像元 435 个。完整 Paper58 的变化 F1 差值 +0.1647，FoM 差值 +0.0455；分配分歧差值 -0.0020（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_changeaware_alloc` 预测变化像元 7,698 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.2020，FoM 差值 -0.1260，转换准确率差值 -0.0700，分配分歧差值 -0.0082（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 4,020 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.1285，FoM 差值 -0.1182，转换准确率差值 -0.0722，分配分歧差值 -0.0097（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 4,020 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.1285，FoM 差值 -0.1182，转换准确率差值 -0.0722，分配分歧差值 -0.0097（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 5,715 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.1730，FoM 差值 -0.1231，转换准确率差值 -0.0722，分配分歧差值 -0.0091（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+573 像元），最大的低估是类别 7（-329 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 11（+4,487 像元），最大的低估是类别 2（-7,069 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 11（+2,086 像元），最大的低估是类别 2（-3,446 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 11（+2,086 像元），最大的低估是类别 2（-3,446 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 11（+3,214 像元），最大的低估是类别 2（-5,113 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+177 像元），最大的低估是类别 2（-151 像元）。

### `xiangzhen_record_031108`

![xiangzhen_record_031108](figures/same_grid_xiangzhen_record_031108.png)

#### 单区域判读

真实变化像元 430 个。完整 Paper58 预测变化像元 1,284 个，GeoSOS-FLUS 预测变化像元 1,041 个。完整 Paper58 的变化 F1 差值 -0.0227，FoM 差值 -0.0217；分配分歧差值 +0.0079（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_changeaware_alloc` 预测变化像元 1,416 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0520，FoM 差值 -0.0602，转换准确率差值 -0.0767，分配分歧差值 +0.1403（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 739 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0059，FoM 差值 -0.0321，转换准确率差值 -0.1721，分配分歧差值 +0.0256（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 739 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0059，FoM 差值 -0.0321，转换准确率差值 -0.1721，分配分歧差值 +0.0256（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 1,090 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0334，FoM 差值 -0.0501，转换准确率差值 -0.1233，分配分歧差值 +0.0780（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 11（+825 像元），最大的低估是类别 5（-692 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 2（+231 像元），最大的低估是类别 11（-420 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 2（+266 像元），最大的低估是类别 11（-361 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 2（+266 像元），最大的低估是类别 11（-361 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 2（+259 像元），最大的低估是类别 11（-422 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 11（+651 像元），最大的低估是类别 5（-546 像元）。

### `xiangzhen_record_031426`

![xiangzhen_record_031426](figures/same_grid_xiangzhen_record_031426.png)

#### 单区域判读

真实变化像元 3,600 个。完整 Paper58 预测变化像元 10,513 个，GeoSOS-FLUS 预测变化像元 5,422 个。完整 Paper58 的变化 F1 差值 -0.0235，FoM 差值 -0.0308；分配分歧差值 +0.1098（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_changeaware_alloc` 预测变化像元 11,154 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0055，FoM 差值 -0.0300，转换准确率差值 +0.0797，分配分歧差值 +0.0492（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 5,825 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0368，FoM 差值 -0.0040，转换准确率差值 -0.0022，分配分歧差值 -0.0323（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 5,825 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0368，FoM 差值 -0.0040，转换准确率差值 -0.0022，分配分歧差值 -0.0323（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 8,074 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0271，FoM 差值 -0.0136，转换准确率差值 +0.0431，分配分歧差值 +0.0005（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+1,642 像元），最大的低估是类别 11（-1,147 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 7（+3,722 像元），最大的低估是类别 5（-3,630 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 7（+1,674 像元），最大的低估是类别 5（-1,445 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+1,674 像元），最大的低估是类别 5（-1,445 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 7（+2,522 像元），最大的低估是类别 5（-2,418 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+1,116 像元），最大的低估是类别 11（-1,418 像元）。

### `xiangzhen_record_031513`

![xiangzhen_record_031513](figures/same_grid_xiangzhen_record_031513.png)

#### 单区域判读

真实变化像元 117 个。完整 Paper58 预测变化像元 258 个，GeoSOS-FLUS 预测变化像元 189 个。完整 Paper58 的变化 F1 差值 +0.0360，FoM 差值 +0.0226；分配分歧差值 +0.0054（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_changeaware_alloc` 预测变化像元 261 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0812，FoM 差值 +0.0252，转换准确率差值 +0.1111，分配分歧差值 -0.0078（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 136 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1273，FoM 差值 +0.0728，转换准确率差值 +0.0684，分配分歧差值 -0.0072（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 167 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1123，FoM 差值 +0.0645，转换准确率差值 +0.0940，分配分歧差值 -0.0075（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 0 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.2680，FoM 差值 -0.1094，转换准确率差值 -0.2479，分配分歧差值 +0.0018（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 7（+177 像元），最大的低估是类别 11（-135 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 7（+212 像元），最大的低估是类别 11（-135 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 7（+109 像元），最大的低估是类别 11（-74 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+133 像元），最大的低估是类别 11（-88 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 7（+14 像元），最大的低估是类别 11（-15 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 7（+155 像元），最大的低估是类别 11（-118 像元）。

### `xiangzhen_record_033687`

![xiangzhen_record_033687](figures/same_grid_xiangzhen_record_033687.png)

#### 单区域判读

真实变化像元 272 个。完整 Paper58 预测变化像元 1,461 个，GeoSOS-FLUS 预测变化像元 977 个。完整 Paper58 的变化 F1 差值 +0.0157，FoM 差值 -0.0008；分配分歧差值 +0.0129（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_changeaware_alloc` 预测变化像元 1,321 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0147，FoM 差值 +0.0036，转换准确率差值 +0.1029，分配分歧差值 -0.0041（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 690 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0286，FoM 差值 +0.0078，转换准确率差值 -0.0515，分配分歧差值 -0.0240（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 886 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0366，FoM 差值 +0.0142，转换准确率差值 +0.0257，分配分歧差值 -0.0195（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 1,063 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0332，FoM 差值 +0.0147，转换准确率差值 +0.0809，分配分歧差值 -0.0166（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+401 像元），最大的低估是类别 7（-556 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 5（+752 像元），最大的低估是类别 7（-692 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 5（+565 像元），最大的低估是类别 7（-500 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+637 像元），最大的低估是类别 7（-572 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 5（+732 像元），最大的低估是类别 7（-667 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+401 像元），最大的低估是类别 7（-346 像元）。

### `xiangzhen_record_038533`

![xiangzhen_record_038533](figures/same_grid_xiangzhen_record_038533.png)

#### 单区域判读

真实变化像元 153 个。完整 Paper58 预测变化像元 648 个，GeoSOS-FLUS 预测变化像元 405 个。完整 Paper58 的变化 F1 差值 +0.0086，FoM 差值 -0.0010；分配分歧差值 +0.0216（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_changeaware_alloc` 预测变化像元 1,042 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0224，FoM 差值 -0.0136，转换准确率差值 +0.1438，分配分歧差值 -0.0147（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 544 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0044，FoM 差值 +0.0048，转换准确率差值 +0.0719，分配分歧差值 -0.0248（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 544 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0044，FoM 差值 +0.0048，转换准确率差值 +0.0719，分配分歧差值 -0.0248（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 184 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0116，FoM 差值 -0.0014，转换准确率差值 -0.0850，分配分歧差值 -0.0255（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+81 像元），最大的低估是类别 7（-39 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 5（+809 像元），最大的低估是类别 7（-660 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 5（+439 像元），最大的低估是类别 7（-322 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+439 像元），最大的低估是类别 7（-322 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 5（+138 像元），最大的低估是类别 7（-66 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+81 像元），最大的低估是类别 7（-39 像元）。

### `xiangzhen_record_038743`

![xiangzhen_record_038743](figures/same_grid_xiangzhen_record_038743.png)

#### 单区域判读

真实变化像元 189 个。完整 Paper58 预测变化像元 998 个，GeoSOS-FLUS 预测变化像元 979 个。完整 Paper58 的变化 F1 差值 +0.0006，FoM 差值 -0.0014；分配分歧差值 +0.0027（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_changeaware_alloc` 预测变化像元 1,056 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0059，FoM 差值 -0.0058，转换准确率差值 +0.0000，分配分歧差值 +0.0035（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 551 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0379，FoM 差值 +0.0258，转换准确率差值 -0.1005，分配分歧差值 -0.0019（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 551 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0379，FoM 差值 +0.0258，转换准确率差值 -0.1005，分配分歧差值 -0.0019（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 14 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.1434，FoM 差值 -0.0740，转换准确率差值 -0.4868，分配分歧差值 +0.0104（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+864 像元），最大的低估是类别 7（-759 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 5（+916 像元），最大的低估是类别 7（-799 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 5（+490 像元），最大的低估是类别 7（-386 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+490 像元），最大的低估是类别 7（-386 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 7（+24 像元），最大的低估是类别 1（-23 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+864 像元），最大的低估是类别 7（-759 像元）。

### `xiangzhen_record_042889`

![xiangzhen_record_042889](figures/same_grid_xiangzhen_record_042889.png)

#### 单区域判读

真实变化像元 1,712 个。完整 Paper58 预测变化像元 6,282 个，GeoSOS-FLUS 预测变化像元 3,368 个。完整 Paper58 的变化 F1 差值 -0.0512，FoM 差值 -0.0429；分配分歧差值 -0.0012（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_changeaware_alloc` 预测变化像元 3,954 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1028，FoM 差值 +0.0018，转换准确率差值 +0.0228，分配分歧差值 -0.0112（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo` 预测变化像元 2,065 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1517，FoM 差值 +0.0579，转换准确率差值 -0.0239，分配分歧差值 -0.0096（该指标越低越好）。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 2,065 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1517，FoM 差值 +0.0579，转换准确率差值 -0.0239，分配分歧差值 -0.0096（该指标越低越好）。`paper58_spatial_demand_regression_holdoutcal` 预测变化像元 1,893 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1379，FoM 差值 +0.0584，转换准确率差值 -0.0362，分配分歧差值 -0.0084（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 11（+4,385 像元），最大的低估是类别 2（-4,124 像元）。paper58_changeaware_alloc 相对结束年份真值最大的高估是类别 5（+2,873 像元），最大的低估是类别 11（-1,031 像元）。paper58_spatial_demand_ratio_external_loo 相对结束年份真值最大的高估是类别 5（+1,312 像元），最大的低估是类别 11（-862 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+1,312 像元），最大的低估是类别 11（-862 像元）。paper58_spatial_demand_regression_holdoutcal 相对结束年份真值最大的高估是类别 5（+1,191 像元），最大的低估是类别 11（-798 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 11（+1,932 像元），最大的低估是类别 2（-1,842 像元）。
