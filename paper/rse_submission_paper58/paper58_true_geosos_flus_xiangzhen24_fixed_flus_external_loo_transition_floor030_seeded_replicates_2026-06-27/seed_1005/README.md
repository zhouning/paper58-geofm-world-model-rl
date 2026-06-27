# 完整 Paper58 与 GeoSOS-FLUS 同网格严格对比报告

本报告使用完整 Paper58 latent-dynamics 预测图作为 Paper58 端结果：起始年份 AlphaEarth embedding 经 LatentDynamicsNet 预测下一年 embedding，再由 decoder 解码为 LULC 类别。评估时，Paper58、GeoSOS-FLUS、起始真值和结束真值都落在同一 ESRI/AlphaEarth 标签网格上。

证据边界：这是比 `paper58_proxy` 更严格的同网格严格对比，但这些 AlphaEarth/ESRI 区域没有 GeoSOS-FLUS 原生 ANN 驱动因子与训练样本。因此 GeoSOS-FLUS 的适宜性层由完整 Paper58 预测图转换为概率，需求量使用非真值来源推导；本实验主要比较完整 Paper58 预测与 GeoSOS-FLUS 控制台 CA 分配器在同网格条件下的结果。

## 运行说明

- Paper58 端使用 src/adk_world_model/weights/latent_dynamics_v1.pt 和 lulc_decoder_v1.pkl 生成的完整 latent-dynamics LULC 预测。
- GeoSOS-FLUS 控制台在同一 ESRI/AlphaEarth 标签网格上运行；适宜性层由完整 Paper58 预测图转换为 one-hot 概率，需求量来源为 `paper58_prediction`，不使用目标结束年份真值需求。
- 额外方法 `paper58_spatial_demand_ratio_external_loo_transition_floor030` 从 `paper/rse_submission_paper58/paper58_spatial_demand_ratio_external_loo_transition_floor030_p25x15_min005_tw10_sp05_xiangzhen24_2026-06-27/predictions` 读取，用于评估 Paper58 优化后输出。

## 如何阅读误差图

- 第一行展示起始年份真值、结束年份真值、完整 Paper58 结果、额外 Paper58 优化结果和 GeoSOS-FLUS 结果。
- 第二行展示变化误差：蓝色是真实变化，绿色是命中，红色是漏判，金色是误报。
- 绿色只代表变化位置命中；类别是否正确需要结合 `FoM` 和 `转换准确率`。

## 总体对比结论

- `变化 F1` 衡量模型是否找到了真实变化像元。`paper58_latent_dynamics` 均值=0.2767，`geosos_flus_console` 均值=0.2664，差值=+0.0103 (更优)。
- `FoM` 要求变化位置和目标类别同时正确。`paper58_latent_dynamics` 均值=0.1141，`geosos_flus_console` 均值=0.1203，差值=-0.0062 (更差)。
- `转换准确率` 只在真实变化像元上检查目标类别是否命中。`paper58_latent_dynamics` 均值=0.3711，`geosos_flus_console` 均值=0.2804，差值=+0.0907 (更优)。
- `分配分歧` 越低越好，表示空间位置和类别分配错配。`paper58_latent_dynamics` 均值=0.0873，`geosos_flus_console` 均值=0.0582，差值=+0.0291 (更差)。
- `paper58_spatial_demand_ratio_external_loo_transition_floor030` 是 Paper58 优化后输出；相对 GeoSOS-FLUS：变化 F1 差值=+0.0219 (更优)，FoM 差值=+0.0045 (更优)，转换准确率差值=+0.0059 (更优)，分配分歧差值=+0.0002 (更差)。四项指标中有 3/4 项优于 GeoSOS-FLUS。

## 指标表

| 方法 | 区域 | 变化 F1 | FoM | 转换准确率 | 分配分歧 |
| --- | --- | ---: | ---: | ---: | ---: |
| `paper58_latent_dynamics` | `xiangzhen_record_000191` | 0.3760 | 0.1502 | 0.3740 | 0.0974 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_000191` | 0.3583 | 0.1641 | 0.2987 | 0.0518 |
| `geosos_flus_console` | `xiangzhen_record_000191` | 0.2244 | 0.0781 | 0.1091 | 0.0463 |
| `paper58_latent_dynamics` | `xiangzhen_record_002058` | 0.2265 | 0.1037 | 0.3378 | 0.1143 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_002058` | 0.3399 | 0.1893 | 0.2854 | 0.1254 |
| `geosos_flus_console` | `xiangzhen_record_002058` | 0.2924 | 0.1544 | 0.2843 | 0.0802 |
| `paper58_latent_dynamics` | `xiangzhen_record_002815` | 0.3584 | 0.0907 | 0.2861 | 0.1140 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_002815` | 0.2765 | 0.0874 | 0.1778 | 0.0716 |
| `geosos_flus_console` | `xiangzhen_record_002815` | 0.3247 | 0.1186 | 0.2553 | 0.1066 |
| `paper58_latent_dynamics` | `xiangzhen_record_003053` | 0.1441 | 0.0776 | 0.4146 | 0.0391 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_003053` | 0.1402 | 0.0754 | 0.3659 | 0.0069 |
| `geosos_flus_console` | `xiangzhen_record_003053` | 0.1592 | 0.0865 | 0.3902 | 0.0295 |
| `paper58_latent_dynamics` | `xiangzhen_record_003882` | 0.3261 | 0.1487 | 0.4064 | 0.0807 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_003882` | 0.3064 | 0.1246 | 0.2449 | 0.0455 |
| `geosos_flus_console` | `xiangzhen_record_003882` | 0.3343 | 0.1597 | 0.3685 | 0.0642 |
| `paper58_latent_dynamics` | `xiangzhen_record_010308` | 0.2954 | 0.1101 | 0.2378 | 0.0917 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_010308` | 0.2953 | 0.1213 | 0.2445 | 0.1165 |
| `geosos_flus_console` | `xiangzhen_record_010308` | 0.2989 | 0.1206 | 0.2327 | 0.0763 |
| `paper58_latent_dynamics` | `xiangzhen_record_015068` | 0.4304 | 0.2036 | 0.4838 | 0.1102 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_015068` | 0.2759 | 0.1297 | 0.2176 | 0.0651 |
| `geosos_flus_console` | `xiangzhen_record_015068` | 0.4244 | 0.2186 | 0.3785 | 0.0775 |
| `paper58_latent_dynamics` | `xiangzhen_record_019130` | 0.2404 | 0.0525 | 0.2439 | 0.0591 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_019130` | 0.2647 | 0.0650 | 0.1870 | 0.0524 |
| `geosos_flus_console` | `xiangzhen_record_019130` | 0.2770 | 0.0708 | 0.2114 | 0.0388 |
| `paper58_latent_dynamics` | `xiangzhen_record_019254` | 0.2220 | 0.0728 | 0.1308 | 0.0862 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_019254` | 0.1291 | 0.0392 | 0.0575 | 0.0357 |
| `geosos_flus_console` | `xiangzhen_record_019254` | 0.1421 | 0.0484 | 0.0710 | 0.0467 |
| `paper58_latent_dynamics` | `xiangzhen_record_020866` | 0.3589 | 0.1658 | 0.4521 | 0.1014 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_020866` | 0.3535 | 0.1433 | 0.2920 | 0.1044 |
| `geosos_flus_console` | `xiangzhen_record_020866` | 0.3609 | 0.1758 | 0.4250 | 0.0845 |
| `paper58_latent_dynamics` | `xiangzhen_record_021584` | 0.0559 | 0.0245 | 0.3303 | 0.0208 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_021584` | 0.3013 | 0.1700 | 0.4924 | 0.0083 |
| `geosos_flus_console` | `xiangzhen_record_021584` | 0.2678 | 0.1435 | 0.2783 | 0.0141 |
| `paper58_latent_dynamics` | `xiangzhen_record_021766` | 0.2737 | 0.1424 | 0.4410 | 0.2159 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_021766` | 0.3299 | 0.1769 | 0.3241 | 0.1501 |
| `geosos_flus_console` | `xiangzhen_record_021766` | 0.1490 | 0.0734 | 0.1047 | 0.0524 |
| `paper58_latent_dynamics` | `xiangzhen_record_022880` | 0.1753 | 0.0873 | 0.4171 | 0.1125 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_022880` | 0.2591 | 0.1467 | 0.3490 | 0.0081 |
| `geosos_flus_console` | `xiangzhen_record_022880` | 0.1330 | 0.0705 | 0.1101 | 0.0360 |
| `paper58_latent_dynamics` | `xiangzhen_record_025932` | 0.4943 | 0.2162 | 0.3727 | 0.2028 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_025932` | 0.4757 | 0.1996 | 0.2883 | 0.1640 |
| `geosos_flus_console` | `xiangzhen_record_025932` | 0.4855 | 0.2188 | 0.3497 | 0.1949 |
| `paper58_latent_dynamics` | `xiangzhen_record_027267` | 0.2837 | 0.0773 | 0.2116 | 0.0313 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_027267` | 0.3383 | 0.1160 | 0.2519 | 0.0301 |
| `geosos_flus_console` | `xiangzhen_record_027267` | 0.2454 | 0.0723 | 0.1549 | 0.0253 |
| `paper58_latent_dynamics` | `xiangzhen_record_028657` | 0.2950 | 0.1148 | 0.3953 | 0.0383 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_028657` | 0.2720 | 0.1032 | 0.3043 | 0.0225 |
| `geosos_flus_console` | `xiangzhen_record_028657` | 0.2918 | 0.1228 | 0.3290 | 0.0308 |
| `paper58_latent_dynamics` | `xiangzhen_record_030462` | 0.4608 | 0.1805 | 0.4763 | 0.0230 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_030462` | 0.1676 | 0.0169 | 0.1558 | 0.0152 |
| `geosos_flus_console` | `xiangzhen_record_030462` | 0.3082 | 0.1457 | 0.2438 | 0.0245 |
| `paper58_latent_dynamics` | `xiangzhen_record_031108` | 0.3186 | 0.1513 | 0.5070 | 0.0891 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_031108` | 0.3353 | 0.1408 | 0.3186 | 0.1067 |
| `geosos_flus_console` | `xiangzhen_record_031108` | 0.3354 | 0.1739 | 0.4395 | 0.0505 |
| `paper58_latent_dynamics` | `xiangzhen_record_031426` | 0.2576 | 0.1057 | 0.3608 | 0.2593 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_031426` | 0.3179 | 0.1325 | 0.2917 | 0.1172 |
| `geosos_flus_console` | `xiangzhen_record_031426` | 0.2800 | 0.1341 | 0.2894 | 0.1510 |
| `paper58_latent_dynamics` | `xiangzhen_record_031513` | 0.3040 | 0.1321 | 0.3590 | 0.0297 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_031513` | 0.3803 | 0.1739 | 0.3419 | 0.0168 |
| `geosos_flus_console` | `xiangzhen_record_031513` | 0.2970 | 0.1246 | 0.2991 | 0.0255 |
| `paper58_latent_dynamics` | `xiangzhen_record_033687` | 0.1743 | 0.0740 | 0.4301 | 0.0501 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_033687` | 0.1952 | 0.0890 | 0.3419 | 0.0177 |
| `geosos_flus_console` | `xiangzhen_record_033687` | 0.1484 | 0.0724 | 0.3088 | 0.0378 |
| `paper58_latent_dynamics` | `xiangzhen_record_038533` | 0.1448 | 0.0606 | 0.2941 | 0.0635 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_038533` | 0.1406 | 0.0664 | 0.2810 | 0.0171 |
| `geosos_flus_console` | `xiangzhen_record_038533` | 0.1367 | 0.0637 | 0.2157 | 0.0416 |
| `paper58_latent_dynamics` | `xiangzhen_record_038743` | 0.1735 | 0.0876 | 0.5026 | 0.0201 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_038743` | 0.2108 | 0.1148 | 0.4021 | 0.0154 |
| `geosos_flus_console` | `xiangzhen_record_038743` | 0.1729 | 0.0881 | 0.4974 | 0.0175 |
| `paper58_latent_dynamics` | `xiangzhen_record_042889` | 0.2519 | 0.1079 | 0.4404 | 0.0447 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_042889` | 0.4549 | 0.2087 | 0.3557 | 0.0363 |
| `geosos_flus_console` | `xiangzhen_record_042889` | 0.3040 | 0.1517 | 0.3826 | 0.0449 |

## 原始输入与结果图

### `xiangzhen_record_000191`

![xiangzhen_record_000191](figures/same_grid_xiangzhen_record_000191.png)

#### 单区域判读

真实变化像元 385 个。完整 Paper58 预测变化像元 796 个，GeoSOS-FLUS 预测变化像元 221 个。完整 Paper58 的变化 F1 差值 +0.1515，FoM 差值 +0.0721；分配分歧差值 +0.0511（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 469 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1339，FoM 差值 +0.0860，转换准确率差值 +0.1896，分配分歧差值 +0.0055（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+242 像元），最大的低估是类别 7（-140 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+221 像元），最大的低估是类别 2（-157 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+218 像元），最大的低估是类别 7（-140 像元）。

### `xiangzhen_record_002058`

![xiangzhen_record_002058](figures/same_grid_xiangzhen_record_002058.png)

#### 单区域判读

真实变化像元 4,678 个。完整 Paper58 预测变化像元 12,499 个，GeoSOS-FLUS 预测变化像元 5,410 个。完整 Paper58 的变化 F1 差值 -0.0660，FoM 差值 -0.0507；分配分歧差值 +0.0341（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 3,819 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0475，FoM 差值 +0.0349，转换准确率差值 +0.0011，分配分歧差值 +0.0452（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+7,942 像元），最大的低估是类别 11（-9,263 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 2（+1,739 像元），最大的低估是类别 7（-880 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 2（+2,893 像元），最大的低估是类别 11（-4,232 像元）。

### `xiangzhen_record_002815`

![xiangzhen_record_002815](figures/same_grid_xiangzhen_record_002815.png)

#### 单区域判读

真实变化像元 748 个。完整 Paper58 预测变化像元 2,126 个，GeoSOS-FLUS 预测变化像元 1,174 个。完整 Paper58 的变化 F1 差值 +0.0337，FoM 差值 -0.0279；分配分歧差值 +0.0074（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,017 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0482，FoM 差值 -0.0312，转换准确率差值 -0.0775，分配分歧差值 -0.0350（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+1,105 像元），最大的低估是类别 2（-685 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 2（+901 像元），最大的低估是类别 11（-625 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+572 像元），最大的低估是类别 11（-422 像元）。

### `xiangzhen_record_003053`

![xiangzhen_record_003053](figures/same_grid_xiangzhen_record_003053.png)

#### 单区域判读

真实变化像元 82 个。完整 Paper58 预测变化像元 390 个，GeoSOS-FLUS 预测变化像元 320 个。完整 Paper58 的变化 F1 差值 -0.0151，FoM 差值 -0.0089；分配分歧差值 +0.0096（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 346 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0190，FoM 差值 -0.0111，转换准确率差值 -0.0244，分配分歧差值 -0.0226（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+134 像元），最大的低估是类别 7（-134 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+320 像元），最大的低估是类别 7（-320 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+134 像元），最大的低估是类别 7（-134 像元）。

### `xiangzhen_record_003882`

![xiangzhen_record_003882](figures/same_grid_xiangzhen_record_003882.png)

#### 单区域判读

真实变化像元 1,319 个。完整 Paper58 预测变化像元 2,987 个，GeoSOS-FLUS 预测变化像元 2,336 个。完整 Paper58 的变化 F1 差值 -0.0083，FoM 差值 -0.0109；分配分歧差值 +0.0165（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,742 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0279，FoM 差值 -0.0350，转换准确率差值 -0.1236，分配分歧差值 -0.0187（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 2（+877 像元），最大的低估是类别 11（-590 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+768 像元），最大的低估是类别 11（-577 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 2（+816 像元），最大的低估是类别 11（-564 像元）。

### `xiangzhen_record_010308`

![xiangzhen_record_010308](figures/same_grid_xiangzhen_record_010308.png)

#### 单区域判读

真实变化像元 593 个。完整 Paper58 预测变化像元 910 个，GeoSOS-FLUS 预测变化像元 752 个。完整 Paper58 的变化 F1 差值 -0.0035，FoM 差值 -0.0106；分配分歧差值 +0.0153（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 809 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0036，FoM 差值 +0.0007，转换准确率差值 +0.0118，分配分歧差值 +0.0402（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+634 像元），最大的低估是类别 11（-379 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+452 像元），最大的低估是类别 7（-460 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+621 像元），最大的低估是类别 11（-363 像元）。

### `xiangzhen_record_015068`

![xiangzhen_record_015068](figures/same_grid_xiangzhen_record_015068.png)

#### 单区域判读

真实变化像元 864 个。完整 Paper58 预测变化像元 1,752 个，GeoSOS-FLUS 预测变化像元 1,035 个。完整 Paper58 的变化 F1 差值 +0.0060，FoM 差值 -0.0150；分配分歧差值 +0.0327（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 818 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.1486，FoM 差值 -0.0889，转换准确率差值 -0.1609，分配分歧差值 -0.0124（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 2（+412 像元），最大的低估是类别 7（-667 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+719 像元），最大的低估是类别 2（-432 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+288 像元），最大的低估是类别 7（-471 像元）。

### `xiangzhen_record_019130`

![xiangzhen_record_019130](figures/same_grid_xiangzhen_record_019130.png)

#### 单区域判读

真实变化像元 123 个。完整 Paper58 预测变化像元 526 个，GeoSOS-FLUS 预测变化像元 303 个。完整 Paper58 的变化 F1 差值 -0.0366，FoM 差值 -0.0183；分配分歧差值 +0.0203（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 285 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0123，FoM 差值 -0.0059，转换准确率差值 -0.0244，分配分歧差值 +0.0136（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+149 像元），最大的低估是类别 11（-342 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+77 像元），最大的低估是类别 11（-154 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+86 像元），最大的低估是类别 11（-209 像元）。

### `xiangzhen_record_019254`

![xiangzhen_record_019254](figures/same_grid_xiangzhen_record_019254.png)

#### 单区域判读

真实变化像元 8,037 个。完整 Paper58 预测变化像元 8,208 个，GeoSOS-FLUS 预测变化像元 4,657 个。完整 Paper58 的变化 F1 差值 +0.0799，FoM 差值 +0.0244；分配分歧差值 +0.0394（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 4,556 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0130，FoM 差值 -0.0092，转换准确率差值 -0.0136，分配分歧差值 -0.0110（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 2（+5,769 像元），最大的低估是类别 11（-9,528 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 2（+6,733 像元），最大的低估是类别 11（-9,219 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 2（+5,769 像元），最大的低估是类别 11（-8,811 像元）。

### `xiangzhen_record_020866`

![xiangzhen_record_020866](figures/same_grid_xiangzhen_record_020866.png)

#### 单区域判读

真实变化像元 887 个。完整 Paper58 预测变化像元 2,061 个，GeoSOS-FLUS 预测变化像元 1,729 个。完整 Paper58 的变化 F1 差值 -0.0020，FoM 差值 -0.0101；分配分歧差值 +0.0169（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,308 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0073，FoM 差值 -0.0325，转换准确率差值 -0.1330，分配分歧差值 +0.0199（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 2（+839 像元），最大的低估是类别 7（-423 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+205 像元），最大的低估是类别 11（-213 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 2（+783 像元），最大的低估是类别 7（-415 像元）。

### `xiangzhen_record_021584`

![xiangzhen_record_021584](figures/same_grid_xiangzhen_record_021584.png)

#### 单区域判读

真实变化像元 327 个。完整 Paper58 预测变化像元 4,217 个，GeoSOS-FLUS 预测变化像元 405 个。完整 Paper58 的变化 F1 差值 -0.2119，FoM 差值 -0.1191；分配分歧差值 +0.0067（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 788 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0336，FoM 差值 +0.0265，转换准确率差值 +0.2141，分配分歧差值 -0.0057（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+3,660 像元），最大的低估是类别 11（-3,481 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 11（+526 像元），最大的低估是类别 7（-317 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 11（+75 像元），最大的低估是类别 7（-59 像元）。

### `xiangzhen_record_021766`

![xiangzhen_record_021766](figures/same_grid_xiangzhen_record_021766.png)

#### 单区域判读

真实变化像元 898 个。完整 Paper58 预测变化像元 2,324 个，GeoSOS-FLUS 预测变化像元 485 个。完整 Paper58 的变化 F1 差值 +0.1248，FoM 差值 +0.0690；分配分歧差值 +0.1635（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,072 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1810，FoM 差值 +0.1035，转换准确率差值 +0.2194，分配分歧差值 +0.0977（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+1,342 像元），最大的低估是类别 11（-714 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+629 像元），最大的低估是类别 7（-377 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+929 像元），最大的低估是类别 11（-714 像元）。

### `xiangzhen_record_022880`

![xiangzhen_record_022880](figures/same_grid_xiangzhen_record_022880.png)

#### 单区域判读

真实变化像元 808 个。完整 Paper58 预测变化像元 3,425 个，GeoSOS-FLUS 预测变化像元 545 个。完整 Paper58 的变化 F1 差值 +0.0423，FoM 差值 +0.0168；分配分歧差值 +0.0765（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,400 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1260，FoM 差值 +0.0763，转换准确率差值 +0.2389，分配分歧差值 -0.0279（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 11（+210 像元），最大的低估是类别 7（-241 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+1,403 像元），最大的低估是类别 11（-1,083 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+129 像元），最大的低估是类别 11（-134 像元）。

### `xiangzhen_record_025932`

![xiangzhen_record_025932](figures/same_grid_xiangzhen_record_025932.png)

#### 单区域判读

真实变化像元 652 个。完整 Paper58 预测变化像元 841 个，GeoSOS-FLUS 预测变化像元 724 个。完整 Paper58 的变化 F1 差值 +0.0088，FoM 差值 -0.0026；分配分歧差值 +0.0079（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 584 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0097，FoM 差值 -0.0192，转换准确率差值 -0.0613，分配分歧差值 -0.0309（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 7（+219 像元），最大的低估是类别 2（-97 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+216 像元），最大的低估是类别 5（-183 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 7（+177 像元），最大的低估是类别 5（-80 像元）。

### `xiangzhen_record_027267`

![xiangzhen_record_027267](figures/same_grid_xiangzhen_record_027267.png)

#### 单区域判读

真实变化像元 794 个。完整 Paper58 预测变化像元 1,737 个，GeoSOS-FLUS 预测变化像元 1,146 个。完整 Paper58 的变化 F1 差值 +0.0383，FoM 差值 +0.0051；分配分歧差值 +0.0061（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,281 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0930，FoM 差值 +0.0437，转换准确率差值 +0.0970，分配分歧差值 +0.0048（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+1,178 像元），最大的低估是类别 7（-642 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+411 像元），最大的低估是类别 11（-418 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+909 像元），最大的低估是类别 7（-511 像元）。

### `xiangzhen_record_028657`

![xiangzhen_record_028657](figures/same_grid_xiangzhen_record_028657.png)

#### 单区域判读

真实变化像元 769 个。完整 Paper58 预测变化像元 2,336 个，GeoSOS-FLUS 预测变化像元 1,644 个。完整 Paper58 的变化 F1 差值 +0.0033，FoM 差值 -0.0079；分配分歧差值 +0.0075（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,856 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0198，FoM 差值 -0.0196，转换准确率差值 -0.0247，分配分歧差值 -0.0083（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+681 像元），最大的低估是类别 7（-465 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+1,198 像元），最大的低估是类别 2（-588 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+425 像元），最大的低估是类别 7（-342 像元）。

### `xiangzhen_record_030462`

![xiangzhen_record_030462](figures/same_grid_xiangzhen_record_030462.png)

#### 单区域判读

真实变化像元 443 个。完整 Paper58 预测变化像元 1,076 个，GeoSOS-FLUS 预测变化像元 433 个。完整 Paper58 的变化 F1 差值 +0.1526，FoM 差值 +0.0347；分配分歧差值 -0.0015（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 4,020 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.1406，FoM 差值 -0.1289，转换准确率差值 -0.0880，分配分歧差值 -0.0092（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+573 像元），最大的低估是类别 7（-329 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 11（+2,086 像元），最大的低估是类别 2（-3,446 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+170 像元），最大的低估是类别 2（-151 像元）。

### `xiangzhen_record_031108`

![xiangzhen_record_031108](figures/same_grid_xiangzhen_record_031108.png)

#### 单区域判读

真实变化像元 430 个。完整 Paper58 预测变化像元 1,284 个，GeoSOS-FLUS 预测变化像元 876 个。完整 Paper58 的变化 F1 差值 -0.0168，FoM 差值 -0.0226；分配分歧差值 +0.0385（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 739 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0000，FoM 差值 -0.0331，转换准确率差值 -0.1209，分配分歧差值 +0.0562（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 11（+825 像元），最大的低估是类别 5（-692 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 2（+266 像元），最大的低估是类别 11（-361 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 11（+675 像元），最大的低估是类别 5（-529 像元）。

### `xiangzhen_record_031426`

![xiangzhen_record_031426](figures/same_grid_xiangzhen_record_031426.png)

#### 单区域判读

真实变化像元 3,600 个。完整 Paper58 预测变化像元 10,513 个，GeoSOS-FLUS 预测变化像元 5,436 个。完整 Paper58 的变化 F1 差值 -0.0224，FoM 差值 -0.0284；分配分歧差值 +0.1084（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 5,825 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0379，FoM 差值 -0.0016，转换准确率差值 +0.0022，分配分歧差值 -0.0338（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+1,642 像元），最大的低估是类别 11（-1,147 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+1,674 像元），最大的低估是类别 5（-1,445 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+1,087 像元），最大的低估是类别 11（-1,399 像元）。

### `xiangzhen_record_031513`

![xiangzhen_record_031513](figures/same_grid_xiangzhen_record_031513.png)

#### 单区域判读

真实变化像元 117 个。完整 Paper58 预测变化像元 258 个，GeoSOS-FLUS 预测变化像元 213 个。完整 Paper58 的变化 F1 差值 +0.0070，FoM 差值 +0.0075；分配分歧差值 +0.0042（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 167 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0833，FoM 差值 +0.0494，转换准确率差值 +0.0427，分配分歧差值 -0.0087（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 7（+177 像元），最大的低估是类别 11（-135 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+133 像元），最大的低估是类别 11（-88 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 7（+161 像元），最大的低估是类别 11（-123 像元）。

### `xiangzhen_record_033687`

![xiangzhen_record_033687](figures/same_grid_xiangzhen_record_033687.png)

#### 单区域判读

真实变化像元 272 个。完整 Paper58 预测变化像元 1,461 个，GeoSOS-FLUS 预测变化像元 981 个。完整 Paper58 的变化 F1 差值 +0.0258，FoM 差值 +0.0015；分配分歧差值 +0.0123（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 886 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0467，FoM 差值 +0.0166，转换准确率差值 +0.0331，分配分歧差值 -0.0201（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+401 像元），最大的低估是类别 7（-556 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+637 像元），最大的低估是类别 7（-572 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+401 像元），最大的低估是类别 7（-348 像元）。

### `xiangzhen_record_038533`

![xiangzhen_record_038533](figures/same_grid_xiangzhen_record_038533.png)

#### 单区域判读

真实变化像元 153 个。完整 Paper58 预测变化像元 648 个，GeoSOS-FLUS 预测变化像元 403 个。完整 Paper58 的变化 F1 差值 +0.0081，FoM 差值 -0.0031；分配分歧差值 +0.0219（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 544 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0039，FoM 差值 +0.0027，转换准确率差值 +0.0654，分配分歧差值 -0.0245（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+81 像元），最大的低估是类别 7（-39 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+439 像元），最大的低估是类别 7（-322 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+81 像元），最大的低估是类别 7（-39 像元）。

### `xiangzhen_record_038743`

![xiangzhen_record_038743](figures/same_grid_xiangzhen_record_038743.png)

#### 单区域判读

真实变化像元 189 个。完整 Paper58 预测变化像元 998 个，GeoSOS-FLUS 预测变化像元 979 个。完整 Paper58 的变化 F1 差值 +0.0006，FoM 差值 -0.0005；分配分歧差值 +0.0026（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 551 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0379，FoM 差值 +0.0267，转换准确率差值 -0.0952，分配分歧差值 -0.0021（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+864 像元），最大的低估是类别 7（-759 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+490 像元），最大的低估是类别 7（-386 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+864 像元），最大的低估是类别 7（-759 像元）。

### `xiangzhen_record_042889`

![xiangzhen_record_042889](figures/same_grid_xiangzhen_record_042889.png)

#### 单区域判读

真实变化像元 1,712 个。完整 Paper58 预测变化像元 6,282 个，GeoSOS-FLUS 预测变化像元 3,380 个。完整 Paper58 的变化 F1 差值 -0.0521，FoM 差值 -0.0438；分配分歧差值 -0.0002（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 2,065 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1509，FoM 差值 +0.0570，转换准确率差值 -0.0269，分配分歧差值 -0.0086（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 11（+4,385 像元），最大的低估是类别 2（-4,124 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+1,312 像元），最大的低估是类别 11（-862 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 11（+1,924 像元），最大的低估是类别 2（-1,828 像元）。
