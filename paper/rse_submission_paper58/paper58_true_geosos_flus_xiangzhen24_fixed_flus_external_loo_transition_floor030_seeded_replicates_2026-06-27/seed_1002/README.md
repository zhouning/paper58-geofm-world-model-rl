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

- `变化 F1` 衡量模型是否找到了真实变化像元。`paper58_latent_dynamics` 均值=0.2767，`geosos_flus_console` 均值=0.2662，差值=+0.0105 (更优)。
- `FoM` 要求变化位置和目标类别同时正确。`paper58_latent_dynamics` 均值=0.1141，`geosos_flus_console` 均值=0.1198，差值=-0.0057 (更差)。
- `转换准确率` 只在真实变化像元上检查目标类别是否命中。`paper58_latent_dynamics` 均值=0.3711，`geosos_flus_console` 均值=0.2796，差值=+0.0915 (更优)。
- `分配分歧` 越低越好，表示空间位置和类别分配错配。`paper58_latent_dynamics` 均值=0.0873，`geosos_flus_console` 均值=0.0598，差值=+0.0275 (更差)。
- `paper58_spatial_demand_ratio_external_loo_transition_floor030` 是 Paper58 优化后输出；相对 GeoSOS-FLUS：变化 F1 差值=+0.0221 (更优)，FoM 差值=+0.0050 (更优)，转换准确率差值=+0.0067 (更优)，分配分歧差值=-0.0014 (更优)。四项指标中有 4/4 项优于 GeoSOS-FLUS。

## 指标表

| 方法 | 区域 | 变化 F1 | FoM | 转换准确率 | 分配分歧 |
| --- | --- | ---: | ---: | ---: | ---: |
| `paper58_latent_dynamics` | `xiangzhen_record_000191` | 0.3760 | 0.1502 | 0.3740 | 0.0974 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_000191` | 0.3583 | 0.1641 | 0.2987 | 0.0518 |
| `geosos_flus_console` | `xiangzhen_record_000191` | 0.2267 | 0.0820 | 0.1299 | 0.0540 |
| `paper58_latent_dynamics` | `xiangzhen_record_002058` | 0.2265 | 0.1037 | 0.3378 | 0.1143 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_002058` | 0.3399 | 0.1893 | 0.2854 | 0.1254 |
| `geosos_flus_console` | `xiangzhen_record_002058` | 0.2932 | 0.1560 | 0.2850 | 0.0796 |
| `paper58_latent_dynamics` | `xiangzhen_record_002815` | 0.3584 | 0.0907 | 0.2861 | 0.1140 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_002815` | 0.2765 | 0.0874 | 0.1778 | 0.0716 |
| `geosos_flus_console` | `xiangzhen_record_002815` | 0.3347 | 0.1247 | 0.2687 | 0.1063 |
| `paper58_latent_dynamics` | `xiangzhen_record_003053` | 0.1441 | 0.0776 | 0.4146 | 0.0391 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_003053` | 0.1402 | 0.0754 | 0.3659 | 0.0069 |
| `geosos_flus_console` | `xiangzhen_record_003053` | 0.1475 | 0.0796 | 0.3902 | 0.0342 |
| `paper58_latent_dynamics` | `xiangzhen_record_003882` | 0.3261 | 0.1487 | 0.4064 | 0.0807 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_003882` | 0.3064 | 0.1246 | 0.2449 | 0.0455 |
| `geosos_flus_console` | `xiangzhen_record_003882` | 0.3391 | 0.1623 | 0.3700 | 0.0628 |
| `paper58_latent_dynamics` | `xiangzhen_record_010308` | 0.2954 | 0.1101 | 0.2378 | 0.0917 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_010308` | 0.2953 | 0.1213 | 0.2445 | 0.1165 |
| `geosos_flus_console` | `xiangzhen_record_010308` | 0.3037 | 0.1205 | 0.2327 | 0.0786 |
| `paper58_latent_dynamics` | `xiangzhen_record_015068` | 0.4304 | 0.2036 | 0.4838 | 0.1102 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_015068` | 0.2759 | 0.1297 | 0.2176 | 0.0651 |
| `geosos_flus_console` | `xiangzhen_record_015068` | 0.4305 | 0.2216 | 0.3796 | 0.0735 |
| `paper58_latent_dynamics` | `xiangzhen_record_019130` | 0.2404 | 0.0525 | 0.2439 | 0.0591 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_019130` | 0.2647 | 0.0650 | 0.1870 | 0.0524 |
| `geosos_flus_console` | `xiangzhen_record_019130` | 0.2727 | 0.0650 | 0.1707 | 0.0418 |
| `paper58_latent_dynamics` | `xiangzhen_record_019254` | 0.2220 | 0.0728 | 0.1308 | 0.0862 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_019254` | 0.1291 | 0.0392 | 0.0575 | 0.0357 |
| `geosos_flus_console` | `xiangzhen_record_019254` | 0.1441 | 0.0503 | 0.0739 | 0.0466 |
| `paper58_latent_dynamics` | `xiangzhen_record_020866` | 0.3589 | 0.1658 | 0.4521 | 0.1014 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_020866` | 0.3535 | 0.1433 | 0.2920 | 0.1044 |
| `geosos_flus_console` | `xiangzhen_record_020866` | 0.3589 | 0.1727 | 0.4273 | 0.0889 |
| `paper58_latent_dynamics` | `xiangzhen_record_021584` | 0.0559 | 0.0245 | 0.3303 | 0.0208 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_021584` | 0.3013 | 0.1700 | 0.4924 | 0.0083 |
| `geosos_flus_console` | `xiangzhen_record_021584` | 0.2803 | 0.1489 | 0.2905 | 0.0140 |
| `paper58_latent_dynamics` | `xiangzhen_record_021766` | 0.2737 | 0.1424 | 0.4410 | 0.2159 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_021766` | 0.3299 | 0.1769 | 0.3241 | 0.1501 |
| `geosos_flus_console` | `xiangzhen_record_021766` | 0.1515 | 0.0725 | 0.1024 | 0.0524 |
| `paper58_latent_dynamics` | `xiangzhen_record_022880` | 0.1753 | 0.0873 | 0.4171 | 0.1125 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_022880` | 0.2591 | 0.1467 | 0.3490 | 0.0081 |
| `geosos_flus_console` | `xiangzhen_record_022880` | 0.1324 | 0.0701 | 0.1064 | 0.0346 |
| `paper58_latent_dynamics` | `xiangzhen_record_025932` | 0.4943 | 0.2162 | 0.3727 | 0.2028 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_025932` | 0.4757 | 0.1996 | 0.2883 | 0.1640 |
| `geosos_flus_console` | `xiangzhen_record_025932` | 0.4878 | 0.2199 | 0.3451 | 0.1930 |
| `paper58_latent_dynamics` | `xiangzhen_record_027267` | 0.2837 | 0.0773 | 0.2116 | 0.0313 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_027267` | 0.3383 | 0.1160 | 0.2519 | 0.0301 |
| `geosos_flus_console` | `xiangzhen_record_027267` | 0.2495 | 0.0713 | 0.1524 | 0.0253 |
| `paper58_latent_dynamics` | `xiangzhen_record_028657` | 0.2950 | 0.1148 | 0.3953 | 0.0383 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_028657` | 0.2720 | 0.1032 | 0.3043 | 0.0225 |
| `geosos_flus_console` | `xiangzhen_record_028657` | 0.2816 | 0.1182 | 0.3264 | 0.0328 |
| `paper58_latent_dynamics` | `xiangzhen_record_030462` | 0.4608 | 0.1805 | 0.4763 | 0.0230 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_030462` | 0.1676 | 0.0169 | 0.1558 | 0.0152 |
| `geosos_flus_console` | `xiangzhen_record_030462` | 0.3234 | 0.1546 | 0.2551 | 0.0233 |
| `paper58_latent_dynamics` | `xiangzhen_record_031108` | 0.3186 | 0.1513 | 0.5070 | 0.0891 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_031108` | 0.3353 | 0.1408 | 0.3186 | 0.1067 |
| `geosos_flus_console` | `xiangzhen_record_031108` | 0.3378 | 0.1732 | 0.4953 | 0.0818 |
| `paper58_latent_dynamics` | `xiangzhen_record_031426` | 0.2576 | 0.1057 | 0.3608 | 0.2593 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_031426` | 0.3179 | 0.1325 | 0.2917 | 0.1172 |
| `geosos_flus_console` | `xiangzhen_record_031426` | 0.2794 | 0.1354 | 0.2892 | 0.1485 |
| `paper58_latent_dynamics` | `xiangzhen_record_031513` | 0.3040 | 0.1321 | 0.3590 | 0.0297 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_031513` | 0.3803 | 0.1739 | 0.3419 | 0.0168 |
| `geosos_flus_console` | `xiangzhen_record_031513` | 0.2526 | 0.0964 | 0.2051 | 0.0231 |
| `paper58_latent_dynamics` | `xiangzhen_record_033687` | 0.1743 | 0.0740 | 0.4301 | 0.0501 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_033687` | 0.1952 | 0.0890 | 0.3419 | 0.0177 |
| `geosos_flus_console` | `xiangzhen_record_033687` | 0.1579 | 0.0753 | 0.3199 | 0.0373 |
| `paper58_latent_dynamics` | `xiangzhen_record_038533` | 0.1448 | 0.0606 | 0.2941 | 0.0635 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_038533` | 0.1406 | 0.0664 | 0.2810 | 0.0171 |
| `geosos_flus_console` | `xiangzhen_record_038533` | 0.1290 | 0.0632 | 0.2157 | 0.0420 |
| `paper58_latent_dynamics` | `xiangzhen_record_038743` | 0.1735 | 0.0876 | 0.5026 | 0.0201 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_038743` | 0.2108 | 0.1148 | 0.4021 | 0.0154 |
| `geosos_flus_console` | `xiangzhen_record_038743` | 0.1713 | 0.0889 | 0.4974 | 0.0159 |
| `paper58_latent_dynamics` | `xiangzhen_record_042889` | 0.2519 | 0.1079 | 0.4404 | 0.0447 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_042889` | 0.4549 | 0.2087 | 0.3557 | 0.0363 |
| `geosos_flus_console` | `xiangzhen_record_042889` | 0.3033 | 0.1516 | 0.3808 | 0.0449 |

## 原始输入与结果图

### `xiangzhen_record_000191`

![xiangzhen_record_000191](figures/same_grid_xiangzhen_record_000191.png)

#### 单区域判读

真实变化像元 385 个。完整 Paper58 预测变化像元 796 个，GeoSOS-FLUS 预测变化像元 303 个。完整 Paper58 的变化 F1 差值 +0.1492，FoM 差值 +0.0682；分配分歧差值 +0.0434（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 469 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1316，FoM 差值 +0.0821，转换准确率差值 +0.1688，分配分歧差值 -0.0022（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+242 像元），最大的低估是类别 7（-140 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+221 像元），最大的低估是类别 2（-157 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+242 像元），最大的低估是类别 7（-141 像元）。

### `xiangzhen_record_002058`

![xiangzhen_record_002058](figures/same_grid_xiangzhen_record_002058.png)

#### 单区域判读

真实变化像元 4,678 个。完整 Paper58 预测变化像元 12,499 个，GeoSOS-FLUS 预测变化像元 5,336 个。完整 Paper58 的变化 F1 差值 -0.0667，FoM 差值 -0.0523；分配分歧差值 +0.0348（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 3,819 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0467，FoM 差值 +0.0333，转换准确率差值 +0.0004，分配分歧差值 +0.0458（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+7,942 像元），最大的低估是类别 11（-9,263 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 2（+1,739 像元），最大的低估是类别 7（-880 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 2（+2,861 像元），最大的低估是类别 11（-4,177 像元）。

### `xiangzhen_record_002815`

![xiangzhen_record_002815](figures/same_grid_xiangzhen_record_002815.png)

#### 单区域判读

真实变化像元 748 个。完整 Paper58 预测变化像元 2,126 个，GeoSOS-FLUS 预测变化像元 1,188 个。完整 Paper58 的变化 F1 差值 +0.0237，FoM 差值 -0.0340；分配分歧差值 +0.0077（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,017 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0582，FoM 差值 -0.0372，转换准确率差值 -0.0909，分配分歧差值 -0.0347（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+1,105 像元），最大的低估是类别 2（-685 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 2（+901 像元），最大的低估是类别 11（-625 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+604 像元），最大的低估是类别 11（-404 像元）。

### `xiangzhen_record_003053`

![xiangzhen_record_003053](figures/same_grid_xiangzhen_record_003053.png)

#### 单区域判读

真实变化像元 82 个。完整 Paper58 预测变化像元 390 个，GeoSOS-FLUS 预测变化像元 352 个。完整 Paper58 的变化 F1 差值 -0.0034，FoM 差值 -0.0020；分配分歧差值 +0.0049（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 346 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0073，FoM 差值 -0.0042，转换准确率差值 -0.0244，分配分歧差值 -0.0272（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+134 像元），最大的低估是类别 7（-134 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+320 像元），最大的低估是类别 7（-320 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+134 像元），最大的低估是类别 7（-134 像元）。

### `xiangzhen_record_003882`

![xiangzhen_record_003882](figures/same_grid_xiangzhen_record_003882.png)

#### 单区域判读

真实变化像元 1,319 个。完整 Paper58 预测变化像元 2,987 个，GeoSOS-FLUS 预测变化像元 2,302 个。完整 Paper58 的变化 F1 差值 -0.0131，FoM 差值 -0.0136；分配分歧差值 +0.0179（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,742 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0327，FoM 差值 -0.0377，转换准确率差值 -0.1251，分配分歧差值 -0.0173（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 2（+877 像元），最大的低估是类别 11（-590 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+768 像元），最大的低估是类别 11（-577 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 2（+815 像元），最大的低估是类别 11（-563 像元）。

### `xiangzhen_record_010308`

![xiangzhen_record_010308](figures/same_grid_xiangzhen_record_010308.png)

#### 单区域判读

真实变化像元 593 个。完整 Paper58 预测变化像元 910 个，GeoSOS-FLUS 预测变化像元 757 个。完整 Paper58 的变化 F1 差值 -0.0083，FoM 差值 -0.0105；分配分歧差值 +0.0130（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 809 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0084，FoM 差值 +0.0008，转换准确率差值 +0.0118，分配分歧差值 +0.0378（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+634 像元），最大的低估是类别 11（-379 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+452 像元），最大的低估是类别 7（-460 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+610 像元），最大的低估是类别 11（-368 像元）。

### `xiangzhen_record_015068`

![xiangzhen_record_015068](figures/same_grid_xiangzhen_record_015068.png)

#### 单区域判读

真实变化像元 864 个。完整 Paper58 预测变化像元 1,752 个，GeoSOS-FLUS 预测变化像元 1,022 个。完整 Paper58 的变化 F1 差值 -0.0001，FoM 差值 -0.0180；分配分歧差值 +0.0367（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 818 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.1547，FoM 差值 -0.0920，转换准确率差值 -0.1620，分配分歧差值 -0.0084（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 2（+412 像元），最大的低估是类别 7（-667 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+719 像元），最大的低估是类别 2（-432 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+288 像元），最大的低估是类别 7（-483 像元）。

### `xiangzhen_record_019130`

![xiangzhen_record_019130](figures/same_grid_xiangzhen_record_019130.png)

#### 单区域判读

真实变化像元 123 个。完整 Paper58 预测变化像元 526 个，GeoSOS-FLUS 预测变化像元 251 个。完整 Paper58 的变化 F1 差值 -0.0324，FoM 差值 -0.0125；分配分歧差值 +0.0173（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 285 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0080，FoM 差值 -0.0000，转换准确率差值 +0.0163，分配分歧差值 +0.0106（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+149 像元），最大的低估是类别 11（-342 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+77 像元），最大的低估是类别 11（-154 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 7（+78 像元），最大的低估是类别 11（-160 像元）。

### `xiangzhen_record_019254`

![xiangzhen_record_019254](figures/same_grid_xiangzhen_record_019254.png)

#### 单区域判读

真实变化像元 8,037 个。完整 Paper58 预测变化像元 8,208 个，GeoSOS-FLUS 预测变化像元 4,680 个。完整 Paper58 的变化 F1 差值 +0.0779，FoM 差值 +0.0224；分配分歧差值 +0.0396（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 4,556 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0149，FoM 差值 -0.0111，转换准确率差值 -0.0164，分配分歧差值 -0.0108（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 2（+5,769 像元），最大的低估是类别 11（-9,528 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 2（+6,733 像元），最大的低估是类别 11（-9,219 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 2（+5,769 像元），最大的低估是类别 11（-8,803 像元）。

### `xiangzhen_record_020866`

![xiangzhen_record_020866](figures/same_grid_xiangzhen_record_020866.png)

#### 单区域判读

真实变化像元 887 个。完整 Paper58 预测变化像元 2,061 个，GeoSOS-FLUS 预测变化像元 1,788 个。完整 Paper58 的变化 F1 差值 +0.0000，FoM 差值 -0.0069；分配分歧差值 +0.0125（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,308 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0053，FoM 差值 -0.0293，转换准确率差值 -0.1353，分配分歧差值 +0.0155（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 2（+839 像元），最大的低估是类别 7（-423 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+205 像元），最大的低估是类别 11（-213 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 2（+781 像元），最大的低估是类别 7（-413 像元）。

### `xiangzhen_record_021584`

![xiangzhen_record_021584](figures/same_grid_xiangzhen_record_021584.png)

#### 单区域判读

真实变化像元 327 个。完整 Paper58 预测变化像元 4,217 个，GeoSOS-FLUS 预测变化像元 415 个。完整 Paper58 的变化 F1 差值 -0.2244，FoM 差值 -0.1245；分配分歧差值 +0.0068（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 788 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0210，FoM 差值 +0.0211，转换准确率差值 +0.2018，分配分歧差值 -0.0056（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+3,660 像元），最大的低估是类别 11（-3,481 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 11（+526 像元），最大的低估是类别 7（-317 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 11（+89 像元），最大的低估是类别 7（-59 像元）。

### `xiangzhen_record_021766`

![xiangzhen_record_021766](figures/same_grid_xiangzhen_record_021766.png)

#### 单区域判读

真实变化像元 898 个。完整 Paper58 预测变化像元 2,324 个，GeoSOS-FLUS 预测变化像元 475 个。完整 Paper58 的变化 F1 差值 +0.1222，FoM 差值 +0.0699；分配分歧差值 +0.1635（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,072 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1785，FoM 差值 +0.1044，转换准确率差值 +0.2216，分配分歧差值 +0.0977（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+1,342 像元），最大的低估是类别 11（-714 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+629 像元），最大的低估是类别 7（-377 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+915 像元），最大的低估是类别 11（-714 像元）。

### `xiangzhen_record_022880`

![xiangzhen_record_022880](figures/same_grid_xiangzhen_record_022880.png)

#### 单区域判读

真实变化像元 808 个。完整 Paper58 预测变化像元 3,425 个，GeoSOS-FLUS 预测变化像元 506 个。完整 Paper58 的变化 F1 差值 +0.0429，FoM 差值 +0.0172；分配分歧差值 +0.0779（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,400 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1266，FoM 差值 +0.0766，转换准确率差值 +0.2426，分配分歧差值 -0.0265（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 11（+210 像元），最大的低估是类别 7（-241 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+1,403 像元），最大的低估是类别 11（-1,083 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+140 像元），最大的低估是类别 11（-149 像元）。

### `xiangzhen_record_025932`

![xiangzhen_record_025932](figures/same_grid_xiangzhen_record_025932.png)

#### 单区域判读

真实变化像元 652 个。完整 Paper58 预测变化像元 841 个，GeoSOS-FLUS 预测变化像元 701 个。完整 Paper58 的变化 F1 差值 +0.0065，FoM 差值 -0.0037；分配分歧差值 +0.0098（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 584 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0121，FoM 差值 -0.0204，转换准确率差值 -0.0567，分配分歧差值 -0.0290（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 7（+219 像元），最大的低估是类别 2（-97 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+216 像元），最大的低估是类别 5（-183 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 7（+167 像元），最大的低估是类别 5（-69 像元）。

### `xiangzhen_record_027267`

![xiangzhen_record_027267](figures/same_grid_xiangzhen_record_027267.png)

#### 单区域判读

真实变化像元 794 个。完整 Paper58 预测变化像元 1,737 个，GeoSOS-FLUS 预测变化像元 1,146 个。完整 Paper58 的变化 F1 差值 +0.0342，FoM 差值 +0.0061；分配分歧差值 +0.0060（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,281 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0888，FoM 差值 +0.0447，转换准确率差值 +0.0995，分配分歧差值 +0.0047（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+1,178 像元），最大的低估是类别 7（-642 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+411 像元），最大的低估是类别 11（-418 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+905 像元），最大的低估是类别 7（-509 像元）。

### `xiangzhen_record_028657`

![xiangzhen_record_028657](figures/same_grid_xiangzhen_record_028657.png)

#### 单区域判读

真实变化像元 769 个。完整 Paper58 预测变化像元 2,336 个，GeoSOS-FLUS 预测变化像元 1,703 个。完整 Paper58 的变化 F1 差值 +0.0135，FoM 差值 -0.0033；分配分歧差值 +0.0055（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 1,856 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0096，FoM 差值 -0.0150，转换准确率差值 -0.0221，分配分歧差值 -0.0103（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+681 像元），最大的低估是类别 7（-465 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+1,198 像元），最大的低估是类别 2（-588 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+415 像元），最大的低估是类别 7（-328 像元）。

### `xiangzhen_record_030462`

![xiangzhen_record_030462](figures/same_grid_xiangzhen_record_030462.png)

#### 单区域判读

真实变化像元 443 个。完整 Paper58 预测变化像元 1,076 个，GeoSOS-FLUS 预测变化像元 429 个。完整 Paper58 的变化 F1 差值 +0.1374，FoM 差值 +0.0259；分配分歧差值 -0.0003（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 4,020 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.1558，FoM 差值 -0.1377，转换准确率差值 -0.0993，分配分歧差值 -0.0080（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+573 像元），最大的低估是类别 7（-329 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 11（+2,086 像元），最大的低估是类别 2（-3,446 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+162 像元），最大的低估是类别 2（-151 像元）。

### `xiangzhen_record_031108`

![xiangzhen_record_031108](figures/same_grid_xiangzhen_record_031108.png)

#### 单区域判读

真实变化像元 430 个。完整 Paper58 预测变化像元 1,284 个，GeoSOS-FLUS 预测变化像元 1,050 个。完整 Paper58 的变化 F1 差值 -0.0193，FoM 差值 -0.0219；分配分歧差值 +0.0073（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 739 个；相对 GeoSOS-FLUS，变化 F1 差值 -0.0025，FoM 差值 -0.0324，转换准确率差值 -0.1767，分配分歧差值 +0.0249（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 11（+825 像元），最大的低估是类别 5（-692 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 2（+266 像元），最大的低估是类别 11（-361 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 11（+656 像元），最大的低估是类别 5（-551 像元）。

### `xiangzhen_record_031426`

![xiangzhen_record_031426](figures/same_grid_xiangzhen_record_031426.png)

#### 单区域判读

真实变化像元 3,600 个。完整 Paper58 预测变化像元 10,513 个，GeoSOS-FLUS 预测变化像元 5,335 个。完整 Paper58 的变化 F1 差值 -0.0217，FoM 差值 -0.0298；分配分歧差值 +0.1109（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 5,825 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0385，FoM 差值 -0.0030，转换准确率差值 +0.0025，分配分歧差值 -0.0313（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+1,642 像元），最大的低估是类别 11（-1,147 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+1,674 像元），最大的低估是类别 5（-1,445 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+1,075 像元），最大的低估是类别 11（-1,408 像元）。

### `xiangzhen_record_031513`

![xiangzhen_record_031513](figures/same_grid_xiangzhen_record_031513.png)

#### 单区域判读

真实变化像元 117 个。完整 Paper58 预测变化像元 258 个，GeoSOS-FLUS 预测变化像元 168 个。完整 Paper58 的变化 F1 差值 +0.0514，FoM 差值 +0.0357；分配分歧差值 +0.0066（该指标越低越好）。该区域完整 Paper58 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 167 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1277，FoM 差值 +0.0775，转换准确率差值 +0.1368，分配分歧差值 -0.0063（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 7（+177 像元），最大的低估是类别 11（-135 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 7（+133 像元），最大的低估是类别 11（-88 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 7（+148 像元），最大的低估是类别 11（-116 像元）。

### `xiangzhen_record_033687`

![xiangzhen_record_033687](figures/same_grid_xiangzhen_record_033687.png)

#### 单区域判读

真实变化像元 272 个。完整 Paper58 预测变化像元 1,461 个，GeoSOS-FLUS 预测变化像元 982 个。完整 Paper58 的变化 F1 差值 +0.0164，FoM 差值 -0.0014；分配分歧差值 +0.0128（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 886 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0373，FoM 差值 +0.0137，转换准确率差值 +0.0221，分配分歧差值 -0.0196（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+401 像元），最大的低估是类别 7（-556 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+637 像元），最大的低估是类别 7（-572 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+401 像元），最大的低估是类别 7（-349 像元）。

### `xiangzhen_record_038533`

![xiangzhen_record_038533](figures/same_grid_xiangzhen_record_038533.png)

#### 单区域判读

真实变化像元 153 个。完整 Paper58 预测变化像元 648 个，GeoSOS-FLUS 预测变化像元 405 个。完整 Paper58 的变化 F1 差值 +0.0158，FoM 差值 -0.0027；分配分歧差值 +0.0215（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 544 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0116，FoM 差值 +0.0031，转换准确率差值 +0.0654，分配分歧差值 -0.0249（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+81 像元），最大的低估是类别 7（-39 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+439 像元），最大的低估是类别 7（-322 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+81 像元），最大的低估是类别 7（-39 像元）。

### `xiangzhen_record_038743`

![xiangzhen_record_038743](figures/same_grid_xiangzhen_record_038743.png)

#### 单区域判读

真实变化像元 189 个。完整 Paper58 预测变化像元 998 个，GeoSOS-FLUS 预测变化像元 967 个。完整 Paper58 的变化 F1 差值 +0.0023，FoM 差值 -0.0013；分配分歧差值 +0.0042（该指标越低越好）。该区域两个方法各有优势，需要结合误差图判断漏判和误报。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 551 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.0395，FoM 差值 +0.0259，转换准确率差值 -0.0952，分配分歧差值 -0.0005（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 5（+864 像元），最大的低估是类别 7（-759 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+490 像元），最大的低估是类别 7（-386 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 5（+864 像元），最大的低估是类别 7（-759 像元）。

### `xiangzhen_record_042889`

![xiangzhen_record_042889](figures/same_grid_xiangzhen_record_042889.png)

#### 单区域判读

真实变化像元 1,712 个。完整 Paper58 预测变化像元 6,282 个，GeoSOS-FLUS 预测变化像元 3,359 个。完整 Paper58 的变化 F1 差值 -0.0514，FoM 差值 -0.0436；分配分歧差值 -0.0002（该指标越低越好）。该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。`paper58_spatial_demand_ratio_external_loo_transition_floor030` 预测变化像元 2,065 个；相对 GeoSOS-FLUS，变化 F1 差值 +0.1516，FoM 差值 +0.0571，转换准确率差值 -0.0251，分配分歧差值 -0.0087（该指标越低越好）。

#### 类别面积偏差

完整 Paper58 相对结束年份真值最大的高估是类别 11（+4,385 像元），最大的低估是类别 2（-4,124 像元）。paper58_spatial_demand_ratio_external_loo_transition_floor030 相对结束年份真值最大的高估是类别 5（+1,312 像元），最大的低估是类别 11（-862 像元）。GeoSOS-FLUS 相对结束年份真值最大的高估是类别 11（+1,903 像元），最大的低估是类别 2（-1,837 像元）。
