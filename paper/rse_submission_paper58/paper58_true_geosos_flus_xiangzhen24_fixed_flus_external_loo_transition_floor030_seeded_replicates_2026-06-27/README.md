# Paper58 与 GeoSOS-FLUS 多随机种子稳健性报告

本报告固定 GeoSOS-FLUS 环境变量 `FLUS_RANDOM_SEED`，对同一输入重复运行 5 次。Paper58 输出是确定性的；GeoSOS-FLUS 的 CA 分配含随机项，因此需要用重复实验报告均值和波动范围。

- 随机种子：1001, 1002, 1003, 1004, 1005
- 主要比较：`paper58_spatial_demand_ratio_external_loo_transition_floor030` 相对 `geosos_flus_console`

## 方法均值与波动

| 方法 | 区域 | n | 变化 F1 均值±标准差 | FoM 均值±标准差 | 转换准确率均值±标准差 | 分配分歧均值±标准差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `geosos_flus_console` | `xiangzhen_record_000191` | 5 | 0.2218±0.0087 | 0.0800±0.0050 | 0.1174±0.0150 | 0.0482±0.0039 |
| `geosos_flus_console` | `xiangzhen_record_002058` | 5 | 0.2924±0.0023 | 0.1551±0.0015 | 0.2843±0.0024 | 0.0807±0.0008 |
| `geosos_flus_console` | `xiangzhen_record_002815` | 5 | 0.3283±0.0043 | 0.1222±0.0026 | 0.2618±0.0053 | 0.1069±0.0007 |
| `geosos_flus_console` | `xiangzhen_record_003053` | 5 | 0.1480±0.0080 | 0.0799±0.0047 | 0.3756±0.0159 | 0.0320±0.0028 |
| `geosos_flus_console` | `xiangzhen_record_003882` | 5 | 0.3378±0.0027 | 0.1613±0.0015 | 0.3709±0.0030 | 0.0636±0.0005 |
| `geosos_flus_console` | `xiangzhen_record_010308` | 5 | 0.3007±0.0020 | 0.1200±0.0009 | 0.2314±0.0014 | 0.0767±0.0011 |
| `geosos_flus_console` | `xiangzhen_record_015068` | 5 | 0.4291±0.0061 | 0.2209±0.0038 | 0.3796±0.0080 | 0.0750±0.0017 |
| `geosos_flus_console` | `xiangzhen_record_019130` | 5 | 0.2592±0.0199 | 0.0684±0.0041 | 0.1919±0.0204 | 0.0419±0.0021 |
| `geosos_flus_console` | `xiangzhen_record_019254` | 5 | 0.1444±0.0020 | 0.0494±0.0010 | 0.0724±0.0015 | 0.0464±0.0004 |
| `geosos_flus_console` | `xiangzhen_record_020866` | 5 | 0.3607±0.0018 | 0.1753±0.0016 | 0.4273±0.0033 | 0.0857±0.0019 |
| `geosos_flus_console` | `xiangzhen_record_021584` | 5 | 0.2736±0.0078 | 0.1457±0.0028 | 0.2862±0.0060 | 0.0142±0.0002 |
| `geosos_flus_console` | `xiangzhen_record_021766` | 5 | 0.1490±0.0094 | 0.0717±0.0048 | 0.1031±0.0068 | 0.0544±0.0020 |
| `geosos_flus_console` | `xiangzhen_record_022880` | 5 | 0.1340±0.0028 | 0.0710±0.0016 | 0.1101±0.0032 | 0.0356±0.0008 |
| `geosos_flus_console` | `xiangzhen_record_025932` | 5 | 0.4903±0.0035 | 0.2226±0.0035 | 0.3457±0.0076 | 0.1907±0.0032 |
| `geosos_flus_console` | `xiangzhen_record_027267` | 5 | 0.2486±0.0074 | 0.0734±0.0036 | 0.1552±0.0078 | 0.0248±0.0005 |
| `geosos_flus_console` | `xiangzhen_record_028657` | 5 | 0.2862±0.0042 | 0.1202±0.0021 | 0.3267±0.0027 | 0.0317±0.0007 |
| `geosos_flus_console` | `xiangzhen_record_030462` | 5 | 0.3192±0.0064 | 0.1515±0.0035 | 0.2515±0.0054 | 0.0237±0.0006 |
| `geosos_flus_console` | `xiangzhen_record_031108` | 5 | 0.3315±0.0115 | 0.1714±0.0045 | 0.4544±0.0372 | 0.0630±0.0160 |
| `geosos_flus_console` | `xiangzhen_record_031426` | 5 | 0.2799±0.0004 | 0.1355±0.0009 | 0.2909±0.0023 | 0.1493±0.0017 |
| `geosos_flus_console` | `xiangzhen_record_031513` | 5 | 0.2905±0.0264 | 0.1212±0.0189 | 0.2803±0.0539 | 0.0241±0.0015 |
| `geosos_flus_console` | `xiangzhen_record_033687` | 5 | 0.1530±0.0039 | 0.0726±0.0022 | 0.3081±0.0080 | 0.0375±0.0007 |
| `geosos_flus_console` | `xiangzhen_record_038533` | 5 | 0.1360±0.0041 | 0.0645±0.0014 | 0.2196±0.0058 | 0.0418±0.0016 |
| `geosos_flus_console` | `xiangzhen_record_038743` | 5 | 0.1729±0.0012 | 0.0888±0.0007 | 0.4995±0.0029 | 0.0168±0.0008 |
| `geosos_flus_console` | `xiangzhen_record_042889` | 5 | 0.3033±0.0041 | 0.1510±0.0024 | 0.3803±0.0037 | 0.0449±0.0002 |
| `paper58_latent_dynamics` | `xiangzhen_record_000191` | 5 | 0.3760±0.0000 | 0.1502±0.0000 | 0.3740±0.0000 | 0.0974±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_002058` | 5 | 0.2265±0.0000 | 0.1037±0.0000 | 0.3378±0.0000 | 0.1143±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_002815` | 5 | 0.3584±0.0000 | 0.0907±0.0000 | 0.2861±0.0000 | 0.1140±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_003053` | 5 | 0.1441±0.0000 | 0.0776±0.0000 | 0.4146±0.0000 | 0.0391±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_003882` | 5 | 0.3261±0.0000 | 0.1487±0.0000 | 0.4064±0.0000 | 0.0807±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_010308` | 5 | 0.2954±0.0000 | 0.1101±0.0000 | 0.2378±0.0000 | 0.0917±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_015068` | 5 | 0.4304±0.0000 | 0.2036±0.0000 | 0.4838±0.0000 | 0.1102±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_019130` | 5 | 0.2404±0.0000 | 0.0525±0.0000 | 0.2439±0.0000 | 0.0591±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_019254` | 5 | 0.2220±0.0000 | 0.0728±0.0000 | 0.1308±0.0000 | 0.0862±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_020866` | 5 | 0.3589±0.0000 | 0.1658±0.0000 | 0.4521±0.0000 | 0.1014±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_021584` | 5 | 0.0559±0.0000 | 0.0245±0.0000 | 0.3303±0.0000 | 0.0208±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_021766` | 5 | 0.2737±0.0000 | 0.1424±0.0000 | 0.4410±0.0000 | 0.2159±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_022880` | 5 | 0.1753±0.0000 | 0.0873±0.0000 | 0.4171±0.0000 | 0.1125±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_025932` | 5 | 0.4943±0.0000 | 0.2162±0.0000 | 0.3727±0.0000 | 0.2028±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_027267` | 5 | 0.2837±0.0000 | 0.0773±0.0000 | 0.2116±0.0000 | 0.0313±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_028657` | 5 | 0.2950±0.0000 | 0.1148±0.0000 | 0.3953±0.0000 | 0.0383±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_030462` | 5 | 0.4608±0.0000 | 0.1805±0.0000 | 0.4763±0.0000 | 0.0230±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_031108` | 5 | 0.3186±0.0000 | 0.1513±0.0000 | 0.5070±0.0000 | 0.0891±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_031426` | 5 | 0.2576±0.0000 | 0.1057±0.0000 | 0.3608±0.0000 | 0.2593±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_031513` | 5 | 0.3040±0.0000 | 0.1321±0.0000 | 0.3590±0.0000 | 0.0297±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_033687` | 5 | 0.1743±0.0000 | 0.0740±0.0000 | 0.4301±0.0000 | 0.0501±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_038533` | 5 | 0.1448±0.0000 | 0.0606±0.0000 | 0.2941±0.0000 | 0.0635±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_038743` | 5 | 0.1735±0.0000 | 0.0876±0.0000 | 0.5026±0.0000 | 0.0201±0.0000 |
| `paper58_latent_dynamics` | `xiangzhen_record_042889` | 5 | 0.2519±0.0000 | 0.1079±0.0000 | 0.4404±0.0000 | 0.0447±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_000191` | 5 | 0.3583±0.0000 | 0.1641±0.0000 | 0.2987±0.0000 | 0.0518±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_002058` | 5 | 0.3399±0.0000 | 0.1893±0.0000 | 0.2854±0.0000 | 0.1254±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_002815` | 5 | 0.2765±0.0000 | 0.0874±0.0000 | 0.1778±0.0000 | 0.0716±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_003053` | 5 | 0.1402±0.0000 | 0.0754±0.0000 | 0.3659±0.0000 | 0.0069±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_003882` | 5 | 0.3064±0.0000 | 0.1246±0.0000 | 0.2449±0.0000 | 0.0455±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_010308` | 5 | 0.2953±0.0000 | 0.1213±0.0000 | 0.2445±0.0000 | 0.1165±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_015068` | 5 | 0.2759±0.0000 | 0.1297±0.0000 | 0.2176±0.0000 | 0.0651±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_019130` | 5 | 0.2647±0.0000 | 0.0650±0.0000 | 0.1870±0.0000 | 0.0524±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_019254` | 5 | 0.1291±0.0000 | 0.0392±0.0000 | 0.0575±0.0000 | 0.0357±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_020866` | 5 | 0.3535±0.0000 | 0.1433±0.0000 | 0.2920±0.0000 | 0.1044±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_021584` | 5 | 0.3013±0.0000 | 0.1700±0.0000 | 0.4924±0.0000 | 0.0083±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_021766` | 5 | 0.3299±0.0000 | 0.1769±0.0000 | 0.3241±0.0000 | 0.1501±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_022880` | 5 | 0.2591±0.0000 | 0.1467±0.0000 | 0.3490±0.0000 | 0.0081±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_025932` | 5 | 0.4757±0.0000 | 0.1996±0.0000 | 0.2883±0.0000 | 0.1640±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_027267` | 5 | 0.3383±0.0000 | 0.1160±0.0000 | 0.2519±0.0000 | 0.0301±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_028657` | 5 | 0.2720±0.0000 | 0.1032±0.0000 | 0.3043±0.0000 | 0.0225±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_030462` | 5 | 0.1676±0.0000 | 0.0169±0.0000 | 0.1558±0.0000 | 0.0152±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_031108` | 5 | 0.3353±0.0000 | 0.1408±0.0000 | 0.3186±0.0000 | 0.1067±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_031426` | 5 | 0.3179±0.0000 | 0.1325±0.0000 | 0.2917±0.0000 | 0.1172±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_031513` | 5 | 0.3803±0.0000 | 0.1739±0.0000 | 0.3419±0.0000 | 0.0168±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_033687` | 5 | 0.1952±0.0000 | 0.0890±0.0000 | 0.3419±0.0000 | 0.0177±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_038533` | 5 | 0.1406±0.0000 | 0.0664±0.0000 | 0.2810±0.0000 | 0.0171±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_038743` | 5 | 0.2108±0.0000 | 0.1148±0.0000 | 0.4021±0.0000 | 0.0154±0.0000 |
| `paper58_spatial_demand_ratio_external_loo_transition_floor030` | `xiangzhen_record_042889` | 5 | 0.4549±0.0000 | 0.2087±0.0000 | 0.3557±0.0000 | 0.0363±0.0000 |

## 按种子总体均值的稳健性

该表先在每个随机种子内对全部区域求平均，再比较 Paper58 与 GeoSOS-FLUS；它对应“每次完整实验总体是否胜出”。

| 指标 | 平均差值 | 标准差 | 最小差值 | 最大差值 | 胜出种子 | 胜出率 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 变化 F1 | 0.0220 | 0.0006 | 0.0214 | 0.0229 | 5/5 | 1.0000 | 越高越好 |
| FoM | 0.0042 | 0.0007 | 0.0034 | 0.0050 | 5/5 | 1.0000 | 越高越好 |
| 转换准确率 | 0.0061 | 0.0013 | 0.0045 | 0.0079 | 5/5 | 1.0000 | 越高越好 |
| 分配分歧 | -0.0004 | 0.0007 | -0.0014 | 0.0004 | 3/5 | 0.6000 | 越低越好 |

### 每个种子的总体差值

| 种子 | 指标 | Paper58 均值 | GeoSOS-FLUS 均值 | 差值 | 是否胜出 |
| ---: | --- | ---: | ---: | ---: | --- |
| 1001 | 变化 F1 | 0.2883 | 0.2665 | 0.0217 | 是 |
| 1001 | FoM | 0.1248 | 0.1212 | 0.0036 | 是 |
| 1001 | 转换准确率 | 0.2862 | 0.2818 | 0.0045 | 是 |
| 1001 | 分配分歧 | 0.0584 | 0.0592 | -0.0008 | 是 |
| 1002 | 变化 F1 | 0.2883 | 0.2662 | 0.0221 | 是 |
| 1002 | FoM | 0.1248 | 0.1198 | 0.0050 | 是 |
| 1002 | 转换准确率 | 0.2862 | 0.2796 | 0.0067 | 是 |
| 1002 | 分配分歧 | 0.0584 | 0.0598 | -0.0014 | 是 |
| 1003 | 变化 F1 | 0.2883 | 0.2654 | 0.0229 | 是 |
| 1003 | FoM | 0.1248 | 0.1202 | 0.0045 | 是 |
| 1003 | 转换准确率 | 0.2862 | 0.2784 | 0.0079 | 是 |
| 1003 | 分配分歧 | 0.0584 | 0.0585 | -0.0001 | 是 |
| 1004 | 变化 F1 | 0.2883 | 0.2669 | 0.0214 | 是 |
| 1004 | FoM | 0.1248 | 0.1214 | 0.0034 | 是 |
| 1004 | 转换准确率 | 0.2862 | 0.2807 | 0.0055 | 是 |
| 1004 | 分配分歧 | 0.0584 | 0.0580 | 0.0004 | 否 |
| 1005 | 变化 F1 | 0.2883 | 0.2664 | 0.0219 | 是 |
| 1005 | FoM | 0.1248 | 0.1203 | 0.0045 | 是 |
| 1005 | 转换准确率 | 0.2862 | 0.2804 | 0.0059 | 是 |
| 1005 | 分配分歧 | 0.0584 | 0.0582 | 0.0002 | 否 |

## 区域×种子配对稳健性

该表把每个区域、每个种子都作为一个配对样本，用来暴露哪些区域仍是短板。

| 指标 | 平均差值 | 标准差 | 最小差值 | 最大差值 | 胜出次数 | 胜出率 | 判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 变化 F1 | 0.0220 | 0.0806 | -0.1605 | 0.1950 | 65/120 | 0.5417 | 越高越好 |
| FoM | 0.0042 | 0.0528 | -0.1377 | 0.1122 | 61/120 | 0.5083 | 越高越好 |
| 转换准确率 | 0.0061 | 0.1146 | -0.1767 | 0.2426 | 54/120 | 0.4500 | 越高越好 |
| 分配分歧 | -0.0004 | 0.0301 | -0.0363 | 0.0977 | 81/120 | 0.6750 | 越低越好 |

## 核心判读

- 按每个种子的 24 区域总体均值，`paper58_spatial_demand_ratio_external_loo_transition_floor030` 在 3 个指标上对 `geosos_flus_console` 达到 100% 种子胜出：变化 F1、FoM、转换准确率。
- 按区域×种子配对，`paper58_spatial_demand_ratio_external_loo_transition_floor030` 达到 100% 配对胜出的指标：无。
- `paper58_spatial_demand_ratio_external_loo_transition_floor030` 在以下总体指标上没有超过 `geosos_flus_console`：无。
- `paper58_spatial_demand_ratio_external_loo_transition_floor030` 在以下区域×种子配对指标上完全没有超过 `geosos_flus_console`：无。
- 因此，本轮 24 个真实乡镇样本支持“Paper58 优化后在总体均值层面稳定超过 GeoSOS-FLUS”；但区域配对胜率仍不是 100%，论文表述应保留区域异质性和失败案例。

## 结论边界

- 这份报告检验的是 GeoSOS-FLUS 随机分配项对结论的影响，不改变输入数据、Paper58 权重或 GeoSOS-FLUS 参数。
- 若某个指标胜出率为 1.0000，说明在本组随机种子下该结论不依赖单次 FLUS 随机结果。
- 分配分歧是越低越好；其差值为正表示 Paper58 对应方法仍高于 GeoSOS-FLUS。
