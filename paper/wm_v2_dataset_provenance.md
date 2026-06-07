# 世界模型 v2 背后的原始数据集

> 对应 `D:/adk/data_agent/world_model_v2.py` (v2.2.0) 和 Paper 9 v6 的底层数据清单
> 覆盖 Bishan District, Chongqing（璧山区）县级耕地布局优化任务

---

## 一、概览：三级数据流水

World Model v2 的服务代码最终通过 `county_env.py` 读取**一个预处理后**的 GeoPackage 文件。这个 GPKG 是由**三个原始源数据**经过 DEM-slope pipeline 拼出来的。

```
① GDB.gdb (DLTB 地类图斑)    ─┐
② xiangzhen.shp (乡镇边界)    ├→ dem_slope_zonal.py ─→ DLTB_with_slope.gpkg
③ Copernicus DEM .tif         ─┘                          (Agent 直接读的)
```

PPO / dream_v5 / MPC 三 mode **共用一份底层数据**，差异只在模型训练方式。

---

## 二、三个最原始数据文件

### ① 地类图斑（Third National Land Survey 第三次全国国土调查标准）

| 属性 | 内容 |
|------|------|
| **路径** | `D:/test/现状用地数据/GDB.gdb/` |
| **Layer** | `DLTB`（地类图斑） |
| **格式** | ESRI File GeoDatabase（~140 KB 头 + 多个 `a000000XX.*` 内部分块） |
| **日期** | 源数据 2019-05-16（第三次全国国土调查发布节点） |
| **CRS** | 声称 EPSG:4610 (Xian 1980)，**实际含 CGCS2000 坐标**（`dem_slope_zonal.py` 注释特别标注了"常见元数据错误"） |
| **关键字段** | `BSM`（图斑识别码）、`DLBM`（地类编码，3 位数字如 `011`）、`DLMC`（地类名称）、`QSDWDM`（权属单位代码→township code）、`TBMJ`（图斑面积） |
| **记录数** | 璧山全区几十万图斑级（过滤后得 52,515 swappable parcel） |

**地类编码语义**（在 `county_env.py:36-37` 和 `dem_slope_zonal.py:45-47`）：

| DLBM 前缀 | 类别 | 参与优化？ |
|----------|------|-----------|
| `011` 水田、`012` 水浇地、`013` 旱地 | **耕地 (FARMLAND)** | ✅ 可互换 |
| `031` 有林地、`032` 灌木林地、`033` 其它林地 | **林地 (FOREST)** | ✅ 可互换 |
| `021` 果园、`022` 茶园、`023` 橡胶园 | 园地 | ❌ 冻结 |
| 其它所有前缀 | Other | ❌ 冻结 |

### ② 乡镇行政边界

| 属性 | 内容 |
|------|------|
| **路径** | `D:/test/xiangzhen.shp` + `.dbf / .prj / .sbn / .sbx / .shp.xml / .shx / .cpg / .qpj` |
| **格式** | ESRI Shapefile（`.shp` 主文件 317 MB，`.dbf` 74 MB） |
| **作用** | 提供 13 个乡镇的权属代码 `QSDWDM`，与 DLTB 做空间/属性关联 |
| **日期** | 2026-03-01 导入工作区 |

13 个乡镇代码在 `county_env.py:48-62` 硬编码：

```
500227001 T01-Bishan      (璧城街道)
500227002 T02-Qinggang    (青杠街道)
500227100 T03-Hechuan     (河边镇)
500227101 T04-Laifeng     (来凤街道)
500227102 T05-Guangpu     (广普镇)
500227103 T06-Daxing      (大兴镇)
500227104 T07-Zhengxing   (正兴镇)
500227105 T08-Dalukou     (大路街道)
500227106 T09-Hebian      (河边镇)
500227107 T10-Shihe       (石河街道)
500227108 T11-Baxian      (八塘镇)
500227109 T12-Jianlong    (健龙镇)
500227200 T13-Qinglonghu  (青龙湖镇)
```

### ③ Copernicus DEM（数字高程）

| 属性 | 内容 |
|------|------|
| **路径** | `D:/test/dem_slope_analysis/intermediate/Copernicus_DSM_COG_10_N29_00_E106_00_DEM.tif` |
| **格式** | GeoTIFF（45 MB） |
| **源** | Copernicus GLO-30（AWS S3 开放数据，30 米分辨率 ≈ 1 arcsec） |
| **覆盖** | N29°—N30° × E106°—E107°（含璧山全境） |
| **用途** | 计算 parcel 级坡度（`slope_degrees.npy`）→ zonal statistics 聚合到图斑 |

配套中间产物（由 `dem_slope_zonal.py` 产生）：

| 文件 | 大小 | 内容 |
|------|------|------|
| `dem_full_tile.npy` | 50 MB | 全 tile DEM 数组 |
| `slope_degrees.npy` | 50 MB | 对应的坡度数组（per pixel） |
| `aspect_degrees.npy` | 50 MB | 坡向 |
| `parcels_attributes.csv` | 20 MB | per-parcel 属性表 |
| `pixel_parcel_matches.csv` | 2.6 MB | 像素-图斑对应关系 |

---

## 三、中间产物（Agent 实际读的文件）

### `DLTB_with_slope.gpkg` — 核心工作文件

| 属性 | 内容 |
|------|------|
| **路径** | `D:/test/dem_slope_analysis/output/DLTB_with_slope.gpkg` |
| **大小** | **153 MB** |
| **CRS** | **EPSG:4326**（已从 CGCS2000 重投到 WGS84） |

**Schema**：

| 字段 | 来源 | 说明 |
|------|------|------|
| `BSM` | DLTB 原始 | 图斑识别码 |
| `YSDM` | DLTB 原始 | 要素代码 |
| `DLBM` | DLTB 原始 | 地类编码（3 位，如 `011`） |
| `DLMC` | DLTB 原始 | 地类名称（中文） |
| `QSDWDM` | DLTB 原始 | 权属单位代码（township code） |
| `QSDWMC` | DLTB 原始 | 权属单位名称 |
| `ZLDWDM` | DLTB 原始 | 坐落单位代码 |
| `ZLDWMC` | DLTB 原始 | 坐落单位名称 |
| `TBMJ` | DLTB 原始 | 图斑面积 |
| `SHAPE_Length` | DLTB 原始 | 周长 |
| `SHAPE_Area` | DLTB 原始 | 面积 |
| `category` | pipeline 加的 | 粗分类 (Farmland / Forest / Orchard / Other) |
| `slope_mean` | pipeline 加的 | parcel 级平均坡度（度） |
| `slope_max` | pipeline 加的 | parcel 级最大坡度 |
| `slope_pixel_count` | pipeline 加的 | zonal 统计的像素数 |
| `geometry` | DLTB 原始 | MultiPolygon，已重投到 WGS84 |

**这是 `county_env.py` Line 29 的 `DLTB_PATH` 指向的文件**，也是 Agent 里所有 PPO/dream_v5/MPC 三 mode 实际读的那一个文件。

### `results_real/blocks/` — block 聚合产物

| 路径 | 内容 |
|------|------|
| `D:/test/results_real/blocks/county/` | county-level block 聚合（2600 blocks） |
| `D:/test/results_real/blocks/township_500227XXX/` | 每个乡镇独立的 block 定义 |
| `D:/test/results_real/blocks/county_summary.json` | 3 KB 汇总元信息 |

这些 block 文件是 **Paper 3 的产物**（hybrid DLTB barriers + AgglomerativeClustering），把 52,515 parcel 聚成 2,600 block。`county_env.py` 启动时 load 这里的 block 定义。

---

## 四、数据流完整图

```
原始数据（上游）                    预处理脚本                        Agent 读取
─────────────────────              ────────────────                ──────────
D:/test/现状用地数据/GDB.gdb    ─┐
  └─ DLTB layer                   │
D:/test/xiangzhen.shp           ─┼─→ dem_slope_zonal.py    ───→  DLTB_with_slope.gpkg
(13 townships)                    │   (step 1: read GDB,            │ (153 MB)
D:/test/dem_slope_analysis/      ─┘    step 2: DEM+slope,           │
  intermediate/Copernicus_            step 3: zonal stats,         │ ← county_env.py
  DSM_COG_10_N29_00_E106_00_          step 4: write GPKG)          │   DLTB_PATH
  DEM.tif (45 MB, Copernicus)                                       │
                                                                    │
                    paper3 block clustering    ───→  results_real/blocks/county/
                                                         │ (2600 blocks)
                                                         │
                                                         ↓
                              county_env.CountyLevelEnv.__init__
                                                         │
                                                         ↓
                                WorldModelV2Service._create_env()
                                                         │
                                                         ↓
                                   PPO / dream_v5 / MPC 三 mode 共用
```

---

## 五、数据规模核对

| 指标 | 数值 | 来源 |
|------|------|------|
| 总 parcel（图斑） | ~几十万 | GDB.gdb DLTB layer |
| Swappable parcel（仅 farm + forest） | **52,515** | county_env init log |
| Townships | 13 | xiangzhen.shp |
| Blocks | **2,600** | results_real/blocks/county_summary.json |
| Cross-township parcel edges | 3,290 | county_env 启动时打印 |
| Initial avg farmland slope | **9.6157°** | 所有 paper 都用这个基准 |
| Initial contiguity | 3.5852 | Paper 4 / 9 基准 |
| Initial baimu fang | **109 patches / 46,844 ha** | 所有 paper 基准 |
| DEM tile size | 3600 × 3600 pixels (1° × 1°) | Copernicus GLO-30, 30m res |

---

## 六、数据敏感性与可开源性

| 文件 | 能否公开 | 备注 |
|------|---------|------|
| `GDB.gdb`（DLTB） | ❌ 不能 | 第三次全国国土调查原始数据，国土部门管制 |
| `xiangzhen.shp` | ⚠️ 可能受限 | 行政边界一般不敏感但包含详细属性 |
| Copernicus DEM `.tif` | ✅ 开源 | ESA 开放数据，AWS S3 直接下载 |
| `DLTB_with_slope.gpkg` | ❌ 衍生品仍受管制 | 含原始 DLBM/QSDWDM/BSM 所以一并受限 |
| `parcels_attributes.csv` | ❌ 同上 | parcel 级属性表 |
| **Block-level features** (17-d) | ✅ Paper 9 承诺公开 | 聚合后已脱敏 |
| **Pairwise dataset** D_pw | ✅ Paper 9 承诺公开 | 只含 block features + action + reward，不含 parcel 级敏感信息 |

### Paper 9 v6 Data Availability 的底层依据

Paper 9 v6 的 Data Availability 段承诺：

> "Raw parcel-level data cannot be publicly shared. Aggregated block-level features, pairwise dataset, seeds/hyperparameters/logs, and a planned synthetic benchmark will be released."

这一承诺的数据安全边界就在 **block features**：

- Parcel 级别（`BSM / DLBM / QSDWDM / TBMJ / geometry`）→ 受国土调查管制，**不能开源**
- Block 级别（17 维匿名数值特征：farmland area, forest area, slope stats, adjacency counts, investment indicators）→ 已脱敏，**可以开源**
- Pairwise dataset 也是在 block features 上收集的，不含 parcel 级身份信息

所以 reviewer 拿到 block features + pairwise data + code，**可以完整复现 contrastive world model 训练和 MPC 评估管线**，不需要触碰受管制的 parcel 级原始数据。

---

## 七、关键文件路径速查

| 用途 | 路径 |
|------|------|
| Agent 主入口（Python） | `D:/adk/data_agent/world_model_v2.py` |
| Env 定义 | `D:/test/county_env.py` |
| Agent 读取的核心数据 | `D:/test/dem_slope_analysis/output/DLTB_with_slope.gpkg` |
| Block 定义 | `D:/test/results_real/blocks/county/` |
| 原始 DLTB | `D:/test/现状用地数据/GDB.gdb/` (layer: `DLTB`) |
| 乡镇边界 | `D:/test/xiangzhen.shp` |
| 原始 DEM | `D:/test/dem_slope_analysis/intermediate/Copernicus_DSM_COG_10_N29_00_E106_00_DEM.tif` |
| 预处理脚本 | `D:/test/dem_slope_analysis/dem_slope_zonal.py` |

---

*生成于 2026-05-08，对应 World Model v2 v2.2.0 和 Paper 9 v6 的底层数据状态。*
