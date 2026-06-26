# Paper58-LAS 与 GeoSOS-FLUS/FLUS Console 对比结果图文报告

生成日期：2026-06-26  
代码状态：`1895b40`  
核心结果来源：`paper/rse_submission_paper58/las_results_batch5_adaptive_demand_grid_w0.75_l1_0.14_change_0.22_neigh_adapt_budget_gate_low0.13_high0.30_scale_0.82_0.85_with_flus`

> 术语边界：这里的 GeoSOS-FLUS 对比结果，严格说是 **matched official FLUS console baseline**。它使用官方 FLUS console 路线构造可复现实验基线，并与 Paper58-LAS 在同一 Batch 5 holdout、同一评价指标、同一成对区域上比较；它还不是 GeoSOS-FLUS native GUI/完整传统驱动因子工作流的最终系统级对比。

## 一句话结论

在 7 个 Batch 5 tier-1 matched holdout 区域上，当前最稳妥的 Paper58-LAS adaptive change-budget gate 相对 official FLUS console baseline 取得 `Change F1` 平均优势 `+0.2049`，bootstrap CI `+0.0902 to +0.3285`，F1 sign test `7/7` positive, `p=0.015625`；`FoM` 平均优势 `+0.0802`，bootstrap CI `+0.0355 to +0.1315`。

## 结果在哪里

| 内容 | 位置 |
| --- | --- |
| 最新核心进度摘要 | `docs/current_work_progress_2026-06-25.md` |
| 最新 matched FLUS 比较目录 | `paper/rse_submission_paper58/las_results_batch5_adaptive_demand_grid_w0.75_l1_0.14_change_0.22_neigh_adapt_budget_gate_low0.13_high0.30_scale_0.82_0.85_with_flus` |
| 方法均值与 bootstrap/sign-test 汇总 | `paper/rse_submission_paper58/las_results_batch5_adaptive_demand_grid_w0.75_l1_0.14_change_0.22_neigh_adapt_budget_gate_low0.13_high0.30_scale_0.82_0.85_with_flus/comparison_las_vs_flus/las_comparison_summary.json` |
| 逐区域 paper58_las vs flus 对比 | `paper/rse_submission_paper58/las_results_batch5_adaptive_demand_grid_w0.75_l1_0.14_change_0.22_neigh_adapt_budget_gate_low0.13_high0.30_scale_0.82_0.85_with_flus/comparison_las_vs_flus/las_comparison_by_area.csv` |
| 三方法逐区域指标 | `paper/rse_submission_paper58/las_results_batch5_adaptive_demand_grid_w0.75_l1_0.14_change_0.22_neigh_adapt_budget_gate_low0.13_high0.30_scale_0.82_0.85_with_flus/las_metrics_by_method.csv` |
| LOAO 候选选择审计目录 | `paper/rse_submission_paper58/las_results_batch5_budget_gate_loo_audit_same_registry` |
| 历史长报告 | `paper/rse_submission_paper58/paper58_geosos_flus_comparison_report_2026-06-24.md` |
| 旧版图文报告 | `paper/rse_submission_paper58/visual_reports/paper58_vs_flus_transition_prior_2026-06-24/README.md` |

## 对比方法是否科学

本报告只使用 matched comparison 与 leave-one-area-out candidate selection audit 两类证据：

1. matched comparison：同一批 7 个区域逐区域成对比较 Paper58-LAS 与 official FLUS console baseline，核心指标包括 Change F1、FoM、Recall、Transition accuracy，以及 quantity/allocation disagreement。
2. bootstrap CI：以区域为成对单元估计平均优势的不确定性。
3. paired sign test：检验优势方向是否在区域间稳定，而不是只靠少数区域拉高均值。
4. LOAO selection：每次留出一个区域，只用另外 6 个区域选择候选设置，再在留出区域评价，避免 post-hoc per-area tuning。
5. 显式边界：不能把当前结果直接写成“已经全面超过 GeoSOS-FLUS native workflow”。当前可证明的是超过 matched official FLUS console baseline。

![Method schematic](figures/fig5_method_schematic.png)

## 总体指标：Paper58-LAS 已明显高于 FLUS console 基线

| 方法 | Change F1 | FoM | Recall | Transition acc. | Allocation disagr. |
| --- | --- | --- | --- | --- | --- |
| Official FLUS console | 0.101 | 0.030 | 0.123 | 0.084 | 0.049 |
| Paper58-LAS | 0.306 | 0.110 | 0.519 | 0.334 | 0.088 |

![Mean metrics](figures/fig1_method_mean_metrics.png)

从均值看，Paper58-LAS 的 Change F1 从 FLUS console 的 `0.101` 提高到 `0.306`，FoM 从 `0.030` 提高到 `0.110`。Recall 和 transition accuracy 也同步提高，这说明优势不是只来自一个单指标。

## 平均优势与置信区间

| 指标 | 平均优势 | bootstrap CI | sign-test positive/effective | two-sided p |
| --- | --- | --- | --- | --- |
| Change F1 | +0.2049 | [+0.0902, +0.3285] | 7/7 | 0.015625 |
| FoM | +0.0802 | [+0.0355, +0.1315] | 6/7 | 0.125 |
| Recall | +0.3954 | [+0.2280, +0.5735] | 7/7 | 0.015625 |
| Transition acc. | +0.2495 | [+0.1309, +0.3849] | 6/6 | 0.03125 |
| Quantity disagr. | -0.0154 | [-0.0467, +0.0108] | 1/4 | 0.625 |
| Allocation disagr. | -0.0392 | [-0.0742, -0.0121] | 1/7 | 0.125 |

![Advantage with CI](figures/fig2_advantage_ci.png)

解释要点：

- Change F1：`7/7` 区域为正，`p=0.015625`，这是当前最硬的主证据。
- FoM：`6/7` 区域为正，均值优势和 bootstrap CI low 均为正，但 sign test 受小样本限制没有达到同等强度。
- Recall 与 transition accuracy：均为正优势，说明 Paper58-LAS 更能抓住实际变化及其转移方向。
- Allocation disagreement：当前为负优势，表示 Paper58-LAS 的空间分配误差仍高于 FLUS console，这是下一轮优化的主要缺口。

## 逐区域结果

| 区域 | 地类层 | F1 advantage | FoM advantage | Transition acc. advantage | Allocation disagr. advantage |
| --- | --- | --- | --- | --- | --- |
| dabie_forest_edge_holdout | Forest | +0.1504 | +0.0882 | +0.2609 | -0.0208 |
| huaibei_irrigation_plain_holdout | Agriculture | +0.0011 | -0.0127 | +0.0000 | -0.0227 |
| liaohe_delta_wetland_holdout | Wetland | +0.4639 | +0.1213 | +0.2419 | -0.0529 |
| renqiu_baiyangdian_edge_holdout | Urban | +0.1208 | +0.0619 | +0.1277 | -0.0454 |
| wenan_lakeplain_newtown_holdout | Urban | +0.0870 | +0.0455 | +0.4000 | -0.1341 |
| wuxi_taihu_dense_edge_holdout | Urban | +0.1614 | +0.0474 | +0.1481 | -0.0113 |
| xilingol_grassland_margin_holdout | Grassland | +0.4496 | +0.2100 | +0.5676 | +0.0127 |

![Area advantages](figures/fig3_area_advantages.png)

区域层面的读法：

- F1 在 7 个区域全部为正，说明当前结论不是单一区域驱动。
- FoM 在 6 个区域为正，唯一负例是 `huaibei_irrigation_plain_holdout`，其 FoM advantage 为 `-0.0127`。
- `liaohe_delta_wetland_holdout` 与 `xilingol_grassland_margin_holdout` 是当前增益最明显的两个区域。
- `wenan_lakeplain_newtown_holdout` 的 F1/FoM 虽为正，但 allocation disagreement 劣势最大，说明空间 placement 仍需优化。

## LOAO 候选选择审计

LOAO 候选池：`adaptive082, adaptive085, adaptive_budget_gate`。主指标为 `change_f1`，tie-break 指标为 `fom`。

![LOAO selection](figures/fig4_loao_selection.png)

LOAO 审计中，`adaptive_budget_gate` 在 7 次留一审计里全部被选中。对应 held-out 平均优势与 matched comparison 一致：Change F1 `+0.2049`，FoM `+0.0802`。这比“在 7 个区域上直接挑最优设置”更可靠，因为每次被评估区域没有参与候选选择。

## 当前可以说什么，不能说什么

可以说：

- Paper58-LAS 当前 adaptive change-budget gate 已经超过 matched official FLUS console baseline。
- 超越证据主要体现在 Change F1、FoM、Recall 与 Transition accuracy。
- LOAO candidate selection 结果支持该候选不是简单的逐区域事后调参产物。

不能说：

- 还不能说 Paper58-LAS 已经全面超过 GeoSOS-FLUS native workflow。
- 还不能说空间 allocation 已全面优于 FLUS，因为 allocation disagreement 仍是负优势。
- 不能把 side-base gate、churn gate、hard-label de-emphasis 等全样本小幅增益候选写成默认模型，除非它们也通过同样的 LOAO 选择协议。

## 下一步建议

最直接的模型优化方向不是继续只压低总变化量，而是减少 `huaibei`、`wenan` 这类区域的 reciprocal swap churn 与空间误配；方法上应继续用同一 registry 的 matched comparison + LOAO explicit-priority selection 验证。若要支撑更强的“超过 GeoSOS-FLUS”表述，需要补充 native GeoSOS-FLUS workflow 对比。
