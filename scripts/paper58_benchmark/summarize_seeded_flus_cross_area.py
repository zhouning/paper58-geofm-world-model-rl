from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper58_benchmark.run_seeded_flus_replicates import LOWER_IS_BETTER, METRICS


METRIC_LABELS = {
    "change_f1": "变化 F1",
    "fom": "FoM",
    "transition_accuracy": "转换准确率",
    "allocation_disagreement": "分配分歧",
}


@dataclass(frozen=True)
class CrossAreaSeededSummary:
    created_at_utc: str
    seeded_report_dirs: list[Path]
    area_delta_rows: list[dict[str, Any]]
    metric_rows: list[dict[str, Any]]
    challenger: str
    baseline: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_ready(row.get(field, "")) for field in fields})


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any) -> float:
    return float(str(value))


def _int(value: Any) -> int:
    return int(float(str(value)))


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.{digits}f}"


def _infer_area(report_dir: Path) -> str:
    method_rows = _read_csv(Path(report_dir) / "seeded_metric_summary_by_method.csv")
    areas = sorted({str(row.get("area", "")).strip() for row in method_rows if row.get("area")})
    if not areas:
        raise ValueError(f"cannot infer area from {report_dir}")
    return "+".join(areas)


def _benefit(metric: str, mean_delta: float) -> float:
    return -mean_delta if metric in LOWER_IS_BETTER else mean_delta


def _load_area_rows(
    report_dir: Path,
    challenger: str,
    baseline: str,
) -> list[dict[str, Any]]:
    area = _infer_area(report_dir)
    rows: list[dict[str, Any]] = []
    for row in _read_csv(Path(report_dir) / "seeded_delta_summary.csv"):
        if row.get("challenger") != challenger or row.get("baseline") != baseline:
            continue
        metric = str(row["metric"])
        n = _int(row["n"])
        n_better = _int(row["n_better"])
        mean_delta = _float(row["mean_delta"])
        higher_is_better = _bool(row["higher_is_better"])
        benefit = _benefit(metric, mean_delta)
        rows.append(
            {
                "area": area,
                "report_dir": Path(report_dir),
                "challenger": challenger,
                "baseline": baseline,
                "metric": metric,
                "metric_label": _metric_label(metric),
                "n": n,
                "mean_delta": mean_delta,
                "std_delta": _float(row["std_delta"]),
                "min_delta": _float(row["min_delta"]),
                "max_delta": _float(row["max_delta"]),
                "n_better": n_better,
                "better_rate": _float(row["better_rate"]),
                "higher_is_better": higher_is_better,
                "mean_benefit": benefit,
                "area_better": benefit > 0.0,
                "all_seed_better": n > 0 and n_better == n,
                "all_seed_worse": n > 0 and n_better == 0,
            }
        )
    return rows


def summarize_cross_area(
    seeded_report_dirs: list[Path],
    challenger: str = "paper58_change_gate",
    baseline: str = "geosos_flus_console",
) -> CrossAreaSeededSummary:
    area_delta_rows: list[dict[str, Any]] = []
    for report_dir in seeded_report_dirs:
        area_delta_rows.extend(_load_area_rows(Path(report_dir), challenger, baseline))

    metric_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        subset = [row for row in area_delta_rows if row["metric"] == metric]
        if not subset:
            continue
        n_total = sum(int(row["n"]) for row in subset)
        n_better_total = sum(int(row["n_better"]) for row in subset)
        mean_deltas = np.asarray([float(row["mean_delta"]) for row in subset], dtype=np.float64)
        benefits = np.asarray([float(row["mean_benefit"]) for row in subset], dtype=np.float64)
        metric_rows.append(
            {
                "metric": metric,
                "metric_label": _metric_label(metric),
                "n_areas": len(subset),
                "n_seed_pairs": n_total,
                "mean_area_delta": float(np.mean(mean_deltas)),
                "median_area_delta": float(np.median(mean_deltas)),
                "mean_area_benefit": float(np.mean(benefits)),
                "median_area_benefit": float(np.median(benefits)),
                "area_better_count": sum(1 for row in subset if bool(row["area_better"])),
                "area_all_seed_better_count": sum(1 for row in subset if bool(row["all_seed_better"])),
                "area_all_seed_worse_count": sum(1 for row in subset if bool(row["all_seed_worse"])),
                "weighted_seed_better_rate": float(n_better_total / n_total) if n_total else None,
                "higher_is_better": metric not in LOWER_IS_BETTER,
            }
        )

    return CrossAreaSeededSummary(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        seeded_report_dirs=[Path(path) for path in seeded_report_dirs],
        area_delta_rows=area_delta_rows,
        metric_rows=metric_rows,
        challenger=challenger,
        baseline=baseline,
    )


def _relative_link(output_dir: Path, target: Path) -> str:
    try:
        return Path(target).resolve().relative_to(Path(output_dir).resolve()).as_posix()
    except ValueError:
        return Path("../" + Path(target).resolve().relative_to(Path(output_dir).resolve().parent).as_posix()).as_posix()


def _write_manifest(summary: CrossAreaSeededSummary, output_dir: Path, same_grid_report_dirs: list[Path]) -> None:
    payload = {
        "created_at_utc": summary.created_at_utc,
        "challenger": summary.challenger,
        "baseline": summary.baseline,
        "seeded_report_dirs": summary.seeded_report_dirs,
        "same_grid_report_dirs": same_grid_report_dirs,
        "metric_rows": summary.metric_rows,
        "area_delta_rows": summary.area_delta_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _make_figures(summary: CrossAreaSeededSummary, output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    area_order = sorted({str(row["area"]) for row in summary.area_delta_rows})
    metric_order = [metric for metric in METRICS if any(row["metric"] == metric for row in summary.area_delta_rows)]
    data = {
        (str(row["area"]), str(row["metric"])): float(row["mean_benefit"])
        for row in summary.area_delta_rows
    }
    x = np.arange(len(area_order))
    width = 0.18 if len(metric_order) > 3 else 0.24
    colors = ["#2C7FB8", "#3B8C6E", "#B48A2C", "#B24C4A"]
    fig, ax = plt.subplots(figsize=(max(7.2, len(area_order) * 1.5), 4.0))
    for index, metric in enumerate(metric_order):
        values = [data.get((area, metric), 0.0) for area in area_order]
        ax.bar(x + (index - (len(metric_order) - 1) / 2) * width, values, width=width, label=_metric_label(metric), color=colors[index % len(colors)])
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_ylabel("Paper58 相对 GeoSOS-FLUS 的收益（正值更好）")
    ax.set_title("跨区域指标收益", loc="left", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([area.replace("_", "\n") for area in area_order], fontsize=7)
    ax.legend(ncol=min(4, len(metric_order)), loc="upper right")
    fig.tight_layout()
    path1 = figure_dir / "cross_area_metric_benefit.png"
    fig.savefig(path1, dpi=220, bbox_inches="tight")
    plt.close(fig)

    labels = [_metric_label(str(row["metric"])) for row in summary.metric_rows]
    rates = [float(row["weighted_seed_better_rate"]) for row in summary.metric_rows]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    bars = ax.bar(labels, rates, color=["#3B8C6E" if value >= 0.5 else "#B24C4A" for value in rates])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("固定种子胜出率")
    ax.set_title("跨区域加权种子胜出率", loc="left", fontweight="bold")
    for bar, value in zip(bars, rates, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path2 = figure_dir / "cross_area_seed_win_rate.png"
    fig.savefig(path2, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return [path1.relative_to(output_dir).as_posix(), path2.relative_to(output_dir).as_posix()]


def _write_tables(summary: CrossAreaSeededSummary, output_dir: Path) -> None:
    area_fields = [
        "area",
        "metric",
        "metric_label",
        "n",
        "mean_delta",
        "std_delta",
        "min_delta",
        "max_delta",
        "n_better",
        "better_rate",
        "higher_is_better",
        "mean_benefit",
        "area_better",
        "all_seed_better",
        "all_seed_worse",
        "report_dir",
    ]
    metric_fields = [
        "metric",
        "metric_label",
        "n_areas",
        "n_seed_pairs",
        "mean_area_delta",
        "median_area_delta",
        "mean_area_benefit",
        "median_area_benefit",
        "area_better_count",
        "area_all_seed_better_count",
        "area_all_seed_worse_count",
        "weighted_seed_better_rate",
        "higher_is_better",
    ]
    _write_csv(output_dir / "area_delta_summary.csv", summary.area_delta_rows, area_fields)
    _write_csv(output_dir / "cross_area_metric_summary.csv", summary.metric_rows, metric_fields)


def write_cross_area_report(
    summary: CrossAreaSeededSummary,
    output_dir: Path,
    same_grid_report_dirs: list[Path] | None = None,
    create_figures: bool = True,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    same_grid_dirs = [Path(path) for path in (same_grid_report_dirs or [])]
    _write_tables(summary, output)
    _write_manifest(summary, output, same_grid_dirs)
    figure_links = _make_figures(summary, output) if create_figures else []

    all_area_count = max((int(row["n_areas"]) for row in summary.metric_rows), default=0)
    all_area_wins = [row for row in summary.metric_rows if int(row["area_better_count"]) == int(row["n_areas"])]
    clear_losses = [row for row in summary.metric_rows if int(row["area_better_count"]) == 0]
    mixed = [row for row in summary.metric_rows if row not in all_area_wins and row not in clear_losses]

    lines = [
        "# Paper58-change-gate 与 GeoSOS-FLUS 跨区域稳健性报告",
        "",
        "## 实验对象",
        "",
        f"- Paper58 方法：`{summary.challenger}`，即 Paper58 latent-dynamics 输出后接非 oracle 的变化置信度与邻域门控。",
        f"- 对照方法：`{summary.baseline}`，即 `/Users/zhouning/FLUS_console_crossplatform` 编译出的 GeoSOS-FLUS 同源 CLI。",
        "- 数据：真实同网格 AlphaEarth/ESRI 2020→2021 输入，外加 GeoSOS-FLUS 风格 Dynamic World 栅格样本区域。",
        "- 随机性处理：GeoSOS-FLUS 每个区域固定 30 个随机种子重复运行；Paper58 输出为确定性结果。",
        "",
    ]
    if figure_links:
        lines.extend(
            [
                "## 汇总图",
                "",
                f"![跨区域指标收益]({figure_links[0]})",
                "",
                "上图中正值表示 Paper58-change-gate 优于 GeoSOS-FLUS；对“分配分歧”已自动按越低越好转换为收益。",
                "",
                f"![跨区域加权种子胜出率]({figure_links[1]})",
                "",
            ]
        )

    lines.extend(
        [
            "## 跨区域指标汇总",
            "",
            "| 指标 | 区域数 | 种子对数 | 平均差值 | 平均收益 | 区域胜出数 | 全种子胜出区域数 | 全种子落后区域数 | 加权种子胜出率 | 指标方向 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary.metric_rows:
        direction = "越高越好" if bool(row["higher_is_better"]) else "越低越好"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["metric_label"]),
                    str(row["n_areas"]),
                    str(row["n_seed_pairs"]),
                    _fmt(row["mean_area_delta"]),
                    _fmt(row["mean_area_benefit"]),
                    f"{row['area_better_count']}/{row['n_areas']}",
                    f"{row['area_all_seed_better_count']}/{row['n_areas']}",
                    f"{row['area_all_seed_worse_count']}/{row['n_areas']}",
                    _fmt(row["weighted_seed_better_rate"]),
                    direction,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 区域级差异",
            "",
            "| 区域 | 指标 | 平均差值 | 平均收益 | 胜出次数 | 胜出率 | 判读 |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(summary.area_delta_rows, key=lambda item: (str(item["area"]), str(item["metric"]))):
        verdict = "Paper58 更好" if bool(row["area_better"]) else "GeoSOS-FLUS 更好或持平"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['area']}`",
                    str(row["metric_label"]),
                    _fmt(row["mean_delta"]),
                    _fmt(row["mean_benefit"]),
                    f"{row['n_better']}/{row['n']}",
                    _fmt(row["better_rate"]),
                    verdict,
                ]
            )
            + " |"
        )

    if same_grid_dirs:
        lines.extend(
            [
                "",
                "## 图文报告入口",
                "",
                "下面这些单区域报告包含起始真实图、终期真实图、Paper58 预测、Paper58-change-gate 预测、GeoSOS-FLUS 预测以及变化误差图。图中 `error` 指“变化误差图”，不是程序错误；通常表示命中、漏报和虚报的空间位置。",
                "",
                "| 报告目录 | README | HTML | 主要空间图 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for report_dir in same_grid_dirs:
            readme = Path(report_dir) / "README.md"
            html = Path(report_dir) / "report.html"
            pngs = sorted((Path(report_dir) / "figures").glob("*.png"))
            fig = pngs[0] if pngs else Path(report_dir) / "figures"
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{Path(report_dir).name}`",
                        f"[README]({_relative_link(output, readme)})" if readme.exists() else "NA",
                        f"[HTML]({_relative_link(output, html)})" if html.exists() else "NA",
                        f"[PNG]({_relative_link(output, fig)})" if fig.exists() else "NA",
                    ]
                )
                + " |"
            )

    win_labels = "、".join(str(row["metric_label"]) for row in all_area_wins) if all_area_wins else "无"
    mixed_labels = "、".join(str(row["metric_label"]) for row in mixed) if mixed else "无"
    loss_labels = "、".join(str(row["metric_label"]) for row in clear_losses) if clear_losses else "无"
    majority_wins = [row for row in summary.metric_rows if int(row["area_better_count"]) > int(row["n_areas"]) / 2]
    majority_labels = "、".join(str(row["metric_label"]) for row in majority_wins) if majority_wins else "无"
    allocation_row = next((row for row in summary.metric_rows if row["metric"] == "allocation_disagreement"), None)
    change_row = next((row for row in summary.metric_rows if row["metric"] == "change_f1"), None)
    transition_row = next((row for row in summary.metric_rows if row["metric"] == "transition_accuracy"), None)
    if allocation_row is None:
        allocation_sentence = "- 分配分歧未进入本轮汇总，不能判断空间分配误差是否改善。"
        allocation_user_sentence = "- 对实际用户而言，当前报告没有覆盖空间分配分歧，因此不能据此判断规划图斑是否更稳。"
    else:
        allocation_sentence = (
            f"- 分配分歧的区域胜出数为 {allocation_row['area_better_count']}/{allocation_row['n_areas']}，"
            f"平均收益为 {_fmt(allocation_row['mean_area_benefit'])}；该项仍是是否“全面超过”的关键瓶颈。"
        )
        allocation_user_sentence = (
            f"- 如果用户更关心“少错改、不乱扩张”，需要优先看分配分歧：本轮 Paper58 在 "
            f"{allocation_row['area_better_count']}/{allocation_row['n_areas']} 个区域优于 GeoSOS-FLUS，"
            "因此还不能作为无条件优势来宣传。"
        )
    if change_row is None:
        change_user_sentence = "- 当前报告没有变化 F1，不能判断真实变化斑块的捕捉能力。"
    else:
        change_user_sentence = (
            f"- 如果用户更关心“尽量抓住真实变化斑块”，变化 F1 在 "
            f"{change_row['area_better_count']}/{change_row['n_areas']} 个区域胜出，"
            f"加权种子胜出率 {_fmt(change_row['weighted_seed_better_rate'])}；失败区域需要单独排查漏判。"
        )
    if transition_row is None:
        transition_sentence = "- 当前报告没有转换准确率，不能判断变化后的目标类别是否更准。"
    else:
        transition_sentence = (
            f"- 转换准确率的区域胜出数为 {transition_row['area_better_count']}/{transition_row['n_areas']}，"
            f"加权种子胜出率 {_fmt(transition_row['weighted_seed_better_rate'])}；这反映 Paper58 的语义转移优势是否稳定。"
        )
    lines.extend(
        [
            "",
            "## 跨区域稳健性结论",
            "",
            f"- 在 {all_area_count} 个真实区域中，所有区域都胜出的指标是：{win_labels}。",
            f"- 多数区域胜出的指标是：{majority_labels}。",
            f"- 区域间表现分化的指标是：{mixed_labels}。",
            f"- 所有区域都没有胜出的指标是：{loss_labels}。",
            "- 因此，本轮跨区域证据不能表述为 Paper58 已经在所有指标上完全超过 GeoSOS-FLUS。",
            f"- 更严谨的表述是：Paper58 当前稳定优势集中在 {win_labels}；{mixed_labels} 仍受区域差异影响。",
            allocation_sentence,
            transition_sentence,
            "",
            "## 对实际用户的含义",
            "",
            allocation_user_sentence,
            change_user_sentence,
            "- 对业务决策来说，这意味着 Paper58 已能在部分核心精度上超过 GeoSOS-FLUS，但仍需要把失败区域作为风险提示，而不是只展示平均值。",
            "",
            "## 下一轮算法优化重点",
            "",
            "1. 将固定 `keep_fraction` 改为按区域变化强度、类别转移概率和不确定性自适应的门控。",
            "2. 增加类别级阈值，而不是对所有转移使用同一个变化保留比例。",
            "3. 在门控后加入轻量空间分配校正，使 FoM 和 transition accuracy 不被过度保守策略拉低。",
            "4. 用 Dongguan/Kunshan/Caidian 作开发集、Anxin 作失败验证集，避免继续过拟合单一区域。",
            "",
            "## 文件",
            "",
            "- `cross_area_metric_summary.csv`：跨区域指标汇总。",
            "- `area_delta_summary.csv`：每个区域、每个指标的差值和胜率。",
            "- `summary.json`：机器可读运行清单。",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize seeded Paper58 vs GeoSOS-FLUS results across areas.")
    parser.add_argument("--seeded-report-dir", action="append", type=Path, required=True)
    parser.add_argument("--same-grid-report-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--challenger", default="paper58_change_gate")
    parser.add_argument("--baseline", default="geosos_flus_console")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    summary = summarize_cross_area(
        seeded_report_dirs=args.seeded_report_dir,
        challenger=args.challenger,
        baseline=args.baseline,
    )
    write_cross_area_report(
        summary,
        output_dir=args.output_dir,
        same_grid_report_dirs=args.same_grid_report_dir,
        create_figures=not args.no_figures,
    )
    print(
        "Cross-area seeded FLUS report complete: "
        f"areas={len(set(row['area'] for row in summary.area_delta_rows))}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
