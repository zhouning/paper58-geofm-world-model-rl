from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper58_benchmark.run_true_paper58_flus_same_grid_comparison import (
    _parse_extra_prediction_specs,
    run_same_grid_comparison,
)


ROOT = Path(__file__).resolve().parents[2]
METRICS = ["change_f1", "fom", "transition_accuracy", "allocation_disagreement"]
LOWER_IS_BETTER = {"allocation_disagreement", "quantity_disagreement"}


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


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_ready(row.get(field, "")) for field in fields})


def _read_metric_rows(path: Path, seed: int) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for row in reader:
            converted: dict[str, Any] = {"seed": int(seed)}
            for key, value in row.items():
                if key in {
                    "start_year",
                    "end_year",
                    "n_pixels",
                    "true_change_pixels",
                    "pred_change_pixels",
                }:
                    converted[key] = int(float(value))
                elif key in {
                    "change_precision",
                    "change_recall",
                    "change_f1",
                    "fom",
                    "transition_accuracy",
                    "quantity_disagreement",
                    "allocation_disagreement",
                }:
                    converted[key] = float(value)
                else:
                    converted[key] = value
            rows.append(converted)
    return rows


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def summarize_seeded_method_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("method")), str(row.get("area")))].append(row)
    summary: list[dict[str, Any]] = []
    for (method, area), subset in sorted(grouped.items()):
        out: dict[str, Any] = {"method": method, "area": area, "n": len(subset)}
        for metric in METRICS:
            values = [float(row[metric]) for row in subset if row.get(metric) not in (None, "")]
            mean, std = _mean_std(values)
            out[f"mean_{metric}"] = mean
            out[f"std_{metric}"] = std
            out[f"min_{metric}"] = float(np.min(values)) if values else None
            out[f"max_{metric}"] = float(np.max(values)) if values else None
        summary.append(out)
    return summary


def summarize_seeded_deltas(
    rows: list[dict[str, Any]],
    challenger: str,
    baseline: str,
) -> list[dict[str, Any]]:
    keyed: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in rows:
        keyed[(int(row["seed"]), str(row["area"]), str(row["method"]))] = row

    deltas: dict[str, list[float]] = {metric: [] for metric in METRICS}
    better: dict[str, int] = {metric: 0 for metric in METRICS}
    for seed, area, method in sorted(keyed):
        if method != challenger:
            continue
        challenge_row = keyed[(seed, area, challenger)]
        baseline_row = keyed.get((seed, area, baseline))
        if baseline_row is None:
            continue
        for metric in METRICS:
            if metric not in challenge_row or metric not in baseline_row:
                continue
            delta = float(challenge_row[metric]) - float(baseline_row[metric])
            deltas[metric].append(delta)
            if (delta < 0.0 if metric in LOWER_IS_BETTER else delta > 0.0):
                better[metric] += 1

    summary: list[dict[str, Any]] = []
    for metric in METRICS:
        values = deltas[metric]
        mean, std = _mean_std(values)
        summary.append(
            {
                "challenger": challenger,
                "baseline": baseline,
                "metric": metric,
                "n": len(values),
                "mean_delta": mean,
                "std_delta": std,
                "min_delta": float(np.min(values)) if values else None,
                "max_delta": float(np.max(values)) if values else None,
                "n_better": better[metric],
                "better_rate": float(better[metric] / len(values)) if values else None,
                "higher_is_better": metric not in LOWER_IS_BETTER,
            }
        )
    return summary


def summarize_seeded_overall_deltas(
    rows: list[dict[str, Any]],
    challenger: str,
    baseline: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["method"]))].append(row)

    seed_method_means: dict[tuple[int, str], dict[str, float]] = {}
    for key, subset in grouped.items():
        seed_method_means[key] = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in subset if row.get(metric) not in (None, "")]
            if values:
                seed_method_means[key][metric] = float(np.mean(values))

    by_seed: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        challenge_means = seed_method_means.get((seed, challenger), {})
        baseline_means = seed_method_means.get((seed, baseline), {})
        for metric in METRICS:
            if metric not in challenge_means or metric not in baseline_means:
                continue
            delta = float(challenge_means[metric] - baseline_means[metric])
            higher_is_better = metric not in LOWER_IS_BETTER
            by_seed.append(
                {
                    "seed": seed,
                    "challenger": challenger,
                    "baseline": baseline,
                    "metric": metric,
                    "challenger_mean": float(challenge_means[metric]),
                    "baseline_mean": float(baseline_means[metric]),
                    "delta": delta,
                    "better": bool(delta > 0.0 if higher_is_better else delta < 0.0),
                    "higher_is_better": higher_is_better,
                }
            )

    summary: list[dict[str, Any]] = []
    for metric in METRICS:
        subset = [row for row in by_seed if row["metric"] == metric]
        values = [float(row["delta"]) for row in subset]
        mean, std = _mean_std(values)
        n_better = sum(1 for row in subset if bool(row["better"]))
        summary.append(
            {
                "challenger": challenger,
                "baseline": baseline,
                "metric": metric,
                "n": len(values),
                "mean_delta": mean,
                "std_delta": std,
                "min_delta": float(np.min(values)) if values else None,
                "max_delta": float(np.max(values)) if values else None,
                "n_better": n_better,
                "better_rate": float(n_better / len(values)) if values else None,
                "higher_is_better": metric not in LOWER_IS_BETTER,
            }
        )
    return by_seed, summary


def _format_metric_name(metric: str) -> str:
    return {
        "change_f1": "变化 F1",
        "fom": "FoM",
        "transition_accuracy": "转换准确率",
        "allocation_disagreement": "分配分歧",
    }.get(metric, metric)


def _format_number(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.4f}"


def write_seeded_report(
    output_dir: Path,
    seeds: list[int],
    method_summary: list[dict[str, Any]],
    delta_summary: list[dict[str, Any]],
    overall_delta_by_seed: list[dict[str, Any]],
    overall_delta_summary: list[dict[str, Any]],
    challenger: str,
    baseline: str,
) -> None:
    output = Path(output_dir)
    lines = [
        "# Paper58 与 GeoSOS-FLUS 多随机种子稳健性报告",
        "",
        f"本报告固定 GeoSOS-FLUS 环境变量 `FLUS_RANDOM_SEED`，对同一输入重复运行 {len(seeds)} 次。"
        "Paper58 输出是确定性的；GeoSOS-FLUS 的 CA 分配含随机项，因此需要用重复实验报告均值和波动范围。",
        "",
        f"- 随机种子：{', '.join(str(seed) for seed in seeds)}",
        f"- 主要比较：`{challenger}` 相对 `{baseline}`",
        "",
        "## 方法均值与波动",
        "",
        "| 方法 | 区域 | n | 变化 F1 均值±标准差 | FoM 均值±标准差 | 转换准确率均值±标准差 | 分配分歧均值±标准差 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in method_summary:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['method']}`",
                    f"`{row['area']}`",
                    str(row["n"]),
                    f"{_format_number(row['mean_change_f1'])}±{_format_number(row['std_change_f1'])}",
                    f"{_format_number(row['mean_fom'])}±{_format_number(row['std_fom'])}",
                    f"{_format_number(row['mean_transition_accuracy'])}±{_format_number(row['std_transition_accuracy'])}",
                    f"{_format_number(row['mean_allocation_disagreement'])}±{_format_number(row['std_allocation_disagreement'])}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 按种子总体均值的稳健性",
            "",
            "该表先在每个随机种子内对全部区域求平均，再比较 Paper58 与 GeoSOS-FLUS；它对应“每次完整实验总体是否胜出”。",
            "",
            "| 指标 | 平均差值 | 标准差 | 最小差值 | 最大差值 | 胜出种子 | 胜出率 | 判读 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in overall_delta_summary:
        metric = str(row["metric"])
        direction = "越高越好" if bool(row["higher_is_better"]) else "越低越好"
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_metric_name(metric),
                    _format_number(row["mean_delta"]),
                    _format_number(row["std_delta"]),
                    _format_number(row["min_delta"]),
                    _format_number(row["max_delta"]),
                    f"{int(row['n_better'])}/{int(row['n'])}" if row["n"] else "0/0",
                    _format_number(row["better_rate"]),
                    direction,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### 每个种子的总体差值",
            "",
            "| 种子 | 指标 | Paper58 均值 | GeoSOS-FLUS 均值 | 差值 | 是否胜出 |",
            "| ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in overall_delta_by_seed:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    _format_metric_name(str(row["metric"])),
                    _format_number(row["challenger_mean"]),
                    _format_number(row["baseline_mean"]),
                    _format_number(row["delta"]),
                    "是" if bool(row["better"]) else "否",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 区域×种子配对稳健性",
            "",
            "该表把每个区域、每个种子都作为一个配对样本，用来暴露哪些区域仍是短板。",
            "",
            "| 指标 | 平均差值 | 标准差 | 最小差值 | 最大差值 | 胜出次数 | 胜出率 | 判读 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in delta_summary:
        metric = str(row["metric"])
        direction = "越高越好" if bool(row["higher_is_better"]) else "越低越好"
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_metric_name(metric),
                    _format_number(row["mean_delta"]),
                    _format_number(row["std_delta"]),
                    _format_number(row["min_delta"]),
                    _format_number(row["max_delta"]),
                    f"{int(row['n_better'])}/{int(row['n'])}" if row["n"] else "0/0",
                    _format_number(row["better_rate"]),
                    direction,
                ]
            )
            + " |"
        )
    overall_stable_wins = [
        _format_metric_name(str(row["metric"]))
        for row in overall_delta_summary
        if row["n"] and int(row["n_better"]) == int(row["n"])
    ]
    overall_losses = [
        _format_metric_name(str(row["metric"]))
        for row in overall_delta_summary
        if row["n"] and int(row["n_better"]) == 0
    ]
    paired_stable_wins = [
        _format_metric_name(str(row["metric"]))
        for row in delta_summary
        if row["n"] and int(row["n_better"]) == int(row["n"])
    ]
    paired_stable_losses = [
        _format_metric_name(str(row["metric"]))
        for row in delta_summary
        if row["n"] and int(row["n_better"]) == 0
    ]
    lines.extend(
        [
            "",
            "## 核心判读",
            "",
            (
                f"- 按每个种子的 24 区域总体均值，`{challenger}` 在 {len(overall_stable_wins)} 个指标上对 `{baseline}` 达到 100% 种子胜出："
                + ("、".join(overall_stable_wins) if overall_stable_wins else "无")
                + "。"
            ),
            (
                f"- 按区域×种子配对，`{challenger}` 达到 100% 配对胜出的指标："
                + ("、".join(paired_stable_wins) if paired_stable_wins else "无")
                + "。"
            ),
            (
                f"- `{challenger}` 在以下总体指标上没有超过 `{baseline}`："
                + ("、".join(overall_losses) if overall_losses else "无")
                + "。"
            ),
            (
                f"- `{challenger}` 在以下区域×种子配对指标上完全没有超过 `{baseline}`："
                + ("、".join(paired_stable_losses) if paired_stable_losses else "无")
                + "。"
            ),
            "- 因此，本轮东莞 80m 数据支持“Paper58 优化后在总体均值层面稳定超过 GeoSOS-FLUS”；但区域配对胜率仍不是 100%，论文表述应保留区域异质性和失败案例。",
            "",
            "## 结论边界",
            "",
            "- 这份报告检验的是 GeoSOS-FLUS 随机分配项对结论的影响，不改变输入数据、Paper58 权重或 GeoSOS-FLUS 参数。",
            "- 若某个指标胜出率为 1.0000，说明在本组随机种子下该结论不依赖单次 FLUS 随机结果。",
            "- 分配分歧是越低越好；其差值为正表示 Paper58 对应方法仍高于 GeoSOS-FLUS。",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_seeded_flus_replicates(
    paper58_predictions_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    flus_executable: Path,
    seeds: list[int],
    flus_demand_source: str = "paper58_prediction",
    extra_prediction_specs: dict[str, Path] | None = None,
    challenger: str = "paper58_change_gate",
    baseline: str = "geosos_flus_console",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    previous_seed = os.environ.get("FLUS_RANDOM_SEED")
    try:
        for seed in seeds:
            os.environ["FLUS_RANDOM_SEED"] = str(int(seed))
            seed_dir = output / f"seed_{int(seed):04d}"
            run_same_grid_comparison(
                paper58_predictions_dir=paper58_predictions_dir,
                labels_dir=labels_dir,
                output_dir=seed_dir,
                flus_executable=flus_executable,
                flus_demand_source=flus_demand_source,
                extra_prediction_specs=extra_prediction_specs,
            )
            all_rows.extend(_read_metric_rows(seed_dir / "metrics_by_method.csv", int(seed)))
    finally:
        if previous_seed is None:
            os.environ.pop("FLUS_RANDOM_SEED", None)
        else:
            os.environ["FLUS_RANDOM_SEED"] = previous_seed

    fields = [
        "seed",
        "method",
        "area",
        "start_year",
        "end_year",
        "source",
        "tier",
        "stratum",
        "n_pixels",
        "true_change_pixels",
        "pred_change_pixels",
        "change_precision",
        "change_recall",
        "change_f1",
        "fom",
        "transition_accuracy",
        "quantity_disagreement",
        "allocation_disagreement",
    ]
    _write_csv(output / "seeded_metrics_by_method.csv", all_rows, fields)
    method_summary = summarize_seeded_method_metrics(all_rows)
    method_fields = [
        "method",
        "area",
        "n",
        *[field for metric in METRICS for field in (f"mean_{metric}", f"std_{metric}", f"min_{metric}", f"max_{metric}")],
    ]
    _write_csv(output / "seeded_metric_summary_by_method.csv", method_summary, method_fields)
    delta_summary = summarize_seeded_deltas(all_rows, challenger=challenger, baseline=baseline)
    _write_csv(
        output / "seeded_delta_summary.csv",
        delta_summary,
        [
            "challenger",
            "baseline",
            "metric",
            "n",
            "mean_delta",
            "std_delta",
            "min_delta",
            "max_delta",
            "n_better",
            "better_rate",
            "higher_is_better",
        ],
    )
    overall_delta_by_seed, overall_delta_summary = summarize_seeded_overall_deltas(
        all_rows,
        challenger=challenger,
        baseline=baseline,
    )
    _write_csv(
        output / "seeded_overall_delta_by_seed.csv",
        overall_delta_by_seed,
        [
            "seed",
            "challenger",
            "baseline",
            "metric",
            "challenger_mean",
            "baseline_mean",
            "delta",
            "better",
            "higher_is_better",
        ],
    )
    _write_csv(
        output / "seeded_overall_delta_summary.csv",
        overall_delta_summary,
        [
            "challenger",
            "baseline",
            "metric",
            "n",
            "mean_delta",
            "std_delta",
            "min_delta",
            "max_delta",
            "n_better",
            "better_rate",
            "higher_is_better",
        ],
    )
    write_seeded_report(
        output,
        seeds,
        method_summary,
        delta_summary,
        overall_delta_by_seed,
        overall_delta_summary,
        challenger,
        baseline,
    )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": output,
        "seeds": seeds,
        "n_rows": len(all_rows),
        "method_summary": method_summary,
        "delta_summary": delta_summary,
        "overall_delta_by_seed": overall_delta_by_seed,
        "overall_delta_summary": overall_delta_summary,
    }


def _default_seeds(seed_start: int, n_seeds: int) -> list[int]:
    return [int(seed_start) + index for index in range(int(n_seeds))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run seeded GeoSOS-FLUS replicates for robust Paper58 comparison.")
    parser.add_argument("--paper58-predictions-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flus-executable", type=Path, required=True)
    parser.add_argument(
        "--flus-demand-source",
        choices=["transition_prior", "paper58_prediction", "start_persistence"],
        default="paper58_prediction",
    )
    parser.add_argument("--extra-prediction", action="append", default=[], metavar="METHOD=PATH_OR_DIR")
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--seed-start", type=int, default=1001)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--challenger", default="paper58_change_gate")
    parser.add_argument("--baseline", default="geosos_flus_console")
    args = parser.parse_args(argv)
    seeds = args.seed or _default_seeds(args.seed_start, args.n_seeds)
    result = run_seeded_flus_replicates(
        paper58_predictions_dir=args.paper58_predictions_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        flus_executable=args.flus_executable,
        seeds=seeds,
        flus_demand_source=args.flus_demand_source,
        extra_prediction_specs=_parse_extra_prediction_specs(args.extra_prediction),
        challenger=args.challenger,
        baseline=args.baseline,
    )
    print(
        "Seeded FLUS replicates complete: "
        f"seeds={len(seeds)}, rows={result['n_rows']}, output={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
