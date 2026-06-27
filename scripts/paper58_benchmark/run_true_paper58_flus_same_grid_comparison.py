from __future__ import annotations

import argparse
import csv
import html
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from scripts.paper58_benchmark.flus import load_flus_prediction
from scripts.paper58_benchmark.flus_case import decode_flus_geotiff, find_flus_simulation_result, write_flus_case
from scripts.paper58_benchmark.las_demand import derive_demand
from scripts.paper58_benchmark.las_metrics import method_metric_row
from scripts.paper58_benchmark.las_suitability import class_values_from_maps, one_hot_probability_cube, transition_prior_from_pairs


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_DIR = ROOT / "data" / "independent_change_labels" / "labels"
DEFAULT_PAPER58_PREDICTIONS_DIR = ROOT / "paper" / "rse_submission_paper58" / "paper58_true_repo_weights_subset_2026-06-27"
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "rse_submission_paper58" / "paper58_true_geosos_flus_same_grid_2026-06-27"
PREDICTION_RE = re.compile(r"^(?P<area>.+)_lulc_pred_(?P<start_year>\d{4})_(?P<end_year>\d{4})\.npy$")
ConsoleRunner = Callable[[Path], None]
METRIC_FIELDS = [
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
CLASS_COLORS = {
    0: "#FFFFFF",
    1: "#419BDF",
    2: "#397D49",
    4: "#7A87C6",
    5: "#E49635",
    7: "#DFC35A",
    8: "#C4281B",
    9: "#A59B8F",
    10: "#B39FE1",
    11: "#E3E2C3",
}
plt.rcParams.update(
    {
        "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)


@dataclass
class SameGridSample:
    area: str
    start_year: int
    end_year: int
    start: np.ndarray
    end: np.ndarray
    paper58_prediction: np.ndarray
    valid_mask: np.ndarray
    source: str = "AlphaEarth/ESRI same-grid true Paper58 latent-dynamics"
    prediction_path: Path | None = None


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_ready(row.get(field, "")) for field in fields})


def _parse_prediction(path: Path) -> tuple[str, int, int] | None:
    match = PREDICTION_RE.match(path.name)
    if match is None:
        return None
    return match.group("area"), int(match.group("start_year")), int(match.group("end_year"))


def discover_same_grid_samples(
    paper58_predictions_dir: Path = DEFAULT_PAPER58_PREDICTIONS_DIR,
    labels_dir: Path = DEFAULT_LABELS_DIR,
) -> list[SameGridSample]:
    samples: list[SameGridSample] = []
    prediction_root = Path(paper58_predictions_dir)
    label_root = Path(labels_dir)
    for prediction_path in sorted(prediction_root.glob("*_lulc_pred_*_*.npy")):
        parsed = _parse_prediction(prediction_path)
        if parsed is None:
            continue
        area, start_year, end_year = parsed
        start_path = label_root / f"{area}_lulc_{start_year}.npy"
        end_path = label_root / f"{area}_lulc_{end_year}.npy"
        if not start_path.exists() or not end_path.exists():
            continue
        start = np.load(start_path).astype(np.int32, copy=False)
        end = np.load(end_path).astype(np.int32, copy=False)
        prediction = np.load(prediction_path).astype(np.int32, copy=False)
        if start.shape != end.shape or start.shape != prediction.shape:
            raise ValueError(
                f"shape mismatch for {area} {start_year}-{end_year}: "
                f"start={start.shape}, end={end.shape}, pred={prediction.shape}"
            )
        valid_mask = (start != 0) & (end != 0) & (prediction != 0)
        samples.append(
            SameGridSample(
                area=area,
                start_year=start_year,
                end_year=end_year,
                start=start,
                end=end,
                paper58_prediction=prediction,
                valid_mask=valid_mask,
                prediction_path=prediction_path,
            )
        )
    return samples


def _valid_values(sample: SameGridSample, array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.shape != sample.start.shape:
        raise ValueError(f"shape mismatch for {sample.area}: {arr.shape} vs {sample.start.shape}")
    return arr[np.asarray(sample.valid_mask, dtype=bool)]


def _valid_training_pair(sample: SameGridSample) -> tuple[np.ndarray, np.ndarray]:
    return np.where(sample.valid_mask, sample.start, 0), np.where(sample.valid_mask, sample.end, 0)


def training_pairs_for_sample(sample: SameGridSample, all_samples: list[SameGridSample]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [_valid_training_pair(other) for other in all_samples if other.area != sample.area]


def evaluate_predictions(
    sample: SameGridSample,
    flus_prediction: np.ndarray | None = None,
    extra_predictions: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = _valid_values(sample, sample.start)
    end = _valid_values(sample, sample.end)
    paper58 = _valid_values(sample, sample.paper58_prediction)
    base = method_metric_row("paper58_latent_dynamics", sample.area, "same_grid", sample.source, start, end, paper58)
    rows.append({**base, "start_year": sample.start_year, "end_year": sample.end_year, "source": sample.source})
    for method, prediction in sorted((extra_predictions or {}).items()):
        extra = _valid_values(sample, prediction)
        extra_row = method_metric_row(str(method), sample.area, "same_grid", sample.source, start, end, extra)
        rows.append({**extra_row, "start_year": sample.start_year, "end_year": sample.end_year, "source": sample.source})
    if flus_prediction is not None:
        flus = _valid_values(sample, flus_prediction)
        flus_row = method_metric_row("geosos_flus_console", sample.area, "same_grid", sample.source, start, end, flus)
        rows.append({**flus_row, "start_year": sample.start_year, "end_year": sample.end_year, "source": sample.source})
    return rows


def run_flus_console_for_sample(
    sample: SameGridSample,
    training_pairs: list[tuple[np.ndarray, np.ndarray]],
    case_root: Path,
    prediction_dir: Path,
    flus_executable: Path,
    console_runner: ConsoleRunner | None = None,
    demand_source: str = "transition_prior",
) -> Path:
    classes = class_values_from_maps(sample.start, sample.end, sample.paper58_prediction)
    probability = one_hot_probability_cube(sample.paper58_prediction, classes, confidence=0.95, floor=0.01)
    prior = transition_prior_from_pairs(training_pairs, classes)
    effective_demand_source = "paper58_prediction" if demand_source == "transition_prior" and not training_pairs else demand_source
    demand = derive_demand(
        sample.start,
        sample.end,
        sample.paper58_prediction,
        demand_source=effective_demand_source,
        class_values=classes,
        transition_prior=prior,
    )
    case_dir = Path(case_root) / f"{sample.area}_{sample.start_year}_{sample.end_year}"
    write_flus_case(
        output_dir=case_dir,
        start_map=sample.start,
        probability_cube=probability,
        class_values=classes,
        future_demand=demand,
        end_year=sample.end_year,
        restrict_mask=sample.valid_mask.astype(np.uint8),
    )
    if console_runner is None:
        subprocess.run([str(flus_executable)], cwd=case_dir, check=True)
    else:
        console_runner(case_dir)
    encoded = find_flus_simulation_result(case_dir, sample.end_year)
    decoded = Path(prediction_dir) / f"{sample.area}_{sample.start_year}_{sample.end_year}_flus.tif"
    return decode_flus_geotiff(encoded, decoded, classes)


def _indexed(array: np.ndarray, class_to_index: dict[int, int]) -> np.ndarray:
    out = np.zeros(np.asarray(array).shape, dtype=np.int16)
    for cls, index in class_to_index.items():
        out[np.asarray(array) == cls] = index
    return out


def _class_cmap(arrays: list[np.ndarray]) -> tuple[ListedColormap, dict[int, int], list[Patch]]:
    classes = sorted({int(value) for array in arrays for value in np.unique(np.asarray(array))})
    class_to_index = {cls: index for index, cls in enumerate(classes)}
    colors = [CLASS_COLORS.get(cls, "#D0D0D0") for cls in classes]
    patches = [Patch(color=color, label=f"类别 {cls}" if cls else "边界外") for cls, color in zip(classes, colors, strict=False)]
    return ListedColormap(colors), class_to_index, patches


def _error_map(sample: SameGridSample, prediction: np.ndarray) -> np.ndarray:
    true_change = sample.end != sample.start
    pred_change = np.asarray(prediction) != sample.start
    valid = np.asarray(sample.valid_mask, dtype=bool)
    error = np.full(sample.start.shape, 4, dtype=np.int8)
    error[valid] = 0
    error[valid & true_change & pred_change] = 1
    error[valid & true_change & ~pred_change] = 2
    error[valid & ~true_change & pred_change] = 3
    return error


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _save_spatial_panel(
    output_dir: Path,
    sample: SameGridSample,
    flus_prediction: np.ndarray | None,
    extra_predictions: dict[str, np.ndarray] | None = None,
) -> str:
    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    map_panels = [
        ("起始年份 ESRI 真值", sample.start),
        ("结束年份 ESRI 真值", sample.end),
        ("完整 Paper58 模型", sample.paper58_prediction),
    ]
    for method, prediction in sorted((extra_predictions or {}).items()):
        map_panels.append((str(method), prediction))
    if flus_prediction is not None:
        map_panels.append(("GeoSOS-FLUS 控制台", flus_prediction))
    error_panels = [
        ("真实变化位置", (sample.end != sample.start).astype(np.int8)),
        ("Paper58 变化误差", _error_map(sample, sample.paper58_prediction)),
    ]
    for method, prediction in sorted((extra_predictions or {}).items()):
        error_panels.append((f"{method} 变化误差", _error_map(sample, prediction)))
    if flus_prediction is not None:
        error_panels.append(("GeoSOS-FLUS 变化误差", _error_map(sample, flus_prediction)))

    cmap, class_to_index, class_handles = _class_cmap([array for _, array in map_panels])
    ncols = max(len(map_panels), len(error_panels))
    fig, axes = plt.subplots(2, ncols, figsize=(max(8.0, ncols * 2.0), 5.0), constrained_layout=True)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    for index, (title, array) in enumerate(map_panels):
        axes[0, index].imshow(_indexed(array, class_to_index), cmap=cmap, interpolation="nearest")
        axes[0, index].set_title(title, loc="left", fontsize=8, fontweight="bold")
        axes[0, index].set_frame_on(True)
    change_cmap = ListedColormap(["#F0F0F0", "#2C7FB8"])
    error_cmap = ListedColormap(["#F0F0F0", "#3B8C6E", "#B24C4A", "#B48A2C", "#FFFFFF"])
    for index, (title, array) in enumerate(error_panels):
        if index == 0:
            axes[1, index].imshow(array, cmap=change_cmap, interpolation="nearest", vmin=0, vmax=1)
        else:
            axes[1, index].imshow(array, cmap=error_cmap, interpolation="nearest", vmin=0, vmax=4)
        axes[1, index].set_title(title, loc="left", fontsize=8, fontweight="bold")
        axes[1, index].set_frame_on(True)
    error_handles = [
        Patch(color="#2C7FB8", label="真实变化"),
        Patch(color="#3B8C6E", label="命中"),
        Patch(color="#B24C4A", label="漏判"),
        Patch(color="#B48A2C", label="误报"),
    ]
    fig.suptitle(f"{sample.area} {sample.start_year}-{sample.end_year}", fontsize=10, fontweight="bold")
    fig.legend(handles=class_handles + error_handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=6, fontsize=7)
    stem = f"same_grid_{_slug(sample.area)}"
    fig.savefig(figure_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return f"figures/{stem}.png"


def _metric_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in rows})
    summary = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        summary.append(
            {
                "method": method,
                "n": len(subset),
                "mean_change_f1": float(np.mean([row["change_f1"] for row in subset])) if subset else None,
                "mean_fom": float(np.mean([row["fom"] for row in subset])) if subset else None,
                "mean_transition_accuracy": float(np.mean([row["transition_accuracy"] for row in subset])) if subset else None,
                "mean_allocation_disagreement": float(np.mean([row["allocation_disagreement"] for row in subset])) if subset else None,
            }
        )
    return summary


def _metric_mean(rows: list[dict[str, Any]], method: str, metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if str(row.get("method")) == method and row.get(metric) not in (None, "")]
    return float(np.mean(values)) if values else None


def _format_delta(left: float | None, right: float | None, higher_is_better: bool = True) -> str:
    if left is None or right is None:
        return "暂无数据"
    delta = left - right
    better = delta > 0 if higher_is_better else delta < 0
    direction = "更优" if better else "更差" if delta != 0 else "持平"
    return f"{delta:+.4f} ({direction})"


def _overall_comparison(rows: list[dict[str, Any]]) -> list[str]:
    paper58_f1 = _metric_mean(rows, "paper58_latent_dynamics", "change_f1")
    flus_f1 = _metric_mean(rows, "geosos_flus_console", "change_f1")
    paper58_fom = _metric_mean(rows, "paper58_latent_dynamics", "fom")
    flus_fom = _metric_mean(rows, "geosos_flus_console", "fom")
    paper58_trans = _metric_mean(rows, "paper58_latent_dynamics", "transition_accuracy")
    flus_trans = _metric_mean(rows, "geosos_flus_console", "transition_accuracy")
    paper58_alloc = _metric_mean(rows, "paper58_latent_dynamics", "allocation_disagreement")
    flus_alloc = _metric_mean(rows, "geosos_flus_console", "allocation_disagreement")
    if flus_f1 is None:
        return ["- 本次只生成了完整 Paper58 latent-dynamics 指标，未生成 GeoSOS-FLUS 匹配结果。"]
    items = [
        (
            "- `变化 F1` 衡量模型是否找到了真实变化像元。"
            f"`paper58_latent_dynamics` 均值={paper58_f1:.4f}，`geosos_flus_console` 均值={flus_f1:.4f}，"
            f"差值={_format_delta(paper58_f1, flus_f1)}。"
        ),
        (
            "- `FoM` 要求变化位置和目标类别同时正确。"
            f"`paper58_latent_dynamics` 均值={paper58_fom:.4f}，`geosos_flus_console` 均值={flus_fom:.4f}，"
            f"差值={_format_delta(paper58_fom, flus_fom)}。"
        ),
        (
            "- `转换准确率` 只在真实变化像元上检查目标类别是否命中。"
            f"`paper58_latent_dynamics` 均值={paper58_trans:.4f}，`geosos_flus_console` 均值={flus_trans:.4f}，"
            f"差值={_format_delta(paper58_trans, flus_trans)}。"
        ),
        (
            "- `分配分歧` 越低越好，表示空间位置和类别分配错配。"
            f"`paper58_latent_dynamics` 均值={paper58_alloc:.4f}，`geosos_flus_console` 均值={flus_alloc:.4f}，"
            f"差值={_format_delta(paper58_alloc, flus_alloc, higher_is_better=False)}。"
        ),
    ]
    extra_methods = sorted(
        {
            str(row["method"])
            for row in rows
            if str(row.get("method")) not in {"paper58_latent_dynamics", "geosos_flus_console"}
        }
    )
    for method in extra_methods:
        method_f1 = _metric_mean(rows, method, "change_f1")
        method_fom = _metric_mean(rows, method, "fom")
        method_trans = _metric_mean(rows, method, "transition_accuracy")
        method_alloc = _metric_mean(rows, method, "allocation_disagreement")
        wins = [
            method_f1 is not None and flus_f1 is not None and method_f1 > flus_f1,
            method_fom is not None and flus_fom is not None and method_fom > flus_fom,
            method_trans is not None and flus_trans is not None and method_trans > flus_trans,
            method_alloc is not None and flus_alloc is not None and method_alloc < flus_alloc,
        ]
        items.append(
            (
                f"- `{method}` 是 Paper58 优化后输出；相对 GeoSOS-FLUS："
                f"变化 F1 差值={_format_delta(method_f1, flus_f1)}，"
                f"FoM 差值={_format_delta(method_fom, flus_fom)}，"
                f"转换准确率差值={_format_delta(method_trans, flus_trans)}，"
                f"分配分歧差值={_format_delta(method_alloc, flus_alloc, higher_is_better=False)}。"
                f"四项指标中有 {sum(bool(win) for win in wins)}/4 项优于 GeoSOS-FLUS。"
            )
        )
    return items


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_没有生成指标行。_"
    fields = ["method", "area", "change_f1", "fom", "transition_accuracy", "allocation_disagreement"]
    lines = ["| 方法 | 区域 | 变化 F1 | FoM | 转换准确率 | 分配分歧 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        values = []
        for field_name in fields:
            if field_name in {"method", "area"}:
                values.append(f"`{row[field_name]}`")
            else:
                values.append(f"{float(row[field_name]):.4f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _html_metric_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>没有生成指标行。</p>"
    fields = ["method", "area", "change_f1", "fom", "transition_accuracy", "allocation_disagreement"]
    headers = ["方法", "区域", "变化 F1", "FoM", "转换准确率", "分配分歧"]
    body = []
    for row in rows:
        cells = []
        for field_name in fields:
            if field_name in {"method", "area"}:
                cells.append(f"<td><code>{html.escape(str(row[field_name]))}</code></td>")
            else:
                cells.append(f"<td>{float(row[field_name]):.4f}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{header}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _plain_markdown_text(value: str) -> str:
    return value.lstrip("- ").replace("`", "").replace("**", "")


def _html_bullets(items: list[str]) -> str:
    return "<ul>\n" + "\n".join(f"<li>{html.escape(_plain_markdown_text(item))}</li>" for item in items) + "\n</ul>"


def _row_by_method(rows: list[dict[str, Any]], area: str) -> dict[str, dict[str, Any]]:
    return {str(row["method"]): row for row in rows if str(row.get("area")) == area}


def _area_interpretation(sample: SameGridSample, rows: list[dict[str, Any]]) -> str:
    by_method = _row_by_method(rows, sample.area)
    paper58 = by_method.get("paper58_latent_dynamics")
    flus = by_method.get("geosos_flus_console")
    if paper58 is None:
        return "该区域没有完整 Paper58 指标。"
    if flus is None:
        return (
            f"真实变化像元 {int(paper58['true_change_pixels']):,} 个。"
            f"完整 Paper58 预测变化像元 {int(paper58['pred_change_pixels']):,} 个，"
            f"变化 F1={float(paper58['change_f1']):.4f}。"
        )
    f1_delta = float(paper58["change_f1"]) - float(flus["change_f1"])
    fom_delta = float(paper58["fom"]) - float(flus["fom"])
    alloc_delta = float(paper58["allocation_disagreement"]) - float(flus["allocation_disagreement"])
    if f1_delta > 0 and fom_delta > 0:
        conclusion = "该区域完整 Paper58 在变化检出和变化命中质量上更强。"
    elif f1_delta < 0 and fom_delta < 0:
        conclusion = "该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。"
    else:
        conclusion = "该区域两个方法各有优势，需要结合误差图判断漏判和误报。"
    text = (
        f"真实变化像元 {int(paper58['true_change_pixels']):,} 个。"
        f"完整 Paper58 预测变化像元 {int(paper58['pred_change_pixels']):,} 个，"
        f"GeoSOS-FLUS 预测变化像元 {int(flus['pred_change_pixels']):,} 个。"
        f"完整 Paper58 的变化 F1 差值 {f1_delta:+.4f}，FoM 差值 {fom_delta:+.4f}；"
        f"分配分歧差值 {alloc_delta:+.4f}（该指标越低越好）。"
        f"{conclusion}"
    )
    for method, extra in sorted(by_method.items()):
        if method in {"paper58_latent_dynamics", "geosos_flus_console"}:
            continue
        f1_extra_delta = float(extra["change_f1"]) - float(flus["change_f1"])
        fom_extra_delta = float(extra["fom"]) - float(flus["fom"])
        trans_extra_delta = float(extra["transition_accuracy"]) - float(flus["transition_accuracy"])
        alloc_extra_delta = float(extra["allocation_disagreement"]) - float(flus["allocation_disagreement"])
        text += (
            f"`{method}` 预测变化像元 {int(extra['pred_change_pixels']):,} 个；"
            f"相对 GeoSOS-FLUS，变化 F1 差值 {f1_extra_delta:+.4f}，"
            f"FoM 差值 {fom_extra_delta:+.4f}，转换准确率差值 {trans_extra_delta:+.4f}，"
            f"分配分歧差值 {alloc_extra_delta:+.4f}（该指标越低越好）。"
        )
    return text


def _class_bias_sentence(sample: SameGridSample, prediction: np.ndarray, label: str) -> str:
    valid = np.asarray(sample.valid_mask, dtype=bool)
    true_end = np.asarray(sample.end)[valid]
    pred = np.asarray(prediction)[valid]
    classes = sorted({int(value) for value in np.unique(true_end)} | {int(value) for value in np.unique(pred)})
    if not classes:
        return f"{label} 没有可统计的有效像元。"
    diffs = []
    for cls in classes:
        diff = int(np.count_nonzero(pred == cls) - np.count_nonzero(true_end == cls))
        diffs.append((cls, diff))
    over_cls, over_diff = max(diffs, key=lambda item: item[1])
    under_cls, under_diff = min(diffs, key=lambda item: item[1])
    return (
        f"{label} 相对结束年份真值最大的高估是类别 {over_cls}（{over_diff:+,} 像元），"
        f"最大的低估是类别 {under_cls}（{under_diff:+,} 像元）。"
    )


def _class_bias_interpretation(
    sample: SameGridSample,
    flus_prediction: np.ndarray | None,
    extra_predictions: dict[str, np.ndarray] | None = None,
) -> str:
    sentences = [_class_bias_sentence(sample, sample.paper58_prediction, "完整 Paper58")]
    for method, prediction in sorted((extra_predictions or {}).items()):
        sentences.append(_class_bias_sentence(sample, prediction, method))
    if flus_prediction is not None:
        sentences.append(_class_bias_sentence(sample, flus_prediction, "GeoSOS-FLUS"))
    return "".join(sentences)


def _extra_predictions_for_area(
    extra_predictions: dict[str, dict[str, np.ndarray]] | None,
    area: str,
) -> dict[str, np.ndarray]:
    return {
        method: predictions[area]
        for method, predictions in (extra_predictions or {}).items()
        if area in predictions
    }


def _parse_extra_prediction_specs(values: list[str] | None) -> dict[str, Path]:
    specs: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"extra prediction must be formatted as METHOD=PATH_OR_DIR, got {value!r}")
        method, path = value.split("=", 1)
        method_key = method.strip()
        if not method_key:
            raise ValueError(f"extra prediction method is empty in {value!r}")
        specs[method_key] = Path(path)
    return specs


def _load_extra_prediction_for_sample(method: str, path_or_dir: Path, sample: SameGridSample) -> np.ndarray:
    path = Path(path_or_dir)
    if path.is_dir():
        candidates = [
            path / f"{sample.area}_lulc_pred_{sample.start_year}_{sample.end_year}.npy",
            path / "predictions" / f"{sample.area}_lulc_pred_{sample.start_year}_{sample.end_year}.npy",
            path / f"{sample.area}_{sample.start_year}_{sample.end_year}_{method}.npy",
            path / "simulated" / f"{sample.area}_{sample.start_year}_{sample.end_year}_{method}.npy",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not path.exists():
        raise FileNotFoundError(f"extra prediction for {method}/{sample.area} not found: {path}")
    prediction = np.load(path).astype(np.int32, copy=False)
    if prediction.shape != sample.start.shape:
        raise ValueError(
            f"extra prediction shape mismatch for {method}/{sample.area}: "
            f"{prediction.shape} vs {sample.start.shape}"
        )
    return prediction


def write_same_grid_report(
    output_dir: Path,
    samples: list[SameGridSample],
    flus_predictions: dict[str, np.ndarray],
    extra_predictions: dict[str, dict[str, np.ndarray]] | None = None,
    metric_rows: list[dict[str, Any]] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = metric_rows or [
        row
        for sample in samples
        for row in evaluate_predictions(
            sample,
            flus_predictions.get(sample.area),
            _extra_predictions_for_area(extra_predictions, sample.area),
        )
    ]
    _write_csv(output / "metrics_by_method.csv", rows, METRIC_FIELDS)
    summary_rows = _metric_summary(rows)
    _write_csv(
        output / "metric_summary_by_method.csv",
        summary_rows,
        ["method", "n", "mean_change_f1", "mean_fom", "mean_transition_accuracy", "mean_allocation_disagreement"],
    )
    spatial = [
        (
            sample.area,
            _save_spatial_panel(
                output,
                sample,
                flus_predictions.get(sample.area),
                _extra_predictions_for_area(extra_predictions, sample.area),
            ),
        )
        for sample in samples
    ]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(samples),
        "n_metric_rows": len(rows),
        "notes": notes or [],
        "metric_summary_by_method": summary_rows,
        "spatial_figures": [{"area": area, "path": path} for area, path in spatial],
    }
    _write_json(output / "summary.json", summary)
    note_lines = [f"- {note}" for note in (notes or [])]
    readme = [
        "# 完整 Paper58 与 GeoSOS-FLUS 同网格严格对比报告",
        "",
        "本报告使用完整 Paper58 latent-dynamics 预测图作为 Paper58 端结果：起始年份 AlphaEarth embedding 经 LatentDynamicsNet 预测下一年 embedding，再由 decoder 解码为 LULC 类别。评估时，Paper58、GeoSOS-FLUS、起始真值和结束真值都落在同一 ESRI/AlphaEarth 标签网格上。",
        "",
        "证据边界：这是比 `paper58_proxy` 更严格的同网格严格对比，但这些 AlphaEarth/ESRI 区域没有 GeoSOS-FLUS 原生 ANN 驱动因子与训练样本。因此 GeoSOS-FLUS 的适宜性层由完整 Paper58 预测图转换为概率，需求量使用非真值来源推导；本实验主要比较完整 Paper58 预测与 GeoSOS-FLUS 控制台 CA 分配器在同网格条件下的结果。",
        "",
        "## 运行说明",
        "",
        *(note_lines or ["- 没有额外说明。"]),
        "",
        "## 如何阅读误差图",
        "",
        "- 第一行展示起始年份真值、结束年份真值、完整 Paper58 结果、额外 Paper58 优化结果和 GeoSOS-FLUS 结果。",
        "- 第二行展示变化误差：蓝色是真实变化，绿色是命中，红色是漏判，金色是误报。",
        "- 绿色只代表变化位置命中；类别是否正确需要结合 `FoM` 和 `转换准确率`。",
        "",
        "## 总体对比结论",
        "",
        *_overall_comparison(rows),
        "",
        "## 指标表",
        "",
        _markdown_table(rows),
        "",
        "## 原始输入与结果图",
        "",
    ]
    for area, figure in spatial:
        sample = next(sample for sample in samples if sample.area == area)
        readme.extend(
            [
                f"### `{area}`",
                "",
                f"![{area}]({figure})",
                "",
                "#### 单区域判读",
                "",
                _area_interpretation(sample, rows),
                "",
                "#### 类别面积偏差",
                "",
                _class_bias_interpretation(
                    sample,
                    flus_predictions.get(sample.area),
                    _extra_predictions_for_area(extra_predictions, sample.area),
                ),
                "",
            ]
        )
    (output / "README.md").write_text("\n".join(readme), encoding="utf-8")

    overview_html = _html_bullets(_overall_comparison(rows))
    notes_html = _html_bullets([f"- {note}" for note in (notes or ["没有额外说明。"])])
    metric_html = _html_metric_table(rows)
    cards = "\n".join(
        (
            f'<section><h2>{html.escape(area)}</h2>'
            f'<img src="{html.escape(path)}" alt="{html.escape(area)}">'
            f'<h3>单区域判读</h3><p>{html.escape(_plain_markdown_text(_area_interpretation(next(sample for sample in samples if sample.area == area), rows)))}</p>'
            f'<h3>类别面积偏差</h3><p>{html.escape(_class_bias_interpretation(next(sample for sample in samples if sample.area == area), flus_predictions.get(area), _extra_predictions_for_area(extra_predictions, area)))}</p>'
            "</section>"
        )
        for area, path in spatial
    )
    (output / "report.html").write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>完整 Paper58 与 GeoSOS-FLUS 同网格严格对比报告</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; color: #1f2328; background: #f6f7f8; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    section {{ background: white; border: 1px solid #d8dee4; border-radius: 8px; margin: 16px 0; padding: 16px; }}
    h1, h2, h3 {{ line-height: 1.25; }}
    p, li {{ line-height: 1.7; }}
    img {{ width: 100%; height: auto; display: block; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  </style>
</head>
<body><main>
<h1>完整 Paper58 与 GeoSOS-FLUS 同网格严格对比报告</h1>
<p>本报告比较完整 Paper58 latent-dynamics 预测图与 GeoSOS-FLUS 控制台在同一 ESRI/AlphaEarth 标签网格上的结果。</p>
<p><strong>证据边界：</strong>GeoSOS-FLUS 的适宜性层由完整 Paper58 预测图转换为概率，需求量使用非目标真值来源推导；具体 demand source 见运行说明。</p>
<section>
<h2>运行说明</h2>
{notes_html}
</section>
<section>
<h2>总体对比结论</h2>
{overview_html}
</section>
<section>
<h2>指标表</h2>
{metric_html}
</section>
{cards}
</main></body></html>
""",
        encoding="utf-8",
    )
    return {"output_dir": str(output), "n_samples": len(samples), "n_metric_rows": len(rows), "n_spatial_panels": len(spatial)}


def run_same_grid_comparison(
    paper58_predictions_dir: Path = DEFAULT_PAPER58_PREDICTIONS_DIR,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    flus_executable: Path | None = None,
    max_samples: int | None = None,
    console_runner: ConsoleRunner | None = None,
    flus_demand_source: str = "transition_prior",
    extra_prediction_specs: dict[str, Path] | None = None,
) -> dict[str, Any]:
    samples = discover_same_grid_samples(paper58_predictions_dir, labels_dir)
    if max_samples is not None:
        samples = samples[: int(max_samples)]
    if not samples:
        raise ValueError(f"no same-grid Paper58 predictions found under {paper58_predictions_dir}")
    output = Path(output_dir)
    flus_dir = output / "maps" / "geosos_flus_console"
    paper58_dir = output / "maps" / "paper58_latent_dynamics"
    flus_dir.mkdir(parents=True, exist_ok=True)
    paper58_dir.mkdir(parents=True, exist_ok=True)
    flus_predictions: dict[str, np.ndarray] = {}
    extra_predictions: dict[str, dict[str, np.ndarray]] = {method: {} for method in (extra_prediction_specs or {})}
    metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for sample in samples:
        np.save(paper58_dir / f"{sample.area}_{sample.start_year}_{sample.end_year}_paper58_latent_dynamics.npy", sample.paper58_prediction)
        sample_extra: dict[str, np.ndarray] = {}
        for method, source in (extra_prediction_specs or {}).items():
            try:
                prediction = _load_extra_prediction_for_sample(method, source, sample)
                sample_extra[method] = prediction
                extra_predictions.setdefault(method, {})[sample.area] = prediction
                method_dir = output / "maps" / _slug(method)
                method_dir.mkdir(parents=True, exist_ok=True)
                np.save(method_dir / f"{sample.area}_{sample.start_year}_{sample.end_year}_{_slug(method)}.npy", prediction)
            except Exception as exc:
                failures.append({"area": sample.area, "reason": f"{method}: {type(exc).__name__}: {exc}"})
        flus_pred = None
        if flus_executable is not None:
            try:
                flus_path = run_flus_console_for_sample(
                    sample=sample,
                    training_pairs=training_pairs_for_sample(sample, samples),
                    case_root=output / "flus_cases",
                    prediction_dir=flus_dir,
                    flus_executable=Path(flus_executable),
                    console_runner=console_runner,
                    demand_source=flus_demand_source,
                )
                flus_pred = load_flus_prediction(flus_path, expected_shape=sample.start.shape, allowed_classes=set(class_values_from_maps(sample.start, sample.end, sample.paper58_prediction)))
                flus_predictions[sample.area] = flus_pred
            except Exception as exc:
                failures.append({"area": sample.area, "reason": f"{type(exc).__name__}: {exc}"})
        metric_rows.extend(evaluate_predictions(sample, flus_pred, sample_extra))
    _write_csv(output / "flus_failures.csv", failures, ["area", "reason"])
    notes = [
        "Paper58 端使用 src/adk_world_model/weights/latent_dynamics_v1.pt 和 lulc_decoder_v1.pkl 生成的完整 latent-dynamics LULC 预测。",
        f"GeoSOS-FLUS 控制台在同一 ESRI/AlphaEarth 标签网格上运行；适宜性层由完整 Paper58 预测图转换为 one-hot 概率，需求量来源为 `{flus_demand_source}`，不使用目标结束年份真值需求。",
    ]
    if failures:
        notes.append(f"GeoSOS-FLUS 控制台有 {len(failures)} 个样本运行失败；详见 flus_failures.csv。")
    for method, source in (extra_prediction_specs or {}).items():
        notes.append(f"额外方法 `{method}` 从 `{source}` 读取，用于评估 Paper58 优化后输出。")
    result = write_same_grid_report(output, samples, flus_predictions, extra_predictions, metric_rows, notes)
    _write_json(
        output / "run_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "paper58_predictions_dir": Path(paper58_predictions_dir),
            "labels_dir": Path(labels_dir),
            "output_dir": output,
            "flus_executable": Path(flus_executable) if flus_executable else None,
            "flus_demand_source": flus_demand_source,
            "extra_prediction_specs": extra_prediction_specs or {},
            "samples": [
                {
                    "area": sample.area,
                    "start_year": sample.start_year,
                    "end_year": sample.end_year,
                    "shape": list(sample.start.shape),
                    "valid_pixels": int(np.count_nonzero(sample.valid_mask)),
                    "changed_pixels": int(np.count_nonzero((sample.end != sample.start) & sample.valid_mask)),
                    "prediction_path": sample.prediction_path,
                }
                for sample in samples
            ],
            "failures": failures,
            "result": result,
        },
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run strict same-grid comparison of true Paper58 latent-dynamics against GeoSOS-FLUS console.")
    parser.add_argument("--paper58-predictions-dir", type=Path, default=DEFAULT_PAPER58_PREDICTIONS_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flus-executable", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--flus-demand-source",
        choices=["transition_prior", "paper58_prediction", "start_persistence"],
        default="transition_prior",
    )
    parser.add_argument(
        "--extra-prediction",
        action="append",
        default=[],
        metavar="METHOD=PATH_OR_DIR",
        help="Optional extra prediction method to include in metrics and figures.",
    )
    args = parser.parse_args(argv)
    result = run_same_grid_comparison(
        paper58_predictions_dir=args.paper58_predictions_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        flus_executable=args.flus_executable,
        max_samples=args.max_samples,
        flus_demand_source=args.flus_demand_source,
        extra_prediction_specs=_parse_extra_prediction_specs(args.extra_prediction),
    )
    print(
        "Same-grid true Paper58 vs GeoSOS-FLUS comparison: "
        f"{result['n_samples']} sample(s), "
        f"{result['n_metric_rows']} metric row(s), "
        f"output={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
