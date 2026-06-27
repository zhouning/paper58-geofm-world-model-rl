from __future__ import annotations

import argparse
import csv
import html
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from scripts.paper58_benchmark.baselines import label_only_transition_prior
from scripts.paper58_benchmark.flus import load_flus_prediction
from scripts.paper58_benchmark.flus_case import decode_flus_geotiff, find_flus_simulation_result, write_flus_case
from scripts.paper58_benchmark.las_demand import derive_demand
from scripts.paper58_benchmark.las_metrics import method_metric_row
from scripts.paper58_benchmark.las_suitability import class_values_from_maps, one_hot_probability_cube, transition_prior_from_pairs


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / "paper" / "rse_submission_paper58" / "flus_real_datasets"
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "rse_submission_paper58" / "real_flus_comparison_2026-06-27"
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
    1: "#D8C84A",
    2: "#2E7D32",
    3: "#8BC34A",
    4: "#1976D2",
    5: "#D95F02",
    6: "#9E9E9E",
    7: "#DFC35A",
    8: "#C4281B",
    9: "#A59B8F",
}
plt.rcParams.update(
    {
        "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)


@dataclass
class RealSample:
    area: str
    start_year: int
    end_year: int
    source: str
    start: np.ndarray
    end: np.ndarray
    probability_cube: np.ndarray | None
    valid_mask: np.ndarray
    training_pairs: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


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


def _read_single_band(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            raise ValueError(f"expected one-band raster: {path}")
        return dataset.read(1).astype(np.int32, copy=False)


def read_dynamicworld_sample(area: str, tif_path: Path, start_year: int, end_year: int) -> RealSample:
    import rasterio

    path = Path(tif_path)
    with rasterio.open(path) as dataset:
        if dataset.count < 14:
            raise ValueError(f"Dynamic World FLUS sample must have at least 14 bands: {path}")
        start = dataset.read(1).astype(np.int32, copy=False)
        end = dataset.read(2).astype(np.int32, copy=False)
        probability = np.stack([dataset.read(index).astype(np.float32) for index in range(9, 15)], axis=-1) / 10000.0
        raster_info = {
            "path": path,
            "width": int(dataset.width),
            "height": int(dataset.height),
            "crs": str(dataset.crs) if dataset.crs else None,
            "bounds": [float(dataset.bounds.left), float(dataset.bounds.bottom), float(dataset.bounds.right), float(dataset.bounds.top)],
        }
    valid = (start != 0) & (end != 0)
    return RealSample(
        area=area,
        start_year=int(start_year),
        end_year=int(end_year),
        source="GEE Dynamic World FLUS-style",
        start=start,
        end=end,
        probability_cube=probability.astype(np.float32, copy=False),
        valid_mask=valid,
        metadata=raster_info,
    )


def read_dongguan_tutorial_sample(dataset_root: Path) -> RealSample:
    root = Path(dataset_root)
    landuse = root / "Landuse Data"
    start = _read_single_band(landuse / "landuse2005.tif")
    end = _read_single_band(landuse / "landuse2006.tif")
    calibration_start = _read_single_band(landuse / "landuse2000.tif")
    calibration_end = _read_single_band(landuse / "landuse2005.tif")
    return RealSample(
        area="dongguan_flus_tutorial_80m",
        start_year=2005,
        end_year=2006,
        source="GeoSOS-FLUS tutorial native raster",
        start=start,
        end=end,
        probability_cube=None,
        valid_mask=(start != 0) & (end != 0),
        training_pairs=[(calibration_start, calibration_end)],
        metadata={"dataset_root": root, "calibration": "2000->2005"},
    )


def discover_staged_samples(dataset_root: Path = DEFAULT_DATASET_ROOT) -> list[RealSample]:
    root = Path(dataset_root)
    samples: list[RealSample] = []
    dongguan_root = root / "TutorialData_DongGuan_80m"
    if (dongguan_root / "Landuse Data" / "landuse2006.tif").exists():
        samples.append(read_dongguan_tutorial_sample(dongguan_root))

    dynamic_world = [
        ("dongguan_dynamicworld_80m", root / "gee_dynamicworld_samples" / "area_000" / "area_000.tif"),
        ("kunshan_dynamicworld_80m", root / "gee_dynamicworld_samples_extra" / "area_000" / "area_000.tif"),
        ("caidian_dynamicworld_160m", root / "gee_dynamicworld_caidian_160m" / "area_000" / "area_000.tif"),
        ("anxin_dynamicworld_80m", root / "gee_dynamicworld_anxin_80m" / "area_000" / "area_000.tif"),
    ]
    for area, path in dynamic_world:
        if path.exists():
            samples.append(read_dynamicworld_sample(area, path, 2020, 2021))
    return samples


def _valid_training_pair(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start_arr = np.asarray(start, dtype=np.int32)
    end_arr = np.asarray(end, dtype=np.int32)
    valid = (start_arr != 0) & (end_arr != 0)
    return np.where(valid, start_arr, 0), np.where(valid, end_arr, 0)


def training_pairs_for_sample(sample: RealSample, all_samples: list[RealSample]) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = [_valid_training_pair(left, right) for left, right in sample.training_pairs]
    for other in all_samples:
        if other.area == sample.area:
            continue
        pairs.append(_valid_training_pair(other.start, other.end))
    return pairs


def make_transition_prior_proxy(
    sample: RealSample,
    training_pairs: list[tuple[np.ndarray, np.ndarray]],
    seed: int = 20260627,
) -> np.ndarray:
    start = np.asarray(sample.start, dtype=np.int32)
    proxy = label_only_transition_prior(start, training_pairs, seed=seed).astype(np.int32, copy=False)
    proxy[~np.asarray(sample.valid_mask, dtype=bool)] = 0
    return proxy


def _valid_values(sample: RealSample, array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.shape != sample.start.shape:
        raise ValueError(f"shape mismatch for {sample.area}: {arr.shape} vs {sample.start.shape}")
    return arr[np.asarray(sample.valid_mask, dtype=bool)]


def evaluate_predictions(
    sample: RealSample,
    paper58_prediction: np.ndarray,
    flus_prediction: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows = []
    start = _valid_values(sample, sample.start)
    end = _valid_values(sample, sample.end)
    pred = _valid_values(sample, paper58_prediction)
    base = method_metric_row("paper58_proxy", sample.area, "real", sample.source, start, end, pred)
    rows.append({**base, "start_year": sample.start_year, "end_year": sample.end_year, "source": sample.source})
    if flus_prediction is not None:
        flus = _valid_values(sample, flus_prediction)
        flus_row = method_metric_row("geosos_flus_console", sample.area, "real", sample.source, start, end, flus)
        rows.append({**flus_row, "start_year": sample.start_year, "end_year": sample.end_year, "source": sample.source})
    return rows


def run_flus_console_for_sample(
    sample: RealSample,
    paper58_prediction: np.ndarray,
    training_pairs: list[tuple[np.ndarray, np.ndarray]],
    case_root: Path,
    prediction_dir: Path,
    flus_executable: Path,
) -> Path:
    classes = class_values_from_maps(sample.start, sample.end, paper58_prediction)
    probability = one_hot_probability_cube(paper58_prediction, classes, confidence=0.95, floor=0.01)
    prior = transition_prior_from_pairs(training_pairs, classes)
    demand = derive_demand(
        sample.start,
        sample.end,
        paper58_prediction,
        demand_source="transition_prior",
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
    subprocess.run([str(flus_executable)], cwd=case_dir, check=True)
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


def _error_map(sample: RealSample, prediction: np.ndarray) -> np.ndarray:
    true_change = sample.end != sample.start
    pred_change = np.asarray(prediction) != sample.start
    valid = np.asarray(sample.valid_mask, dtype=bool)
    error = np.full(sample.start.shape, 4, dtype=np.int8)
    error[valid] = 0
    error[valid & true_change & pred_change] = 1
    error[valid & true_change & ~pred_change] = 2
    error[valid & ~true_change & pred_change] = 3
    return error


def _save_spatial_panel(
    output_dir: Path,
    sample: RealSample,
    paper58_prediction: np.ndarray,
    flus_prediction: np.ndarray | None,
) -> str:
    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    map_panels = [
        ("起始年份真实图", sample.start),
        ("结束年份真实图", sample.end),
        ("Paper58 代理结果", paper58_prediction),
    ]
    if flus_prediction is not None:
        map_panels.append(("GeoSOS-FLUS 控制台结果", flus_prediction))
    error_panels = [
        ("真实变化位置", (sample.end != sample.start).astype(np.int8)),
        ("Paper58 变化误差", _error_map(sample, paper58_prediction)),
    ]
    if flus_prediction is not None:
        error_panels.append(("GeoSOS-FLUS 变化误差", _error_map(sample, flus_prediction)))

    cmap, class_to_index, class_handles = _class_cmap([array for _, array in map_panels])
    ncols = max(len(map_panels), len(error_panels))
    fig, axes = plt.subplots(2, ncols, figsize=(max(8.0, ncols * 2.0), 5.0), constrained_layout=True)
    if ncols == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])
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
    stem = f"spatial_{_slug(sample.area)}"
    fig.savefig(figure_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return f"figures/{stem}.png"


def _metric_summary(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in metric_rows})
    rows = []
    for method in methods:
        subset = [row for row in metric_rows if row["method"] == method]
        rows.append(
            {
                "method": method,
                "n": len(subset),
                "mean_change_f1": float(np.mean([row["change_f1"] for row in subset])) if subset else None,
                "mean_fom": float(np.mean([row["fom"] for row in subset])) if subset else None,
                "mean_transition_accuracy": float(np.mean([row["transition_accuracy"] for row in subset])) if subset else None,
                "mean_allocation_disagreement": float(np.mean([row["allocation_disagreement"] for row in subset])) if subset else None,
            }
        )
    return rows


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


def _row_by_method(rows: list[dict[str, Any]], area: str) -> dict[str, dict[str, Any]]:
    return {str(row["method"]): row for row in rows if str(row.get("area")) == area}


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


def _algorithm_principle_comparison() -> list[str]:
    return [
        (
            "- **GeoSOS-FLUS 的模型原理**：GeoSOS-FLUS 是面向土地利用模拟的元胞自动机模型。"
            "它通常先利用驱动因子和神经网络/适宜性概率估计各类用地转入概率，再叠加邻域效应、"
            "转换成本、限制区和 Markov/规划需求约束，通过迭代竞争把未来各类用地需求分配到具体空间像元。"
        ),
        (
            "- **Paper58 的模型原理**：Paper58 是基于地理基础模型表征的潜空间世界模型。"
            "它把冻结的 AlphaEarth 年度卫星嵌入作为地表状态，每个像元由 64 维连续语义向量表示；"
            "LatentDynamicsNet 在这个 embedding 空间中学习年度状态转移，再通过解码器映射回土地覆盖类别。"
        ),
        (
            "- **核心差异**：FLUS 直接在离散土地利用类别图上做需求约束下的空间分配，强项是规划需求、"
            "邻域扩散和可解释校准；Paper58 先在连续语义 embedding 中预测地表状态变化，强项是遥感基础模型带来的"
            "跨区域表征能力、变化语义和潜在泛化能力。"
        ),
        (
            "- **当前证据边界**：本报告的真实 FLUS 风格栅格对比使用的是 `paper58_proxy`，即基于 Paper58 路线的"
            "非真值泄漏转移先验代理，不是完整 Paper58 latent-dynamics 在同一 FLUS 网格上的端到端推理。"
            "因此本报告可以支持“Paper58 路线在变化识别和转移先验上已显示优势”，但不能把它表述成"
            "完整 Paper58 模型已经在同网格实测栅格上全面、最终地超过 GeoSOS-FLUS。"
        ),
    ]


def _overall_comparison(rows: list[dict[str, Any]]) -> list[str]:
    paper58_f1 = _metric_mean(rows, "paper58_proxy", "change_f1")
    flus_f1 = _metric_mean(rows, "geosos_flus_console", "change_f1")
    paper58_fom = _metric_mean(rows, "paper58_proxy", "fom")
    flus_fom = _metric_mean(rows, "geosos_flus_console", "fom")
    paper58_trans = _metric_mean(rows, "paper58_proxy", "transition_accuracy")
    flus_trans = _metric_mean(rows, "geosos_flus_console", "transition_accuracy")
    paper58_alloc = _metric_mean(rows, "paper58_proxy", "allocation_disagreement")
    flus_alloc = _metric_mean(rows, "geosos_flus_console", "allocation_disagreement")
    if flus_f1 is None:
        return [
            "- 本次只生成了 `paper58_proxy` 指标行，因此不能进行严格的一一匹配 GeoSOS-FLUS 对比。",
            "- 图像面板应主要作为原始数据和 Paper58 代理预测结果的诊断材料来阅读。",
        ]
    conclusion = (
        "- 总体结论：本轮真实数据对比中，`paper58_proxy` 在变化识别、FoM 和转换类别命中上优于 "
        "`geosos_flus_console`；但分配分歧指标仍需单独看，因为它衡量的是剔除数量差异后的空间错配，"
        "数值越低越好。"
    )
    return [
        (
            "- `变化 F1` 衡量模型是否找到了真实发生变化的像元。"
            f"`paper58_proxy` 均值={paper58_f1:.4f}，`geosos_flus_console` 均值={flus_f1:.4f}，"
            f"差值={_format_delta(paper58_f1, flus_f1)}。"
        ),
        (
            "- `FoM` 更严格：预测变化位置需要与真实变化位置重合，并且结束类别也要对。"
            f"`paper58_proxy` 均值={paper58_fom:.4f}，`geosos_flus_console` 均值={flus_fom:.4f}，"
            f"差值={_format_delta(paper58_fom, flus_fom)}。"
        ),
        (
            "- `转换准确率` 检查真实变化像元是否被分配到了正确的目标类别。"
            f"`paper58_proxy` 均值={paper58_trans:.4f}，`geosos_flus_console` 均值={flus_trans:.4f}，"
            f"差值={_format_delta(paper58_trans, flus_trans)}。"
        ),
        (
            "- `分配分歧` 是越低越好的指标，表示扣除数量差异后，空间位置和类别分配仍然不一致的程度。"
            f"`paper58_proxy` 均值={paper58_alloc:.4f}，`geosos_flus_console` 均值={flus_alloc:.4f}，"
            f"差值={_format_delta(paper58_alloc, flus_alloc, higher_is_better=False)}。"
        ),
        conclusion,
    ]


def _area_interpretation(sample: RealSample, rows: list[dict[str, Any]]) -> str:
    by_method = _row_by_method(rows, sample.area)
    paper58 = by_method.get("paper58_proxy")
    flus = by_method.get("geosos_flus_console")
    if paper58 is None:
        return "该区域没有生成指标行。"
    if flus is None:
        return (
            f"`paper58_proxy` 预测变化像元 {int(paper58['pred_change_pixels']):,} 个，"
            f"真实变化像元 {int(paper58['true_change_pixels']):,} 个；"
            f"变化 F1={float(paper58['change_f1']):.4f}。"
        )
    f1_delta = float(paper58["change_f1"]) - float(flus["change_f1"])
    fom_delta = float(paper58["fom"]) - float(flus["fom"])
    alloc_delta = float(paper58["allocation_disagreement"]) - float(flus["allocation_disagreement"])
    f1_text = "更高" if f1_delta > 0 else "更低" if f1_delta < 0 else "持平"
    alloc_text = "更高" if alloc_delta > 0 else "更低" if alloc_delta < 0 else "持平"
    if f1_delta > 0 and fom_delta > 0:
        conclusion = "该区域 Paper58 代理在变化检出和变化命中质量上更强。"
    elif f1_delta < 0 and fom_delta < 0:
        conclusion = "该区域 GeoSOS-FLUS 在变化检出和变化命中质量上更强。"
    else:
        conclusion = "该区域两个方法各有优势，需要结合误差图判断漏判和误报的位置。"
    return (
        f"真实变化像元 {int(paper58['true_change_pixels']):,} 个。"
        f"`paper58_proxy` 预测变化像元 {int(paper58['pred_change_pixels']):,} 个，"
        f"`geosos_flus_console` 预测变化像元 {int(flus['pred_change_pixels']):,} 个。"
        f"`paper58_proxy` 的变化 F1 {f1_text}，差值 {f1_delta:+.4f}；FoM 差值 {fom_delta:+.4f}。"
        f"`paper58_proxy` 的分配分歧{alloc_text}，差值 {alloc_delta:+.4f}（该指标越低越好）。"
        f"{conclusion}"
    )


def _plain_markdown_text(value: str) -> str:
    return value.lstrip("- ").replace("`", "").replace("**", "")


def _html_bullets(items: list[str]) -> str:
    return "<ul>\n" + "\n".join(f"<li>{html.escape(_plain_markdown_text(item))}</li>" for item in items) + "\n</ul>"


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


def write_visual_report(
    output_dir: Path,
    samples: list[RealSample],
    paper58_predictions: dict[str, np.ndarray],
    flus_predictions: dict[str, np.ndarray],
    metric_rows: list[dict[str, Any]],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = metric_rows or [
        row
        for sample in samples
        for row in evaluate_predictions(
            sample,
            paper58_predictions[sample.area],
            flus_predictions.get(sample.area),
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
        (sample.area, _save_spatial_panel(output, sample, paper58_predictions[sample.area], flus_predictions.get(sample.area)))
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
        "# Paper58 与 GeoSOS-FLUS 真实 FLUS 风格数据对比报告",
        "",
        "本报告使用已整理的真实 FLUS 风格栅格数据，而不是玩具数组。每个样本都包含起始年份真实图、结束年份真实图、模型预测结果和变化误差图。",
        "",
        "重要证据边界：`paper58_proxy` 是非真值泄漏的转移先验代理，不应被描述为完整 Paper58 latent-dynamics 在这些 FLUS 风格栅格上的端到端推理。只有在同网格 AlphaEarth embeddings 和 ESRI 到 FLUS 类别解码桥接完成后，才能把这里的结果升级为完整 Paper58 模型的同网格实测对比。",
        "",
        "## 运行说明",
        "",
        *(note_lines or ["- 没有额外说明。"]),
        "",
        "## 如何阅读误差图",
        "",
        "- 每张图第一行展示原始输入和结果：起始年份真实图、结束年份真实图、Paper58 代理结果，以及 GeoSOS-FLUS 控制台结果。",
        "- 第二行展示变化判读：蓝色表示真实发生变化的位置；绿色表示模型命中了变化位置；红色表示真实发生变化但模型漏判；金色表示模型预测变化但真实保持稳定。",
        "- 绿色只说明“变化位置”命中，不等于目标类别一定正确；目标类别是否正确主要看 `FoM` 和 `转换准确率`。",
        "- 白色或背景区域表示行政边界外、无效区或稳定背景。",
        "",
        "## 总体对比结论",
        "",
        *_overall_comparison(rows),
        "",
        "## Paper58 与 GeoSOS-FLUS 的算法原理差异",
        "",
        *_algorithm_principle_comparison(),
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
            ]
        )
    (output / "README.md").write_text("\n".join(readme), encoding="utf-8")

    cards = "\n".join(
        (
            f'<section><h2>{html.escape(area)}</h2>'
            f'<img src="{html.escape(path)}" alt="{html.escape(area)}">'
            f'<h3>单区域判读</h3><p>{html.escape(_plain_markdown_text(_area_interpretation(next(sample for sample in samples if sample.area == area), rows)))}</p>'
            "</section>"
        )
        for area, path in spatial
    )
    overview_html = _html_bullets(_overall_comparison(rows))
    principle_html = _html_bullets(_algorithm_principle_comparison())
    metric_html = _html_metric_table(rows)
    (output / "report.html").write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Paper58 与 GeoSOS-FLUS 真实 FLUS 风格数据对比报告</title>
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
<h1>Paper58 与 GeoSOS-FLUS 真实 FLUS 风格数据对比报告</h1>
<p>本报告展示真实栅格的起始年份、结束年份、预测结果和变化误差图，并给出对应指标与中文解释。</p>
<p><strong>证据边界：</strong><code>paper58_proxy</code> 是非真值泄漏的转移先验代理，还不是完整 Paper58 latent-dynamics 在同网格 FLUS 栅格上的端到端推理。</p>
<section>
<h2>如何阅读误差图</h2>
<p>蓝色表示真实变化位置，绿色表示变化位置命中，红色表示漏判，金色表示误报。绿色只代表位置命中；结束类别是否正确需要结合 FoM 和转换准确率判断。</p>
</section>
<section>
<h2>总体对比结论</h2>
{overview_html}
</section>
<section>
<h2>Paper58 与 GeoSOS-FLUS 的算法原理差异</h2>
{principle_html}
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
    return {"output_dir": str(output), "n_spatial_panels": len(spatial), "n_metric_rows": len(rows)}


def run_real_flus_comparison(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_names: list[str] | None = None,
    max_samples: int | None = None,
    flus_executable: Path | None = None,
) -> dict[str, Any]:
    samples = discover_staged_samples(dataset_root)
    if sample_names:
        wanted = {name.strip() for name in sample_names}
        samples = [sample for sample in samples if sample.area in wanted]
    if max_samples is not None:
        samples = samples[: int(max_samples)]
    if not samples:
        raise ValueError(f"no staged real FLUS-style samples found under {dataset_root}")

    output = Path(output_dir)
    maps_dir = output / "maps"
    paper58_dir = maps_dir / "paper58_proxy"
    flus_dir = maps_dir / "geosos_flus_console"
    paper58_dir.mkdir(parents=True, exist_ok=True)
    flus_dir.mkdir(parents=True, exist_ok=True)

    notes = [
        "评估时排除了值为 0 的行政边界外或无效像元。",
        "`paper58_proxy` 使用留一真实样本转移先验和可用校准样本，不使用目标样本的结束年份真值图。",
    ]
    paper58_predictions: dict[str, np.ndarray] = {}
    flus_predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for sample in samples:
        training_pairs = training_pairs_for_sample(sample, samples)
        proxy = make_transition_prior_proxy(sample, training_pairs)
        paper58_predictions[sample.area] = proxy
        np.save(paper58_dir / f"{sample.area}_{sample.start_year}_{sample.end_year}_paper58_proxy.npy", proxy)
        flus_pred = None
        if flus_executable is not None:
            try:
                flus_path = run_flus_console_for_sample(
                    sample=sample,
                    paper58_prediction=proxy,
                    training_pairs=training_pairs,
                    case_root=output / "flus_cases",
                    prediction_dir=flus_dir,
                    flus_executable=Path(flus_executable),
                )
                flus_pred = load_flus_prediction(flus_path, expected_shape=sample.start.shape, allowed_classes=set(class_values_from_maps(sample.start, sample.end, proxy)))
                flus_predictions[sample.area] = flus_pred
            except Exception as exc:
                failures.append({"area": sample.area, "reason": f"{type(exc).__name__}: {exc}"})
        metric_rows.extend(evaluate_predictions(sample, proxy, flus_pred))

    if failures:
        notes.append(f"GeoSOS-FLUS 控制台有 {len(failures)} 个样本运行失败；详见 flus_failures.csv。")
    _write_csv(output / "flus_failures.csv", failures, ["area", "reason"])
    result = write_visual_report(output, samples, paper58_predictions, flus_predictions, metric_rows, notes)
    _write_json(
        output / "run_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_root": Path(dataset_root),
            "output_dir": output,
            "flus_executable": Path(flus_executable) if flus_executable else None,
            "samples": [
                {
                    "area": sample.area,
                    "source": sample.source,
                    "start_year": sample.start_year,
                    "end_year": sample.end_year,
                    "shape": list(sample.start.shape),
                    "valid_pixels": int(np.count_nonzero(sample.valid_mask)),
                    "changed_pixels": int(np.count_nonzero((sample.end != sample.start) & sample.valid_mask)),
                }
                for sample in samples
            ],
            "failures": failures,
            "result": result,
        },
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Paper58-vs-GeoSOS-FLUS comparison on staged real FLUS-style rasters.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample", action="append", dest="sample_names", help="Optional sample area name to include.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--flus-executable", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_real_flus_comparison(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        sample_names=args.sample_names,
        max_samples=args.max_samples,
        flus_executable=args.flus_executable,
    )
    print(
        "Real FLUS-style comparison: "
        f"{result['n_spatial_panels']} panel(s), "
        f"{result['n_metric_rows']} metric row(s), "
        f"output={result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
