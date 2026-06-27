from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper58_benchmark.las_metrics import method_metric_row
from scripts.paper58_benchmark.run_seeded_flus_replicates import LOWER_IS_BETTER, METRICS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KEEP_FRACTIONS = [0.0, 0.25, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0]


@dataclass(frozen=True)
class GateSweepCase:
    area: str
    start_year: int
    end_year: int
    start_map: np.ndarray
    end_map: np.ndarray
    prediction_map: np.ndarray
    score_map: np.ndarray
    source_dir: Path


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


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def apply_scored_change_gate(
    start_map: np.ndarray,
    prediction_map: np.ndarray,
    score_map: np.ndarray,
    keep_fraction: float,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = np.asarray(start_map)
    prediction = np.asarray(prediction_map)
    score = np.asarray(score_map, dtype=np.float32)
    if start.shape != prediction.shape or start.shape != score.shape:
        raise ValueError(f"shape mismatch: start={start.shape}, prediction={prediction.shape}, score={score.shape}")
    keep = float(keep_fraction)
    if keep < 0.0 or keep > 1.0:
        raise ValueError(f"keep_fraction must be in [0, 1]: {keep}")
    valid = np.ones(start.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != start.shape:
        raise ValueError(f"valid_mask shape {valid.shape} does not match start shape {start.shape}")

    candidates = valid & (prediction != start)
    candidate_indices = np.flatnonzero(candidates.ravel())
    keep_count = int(round(candidate_indices.size * keep))
    gated = start.copy()
    kept_indices = np.array([], dtype=np.int64)
    if keep_count > 0 and candidate_indices.size > 0:
        order = candidate_indices[np.argsort(score.ravel()[candidate_indices])[::-1]]
        kept_indices = order[:keep_count]
        gated.ravel()[kept_indices] = prediction.ravel()[kept_indices]
    return gated.astype(prediction.dtype, copy=False), {
        "candidate_change_pixels": int(candidate_indices.size),
        "kept_change_pixels": int(kept_indices.size),
        "keep_fraction": keep,
    }


def _valid_values(array: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    return np.asarray(array)[np.asarray(valid_mask, dtype=bool)]


def sweep_scored_change_gate(
    area: str,
    start_map: np.ndarray,
    end_map: np.ndarray,
    prediction_map: np.ndarray,
    score_map: np.ndarray,
    keep_fractions: list[float],
    start_year: int = 2020,
    end_year: int = 2021,
) -> list[dict[str, Any]]:
    start = np.asarray(start_map)
    end = np.asarray(end_map)
    prediction = np.asarray(prediction_map)
    score = np.asarray(score_map)
    if start.shape != end.shape or start.shape != prediction.shape or start.shape != score.shape:
        raise ValueError(
            f"shape mismatch: start={start.shape}, end={end.shape}, prediction={prediction.shape}, score={score.shape}"
        )
    valid = (start != 0) & (end != 0) & (prediction != 0)
    rows: list[dict[str, Any]] = []
    for keep_fraction in keep_fractions:
        gated, diagnostics = apply_scored_change_gate(start, prediction, score, keep_fraction, valid_mask=valid)
        row = method_metric_row(
            method="paper58_change_gate_sweep",
            area=area,
            tier="same_grid",
            stratum="change_gate_sweep",
            start_map=_valid_values(start, valid),
            true_map=_valid_values(end, valid),
            pred_map=_valid_values(gated, valid),
        )
        rows.append(
            {
                **row,
                "start_year": int(start_year),
                "end_year": int(end_year),
                "keep_fraction": float(keep_fraction),
                "candidate_change_pixels": diagnostics["candidate_change_pixels"],
                "kept_change_pixels": diagnostics["kept_change_pixels"],
            }
        )
    return rows


def best_fraction_by_metric(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        metric_rows = [row for row in rows if metric in row]
        if not metric_rows:
            continue
        reverse = metric not in LOWER_IS_BETTER
        best[metric] = sorted(metric_rows, key=lambda row: float(row[metric]), reverse=reverse)[0]
    return best


def load_case_from_change_gate_dir(change_gate_dir: Path) -> GateSweepCase:
    root = Path(change_gate_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    area = str(manifest["area"])
    start_year = int(manifest["start_year"])
    end_year = int(manifest["end_year"])
    start_path = _resolve_path(manifest["inputs"]["start"])
    prediction_path = _resolve_path(manifest["inputs"]["paper58_prediction"])
    end_path = start_path.with_name(f"{area}_lulc_{end_year}.npy")
    score_path = root / "diagnostics" / f"{area}_change_gate_score.npy"
    return GateSweepCase(
        area=area,
        start_year=start_year,
        end_year=end_year,
        start_map=np.load(start_path).astype(np.int32, copy=False),
        end_map=np.load(end_path).astype(np.int32, copy=False),
        prediction_map=np.load(prediction_path).astype(np.int32, copy=False),
        score_map=np.load(score_path).astype(np.float32, copy=False),
        source_dir=root,
    )


def _make_figures(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    areas = sorted({str(row["area"]) for row in rows})
    labels = {
        "change_f1": "变化 F1",
        "fom": "FoM",
        "transition_accuracy": "转换准确率",
        "allocation_disagreement": "分配分歧",
    }
    links: list[str] = []
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(6.8, 3.6))
        for area in areas:
            subset = sorted([row for row in rows if row["area"] == area], key=lambda row: float(row["keep_fraction"]))
            ax.plot(
                [float(row["keep_fraction"]) for row in subset],
                [float(row[metric]) for row in subset],
                marker="o",
                linewidth=1.4,
                label=area.replace("_dynamicworld", "").replace("_realistic", ""),
            )
        ax.set_xlabel("keep_fraction")
        ax.set_ylabel(labels.get(metric, metric))
        direction = "越低越好" if metric in LOWER_IS_BETTER else "越高越好"
        ax.set_title(f"{labels.get(metric, metric)} 对门控强度的敏感性（{direction}）", loc="left", fontweight="bold")
        ax.legend(fontsize=6)
        fig.tight_layout()
        path = figure_dir / f"sweep_{metric}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        links.append(path.relative_to(output_dir).as_posix())
    return links


def write_sweep_report(
    rows: list[dict[str, Any]],
    output_dir: Path,
    keep_fractions: list[float],
    create_figures: bool = True,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "area",
        "start_year",
        "end_year",
        "keep_fraction",
        "candidate_change_pixels",
        "kept_change_pixels",
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
    _write_csv(output / "sweep_metrics.csv", rows, fields)
    best_rows: list[dict[str, Any]] = []
    for area in sorted({str(row["area"]) for row in rows}):
        best = best_fraction_by_metric([row for row in rows if row["area"] == area])
        for metric, row in best.items():
            best_rows.append(
                {
                    "area": area,
                    "metric": metric,
                    "keep_fraction": row["keep_fraction"],
                    "metric_value": row[metric],
                    "change_f1": row["change_f1"],
                    "fom": row["fom"],
                    "transition_accuracy": row["transition_accuracy"],
                    "allocation_disagreement": row["allocation_disagreement"],
                }
            )
    _write_csv(
        output / "best_by_metric.csv",
        best_rows,
        [
            "area",
            "metric",
            "keep_fraction",
            "metric_value",
            "change_f1",
            "fom",
            "transition_accuracy",
            "allocation_disagreement",
        ],
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "keep_fractions": keep_fractions,
        "n_rows": len(rows),
    }
    (output / "summary.json").write_text(json.dumps(_json_ready(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    figure_links = _make_figures(rows, output) if create_figures else []

    lines = [
        "# Paper58 change-gate 门控强度诊断",
        "",
        "本报告复用 `apply_paper58_change_gate.py` 已保存的 score 栅格，不重新训练模型，也不重新调用 AlphaEarth。目的只是回答：固定 `keep_fraction=0.65` 是否适合所有真实区域。",
        "",
        f"- 扫描 keep_fraction：{', '.join(f'{value:.2f}' for value in keep_fractions)}",
        "- `keep_fraction=1.00` 等价于不门控，保留 Paper58 原始 latent-dynamics 变化。",
        "- `keep_fraction=0.00` 等价于全部回退到起始图，只作为诊断下界，不是可用模型。",
        "",
    ]
    if figure_links:
        lines.extend(["## 敏感性曲线", ""])
        for link in figure_links:
            lines.extend([f"![{Path(link).stem}]({link})", ""])
    lines.extend(
        [
            "## 每个区域各指标最优 keep_fraction",
            "",
            "| 区域 | 指标 | 最优 keep_fraction | 指标值 | 同时的 F1 | 同时的 FoM | 同时的转换准确率 | 同时的分配分歧 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    label = {
        "change_f1": "变化 F1",
        "fom": "FoM",
        "transition_accuracy": "转换准确率",
        "allocation_disagreement": "分配分歧",
    }
    for row in best_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['area']}`",
                    label.get(str(row["metric"]), str(row["metric"])),
                    f"{float(row['keep_fraction']):.2f}",
                    f"{float(row['metric_value']):.4f}",
                    f"{float(row['change_f1']):.4f}",
                    f"{float(row['fom']):.4f}",
                    f"{float(row['transition_accuracy']):.4f}",
                    f"{float(row['allocation_disagreement']):.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 诊断结论",
            "",
            "- 如果不同区域或不同指标的最优 keep_fraction 明显不一致，说明固定 0.65 不是最终算法形态。",
            "- 若 `keep_fraction=1.00` 在转换准确率上经常最好，说明当前门控主要在压低虚报，但会切掉部分真实类别转换。",
            "- 下一轮 Paper58 优化应从固定比例门控升级为自适应门控，并显式加入召回率/转换准确率保护项。",
            "",
            "## 文件",
            "",
            "- `sweep_metrics.csv`：所有区域、所有 keep_fraction 的完整指标。",
            "- `best_by_metric.csv`：每个区域每个指标的最优 keep_fraction。",
            "- `summary.json`：运行清单。",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_sweep(
    change_gate_dirs: list[Path],
    output_dir: Path,
    keep_fractions: list[float],
    create_figures: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for change_gate_dir in change_gate_dirs:
        case = load_case_from_change_gate_dir(Path(change_gate_dir))
        rows.extend(
            sweep_scored_change_gate(
                area=case.area,
                start_year=case.start_year,
                end_year=case.end_year,
                start_map=case.start_map,
                end_map=case.end_map,
                prediction_map=case.prediction_map,
                score_map=case.score_map,
                keep_fractions=keep_fractions,
            )
        )
    write_sweep_report(rows, output_dir=output_dir, keep_fractions=keep_fractions, create_figures=create_figures)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep Paper58 change-gate keep fractions from saved score rasters.")
    parser.add_argument("--change-gate-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-fraction", action="append", type=float, default=[])
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    keep_fractions = args.keep_fraction or DEFAULT_KEEP_FRACTIONS
    rows = run_sweep(
        change_gate_dirs=args.change_gate_dir,
        output_dir=args.output_dir,
        keep_fractions=keep_fractions,
        create_figures=not args.no_figures,
    )
    print(
        "Paper58 change-gate sweep complete: "
        f"cases={len(args.change_gate_dir)}, rows={len(rows)}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
