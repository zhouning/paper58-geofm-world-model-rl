from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from scripts.paper58_benchmark.evaluate_benchmark import _read_registry
from scripts.paper58_benchmark.evaluate_las import _path
from scripts.paper58_benchmark.flus import load_flus_prediction


BLUE = "#2C7FB8"
GRAY = "#8A8F93"
RED = "#B24C4A"
GREEN = "#3B8C6E"
GOLD = "#B48A2C"
INK = "#222222"
LULC_COLORS = {
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
METRICS = [
    ("change_f1", "Change F1"),
    ("fom", "FoM"),
    ("transition_accuracy", "Transition acc."),
    ("allocation_disagreement", "Allocation disagr."),
]


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.frameon": False,
        }
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _float(row: dict, key: str) -> float:
    return float(row[key])


def _class_index_array(array: np.ndarray, class_to_index: dict[int, int]) -> np.ndarray:
    indexed = np.zeros(array.shape, dtype=int)
    for cls, index in class_to_index.items():
        indexed[np.asarray(array) == cls] = index
    return indexed


def _lulc_cmap(arrays: list[np.ndarray]) -> tuple[ListedColormap, dict[int, int]]:
    classes = sorted({int(value) for array in arrays for value in np.unique(np.asarray(array))})
    class_to_index = {cls: index for index, cls in enumerate(classes)}
    colors = [LULC_COLORS.get(cls, "#D0D0D0") for cls in classes]
    return ListedColormap(colors), class_to_index


def _save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _simulated_map_path(result_dir: Path, row: dict) -> Path:
    area = str(row.get("area"))
    start_year = int(row.get("start_year"))
    end_year = int(row.get("end_year"))
    return Path(result_dir) / "simulated" / f"{area}_{start_year}_{end_year}_paper58_las.npy"


def _load_result_map(result_dir: Path, row: dict) -> np.ndarray:
    path = _simulated_map_path(result_dir, row)
    if not path.exists():
        raise FileNotFoundError(f"missing LAS simulated map: {path}")
    return np.load(path).astype(np.int32, copy=False)


def _change_error_map(start: np.ndarray, end: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    true_change = np.asarray(end) != np.asarray(start)
    pred_change = np.asarray(prediction) != np.asarray(start)
    error = np.zeros(np.asarray(start).shape, dtype=np.int8)
    error[true_change & pred_change] = 1
    error[true_change & ~pred_change] = 2
    error[~true_change & pred_change] = 3
    return error


def _make_metric_advantage_figure(summary: dict, figure_dir: Path) -> str:
    advantages = summary.get("advantages", {})
    labels: list[str] = []
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for metric, label in METRICS:
        payload = advantages.get(metric)
        if not payload:
            continue
        ci = payload.get("bootstrap_ci", {})
        mean = float(payload["mean_advantage"])
        labels.append(label)
        means.append(mean)
        lows.append(float(ci.get("ci_low", mean)))
        highs.append(float(ci.get("ci_high", mean)))

    y = np.arange(len(labels))
    left = [mean - low for mean, low in zip(means, lows, strict=False)]
    right = [high - mean for mean, high in zip(means, highs, strict=False)]
    colors = [GREEN if value >= 0 else RED for value in means]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.axvline(0.0, color=INK, linewidth=0.8)
    ax.barh(y, means, color=colors, height=0.52)
    ax.errorbar(means, y, xerr=[left, right], fmt="none", color=INK, linewidth=0.8, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Paper58-LAS advantage over GeoSOS-FLUS")
    ax.set_title("Matched metric advantages with bootstrap intervals", loc="left", fontweight="bold")
    for yi, value in zip(y, means, strict=False):
        ax.text(value, yi + 0.26, f"{value:+.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    _save_figure(fig, figure_dir, "fig1_metric_advantages")
    return "figures/fig1_metric_advantages.png"


def _make_area_advantage_figure(rows: list[dict[str, str]], figure_dir: Path) -> str:
    labels = [str(row["area"]).replace("_holdout", "").replace("_", "\n") for row in rows]
    x = np.arange(len(rows))
    width = 0.24
    series = [
        ("change_f1_advantage", "F1", BLUE),
        ("fom_advantage", "FoM", GREEN),
        ("allocation_disagreement_advantage", "Alloc. disagr.", RED),
    ]
    fig, ax = plt.subplots(figsize=(max(7.2, len(rows) * 0.95), 3.6))
    for index, (key, label, color) in enumerate(series):
        values = [_float(row, key) for row in rows]
        ax.bar(x + (index - 1) * width, values, width=width, color=color, label=label)
    ax.axhline(0.0, color=INK, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=6)
    ax.set_ylabel("Advantage")
    ax.set_title("Area-level advantages", loc="left", fontweight="bold")
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    _save_figure(fig, figure_dir, "fig2_area_advantages")
    return "figures/fig2_area_advantages.png"


def _make_spatial_panel(
    row: dict,
    primary_result_dir: Path,
    figure_dir: Path,
    extra_result_dirs: dict[str, Path],
) -> str:
    start = np.load(_path(row.get("label_start_path"))).astype(np.int32, copy=False)
    end = np.load(_path(row.get("label_end_path"))).astype(np.int32, copy=False)
    direct = np.load(_path(row.get("prediction_path"))).astype(np.int32, copy=False)
    class_values = sorted({int(value) for array in (start, end, direct) for value in np.unique(array)})
    flus_path = _path(row.get("flus_prediction_path"))
    if flus_path is None:
        raise FileNotFoundError(f"missing FLUS prediction path for {row.get('area')}")
    flus = load_flus_prediction(flus_path, expected_shape=start.shape, allowed_classes=set(class_values))
    las = _load_result_map(primary_result_dir, row)
    extra_maps = {name: _load_result_map(path, row) for name, path in extra_result_dirs.items()}

    map_arrays = [start, end, direct, flus, las, *extra_maps.values()]
    cmap, class_to_index = _lulc_cmap(map_arrays)
    map_panels = [
        ("Reference start", start),
        ("Observed end", end),
        ("Paper58 direct", direct),
        ("GeoSOS-FLUS", flus),
        ("Paper58-LAS", las),
        *[(name, array) for name, array in extra_maps.items()],
    ]
    error_panels = [
        ("Reference change", end != start, "change"),
        ("Direct error", _change_error_map(start, end, direct), "error"),
        ("GeoSOS-FLUS error", _change_error_map(start, end, flus), "error"),
        ("Paper58-LAS error", _change_error_map(start, end, las), "error"),
        *[(f"{name} error", _change_error_map(start, end, array), "error") for name, array in extra_maps.items()],
    ]
    ncols = max(len(map_panels), len(error_panels))
    fig, axes = plt.subplots(2, ncols, figsize=(max(9.2, ncols * 1.35), 4.2), constrained_layout=True)
    if ncols == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    for index, (title, array) in enumerate(map_panels):
        ax = axes[0, index]
        ax.imshow(_class_index_array(array, class_to_index), cmap=cmap, interpolation="nearest")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=7)
        ax.set_frame_on(True)

    change_cmap = ListedColormap(["#F0F0F0", BLUE])
    error_cmap = ListedColormap(["#F0F0F0", GREEN, RED, GOLD])
    for index, (title, array, kind) in enumerate(error_panels):
        ax = axes[1, index]
        if kind == "change":
            ax.imshow(array, cmap=change_cmap, interpolation="nearest", vmin=0, vmax=1)
        else:
            ax.imshow(array, cmap=error_cmap, interpolation="nearest", vmin=0, vmax=3)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=7)
        ax.set_frame_on(True)

    area = str(row.get("area"))
    years = f"{row.get('start_year')}-{row.get('end_year')}"
    fig.suptitle(f"{area} {years}: raw maps, model outputs, and change errors", fontsize=9, fontweight="bold")
    class_handles = [Patch(color=LULC_COLORS.get(cls, "#D0D0D0"), label=f"class {cls}") for cls in sorted(class_to_index)]
    error_handles = [
        Patch(color=BLUE, label="true change"),
        Patch(color=GREEN, label="hit"),
        Patch(color=RED, label="miss"),
        Patch(color=GOLD, label="false alarm"),
    ]
    fig.legend(
        handles=class_handles + error_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=min(10, len(class_handles) + len(error_handles)),
        fontsize=6,
    )
    stem = f"spatial_{_slug(area)}"
    _save_figure(fig, figure_dir, stem)
    return f"figures/{stem}.png"


def _copy_metric_rows_for_markdown(rows: list[dict[str, str]]) -> str:
    fields = [
        ("area", "Area"),
        ("change_f1_advantage", "F1 adv."),
        ("fom_advantage", "FoM adv."),
        ("transition_accuracy_advantage", "Trans. adv."),
        ("allocation_disagreement_advantage", "Alloc. adv."),
    ]
    lines = ["| " + " | ".join(label for _, label in fields) + " |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        values = []
        for key, _ in fields:
            if key == "area":
                values.append(f"`{row[key]}`")
            else:
                values.append(f"{float(row[key]):+.4f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_readme(
    output_dir: Path,
    title: str,
    result_dir: Path,
    metric_figure: str,
    area_figure: str,
    spatial_figures: list[tuple[str, str]],
    comparison_rows: list[dict[str, str]],
) -> None:
    content = [
        f"# {title}",
        "",
        "This report combines metric comparison charts with raw spatial evidence panels.",
        "",
        f"- Primary result directory: `{result_dir}`",
        "- Spatial panels show reference start, observed end, Paper58 direct prediction, GeoSOS-FLUS output, Paper58-LAS output, and change-error maps.",
        "- Error-map colors: green = change hit, red = missed observed change, gold = false alarm.",
        "",
        "## Metric Summary",
        "",
        f"![Metric advantages]({metric_figure})",
        "",
        f"![Area advantages]({area_figure})",
        "",
        _copy_metric_rows_for_markdown(comparison_rows),
        "",
        "## Raw And Result Spatial Panels",
        "",
    ]
    for area, figure in spatial_figures:
        content.extend([f"### `{area}`", "", f"![{area}]({figure})", ""])
    (output_dir / "README.md").write_text("\n".join(content), encoding="utf-8")


def _write_html(
    output_dir: Path,
    title: str,
    metric_figure: str,
    area_figure: str,
    spatial_figures: list[tuple[str, str]],
) -> None:
    figure_cards = "\n".join(
        f'<section class="card"><h2>{html.escape(area)}</h2><img src="{html.escape(figure)}" alt="{html.escape(area)}"></section>'
        for area, figure in spatial_figures
    )
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: #1f2328; background: #f6f7f8; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    h2 {{ font-size: 18px; margin: 0 0 12px; }}
    p {{ line-height: 1.5; }}
    .card {{ background: white; border: 1px solid #d7dce0; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    img {{ width: 100%; height: auto; display: block; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <p>Metric charts plus raw/reference/result spatial panels. Green error pixels are hits, red are missed observed changes, and gold are false alarms.</p>
  <section class="card"><h2>Metric Advantages</h2><img src="{html.escape(metric_figure)}" alt="Metric advantages"></section>
  <section class="card"><h2>Area Advantages</h2><img src="{html.escape(area_figure)}" alt="Area advantages"></section>
  <div class="grid">
    {figure_cards}
  </div>
</main>
</body>
</html>
"""
    (output_dir / "report.html").write_text(content, encoding="utf-8")


def make_las_visual_report(
    registry_path: Path,
    las_result_dir: Path,
    output_dir: Path,
    title: str = "Paper58-LAS vs GeoSOS-FLUS Visual Report",
    extra_result_dirs: dict[str, Path] | None = None,
    max_areas: int | None = None,
) -> dict:
    _apply_style()
    registry_rows = [row for row in _read_registry(Path(registry_path)) if row.get("qc_status") == "include"]
    if max_areas is not None:
        registry_rows = registry_rows[: int(max_areas)]
    output = Path(output_dir)
    figure_dir = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    result_dir = Path(las_result_dir)
    comparison_dir = result_dir / "comparison_vs_flus"
    summary = _load_json(comparison_dir / "las_comparison_summary.json")
    comparison_rows = _read_csv_rows(comparison_dir / "las_comparison_by_area.csv")
    metric_figure = _make_metric_advantage_figure(summary, figure_dir)
    area_figure = _make_area_advantage_figure(comparison_rows, figure_dir)

    extras = {str(name): Path(path) for name, path in (extra_result_dirs or {}).items()}
    spatial_figures = [
        (str(row.get("area")), _make_spatial_panel(row, result_dir, figure_dir, extras))
        for row in registry_rows
    ]
    _write_readme(output, title, result_dir, metric_figure, area_figure, spatial_figures, comparison_rows)
    _write_html(output, title, metric_figure, area_figure, spatial_figures)
    return {
        "output_dir": str(output),
        "figures": [metric_figure, area_figure, *[figure for _, figure in spatial_figures]],
        "n_spatial_panels": len(spatial_figures),
    }


def _parse_extra_result(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("extra result must be NAME=DIR")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("extra result name cannot be empty")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a visual Paper58-LAS vs GeoSOS-FLUS report.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--las-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Paper58-LAS vs GeoSOS-FLUS Visual Report")
    parser.add_argument("--extra-result", action="append", type=_parse_extra_result, default=[])
    parser.add_argument("--max-areas", type=int, default=None)
    args = parser.parse_args()
    result = make_las_visual_report(
        registry_path=args.registry,
        las_result_dir=args.las_result_dir,
        output_dir=args.output_dir,
        title=args.title,
        extra_result_dirs=dict(args.extra_result),
        max_areas=args.max_areas,
    )
    print(
        "LAS visual report: "
        f"{result['n_spatial_panels']} spatial panel(s), output={result['output_dir']}"
    )


if __name__ == "__main__":
    main()
