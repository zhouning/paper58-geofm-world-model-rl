from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / "paper" / "rse_submission_paper58" / "flus_real_datasets" / "TutorialData_DongGuan_80m"
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "rse_submission_paper58" / "flus_real_datasets" / "dongguan_80m_audit"
DEFAULT_SOURCE_ZIP = Path("/Users/zhouning/Downloads/1TutorialData_DongGuan_80m.zip")
LANDUSE_COLORS = {
    0: "#F7F7F7",
    1: "#D8C84A",
    2: "#2E7D32",
    3: "#8BC34A",
    4: "#1976D2",
    5: "#D95F02",
    6: "#9E9E9E",
}


@dataclass(frozen=True)
class RasterPayload:
    path: Path
    year: int
    array: np.ndarray
    summary: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_ready(row.get(field, "")) for field in fieldnames})


def _child_text(element: ET.Element, name: str) -> str | None:
    child = element.find(name)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def parse_landuse_classes(path: Path) -> dict[int, dict[str, Any]]:
    root = ET.parse(path).getroot()
    roles: dict[int, str] = {}
    role_sections = {
        "ConvertValues": "convertible",
        "NotToConvertValues": "fixed",
        "UrbanValues": "urban",
    }
    for section_name, role in role_sections.items():
        section = root.find(section_name)
        if section is None:
            continue
        for item in section.findall("StructLanduseInfo"):
            value = _child_text(item, "LanduseTypeValue")
            if value is not None:
                roles[int(value)] = role

    classes: dict[int, dict[str, Any]] = {}
    all_types = root.find("AllTypes")
    if all_types is None:
        return classes
    for item in all_types.findall("StructLanduseInfo"):
        value_text = _child_text(item, "LanduseTypeValue")
        if value_text is None:
            continue
        value = int(value_text)
        classes[value] = {
            "class_value": value,
            "class_name": _child_text(item, "LanduseTypeEnName") or f"class_{value}",
            "class_name_zh": _child_text(item, "LanduseTypeChsName") or "",
            "role": roles.get(value, "unknown"),
            "color_int": int(_child_text(item, "LanduseTypeColorIntValue") or 0),
        }
    return dict(sorted(classes.items()))


def parse_suitability_matrix(path: Path) -> list[dict[str, Any]]:
    root = ET.parse(path).getroot()
    rows: list[dict[str, Any]] = []
    for row_index, table in enumerate(root.findall("Table1"), start=1):
        values = {child.tag: int((child.text or "0").strip()) for child in list(table)}
        rows.append({"row_index": row_index, "values": values})
    return rows


def load_landuse_array(path: Path) -> np.ndarray:
    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            raise ValueError(f"expected one raster band in {path}, got {dataset.count}")
        return dataset.read(1)


def summarize_landuse_raster(
    path: Path,
    year: int,
    class_names: dict[int, str] | None = None,
    outside_values: set[int] | None = None,
) -> dict[str, Any]:
    names = class_names or {}
    outside = outside_values if outside_values is not None else {0}
    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            raise ValueError(f"expected one raster band in {path}, got {dataset.count}")
        array = dataset.read(1)
        total = int(array.size)
        valid_mask = ~np.isin(array, list(outside))
        valid_array = array[valid_mask]
        unique, counts = np.unique(valid_array, return_counts=True)
        valid_total = int(valid_array.size)
        class_counts = [
            {
                "class_value": int(value),
                "class_name": names.get(int(value), f"class_{int(value)}"),
                "pixels": int(count),
                "share": float(count / valid_total) if valid_total else 0.0,
            }
            for value, count in zip(unique, counts, strict=False)
        ]
        bounds = dataset.bounds
        transform = dataset.transform
        return {
            "path": Path(path),
            "year": int(year),
            "width": int(dataset.width),
            "height": int(dataset.height),
            "pixel_count": total,
            "valid_pixel_count": valid_total,
            "outside_pixel_count": int(total - valid_total),
            "outside_values": sorted(int(value) for value in outside),
            "dtype": str(array.dtype),
            "crs": str(dataset.crs) if dataset.crs else None,
            "bounds": [float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)],
            "transform": [float(value) for value in tuple(transform)[:6]],
            "resolution": [float(abs(transform.a)), float(abs(transform.e))],
            "class_counts": class_counts,
        }


def transition_count_rows(
    start: np.ndarray,
    end: np.ndarray,
    start_year: int,
    end_year: int,
    class_names: dict[int, str] | None = None,
    outside_values: set[int] | None = None,
) -> list[dict[str, Any]]:
    if start.shape != end.shape:
        raise ValueError(f"shape mismatch: {start.shape} vs {end.shape}")
    names = class_names or {}
    outside = outside_values if outside_values is not None else {0}
    valid_mask = ~np.isin(start, list(outside)) & ~np.isin(end, list(outside))
    valid_start = start[valid_mask]
    valid_end = end[valid_mask]
    counter: Counter[tuple[int, int]] = Counter(zip(valid_start.ravel().astype(int), valid_end.ravel().astype(int), strict=False))
    rows = []
    for from_class, to_class in sorted(counter):
        rows.append(
            {
                "period": f"{int(start_year)}_{int(end_year)}",
                "from_class": int(from_class),
                "from_name": names.get(int(from_class), f"class_{int(from_class)}"),
                "to_class": int(to_class),
                "to_name": names.get(int(to_class), f"class_{int(to_class)}"),
                "pixels": int(counter[(from_class, to_class)]),
                "changed": bool(from_class != to_class),
            }
        )
    return rows


def transition_summary(
    start: np.ndarray,
    end: np.ndarray,
    start_year: int,
    end_year: int,
    urban_class: int = 5,
    outside_values: set[int] | None = None,
) -> dict[str, Any]:
    if start.shape != end.shape:
        raise ValueError(f"shape mismatch: {start.shape} vs {end.shape}")
    outside = outside_values if outside_values is not None else {0}
    valid_mask = ~np.isin(start, list(outside)) & ~np.isin(end, list(outside))
    valid_start = start[valid_mask]
    valid_end = end[valid_mask]
    changed = valid_start != valid_end
    total = int(valid_start.size)
    to_urban = changed & (valid_end == urban_class)
    from_urban = changed & (valid_start == urban_class)
    return {
        "period": f"{int(start_year)}_{int(end_year)}",
        "start_year": int(start_year),
        "end_year": int(end_year),
        "n_pixels": total,
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_share": float(np.count_nonzero(changed) / total) if total else 0.0,
        "persistent_pixels": int(total - np.count_nonzero(changed)),
        "to_urban_pixels": int(np.count_nonzero(to_urban)),
        "from_urban_pixels": int(np.count_nonzero(from_urban)),
    }


def comparison_protocol(years: list[int]) -> dict[str, Any]:
    ordered = sorted(int(year) for year in years)
    if len(ordered) < 3:
        raise ValueError("at least three land-use years are required for calibration and validation")
    return {
        "calibration": {
            "start_year": ordered[0],
            "end_year": ordered[1],
            "role": "calibrate transition demand and suitability from real FLUS-style inputs",
        },
        "validation": {
            "start_year": ordered[1],
            "end_year": ordered[2],
            "role": "matched Paper58 vs GeoSOS-FLUS simulation target",
        },
    }


def _load_landuse_payloads(dataset_root: Path, classes: dict[int, dict[str, Any]]) -> list[RasterPayload]:
    class_names = {value: payload["class_name"] for value, payload in classes.items()}
    landuse_dir = dataset_root / "Landuse Data"
    paths = [
        (2000, landuse_dir / "landuse2000.tif"),
        (2005, landuse_dir / "landuse2005.tif"),
        (2006, landuse_dir / "landuse2006.tif"),
    ]
    payloads = []
    for year, path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing land-use raster: {path}")
        summary = summarize_landuse_raster(path, year=year, class_names=class_names)
        payloads.append(RasterPayload(path=path, year=year, array=load_landuse_array(path), summary=summary))
    shapes = {payload.array.shape for payload in payloads}
    if len(shapes) != 1:
        raise ValueError(f"land-use rasters are not on the same grid: {sorted(shapes)}")
    return payloads


def summarize_variable_grids(dataset_root: Path) -> list[dict[str, Any]]:
    variables_root = dataset_root / "Variables Data"
    rows = []
    if not variables_root.exists():
        return rows
    for path in sorted(variables_root.iterdir()):
        if not path.is_dir() or path.name == "info":
            continue
        row: dict[str, Any] = {"name": path.name, "path": path, "readable": False}
        try:
            with rasterio.open(path) as dataset:
                array = dataset.read(1, masked=True)
                compressed = array.compressed()
                row.update(
                    {
                        "readable": True,
                        "width": int(dataset.width),
                        "height": int(dataset.height),
                        "dtype": str(array.dtype),
                        "crs": str(dataset.crs) if dataset.crs else None,
                        "min": float(compressed.min()) if compressed.size else None,
                        "max": float(compressed.max()) if compressed.size else None,
                        "mean": float(compressed.mean()) if compressed.size else None,
                    }
                )
        except Exception as exc:  # pragma: no cover - depends on GDAL ArcInfo Grid support
            row["error"] = str(exc)
        rows.append(row)
    return rows


def _indexed_classes(array: np.ndarray, classes: list[int]) -> np.ndarray:
    index = np.zeros(array.shape, dtype=np.int16)
    class_to_index = {int(cls): offset for offset, cls in enumerate(classes)}
    for cls, offset in class_to_index.items():
        index[array == cls] = offset
    return index


def _save_landuse_map(
    array: np.ndarray,
    year: int,
    class_names: dict[int, str],
    figure_dir: Path,
    title_prefix: str = "Dongguan FLUS tutorial land use",
) -> str:
    classes = sorted({int(value) for value in np.unique(array)})
    colors = [LANDUSE_COLORS.get(cls, "#CCCCCC") for cls in classes]
    indexed = _indexed_classes(array, classes)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.imshow(indexed, cmap=ListedColormap(colors), interpolation="nearest")
    ax.set_title(f"{title_prefix} {year}", loc="left", fontweight="bold")
    ax.set_axis_off()
    patches = [
        Patch(color=color, label=f"{cls} {'outside boundary' if cls == 0 else class_names.get(cls, f'class_{cls}')}")
        for cls, color in zip(classes, colors, strict=False)
    ]
    ax.legend(handles=patches, loc="lower left", bbox_to_anchor=(0, -0.18), ncol=3, fontsize=7, frameon=False)
    fig.tight_layout()
    path = figure_dir / f"landuse_{year}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _save_change_map(start: np.ndarray, end: np.ndarray, start_year: int, end_year: int, figure_dir: Path) -> str:
    code = np.full(start.shape, 4, dtype=np.uint8)
    valid = (start != 0) & (end != 0)
    changed = (start != end) & valid
    code[valid] = 0
    code[changed] = 3
    code[changed & (end == 5)] = 1
    code[changed & (start == 5)] = 2
    labels = {
        0: "persistent",
        1: "changed to construction",
        2: "changed from construction",
        3: "other transition",
        4: "outside boundary",
    }
    colors = ["#E5E5E5", "#D95F02", "#7B3294", "#3B8C6E", "#FFFFFF"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.imshow(code, cmap=ListedColormap(colors), interpolation="nearest", vmin=0, vmax=4)
    ax.set_title(f"Observed transition map {start_year}-{end_year}", loc="left", fontweight="bold")
    ax.set_axis_off()
    patches = [Patch(color=colors[key], label=labels[key]) for key in sorted(labels)]
    ax.legend(handles=patches, loc="lower left", bbox_to_anchor=(0, -0.15), ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    path = figure_dir / f"change_{start_year}_{end_year}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _save_class_count_chart(summaries: list[dict[str, Any]], figure_dir: Path) -> str:
    classes = sorted({int(row["class_value"]) for summary in summaries for row in summary["class_counts"]})
    years = [int(summary["year"]) for summary in summaries]
    values_by_class = {cls: [] for cls in classes}
    for summary in summaries:
        counts = {int(row["class_value"]): int(row["pixels"]) for row in summary["class_counts"]}
        for cls in classes:
            values_by_class[cls].append(counts.get(cls, 0))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bottom = np.zeros(len(years), dtype=float)
    for cls in classes:
        values = np.asarray(values_by_class[cls], dtype=float)
        ax.bar([str(year) for year in years], values, bottom=bottom, color=LANDUSE_COLORS.get(cls, "#CCCCCC"), label=str(cls))
        bottom += values
    ax.set_ylabel("Pixels")
    ax.set_title("Class composition by year", loc="left", fontweight="bold")
    ax.legend(title="Class", ncol=len(classes), fontsize=7, title_fontsize=7, frameon=False)
    fig.tight_layout()
    path = figure_dir / "class_counts_by_year.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _save_change_summary_chart(rows: list[dict[str, Any]], figure_dir: Path) -> str:
    periods = [row["period"] for row in rows]
    changed = [float(row["changed_share"]) * 100.0 for row in rows]
    to_urban = [int(row["to_urban_pixels"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].bar(periods, changed, color="#3B8C6E")
    axes[0].set_ylabel("Changed pixels (%)")
    axes[0].set_title("Observed change intensity", loc="left", fontweight="bold", fontsize=9)
    axes[1].bar(periods, to_urban, color="#D95F02")
    axes[1].set_ylabel("Pixels")
    axes[1].set_title("Transitions to construction", loc="left", fontweight="bold", fontsize=9)
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = figure_dir / "change_summary.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _write_report(
    output_dir: Path,
    audit: dict[str, Any],
    landuse_figures: list[str],
    change_figures: list[str],
    chart_figures: list[str],
) -> Path:
    lines = [
        "# FLUS Real Dataset Audit: Dongguan 80m",
        "",
        f"- Source zip: `{audit['source_zip']}`",
        f"- Dataset root: `{audit['dataset_root']}`",
        "- Status: usable as a real FLUS-style benchmark dataset.",
        "- Proposed protocol: calibrate on 2000->2005, compare Paper58 and GeoSOS-FLUS on 2005->2006.",
        "",
        "## Dataset Structure",
        "",
        f"- Land-use rasters: {len(audit['landuse_maps'])} years on one grid.",
        f"- Grid: {audit['landuse_maps'][0]['width']} x {audit['landuse_maps'][0]['height']} pixels.",
        f"- Valid in-boundary pixels: {audit['landuse_maps'][0]['valid_pixel_count']:,}.",
        f"- Outside-boundary pixels excluded from metrics: {audit['landuse_maps'][0]['outside_pixel_count']:,}.",
        f"- Resolution: {audit['landuse_maps'][0]['resolution'][0]:.1f} m.",
        f"- Classes: {len(audit['classes'])} FLUS land-use types.",
        f"- Driving variables: {len(audit['variables'])} ArcInfo Grid distance variables.",
        "",
        "## Raw Land-Use Inputs",
        "",
    ]
    for figure in landuse_figures:
        lines.append(f"![{figure}](figures/{figure})")
        lines.append("")
    lines.extend(
        [
            "## Observed Change Targets",
            "",
        ]
    )
    for figure in change_figures:
        lines.append(f"![{figure}](figures/{figure})")
        lines.append("")
    lines.extend(
        [
            "## Dataset Metrics",
            "",
        ]
    )
    for figure in chart_figures:
        lines.append(f"![{figure}](figures/{figure})")
        lines.append("")
    lines.extend(
        [
            "## Output Tables",
            "",
            "- `audit.json`: complete machine-readable audit.",
            "- `class_counts.csv`: class counts by year.",
            "- `transition_counts.csv`: observed class-to-class transition matrix rows.",
            "- `transition_summary.csv`: change intensity and construction transitions.",
            "",
        ]
    )
    path = output_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def audit_dataset(dataset_root: Path, output_dir: Path, source_zip: Path | None = None) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    classes = parse_landuse_classes(dataset_root / "Config Files" / "DefaultLanduseInfo.xml")
    class_names = {value: payload["class_name"] for value, payload in classes.items()}
    payloads = _load_landuse_payloads(dataset_root, classes)
    transition_rows: list[dict[str, Any]] = []
    transition_summaries: list[dict[str, Any]] = []
    change_figures: list[str] = []
    for previous, current in zip(payloads, payloads[1:], strict=False):
        transition_rows.extend(
            transition_count_rows(previous.array, current.array, previous.year, current.year, class_names=class_names)
        )
        transition_summaries.append(transition_summary(previous.array, current.array, previous.year, current.year))
        change_figures.append(_save_change_map(previous.array, current.array, previous.year, current.year, figure_dir))

    landuse_figures = [_save_landuse_map(payload.array, payload.year, class_names, figure_dir) for payload in payloads]
    chart_figures = [
        _save_class_count_chart([payload.summary for payload in payloads], figure_dir),
        _save_change_summary_chart(transition_summaries, figure_dir),
    ]
    class_count_rows = [
        {"year": payload.year, **row}
        for payload in payloads
        for row in payload.summary["class_counts"]
    ]
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": "TutorialData_DongGuan_80m",
        "source_zip": source_zip or DEFAULT_SOURCE_ZIP,
        "dataset_root": dataset_root,
        "classes": list(classes.values()),
        "suitability_matrix": parse_suitability_matrix(dataset_root / "Config Files" / "SuitableMatrix.xml"),
        "landuse_maps": [payload.summary for payload in payloads],
        "transitions": transition_summaries,
        "variables": summarize_variable_grids(dataset_root),
        "comparison_protocol": comparison_protocol([payload.year for payload in payloads]),
        "figures": {
            "landuse": [f"figures/{figure}" for figure in landuse_figures],
            "change": [f"figures/{figure}" for figure in change_figures],
            "charts": [f"figures/{figure}" for figure in chart_figures],
        },
    }
    _write_json(output_dir / "audit.json", audit)
    _write_csv(
        output_dir / "class_counts.csv",
        class_count_rows,
        ["year", "class_value", "class_name", "pixels", "share"],
    )
    _write_csv(
        output_dir / "transition_counts.csv",
        transition_rows,
        ["period", "from_class", "from_name", "to_class", "to_name", "pixels", "changed"],
    )
    _write_csv(
        output_dir / "transition_summary.csv",
        transition_summaries,
        [
            "period",
            "start_year",
            "end_year",
            "n_pixels",
            "changed_pixels",
            "changed_share",
            "persistent_pixels",
            "to_urban_pixels",
            "from_urban_pixels",
        ],
    )
    _write_report(output_dir, _json_ready(audit), landuse_figures, change_figures, chart_figures)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit real FLUS-style land-use data for Paper58 comparison.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-zip", type=Path, default=DEFAULT_SOURCE_ZIP)
    args = parser.parse_args(argv)
    audit_dataset(dataset_root=args.dataset_root, output_dir=args.output_dir, source_zip=args.source_zip)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
