from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "paper" / "rse_submission_paper58" / "revision_results"
DEFAULT_FIGURE_DIR = ROOT / "paper" / "rse_submission_paper58" / "figures"
DEFAULT_DECODER_RESULTS = ROOT / "src" / "adk_world_model" / "experiments" / "output" / "world_model_lulc_decode.json"
DEFAULT_LABELS_DIR = ROOT / "data" / "independent_change_labels" / "labels"
DEFAULT_PREDICTIONS_DIR = ROOT / "data" / "independent_change_labels" / "predicted"

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


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.frameon": False,
        }
    )


def _read_csv_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {"area": row["area"]}
            for key in ("persistence", "model", "advantage", "change_pixel_advantage"):
                parsed[key] = float(row[key]) if row.get(key) not in (None, "") else None
            rows.append(parsed)
    return rows


def _read_generic_csv_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                if value in (None, ""):
                    parsed[key] = None
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def load_revision_inputs(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    decoder_path: Path = DEFAULT_DECODER_RESULTS,
) -> dict:
    summary_path = results_dir / "revision_summary.json"
    alpha_path = results_dir / "alphaearth_area_metrics.csv"
    prithvi_path = results_dir / "prithvi_area_metrics.csv"
    category_path = results_dir / "alphaearth_category_summary.csv"
    planning_path = results_dir / "planning_baseline_summary.csv"
    transfer_path = results_dir / "transfer_planning_summary.csv"
    missing = [
        str(p)
        for p in (summary_path, alpha_path, prithvi_path, category_path, planning_path, transfer_path, decoder_path)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing revision result files: " + ", ".join(missing))
    return {
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "alphaearth_rows": _read_csv_rows(alpha_path),
        "prithvi_rows": _read_csv_rows(prithvi_path),
        "category_rows": _read_generic_csv_rows(category_path),
        "planning_baseline_rows": _read_generic_csv_rows(planning_path),
        "transfer_planning_rows": _read_generic_csv_rows(transfer_path),
        "decoder": json.loads(decoder_path.read_text(encoding="utf-8")),
    }


def load_spatial_change_case(
    labels_dir: Path = DEFAULT_LABELS_DIR,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
    area: str = "banzhucun",
    start_year: int = 2020,
    end_year: int = 2023,
) -> dict:
    labels_dir = Path(labels_dir)
    predictions_dir = Path(predictions_dir)
    start = np.load(labels_dir / f"{area}_lulc_{start_year}.npy")
    end = np.load(labels_dir / f"{area}_lulc_{end_year}.npy")
    pred = np.load(predictions_dir / f"{area}_lulc_pred_{start_year}_{end_year}.npy")
    if start.shape != end.shape or start.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch for {area} {start_year}-{end_year}: "
            f"start={start.shape}, end={end.shape}, pred={pred.shape}"
        )

    true_change = end != start
    model_change = pred != start
    return {
        "area": area,
        "start_year": start_year,
        "end_year": end_year,
        "start": start,
        "end": end,
        "pred": pred,
        "true_change": true_change,
        "model_change": model_change,
        "change_hit": true_change & model_change,
        "change_miss": true_change & ~model_change,
        "false_alarm": ~true_change & model_change,
    }


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_per_area_performance(data: dict, out_dir: Path) -> None:
    rows = sorted(data["alphaearth_rows"], key=lambda r: r["advantage"])
    labels = [r["area"].replace("_", " ") for r in rows]
    x = np.arange(len(rows))
    persistence = np.array([r["persistence"] for r in rows])
    model = np.array([r["model"] for r in rows])
    advantage = np.array([r["advantage"] for r in rows])

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.0),
        gridspec_kw={"height_ratios": [2.2, 1.1], "hspace": 0.36},
        sharex=True,
    )
    width = 0.36
    ax_top.bar(x - width / 2, persistence, width, color="#C9CED2", label="Persistence")
    ax_top.bar(x + width / 2, model, width, color=BLUE, label="LatentDynamicsNet")
    ax_top.set_ylabel("Cosine similarity")
    ax_top.set_ylim(max(0.93, float(min(persistence.min(), model.min()) - 0.01)), 1.0)
    ax_top.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax_top.text(
        0.0,
        1.04,
        "a  Area-level prediction quality from valid cached AlphaEarth grids",
        transform=ax_top.transAxes,
        fontweight="bold",
        va="bottom",
    )

    colors = [GREEN if v > 0 else RED for v in advantage]
    ax_bottom.bar(x, advantage, color=colors, width=0.58)
    ax_bottom.axhline(0, color=INK, linewidth=0.8)
    ax_bottom.set_ylabel("Model - persistence")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(labels, rotation=35, ha="right")
    ax_bottom.text(
        0.0,
        1.05,
        "b  Paired advantage by area",
        transform=ax_bottom.transAxes,
        fontweight="bold",
        va="bottom",
    )

    summary = data["summary"]["alphaearth"]
    ci = summary["advantage"]
    y_min = min(float(advantage.min()) - 0.004, -0.022)
    y_max = max(float(advantage.max()) + 0.008, 0.034)
    ax_bottom.set_ylim(y_min, y_max)
    ax_bottom.text(
        0.02,
        0.94,
        f"Mean advantage {ci['mean']:+.4f}\n95% CI [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]\n"
        f"{summary['advantage_sign_test']['n_positive']}/{summary['n_areas']} areas positive",
        transform=ax_bottom.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    save_figure(fig, out_dir, "fig_rse_revision_area_performance")


def make_encoder_diagnostic(data: dict, out_dir: Path) -> None:
    summary = data["summary"]
    labels = ["AlphaEarth", "Prithvi-100M CLS"]
    persistence = [
        summary["alphaearth"]["persistence"]["mean"],
        summary["prithvi"]["persistence"]["mean"],
    ]
    model = [
        summary["alphaearth"]["model"]["mean"],
        summary["prithvi"]["model"]["mean"],
    ]
    advantage = [
        summary["alphaearth"]["advantage"]["mean"],
        summary["prithvi"]["advantage"]["mean"],
    ]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.34})
    x = np.arange(2)
    width = 0.32
    ax_left.bar(x - width / 2, persistence, width, color="#C9CED2", label="Persistence")
    ax_left.bar(x + width / 2, model, width, color=BLUE, label="LatentDynamicsNet")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)
    ax_left.set_ylabel("Mean cosine similarity")
    ax_left.set_ylim(0.96, 1.002)
    ax_left.legend(loc="lower right")
    ax_left.text(0, 1.04, "a  Encoder-state temporal structure", transform=ax_left.transAxes, fontweight="bold")

    ax_right.axhline(0, color=INK, linewidth=0.8)
    ax_right.bar(x, advantage, color=[GREEN, RED], width=0.5)
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(labels)
    ax_right.set_ylabel("Mean advantage")
    ax_right.set_ylabel("Mean advantage")
    ax_right.set_yticks([-0.000002, 0.0, 0.0025, 0.005])
    ax_right.set_yticklabels(["-2e-6", "0", "0.0025", "0.0050"])
    ax_right.text(0, 1.04, "b  Advantage over persistence", transform=ax_right.transAxes, fontweight="bold")
    ax_right.text(
        0.97,
        0.1,
        "Prithvi CLS is nearly stationary\nacross years in this cache",
        transform=ax_right.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    save_figure(fig, out_dir, "fig_rse_revision_encoder_diagnostic")


def make_decoder_confusion(data: dict, out_dir: Path) -> None:
    decoder = data["decoder"]
    cm = np.array(decoder["confusion_matrix"], dtype=float)
    classes = [str(c) for c in decoder["classes"]]
    row_totals = cm.sum(axis=1, keepdims=True)
    row_norm = np.divide(cm, row_totals, out=np.zeros_like(cm), where=row_totals > 0)
    macro_f1 = decoder.get("classification_report", {}).get("macro avg", {}).get("f1-score")

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted class code")
    ax.set_ylabel("Reference class code")
    ax.text(
        0.0,
        1.08,
        "LULC decoder cross-validation",
        transform=ax.transAxes,
        fontweight="bold",
        va="bottom",
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue
            color = "white" if row_norm[i, j] > 0.55 else INK
            ax.text(j, i, f"{row_norm[i, j] * 100:.0f}\n({int(cm[i, j])})", ha="center", va="center", color=color, fontsize=6)

    summary = f"n={decoder['n_samples']:,}; OA={decoder['overall_accuracy']:.3f}"
    if macro_f1 is not None:
        summary += f"; macro-F1={macro_f1:.3f}"
    ax.text(0.0, -0.24, summary, transform=ax.transAxes, ha="left", va="top", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized proportion")
    save_figure(fig, out_dir, "fig_rse_revision_decoder_confusion")


def _lulc_cmap_and_norm(arrays: list[np.ndarray]) -> tuple[ListedColormap, dict[int, int]]:
    classes = sorted({int(value) for arr in arrays for value in np.unique(arr)})
    colors = [LULC_COLORS.get(cls, "#D0D0D0") for cls in classes]
    return ListedColormap(colors), {cls: index for index, cls in enumerate(classes)}


def _class_index_array(array: np.ndarray, class_to_index: dict[int, int]) -> np.ndarray:
    indexed = np.zeros(array.shape, dtype=int)
    for cls, index in class_to_index.items():
        indexed[array == cls] = index
    return indexed


def make_spatial_change_validation(
    out_dir: Path,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
    area: str = "banzhucun",
    start_year: int = 2020,
    end_year: int = 2023,
) -> None:
    case = load_spatial_change_case(labels_dir, predictions_dir, area, start_year, end_year)
    cmap, class_to_index = _lulc_cmap_and_norm([case["start"], case["end"], case["pred"]])

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), constrained_layout=True)
    map_panels = [
        ("a  Reference start", case["start"]),
        ("b  Reference end", case["end"]),
        ("c  Model prediction", case["pred"]),
    ]
    for ax, (title, array) in zip(axes[0], map_panels):
        ax.imshow(_class_index_array(array, class_to_index), cmap=cmap, interpolation="nearest")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1, 0].imshow(case["true_change"], cmap=ListedColormap(["#F0F0F0", BLUE]), interpolation="nearest")
    axes[1, 0].set_title("d  Reference change", loc="left", fontweight="bold", fontsize=8)
    axes[1, 1].imshow(case["model_change"], cmap=ListedColormap(["#F0F0F0", GOLD]), interpolation="nearest")
    axes[1, 1].set_title("e  Predicted change", loc="left", fontweight="bold", fontsize=8)

    error = np.zeros(case["start"].shape, dtype=int)
    error[case["change_hit"]] = 1
    error[case["change_miss"]] = 2
    error[case["false_alarm"]] = 3
    axes[1, 2].imshow(error, cmap=ListedColormap(["#F0F0F0", GREEN, RED, GOLD]), interpolation="nearest")
    axes[1, 2].set_title("f  Change-detection error", loc="left", fontweight="bold", fontsize=8)
    for ax in axes[1]:
        ax.set_xticks([])
        ax.set_yticks([])

    true_change = int(np.count_nonzero(case["true_change"]))
    hit = int(np.count_nonzero(case["change_hit"]))
    miss = int(np.count_nonzero(case["change_miss"]))
    false_alarm = int(np.count_nonzero(case["false_alarm"]))
    fig.suptitle(
        f"{area.replace('_', ' ').title()} {start_year}-{end_year}: "
        f"true change={true_change:,}, hit={hit:,}, miss={miss:,}, false alarm={false_alarm:,}",
        fontsize=9,
        fontweight="bold",
    )
    lulc_handles = [
        Patch(color=LULC_COLORS.get(cls, "#D0D0D0"), label=f"class {cls}")
        for cls in sorted(class_to_index)
    ]
    error_handles = [
        Patch(color=GREEN, label="change hit"),
        Patch(color=RED, label="miss"),
        Patch(color=GOLD, label="false alarm"),
    ]
    fig.legend(
        handles=lulc_handles + error_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=min(8, len(lulc_handles) + len(error_handles)),
        fontsize=6,
    )
    save_figure(fig, out_dir, "fig_rse_revision_spatial_change_validation")


def make_planning_dropout(data: dict, out_dir: Path) -> None:
    stats = data["summary"].get("planning_dropout_statistics", {})
    configs = stats.get("configs", {})
    order = ["full", "dropout0.3", "dropout1.0"]
    labels = ["Full features", "30% feature dropout", "Embedding only"]
    slope_mean = np.array([configs[k]["slope_mean"] for k in order])
    slope_std = np.array([configs[k]["slope_std"] for k in order])
    cont_mean = np.array([configs[k]["cont_mean"] for k in order])
    cont_std = np.array([configs[k]["cont_std"] for k in order])
    n = np.array([configs[k]["n"] for k in order], dtype=float)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.34})
    x = np.arange(len(order))
    ax_left.errorbar(x, slope_mean, yerr=1.96 * slope_std / np.sqrt(n), fmt="o", color=BLUE, capsize=3)
    ax_left.axhline(0, color=INK, linewidth=0.7)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, rotation=20, ha="right")
    ax_left.set_ylabel("Slope change (%)")
    ax_left.text(0, 1.04, "a  Terrain-slope objective", transform=ax_left.transAxes, fontweight="bold")

    ax_right.errorbar(x, cont_mean, yerr=1.96 * cont_std / np.sqrt(n), fmt="o", color=GOLD, capsize=3)
    ax_right.axhline(0, color=INK, linewidth=0.7)
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(labels, rotation=20, ha="right")
    ax_right.set_ylabel("Contiguity change")
    ax_right.text(0, 1.04, "b  Spatial-contiguity objective", transform=ax_right.transAxes, fontweight="bold")
    save_figure(fig, out_dir, "fig_rse_revision_planning_dropout")


def make_all_figures(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    out_dir: Path = DEFAULT_FIGURE_DIR,
    decoder_path: Path = DEFAULT_DECODER_RESULTS,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
) -> None:
    apply_style()
    data = load_revision_inputs(results_dir, decoder_path)
    make_per_area_performance(data, out_dir)
    make_encoder_diagnostic(data, out_dir)
    make_decoder_confusion(data, out_dir)
    make_spatial_change_validation(out_dir, labels_dir, predictions_dir)
    make_planning_dropout(data, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make data-traceable Paper58 RSE revision figures.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--decoder-results", type=Path, default=DEFAULT_DECODER_RESULTS)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    args = parser.parse_args()
    make_all_figures(args.results_dir, args.out_dir, args.decoder_results, args.labels_dir, args.predictions_dir)
    print(f"Wrote revision figures to {args.out_dir}")


if __name__ == "__main__":
    main()
