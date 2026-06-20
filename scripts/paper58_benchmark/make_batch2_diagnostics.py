from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from scripts.paper58_benchmark.baselines import spatial_shuffle_prediction
from scripts.paper58_benchmark.evaluate_benchmark import binary_change_metrics
from scripts.paper58_benchmark.statistics import clustered_bootstrap_ci


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "paper" / "rse_submission_paper58" / "benchmark_results_batch2"
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "rse_submission_paper58" / "diagnostics_batch2"
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


def _class_index_array(array: np.ndarray, class_to_index: dict[int, int]) -> np.ndarray:
    indexed = np.zeros(array.shape, dtype=int)
    for cls, index in class_to_index.items():
        indexed[array == cls] = index
    return indexed


def _lulc_cmap(arrays: list[np.ndarray]) -> tuple[ListedColormap, dict[int, int]]:
    classes = sorted({int(value) for array in arrays for value in np.unique(array)})
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    colors = [LULC_COLORS.get(cls, "#D0D0D0") for cls in classes]
    return ListedColormap(colors), class_to_index


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _read_metrics(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        parsed.append(
            {
                "area": row["area"],
                "stratum": row["stratum"],
                "primary_change_advantage": float(row["primary_change_advantage"]),
                "spatial_change_advantage": float(row["spatial_change_advantage"]),
                "model_change_f1": float(row["model_change_f1"]),
                "spatial_shuffle_change_f1": float(row["spatial_shuffle_change_f1"]),
            }
        )
    return parsed


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_xiongan_spatial_panel(
    out_dir: Path,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
    area: str = "xiong_an_fringe_holdout",
    start_year: int = 2020,
    end_year: int = 2021,
) -> dict:
    start = np.load(Path(labels_dir) / f"{area}_lulc_{start_year}.npy")
    end = np.load(Path(labels_dir) / f"{area}_lulc_{end_year}.npy")
    pred = np.load(Path(predictions_dir) / f"{area}_lulc_pred_{start_year}_{end_year}.npy")
    shuffle = spatial_shuffle_prediction(pred)

    true_change = end != start
    model_change = pred != start
    shuffle_change = shuffle != start
    hit = true_change & model_change
    miss = true_change & ~model_change
    false_alarm = ~true_change & model_change
    shuffle_hit = true_change & shuffle_change
    shuffle_miss = true_change & ~shuffle_change
    shuffle_false_alarm = ~true_change & shuffle_change

    model_metrics = binary_change_metrics(true_change, model_change)
    shuffle_metrics = binary_change_metrics(true_change, shuffle_change)

    cmap, class_to_index = _lulc_cmap([start, end, pred])
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.8), constrained_layout=True)
    map_panels = [
        ("a  Reference start", start),
        ("b  Reference end", end),
        ("c  Model prediction", pred),
        ("d  Shuffled prediction", shuffle),
    ]
    for ax, (title, array) in zip(axes[0], map_panels):
        ax.imshow(_class_index_array(array, class_to_index), cmap=cmap, interpolation="nearest")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1, 0].imshow(true_change, cmap=ListedColormap(["#F0F0F0", BLUE]), interpolation="nearest")
    axes[1, 0].set_title("e  Reference change", loc="left", fontweight="bold", fontsize=8)
    axes[1, 1].imshow(model_change, cmap=ListedColormap(["#F0F0F0", GOLD]), interpolation="nearest")
    axes[1, 1].set_title("f  Predicted change", loc="left", fontweight="bold", fontsize=8)

    model_error = np.zeros(start.shape, dtype=int)
    model_error[hit] = 1
    model_error[miss] = 2
    model_error[false_alarm] = 3
    axes[1, 2].imshow(model_error, cmap=ListedColormap(["#F0F0F0", GREEN, RED, GOLD]), interpolation="nearest")
    axes[1, 2].set_title("g  Model change error", loc="left", fontweight="bold", fontsize=8)

    shuffle_error = np.zeros(start.shape, dtype=int)
    shuffle_error[shuffle_hit] = 1
    shuffle_error[shuffle_miss] = 2
    shuffle_error[shuffle_false_alarm] = 3
    axes[1, 3].imshow(shuffle_error, cmap=ListedColormap(["#F0F0F0", GREEN, RED, GOLD]), interpolation="nearest")
    axes[1, 3].set_title("h  Shuffle change error", loc="left", fontweight="bold", fontsize=8)

    for ax in axes[1]:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Xiong'an spatial failure diagnostic {start_year}-{end_year}: "
        f"model F1={model_metrics['f1']:.3f}, shuffle F1={shuffle_metrics['f1']:.3f}",
        fontsize=9,
        fontweight="bold",
    )
    lulc_handles = [Patch(color=LULC_COLORS.get(cls, "#D0D0D0"), label=f"class {cls}") for cls in sorted(class_to_index)]
    error_handles = [
        Patch(color=GREEN, label="change hit"),
        Patch(color=RED, label="miss"),
        Patch(color=GOLD, label="false alarm"),
    ]
    fig.legend(
        handles=lulc_handles + error_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=min(9, len(lulc_handles) + len(error_handles)),
        fontsize=6,
    )
    _save_figure(fig, out_dir, "fig_batch2_xiongan_spatial_failure")

    return {
        "area": area,
        "start_year": start_year,
        "end_year": end_year,
        "true_change_pixels": int(np.count_nonzero(true_change)),
        "model_change_pixels": int(np.count_nonzero(model_change)),
        "shuffle_change_pixels": int(np.count_nonzero(shuffle_change)),
        "model_change_f1": float(model_metrics["f1"]),
        "shuffle_change_f1": float(shuffle_metrics["f1"]),
        "model_hits": int(np.count_nonzero(hit)),
        "model_misses": int(np.count_nonzero(miss)),
        "model_false_alarms": int(np.count_nonzero(false_alarm)),
        "shuffle_hits": int(np.count_nonzero(shuffle_hit)),
        "shuffle_misses": int(np.count_nonzero(shuffle_miss)),
        "shuffle_false_alarms": int(np.count_nonzero(shuffle_false_alarm)),
    }


def make_batch2_sensitivity_tables(
    out_dir: Path,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> dict:
    metric_rows = _read_metrics(Path(results_dir) / "benchmark_metrics_by_pair.csv")
    sorted_rows = sorted(metric_rows, key=lambda row: row["spatial_change_advantage"])
    _write_csv(
        Path(out_dir) / "batch2_spatial_advantage_ranked.csv",
        sorted_rows,
        [
            "area",
            "stratum",
            "primary_change_advantage",
            "spatial_change_advantage",
            "model_change_f1",
            "spatial_shuffle_change_f1",
        ],
    )

    sensitivity_rows = []
    for dropped in [None] + [row["area"] for row in sorted_rows]:
        subset = [row for row in metric_rows if row["area"] != dropped] if dropped else metric_rows
        ci = clustered_bootstrap_ci(subset, "spatial_change_advantage", n_boot=5000, seed=43)
        sensitivity_rows.append(
            {
                "dropped_area": "" if dropped is None else dropped,
                "n_rows": ci["n_rows"],
                "n_clusters": ci["n_clusters"],
                "mean": ci["mean"],
                "median": ci["median"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
            }
        )
    _write_csv(
        Path(out_dir) / "batch2_spatial_leave_one_out.csv",
        sensitivity_rows,
        ["dropped_area", "n_rows", "n_clusters", "mean", "median", "ci_low", "ci_high"],
    )

    xiongan = next(row for row in metric_rows if row["area"] == "xiong_an_fringe_holdout")
    return {
        "batch2_spatial_ci_low": sensitivity_rows[0]["ci_low"],
        "xiongan_spatial_advantage": xiongan["spatial_change_advantage"],
        "xiongan_primary_advantage": xiongan["primary_change_advantage"],
        "drop_xiongan_spatial_ci_low": next(
            row["ci_low"] for row in sensitivity_rows if row["dropped_area"] == "xiong_an_fringe_holdout"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Paper58 Batch 2 failure diagnostics.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    args = parser.parse_args()

    apply_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    xiongan_summary = make_xiongan_spatial_panel(
        out_dir=args.output_dir,
        labels_dir=args.labels_dir,
        predictions_dir=args.predictions_dir,
    )
    sensitivity_summary = make_batch2_sensitivity_tables(
        out_dir=args.output_dir,
        results_dir=args.results_dir,
    )
    summary_path = args.output_dir / "batch2_diagnostic_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"xiongan_model_change_f1={xiongan_summary['model_change_f1']}",
                f"xiongan_shuffle_change_f1={xiongan_summary['shuffle_change_f1']}",
                f"batch2_spatial_ci_low={sensitivity_summary['batch2_spatial_ci_low']}",
                f"drop_xiongan_spatial_ci_low={sensitivity_summary['drop_xiongan_spatial_ci_low']}",
                f"xiongan_spatial_advantage={sensitivity_summary['xiongan_spatial_advantage']}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote Batch 2 diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
