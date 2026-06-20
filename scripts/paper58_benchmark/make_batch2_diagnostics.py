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
DEFAULT_EMBEDDINGS_DIR = ROOT / "data" / "independent_change_labels" / "embeddings"
DEFAULT_PREDICTIONS_DIR = ROOT / "data" / "independent_change_labels" / "predicted"
DEFAULT_DECODER_PATH = ROOT / "src" / "adk_world_model" / "weights" / "lulc_decoder_v1.pkl"

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


def _load_decoder(path: Path):
    import joblib

    return joblib.load(path)


def _decode_lulc(embedding_grid: np.ndarray, decoder) -> np.ndarray:
    h, w = embedding_grid.shape[:2]
    return decoder.predict(embedding_grid.reshape(-1, embedding_grid.shape[-1])).reshape(h, w).astype(np.int32)


def _pixel_accuracy(reference: np.ndarray, prediction: np.ndarray) -> float:
    if reference.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: reference={reference.shape}, prediction={prediction.shape}")
    return float(np.mean(reference == prediction)) if reference.size else 0.0


def _shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    shifted = np.zeros(mask.shape, dtype=bool)
    src_y0 = max(0, -dy)
    src_y1 = mask.shape[0] - max(0, dy)
    src_x0 = max(0, -dx)
    src_x1 = mask.shape[1] - max(0, dx)
    dst_y0 = max(0, dy)
    dst_x0 = max(0, dx)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return shifted


def _centroid(mask: np.ndarray) -> tuple[float | None, float | None]:
    points = np.argwhere(mask)
    if points.size == 0:
        return None, None
    yx = points.mean(axis=0)
    return float(yx[0]), float(yx[1])


def best_shift_diagnostic(
    true_change: np.ndarray,
    model_change: np.ndarray,
    max_shift: int = 4,
) -> dict:
    raw_f1 = binary_change_metrics(true_change, model_change)["f1"]
    best_f1 = raw_f1
    best_dy = 0
    best_dx = 0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = _shift_mask(model_change, dy, dx)
            shifted_f1 = binary_change_metrics(true_change, shifted)["f1"]
            if shifted_f1 > best_f1:
                best_f1 = shifted_f1
                best_dy = dy
                best_dx = dx

    true_y, true_x = _centroid(true_change)
    model_y, model_x = _centroid(model_change)
    return {
        "raw_change_f1": float(raw_f1),
        "best_shift_change_f1": float(best_f1),
        "best_dy": int(best_dy),
        "best_dx": int(best_dx),
        "centroid_true_y": true_y,
        "centroid_true_x": true_x,
        "centroid_model_y": model_y,
        "centroid_model_x": model_x,
    }


def transition_count_rows(
    start: np.ndarray,
    end: np.ndarray,
    change_mask: np.ndarray,
    source: str,
    limit: int = 12,
) -> list[dict]:
    counts: dict[tuple[int, int], int] = {}
    ys, xs = np.where(change_mask)
    for y, x in zip(ys, xs):
        key = (int(start[y, x]), int(end[y, x]))
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [
        {
            "source": source,
            "start_class": start_class,
            "end_class": end_class,
            "n_pixels": n_pixels,
        }
        for (start_class, end_class), n_pixels in ranked
    ]


def _class_count_summary(values: np.ndarray, limit: int = 4) -> str:
    counts: dict[int, int] = {}
    for value in values.ravel():
        class_id = int(value)
        counts[class_id] = counts.get(class_id, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return ";".join(f"{class_id}:{n_pixels}" for class_id, n_pixels in ranked)


def _probability_grid(embedding_grid: np.ndarray, decoder) -> tuple[np.ndarray, list[int]] | None:
    if not hasattr(decoder, "predict_proba") or not hasattr(decoder, "classes_"):
        return None
    h, w = embedding_grid.shape[:2]
    probabilities = decoder.predict_proba(embedding_grid.reshape(-1, embedding_grid.shape[-1]))
    classes = [int(value) for value in decoder.classes_]
    return probabilities.reshape(h, w, len(classes)), classes


def _rounded_float(value: float) -> float:
    return round(float(value), 6)


def make_batch2_alignment_table(
    out_dir: Path,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
    areas: list[str] | None = None,
    start_year: int = 2020,
    end_year: int = 2021,
    max_shift: int = 4,
) -> list[dict]:
    if areas is None:
        areas = [
            path.name.removesuffix(f"_lulc_pred_{start_year}_{end_year}.npy")
            for path in sorted(Path(predictions_dir).glob(f"*_lulc_pred_{start_year}_{end_year}.npy"))
        ]

    rows = []
    for area in areas:
        start = np.load(Path(labels_dir) / f"{area}_lulc_{start_year}.npy")
        end = np.load(Path(labels_dir) / f"{area}_lulc_{end_year}.npy")
        pred = np.load(Path(predictions_dir) / f"{area}_lulc_pred_{start_year}_{end_year}.npy")
        true_change = end != start
        model_change = pred != start
        rows.append({"area": area, **best_shift_diagnostic(true_change, model_change, max_shift=max_shift)})

    _write_csv(
        Path(out_dir) / "batch2_spatial_alignment_shift.csv",
        rows,
        [
            "area",
            "raw_change_f1",
            "best_shift_change_f1",
            "best_dy",
            "best_dx",
            "centroid_true_y",
            "centroid_true_x",
            "centroid_model_y",
            "centroid_model_x",
        ],
    )
    return rows


def make_embedding_decoder_audit_table(
    out_dir: Path,
    decoder,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
    areas: list[str] | None = None,
    start_year: int = 2020,
    end_year: int = 2021,
    max_shift: int = 4,
) -> list[dict]:
    if areas is None:
        areas = [
            path.name.removesuffix(f"_lulc_pred_{start_year}_{end_year}.npy")
            for path in sorted(Path(predictions_dir).glob(f"*_lulc_pred_{start_year}_{end_year}.npy"))
        ]

    rows = []
    for area in areas:
        start = np.load(Path(labels_dir) / f"{area}_lulc_{start_year}.npy")
        end = np.load(Path(labels_dir) / f"{area}_lulc_{end_year}.npy")
        pred = np.load(Path(predictions_dir) / f"{area}_lulc_pred_{start_year}_{end_year}.npy")
        emb_start = np.load(Path(embeddings_dir) / f"{area}_emb_{start_year}.npy")
        emb_end = np.load(Path(embeddings_dir) / f"{area}_emb_{end_year}.npy")

        decoded_start = _decode_lulc(emb_start, decoder)
        decoded_end = _decode_lulc(emb_end, decoder)
        true_change = end != start
        decoded_observed_change = decoded_end != decoded_start
        model_change = pred != start
        model_change_decoded_start = pred != decoded_start
        decoded_metrics = binary_change_metrics(true_change, decoded_observed_change)
        decoded_shift = best_shift_diagnostic(true_change, decoded_observed_change, max_shift=max_shift)

        rows.append(
            {
                "area": area,
                "start_decode_accuracy": _pixel_accuracy(start, decoded_start),
                "end_decode_accuracy": _pixel_accuracy(end, decoded_end),
                "true_change_pixels": int(np.count_nonzero(true_change)),
                "decoded_observed_change_pixels": int(np.count_nonzero(decoded_observed_change)),
                "decoded_observed_change_precision": float(decoded_metrics["precision"]),
                "decoded_observed_change_recall": float(decoded_metrics["recall"]),
                "decoded_observed_change_f1": float(decoded_metrics["f1"]),
                "decoded_observed_best_shift_change_f1": float(decoded_shift["best_shift_change_f1"]),
                "decoded_observed_best_dy": int(decoded_shift["best_dy"]),
                "decoded_observed_best_dx": int(decoded_shift["best_dx"]),
                "model_change_f1_label_start": float(binary_change_metrics(true_change, model_change)["f1"]),
                "model_change_f1_decoded_start": float(
                    binary_change_metrics(true_change, model_change_decoded_start)["f1"]
                ),
                "model_end_accuracy": _pixel_accuracy(end, pred),
                "model_start_accuracy": _pixel_accuracy(start, pred),
            }
        )

    _write_csv(
        Path(out_dir) / "batch2_embedding_decoder_audit.csv",
        rows,
        [
            "area",
            "start_decode_accuracy",
            "end_decode_accuracy",
            "true_change_pixels",
            "decoded_observed_change_pixels",
            "decoded_observed_change_precision",
            "decoded_observed_change_recall",
            "decoded_observed_change_f1",
            "decoded_observed_best_shift_change_f1",
            "decoded_observed_best_dy",
            "decoded_observed_best_dx",
            "model_change_f1_label_start",
            "model_change_f1_decoded_start",
            "model_end_accuracy",
            "model_start_accuracy",
        ],
    )
    return rows


def make_transition_table(
    out_dir: Path,
    decoder,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
    area: str = "xiong_an_fringe_holdout",
    start_year: int = 2020,
    end_year: int = 2021,
    limit: int = 12,
) -> list[dict]:
    start = np.load(Path(labels_dir) / f"{area}_lulc_{start_year}.npy")
    end = np.load(Path(labels_dir) / f"{area}_lulc_{end_year}.npy")
    pred = np.load(Path(predictions_dir) / f"{area}_lulc_pred_{start_year}_{end_year}.npy")
    emb_start = np.load(Path(embeddings_dir) / f"{area}_emb_{start_year}.npy")
    emb_end = np.load(Path(embeddings_dir) / f"{area}_emb_{end_year}.npy")
    decoded_start = _decode_lulc(emb_start, decoder)
    decoded_end = _decode_lulc(emb_end, decoder)

    rows = (
        transition_count_rows(start, end, end != start, source="reference_change", limit=limit)
        + transition_count_rows(start, pred, pred != start, source="model_change", limit=limit)
        + transition_count_rows(
            decoded_start,
            decoded_end,
            decoded_end != decoded_start,
            source="decoded_observed_change",
            limit=limit,
        )
    )
    _write_csv(
        Path(out_dir) / f"{area}_transition_counts.csv",
        rows,
        ["source", "start_class", "end_class", "n_pixels"],
    )
    return rows


def make_transition_fate_table(
    out_dir: Path,
    decoder,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR,
    predictions_dir: Path = DEFAULT_PREDICTIONS_DIR,
    area: str = "xiong_an_fringe_holdout",
    start_year: int = 2020,
    end_year: int = 2021,
    top_n_true_transitions: int = 8,
) -> list[dict]:
    start = np.load(Path(labels_dir) / f"{area}_lulc_{start_year}.npy")
    end = np.load(Path(labels_dir) / f"{area}_lulc_{end_year}.npy")
    pred = np.load(Path(predictions_dir) / f"{area}_lulc_pred_{start_year}_{end_year}.npy")
    emb_start = np.load(Path(embeddings_dir) / f"{area}_emb_{start_year}.npy")
    emb_end = np.load(Path(embeddings_dir) / f"{area}_emb_{end_year}.npy")
    decoded_start = _decode_lulc(emb_start, decoder)
    decoded_end = _decode_lulc(emb_end, decoder)
    probability_grid = _probability_grid(emb_end, decoder)

    transition_counts: dict[tuple[int, int], int] = {}
    changed = start != end
    for start_class, end_class in zip(start[changed].ravel(), end[changed].ravel()):
        key = (int(start_class), int(end_class))
        transition_counts[key] = transition_counts.get(key, 0) + 1
    ranked = sorted(transition_counts.items(), key=lambda item: (-item[1], item[0]))[:top_n_true_transitions]

    rows = []
    for (start_class, end_class), n_pixels in ranked:
        mask = (start == start_class) & (end == end_class)
        probability_fields = {
            "mean_true_end_prob": None,
            "median_true_end_prob": None,
            "top_mean_prob_class": None,
            "top_mean_prob": None,
            "second_mean_prob_class": None,
            "second_mean_prob": None,
        }
        if probability_grid is not None:
            probabilities, classes = probability_grid
            class_to_index = {class_id: index for index, class_id in enumerate(classes)}
            masked_probabilities = probabilities[mask]
            mean_probabilities = masked_probabilities.mean(axis=0)
            ranked_probability_indices = sorted(
                range(len(classes)),
                key=lambda index: (-mean_probabilities[index], classes[index]),
            )
            true_end_index = class_to_index.get(end_class)
            if true_end_index is not None:
                true_end_probabilities = masked_probabilities[:, true_end_index]
                probability_fields["mean_true_end_prob"] = _rounded_float(true_end_probabilities.mean())
                probability_fields["median_true_end_prob"] = _rounded_float(np.median(true_end_probabilities))
            if ranked_probability_indices:
                top_index = ranked_probability_indices[0]
                probability_fields["top_mean_prob_class"] = classes[top_index]
                probability_fields["top_mean_prob"] = _rounded_float(mean_probabilities[top_index])
            if len(ranked_probability_indices) > 1:
                second_index = ranked_probability_indices[1]
                probability_fields["second_mean_prob_class"] = classes[second_index]
                probability_fields["second_mean_prob"] = _rounded_float(mean_probabilities[second_index])
        rows.append(
            {
                "true_transition": f"{start_class}->{end_class}",
                "n_true_pixels": n_pixels,
                "decoded_start_top": _class_count_summary(decoded_start[mask]),
                "decoded_end_top": _class_count_summary(decoded_end[mask]),
                "model_end_top": _class_count_summary(pred[mask]),
                **probability_fields,
            }
        )

    _write_csv(
        Path(out_dir) / f"{area}_transition_fate.csv",
        rows,
        [
            "true_transition",
            "n_true_pixels",
            "decoded_start_top",
            "decoded_end_top",
            "model_end_top",
            "mean_true_end_prob",
            "median_true_end_prob",
            "top_mean_prob_class",
            "top_mean_prob",
            "second_mean_prob_class",
            "second_mean_prob",
        ],
    )
    return rows


def make_decoder_true_end_confidence_table(
    out_dir: Path,
    decoder,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR,
    areas: list[str] | None = None,
    start_year: int = 2020,
    end_year: int = 2021,
) -> list[dict]:
    if areas is None:
        areas = [
            path.name.removesuffix(f"_emb_{end_year}.npy")
            for path in sorted(Path(embeddings_dir).glob(f"*_emb_{end_year}.npy"))
        ]

    rows = []
    for area in areas:
        start = np.load(Path(labels_dir) / f"{area}_lulc_{start_year}.npy")
        end = np.load(Path(labels_dir) / f"{area}_lulc_{end_year}.npy")
        emb_end = np.load(Path(embeddings_dir) / f"{area}_emb_{end_year}.npy")
        probability_grid = _probability_grid(emb_end, decoder)
        if probability_grid is None:
            continue

        probabilities, classes = probability_grid
        class_to_index = {class_id: index for index, class_id in enumerate(classes)}
        decoded_end = _decode_lulc(emb_end, decoder)
        changed = start != end
        for end_class in sorted({int(value) for value in np.unique(end[changed])}):
            end_class_index = class_to_index.get(end_class)
            if end_class_index is None:
                continue
            mask = changed & (end == end_class)
            true_end_probabilities = probabilities[mask, end_class_index]

            decoded_counts: dict[int, int] = {}
            for value in decoded_end[mask].ravel():
                class_id = int(value)
                decoded_counts[class_id] = decoded_counts.get(class_id, 0) + 1
            top_pred_class, top_pred_count = sorted(
                decoded_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[0]

            rows.append(
                {
                    "area": area,
                    "true_end_class": end_class,
                    "n_pixels": int(np.count_nonzero(mask)),
                    "mean_true_end_prob": _rounded_float(true_end_probabilities.mean()),
                    "median_true_end_prob": _rounded_float(np.median(true_end_probabilities)),
                    "top_pred_class": top_pred_class,
                    "top_pred_count": top_pred_count,
                }
            )

    rows.sort(key=lambda row: (row["true_end_class"], row["mean_true_end_prob"], row["area"]))
    _write_csv(
        Path(out_dir) / "batch2_decoder_true_end_confidence_by_area.csv",
        rows,
        [
            "area",
            "true_end_class",
            "n_pixels",
            "mean_true_end_prob",
            "median_true_end_prob",
            "top_pred_class",
            "top_pred_count",
        ],
    )
    return rows


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
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--decoder", type=Path, default=DEFAULT_DECODER_PATH)
    args = parser.parse_args()

    apply_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decoder = _load_decoder(args.decoder)
    xiongan_summary = make_xiongan_spatial_panel(
        out_dir=args.output_dir,
        labels_dir=args.labels_dir,
        predictions_dir=args.predictions_dir,
    )
    sensitivity_summary = make_batch2_sensitivity_tables(
        out_dir=args.output_dir,
        results_dir=args.results_dir,
    )
    areas = [row["area"] for row in _read_metrics(args.results_dir / "benchmark_metrics_by_pair.csv")]
    alignment_rows = make_batch2_alignment_table(
        out_dir=args.output_dir,
        labels_dir=args.labels_dir,
        predictions_dir=args.predictions_dir,
        areas=areas,
    )
    decoder_audit_rows = make_embedding_decoder_audit_table(
        out_dir=args.output_dir,
        decoder=decoder,
        labels_dir=args.labels_dir,
        embeddings_dir=args.embeddings_dir,
        predictions_dir=args.predictions_dir,
        areas=areas,
    )
    transition_rows = make_transition_table(
        out_dir=args.output_dir,
        decoder=decoder,
        labels_dir=args.labels_dir,
        embeddings_dir=args.embeddings_dir,
        predictions_dir=args.predictions_dir,
        area="xiong_an_fringe_holdout",
    )
    transition_fate_rows = make_transition_fate_table(
        out_dir=args.output_dir,
        decoder=decoder,
        labels_dir=args.labels_dir,
        embeddings_dir=args.embeddings_dir,
        predictions_dir=args.predictions_dir,
        area="xiong_an_fringe_holdout",
    )
    true_end_confidence_rows = make_decoder_true_end_confidence_table(
        out_dir=args.output_dir,
        decoder=decoder,
        labels_dir=args.labels_dir,
        embeddings_dir=args.embeddings_dir,
        areas=areas,
    )
    xiongan_alignment = next(row for row in alignment_rows if row["area"] == "xiong_an_fringe_holdout")
    xiongan_decoder_audit = next(row for row in decoder_audit_rows if row["area"] == "xiong_an_fringe_holdout")
    xiongan_transition_top = next(
        (
            row
            for row in transition_rows
            if row["source"] == "reference_change"
        ),
        None,
    )
    xiongan_transition_fate_top = transition_fate_rows[0] if transition_fate_rows else None
    xiongan_class11_confidence = next(
        (
            row
            for row in true_end_confidence_rows
            if row["area"] == "xiong_an_fringe_holdout" and row["true_end_class"] == 11
        ),
        None,
    )
    summary_path = args.output_dir / "batch2_diagnostic_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"xiongan_model_change_f1={xiongan_summary['model_change_f1']}",
                f"xiongan_shuffle_change_f1={xiongan_summary['shuffle_change_f1']}",
                f"xiongan_best_shift_change_f1={xiongan_alignment['best_shift_change_f1']}",
                f"xiongan_best_shift_dy={xiongan_alignment['best_dy']}",
                f"xiongan_best_shift_dx={xiongan_alignment['best_dx']}",
                f"xiongan_start_decode_accuracy={xiongan_decoder_audit['start_decode_accuracy']}",
                f"xiongan_end_decode_accuracy={xiongan_decoder_audit['end_decode_accuracy']}",
                f"xiongan_decoded_observed_change_f1={xiongan_decoder_audit['decoded_observed_change_f1']}",
                (
                    "xiongan_decoded_observed_best_shift_change_f1="
                    f"{xiongan_decoder_audit['decoded_observed_best_shift_change_f1']}"
                ),
                f"xiongan_decoded_observed_best_shift_dy={xiongan_decoder_audit['decoded_observed_best_dy']}",
                f"xiongan_decoded_observed_best_shift_dx={xiongan_decoder_audit['decoded_observed_best_dx']}",
                f"batch2_spatial_ci_low={sensitivity_summary['batch2_spatial_ci_low']}",
                f"drop_xiongan_spatial_ci_low={sensitivity_summary['drop_xiongan_spatial_ci_low']}",
                f"xiongan_spatial_advantage={sensitivity_summary['xiongan_spatial_advantage']}",
                (
                    "xiongan_reference_top_transition="
                    f"{xiongan_transition_top['start_class']}->{xiongan_transition_top['end_class']}:"
                    f"{xiongan_transition_top['n_pixels']}"
                )
                if xiongan_transition_top is not None
                else "xiongan_reference_top_transition=",
                (
                    "xiongan_reference_top_transition_fate="
                    f"{xiongan_transition_fate_top['true_transition']};"
                    f"decoded_end={xiongan_transition_fate_top['decoded_end_top']};"
                    f"model_end={xiongan_transition_fate_top['model_end_top']};"
                    f"mean_true_end_prob={xiongan_transition_fate_top['mean_true_end_prob']};"
                    f"top_mean_prob={xiongan_transition_fate_top['top_mean_prob_class']}:"
                    f"{xiongan_transition_fate_top['top_mean_prob']}"
                )
                if xiongan_transition_fate_top is not None
                else "xiongan_reference_top_transition_fate=",
                (
                    "xiongan_true_end_class11_mean_prob="
                    f"{xiongan_class11_confidence['mean_true_end_prob']}"
                )
                if xiongan_class11_confidence is not None
                else "xiongan_true_end_class11_mean_prob=",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote Batch 2 diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
