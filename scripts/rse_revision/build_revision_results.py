from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENCODER_ABLATION = ROOT / "experiments" / "paper8" / "results" / "paper8_ablation_encoder.json"
DEFAULT_DROPOUT_STATS = ROOT / "experiments" / "paper8" / "results" / "dual_rep" / "dropout_statistical_tests.json"
DEFAULT_INTERVENTION_RESULTS = ROOT / "experiments" / "paper8" / "results" / "intervention"
DEFAULT_DUAL_REP_RESULTS = ROOT / "experiments" / "paper8" / "results" / "dual_rep"
DEFAULT_TRANSFER_HEPING_RESULTS = ROOT / "experiments" / "paper8" / "results" / "transfer_heping"
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "rse_submission_paper58" / "revision_results"

AREA_CATEGORIES = {
    "yangtze_delta": "Urban",
    "jing_jin_ji": "Urban",
    "chengdu_plain": "Urban",
    "pearl_river": "Urban",
    "northeast_plain": "Agriculture",
    "north_china_plain": "Agriculture",
    "jianghan_plain": "Agriculture",
    "hetao": "Agriculture",
    "yunnan_eco": "Ecology",
    "daxinganling": "Forest",
    "wuyi_mountain": "Forest",
    "qinghai_edge": "Plateau",
    "guanzhong": "Mixed",
    "minnan_coast": "Mixed",
    "bishan": "Mixed",
    "banzhucun": "Mixed",
    "poyang_lake": "Wetland",
}


def bootstrap_ci(values: list[float], n_boot: int = 5000, seed: int = 42) -> dict:
    clean = np.array([v for v in values if math.isfinite(v)], dtype=float)
    if clean.size == 0:
        return {"n": 0, "mean": None, "median": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    boot = rng.choice(clean, size=(n_boot, clean.size), replace=True).mean(axis=1)
    return {
        "n": int(clean.size),
        "mean": float(clean.mean()),
        "median": float(np.median(clean)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def paired_sign_test(values: list[float]) -> dict:
    clean = [v for v in values if math.isfinite(v)]
    n_positive = sum(v > 0 for v in clean)
    n_negative = sum(v < 0 for v in clean)
    n_tie = sum(v == 0 for v in clean)
    n_effective = n_positive + n_negative
    if n_effective == 0:
        p_value = 1.0
    else:
        k = min(n_positive, n_negative)
        tail = sum(math.comb(n_effective, i) for i in range(k + 1)) / (2 ** n_effective)
        p_value = min(1.0, 2.0 * tail)
    return {
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "n_tie": int(n_tie),
        "n_effective": int(n_effective),
        "two_sided_p": float(p_value),
    }


def _valid_metric(value: object) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _is_zero_placeholder(row: dict) -> bool:
    persistence = float(row.get("persistence_mean", 0.0) or 0.0)
    model = float(row.get("ldn_mean", 0.0) or 0.0)
    advantage = float(row.get("advantage", 0.0) or 0.0)
    return persistence == 0.0 and model == 0.0 and advantage == 0.0


def load_encoder_ablation_rows(path: Path, encoder: str = "alphaearth") -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    section = payload.get(encoder, {})
    rows = []
    for area, row in section.get("areas", {}).items():
        if _is_zero_placeholder(row):
            continue
        if not (_valid_metric(row.get("persistence_mean")) and _valid_metric(row.get("ldn_mean"))):
            continue
        rows.append(
            {
                "area": area,
                "persistence": float(row["persistence_mean"]),
                "model": float(row["ldn_mean"]),
                "advantage": float(row.get("advantage", float(row["ldn_mean"]) - float(row["persistence_mean"]))),
                "change_pixel_advantage": (
                    float(row["change_pixel_advantage"])
                    if _valid_metric(row.get("change_pixel_advantage"))
                    else None
                ),
            }
        )
    return rows


def summarize_encoder_rows(rows: list[dict]) -> dict:
    advantages = [r["advantage"] for r in rows]
    persistence = [r["persistence"] for r in rows]
    model = [r["model"] for r in rows]
    change_advantages = [r["change_pixel_advantage"] for r in rows if r["change_pixel_advantage"] is not None]
    return {
        "n_areas": len(rows),
        "persistence": bootstrap_ci(persistence),
        "model": bootstrap_ci(model),
        "advantage": bootstrap_ci(advantages),
        "advantage_sign_test": paired_sign_test(advantages),
        "change_pixel_advantage": bootstrap_ci(change_advantages),
        "areas_positive": [r["area"] for r in rows if r["advantage"] > 0],
        "areas_negative": [r["area"] for r in rows if r["advantage"] < 0],
    }


def summarize_rows_by_category(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        category = AREA_CATEGORIES.get(str(row.get("area", "")))
        if category is None:
            continue
        advantage = row.get("advantage")
        if advantage is None:
            continue
        try:
            numeric_advantage = float(advantage)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric_advantage):
            continue
        grouped.setdefault(category, []).append({**row, "advantage": numeric_advantage})

    summary = {}
    for category in sorted(grouped):
        values = [row["advantage"] for row in grouped[category]]
        summary[category] = {
            "n": len(values),
            "mean_advantage": float(mean(values)),
            "median_advantage": float(median(values)),
            "n_positive": int(sum(v > 0 for v in values)),
            "n_negative": int(sum(v < 0 for v in values)),
            "areas": [row["area"] for row in grouped[category]],
        }
    return summary


def _read_planning_jsons(directory: Path, pattern: str) -> list[dict]:
    rows = []
    for path in sorted(directory.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not _valid_metric(payload.get("mean_slope")):
            continue
        rows.append(
            {
                "source_file": str(path),
                "seed": payload.get("seed"),
                "mean_slope": float(payload["mean_slope"]),
                "mean_cont": float(payload["mean_cont"]) if _valid_metric(payload.get("mean_cont")) else None,
                "mean_reward": float(payload["mean_reward"]) if _valid_metric(payload.get("mean_reward")) else None,
            }
        )
    return rows


def summarize_planning_runs(run_specs: dict[str, tuple[Path, str]]) -> dict:
    summary = {}
    for label, (directory, pattern) in run_specs.items():
        rows = _read_planning_jsons(directory, pattern)
        slopes = [row["mean_slope"] for row in rows]
        conts = [row["mean_cont"] for row in rows if row["mean_cont"] is not None]
        rewards = [row["mean_reward"] for row in rows if row["mean_reward"] is not None]
        summary[label] = {
            "n": len(rows),
            "slope_mean": float(mean(slopes)) if slopes else None,
            "slope_std": float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0 if slopes else None,
            "cont_mean": float(mean(conts)) if conts else None,
            "cont_std": float(np.std(conts, ddof=1)) if len(conts) > 1 else 0.0 if conts else None,
            "reward_mean": float(mean(rewards)) if rewards else None,
            "reward_std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0 if rewards else None,
        }

    full_slope = summary.get("full", {}).get("slope_mean")
    if full_slope not in (None, 0.0):
        for label, item in summary.items():
            slope = item.get("slope_mean")
            item["slope_retention_vs_full"] = float(slope / full_slope) if slope is not None else None
    return summary


def _read_transfer_jsons(directory: Path, pattern: str) -> list[dict]:
    rows = []
    for path in sorted(directory.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not _valid_metric(payload.get("mean_reward")):
            continue
        rows.append(
            {
                "source_file": str(path),
                "seed": payload.get("seed"),
                "mean_reward": float(payload["mean_reward"]),
                "mean_cropland_change": (
                    float(payload["mean_cropland_change"])
                    if _valid_metric(payload.get("mean_cropland_change"))
                    else None
                ),
            }
        )
    return rows


def summarize_transfer_planning_runs(directory: Path) -> dict:
    summary = {}
    for label in ("random", "greedy", "transfer"):
        rows = _read_transfer_jsons(directory, f"{label}_eval_seed*.json")
        rewards = [row["mean_reward"] for row in rows]
        crop_changes = [row["mean_cropland_change"] for row in rows if row["mean_cropland_change"] is not None]
        summary[label] = {
            "n": len(rows),
            "reward_mean": float(mean(rewards)) if rewards else None,
            "reward_std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0 if rewards else None,
            "crop_change_mean": float(mean(crop_changes)) if crop_changes else None,
            "crop_change_std": (
                float(np.std(crop_changes, ddof=1))
                if len(crop_changes) > 1
                else 0.0
                if crop_changes
                else None
            ),
        }

    random_reward = summary.get("random", {}).get("reward_mean")
    greedy_reward = summary.get("greedy", {}).get("reward_mean")
    for item in summary.values():
        reward = item.get("reward_mean")
        item["reward_delta_vs_random"] = (
            float(reward - random_reward)
            if reward is not None and random_reward is not None
            else None
        )
        item["reward_delta_vs_greedy"] = (
            float(reward - greedy_reward)
            if reward is not None and greedy_reward is not None
            else None
        )
    return summary


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["area", "persistence", "model", "advantage", "change_pixel_advantage"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_category_summary_csv(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "n", "mean_advantage", "median_advantage", "n_positive", "n_negative", "areas"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for category, row in summary.items():
            writer.writerow({**row, "category": category, "areas": ";".join(row["areas"])})


def write_planning_summary_csv(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "configuration",
        "n",
        "slope_mean",
        "slope_std",
        "cont_mean",
        "cont_std",
        "reward_mean",
        "reward_std",
        "slope_retention_vs_full",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, row in summary.items():
            writer.writerow({**row, "configuration": label})


def write_transfer_summary_csv(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "configuration",
        "n",
        "reward_mean",
        "reward_std",
        "crop_change_mean",
        "crop_change_std",
        "reward_delta_vs_random",
        "reward_delta_vs_greedy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, row in summary.items():
            writer.writerow({**row, "configuration": label})


def build_revision_results(
    encoder_ablation_path: Path = DEFAULT_ENCODER_ABLATION,
    dropout_stats_path: Path = DEFAULT_DROPOUT_STATS,
    intervention_results_dir: Path = DEFAULT_INTERVENTION_RESULTS,
    dual_rep_results_dir: Path = DEFAULT_DUAL_REP_RESULTS,
    transfer_heping_results_dir: Path = DEFAULT_TRANSFER_HEPING_RESULTS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    alphaearth_rows = load_encoder_ablation_rows(encoder_ablation_path, "alphaearth")
    prithvi_rows = load_encoder_ablation_rows(encoder_ablation_path, "prithvi")
    category_summary = summarize_rows_by_category(alphaearth_rows)
    planning_baselines = summarize_planning_runs(
        {
            "embedding_intervention": (intervention_results_dir, "ppo_eval_seed*.json"),
            "full": (dual_rep_results_dir, "full_eval_seed*.json"),
            "dropout0.3": (dual_rep_results_dir, "dropout0.3_eval_seed*.json"),
            "dropout1.0": (dual_rep_results_dir, "dropout1.0_eval_seed*.json"),
        }
    )
    transfer_planning = summarize_transfer_planning_runs(transfer_heping_results_dir)

    results = {
        "source_files": {
            "encoder_ablation": str(encoder_ablation_path),
            "dropout_statistics": str(dropout_stats_path) if dropout_stats_path.exists() else None,
            "intervention_results_dir": str(intervention_results_dir),
            "dual_rep_results_dir": str(dual_rep_results_dir),
            "transfer_heping_results_dir": str(transfer_heping_results_dir),
        },
        "alphaearth": summarize_encoder_rows(alphaearth_rows),
        "prithvi": summarize_encoder_rows(prithvi_rows),
        "alphaearth_category_summary": category_summary,
        "planning_baselines": planning_baselines,
        "transfer_planning": transfer_planning,
        "notes": [
            "Rows with persistence=model=advantage=0 are treated as missing-cache placeholders and excluded.",
            "Bootstrap confidence intervals are computed over area-level means.",
            "The sign test uses paired area-level model-minus-persistence advantages.",
            "The planning baseline summary compares configurations with the same slope/contiguity evaluation metrics.",
            "The transfer planning summary is an embedding-space diagnostic and is not directly comparable with real county-level slope/contiguity metrics.",
        ],
    }

    if dropout_stats_path.exists():
        results["planning_dropout_statistics"] = json.loads(dropout_stats_path.read_text(encoding="utf-8"))

    (output_dir / "revision_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_rows_csv(output_dir / "alphaearth_area_metrics.csv", alphaearth_rows)
    write_rows_csv(output_dir / "prithvi_area_metrics.csv", prithvi_rows)
    write_category_summary_csv(output_dir / "alphaearth_category_summary.csv", category_summary)
    write_planning_summary_csv(output_dir / "planning_baseline_summary.csv", planning_baselines)
    write_transfer_summary_csv(output_dir / "transfer_planning_summary.csv", transfer_planning)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Paper58 RSE revision result tables.")
    parser.add_argument("--encoder-ablation", type=Path, default=DEFAULT_ENCODER_ABLATION)
    parser.add_argument("--dropout-stats", type=Path, default=DEFAULT_DROPOUT_STATS)
    parser.add_argument("--intervention-results-dir", type=Path, default=DEFAULT_INTERVENTION_RESULTS)
    parser.add_argument("--dual-rep-results-dir", type=Path, default=DEFAULT_DUAL_REP_RESULTS)
    parser.add_argument("--transfer-heping-results-dir", type=Path, default=DEFAULT_TRANSFER_HEPING_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    results = build_revision_results(
        args.encoder_ablation,
        args.dropout_stats,
        args.intervention_results_dir,
        args.dual_rep_results_dir,
        args.transfer_heping_results_dir,
        args.output_dir,
    )
    ae = results["alphaearth"]["advantage"]
    print(
        "AlphaEarth advantage: "
        f"n={ae['n']}, mean={ae['mean']:.6f}, 95% CI=[{ae['ci_low']:.6f}, {ae['ci_high']:.6f}]"
    )


if __name__ == "__main__":
    main()
