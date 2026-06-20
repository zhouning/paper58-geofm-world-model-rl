from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.paper58_benchmark.statistics import clustered_bootstrap_ci


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH2_RESULTS_DIR = ROOT / "paper" / "rse_submission_paper58" / "benchmark_results_batch2"
DEFAULT_BATCH3_RESULTS_DIR = ROOT / "paper" / "rse_submission_paper58" / "benchmark_results_batch3"
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "rse_submission_paper58" / "diagnostics_batch23"

REQUIRED_METRIC_FIELDS = (
    "area",
    "start_year",
    "end_year",
    "tier",
    "stratum",
    "true_change_pixels",
    "model_change_f1",
    "primary_change_advantage",
    "spatial_shuffle_change_f1",
    "spatial_change_advantage",
)


def _read_metrics(path: Path, batch_name: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or ())
        missing = [field for field in REQUIRED_METRIC_FIELDS if field not in fieldnames]
        if missing:
            raise ValueError(f"{path.name} missing required columns: {', '.join(missing)}")
        rows = []
        for row in reader:
            rows.append(
                {
                    "batch": batch_name,
                    "area": row["area"],
                    "start_year": int(row["start_year"]),
                    "end_year": int(row["end_year"]),
                    "tier": row["tier"],
                    "stratum": row["stratum"],
                    "true_change_pixels": int(row["true_change_pixels"]),
                    "model_change_f1": float(row["model_change_f1"]),
                    "primary_change_advantage": float(row["primary_change_advantage"]),
                    "spatial_shuffle_change_f1": float(row["spatial_shuffle_change_f1"]),
                    "spatial_change_advantage": float(row["spatial_change_advantage"]),
                }
            )
    return rows


def _read_gate(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _rounded(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _urban_summary(rows: list[dict], n_boot: int) -> dict:
    urban_rows = [row for row in rows if row["stratum"] == "Urban" and row["tier"] == "tier1"]
    spatial_values = [row["spatial_change_advantage"] for row in urban_rows]
    primary_values = [row["primary_change_advantage"] for row in urban_rows]
    spatial_ci = clustered_bootstrap_ci(urban_rows, "spatial_change_advantage", n_boot=n_boot, seed=52)
    primary_ci = clustered_bootstrap_ci(urban_rows, "primary_change_advantage", n_boot=n_boot, seed=51)
    return {
        "n": len(urban_rows),
        "areas": [row["area"] for row in urban_rows],
        "mean_primary_change_advantage": _rounded(sum(primary_values) / len(primary_values)) if primary_values else None,
        "mean_spatial_change_advantage": _rounded(sum(spatial_values) / len(spatial_values)) if spatial_values else None,
        "n_negative_spatial": int(sum(value < 0 for value in spatial_values)),
        "n_positive_spatial": int(sum(value > 0 for value in spatial_values)),
        "primary_ci_low": _rounded(primary_ci["ci_low"]),
        "spatial_ci_low": _rounded(spatial_ci["ci_low"]),
    }


def _gate_comparison(batch2_gate: dict, batch3_gate: dict) -> dict:
    return {
        "batch2_status": batch2_gate.get("status"),
        "batch2_primary_ci_low": batch2_gate.get("tier1_primary_change", {}).get("ci_low"),
        "batch2_spatial_ci_low": batch2_gate.get("tier1_spatial_change", {}).get("ci_low"),
        "batch2_positive_tier1_strata": batch2_gate.get("positive_tier1_strata"),
        "batch3_status": batch3_gate.get("status"),
        "batch3_primary_ci_low": batch3_gate.get("tier1_primary_change", {}).get("ci_low"),
        "batch3_spatial_ci_low": batch3_gate.get("tier1_spatial_change", {}).get("ci_low"),
        "batch3_positive_tier1_strata": batch3_gate.get("positive_tier1_strata"),
    }


def _ranked_rows(batch2_rows: list[dict], batch3_rows: list[dict]) -> list[dict]:
    rows = batch2_rows + batch3_rows
    return sorted(
        rows,
        key=lambda row: (row["spatial_change_advantage"], row["primary_change_advantage"], row["batch"], row["area"]),
    )


def build_comparison_diagnostics(
    batch2_results_dir: Path = DEFAULT_BATCH2_RESULTS_DIR,
    batch3_results_dir: Path = DEFAULT_BATCH3_RESULTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_boot: int = 5000,
) -> dict:
    batch2_results_dir = Path(batch2_results_dir)
    batch3_results_dir = Path(batch3_results_dir)
    output_dir = Path(output_dir)

    batch2_rows = _read_metrics(batch2_results_dir / "benchmark_metrics_by_pair.csv", "batch2")
    batch3_rows = _read_metrics(batch3_results_dir / "benchmark_metrics_by_pair.csv", "batch3")
    batch2_gate = _read_gate(batch2_results_dir / "benchmark_gate_report.json")
    batch3_gate = _read_gate(batch3_results_dir / "benchmark_gate_report.json")

    ranked_rows = _ranked_rows(batch2_rows, batch3_rows)
    gate_comparison = _gate_comparison(batch2_gate, batch3_gate)
    urban_comparison = {
        "batch2": _urban_summary(batch2_rows, n_boot=n_boot),
        "batch3": _urban_summary(batch3_rows, n_boot=n_boot),
    }
    headline = {
        "lowest_spatial_area": {
            "batch": ranked_rows[0]["batch"],
            "area": ranked_rows[0]["area"],
            "spatial_change_advantage": ranked_rows[0]["spatial_change_advantage"],
            "primary_change_advantage": ranked_rows[0]["primary_change_advantage"],
        },
        "best_spatial_area": {
            "batch": ranked_rows[-1]["batch"],
            "area": ranked_rows[-1]["area"],
            "spatial_change_advantage": ranked_rows[-1]["spatial_change_advantage"],
            "primary_change_advantage": ranked_rows[-1]["primary_change_advantage"],
        },
    }
    summary = {
        "gate_comparison": gate_comparison,
        "urban_comparison": urban_comparison,
        "headline": headline,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "batch23_spatial_advantage_ranked.csv",
        ranked_rows,
        [
            "batch",
            "area",
            "stratum",
            "true_change_pixels",
            "model_change_f1",
            "primary_change_advantage",
            "spatial_shuffle_change_f1",
            "spatial_change_advantage",
        ],
    )
    (output_dir / "batch23_comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "batch23_comparison_summary.txt").write_text(
        "\n".join(
            [
                f"batch2_status={gate_comparison['batch2_status']}",
                f"batch2_spatial_ci_low={gate_comparison['batch2_spatial_ci_low']}",
                f"batch3_status={gate_comparison['batch3_status']}",
                f"batch3_spatial_ci_low={gate_comparison['batch3_spatial_ci_low']}",
                f"lowest_spatial_area={headline['lowest_spatial_area']['batch']}:{headline['lowest_spatial_area']['area']}",
                f"best_spatial_area={headline['best_spatial_area']['batch']}:{headline['best_spatial_area']['area']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Paper58 Batch 2 vs Batch 3 comparison diagnostics.")
    parser.add_argument("--batch2-results-dir", type=Path, default=DEFAULT_BATCH2_RESULTS_DIR)
    parser.add_argument("--batch3-results-dir", type=Path, default=DEFAULT_BATCH3_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()
    build_comparison_diagnostics(
        batch2_results_dir=args.batch2_results_dir,
        batch3_results_dir=args.batch3_results_dir,
        output_dir=args.output_dir,
        n_boot=args.n_boot,
    )
    print(f"Wrote Batch 2 vs Batch 3 diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
