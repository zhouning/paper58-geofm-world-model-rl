# -*- coding: utf-8 -*-
"""Extract v4 manuscript numbers from macOS E3/E4/E6 results.

Runs after macOS pushes E3/E4/E6 results. Outputs a JSON with all
placeholder values that need to be substituted into v4 manuscript.

Usage:
    python extract_v4_numbers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
OUTPUT_FILE = HERE / "v4_manuscript_numbers.json"


def extract_e3() -> dict:
    """E3 multi-step rollout summary."""
    csv_path = RESULTS_DIR / "e3_multistep" / "multistep_all_areas.csv"
    if not csv_path.exists():
        return {"status": "not_run", "path": str(csv_path)}

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {"status": "empty"}

    summary = df.groupby("step")[["persistence", "model", "advantage"]].agg(["mean", "std"])

    per_step = {}
    for step in sorted(df["step"].unique()):
        step_df = df[df["step"] == step]
        per_step[int(step)] = {
            "persistence_mean": float(step_df["persistence"].mean()),
            "persistence_std": float(step_df["persistence"].std()),
            "model_mean": float(step_df["model"].mean()),
            "model_std": float(step_df["model"].std()),
            "advantage_mean": float(step_df["advantage"].mean()),
            "advantage_std": float(step_df["advantage"].std()),
            "n_areas": int(len(step_df)),
            "n_pos": int((step_df["advantage"] > 0).sum()),
            "n_neg": int((step_df["advantage"] < 0).sum()),
        }

    # Check if bug is fixed (persistence should be < 1.0)
    max_persist_step1 = df[df["step"] == 1]["persistence"].max()
    bug_fixed = max_persist_step1 < 1.0

    return {
        "status": "complete",
        "n_areas": int(df["area"].nunique()),
        "n_steps": int(df["step"].max()),
        "per_step": per_step,
        "bug_fixed": bug_fixed,
        "max_persist_step1": float(max_persist_step1),
    }


def extract_e4() -> dict:
    """E4 per-year decoder summary."""
    decoder_csv = RESULTS_DIR / "e4_per_year_decoder" / "decoder_by_year.csv"
    delta_csv = RESULTS_DIR / "e4_per_year_decoder" / "per_pair_end_accuracy_delta.csv"

    if not decoder_csv.exists() or not delta_csv.exists():
        return {"status": "not_run"}

    decoder_df = pd.read_csv(decoder_csv)
    delta_df = pd.read_csv(delta_csv)

    # Filter to only OK status
    valid_delta = delta_df[delta_df["status"] == "ok"] if "status" in delta_df.columns else delta_df
    valid_delta = valid_delta.dropna(subset=["delta"]) if "delta" in valid_delta.columns else valid_delta

    per_year = {}
    for _, row in decoder_df.iterrows():
        year = int(row["year"])
        per_year[year] = {
            "n_samples": int(row.get("n_samples", 0)),
            "cv_accuracy": float(row.get("cv_accuracy_mean", row.get("cv_accuracy", 0.0))),
            "macro_f1": float(row.get("cv_macro_f1_mean", row.get("macro_f1", 0.0))),
        }

    if len(valid_delta) > 0:
        delta_summary = {
            "n_total": int(len(valid_delta)),
            "n_improved": int((valid_delta["delta"] > 0).sum()),
            "n_degraded": int((valid_delta["delta"] < 0).sum()),
            "mean_delta": float(valid_delta["delta"].mean()),
            "median_delta": float(valid_delta["delta"].median()),
            "min_delta": float(valid_delta["delta"].min()),
            "max_delta": float(valid_delta["delta"].max()),
            "std_delta": float(valid_delta["delta"].std()),
        }
    else:
        delta_summary = {"status": "no_valid_pairs"}

    return {
        "status": "complete",
        "per_year": per_year,
        "delta_summary": delta_summary,
    }


def extract_e6() -> dict:
    """E6 30-area baseline summary."""
    paired_json = RESULTS_DIR / "e6_expanded_areas" / "expanded_paired_tests.json"
    per_area_csv = RESULTS_DIR / "e6_expanded_areas" / "expanded_per_area.csv"

    if not paired_json.exists():
        return {"status": "not_run"}

    with paired_json.open() as f:
        stats = json.load(f)

    per_area = {}
    if per_area_csv.exists():
        df = pd.read_csv(per_area_csv)
        n_pos = int((df["advantage"] > 0).sum())
        n_neg = int((df["advantage"] < 0).sum())
        per_area = {
            "n_total": int(len(df)),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "min_adv": float(df["advantage"].min()),
            "max_adv": float(df["advantage"].max()),
            "areas_pos": df[df["advantage"] > 0].sort_values("advantage", ascending=False)[["area", "advantage"]].head(10).to_dict("records"),
            "areas_neg_worst": df[df["advantage"] < 0].sort_values("advantage")[["area", "advantage"]].head(10).to_dict("records"),
        }

    # Check that min_years filter worked
    sources_json = RESULTS_DIR / "e6_expanded_areas" / "eval_area_sources.json"
    filter_check = {}
    if sources_json.exists():
        with sources_json.open() as f:
            sources = json.load(f)
        n_paper8 = sources["summary"]["roots"].get(
            [k for k in sources["summary"]["roots"] if "paper8" in k][0] if any("paper8" in k for k in sources["summary"]["roots"]) else "",
            0
        )
        n_indep = sources["summary"]["roots"].get(
            [k for k in sources["summary"]["roots"] if "independent_change" in k][0] if any("independent_change" in k for k in sources["summary"]["roots"]) else "",
            0
        )
        filter_check = {
            "n_paper8": n_paper8,
            "n_independent_change": n_indep,
            "filter_correct": n_indep == 0,  # Should be 0 if min_years=8 is applied
        }

    return {
        "status": "complete",
        "paired_tests": stats,
        "per_area": per_area,
        "filter_check": filter_check,
    }


def main():
    result = {
        "e3": extract_e3(),
        "e4": extract_e4(),
        "e6": extract_e6(),
    }

    # Print summary
    print("=" * 60)
    print("v4 Manuscript Numbers Extraction")
    print("=" * 60)

    print("\n--- E3 (Multi-Step Rollout) ---")
    e3 = result["e3"]
    if e3["status"] == "complete":
        print(f"  Areas: {e3['n_areas']}, Max steps: {e3['n_steps']}")
        print(f"  Bug fixed: {e3['bug_fixed']} (max persist step1 = {e3['max_persist_step1']:.4f})")
        for step, data in sorted(e3["per_step"].items()):
            print(f"  Step {step}: persist={data['persistence_mean']:.4f}, model={data['model_mean']:.4f}, adv={data['advantage_mean']:+.5f} ({data['n_pos']}+/{data['n_neg']}-)")
    else:
        print(f"  Status: {e3['status']}")

    print("\n--- E4 (Per-Year Decoder) ---")
    e4 = result["e4"]
    if e4["status"] == "complete":
        print(f"  Years trained: {sorted(e4['per_year'].keys())}")
        for year in sorted(e4["per_year"].keys()):
            data = e4["per_year"][year]
            print(f"  {year}: n={data['n_samples']}, acc={data['cv_accuracy']:.4f}, f1={data['macro_f1']:.4f}")
        if isinstance(e4["delta_summary"], dict) and "mean_delta" in e4["delta_summary"]:
            d = e4["delta_summary"]
            print(f"  Delta: mean={d['mean_delta']:+.4f}, {d['n_improved']}/{d['n_total']} improved")
    else:
        print(f"  Status: {e4['status']}")

    print("\n--- E6 (30-Area Baseline) ---")
    e6 = result["e6"]
    if e6["status"] == "complete":
        stats = e6["paired_tests"]
        print(f"  n = {stats['n']}")
        print(f"  mean = {stats['mean']:.6f}")
        print(f"  wilcoxon_p = {stats.get('wilcoxon_p', 'N/A'):.4f}" if isinstance(stats.get('wilcoxon_p'), (int, float)) else f"  wilcoxon_p = {stats.get('wilcoxon_p')}")
        print(f"  n_pos/n_neg: {e6['per_area'].get('n_pos', '?')}/{e6['per_area'].get('n_neg', '?')}")
        if e6.get("filter_check"):
            fc = e6["filter_check"]
            print(f"  Filter check: paper8={fc.get('n_paper8')}, indep={fc.get('n_independent_change')}, correct={fc.get('filter_correct')}")
    else:
        print(f"  Status: {e6['status']}")

    # Save to JSON
    OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    print(f"\n✅ Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
