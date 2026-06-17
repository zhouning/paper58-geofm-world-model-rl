from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "paper" / "rse_submission_paper58" / "revision_results"
DEFAULT_DECODER_RESULTS = ROOT / "src" / "adk_world_model" / "experiments" / "output" / "world_model_lulc_decode.json"
DEFAULT_INDEPENDENT_CHANGE_LABELS_DIR = ROOT / "data" / "independent_change_labels"
DEFAULT_INDEPENDENT_CHANGE_SUMMARY = DEFAULT_RESULTS_DIR / "independent_change_validation_summary.json"


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _load_json(path: Path) -> dict:
    if not _exists(path):
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_status(paths: list[Path], expected_summary: bool = True) -> str:
    if not paths:
        return "missing"
    if all(_exists(path) for path in paths):
        return "complete" if expected_summary else "diagnostic"
    return "missing"


def _relative_paths(paths: list[Path]) -> list[str]:
    rel_paths = []
    for path in paths:
        try:
            rel_paths.append(str(path.relative_to(ROOT)))
        except ValueError:
            rel_paths.append(str(path))
    return rel_paths


def _detect_independent_change_label_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    patterns = ("*.csv", "*.json", "*.geojson", "*.gpkg", "*.tif", "*.tiff", "*.parquet")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in directory.rglob(pattern) if path.is_file() and path.stat().st_size > 0)
    return sorted(files)


def _stage(
    stage_id: str,
    question: str,
    status: str,
    claim_supported: bool,
    evidence_files: list[Path],
    manuscript_use: str,
    next_empirical_step: str,
    metric_summary: str = "",
) -> dict:
    return {
        "stage_id": stage_id,
        "question": question,
        "status": status,
        "claim_supported": bool(claim_supported),
        "metric_summary": metric_summary,
        "evidence_files": _relative_paths(evidence_files),
        "manuscript_use": manuscript_use,
        "next_empirical_step": next_empirical_step,
    }


def build_empirical_pipeline_stages(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    decoder_path: Path = DEFAULT_DECODER_RESULTS,
    independent_change_labels_dir: Path = DEFAULT_INDEPENDENT_CHANGE_LABELS_DIR,
) -> list[dict]:
    summary_path = results_dir / "revision_summary.json"
    summary = _load_json(summary_path)

    alpha_path = results_dir / "alphaearth_area_metrics.csv"
    category_path = results_dir / "alphaearth_category_summary.csv"
    prithvi_path = results_dir / "prithvi_area_metrics.csv"
    planning_path = results_dir / "planning_baseline_summary.csv"
    transfer_path = results_dir / "transfer_planning_summary.csv"
    independent_change_summary_path = results_dir / "independent_change_validation_summary.json"
    independent_change_by_area_path = results_dir / "independent_change_validation_by_area.csv"

    alpha_summary = summary.get("alphaearth", {})
    alpha_advantage = alpha_summary.get("advantage", {})
    alpha_metric = ""
    if alpha_summary:
        alpha_metric = (
            f"n={alpha_summary.get('n_areas')}; "
            f"mean advantage={alpha_advantage.get('mean')}; "
            f"positive areas={len(alpha_summary.get('areas_positive', []))}"
        )

    prithvi_summary = summary.get("prithvi", {})
    prithvi_metric = ""
    if prithvi_summary:
        prithvi_metric = (
            f"n={prithvi_summary.get('n_areas')}; "
            f"mean advantage={prithvi_summary.get('advantage', {}).get('mean')}"
        )

    decoder_summary = _load_json(decoder_path)
    decoder_metric = ""
    if decoder_summary:
        decoder_metric = (
            f"n={decoder_summary.get('n_samples')}; "
            f"overall accuracy={decoder_summary.get('overall_accuracy')}"
        )

    planning_summary = summary.get("planning_baselines", {})
    planning_metric = ""
    if planning_summary:
        full = planning_summary.get("full", {})
        dropout = planning_summary.get("dropout0.3", {})
        emb_only = planning_summary.get("dropout1.0", {})
        planning_metric = (
            f"full slope={full.get('slope_mean')}; "
            f"dropout0.3 retention={dropout.get('slope_retention_vs_full')}; "
            f"embedding-only retention={emb_only.get('slope_retention_vs_full')}"
        )

    transfer_summary = summary.get("transfer_planning", {})
    transfer_metric = ""
    if transfer_summary:
        transfer = transfer_summary.get("transfer", {})
        transfer_metric = (
            f"n={transfer.get('n')}; "
            f"delta vs random={transfer.get('reward_delta_vs_random')}; "
            f"delta vs greedy={transfer.get('reward_delta_vs_greedy')}"
        )

    independent_label_files = _detect_independent_change_label_files(independent_change_labels_dir)
    independent_change_summary = _load_json(independent_change_summary_path)
    n_independent_pairs = int(independent_change_summary.get("n_area_year_pairs") or 0)
    independent_change_complete = n_independent_pairs > 0 and _exists(independent_change_summary_path)
    independent_change_metric = f"n_files={len(independent_label_files)}"
    if independent_change_summary:
        mean_metrics = independent_change_summary.get("mean_metrics", {})
        independent_change_metric = (
            f"n_pairs={n_independent_pairs}; "
            f"model change F1={mean_metrics.get('model_change_f1')}; "
            f"shuffled change F1={mean_metrics.get('shuffled_model_change_f1')}; "
            f"transition-prior change F1={mean_metrics.get('transition_prior_change_f1')}; "
            f"persistence change F1={mean_metrics.get('persistence_change_f1')}; "
            f"changed-pixel accuracy={mean_metrics.get('model_changed_pixel_accuracy')}"
        )

    return [
        _stage(
            "area_level_embedding_dynamics",
            "Does LatentDynamicsNet improve annual embedding prediction over persistence?",
            _evidence_status([summary_path, alpha_path, category_path], expected_summary=True),
            _exists(summary_path) and _exists(alpha_path),
            [summary_path, alpha_path, category_path],
            "Primary evidence for bounded baseline-trend embedding dynamics.",
            "Expand valid cached grids and report area-level uncertainty for additional regions.",
            alpha_metric,
        ),
        _stage(
            "encoder_replacement_ablation",
            "Is the result explained merely by using any frozen encoder?",
            _evidence_status([summary_path, prithvi_path], expected_summary=True),
            _exists(summary_path) and _exists(prithvi_path),
            [summary_path, prithvi_path],
            "Ablation evidence that the tested Prithvi CLS state is nearly stationary in this workflow.",
            "Test local token or dense Prithvi features if a like-for-like gridded representation is available.",
            prithvi_metric,
        ),
        _stage(
            "semantic_decoder_validation",
            "Can AlphaEarth embeddings be linked to LULC semantics for visualization?",
            _evidence_status([decoder_path], expected_summary=False),
            _exists(decoder_path),
            [decoder_path],
            "Diagnostic support for qualitative decoded maps, not proof of predicted categorical transitions.",
            "Validate predicted transitions against independent annual LULC labels or visual interpretation.",
            decoder_metric,
        ),
        _stage(
            "planning_feature_dropout",
            "Do GeoFM embeddings add robustness when planning features are partially unavailable?",
            _evidence_status([summary_path, planning_path], expected_summary=False),
            _exists(summary_path) and _exists(planning_path),
            [summary_path, planning_path],
            "Application probe showing embeddings are complementary but insufficient alone.",
            "Run cross-region planning on target regions with locally verified planning constraints.",
            planning_metric,
        ),
        _stage(
            "embedding_space_transfer_probe",
            "Is there measurable zero-shot signal in an embedding-space planning transfer probe?",
            _evidence_status([summary_path, transfer_path], expected_summary=False),
            _exists(summary_path) and _exists(transfer_path),
            [summary_path, transfer_path],
            "Weak transfer diagnostic within the embedding-space reward system.",
            "Repeat transfer under real county-level slope and contiguity metrics.",
            transfer_metric,
        ),
        _stage(
            "independent_change_label_validation",
            "Do predicted embedding changes correspond to independently labelled land-cover transitions?",
            "complete" if independent_change_complete else "missing",
            independent_change_complete,
            [independent_change_summary_path, independent_change_by_area_path]
            if independent_change_complete
            else independent_label_files,
            "Transition-level validation against independent annual LULC labels."
            if independent_change_complete
            else "Not used as a completed result in the current manuscript.",
            "Expand independent annual change labels across more regions and report transition-level uncertainty."
            if independent_change_complete
            else "Acquire or derive annual independent change labels and model-predicted LULC maps, then report transition-level accuracy.",
            independent_change_metric,
        ),
    ]


def write_audit_csv(path: Path, stages: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage_id",
        "question",
        "status",
        "claim_supported",
        "metric_summary",
        "evidence_files",
        "manuscript_use",
        "next_empirical_step",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stage in stages:
            writer.writerow(
                {
                    **stage,
                    "evidence_files": ";".join(stage["evidence_files"]),
                }
            )


def audit_empirical_pipeline(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    decoder_path: Path = DEFAULT_DECODER_RESULTS,
    independent_change_labels_dir: Path = DEFAULT_INDEPENDENT_CHANGE_LABELS_DIR,
    output_dir: Path = DEFAULT_RESULTS_DIR,
) -> dict:
    stages = build_empirical_pipeline_stages(results_dir, decoder_path, independent_change_labels_dir)
    summary = {
        "n_stages": len(stages),
        "n_complete": sum(stage["status"] == "complete" for stage in stages),
        "n_diagnostic": sum(stage["status"] == "diagnostic" for stage in stages),
        "n_missing": sum(stage["status"] == "missing" for stage in stages),
        "independent_change_label_validation_complete": any(
            stage["stage_id"] == "independent_change_label_validation" and stage["status"] == "complete"
            for stage in stages
        ),
    }
    report = {
        "summary": summary,
        "stages": stages,
        "notes": [
            "Complete stages have data files that directly support a manuscript result.",
            "Diagnostic stages support bounded interpretation or visualization but not standalone operational claims.",
            "Missing stages are not used as completed evidence in the manuscript.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "empirical_pipeline_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_audit_csv(output_dir / "empirical_pipeline_audit.csv", stages)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Paper58 empirical evidence chain.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--decoder-results", type=Path, default=DEFAULT_DECODER_RESULTS)
    parser.add_argument("--independent-change-labels-dir", type=Path, default=DEFAULT_INDEPENDENT_CHANGE_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    report = audit_empirical_pipeline(
        args.results_dir,
        args.decoder_results,
        args.independent_change_labels_dir,
        args.output_dir,
    )
    print(
        "Empirical pipeline audit: "
        f"{report['summary']['n_complete']} complete, "
        f"{report['summary']['n_diagnostic']} diagnostic, "
        f"{report['summary']['n_missing']} missing"
    )


if __name__ == "__main__":
    main()
