from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adk_world_model.world_model import (
    N_CONTEXT,
    SCENARIOS,
    SCENARIO_DIM,
    WEIGHTS_PATH,
    DECODER_PATH,
    _build_model,
)
from scripts.paper58_benchmark.holdouts import area_records_for_status, load_holdout_manifest


DEFAULT_EMBEDDING_DIRS = [
    ROOT / "experiments" / "paper8" / "data",
    ROOT / "experiments" / "paper8" / "data" / "village",
    ROOT / "experiments" / "paper8" / "data" / "heping",
]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "independent_change_labels" / "predicted"
DEFAULT_REPORT = ROOT / "data" / "independent_change_labels" / "prediction_readiness_report.json"


def _load_torch():
    import torch

    return torch


def _load_decoder(path: Path):
    import joblib

    return joblib.load(path)


def _find_embeddings(embedding_dirs: list[Path]) -> dict[str, dict[int, Path]]:
    found: dict[str, dict[int, Path]] = {}
    for directory in embedding_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*_emb_*.npy")):
            stem = path.stem
            if stem.endswith("_50m"):
                continue
            parts = stem.rsplit("_emb_", 1)
            if len(parts) != 2:
                continue
            area, year_raw = parts
            try:
                year = int(year_raw)
            except ValueError:
                continue
            normalized_area = "banzhucun" if area == "village" else area
            found.setdefault(normalized_area, {})[year] = path
    return found


def _find_context(area: str, embedding_path: Path) -> Path | None:
    candidates = [
        embedding_path.parent / f"{area}_context.npy",
        embedding_path.parent / "village_context.npy" if area == "banzhucun" else embedding_path.parent / f"{area}_context.npy",
        embedding_path.parent / "bishan_context.npy",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _load_model(weights_path: Path):
    torch = _load_torch()
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = _build_model(
        checkpoint.get("z_dim", 64),
        checkpoint.get("scenario_dim", SCENARIO_DIM),
        checkpoint.get("n_context", N_CONTEXT),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _predict_next_embedding(model, emb: np.ndarray, context: np.ndarray | None) -> np.ndarray:
    torch = _load_torch()
    scenario = np.zeros(SCENARIO_DIM, dtype=np.float32)
    scenario[SCENARIOS["baseline"].id] = 1.0
    z = torch.tensor(emb.transpose(2, 0, 1)).unsqueeze(0).float()
    scenario_t = torch.tensor(scenario).unsqueeze(0)
    context_t = torch.tensor(context).unsqueeze(0).float() if context is not None else None
    with torch.no_grad():
        pred = model(z, scenario_t, context_t)
        pred = torch.nn.functional.normalize(pred, p=2, dim=1)
    return pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)


def _decode_lulc(embedding_grid: np.ndarray, decoder) -> np.ndarray:
    h, w = embedding_grid.shape[:2]
    return decoder.predict(embedding_grid.reshape(-1, 64)).reshape(h, w).astype(np.int32)


def generate_change_validation_predictions(
    embedding_dirs: list[Path] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT,
    weights_path: Path = Path(WEIGHTS_PATH),
    decoder_path: Path = Path(DECODER_PATH),
    areas: list[str] | None = None,
    area_manifest_path: Path | None = None,
    overwrite: bool = False,
) -> dict:
    embedding_dirs = embedding_dirs or DEFAULT_EMBEDDING_DIRS
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    weights_path = Path(weights_path)
    decoder_path = Path(decoder_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    readiness_failures = []
    if not weights_path.exists():
        readiness_failures.append({"component": "latent_dynamics_weights", "path": str(weights_path), "reason": "missing"})
    if not decoder_path.exists():
        readiness_failures.append({"component": "lulc_decoder", "path": str(decoder_path), "reason": "missing"})

    manifest_areas: list[str] | None = None
    if areas is None and area_manifest_path is not None:
        manifest_areas = [record.area for record in load_holdout_manifest(Path(area_manifest_path)) if record.development_contact_status == "none"]
        if not manifest_areas:
            readiness_failures.append(
                {
                    "component": "cached_embeddings",
                    "paths": [str(path) for path in embedding_dirs],
                    "reason": "no_embedding_sequences_found",
                    "candidate_areas": [],
                }
            )
            report = {
                "status": "not_ready",
                "n_predictions": 0,
                "readiness_failures": readiness_failures,
                "next_step": (
                    "Provide LatentDynamicsNet weights, a fitted LULC decoder, and cached annual embeddings; "
                    "then rerun this script before evaluating independent change labels."
                ),
            }
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            return report
        areas = manifest_areas

    embeddings = _find_embeddings([Path(p) for p in embedding_dirs])
    if areas:
        wanted = {area.lower() for area in areas}
        embeddings = {area: years for area, years in embeddings.items() if area.lower() in wanted}
    if not embeddings:
        readiness_failures.append(
            {
                "component": "cached_embeddings",
                "paths": [str(path) for path in embedding_dirs],
                "reason": "no_embedding_sequences_found",
                "candidate_areas": sorted(areas or []),
            }
        )

    if readiness_failures:
        report = {
            "status": "not_ready",
            "n_predictions": 0,
            "readiness_failures": readiness_failures,
            "next_step": (
                "Provide LatentDynamicsNet weights, a fitted LULC decoder, and cached annual embeddings; "
                "then rerun this script before evaluating independent change labels."
            ),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    model = _load_model(weights_path)
    decoder = _load_decoder(decoder_path)
    records = []
    skipped = []
    for area, years_to_path in sorted(embeddings.items()):
        years = sorted(years_to_path)
        for start_year, end_year in zip(years, years[1:]):
            out_path = output_dir / f"{area}_lulc_pred_{start_year}_{end_year}.npy"
            if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
                records.append({"area": area, "start_year": start_year, "end_year": end_year, "path": str(out_path), "status": "cached"})
                continue
            emb = np.load(years_to_path[start_year]).astype(np.float32, copy=False)
            context_path = _find_context(area, years_to_path[start_year])
            context = np.load(context_path).astype(np.float32, copy=False) if context_path else None
            if context is not None and (context.shape[1] != emb.shape[0] or context.shape[2] != emb.shape[1]):
                skipped.append(
                    {
                        "area": area,
                        "start_year": start_year,
                        "end_year": end_year,
                        "reason": "context_shape_mismatch",
                        "embedding_shape": list(emb.shape),
                        "context_shape": list(context.shape),
                    }
                )
                continue
            pred_emb = _predict_next_embedding(model, emb, context)
            pred_lulc = _decode_lulc(pred_emb, decoder)
            np.save(out_path, pred_lulc)
            records.append(
                {
                    "area": area,
                    "start_year": start_year,
                    "end_year": end_year,
                    "embedding_file": str(years_to_path[start_year]),
                    "context_file": str(context_path) if context_path else None,
                    "path": str(out_path),
                    "shape": list(pred_lulc.shape),
                    "status": "generated",
                }
            )

    report = {
        "status": "complete" if records else "no_predictions",
        "n_predictions": len(records),
        "n_skipped": len(skipped),
        "weights_path": str(weights_path),
        "decoder_path": str(decoder_path),
        "output_dir": str(output_dir),
        "records": records,
        "skipped": skipped,
        "next_step": "Run scripts/rse_revision/evaluate_independent_change_validation.py.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predicted LULC maps for Paper58 independent change validation.")
    parser.add_argument("--embedding-dir", type=Path, action="append", dest="embedding_dirs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--weights", type=Path, default=Path(WEIGHTS_PATH))
    parser.add_argument("--decoder", type=Path, default=Path(DECODER_PATH))
    parser.add_argument("--areas", default="", help="Optional comma-separated area filter.")
    parser.add_argument("--area-manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    areas = [item.strip().lower() for item in args.areas.split(",") if item.strip()] or None
    report = generate_change_validation_predictions(
        embedding_dirs=args.embedding_dirs,
        output_dir=args.output_dir,
        report_path=args.report,
        weights_path=args.weights,
        decoder_path=args.decoder,
        areas=areas,
        area_manifest_path=args.area_manifest,
        overwrite=args.overwrite,
    )
    print(
        "Change-validation prediction generation: "
        f"{report['status']}, "
        f"{report.get('n_predictions', 0)} prediction(s)"
    )


if __name__ == "__main__":
    main()
