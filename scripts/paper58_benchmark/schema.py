from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_DIR = ROOT / "paper" / "rse_submission_paper58" / "benchmark_results"
DEFAULT_LABELS_DIR = ROOT / "data" / "independent_change_labels" / "labels"
DEFAULT_PREDICTIONS_DIR = ROOT / "data" / "independent_change_labels" / "predicted"
DEFAULT_INDEPENDENT_EMBEDDINGS_DIR = ROOT / "data" / "independent_change_labels" / "embeddings"
DEFAULT_EXPERIMENT_DATA_DIR = ROOT / "experiments" / "paper8" / "data"

DEVELOPMENT_AREAS = {"banzhucun", "bishan", "heping"}

AREA_STRATA = {
    "yangtze_delta": "Urban",
    "jing_jin_ji": "Urban",
    "chengdu_plain": "Urban",
    "pearl_river": "Urban",
    "northeast_plain": "Agriculture",
    "north_china_plain": "Agriculture",
    "jianghan_plain": "Agriculture",
    "hetao": "Agriculture",
    "daxinganling": "Forest",
    "wuyi_mountain": "Forest",
    "qinghai_edge": "Plateau",
    "guanzhong": "Mixed",
    "minnan_coast": "Mixed",
    "bishan": "Mixed",
    "banzhucun": "Mixed",
    "heping": "Mixed",
    "poyang_lake": "Wetland",
    "yunnan_eco": "Ecology",
}


@dataclass(frozen=True)
class BenchmarkRow:
    area: str
    start_year: int
    end_year: int
    tier: str
    stratum: str
    label_start_path: Path | None
    label_end_path: Path | None
    prediction_path: Path | None
    embedding_start_path: Path | None
    embedding_end_path: Path | None
    context_path: Path | None
    label_shape: tuple[int, ...] | None
    prediction_shape: tuple[int, ...] | None
    embedding_shape: tuple[int, ...] | None
    n_pixels: int
    true_change_pixels: int
    true_change_pct: float
    qc_status: str
    excluded_reason: str


def assign_tier(area: str) -> str:
    return "tier2" if area.lower() in DEVELOPMENT_AREAS else "tier1"


def area_stratum(area: str) -> str:
    return AREA_STRATA.get(area.lower(), "Unknown")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def row_to_dict(row: BenchmarkRow) -> dict[str, Any]:
    return _json_ready(asdict(row))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
