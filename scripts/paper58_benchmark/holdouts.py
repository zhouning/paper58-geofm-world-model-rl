from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOLDOUT_MANIFEST = ROOT / "data" / "independent_change_labels" / "paper58_holdout_areas.json"

VALID_CONTACT_STATUS = {"none", "known_contact", "uncertain"}
KNOWN_TRAINING_AREAS = {
    "yangtze_delta",
    "jing_jin_ji",
    "pearl_river",
    "chengdu_plain",
    "northeast_plain",
    "north_china_plain",
    "jianghan_plain",
    "hetao",
    "yunnan_eco",
    "daxinganling",
    "qinghai_edge",
    "wuyi_mountain",
    "guanzhong",
    "minnan_coast",
    "poyang_lake",
}
KNOWN_DEVELOPMENT_AREAS = {"banzhucun", "bishan", "heping"}
REQUIRED_FIELDS = {
    "area",
    "bbox",
    "stratum",
    "years",
    "data_source",
    "selection_reason",
    "development_contact_status",
    "contact_evidence",
    "expected_role",
    "notes",
}


@dataclass(frozen=True)
class HoldoutArea:
    area: str
    bbox: tuple[float, float, float, float]
    stratum: str
    years: tuple[int, ...]
    data_source: str
    selection_reason: str
    development_contact_status: str
    contact_evidence: str
    expected_role: str
    notes: str

    def as_area_record(self) -> dict[str, Any]:
        return {"name": self.area, "bbox": list(self.bbox)}


def _require_string(row: dict[str, Any], field: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ValueError(f"holdout area {index} field {field} must be a string")
    return value.strip()


def _is_number(value: Any) -> bool:
    return type(value) in (int, float)


def _is_integer(value: Any) -> bool:
    return type(value) is int


def _parse_area(row: dict[str, Any], index: int) -> HoldoutArea:
    if not isinstance(row, dict):
        raise ValueError(f"holdout area {index} must be an object")

    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        raise ValueError(f"holdout area {index} missing required field: {missing[0]}")

    area = _require_string(row, "area", index).lower()
    bbox_raw = row["bbox"]
    years_raw = row["years"]
    if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
        raise ValueError(f"holdout area {index} bbox must contain four numbers")
    if not all(_is_number(value) for value in bbox_raw):
        raise ValueError(f"holdout area {index} bbox must contain four numbers")
    if not isinstance(years_raw, list) or not years_raw:
        raise ValueError(f"holdout area {index} years must be a non-empty list")
    if not all(_is_integer(value) for value in years_raw):
        raise ValueError(f"holdout area {index} years must contain integers")

    status = _require_string(row, "development_contact_status", index)
    if status not in VALID_CONTACT_STATUS:
        raise ValueError(f"holdout area {index} invalid development_contact_status: {status}")

    contact_evidence = _require_string(row, "contact_evidence", index)
    if not contact_evidence:
        raise ValueError(f"holdout area {index} contact_evidence must be non-empty")

    return HoldoutArea(
        area=area,
        bbox=tuple(float(value) for value in bbox_raw),
        stratum=_require_string(row, "stratum", index),
        years=tuple(sorted(set(int(value) for value in years_raw))),
        data_source=_require_string(row, "data_source", index),
        selection_reason=_require_string(row, "selection_reason", index),
        development_contact_status=status,
        contact_evidence=contact_evidence,
        expected_role=_require_string(row, "expected_role", index),
        notes=_require_string(row, "notes", index),
    )


def load_holdout_manifest(path: Path = DEFAULT_HOLDOUT_MANIFEST) -> list[HoldoutArea]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("holdout manifest must contain an object")
    areas = payload.get("areas")
    if not isinstance(areas, list):
        raise ValueError("holdout manifest must contain an 'areas' list")
    parsed = [_parse_area(row, index) for index, row in enumerate(areas)]
    seen: set[str] = set()
    for area in parsed:
        if area.area in seen:
            raise ValueError(f"duplicate holdout area: {area.area}")
        seen.add(area.area)
    return parsed


def manifest_lookup(areas: list[HoldoutArea]) -> dict[str, HoldoutArea]:
    return {area.area: area for area in areas}


def tier_from_provenance(area: str, lookup: dict[str, HoldoutArea]) -> str:
    normalized = area.lower()
    if normalized in KNOWN_TRAINING_AREAS or normalized in KNOWN_DEVELOPMENT_AREAS:
        return "tier2"
    record = lookup.get(normalized)
    if record is None:
        return "review_required"
    if record.development_contact_status == "none":
        return "tier1"
    if record.development_contact_status == "known_contact":
        return "tier2"
    return "review_required"


def area_records_for_status(
    manifest_path: Path = DEFAULT_HOLDOUT_MANIFEST,
    statuses: set[str] | None = None,
) -> list[dict[str, Any]]:
    wanted = {"none"} if statuses is None else statuses
    return [
        area.as_area_record()
        for area in load_holdout_manifest(manifest_path)
        if area.development_contact_status in wanted
    ]
