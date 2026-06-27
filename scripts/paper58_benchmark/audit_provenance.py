from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.paper58_benchmark.schema import DEFAULT_BENCHMARK_DIR, write_csv, write_json


AUDIT_FIELDS = [
    "area",
    "start_year",
    "end_year",
    "tier",
    "stratum",
    "development_contact_status",
    "contact_evidence",
    "qc_status",
    "audit_status",
]


def _normalized_string(value: object) -> str:
    return str(value).strip().lower() if isinstance(value, str) else str(value)


def _read_rows(path: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("benchmark registry must contain a rows list")
    normalized_rows: list[dict] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"registry row {index} must be an object")
        normalized = dict(row)
        normalized["tier"] = _normalized_string(normalized.get("tier"))
        normalized["development_contact_status"] = _normalized_string(normalized.get("development_contact_status"))
        if isinstance(normalized.get("contact_evidence"), str):
            normalized["contact_evidence"] = normalized["contact_evidence"].strip()
        normalized_rows.append(normalized)
    return normalized_rows


def _audit_status(row: dict) -> str:
    if row.get("tier") == "tier1" and row.get("development_contact_status") != "none":
        return "invalid_tier1_contact"
    if row.get("tier") == "tier1" and not str(row.get("contact_evidence", "")).strip():
        return "invalid_tier1_missing_evidence"
    if row.get("tier") == "review_required":
        return "review_required"
    return "ok"


def audit_registry_provenance(
    registry_path: Path = DEFAULT_BENCHMARK_DIR / "benchmark_registry.json",
    output_dir: Path = DEFAULT_BENCHMARK_DIR,
) -> dict:
    rows = _read_rows(Path(registry_path))
    audit_rows = []
    for row in rows:
        audit_rows.append(
            {
                "area": row.get("area"),
                "start_year": row.get("start_year"),
                "end_year": row.get("end_year"),
                "tier": row.get("tier"),
                "stratum": row.get("stratum"),
                "development_contact_status": row.get("development_contact_status"),
                "contact_evidence": row.get("contact_evidence"),
                "qc_status": row.get("qc_status"),
                "audit_status": _audit_status(row),
            }
        )
    invalid = [row for row in audit_rows if str(row["audit_status"]).startswith("invalid_tier1")]
    report = {
        "n_rows": len(rows),
        "tier_counts": dict(sorted(Counter(str(row.get("tier")) for row in rows).items())),
        "invalid_tier1_rows": invalid,
    }
    output_dir = Path(output_dir)
    write_json(output_dir / "benchmark_provenance_audit.json", report)
    write_csv(output_dir / "benchmark_provenance_audit.csv", audit_rows, AUDIT_FIELDS)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Paper58 benchmark provenance tiers.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_BENCHMARK_DIR / "benchmark_registry.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    args = parser.parse_args()
    report = audit_registry_provenance(args.registry, args.output_dir)
    print(
        "Benchmark provenance audit: "
        f"{report['n_rows']} row(s), "
        f"{len(report['invalid_tier1_rows'])} invalid Tier 1 row(s)"
    )


if __name__ == "__main__":
    main()
