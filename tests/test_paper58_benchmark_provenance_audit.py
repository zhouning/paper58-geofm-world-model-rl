import csv
import json
from pathlib import Path

from scripts.paper58_benchmark.audit_provenance import audit_registry_provenance


def test_audit_registry_provenance_writes_json_and_csv(tmp_path: Path):
    registry_path = tmp_path / "benchmark_registry.json"
    output_dir = tmp_path / "out"
    registry_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "strict_external",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Urban",
                        "development_contact_status": "none",
                        "contact_evidence": "toy no-contact evidence",
                        "qc_status": "include",
                    },
                    {
                        "area": "poyang_lake",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier2",
                        "stratum": "Wetland",
                        "development_contact_status": "known_contact",
                        "contact_evidence": "training-list membership",
                        "qc_status": "include",
                    },
                    {
                        "area": "uncertain_external",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "review_required",
                        "stratum": "Mixed",
                        "development_contact_status": "uncertain",
                        "contact_evidence": "not cleared",
                        "qc_status": "include",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = audit_registry_provenance(registry_path=registry_path, output_dir=output_dir)

    assert report["n_rows"] == 3
    assert report["tier_counts"] == {"review_required": 1, "tier1": 1, "tier2": 1}
    assert report["invalid_tier1_rows"] == []
    assert json.loads((output_dir / "benchmark_provenance_audit.json").read_text(encoding="utf-8")) == report
    with (output_dir / "benchmark_provenance_audit.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["audit_status"] for row in rows] == ["ok", "ok", "review_required"]


def test_audit_registry_provenance_reports_invalid_tier1_rows(tmp_path: Path):
    registry_path = tmp_path / "benchmark_registry.json"
    output_dir = tmp_path / "out"
    registry_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "known_contact_external",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Urban",
                        "development_contact_status": "known_contact",
                        "contact_evidence": "training-list membership",
                        "qc_status": "include",
                    },
                    {
                        "area": "missing_evidence_external",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": "tier1",
                        "stratum": "Wetland",
                        "development_contact_status": "none",
                        "contact_evidence": "   ",
                        "qc_status": "include",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = audit_registry_provenance(registry_path=registry_path, output_dir=output_dir)

    assert [row["audit_status"] for row in report["invalid_tier1_rows"]] == [
        "invalid_tier1_contact",
        "invalid_tier1_missing_evidence",
    ]
    with (output_dir / "benchmark_provenance_audit.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["audit_status"] for row in rows] == [
        "invalid_tier1_contact",
        "invalid_tier1_missing_evidence",
    ]


def test_audit_registry_provenance_normalizes_like_evaluator(tmp_path: Path):
    registry_path = tmp_path / "benchmark_registry.json"
    output_dir = tmp_path / "out"
    registry_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "area": "strict_external",
                        "start_year": 2020,
                        "end_year": 2021,
                        "tier": " Tier1 ",
                        "stratum": "Urban",
                        "development_contact_status": " None ",
                        "contact_evidence": "  normalized evidence  ",
                        "qc_status": "include",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = audit_registry_provenance(registry_path=registry_path, output_dir=output_dir)

    assert report["tier_counts"] == {"tier1": 1}
    assert report["invalid_tier1_rows"] == []
    with (output_dir / "benchmark_provenance_audit.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["tier"] == "tier1"
    assert rows[0]["development_contact_status"] == "none"
    assert rows[0]["contact_evidence"] == "normalized evidence"
    assert rows[0]["audit_status"] == "ok"


def test_audit_registry_provenance_rejects_non_object_rows(tmp_path: Path):
    registry_path = tmp_path / "benchmark_registry.json"
    registry_path.write_text(json.dumps({"rows": [{"area": "ok"}, 1]}), encoding="utf-8")

    try:
        audit_registry_provenance(registry_path=registry_path, output_dir=tmp_path / "out")
    except ValueError as exc:
        assert str(exc) == "registry row 1 must be an object"
    else:
        raise AssertionError("expected non-object registry row to raise")
