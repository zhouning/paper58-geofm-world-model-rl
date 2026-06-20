from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.paper58_benchmark.holdouts import HoldoutArea, load_holdout_manifest


def _row(area: HoldoutArea) -> dict:
    return {
        "area": area.area,
        "bbox": list(area.bbox),
        "stratum": area.stratum,
        "years": list(area.years),
        "data_source": area.data_source,
        "selection_reason": area.selection_reason,
        "development_contact_status": area.development_contact_status,
        "contact_evidence": area.contact_evidence,
        "expected_role": area.expected_role,
        "notes": area.notes,
    }


def build_combined_holdout_manifest(
    manifest_paths: list[Path],
    output_path: Path,
) -> list[HoldoutArea]:
    combined: list[HoldoutArea] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        for area in load_holdout_manifest(Path(manifest_path)):
            if area.area in seen:
                raise ValueError(f"duplicate combined holdout area: {area.area}")
            seen.add(area.area)
            combined.append(area)

    payload = {
        "version": 1,
        "created": "2026-06-20",
        "purpose": "Paper58 combined Batch 1 + Batch 2 holdout manifest",
        "source_manifests": [str(Path(path)) for path in manifest_paths],
        "areas": [_row(area) for area in combined],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a combined Paper58 holdout manifest.")
    parser.add_argument("--manifest", dest="manifests", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    combined = build_combined_holdout_manifest(args.manifests, args.output)
    print(f"Combined holdout manifest: {len(combined)} area(s)")


if __name__ == "__main__":
    main()
