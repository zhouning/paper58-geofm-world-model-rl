from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper58_benchmark.flus_native_gee import build_gee_export_specs


def write_specs(path: Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Paper58 FLUS-native GEE export task specs.")
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-root", default="data/flus_native_admin")
    args = parser.parse_args()

    specs = build_gee_export_specs(
        candidate_manifest_path=args.candidate_manifest,
        start_year=args.start_year,
        end_year=args.end_year,
        limit=args.limit,
        output_root=args.output_root,
    )
    write_specs(args.output, specs)
    print(
        "FLUS-native GEE export specs: "
        f"{specs['summary']['n_tasks']} task(s), output={args.output}"
    )


if __name__ == "__main__":
    main()
