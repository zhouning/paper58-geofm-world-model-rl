from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper58_benchmark.flus_native_gee_script import build_gee_javascript


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a GEE JavaScript exporter for FLUS-native task specs.")
    parser.add_argument("--export-specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drive-folder", default="paper58_flus_native")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    js = build_gee_javascript(
        export_specs_path=args.export_specs,
        drive_folder=args.drive_folder,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(js, encoding="utf-8")
    print(f"FLUS-native GEE JavaScript: wrote {args.output}")


if __name__ == "__main__":
    main()
