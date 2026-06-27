from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper58_benchmark.flus_native_admin import (
    build_admin_candidate_manifest,
    build_stratified_admin_candidate_manifest,
)


def write_manifest(path: Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Paper58 FLUS-native administrative candidate manifest.")
    parser.add_argument("--shp", type=Path, required=True, help="Path to township shapefile.")
    parser.add_argument("--output", type=Path, required=True, help="Output manifest JSON path.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows after filtering.")
    parser.add_argument("--province", default=None, help="Optional exact province-name filter.")
    parser.add_argument("--target-scale-m", type=int, default=100, help="Planned FLUS raster scale in meters.")
    parser.add_argument("--encoding", default="utf-8", help="DBF text encoding.")
    parser.add_argument("--stratified-sample-size", type=int, default=None, help="Build a deterministic stratified sample instead of taking rows in source order.")
    parser.add_argument("--min-pixels", type=int, default=2_500, help="Minimum estimated raster pixels for stratified sampling.")
    parser.add_argument("--max-pixels", type=int, default=80_000, help="Maximum estimated raster pixels for stratified sampling.")
    parser.add_argument("--min-width-px", type=int, default=32, help="Minimum estimated raster width for stratified sampling.")
    parser.add_argument("--min-height-px", type=int, default=32, help="Minimum estimated raster height for stratified sampling.")
    parser.add_argument("--max-per-province", type=int, default=2, help="Maximum selected stratified samples per province.")
    parser.add_argument("--seed", default="paper58-flus-native-admin-v1", help="Stable seed for deterministic stratified sampling.")
    args = parser.parse_args()

    if args.stratified_sample_size is not None:
        manifest = build_stratified_admin_candidate_manifest(
            shp_path=args.shp,
            sample_size=args.stratified_sample_size,
            province=args.province,
            target_scale_m=args.target_scale_m,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            min_width_px=args.min_width_px,
            min_height_px=args.min_height_px,
            max_per_province=args.max_per_province,
            seed=args.seed,
            encoding=args.encoding,
        )
    else:
        manifest = build_admin_candidate_manifest(
            shp_path=args.shp,
            limit=args.limit,
            province=args.province,
            target_scale_m=args.target_scale_m,
            encoding=args.encoding,
        )
    write_manifest(args.output, manifest)
    print(
        "FLUS-native admin manifest: "
        f"{manifest['summary']['n_rows']}/{manifest['summary']['source_records']} row(s), "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
