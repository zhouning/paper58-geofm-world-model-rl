from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _write_candidate_manifest(path: Path) -> None:
    payload = {
        "version": 1,
        "summary": {"source_records": 2, "n_rows": 2},
        "rows": [
            {
                "area_id": "xiangzhen_record_000000",
                "source_shp_path": "/Users/zhouning/Downloads/shp/xiangzhen.shp",
                "record_index": 0,
                "province": "河南省",
                "city": "南阳市",
                "county": "桐柏县",
                "town": "吴城镇",
                "bbox": [113.42, 32.37, 113.59, 32.52],
                "target_scale_m": 100,
                "estimated_width_px": 157,
                "estimated_height_px": 161,
                "tags": ["admin_township", "flus_native_candidate"],
            },
            {
                "area_id": "xiangzhen_record_000001",
                "source_shp_path": "/Users/zhouning/Downloads/shp/xiangzhen.shp",
                "record_index": 1,
                "province": "河北省",
                "city": "保定市",
                "county": "安新县",
                "town": "端村镇",
                "bbox": [115.80, 38.80, 115.95, 38.96],
                "target_scale_m": 100,
                "estimated_width_px": 132,
                "estimated_height_px": 178,
                "tags": ["admin_township", "flus_native_candidate"],
            },
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class FlusNativeGeeExportSpecTests(unittest.TestCase):
    def test_export_specs_include_flus_native_raster_stack_and_configurable_drivers(self) -> None:
        from scripts.paper58_benchmark.flus_native_gee import build_gee_export_specs

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "candidates.json"
            _write_candidate_manifest(manifest_path)

            specs = build_gee_export_specs(
                candidate_manifest_path=manifest_path,
                start_year=2020,
                end_year=2021,
                limit=1,
                output_root="data/flus_native_admin",
            )

        self.assertEqual(specs["summary"]["n_tasks"], 1)
        self.assertEqual(
            specs["gee_sources"]["alphaearth"], "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
        )
        self.assertEqual(specs["gee_sources"]["terrain_dem"], "USGS/SRTMGL1_003")
        task = specs["tasks"][0]
        self.assertEqual(task["area_id"], "xiangzhen_record_000000")
        self.assertEqual(task["years"], {"start": 2020, "end": 2021})
        self.assertEqual(task["bbox"], [113.42, 32.37, 113.59, 32.52])
        raster_names = [layer["name"] for layer in task["gee_rasters"]]
        self.assertEqual(
            raster_names,
            [
                "lulc_start",
                "lulc_end",
                "alphaearth_start",
                "alphaearth_end",
                "dem",
                "slope",
                "aspect",
            ],
        )
        derived_names = [layer["name"] for layer in task["derived_layers"]]
        self.assertIn("restriction_mask_from_admin_polygon", derived_names)
        self.assertIn("probability_of_occurrence", derived_names)
        self.assertIn("future_demand", derived_names)
        driver_names = [layer["name"] for layer in task["optional_driver_assets"]]
        self.assertEqual(
            driver_names,
            [
                "distance_to_highway",
                "distance_to_railway",
                "distance_to_road",
                "distance_to_town",
                "distance_to_water",
            ],
        )
        self.assertTrue(all(layer["status"] == "requires_gee_asset" for layer in task["optional_driver_assets"]))

    def test_export_spec_cli_runs_when_executed_by_file_path(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "paper58_benchmark" / "make_flus_native_gee_export_specs.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "candidates.json"
            output_path = root / "export_specs.json"
            _write_candidate_manifest(manifest_path)

            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--candidate-manifest",
                    str(manifest_path),
                    "--output",
                    str(output_path),
                    "--limit",
                    "1",
                    "--start-year",
                    "2020",
                    "--end-year",
                    "2021",
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FLUS-native GEE export specs: 1 task(s)", result.stdout)


if __name__ == "__main__":
    unittest.main()
