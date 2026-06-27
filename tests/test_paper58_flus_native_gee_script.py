from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _write_export_specs(path: Path) -> None:
    payload = {
        "version": 1,
        "gee_sources": {
            "alphaearth": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
            "lulc": "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS",
            "terrain_dem": "USGS/SRTMGL1_003",
        },
        "summary": {"n_tasks": 1, "start_year": 2020, "end_year": 2021},
        "tasks": [
            {
                "area_id": "xiangzhen_record_000000",
                "bbox": [113.42, 32.37, 113.59, 32.52],
                "target_scale_m": 100,
                "gee_rasters": [
                    {
                        "name": "lulc_start",
                        "collection": "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS",
                        "year": 2020,
                        "bands": ["b1"],
                        "scale_m": 100,
                        "output": "data/flus_native_admin/xiangzhen_record_000000/lulc_2020.tif",
                    },
                    {
                        "name": "alphaearth_start",
                        "collection": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
                        "year": 2020,
                        "bands": ["A00", "A01"],
                        "scale_m": 100,
                        "output": "data/flus_native_admin/xiangzhen_record_000000/alphaearth_2020.tif",
                    },
                    {
                        "name": "dem",
                        "collection": "USGS/SRTMGL1_003",
                        "operation": "select:elevation",
                        "bands": ["elevation"],
                        "scale_m": 100,
                        "output": "data/flus_native_admin/xiangzhen_record_000000/dem.tif",
                    },
                    {
                        "name": "slope",
                        "collection": "USGS/SRTMGL1_003",
                        "operation": "ee.Terrain.slope",
                        "bands": ["slope"],
                        "scale_m": 100,
                        "output": "data/flus_native_admin/xiangzhen_record_000000/slope.tif",
                    },
                    {
                        "name": "aspect",
                        "collection": "USGS/SRTMGL1_003",
                        "operation": "ee.Terrain.aspect",
                        "bands": ["aspect"],
                        "scale_m": 100,
                        "output": "data/flus_native_admin/xiangzhen_record_000000/aspect.tif",
                    },
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class FlusNativeGeeScriptTests(unittest.TestCase):
    def test_build_gee_javascript_contains_core_exports(self) -> None:
        from scripts.paper58_benchmark.flus_native_gee_script import build_gee_javascript

        with tempfile.TemporaryDirectory() as tmp:
            specs_path = Path(tmp) / "export_specs.json"
            _write_export_specs(specs_path)
            js = build_gee_javascript(specs_path, drive_folder="paper58_flus_native_preview")

        self.assertIn("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL", js)
        self.assertIn("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS", js)
        self.assertIn("USGS/SRTMGL1_003", js)
        self.assertIn("ee.Geometry.Rectangle([113.42, 32.37, 113.59, 32.52])", js)
        self.assertIn("ee.Terrain.slope", js)
        self.assertIn("ee.Terrain.aspect", js)
        self.assertIn("Export.image.toDrive", js)
        self.assertIn("description: 'xiangzhen_record_000000_lulc_start'", js)
        self.assertIn("folder: 'paper58_flus_native_preview'", js)
        self.assertIn("scale: 100", js)

    def test_gee_javascript_cli_writes_script_when_executed_by_file_path(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "paper58_benchmark" / "write_flus_native_gee_js.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs_path = root / "export_specs.json"
            output_path = root / "export.js"
            _write_export_specs(specs_path)

            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--export-specs",
                    str(specs_path),
                    "--output",
                    str(output_path),
                    "--drive-folder",
                    "paper58_flus_native_preview",
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )

            script_text = output_path.read_text(encoding="utf-8") if output_path.exists() else ""

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FLUS-native GEE JavaScript", result.stdout)
        self.assertIn("Export.image.toDrive", script_text)


if __name__ == "__main__":
    unittest.main()
