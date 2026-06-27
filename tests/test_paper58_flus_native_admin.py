from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


def _write_test_dbf(path: Path) -> None:
    fields = [
        ("省", 18),
        ("市", 18),
        ("县", 18),
        ("乡", 24),
        ("市_县", 32),
        ("省_县", 32),
        ("geom", 48),
    ]
    records = [
        {
            "省": "河南省",
            "市": "南阳市",
            "县": "桐柏县",
            "乡": "吴城镇",
            "市_县": "南阳市桐柏县",
            "省_县": "河南省桐柏县",
            "geom": "SRID=4326;MULTIPOLYGON",
        },
        {
            "省": "河北省",
            "市": "保定市",
            "县": "安新县",
            "乡": "端村镇",
            "市_县": "保定市安新县",
            "省_县": "河北省安新县",
            "geom": "SRID=4326;MULTIPOLYGON",
        },
    ]
    header_len = 32 + 32 * len(fields) + 1
    record_len = 1 + sum(length for _, length in fields)
    header = bytearray(32)
    header[0] = 3
    header[1:4] = bytes([126, 6, 26])
    header[4:8] = struct.pack("<I", len(records))
    header[8:10] = struct.pack("<H", header_len)
    header[10:12] = struct.pack("<H", record_len)
    field_bytes = bytearray()
    for name, length in fields:
        descriptor = bytearray(32)
        encoded = name.encode("utf-8")
        descriptor[: len(encoded)] = encoded
        descriptor[11] = ord("C")
        descriptor[16] = length
        field_bytes.extend(descriptor)
    body = bytearray()
    for record in records:
        row = bytearray(b" ")
        for name, length in fields:
            raw = record[name].encode("utf-8")
            row.extend(raw[:length].ljust(length, b" "))
        body.extend(row)
    path.write_bytes(bytes(header) + bytes(field_bytes) + b"\r" + bytes(body))


def _write_test_shp(path: Path) -> None:
    record_bboxes = [
        (113.0, 32.0, 114.0, 33.0),
        (115.0, 38.0, 116.0, 39.0),
    ]
    records = bytearray()
    for index, bbox in enumerate(record_bboxes, start=1):
        content = struct.pack("<i4d", 5, *bbox)
        records.extend(struct.pack(">2i", index, len(content) // 2))
        records.extend(content)
    file_len_words = (100 + len(records)) // 2
    header = bytearray(100)
    header[:4] = struct.pack(">i", 9994)
    header[24:28] = struct.pack(">i", file_len_words)
    header[28:36] = struct.pack("<2i", 1000, 5)
    header[36:68] = struct.pack("<4d", 113.0, 32.0, 116.0, 39.0)
    path.write_bytes(bytes(header) + bytes(records))


def _write_custom_test_dbf(path: Path, records: list[dict[str, str]]) -> None:
    fields = [
        ("省", 18),
        ("市", 18),
        ("县", 18),
        ("乡", 24),
        ("市_县", 32),
        ("省_县", 32),
        ("geom", 48),
    ]
    header_len = 32 + 32 * len(fields) + 1
    record_len = 1 + sum(length for _, length in fields)
    header = bytearray(32)
    header[0] = 3
    header[1:4] = bytes([126, 6, 26])
    header[4:8] = struct.pack("<I", len(records))
    header[8:10] = struct.pack("<H", header_len)
    header[10:12] = struct.pack("<H", record_len)
    field_bytes = bytearray()
    for name, length in fields:
        descriptor = bytearray(32)
        encoded = name.encode("utf-8")
        descriptor[: len(encoded)] = encoded
        descriptor[11] = ord("C")
        descriptor[16] = length
        field_bytes.extend(descriptor)
    body = bytearray()
    for record in records:
        row = bytearray(b" ")
        for name, length in fields:
            raw = record.get(name, "").encode("utf-8")
            row.extend(raw[:length].ljust(length, b" "))
        body.extend(row)
    path.write_bytes(bytes(header) + bytes(field_bytes) + b"\r" + bytes(body))


def _write_custom_test_shp(path: Path, record_bboxes: list[tuple[float, float, float, float]]) -> None:
    records = bytearray()
    for index, bbox in enumerate(record_bboxes, start=1):
        content = struct.pack("<i4d", 5, *bbox)
        records.extend(struct.pack(">2i", index, len(content) // 2))
        records.extend(content)
    file_len_words = (100 + len(records)) // 2
    xs = [value for bbox in record_bboxes for value in (bbox[0], bbox[2])]
    ys = [value for bbox in record_bboxes for value in (bbox[1], bbox[3])]
    header = bytearray(100)
    header[:4] = struct.pack(">i", 9994)
    header[24:28] = struct.pack(">i", file_len_words)
    header[28:36] = struct.pack("<2i", 1000, 5)
    header[36:68] = struct.pack("<4d", min(xs), min(ys), max(xs), max(ys))
    path.write_bytes(bytes(header) + bytes(records))


def _minimal_tiff_bytes(width: int, height: int, samples_per_pixel: int) -> bytes:
    entries = []
    extra = bytearray()

    def add_entry(tag: int, tag_type: int, count: int, value: int | bytes) -> None:
        if isinstance(value, bytes):
            offset = 8 + 2 + 12 * 4 + 4 + len(extra)
            extra.extend(value)
            entries.append(struct.pack("<HHII", tag, tag_type, count, offset))
        else:
            entries.append(struct.pack("<HHII", tag, tag_type, count, value))

    add_entry(256, 4, 1, width)
    add_entry(257, 4, 1, height)
    add_entry(277, 3, 1, samples_per_pixel)
    add_entry(258, 3, samples_per_pixel, struct.pack("<" + "H" * samples_per_pixel, *([32] * samples_per_pixel)))
    ifd = struct.pack("<H", len(entries)) + b"".join(entries) + struct.pack("<I", 0)
    return b"II*\x00" + struct.pack("<I", 8) + ifd + bytes(extra)


class FlusNativeAdminParserTests(unittest.TestCase):
    def test_dbf_records_decode_chinese_admin_fields(self) -> None:
        from scripts.paper58_benchmark.flus_native_admin import iter_dbf_records, read_dbf_metadata

        with tempfile.TemporaryDirectory() as tmp:
            dbf_path = Path(tmp) / "xiangzhen.dbf"
            _write_test_dbf(dbf_path)

            metadata = read_dbf_metadata(dbf_path)
            records = list(iter_dbf_records(dbf_path))

        self.assertEqual(metadata["records"], 2)
        self.assertEqual(metadata["fields"][0]["normalized_name"], "province")
        self.assertEqual(records[0]["province"], "河南省")
        self.assertEqual(records[0]["city"], "南阳市")
        self.assertEqual(records[0]["county"], "桐柏县")
        self.assertEqual(records[0]["town"], "吴城镇")
        self.assertEqual(records[1]["province_county"], "河北省安新县")

    def test_shp_metadata_and_record_bboxes_are_read_without_gdal(self) -> None:
        from scripts.paper58_benchmark.flus_native_admin import iter_shp_record_bboxes, read_shp_metadata

        with tempfile.TemporaryDirectory() as tmp:
            shp_path = Path(tmp) / "xiangzhen.shp"
            _write_test_shp(shp_path)

            metadata = read_shp_metadata(shp_path)
            bboxes = list(iter_shp_record_bboxes(shp_path))

        self.assertEqual(metadata["shape_type"], 5)
        self.assertEqual(metadata["bbox_xy"], [113.0, 32.0, 116.0, 39.0])
        self.assertEqual(bboxes[0]["record_index"], 0)
        self.assertEqual(bboxes[0]["bbox_xy"], [113.0, 32.0, 114.0, 33.0])
        self.assertEqual(bboxes[1]["record_number"], 2)
        self.assertEqual(bboxes[1]["bbox_xy"], [115.0, 38.0, 116.0, 39.0])

    def test_flus_sample_zip_inspector_reads_configs_and_tiff_headers(self) -> None:
        from scripts.paper58_benchmark.flus_native_admin import inspect_flus_sample_zip

        config_color = "\n".join(
            [
                "[Index, Count, Land Use Type, R, G, B]",
                "1,46989,Urban land,170,0,127",
                "2,54427,Water area,0,0,255",
                "3,59899,Cropland,0,255,0",
                "4,49516,Forest land,0,85,0",
                "5,38090,Orchard,255,255,0",
            ]
        )
        config_mp = "\n".join(
            [
                "[Number of types]",
                "5",
                "[Future Pixels]",
                "80016",
                "54427",
                "43599",
                "42433",
                "28446",
                "[Cost Matrix]",
                "1,0,0,0,0",
                "0,1,0,0,0",
                "1,1,1,1,0",
                "1,0,1,1,0",
                "1,0,1,0,1",
                "[Intensity of neighborhood]",
                "1",
                "0.9",
                "0.5",
                "1",
                "0.1",
                "[Maximum Number Of Iterations]",
                "300",
                "[Size of neighborhood]",
                "3",
                "[Accelerated factor]",
                "0.1",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "flus_sample.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("FLUS_V2.4/testdata/config_color.log", config_color)
                zf.writestr("FLUS_V2.4/testdata/config_mp.log", config_mp)
                zf.writestr(
                    "FLUS_V2.4/testdata/Probability-of-occurrence.tif",
                    _minimal_tiff_bytes(width=768, height=531, samples_per_pixel=5),
                )

            summary = inspect_flus_sample_zip(zip_path)

        self.assertEqual(summary["number_of_types"], 5)
        self.assertEqual(summary["classes"][0]["name"], "Urban land")
        self.assertEqual(summary["future_pixels"], [80016, 54427, 43599, 42433, 28446])
        self.assertEqual(summary["cost_matrix"][2], [1, 1, 1, 1, 0])
        self.assertEqual(summary["neighborhood_weights"], [1.0, 0.9, 0.5, 1.0, 0.1])
        self.assertEqual(summary["maximum_iterations"], 300)
        self.assertEqual(summary["neighborhood_size"], 3)
        self.assertEqual(summary["accelerated_factor"], 0.1)
        self.assertEqual(summary["probability_raster"]["width"], 768)
        self.assertEqual(summary["probability_raster"]["height"], 531)
        self.assertEqual(summary["probability_raster"]["samples_per_pixel"], 5)

    def test_admin_candidate_manifest_filters_by_province_and_estimates_grid(self) -> None:
        from scripts.paper58_benchmark.flus_native_admin import build_admin_candidate_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shp_path = root / "xiangzhen.shp"
            dbf_path = root / "xiangzhen.dbf"
            _write_test_shp(shp_path)
            _write_test_dbf(dbf_path)

            manifest = build_admin_candidate_manifest(
                shp_path=shp_path,
                limit=10,
                province="河南省",
                target_scale_m=100,
            )

        self.assertEqual(manifest["summary"]["n_rows"], 1)
        self.assertEqual(manifest["summary"]["source_records"], 2)
        self.assertEqual(manifest["filters"]["province"], "河南省")
        row = manifest["rows"][0]
        self.assertEqual(row["area_id"], "xiangzhen_record_000000")
        self.assertEqual(row["record_index"], 0)
        self.assertEqual(row["province"], "河南省")
        self.assertEqual(row["city"], "南阳市")
        self.assertEqual(row["county"], "桐柏县")
        self.assertEqual(row["town"], "吴城镇")
        self.assertEqual(row["bbox"], [113.0, 32.0, 114.0, 33.0])
        self.assertEqual(row["target_scale_m"], 100)
        self.assertGreater(row["estimated_width_px"], 0)
        self.assertGreater(row["estimated_height_px"], 0)
        self.assertIn("flus_native_candidate", row["tags"])

    def test_stratified_manifest_samples_across_regions_and_records_selection_metadata(self) -> None:
        from scripts.paper58_benchmark.flus_native_admin import build_stratified_admin_candidate_manifest

        records = [
            {"省": "广东省", "市": "广州市", "县": "番禺区", "乡": "A镇", "市_县": "广州市番禺区", "省_县": "广东省番禺区"},
            {"省": "广东省", "市": "佛山市", "县": "顺德区", "乡": "B镇", "市_县": "佛山市顺德区", "省_县": "广东省顺德区"},
            {"省": "河北省", "市": "保定市", "县": "安新县", "乡": "C镇", "市_县": "保定市安新县", "省_县": "河北省安新县"},
            {"省": "黑龙江省", "市": "哈尔滨市", "县": "宾县", "乡": "D镇", "市_县": "哈尔滨市宾县", "省_县": "黑龙江省宾县"},
            {"省": "四川省", "市": "成都市", "县": "郫都区", "乡": "E镇", "市_县": "成都市郫都区", "省_县": "四川省郫都区"},
            {"省": "新疆维吾尔自治区", "市": "喀什地区", "县": "疏附县", "乡": "F乡", "市_县": "喀什地区疏附县", "省_县": "新疆维吾尔自治区疏附县"},
        ]
        bboxes = [
            (113.2, 22.8, 113.5, 23.1),
            (113.0, 22.6, 113.35, 22.95),
            (115.8, 38.8, 116.1, 39.1),
            (126.1, 45.5, 126.5, 45.9),
            (103.8, 30.6, 104.1, 30.9),
            (75.8, 39.1, 76.2, 39.5),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shp_path = root / "xiangzhen.shp"
            dbf_path = root / "xiangzhen.dbf"
            _write_custom_test_shp(shp_path, bboxes)
            _write_custom_test_dbf(dbf_path, records)

            manifest = build_stratified_admin_candidate_manifest(
                shp_path=shp_path,
                sample_size=4,
                target_scale_m=100,
                min_pixels=1000,
                max_pixels=200000,
                max_per_province=1,
                seed="unit-test",
            )

        rows = manifest["rows"]
        self.assertEqual(manifest["summary"]["sample_method"], "deterministic_macro_region_size_round_robin")
        self.assertEqual(manifest["summary"]["n_rows"], 4)
        self.assertEqual(len({row["province"] for row in rows}), 4)
        self.assertTrue(all("stratified_sample" in row["tags"] for row in rows))
        self.assertTrue(all(row["selection"]["estimated_pixels"] >= 1000 for row in rows))
        self.assertTrue(all(row["selection"]["macro_region"] for row in rows))
        self.assertTrue(all(row["selection"]["size_class"] in {"small", "medium", "large"} for row in rows))

    def test_manifest_cli_runs_when_executed_by_file_path(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "paper58_benchmark" / "make_flus_native_admin_manifest.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shp_path = root / "xiangzhen.shp"
            dbf_path = root / "xiangzhen.dbf"
            output_path = root / "manifest.json"
            _write_test_shp(shp_path)
            _write_test_dbf(dbf_path)

            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--shp",
                    str(shp_path),
                    "--output",
                    str(output_path),
                    "--limit",
                    "1",
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FLUS-native admin manifest: 1/2 row(s)", result.stdout)

    def test_manifest_cli_can_write_stratified_sample(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "paper58_benchmark" / "make_flus_native_admin_manifest.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shp_path = root / "xiangzhen.shp"
            dbf_path = root / "xiangzhen.dbf"
            output_path = root / "manifest.json"
            _write_custom_test_shp(
                shp_path,
                [
                    (113.0, 22.5, 113.4, 22.9),
                    (115.8, 38.8, 116.2, 39.2),
                    (126.1, 45.5, 126.6, 46.0),
                ],
            )
            _write_custom_test_dbf(
                dbf_path,
                [
                    {"省": "广东省", "市": "广州市", "县": "番禺区", "乡": "A镇", "市_县": "广州市番禺区", "省_县": "广东省番禺区"},
                    {"省": "河北省", "市": "保定市", "县": "安新县", "乡": "B镇", "市_县": "保定市安新县", "省_县": "河北省安新县"},
                    {"省": "黑龙江省", "市": "哈尔滨市", "县": "宾县", "乡": "C镇", "市_县": "哈尔滨市宾县", "省_县": "黑龙江省宾县"},
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--shp",
                    str(shp_path),
                    "--output",
                    str(output_path),
                    "--stratified-sample-size",
                    "2",
                    "--min-pixels",
                    "1000",
                    "--max-pixels",
                    "200000",
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["summary"]["sample_method"], "deterministic_macro_region_size_round_robin")
        self.assertEqual(payload["summary"]["n_rows"], 2)


if __name__ == "__main__":
    unittest.main()
