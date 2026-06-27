from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


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


def test_select_manifest_rows_supports_limit_and_area_id(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.fetch_flus_native_admin_same_grid_paper58_inputs import select_manifest_rows

    manifest = {
        "rows": [
            {"area_id": "a", "record_index": 0},
            {"area_id": "b", "record_index": 1},
            {"area_id": "c", "record_index": 2},
        ]
    }

    assert [row["area_id"] for row in select_manifest_rows(manifest, limit=2)] == ["a", "b"]
    assert [row["area_id"] for row in select_manifest_rows(manifest, area_ids=["c", "a"])] == ["a", "c"]


def test_read_admin_geometry_uses_manifest_record_index(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.fetch_flus_native_admin_same_grid_paper58_inputs import read_admin_geometry

    shp_path = tmp_path / "xiangzhen.shp"
    dbf_path = tmp_path / "xiangzhen.dbf"
    _write_custom_test_shp(shp_path, [(113.0, 22.5, 113.4, 22.9), (115.8, 38.8, 116.2, 39.2)])
    _write_custom_test_dbf(
        dbf_path,
        [
            {"省": "广东省", "市": "广州市", "县": "番禺区", "乡": "A镇", "市_县": "广州市番禺区", "省_县": "广东省番禺区"},
            {"省": "河北省", "市": "保定市", "县": "安新县", "乡": "B镇", "市_县": "保定市安新县", "省_县": "河北省安新县"},
        ],
    )
    row = {
        "area_id": "xiangzhen_record_000001",
        "source_shp_path": str(shp_path),
        "record_index": 1,
    }

    result = read_admin_geometry(row, simplify_tolerance=0)

    assert result["properties"]["省"] == "河北省"
    assert result["bounds"] == [115.8, 38.8, 116.2, 39.2]
    assert json.loads(json.dumps(result["geometry_geojson"]))["type"] == "Polygon"


def test_lulc_reference_grid_treats_non_positive_values_as_invalid(tmp_path: Path) -> None:
    from scripts.paper58_benchmark.fetch_flus_native_admin_same_grid_paper58_inputs import read_lulc_reference_grid

    tif_path = tmp_path / "lulc.tif"
    arr = np.array([[1, 2, 0], [-32768, 7, 8]], dtype=np.int16)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=1,
        dtype=arr.dtype,
        crs="EPSG:4326",
        transform=from_origin(100, 30, 0.01, 0.01),
        nodata=-32768,
    ) as dataset:
        dataset.write(arr, 1)

    reference = read_lulc_reference_grid(tif_path)

    assert reference.valid_mask.tolist() == [[True, True, False], [False, True, True]]
