from __future__ import annotations

import csv
import io
import argparse
import hashlib
import json
import math
import struct
import zipfile
from pathlib import Path
from typing import Iterator


ADMIN_FIELD_ALIASES = {
    "省": "province",
    "市": "city",
    "县": "county",
    "乡": "town",
    "市_县": "city_county",
    "省_县": "province_county",
    "geom": "geom",
}

PROVINCE_MACRO_REGIONS = {
    "北京市": "华北",
    "天津市": "华北",
    "河北省": "华北",
    "山西省": "华北",
    "内蒙古自治区": "华北",
    "辽宁省": "东北",
    "吉林省": "东北",
    "黑龙江省": "东北",
    "上海市": "华东",
    "江苏省": "华东",
    "浙江省": "华东",
    "安徽省": "华东",
    "福建省": "华东",
    "江西省": "华东",
    "山东省": "华东",
    "台湾省": "华东",
    "河南省": "华中",
    "湖北省": "华中",
    "湖南省": "华中",
    "广东省": "华南",
    "广西壮族自治区": "华南",
    "海南省": "华南",
    "香港特别行政区": "华南",
    "澳门特别行政区": "华南",
    "重庆市": "西南",
    "四川省": "西南",
    "贵州省": "西南",
    "云南省": "西南",
    "西藏自治区": "西南",
    "陕西省": "西北",
    "甘肃省": "西北",
    "青海省": "西北",
    "宁夏回族自治区": "西北",
    "新疆维吾尔自治区": "西北",
}

STRATIFIED_SAMPLE_METHOD = "deterministic_macro_region_size_round_robin"


def _decode(raw: bytes, encoding: str = "utf-8") -> str:
    return raw.decode(encoding, errors="replace").strip()


def _normalize_field_name(name: str) -> str:
    if name in ADMIN_FIELD_ALIASES:
        return ADMIN_FIELD_ALIASES[name]
    normalized = name.strip().lower().replace(" ", "_")
    return normalized or "field"


def _read_dbf_header(path: Path, encoding: str = "utf-8") -> dict:
    source = Path(path)
    with source.open("rb") as f:
        first = f.read(32)
        if len(first) < 32:
            raise ValueError(f"DBF header is too short: {source}")
        record_count = struct.unpack("<I", first[4:8])[0]
        header_length = struct.unpack("<H", first[8:10])[0]
        record_length = struct.unpack("<H", first[10:12])[0]
        f.seek(0)
        header = f.read(header_length)

    fields = []
    offset = 32
    record_offset = 1
    while offset + 32 <= len(header) and header[offset] != 0x0D:
        descriptor = header[offset : offset + 32]
        raw_name = descriptor[:11].split(b"\0", 1)[0]
        name = _decode(raw_name, encoding=encoding)
        field_length = int(descriptor[16])
        fields.append(
            {
                "name": name,
                "normalized_name": _normalize_field_name(name),
                "type": chr(descriptor[11]),
                "length": field_length,
                "decimal_count": int(descriptor[17]),
                "record_offset": record_offset,
            }
        )
        record_offset += field_length
        offset += 32

    return {
        "path": str(source),
        "records": int(record_count),
        "header_length": int(header_length),
        "record_length": int(record_length),
        "fields": fields,
    }


def read_dbf_metadata(path: Path, encoding: str = "utf-8") -> dict:
    return _read_dbf_header(Path(path), encoding=encoding)


def iter_dbf_records(path: Path, encoding: str = "utf-8") -> Iterator[dict[str, str]]:
    metadata = _read_dbf_header(Path(path), encoding=encoding)
    fields = metadata["fields"]
    with Path(path).open("rb") as f:
        f.seek(int(metadata["header_length"]))
        for _ in range(int(metadata["records"])):
            record = f.read(int(metadata["record_length"]))
            if len(record) < int(metadata["record_length"]):
                break
            if record[:1] == b"*":
                continue
            row: dict[str, str] = {}
            for field in fields:
                start = int(field["record_offset"])
                end = start + int(field["length"])
                row[str(field["normalized_name"])] = _decode(record[start:end], encoding=encoding)
            yield row


def read_shp_metadata(path: Path) -> dict:
    source = Path(path)
    header = source.read_bytes()[:100]
    if len(header) < 100:
        raise ValueError(f"SHP header is too short: {source}")
    file_code = struct.unpack(">i", header[:4])[0]
    file_length_bytes = struct.unpack(">i", header[24:28])[0] * 2
    version, shape_type = struct.unpack("<2i", header[28:36])
    bbox = struct.unpack("<4d", header[36:68])
    return {
        "path": str(source),
        "file_code": int(file_code),
        "file_length_bytes": int(file_length_bytes),
        "version": int(version),
        "shape_type": int(shape_type),
        "bbox_xy": [float(value) for value in bbox],
    }


def iter_shp_record_bboxes(path: Path) -> Iterator[dict[str, int | list[float]]]:
    with Path(path).open("rb") as f:
        f.seek(100)
        sequential_index = 0
        while True:
            record_header = f.read(8)
            if not record_header:
                break
            if len(record_header) < 8:
                raise ValueError(f"Truncated SHP record header in {path}")
            record_number, content_length_words = struct.unpack(">2i", record_header)
            content_length = int(content_length_words) * 2
            content = f.read(content_length)
            if len(content) < content_length:
                raise ValueError(f"Truncated SHP record content in {path}")
            if content_length < 36:
                sequential_index += 1
                continue
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type == 0:
                sequential_index += 1
                continue
            bbox = struct.unpack("<4d", content[4:36])
            yield {
                "record_index": sequential_index,
                "record_number": int(record_number),
                "shape_type": int(shape_type),
                "bbox_xy": [float(value) for value in bbox],
            }
            sequential_index += 1


def _config_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line.strip("[]")
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return sections


def _parse_config_mp(text: str) -> dict:
    sections = _config_sections(text)
    number_of_types = int(sections["Number of types"][0])
    future_pixels = [int(value) for value in sections.get("Future Pixels", [])[:number_of_types]]
    cost_matrix = [
        [int(value.strip()) for value in row.split(",")]
        for row in sections.get("Cost Matrix", [])[:number_of_types]
    ]
    neighborhood_weights = [
        float(value) for value in sections.get("Intensity of neighborhood", [])[:number_of_types]
    ]
    maximum_iterations = int(sections["Maximum Number Of Iterations"][0])
    neighborhood_size = int(sections["Size of neighborhood"][0])
    accelerated_factor = float(sections["Accelerated factor"][0])
    return {
        "number_of_types": number_of_types,
        "future_pixels": future_pixels,
        "cost_matrix": cost_matrix,
        "neighborhood_weights": neighborhood_weights,
        "maximum_iterations": maximum_iterations,
        "neighborhood_size": neighborhood_size,
        "accelerated_factor": accelerated_factor,
    }


def _parse_config_color(text: str) -> list[dict]:
    classes = []
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0].strip().startswith("["):
            continue
        if len(row) < 6:
            continue
        classes.append(
            {
                "index": int(row[0]),
                "count": int(row[1]),
                "name": row[2].strip(),
                "rgb": [int(row[3]), int(row[4]), int(row[5])],
            }
        )
    return classes


_TIFF_TYPE_FORMATS = {
    1: ("B", 1),
    2: ("c", 1),
    3: ("H", 2),
    4: ("I", 4),
    11: ("f", 4),
    12: ("d", 8),
}


def _tiff_values(data: bytes, endian: str, tag_type: int, count: int, value_or_offset: bytes) -> object:
    fmt, size = _TIFF_TYPE_FORMATS.get(tag_type, ("B", 1))
    total_size = size * count
    raw = value_or_offset[:total_size] if total_size <= 4 else data[struct.unpack(endian + "I", value_or_offset)[0] :][:total_size]
    if tag_type == 2:
        return raw.rstrip(b"\0").decode("utf-8", errors="replace")
    values = struct.unpack(endian + fmt * count, raw)
    return values[0] if count == 1 else list(values)


def _read_tiff_header(data: bytes) -> dict:
    if len(data) < 8:
        raise ValueError("TIFF header is too short")
    if data[:2] == b"II":
        endian = "<"
    elif data[:2] == b"MM":
        endian = ">"
    else:
        raise ValueError("TIFF byte order marker is missing")
    magic, ifd_offset = struct.unpack(endian + "HI", data[2:8])
    if magic != 42:
        raise ValueError(f"unsupported TIFF magic: {magic}")
    entry_count = struct.unpack(endian + "H", data[ifd_offset : ifd_offset + 2])[0]
    tags: dict[int, object] = {}
    for index in range(entry_count):
        offset = ifd_offset + 2 + index * 12
        tag, tag_type, count = struct.unpack(endian + "HHI", data[offset : offset + 8])
        value_or_offset = data[offset + 8 : offset + 12]
        tags[int(tag)] = _tiff_values(data, endian, int(tag_type), int(count), value_or_offset)
    return {
        "width": int(tags.get(256, 0)),
        "height": int(tags.get(257, 0)),
        "bits_per_sample": tags.get(258),
        "samples_per_pixel": int(tags.get(277, 1)),
        "sample_format": tags.get(339),
    }


def _first_matching_name(names: list[str], suffix: str) -> str | None:
    for name in names:
        if name.endswith(suffix):
            return name
    return None


def inspect_flus_sample_zip(zip_path: Path) -> dict:
    source = Path(zip_path)
    with zipfile.ZipFile(source) as zf:
        names = zf.namelist()
        config_mp_name = _first_matching_name(names, "config_mp.log")
        config_color_name = _first_matching_name(names, "config_color.log")
        probability_name = _first_matching_name(names, "Probability-of-occurrence.tif")
        if config_mp_name is None:
            raise FileNotFoundError("config_mp.log not found in FLUS sample zip")
        if config_color_name is None:
            raise FileNotFoundError("config_color.log not found in FLUS sample zip")
        config = _parse_config_mp(zf.read(config_mp_name).decode("utf-8", errors="replace"))
        classes = _parse_config_color(zf.read(config_color_name).decode("utf-8", errors="replace"))
        probability_raster = None
        if probability_name is not None:
            probability_raster = _read_tiff_header(zf.read(probability_name))
        return {
            "zip_path": str(source),
            **config,
            "classes": classes,
            "probability_raster": probability_raster,
        }


def _estimate_grid_shape(bbox: list[float], target_scale_m: int) -> tuple[int, int]:
    xmin, ymin, xmax, ymax = bbox
    mid_lat = (float(ymin) + float(ymax)) / 2.0
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = max(1.0, meters_per_degree_lat * math.cos(math.radians(mid_lat)))
    width_m = abs(float(xmax) - float(xmin)) * meters_per_degree_lon
    height_m = abs(float(ymax) - float(ymin)) * meters_per_degree_lat
    scale = max(1, int(target_scale_m))
    return max(1, math.ceil(width_m / scale)), max(1, math.ceil(height_m / scale))


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    xmin, ymin, xmax, ymax = bbox
    return (float(xmin) + float(xmax)) / 2.0, (float(ymin) + float(ymax)) / 2.0


def _macro_region(province: str) -> str:
    return PROVINCE_MACRO_REGIONS.get(province, "未分区")


def _size_class(estimated_pixels: int) -> str:
    if estimated_pixels < 10_000:
        return "small"
    if estimated_pixels < 40_000:
        return "medium"
    return "large"


def _stable_selection_key(row: dict, seed: str) -> str:
    payload = "|".join(
        [
            str(seed),
            str(row.get("area_id", "")),
            str(row.get("province", "")),
            str(row.get("city", "")),
            str(row.get("county", "")),
            str(row.get("town", "")),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_admin_candidate_manifest(
    shp_path: Path,
    limit: int | None = None,
    province: str | None = None,
    target_scale_m: int = 100,
    encoding: str = "utf-8",
) -> dict:
    source_shp = Path(shp_path)
    source_dbf = source_shp.with_suffix(".dbf")
    if not source_dbf.exists():
        raise FileNotFoundError(f"DBF companion not found for {source_shp}: {source_dbf}")

    dbf_metadata = read_dbf_metadata(source_dbf, encoding=encoding)
    shp_metadata = read_shp_metadata(source_shp)
    rows = []
    for admin_row, bbox_row in zip(
        iter_dbf_records(source_dbf, encoding=encoding),
        iter_shp_record_bboxes(source_shp),
    ):
        if province is not None and admin_row.get("province") != province:
            continue
        bbox = [float(value) for value in bbox_row["bbox_xy"]]  # type: ignore[index]
        estimated_width, estimated_height = _estimate_grid_shape(bbox, target_scale_m)
        record_index = int(bbox_row["record_index"])
        rows.append(
            {
                "area_id": f"{source_shp.stem}_record_{record_index:06d}",
                "source_shp_path": str(source_shp),
                "source_dbf_path": str(source_dbf),
                "record_index": record_index,
                "record_number": int(bbox_row["record_number"]),
                "province": admin_row.get("province", ""),
                "city": admin_row.get("city", ""),
                "county": admin_row.get("county", ""),
                "town": admin_row.get("town", ""),
                "city_county": admin_row.get("city_county", ""),
                "province_county": admin_row.get("province_county", ""),
                "bbox": bbox,
                "target_scale_m": int(target_scale_m),
                "estimated_width_px": int(estimated_width),
                "estimated_height_px": int(estimated_height),
                "tags": ["admin_township", "flus_native_candidate"],
            }
        )
        if limit is not None and len(rows) >= int(limit):
            break

    return {
        "version": 1,
        "purpose": "Paper58 FLUS-native administrative-unit candidate manifest",
        "source": {
            "shp_path": str(source_shp),
            "dbf_path": str(source_dbf),
            "shp_metadata": shp_metadata,
            "dbf_records": int(dbf_metadata["records"]),
        },
        "filters": {
            "province": province,
            "limit": limit,
            "target_scale_m": int(target_scale_m),
        },
        "summary": {
            "source_records": int(dbf_metadata["records"]),
            "n_rows": len(rows),
        },
        "rows": rows,
    }


def build_stratified_admin_candidate_manifest(
    shp_path: Path,
    sample_size: int,
    province: str | None = None,
    target_scale_m: int = 100,
    min_pixels: int = 2_500,
    max_pixels: int = 80_000,
    min_width_px: int = 32,
    min_height_px: int = 32,
    max_per_province: int = 2,
    seed: str = "paper58-flus-native-admin-v1",
    encoding: str = "utf-8",
) -> dict:
    base = build_admin_candidate_manifest(
        shp_path=shp_path,
        limit=None,
        province=province,
        target_scale_m=target_scale_m,
        encoding=encoding,
    )
    source_rows = list(base["rows"])
    eligible_rows = []
    for row in source_rows:
        width = int(row["estimated_width_px"])
        height = int(row["estimated_height_px"])
        estimated_pixels = width * height
        if estimated_pixels < int(min_pixels) or estimated_pixels > int(max_pixels):
            continue
        if width < int(min_width_px) or height < int(min_height_px):
            continue
        center_lon, center_lat = _bbox_center([float(value) for value in row["bbox"]])
        macro_region = _macro_region(str(row.get("province", "")))
        size_class = _size_class(estimated_pixels)
        selection_key = _stable_selection_key(row, seed)
        enriched = {
            **row,
            "tags": [
                *list(row.get("tags", [])),
                "stratified_sample",
                f"macro_region:{macro_region}",
                f"size_class:{size_class}",
            ],
            "selection": {
                "method": STRATIFIED_SAMPLE_METHOD,
                "seed": seed,
                "selection_key": selection_key,
                "macro_region": macro_region,
                "size_class": size_class,
                "estimated_pixels": int(estimated_pixels),
                "center_lon": float(center_lon),
                "center_lat": float(center_lat),
            },
        }
        eligible_rows.append(enriched)

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in eligible_rows:
        selection = row["selection"]
        group_key = (str(selection["macro_region"]), str(selection["size_class"]))
        groups.setdefault(group_key, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda item: str(item["selection"]["selection_key"]))

    selected_rows = []
    province_counts: dict[str, int] = {}
    ordered_group_keys = sorted(groups)
    while len(selected_rows) < int(sample_size):
        made_progress = False
        for group_key in ordered_group_keys:
            rows = groups[group_key]
            while rows:
                row = rows.pop(0)
                row_province = str(row.get("province", ""))
                if province_counts.get(row_province, 0) >= int(max_per_province):
                    continue
                selected_rows.append(row)
                province_counts[row_province] = province_counts.get(row_province, 0) + 1
                made_progress = True
                break
            if len(selected_rows) >= int(sample_size):
                break
        if not made_progress:
            break

    for rank, row in enumerate(selected_rows, start=1):
        row["selection"]["selection_rank"] = rank

    base["purpose"] = "Paper58 FLUS-native stratified administrative-unit validation sample manifest"
    base["filters"] = {
        **dict(base["filters"]),
        "stratified_sample_size": int(sample_size),
        "sample_method": STRATIFIED_SAMPLE_METHOD,
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
        "min_width_px": int(min_width_px),
        "min_height_px": int(min_height_px),
        "max_per_province": int(max_per_province),
        "seed": seed,
    }
    base["summary"] = {
        **dict(base["summary"]),
        "sample_method": STRATIFIED_SAMPLE_METHOD,
        "candidate_rows_before_pixel_filter": len(source_rows),
        "eligible_rows_after_pixel_filter": len(eligible_rows),
        "n_rows": len(selected_rows),
        "requested_sample_size": int(sample_size),
        "macro_regions": sorted({str(row["selection"]["macro_region"]) for row in selected_rows}),
        "provinces": sorted({str(row.get("province", "")) for row in selected_rows}),
    }
    base["rows"] = selected_rows
    return base


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect FLUS-native administrative benchmark inputs.")
    parser.add_argument("--inspect-flus-sample", type=Path, help="Path to a GeoSOS-FLUS sample ZIP.")
    args = parser.parse_args()
    if args.inspect_flus_sample:
        _print_json(inspect_flus_sample_zip(args.inspect_flus_sample))
        return
    parser.error("provide --inspect-flus-sample")


if __name__ == "__main__":
    main()
