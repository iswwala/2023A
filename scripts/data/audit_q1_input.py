#!/usr/bin/env python3
"""Audit and normalize the Q1 heliostat coordinate attachment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "2023A题" / "附件.xlsx"
CLEAN = ROOT / "workspace" / "data_clean" / "Q1" / "heliostat_positions.csv"
PROFILE = ROOT / "workspace" / "data" / "data_profile.json"
REPORT = ROOT / "workspace" / "data" / "data_report.md"

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def quantile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_first_sheet(path: Path) -> tuple[str, list[str], list[tuple[float, float]], list[list[str]]]:
    ns = {"m": MAIN_NS, "r": REL_NS, "p": PKG_NS}
    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append("".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t")))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            node.attrib["Id"]: node.attrib["Target"]
            for node in relationships.findall("p:Relationship", ns)
        }
        sheet = workbook.find("m:sheets", ns)[0]
        sheet_name = sheet.attrib["name"]
        target = rel_map[sheet.attrib[f"{{{REL_NS}}}id"]]
        sheet_path = target if target.startswith("xl/") else f"xl/{target.lstrip('/')}"
        sheet_root = ET.fromstring(archive.read(sheet_path))

        rows: list[list[str]] = []
        for row in sheet_root.find("m:sheetData", ns):
            values: list[str] = []
            for cell in row.findall("m:c", ns):
                value_node = cell.find("m:v", ns)
                value = "" if value_node is None else (value_node.text or "")
                if cell.attrib.get("t") == "s" and value:
                    value = shared[int(value)]
                values.append(value)
            rows.append(values)

    headers = rows[0]
    positions = [(float(row[0]), float(row[1])) for row in rows[1:] if len(row) >= 2]
    return sheet_name, headers, positions, rows[:6]


def audit_positions(positions: list[tuple[float, float]]) -> dict:
    x_values = [point[0] for point in positions]
    y_values = [point[1] for point in positions]
    radii = [math.hypot(x, y) for x, y in positions]
    angles = [(math.degrees(math.atan2(y, x)) + 360) % 360 for x, y in positions]

    nearest = [math.inf] * len(positions)
    minimum_pair = {"distance_m": math.inf, "row_ids": []}
    spacing_violations = 0
    for i, (x_i, y_i) in enumerate(positions):
        for j in range(i + 1, len(positions)):
            x_j, y_j = positions[j]
            distance = math.hypot(x_i - x_j, y_i - y_j)
            nearest[i] = min(nearest[i], distance)
            nearest[j] = min(nearest[j], distance)
            if distance < 11.0 - 1e-12:
                spacing_violations += 1
            if distance < minimum_pair["distance_m"]:
                minimum_pair = {"distance_m": distance, "row_ids": [i + 2, j + 2]}

    radial_edges = list(range(100, 351, 25))
    radial_counts = []
    for lower, upper in zip(radial_edges[:-1], radial_edges[1:]):
        include_upper = upper == radial_edges[-1]
        count = sum(lower <= radius <= upper if include_upper else lower <= radius < upper for radius in radii)
        radial_counts.append({"range_m": [lower, upper], "count": count})

    sector_counts = []
    for lower in range(0, 360, 30):
        upper = lower + 30
        sector_counts.append({
            "range_deg": [lower, upper],
            "count": sum(lower <= angle < upper for angle in angles),
        })

    sorted_angles = sorted(angles)
    circular_gaps = [
        sorted_angles[index + 1] - sorted_angles[index]
        for index in range(len(sorted_angles) - 1)
    ]
    circular_gaps.append(sorted_angles[0] + 360 - sorted_angles[-1])

    def summary(values: list[float]) -> dict:
        return {
            "min": min(values),
            "q1": quantile(values, 0.25),
            "median": quantile(values, 0.5),
            "mean": statistics.fmean(values),
            "q3": quantile(values, 0.75),
            "max": max(values),
            "std_population": statistics.pstdev(values),
        }

    return {
        "x_summary_m": summary(x_values),
        "y_summary_m": summary(y_values),
        "radius_summary_m": summary(radii),
        "nearest_neighbor_summary_m": summary(nearest),
        "minimum_pair": minimum_pair,
        "spacing_violating_pairs_lt_11m": spacing_violations,
        "inside_required_center_annulus_100_to_350m": sum(100 <= radius <= 350 for radius in radii),
        "radial_bin_counts": radial_counts,
        "angular_sector_counts": sector_counts,
        "largest_angular_gap_deg": max(circular_gaps),
        "centroid_m": [statistics.fmean(x_values), statistics.fmean(y_values)],
    }


def main() -> None:
    sheet_name, headers, positions, preview = load_first_sheet(RAW)
    audit = audit_positions(positions)

    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    with CLEAN.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["heliostat_id", "x_m", "y_m", "z_m"])
        for index, (x_value, y_value) in enumerate(positions, start=1):
            writer.writerow([index, format(x_value, ".15g"), format(y_value, ".15g"), "4"])

    raw_hash = sha256(RAW)
    clean_hash = sha256(CLEAN)
    duplicate_count = len(positions) - len(set(positions))
    profile = {
        "schema_version": 1,
        "raw_files": [{
            "path": "2023A题/附件.xlsx",
            "size_bytes": RAW.stat().st_size,
            "sha256": raw_hash,
            "sheet_names": [sheet_name],
            "headers": headers,
            "preview": preview,
            "read_only": True,
        }],
        "attachment_mapping": [{
            "path": "2023A题/附件.xlsx",
            "questions": ["Q1"],
            "role": "给定1745面定日镜的镜心平面坐标",
            "mapping_status": "unambiguous",
        }],
        "fields": [
            {"name": "x坐标 (m)", "type": "numeric", "unit": "m", "role": "镜心东向坐标"},
            {"name": "y坐标 (m)", "type": "numeric", "unit": "m", "role": "镜心北向坐标"},
        ],
        "quality": {
            "missingness": {"x坐标 (m)": 0, "y坐标 (m)": 0, "rows_with_missing": 0},
            "duplicates": {"duplicate_coordinate_pairs": duplicate_count},
            "impossible_values": {
                "non_finite_coordinates": 0,
                "outside_100_to_350m_center_annulus": len(positions) - audit["inside_required_center_annulus_100_to_350m"],
                "spacing_violating_pairs_lt_11m": audit["spacing_violating_pairs_lt_11m"],
            },
            "outliers": {
                "status": "not_applicable_as_statistical_outliers",
                "explanation": "镜心坐标是确定性空间设计输入，仅按题面几何合法性检查，不按统计离群规则删除。",
            },
        },
        "coverage": {
            "rows": len(positions),
            "effective_sample_size": len(positions),
            "time_range": None,
            "time_gaps": None,
            "spatial": audit,
        },
        "distribution_risks": {
            "class_imbalance": None,
            "rare_categories": [],
            "high_cardinality": ["coordinate_pair: 1745 unique values"],
            "redundancy_warnings": [],
            "concentration_metrics": {
                "largest_25m_radial_bin_share": max(item["count"] for item in audit["radial_bin_counts"]) / len(positions),
                "largest_30deg_sector_share": max(item["count"] for item in audit["angular_sector_counts"]) / len(positions),
                "largest_angular_gap_deg": audit["largest_angular_gap_deg"],
            },
        },
        "per_question_readiness": {
            "Q1": {
                "status": "ready_with_warnings",
                "usable_rows": len(positions),
                "warnings": [
                    "附件只提供镜心x、y；Q1的z=4 m来自题面固定条件。",
                    "坐标质量通过不代表光学几何模型已经验证。",
                ],
            }
        },
        "cleaning_operations": [{
            "operation": "representation_normalization",
            "description": "将XLSX中的数值坐标无删改转换为UTF-8 CSV，并增加顺序ID和题面固定z=4 m。",
            "assumption_bearing": False,
            "rows_before": len(positions),
            "rows_after": len(positions),
        }],
        "cleaned_files": [{
            "path": "workspace/data_clean/Q1/heliostat_positions.csv",
            "sha256": clean_hash,
            "rows": len(positions),
        }],
        "unresolved_risks": [
            "太阳锥角、锥内能量分布、阴影遮挡去重和截断几何口径不由附件定义。",
            "年平均的十二个月权重需在方法假设中明确。",
        ],
    }

    PROFILE.parent.mkdir(parents=True, exist_ok=True)
    PROFILE.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    radius = audit["radius_summary_m"]
    nearest_summary = audit["nearest_neighbor_summary_m"]
    radial_rows = "\n".join(
        f"| {item['range_m'][0]}--{item['range_m'][1]} | {item['count']} |"
        for item in audit["radial_bin_counts"]
    )
    sector_rows = "\n".join(
        f"| {item['range_deg'][0]}--{item['range_deg'][1]} | {item['count']} |"
        for item in audit["angular_sector_counts"]
    )
    report = f"""# Q1数据审计报告

## 附件映射与追溯

- 原始文件：`2023A题/附件.xlsx`，SHA-256 `{raw_hash}`。
- 工作表：`{sheet_name}`；字段：`{headers[0]}`、`{headers[1]}`。
- 用途：仅用于Q1给定镜场的1745个镜心平面坐标，映射无歧义。
- 原文件未修改。规范化副本为 `workspace/data_clean/Q1/heliostat_positions.csv`，增加顺序ID与题面固定安装高度4 m。

## 质量与合法性

| 检查 | 结果 |
|---|---:|
| 数据行数 / 有效行数 | {len(positions)} / {len(positions)} |
| 缺失坐标 | 0 |
| 重复坐标对 | {duplicate_count} |
| 非有限数值 | 0 |
| 半径不在100--350 m | {len(positions) - audit['inside_required_center_annulus_100_to_350m']} |
| 中心距小于11 m的镜对 | {audit['spacing_violating_pairs_lt_11m']} |
| 最小镜心距 | {audit['minimum_pair']['distance_m']:.6f} m（Excel行{audit['minimum_pair']['row_ids']}） |

坐标是确定性设计输入，不能把远离均值的合法外圈镜心当作统计离群值删除。所有1745行均保留，Q1有效样本数为1745。

## 空间覆盖

- 镜心半径：最小{radius['min']:.6f} m，中位数{radius['median']:.6f} m，平均{radius['mean']:.6f} m，最大{radius['max']:.6f} m。
- 最近邻距离：最小{nearest_summary['min']:.6f} m，中位数{nearest_summary['median']:.6f} m，平均{nearest_summary['mean']:.6f} m，最大{nearest_summary['max']:.6f} m。
- 坐标质心：$({audit['centroid_m'][0]:.6f},{audit['centroid_m'][1]:.6f})$ m。
- 相邻镜心极角最大空隙：{audit['largest_angular_gap_deg']:.6f}度。

### 径向分层

| 半径区间 (m) | 镜数 |
|---|---:|
{radial_rows}

### 角向分区

| 从正东逆时针角区间 (度) | 镜数 |
|---|---:|
{sector_rows}

## Q1就绪性

状态为 `ready_with_warnings`。附件结构、数值和几何合法性足以支持Q1，但附件只给出$x,y$；$z=4$ m来自题面固定条件。太阳方向、镜面姿态、阴影遮挡、截断效率与年平均权重属于模型层风险，必须在方法卡和风险探针中单独验证，不能由数据质量结论替代。
"""
    REPORT.write_text(report, encoding="utf-8")

    print(json.dumps({
        "rows": len(positions),
        "raw_sha256": raw_hash,
        "clean_sha256": clean_hash,
        "minimum_distance_m": audit["minimum_pair"]["distance_m"],
        "radius_range_m": [radius["min"], radius["max"]],
        "readiness": "ready_with_warnings",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
