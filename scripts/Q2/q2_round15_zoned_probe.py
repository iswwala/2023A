#!/usr/bin/env python3
"""Risk probe for symmetric concentric zones with different layout strategies."""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/Q2"))
import q2_optical_model as q2  # noqa: E402
import q2_round13_symmetric_global_pso as symmetric  # noqa: E402

ROUND_DIR = ROOT / "results/Q2/experiments/round15_zoned_probe"
TABLE_DIR = ROUND_DIR / "tables"
METRIC_DIR = ROUND_DIR / "metrics"


@dataclass(frozen=True)
class Specification:
    strategy: str
    size_m: float
    tower_y_m: float
    inner_boundary_m: float
    middle_boundary_m: float
    phases: tuple[float, float, float]


SPECIFICATIONS = (
    Specification("uniform_triangular", 6.70, -40.0, 170.0, 255.0, (0.0, 0.0, 0.0)),
    Specification("zoned_triangular", 6.70, -40.0, 160.0, 250.0, (0.0, 0.25, 0.50)),
    Specification("zoned_triangular", 6.70, -40.0, 180.0, 270.0, (0.50, 0.0, 0.50)),
    Specification("zoned_triangular", 6.70, -20.0, 160.0, 250.0, (0.0, 0.25, 0.50)),
    Specification("zoned_triangular", 6.70, -60.0, 180.0, 270.0, (0.50, 0.0, 0.50)),
    Specification("inner_concentric", 6.70, -40.0, 150.0, 250.0, (0.0, 0.25, 0.50)),
    Specification("inner_concentric", 6.70, -20.0, 170.0, 260.0, (0.0, 0.0, 0.50)),
    Specification("inner_concentric", 6.70, -60.0, 170.0, 260.0, (0.0, 0.0, 0.50)),
    Specification("zoned_triangular", 6.80, -40.0, 160.0, 250.0, (0.0, 0.25, 0.50)),
    Specification("inner_concentric", 6.80, -40.0, 160.0, 250.0, (0.0, 0.25, 0.50)),
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ring_candidates(design: q2.Design, maximum_radius: float) -> np.ndarray:
    spacing = design.width_m + 5.0
    tower = np.asarray([0.0, design.tower_y_m])
    rows: list[np.ndarray] = []
    radius = q2.EXCLUSION_RADIUS + spacing
    while radius < maximum_radius - 1e-9:
        maximum_count = int(math.floor(math.pi / math.asin(spacing / (2.0 * radius))))
        count = max(4, maximum_count - maximum_count % 2)
        angles = 2.0 * math.pi * np.arange(count) / count
        relative = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
        absolute = relative + tower
        legal = np.linalg.norm(absolute, axis=1) <= q2.FIELD_RADIUS + 1e-9
        rows.append(absolute[legal])
        radius += spacing
    return np.vstack(rows) if rows else np.empty((0, 2))


def pair_groups(points: np.ndarray, priority: int) -> list[tuple[int, np.ndarray]]:
    unique = {(round(abs(float(x)), 8), round(float(y), 8)) for x, y in points}
    groups = []
    for abs_x, y in unique:
        if abs_x <= 1e-8:
            group = np.asarray([[0.0, y]])
        else:
            group = np.asarray([[-abs_x, y], [abs_x, y]])
        groups.append((priority, group))
    return groups


def accept_symmetric_groups(
    groups: list[tuple[int, np.ndarray]], minimum_spacing: float
) -> np.ndarray:
    cell_size = minimum_spacing
    cells: dict[tuple[int, int], list[np.ndarray]] = {}
    accepted: list[np.ndarray] = []

    def cell(point: np.ndarray) -> tuple[int, int]:
        return (math.floor(point[0] / cell_size), math.floor(point[1] / cell_size))

    def clear(point: np.ndarray, pending: list[np.ndarray]) -> bool:
        cx, cy = cell(point)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in cells.get((cx + dx, cy + dy), []):
                    if np.linalg.norm(point - other) < minimum_spacing - 1e-8:
                        return False
        return all(np.linalg.norm(point - other) >= minimum_spacing - 1e-8 for other in pending)

    for _, group in sorted(groups, key=lambda item: (item[0], float(np.mean(np.linalg.norm(item[1], axis=1))))):
        pending: list[np.ndarray] = []
        if all(clear(point, pending) and not pending.append(point) for point in group):
            for point in pending:
                accepted.append(point)
                cells.setdefault(cell(point), []).append(point)
    return np.asarray(accepted)


def zoned_layout(spec: Specification, design: q2.Design) -> np.ndarray:
    tower = np.asarray([0.0, design.tower_y_m])
    boundaries = (q2.EXCLUSION_RADIUS, spec.inner_boundary_m, spec.middle_boundary_m, np.inf)
    groups: list[tuple[int, np.ndarray]] = []
    if spec.strategy == "inner_concentric":
        groups.extend(pair_groups(ring_candidates(design, spec.inner_boundary_m), 0))
        zone_start = 1
    else:
        zone_start = 0
    for zone in range(zone_start, 3):
        pool = symmetric.symmetric_triangular_layout(design, spec.phases[zone])
        radius = np.linalg.norm(pool - tower, axis=1)
        selected = pool[(radius >= boundaries[zone] - 1e-9) & (radius < boundaries[zone + 1] - 1e-9)]
        groups.extend(pair_groups(selected, zone + 1))
    points = accept_symmetric_groups(groups, design.width_m + 5.0)
    return points[np.lexsort((points[:, 0], points[:, 1]))]


def main() -> None:
    started = time.perf_counter()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    states = q2.q1.build_solar_states()
    dni = np.asarray([state.dni_kw_m2 for state in states])
    rows: list[dict[str, Any]] = []
    for rank, spec in enumerate(SPECIFICATIONS, start=1):
        design = q2.Design(
            f"M2Z_probe_{rank}", "M2Z", "zoned_probe", 0.0, spec.tower_y_m,
            spec.size_m, spec.size_m, spec.size_m / 2.0 + 0.55,
        )
        points = zoned_layout(spec, design)
        geometry = q2.geometry_checks(points, design)
        symmetry_check = symmetric.symmetry_check(points)
        if geometry["status"] != "PASS" or symmetry_check["status"] != "PASS":
            raise RuntimeError(f"Probe geometry failed: {rank}, {geometry}, {symmetry_check}")
        components, _, runtime = q2.evaluate_resolution(
            points, design, states, 4, 2, (q2.SEEDS[0],), 64
        )
        _, annual = q2.aggregate(components, states, dni, design.width_m * design.height_m)
        row = {
            "candidate_rank": rank,
            "strategy": spec.strategy,
            "size_m": spec.size_m,
            "tower_y_m": spec.tower_y_m,
            "inner_boundary_m": spec.inner_boundary_m,
            "middle_boundary_m": spec.middle_boundary_m,
            "inner_phase": spec.phases[0],
            "middle_phase": spec.phases[1],
            "outer_phase": spec.phases[2],
            "mirror_count": len(points),
            "total_area_m2": len(points) * design.width_m * design.height_m,
            "minimum_spacing_m": geometry["minimum_spacing_m"],
            "unpaired_mirror_count": symmetry_check["unpaired_mirror_count"],
            **annual,
            "power_margin_mw": annual["field_power_mw"] - q2.TARGET_POWER_MW,
            "runtime_seconds": runtime,
        }
        rows.append(row)
        write_csv(TABLE_DIR / "q2_zoned_probe.csv", rows)
        write_csv(
            TABLE_DIR / f"q2_zoned_probe_candidate_{rank}_layout.csv",
            [
                {"heliostat_id": i, "x_m": p[0], "y_m": p[1], "z_m": design.installation_height_m,
                 "width_m": design.width_m, "height_m": design.height_m}
                for i, p in enumerate(points, start=1)
            ],
        )
        print(
            f"rank={rank} strategy={spec.strategy} n={len(points)} "
            f"power={annual['field_power_mw']:.6f} unit={annual['unit_area_power_kw_m2']:.6f}",
            flush=True,
        )
    baseline = rows[0]
    best_unit = max(rows, key=lambda row: row["unit_area_power_kw_m2"])
    best_power = max(rows, key=lambda row: row["field_power_mw"])
    summary = {
        "schema_version": 1,
        "question": "Q2",
        "round": "round15_zoned_probe",
        "status": "success",
        "approved_decision_id": "q2_zoned_layout_revision",
        "samples": [4, 2],
        "seed": q2.SEEDS[0],
        "baseline_uniform_triangular": baseline,
        "best_unit_area_candidate": best_unit,
        "best_power_candidate": best_power,
        "unit_gain_vs_uniform_kw_m2": best_unit["unit_area_power_kw_m2"] - baseline["unit_area_power_kw_m2"],
        "power_gain_vs_uniform_mw": best_power["field_power_mw"] - baseline["field_power_mw"],
        "verdict": "PASS" if best_power["field_power_mw"] > baseline["field_power_mw"] else "CONDITIONAL",
        "runtime_seconds": time.perf_counter() - started,
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    (METRIC_DIR / "q2_zoned_probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROUND_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
