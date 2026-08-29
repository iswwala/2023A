#!/usr/bin/env python3
"""Formal four-seed evaluation of the best symmetric zoned-transition candidate."""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/Q2"))
import q2_optical_model as q2  # noqa: E402

SOURCE = ROOT / "results/Q2/experiments/round16_zoned_transition_pso"
ROUND_DIR = ROOT / "results/Q2/experiments/round17_zoned_formal"
TABLE_DIR = ROUND_DIR / "tables"
METRIC_DIR = ROUND_DIR / "metrics"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started = time.perf_counter()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    source_row = read_csv(SOURCE / "tables/q2_zoned_transition_full_rechecks.csv")[0]
    source_layout = SOURCE / "tables/q2_zoned_transition_candidate_1_layout.csv"
    points = np.asarray(
        [[float(row["x_m"]), float(row["y_m"])] for row in read_csv(source_layout)]
    )
    design = q2.Design(
        "M2ZT_symmetric_zoned_formal", "M2ZT", "main_candidate_finalist",
        0.0, float(source_row["tower_y_m"]), float(source_row["width_m"]),
        float(source_row["height_m"]), float(source_row["installation_height_m"]),
    )
    geometry = q2.geometry_checks(points, design)
    if geometry["status"] != "PASS":
        raise RuntimeError(f"Formal geometry failed: {geometry}")
    states = q2.q1.build_solar_states()
    dni = np.asarray([state.dni_kw_m2 for state in states])
    coarse, coarse_seeds, coarse_runtime = q2.evaluate_resolution(
        points, design, states, 16, 8, q2.SEEDS, 64
    )
    fine, fine_seeds, fine_runtime = q2.evaluate_resolution(
        points, design, states, 32, 16, q2.SEEDS, 64
    )
    monthly, annual = q2.aggregate(fine, states, dni, design.width_m * design.height_m)
    convergence = q2.convergence_metrics(
        design.design_id, coarse, fine, states, design.width_m * design.height_m,
        fine_seeds, (16, 8), (32, 16),
    )
    fine_power = [row["annual"]["field_power_mw"] for row in fine_seeds]
    formal = {
        "design_id": design.design_id,
        "tower_x_m": 0.0,
        "tower_y_m": design.tower_y_m,
        "width_m": design.width_m,
        "height_m": design.height_m,
        "installation_height_m": design.installation_height_m,
        "mirror_count": len(points),
        "total_area_m2": len(points) * design.width_m * design.height_m,
        "minimum_spacing_m": geometry["minimum_spacing_m"],
        "unpaired_mirror_count": 0,
        "inner_boundary_m": float(source_row["inner_boundary_m"]),
        "middle_boundary_m": float(source_row["middle_boundary_m"]),
        "inner_phase": float(source_row["inner_phase"]),
        "middle_phase": float(source_row["middle_phase"]),
        "outer_phase": float(source_row["outer_phase"]),
        **annual,
        "power_margin_mw": annual["field_power_mw"] - q2.TARGET_POWER_MW,
        "fine_seed_power_mw": fine_power,
        "minimum_fine_seed_power_mw": min(fine_power),
        "pooled_mean_feasible": annual["field_power_mw"] >= q2.TARGET_POWER_MW,
        "all_seeds_feasible": min(fine_power) >= q2.TARGET_POWER_MW,
        "convergence_status": convergence["status"],
    }
    write_csv(TABLE_DIR / "q2_final_layout.csv", read_csv(source_layout))
    write_csv(TABLE_DIR / "q2_final_monthly.csv", monthly)
    evidence = {
        "formal": formal,
        "geometry": geometry,
        "convergence": convergence,
        "coarse_seed_summaries": coarse_seeds,
        "fine_seed_summaries": fine_seeds,
    }
    (METRIC_DIR / "q2_zoned_formal.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "question": "Q2",
        "round": "round17_zoned_formal",
        "status": "success",
        "approved_decision_ids": ["q2_tower_shadow_revision", "q2_zoned_layout_revision", "q2_uniform_mirror_shape_revision"],
        "source_layout": str(source_layout.relative_to(ROOT)),
        "formal": formal,
        "seed": list(q2.SEEDS),
        "runtime_seconds": time.perf_counter() - started,
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    (ROUND_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(formal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
