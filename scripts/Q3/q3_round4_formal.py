#!/usr/bin/env python3
"""Formal recheck of the selected Q3 coupled width-position candidate."""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/Q1"))
sys.path.insert(0, str(ROOT / "scripts/Q3"))
import q1_optical_model as q1  # noqa: E402
import q3_optical_model as q3  # noqa: E402

SOURCE = ROOT / os.environ.get(
    "Q3_FORMAL_SOURCE",
    "results/Q3/experiments/round3_width_position/tables/q3_width_position_best_layout.csv",
)
BASELINE = ROOT / "results/Q2/experiments/round17_zoned_formal/metrics/q2_zoned_formal.json"
ROUND_DIR = ROOT / os.environ.get("Q3_FORMAL_ROUND_DIR", "results/Q3/experiments/round4_formal")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started = time.perf_counter()
    table_dir, metric_dir = ROUND_DIR / "tables", ROUND_DIR / "metrics"
    table_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(SOURCE)
    points = np.asarray([[float(r["x_m"]), float(r["y_m"])] for r in rows])
    widths = np.asarray([float(r["width_m"]) for r in rows])
    heights = np.asarray([float(r["height_m"]) for r in rows])
    z = np.asarray([float(r["z_m"]) for r in rows])
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["formal"]
    tower_x = float(os.environ.get("Q3_FORMAL_TOWER_X", baseline["tower_x_m"]))
    tower_y = float(os.environ.get("Q3_FORMAL_TOWER_Y", baseline["tower_y_m"]))
    design = q3.HeterogeneousDesign(
        "M3_width_position_formal", "M3", "main_candidate", tower_x, tower_y,
    )
    geometry = q3.geometry_checks(points, widths, heights, z, design)
    if geometry["status"] != "PASS":
        raise RuntimeError(f"Illegal Q3 finalist: {geometry}")
    states = q1.build_solar_states()
    coarse, coarse_seeds, coarse_runtime = q3.evaluate_resolution(points, widths, heights, z, design, states, 16, 8, q1.SEEDS, 64)
    fine, fine_seeds, fine_runtime = q3.evaluate_resolution(points, widths, heights, z, design, states, 32, 16, q1.SEEDS, 64)
    areas = widths * heights
    coarse_monthly, coarse_annual = q3.aggregate(coarse, states, areas)
    fine_monthly, fine_annual = q3.aggregate(fine, states, areas)
    seed_power = [row["annual"]["field_power_mw"] for row in fine_seeds]
    monthly_delta = [abs(a["unit_area_power_kw_m2"] - b["unit_area_power_kw_m2"]) / b["unit_area_power_kw_m2"] for a, b in zip(coarse_monthly, fine_monthly)]
    convergence = {
        "annual_eta_total_abs_delta": abs(fine_annual["eta_total"] - coarse_annual["eta_total"]),
        "annual_field_power_relative_delta": abs(fine_annual["field_power_mw"] - coarse_annual["field_power_mw"]) / fine_annual["field_power_mw"],
        "max_monthly_unit_power_relative_delta": max(monthly_delta),
        "fine_seed_field_power_mw": seed_power,
        "fine_seed_field_power_mw_std": float(np.std(seed_power)),
        "fine_seed_field_power_mw_range": float(np.ptp(seed_power)),
    }
    convergence["status"] = "PASS" if convergence["annual_eta_total_abs_delta"] <= 0.002 and convergence["annual_field_power_relative_delta"] <= 0.003 and convergence["max_monthly_unit_power_relative_delta"] <= 0.005 else "CONDITIONAL"
    comparison = {
        "main": fine_annual,
        "baseline": {k: baseline[k] for k in ("eta_total", "eta_cos", "eta_sb", "eta_trunc", "field_power_mw", "unit_area_power_kw_m2", "total_area_m2")},
        "power_delta_mw": fine_annual["field_power_mw"] - baseline["field_power_mw"],
        "unit_area_delta_kw_m2": fine_annual["unit_area_power_kw_m2"] - baseline["unit_area_power_kw_m2"],
        "unit_area_relative_gain": (fine_annual["unit_area_power_kw_m2"] - baseline["unit_area_power_kw_m2"]) / baseline["unit_area_power_kw_m2"],
        "pooled_feasible": fine_annual["field_power_mw"] >= 60.0,
        "all_seeds_feasible": min(seed_power) >= 60.0,
        "minimum_seed_power_mw": min(seed_power),
    }
    output = {
        "design": {"design_id": design.design_id, "tower_x_m": design.tower_x_m, "tower_y_m": design.tower_y_m, "mirror_count": len(points)},
        "geometry": geometry, "coarse": coarse_annual, "fine": fine_annual,
        "coarse_seed_summaries": coarse_seeds, "fine_seed_summaries": fine_seeds,
        "convergence": convergence, "comparison": comparison,
        "specifications": {"unique_width_6dp": int(len(np.unique(np.round(widths, 6)))), "unique_height_6dp": int(len(np.unique(np.round(heights, 6)))), "unique_z_6dp": int(len(np.unique(np.round(z, 6)))), "width_range_m": [float(widths.min()), float(widths.max())], "height_range_m": [float(heights.min()), float(heights.max())], "total_area_m2": float(np.sum(areas))},
    }
    write_csv(table_dir / "q3_finalist_layout.csv", rows)
    write_csv(table_dir / "q3_finalist_monthly.csv", fine_monthly)
    (metric_dir / "q3_formal.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1, "question": "Q3", "round": ROUND_DIR.name, "implementation_target": "python", "approved_decision_id": os.environ.get("Q3_FORMAL_DECISION_ID", "q3_global_restart_revision"), "random_seed": list(q1.SEEDS),
        "methods": [{"method_id": "M3G", "role": "main_candidate", "status": "success", "script": "scripts/Q3/q3_round4_formal.py", "execution_time_seconds": coarse_runtime + fine_runtime, "output_files": [str((table_dir / "q3_finalist_layout.csv").relative_to(ROOT)), str((table_dir / "q3_finalist_monthly.csv").relative_to(ROOT)), str((metric_dir / "q3_formal.json").relative_to(ROOT))], "metrics_summary": fine_annual, "warnings": [], "errors": []}, {"method_id": "B3", "role": "usable_baseline", "status": "success", "script": "results/Q2/experiments/round17_zoned_formal", "execution_time_seconds": 0, "output_files": [str(BASELINE.relative_to(ROOT))], "metrics_summary": comparison["baseline"], "warnings": [], "errors": []}],
        "comparison": comparison,
        "fallback_trigger": {"fallback_id": "F3", "condition": "formal candidate fails pooled/all-seed capacity or unit-area improvement", "observed": (not comparison["pooled_feasible"] or not comparison["all_seeds_feasible"] or comparison["unit_area_delta_kw_m2"] <= 0), "evidence": str((metric_dir / "q3_formal.json").relative_to(ROOT))},
        "environment": {"python": sys.version, "platform": platform.platform()}, "runtime_seconds": time.perf_counter() - started,
    }
    (ROUND_DIR / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"comparison": comparison, "convergence": convergence, "specifications": output["specifications"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
