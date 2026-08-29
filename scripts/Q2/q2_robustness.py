#!/usr/bin/env python3
"""Risk-targeted formal robustness checks for the approved Q2 M2 design."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MATPLOTLIB_CONFIG = ROOT / "workspace" / "analysis" / "Q2" / ".mplconfig"
MATPLOTLIB_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG))

import numpy as np
import tensorflow as tf


sys.path.insert(0, str(ROOT / "scripts" / "Q2"))
import q2_optical_model as q2  # noqa: E402


ROUND_DIR = ROOT / "results" / "Q2" / "experiments" / "round1"
SUMMARY_PATH = ROOT / "results" / "Q2" / "robustness" / "q2_robustness_summary.json"
TARGET_POWER_MW = 60.0
SURFACE_SAMPLES = 32
SUN_SAMPLES = 16


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_monthly(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def evaluate(
    design: q2.Design,
    points: np.ndarray,
    states: list[q2.q1.SolarState],
    batch_size: int,
    half_angle: float = q2.q1.SUN_HALF_ANGLE,
) -> dict[str, Any]:
    components, seed_summaries, runtime = q2.evaluate_resolution(
        points,
        design,
        states,
        SURFACE_SAMPLES,
        SUN_SAMPLES,
        q2.SEEDS,
        batch_size,
        half_angle,
    )
    dni = np.asarray([state.dni_kw_m2 for state in states], dtype=np.float64)
    _, annual = q2.aggregate(components, states, dni, design.width_m * design.height_m)
    result = {
        "design": {
            "design_id": design.design_id,
            "tower_xy_m": [design.tower_x_m, design.tower_y_m],
            "spacing_factor": design.spacing_factor,
            "lattice_angle_deg": design.lattice_angle_deg,
            "phase_x_fraction": design.phase_x_fraction,
            "phase_y_fraction": design.phase_y_fraction,
        },
        "mirror_count": len(points),
        "total_area_m2": len(points) * design.width_m * design.height_m,
        "geometry": q2.geometry_checks(points, design),
        "annual": annual,
        "power_margin_mw": annual["field_power_mw"] - TARGET_POWER_MW,
        "seed_field_power_mw": [row["annual"]["field_power_mw"] for row in seed_summaries],
        "seed_power_std_mw": float(
            np.std([row["annual"]["field_power_mw"] for row in seed_summaries])
        ),
        "seed_power_range_mw": float(
            np.ptp([row["annual"]["field_power_mw"] for row in seed_summaries])
        ),
        "runtime_seconds": runtime,
    }
    del components
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    started = time.perf_counter()

    run_summary = read_json(ROUND_DIR / "run_summary.json")
    convergence = read_json(ROUND_DIR / "metrics" / "q2_convergence.json")
    if run_summary["approved_decision_id"] != "q2_method_choice":
        raise RuntimeError("Q2 approved decision mismatch")
    if convergence["status"] != "PASS":
        raise RuntimeError("Base Q2 convergence must pass before robustness checks")

    states = q2.q1.build_solar_states()
    main_design = q2.Design(
        "M2_south40_spacing101",
        "M2",
        "main_candidate",
        0.0,
        -40.0,
        7.0,
        7.0,
        4.0,
        spacing_factor=1.01,
    )
    baseline_design = q2.Design(
        "B2_fixed_7m_centered_hex",
        "B2",
        "usable_baseline",
        0.0,
        0.0,
        7.0,
        7.0,
        4.0,
    )
    main_points = np.loadtxt(
        ROUND_DIR / "tables" / "q2_main_layout.csv", delimiter=",", skiprows=1, usecols=(1, 2)
    )
    baseline_points = np.loadtxt(
        ROUND_DIR / "tables" / "q2_baseline_layout.csv", delimiter=",", skiprows=1, usecols=(1, 2)
    )
    core_main = next(row for row in run_summary["methods"] if row["method_id"] == "M2")["metrics_summary"]
    core_baseline = next(row for row in run_summary["methods"] if row["method_id"] == "B2")["metrics_summary"]

    perturbations = [
        replace(main_design, design_id="tower_west_5m", tower_x_m=-5.0),
        replace(main_design, design_id="tower_east_5m", tower_x_m=5.0),
        replace(main_design, design_id="tower_north_5m", tower_y_m=-35.0),
        replace(main_design, design_id="tower_south_5m", tower_y_m=-45.0),
        replace(main_design, design_id="spacing_minus_0p005", spacing_factor=1.005),
        replace(main_design, design_id="spacing_plus_0p005", spacing_factor=1.015),
        replace(main_design, design_id="angle_minus_5deg", lattice_angle_deg=-5.0),
        replace(main_design, design_id="angle_plus_5deg", lattice_angle_deg=5.0),
        replace(main_design, design_id="phase_x_minus_0p25", phase_x_fraction=-0.25),
        replace(main_design, design_id="phase_x_plus_0p25", phase_x_fraction=0.25),
        replace(main_design, design_id="phase_y_minus_0p25", phase_y_fraction=-0.25),
        replace(main_design, design_id="phase_y_plus_0p25", phase_y_fraction=0.25),
    ]
    local_results: list[dict[str, Any]] = []
    for design in perturbations:
        print(f"local perturbation {design.design_id} started", flush=True)
        row = evaluate(design, q2.hexagonal_layout(design), states, args.batch_size)
        row["unit_area_gain_vs_core_b2_kw_m2"] = (
            row["annual"]["unit_area_power_kw_m2"]
            - core_baseline["unit_area_power_kw_m2"]
        )
        row["unit_area_relative_gain_vs_core_b2"] = (
            row["unit_area_gain_vs_core_b2_kw_m2"]
            / core_baseline["unit_area_power_kw_m2"]
        )
        local_results.append(row)
        print(
            f"local perturbation {design.design_id} power={row['annual']['field_power_mw']:.6f} "
            f"unit={row['annual']['unit_area_power_kw_m2']:.6f} completed",
            flush=True,
        )

    sunshape_results: list[dict[str, Any]] = []
    for factor in (0.95, 1.05):
        paired: dict[str, Any] = {"half_angle_factor": factor, "half_angle_rad": q2.q1.SUN_HALF_ANGLE * factor}
        paired["M2"] = evaluate(
            replace(main_design, design_id=f"M2_sunshape_{factor:.2f}"),
            main_points,
            states,
            args.batch_size,
            q2.q1.SUN_HALF_ANGLE * factor,
        )
        paired["B2"] = evaluate(
            replace(baseline_design, design_id=f"B2_sunshape_{factor:.2f}"),
            baseline_points,
            states,
            args.batch_size,
            q2.q1.SUN_HALF_ANGLE * factor,
        )
        paired["unit_area_gain_kw_m2"] = (
            paired["M2"]["annual"]["unit_area_power_kw_m2"]
            - paired["B2"]["annual"]["unit_area_power_kw_m2"]
        )
        paired["unit_area_relative_gain"] = (
            paired["unit_area_gain_kw_m2"]
            / paired["B2"]["annual"]["unit_area_power_kw_m2"]
        )
        sunshape_results.append(paired)

    main_monthly = read_monthly(ROUND_DIR / "tables" / "q2_main_monthly.csv")
    baseline_monthly = read_monthly(ROUND_DIR / "tables" / "q2_baseline_monthly.csv")
    main_day_power = float(
        np.average([row["field_power_mw"] for row in main_monthly], weights=q2.q1.MONTH_DAYS)
    )
    baseline_day_power = float(
        np.average([row["field_power_mw"] for row in baseline_monthly], weights=q2.q1.MONTH_DAYS)
    )
    main_day_unit = float(
        np.average([row["unit_area_power_kw_m2"] for row in main_monthly], weights=q2.q1.MONTH_DAYS)
    )
    baseline_day_unit = float(
        np.average([row["unit_area_power_kw_m2"] for row in baseline_monthly], weights=q2.q1.MONTH_DAYS)
    )
    month_weighting = {
        "main_day_weighted_power_mw": main_day_power,
        "baseline_day_weighted_power_mw": baseline_day_power,
        "main_day_weighted_unit_area_kw_m2": main_day_unit,
        "baseline_day_weighted_unit_area_kw_m2": baseline_day_unit,
        "unit_area_gain_kw_m2": main_day_unit - baseline_day_unit,
        "unit_area_relative_gain": (main_day_unit - baseline_day_unit) / baseline_day_unit,
    }

    main_convergence = convergence["methods"]["M2"]
    baseline_convergence = convergence["methods"]["B2"]
    main_seed_power = np.asarray(main_convergence["fine_seed_field_power_mw"], dtype=np.float64)
    baseline_seed_power = np.asarray(baseline_convergence["fine_seed_field_power_mw"], dtype=np.float64)
    main_area = len(main_points) * main_design.width_m * main_design.height_m
    baseline_area = len(baseline_points) * baseline_design.width_m * baseline_design.height_m
    paired_seed_gain = main_seed_power * 1000.0 / main_area - baseline_seed_power * 1000.0 / baseline_area
    seed_stability = {
        "main_power_mw": main_seed_power.tolist(),
        "baseline_power_mw": baseline_seed_power.tolist(),
        "paired_unit_area_gain_kw_m2": paired_seed_gain.tolist(),
        "minimum_main_power_mw": float(np.min(main_seed_power)),
        "minimum_paired_unit_area_gain_kw_m2": float(np.min(paired_seed_gain)),
    }
    coarse_gain = (
        main_convergence["coarse"]["annual"]["unit_area_power_kw_m2"]
        - baseline_convergence["coarse"]["annual"]["unit_area_power_kw_m2"]
    )
    fine_gain = (
        main_convergence["fine"]["annual"]["unit_area_power_kw_m2"]
        - baseline_convergence["fine"]["annual"]["unit_area_power_kw_m2"]
    )
    numerical_resolution = {
        "coarse_unit_area_gain_kw_m2": coarse_gain,
        "fine_unit_area_gain_kw_m2": fine_gain,
        "minimum_gain_kw_m2": min(coarse_gain, fine_gain),
        "main_convergence_status": main_convergence["status"],
        "baseline_convergence_status": baseline_convergence["status"],
    }

    local_power_min = min(row["annual"]["field_power_mw"] for row in local_results)
    local_gain_min = min(row["unit_area_gain_vs_core_b2_kw_m2"] for row in local_results)
    sun_power_min = min(row["M2"]["annual"]["field_power_mw"] for row in sunshape_results)
    sun_gain_min = min(row["unit_area_gain_kw_m2"] for row in sunshape_results)
    checks = {
        "local_geometry": {
            "status": "PASS" if all(row["geometry"]["status"] == "PASS" for row in local_results) else "FAIL",
            "threshold": "all perturbed layouts geometrically legal",
            "observed": all(row["geometry"]["status"] == "PASS" for row in local_results),
        },
        "local_capacity": {
            "status": "PASS" if local_power_min >= TARGET_POWER_MW else "FAIL",
            "threshold_mw": TARGET_POWER_MW,
            "observed_minimum_mw": local_power_min,
        },
        "local_unit_area_advantage": {
            "status": "PASS" if local_gain_min > 0.0 else "FAIL",
            "threshold_kw_m2": 0.0,
            "observed_minimum_gain_kw_m2": local_gain_min,
        },
        "sunshape_capacity": {
            "status": "PASS" if sun_power_min >= TARGET_POWER_MW else "FAIL",
            "threshold_mw": TARGET_POWER_MW,
            "observed_minimum_mw": sun_power_min,
        },
        "sunshape_unit_area_advantage": {
            "status": "PASS" if sun_gain_min > 0.0 else "FAIL",
            "threshold_kw_m2": 0.0,
            "observed_minimum_gain_kw_m2": sun_gain_min,
        },
        "month_weighting": {
            "status": "PASS" if main_day_power >= TARGET_POWER_MW and month_weighting["unit_area_gain_kw_m2"] > 0.0 else "FAIL",
            "threshold": "day-weighted M2 power >= 60 MW and unit-area gain > 0",
            "observed": month_weighting,
        },
        "seed_stability": {
            "status": "PASS" if seed_stability["minimum_main_power_mw"] >= TARGET_POWER_MW and seed_stability["minimum_paired_unit_area_gain_kw_m2"] > 0.0 else "FAIL",
            "threshold": "every paired seed keeps M2 power >= 60 MW and unit-area gain > 0",
            "observed": seed_stability,
        },
        "numerical_resolution": {
            "status": "PASS" if convergence["status"] == "PASS" and numerical_resolution["minimum_gain_kw_m2"] > 0.0 else "FAIL",
            "threshold": "base convergence PASS and coarse/fine gain > 0",
            "observed": numerical_resolution,
        },
    }
    overall_status = "PASS" if all(row["status"] == "PASS" for row in checks.values()) else "FAIL"
    summary = {
        "schema_version": 1,
        "question_id": "Q2",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tested_claims": [
            "M2 formal annual power remains at least 60 MW under justified perturbations",
            "M2 unit-area annual power remains greater than the comparable B2 baseline",
        ],
        "source_paths": [
            "results/Q2/experiments/round1/run_summary.json",
            "results/Q2/experiments/round1/metrics/q2_convergence.json",
            "results/Q2/experiments/round1/tables/q2_main_layout.csv",
            "results/Q2/experiments/round1/tables/q2_baseline_layout.csv",
        ],
        "predeclared_perturbations": {
            "tower_coordinate_m": [-5.0, 5.0],
            "spacing_factor_delta": [-0.005, 0.005],
            "lattice_angle_deg": [-5.0, 5.0],
            "phase_fraction": [-0.25, 0.25],
            "sun_half_angle_relative": [-0.05, 0.05],
            "month_weighting": "equal month versus calendar-day weighting",
            "numerical": "four fixed seeds and nested 16x8 versus 32x16 resolution",
        },
        "thresholds": {
            "minimum_main_power_mw": TARGET_POWER_MW,
            "minimum_unit_area_gain_kw_m2": 0.0,
            "geometry": "PASS",
        },
        "core_result": {
            "M2": core_main,
            "B2": core_baseline,
        },
        "local_perturbations": local_results,
        "sunshape_perturbations": sunshape_results,
        "month_weighting": month_weighting,
        "seed_stability": seed_stability,
        "numerical_resolution": numerical_resolution,
        "checks": checks,
        "status": overall_status,
        "limitation": "Checks cover the approved structured-layout neighborhood and stated optical/time assumptions; they do not establish global optimality outside that domain.",
        "fallback_trigger_relevance": {
            "triggered": overall_status != "PASS",
            "reason": "A failed capacity or unit-area-advantage check requires adjust/fallback judgment; no automatic method replacement is authorized.",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "batch_size": args.batch_size,
            "surface_samples": SURFACE_SAMPLES,
            "sun_samples": SUN_SAMPLES,
            "seeds": list(q2.SEEDS),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    q2.write_json(SUMMARY_PATH, summary)
    print(json.dumps({
        "status": overall_status,
        "local_minimum_power_mw": local_power_min,
        "local_minimum_unit_area_gain_kw_m2": local_gain_min,
        "sunshape_minimum_power_mw": sun_power_min,
        "sunshape_minimum_unit_area_gain_kw_m2": sun_gain_min,
        "runtime_seconds": summary["runtime_seconds"],
        "output": str(SUMMARY_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
