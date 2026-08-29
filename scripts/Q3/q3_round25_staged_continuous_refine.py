#!/usr/bin/env python3
"""Staged continuous per-mirror refinement on the feasible Q3 anchor layout."""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/Q1"))
sys.path.insert(0, str(ROOT / "scripts/Q3"))
import q1_optical_model as q1  # noqa: E402
import q3_optical_model as q3  # noqa: E402
import q3_round7_global_pso as common  # noqa: E402

SOURCE = ROOT / os.environ.get(
    "Q3_CONTINUOUS_SOURCE",
    "results/Q3/experiments/round24_feasible_mixed_refine/tables/q3_feasible_mixed_layout.csv",
)
ROUND_DIR = ROOT / os.environ.get(
    "Q3_CONTINUOUS_ROUND_DIR", "results/Q3/experiments/round25_staged_continuous_refine",
)
TABLE_DIR = ROUND_DIR / "tables"
METRIC_DIR = ROUND_DIR / "metrics"
SEED = 20230828
TOWER_Y_M = float(os.environ.get("Q3_CONTINUOUS_TOWER_Y", "-60.0"))
DIMENSION_LOWER_M = 2.0
DIMENSION_UPPER_M = 8.0
STAGES = (
    {"stage": 1, "max_delta_w_m": 2.0, "max_delta_h_m": 2.0, "target_power_mw": 60.10},
    {"stage": 2, "max_delta_w_m": 0.75, "max_delta_h_m": 0.75, "target_power_mw": 60.07},
    {"stage": 3, "max_delta_w_m": 0.25, "max_delta_h_m": 0.25, "target_power_mw": 60.05},
)


def load_layout(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    points = np.asarray([(float(row["x_m"]), float(row["y_m"])) for row in rows])
    widths = np.asarray([float(row["width_m"]) for row in rows])
    heights = np.asarray([float(row["height_m"]) for row in rows])
    installation = np.asarray([float(row["z_m"]) for row in rows])
    return points, widths, heights, installation


def evaluate(points: np.ndarray, widths: np.ndarray, heights: np.ndarray,
             installation: np.ndarray, states: list[q1.SolarState],
             surface_count: int = 8, sun_count: int = 4) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    design = q3.HeterogeneousDesign("M3G_continuous", "M3G", "main_candidate", 0.0, TOWER_Y_M)
    geometry = q3.geometry_checks(points, widths, heights, installation, design)
    if geometry["status"] != "PASS":
        raise RuntimeError(f"geometry failed: {geometry}")
    components, summaries, runtime = q3.evaluate_resolution(
        points, widths, heights, installation, design, states,
        surface_count, sun_count, (SEED,), 64)
    _, annual = q3.aggregate(components, states, widths * heights)
    return {
        **annual, "geometry_status": geometry["status"],
        "minimum_spacing_slack_m": geometry["minimum_spacing_slack_m"],
        "height_exceeds_width_count": geometry["height_exceeds_width_count"],
        "runtime_seconds": runtime, "seed_summaries": summaries,
    }, components


def proposal(current_w: np.ndarray, current_h: np.ndarray, rank_fraction: np.ndarray,
             parameters: np.ndarray, stage: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fraction, exponent, delta_w_scale, delta_h_scale = parameters
    strength = np.where(
        rank_fraction < fraction,
        np.maximum(0.0, 1.0 - rank_fraction / fraction) ** exponent,
        0.0,
    )
    widths = np.maximum(DIMENSION_LOWER_M,
                        current_w - stage["max_delta_w_m"] * delta_w_scale * strength)
    heights = np.maximum(DIMENSION_LOWER_M,
                         current_h - stage["max_delta_h_m"] * delta_h_scale * strength)
    heights = np.minimum(heights, widths)
    return widths, heights, strength


def summarize_sizes(widths: np.ndarray, heights: np.ndarray) -> dict[str, Any]:
    return {
        "width_min_m": float(np.min(widths)), "width_max_m": float(np.max(widths)),
        "height_min_m": float(np.min(heights)), "height_max_m": float(np.max(heights)),
        "unique_width_count_6dp": int(len(np.unique(np.round(widths, 6)))),
        "unique_height_count_6dp": int(len(np.unique(np.round(heights, 6)))),
        "width_quantiles_m": np.quantile(widths, [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist(),
        "height_quantiles_m": np.quantile(heights, [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist(),
    }


def installation_profile(points: np.ndarray, heights: np.ndarray, rank_fraction: np.ndarray,
                         parameters: np.ndarray, maximum_delta_m: float = 0.8) -> np.ndarray:
    radius = np.linalg.norm(points - np.asarray((0.0, TOWER_Y_M)), axis=1)
    radius_score = np.clip((radius - 100.0) / 250.0, 0.0, 1.0) - 0.5
    north_score = np.clip((points[:, 1] - TOWER_Y_M) / 350.0, -1.0, 1.0)
    low_contribution_score = 1.0 - rank_fraction
    bias, radial, north, contribution = parameters
    delta = maximum_delta_m * (
        bias + radial * radius_score + north * north_score
        + contribution * low_contribution_score
    )
    lower = np.maximum(2.0, heights / 2.0 + 0.05)
    return np.clip(3.95 + delta, lower, 6.0)


def main() -> None:
    started = time.perf_counter()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    states = q1.build_solar_states()
    dni = np.asarray([state.dni_kw_m2 for state in states])
    points, widths, heights, installation = load_layout(SOURCE)
    current, components = evaluate(points, widths, heights, installation, states)
    history = [{"stage": 0, "accepted": True, "metrics": current,
                "size_distribution": summarize_sizes(widths, heights)}]

    for stage in STAGES:
        if current["field_power_mw"] < stage["target_power_mw"]:
            history.append({
                "stage": int(stage["stage"]), "target_power_mw": stage["target_power_mw"],
                "accepted": False, "skipped": True,
                "reason": "current layout is below this stage power target; installation-height recovery runs first",
                "accepted_metrics": current, "size_distribution": summarize_sizes(widths, heights),
            })
            candidate = common.Candidate(points, widths, heights, installation, {})
            common.write_csv(TABLE_DIR / f"q3_continuous_stage_{int(stage['stage'])}_layout.csv",
                             common.layout_rows(candidate))
            print(f"stage={int(stage['stage'])} skipped power={current['field_power_mw']:.6f}", flush=True)
            continue
        density = np.mean(dni[:, None] * components["eta_total"], axis=0)
        order = np.argsort(density)
        rank_fraction = np.empty(len(order), dtype=float)
        rank_fraction[order] = np.arange(len(order), dtype=float) / max(1, len(order) - 1)

        def objective(parameters: np.ndarray) -> float:
            candidate_w, candidate_h, _ = proposal(widths, heights, rank_fraction, parameters, stage)
            area = candidate_w * candidate_h
            predicted_power = float(np.sum(density * area) / 1000.0)
            predicted_unit = predicted_power * 1000.0 / float(np.sum(area))
            shortfall = max(0.0, stage["target_power_mw"] - predicted_power)
            return -predicted_unit + 100.0 * shortfall**2

        optimization = differential_evolution(
            objective, bounds=((0.01, 0.35), (0.35, 5.0), (0.0, 1.0), (0.0, 1.0)),
            seed=SEED + int(stage["stage"]), popsize=8, maxiter=35, tol=1e-8,
            polish=True, workers=1,
        )
        proposed_w, proposed_h, strength = proposal(
            widths, heights, rank_fraction, optimization.x, stage)
        trial_scales = (1.0, 0.75, 0.50, 0.25)
        accepted = False
        attempts = []
        for scale in trial_scales:
            trial_w = widths + scale * (proposed_w - widths)
            trial_h = heights + scale * (proposed_h - heights)
            result, trial_components = evaluate(points, trial_w, trial_h, installation, states)
            attempts.append({"scale": scale, "metrics": result,
                             "size_distribution": summarize_sizes(trial_w, trial_h)})
            if (result["field_power_mw"] >= stage["target_power_mw"]
                    and result["unit_area_power_kw_m2"] > current["unit_area_power_kw_m2"]):
                widths, heights = trial_w, trial_h
                current, components = result, trial_components
                accepted = True
                break
        history.append({
            "stage": int(stage["stage"]), "target_power_mw": stage["target_power_mw"],
            "optimizer_parameters": {
                "affected_fraction": float(optimization.x[0]),
                "rank_exponent": float(optimization.x[1]),
                "delta_w_scale": float(optimization.x[2]),
                "delta_h_scale": float(optimization.x[3]),
            },
            "surrogate_objective": float(optimization.fun),
            "optimizer_success": bool(optimization.success),
            "attempts": attempts, "accepted": accepted,
            "accepted_metrics": current, "size_distribution": summarize_sizes(widths, heights),
            "adjusted_mirror_count": int(np.sum(strength > 1e-12)),
        })
        candidate = common.Candidate(points, widths, heights, installation, {})
        common.write_csv(TABLE_DIR / f"q3_continuous_stage_{int(stage['stage'])}_layout.csv",
                         common.layout_rows(candidate))
        print(f"stage={int(stage['stage'])} accepted={accepted} power={current['field_power_mw']:.6f} "
              f"unit={current['unit_area_power_kw_m2']:.9f} "
              f"unique_w={len(np.unique(np.round(widths, 6)))}", flush=True)

    density = np.mean(dni[:, None] * components["eta_total"], axis=0)
    order = np.argsort(density)
    rank_fraction = np.empty(len(order), dtype=float)
    rank_fraction[order] = np.arange(len(order), dtype=float) / max(1, len(order) - 1)
    representative_indices = (0, 2, 5, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57)
    representative_states = [replace(states[index], state_index=new_index)
                             for new_index, index in enumerate(representative_indices)]

    def height_objective(parameters: np.ndarray) -> float:
        candidate_z = installation_profile(points, heights, rank_fraction, parameters)
        result, _ = evaluate(points, widths, heights, candidate_z, representative_states, 1, 1)
        return -result["field_power_mw"]

    height_optimization = differential_evolution(
        height_objective, bounds=((-1.0, 1.0),) * 4, seed=SEED + 100,
        popsize=5, maxiter=4, tol=1e-5, polish=False, workers=1,
    )
    proposed_z = installation_profile(points, heights, rank_fraction, height_optimization.x)
    height_attempts = []
    height_accepted = False
    for scale in (1.0, 0.5, 0.25):
        trial_z = installation + scale * (proposed_z - installation)
        result, trial_components = evaluate(points, widths, heights, trial_z, states)
        height_attempts.append({"scale": scale, "metrics": result,
                                "installation_min_m": float(np.min(trial_z)),
                                "installation_max_m": float(np.max(trial_z)),
                                "unique_installation_count_6dp": int(len(np.unique(np.round(trial_z, 6))))})
        if (result["field_power_mw"] >= 60.05
                and result["unit_area_power_kw_m2"] >= current["unit_area_power_kw_m2"]):
            installation, current, components = trial_z, result, trial_components
            height_accepted = True
            break
    history.append({
        "stage": 4, "name": "continuous_installation_height",
        "bounds_m": [2.0, 6.0], "optimizer_parameters": {
            "bias": float(height_optimization.x[0]),
            "radial": float(height_optimization.x[1]),
            "north": float(height_optimization.x[2]),
            "low_contribution": float(height_optimization.x[3]),
        },
        "representative_state_indices": list(representative_indices),
        "low_fidelity_objective": float(height_optimization.fun),
        "attempts": height_attempts, "accepted": height_accepted,
        "accepted_metrics": current,
        "installation_min_m": float(np.min(installation)),
        "installation_max_m": float(np.max(installation)),
        "unique_installation_count_6dp": int(len(np.unique(np.round(installation, 6)))),
    })
    candidate = common.Candidate(points, widths, heights, installation, {})
    common.write_csv(TABLE_DIR / "q3_continuous_stage_4_layout.csv", common.layout_rows(candidate))
    print(f"stage=4 accepted={height_accepted} power={current['field_power_mw']:.6f} "
          f"unit={current['unit_area_power_kw_m2']:.9f} "
          f"unique_z={len(np.unique(np.round(installation, 6)))}", flush=True)

    candidate = common.Candidate(points, widths, heights, installation, {})
    common.write_csv(TABLE_DIR / "q3_continuous_final_layout.csv", common.layout_rows(candidate))
    summary = {
        "schema_version": 1, "question": "Q3", "round": ROUND_DIR.name, "status": "success",
        "approved_decision_id": "q3_full_dimension_range_revision",
        "method_id": "M3G_staged_continuous_trust_region", "role": "main_candidate_refinement",
        "source": str(SOURCE.relative_to(ROOT)), "q2_layout_read": False,
        "tower": {"x_m": 0.0, "y_m": TOWER_Y_M},
        "global_dimension_bounds_m": [DIMENSION_LOWER_M, DIMENSION_UPPER_M],
        "branch_dimension_bounds_m": [DIMENSION_LOWER_M, 6.8],
        "branch_upper_bound_reason": "fixed zero-slack anchor positions; expansion requires the separate layout-regeneration branch",
        "optimizer": {"name": "differential_evolution_on_frozen_optical_contribution_surrogate",
                      "seed": SEED, "stages": list(STAGES),
                      "installation_height_search": "four-parameter continuous optical search"},
        "evaluation": {"states": 60, "samples": [8, 4], "seeds": [SEED]},
        "stage_history": history, "best_full_feasible": current,
        "size_distribution": summarize_sizes(widths, heights),
        "formal_four_seed_recheck_required": True,
        "runtime_seconds": time.perf_counter() - started,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "warnings": ["This fixed-position branch covers 2--6.8 m; 6.8--8 m requires position regeneration and is not omitted from the global plan."],
    }
    (METRIC_DIR / "q3_continuous_stage_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROUND_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_full_feasible": current,
                      "size_distribution": summary["size_distribution"],
                      "runtime_seconds": summary["runtime_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
