#!/usr/bin/env python3
"""From-scratch global PSO for a heterogeneous Q3 heliostat field."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/Q1"))
sys.path.insert(0, str(ROOT / "scripts/Q2"))
sys.path.insert(0, str(ROOT / "scripts/Q3"))
import q1_optical_model as q1  # noqa: E402
import q2_optical_model as q2  # noqa: E402
import q2_round13_symmetric_global_pso as symmetric  # noqa: E402
import q2_round16_zoned_transition_pso as transition  # noqa: E402
import q3_optical_model as q3  # noqa: E402

ROUND_DIR = ROOT / os.environ.get("Q3_GLOBAL_ROUND_DIR", "results/Q3/experiments/round7_global_pso")
TABLE_DIR = ROUND_DIR / "tables"
METRIC_DIR = ROUND_DIR / "metrics"
Q2_METRIC = ROOT / "results/Q2/experiments/round17_zoned_formal/metrics/q2_zoned_formal.json"
PSO_SEED = 20230828
SCREEN_POWER_FLOOR_MW = 60.35
SCREEN_STATE_INDICES = (0, 2, 5, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57)
NAMES = (
    "pitch_margin_m", "tower_y_m", "zone_count_proxy",
    "boundary_1_m", "boundary_2_m", "boundary_3_m", "boundary_4_m", "boundary_5_m",
    "phase_1", "phase_2", "phase_3", "phase_4", "phase_5", "phase_6", "base_width_m",
    "width_radial_amp_m", "width_north_amp_m", "base_height_ratio",
    "height_ratio_radial_amp", "height_ratio_north_amp", "base_clearance_m",
    "clearance_radial_amp_m",
)
LOWER = np.asarray((0.0, -180.0, 1.0,
                    115.0, 115.0, 115.0, 115.0, 115.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    5.2, -0.8, -0.5, 0.65, -0.20, -0.15, 0.50, -0.15))
UPPER = np.asarray((0.6, 100.0, 6.0,
                    335.0, 335.0, 335.0, 335.0, 335.0,
                    0.99, 0.99, 0.99, 0.99, 0.99, 0.99,
                    7.8, 0.8, 0.5, 1.00, 0.20, 0.15, 0.90, 0.25))


@dataclass
class Candidate:
    points: np.ndarray
    widths: np.ndarray
    heights: np.ndarray
    installation_heights: np.ndarray
    geometry: dict[str, Any]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fields(vector: np.ndarray) -> dict[str, float]:
    values = {name: float(value) for name, value in zip(NAMES, vector)}
    values["zone_count"] = int(np.clip(round(values["zone_count_proxy"]), 1, 6))
    values["proposal_pitch_m"] = values["base_width_m"] + 5.0 + values["pitch_margin_m"]
    return values


def specification(points: np.ndarray, tower: np.ndarray, value: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relative = points - tower
    radius_score = np.clip((np.linalg.norm(relative, axis=1) - 100.0) / 250.0, 0.0, 1.0)
    north_score = np.clip(relative[:, 1] / 350.0, -1.0, 1.0)
    widths = np.clip(value["base_width_m"] + value["width_radial_amp_m"] * radius_score + value["width_north_amp_m"] * north_score, 2.0, 8.0)
    height_ratio = np.clip(
        value["base_height_ratio"]
        + value["height_ratio_radial_amp"] * radius_score
        + value["height_ratio_north_amp"] * north_score,
        2.0 / widths, 1.0,
    )
    heights = widths * height_ratio
    clearance = np.clip(value["base_clearance_m"] + value["clearance_radial_amp_m"] * radius_score, 0.25, 1.25)
    installation = np.clip(heights / 2.0 + clearance, 2.0, 6.0)
    heights = np.minimum(heights, 2.0 * (installation - 0.05))
    return widths, heights, installation


def proposal_pool(value: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    mother_width = np.clip(
        value["base_width_m"] + max(0.0, value["width_radial_amp_m"])
        + abs(value["width_north_amp_m"]) + value["pitch_margin_m"], 2.0, 8.0,
    )
    design = q2.Design(
        "M3G_proposal", "M3G", "proposal", 0.0, value["tower_y_m"],
        mother_width, 6.0, 3.5,
    )
    tower = np.asarray((0.0, value["tower_y_m"]))
    zone_count = int(value["zone_count"])
    internal = sorted(value[f"boundary_{index}_m"] for index in range(1, 6))[:zone_count - 1]
    boundaries = [q2.EXCLUSION_RADIUS, *internal, np.inf]
    groups: list[tuple[int, np.ndarray]] = []
    for zone in range(zone_count):
        phase = value[f"phase_{zone + 1}"]
        pool = symmetric.symmetric_triangular_layout(design, phase)
        radius = np.linalg.norm(pool - tower, axis=1)
        selected = pool[(radius >= boundaries[zone] - 1e-9) & (radius < boundaries[zone + 1] - 1e-9)]
        groups.extend(transition.zoned.pair_groups(selected, zone))
    # A common pool fills only holes that remain legal after all zone interfaces are joined.
    base = symmetric.symmetric_triangular_layout(design, 0.0)
    groups.extend(transition.zoned.pair_groups(base, zone_count + 1))
    points = transition.zoned.accept_symmetric_groups(groups, mother_width + 5.0)
    points = points[np.lexsort((points[:, 0], points[:, 1]))]
    return points, np.zeros(len(points), dtype=np.int64)


def accept_variable_groups(points: np.ndarray, priorities: np.ndarray, widths: np.ndarray) -> np.ndarray:
    lookup = {(round(float(x), 8), round(float(y), 8)): i for i, (x, y) in enumerate(points)}
    keys = sorted({(round(abs(float(x)), 8), round(float(y), 8)) for x, y in points}, key=lambda item: (priorities[lookup.get(item, 0)], math.hypot(item[0], item[1])))
    cell_size = 13.0
    cells: dict[tuple[int, int], list[int]] = {}
    accepted: list[int] = []

    def cell(point: np.ndarray) -> tuple[int, int]:
        return math.floor(point[0] / cell_size), math.floor(point[1] / cell_size)

    def clear(index: int, pending: list[int]) -> bool:
        point = points[index]
        cx, cy = cell(point)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in cells.get((cx + dx, cy + dy), []):
                    required = (widths[index] + widths[other]) / 2.0 + 5.0
                    if np.linalg.norm(point - points[other]) < required - 1e-8:
                        return False
        for other in pending:
            required = (widths[index] + widths[other]) / 2.0 + 5.0
            if np.linalg.norm(point - points[other]) < required - 1e-8:
                return False
        return True

    for abs_x, y in keys:
        coordinates = [(0.0, y)] if abs_x <= 1e-8 else [(-abs_x, y), (abs_x, y)]
        group = [lookup[coordinate] for coordinate in coordinates if coordinate in lookup]
        if len(group) != len(coordinates):
            continue
        pending: list[int] = []
        if all(clear(index, pending) and not pending.append(index) for index in group):
            for index in pending:
                accepted.append(index)
                cells.setdefault(cell(points[index]), []).append(index)
    return np.asarray(accepted, dtype=np.int64)


def generate(vector: np.ndarray) -> tuple[Candidate, q3.HeterogeneousDesign]:
    value = fields(vector)
    tower = np.asarray((0.0, value["tower_y_m"]))
    pool, _ = proposal_pool(value)
    widths, heights, installation = specification(pool, tower, value)
    points = pool
    order = np.lexsort((points[:, 0], points[:, 1]))
    points, widths, heights, installation = points[order], widths[order], heights[order], installation[order]
    design = q3.HeterogeneousDesign("M3G_global", "M3G", "main_candidate", 0.0, value["tower_y_m"])
    geometry = q3.geometry_checks(points, widths, heights, installation, design)
    return Candidate(points, widths, heights, installation, geometry), design


def evaluate(vector: np.ndarray, states: list[q1.SolarState], samples: tuple[int, int]) -> tuple[dict[str, Any], Candidate]:
    value = fields(vector)
    candidate, design = generate(vector)
    if candidate.geometry["status"] != "PASS":
        return {**value, "geometry_status": "FAIL", "field_power_mw": -1e9, "unit_area_power_kw_m2": -1e9, "mirror_count": len(candidate.points), "runtime_seconds": 0.0}, candidate
    components, _, runtime = q3.evaluate_resolution(candidate.points, candidate.widths, candidate.heights, candidate.installation_heights, design, states, samples[0], samples[1], (PSO_SEED,), 64)
    _, annual = q3.aggregate(components, states, candidate.widths * candidate.heights)
    row = {**value, "geometry_status": "PASS", "mirror_count": len(candidate.points), **annual, "power_margin_mw": annual["field_power_mw"] - 60.0, "minimum_spacing_slack_m": candidate.geometry["minimum_spacing_slack_m"], "runtime_seconds": runtime}
    return row, candidate


def fitness(row: dict[str, Any], power_floor: float) -> tuple[int, float, float]:
    power, unit = float(row["field_power_mw"]), float(row["unit_area_power_kw_m2"])
    return (1, unit, power) if power >= power_floor else (0, power, unit)


def layout_rows(candidate: Candidate) -> list[dict[str, Any]]:
    return [{"heliostat_id": i, "x_m": p[0], "y_m": p[1], "z_m": z, "width_m": w, "height_m": h} for i, (p, w, h, z) in enumerate(zip(candidate.points, candidate.widths, candidate.heights, candidate.installation_heights), start=1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--full-count", type=int, default=5)
    parser.add_argument("--screen-samples", type=int, nargs=2, default=(2, 2))
    parser.add_argument("--full-samples", type=int, nargs=2, default=(8, 4))
    args = parser.parse_args()
    started = time.perf_counter()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(PSO_SEED)
    all_states = q1.build_solar_states()
    screen_states = all_states
    positions = rng.uniform(LOWER, UPPER, size=(args.particles, len(LOWER)))
    seed_values = (
        {"pitch_margin_m": 0.05, "tower_y_m": -20.0, "zone_count_proxy": 1.0, "base_width_m": 6.5, "base_height_ratio": 0.96, "base_clearance_m": 0.55},
        {"pitch_margin_m": 0.08, "tower_y_m": -100.0, "zone_count_proxy": 2.0, "boundary_1_m": 210.0, "phase_2": 0.5, "base_width_m": 6.0, "width_radial_amp_m": -0.2, "base_height_ratio": 1.0, "height_ratio_radial_amp": -0.1, "base_clearance_m": 0.60},
        {"pitch_margin_m": 0.10, "tower_y_m": 50.0, "zone_count_proxy": 4.0, "boundary_1_m": 145.0, "boundary_2_m": 205.0, "boundary_3_m": 275.0, "phase_1": 0.5, "phase_3": 0.5, "base_width_m": 7.2, "width_radial_amp_m": -0.25, "base_height_ratio": 0.90, "height_ratio_radial_amp": 0.1, "base_clearance_m": 0.65},
        {"pitch_margin_m": 0.05, "tower_y_m": -50.0, "zone_count_proxy": 6.0, "boundary_1_m": 125.0, "boundary_2_m": 165.0, "boundary_3_m": 210.0, "boundary_4_m": 260.0, "boundary_5_m": 315.0, "phase_2": 0.5, "phase_4": 0.5, "phase_6": 0.5, "base_width_m": 5.5, "width_radial_amp_m": 0.2, "base_height_ratio": 1.0, "height_ratio_radial_amp": -0.05, "base_clearance_m": 0.55},
    )
    for index, overrides in enumerate(seed_values[:args.particles]):
        seed = {name: float((LOWER[i] + UPPER[i]) / 2.0) for i, name in enumerate(NAMES)}
        seed.update(overrides)
        positions[index] = np.asarray([seed[name] for name in NAMES])
    velocities = rng.uniform(-0.03 * (UPPER - LOWER), 0.03 * (UPPER - LOWER), positions.shape)
    personal_positions = positions.copy()
    personal_rows: list[dict[str, Any] | None] = [None] * args.particles
    global_position, global_row = positions[0].copy(), None
    records: list[dict[str, Any]] = []
    vectors: dict[int, np.ndarray] = {}
    for iteration in range(args.iterations):
        for particle in range(args.particles):
            row, _ = evaluate(positions[particle], screen_states, tuple(args.screen_samples))
            evaluation_id = len(records) + 1
            row.update({"evaluation_id": evaluation_id, "iteration": iteration, "particle": particle})
            records.append(row)
            vectors[evaluation_id] = positions[particle].copy()
            if personal_rows[particle] is None or fitness(row, SCREEN_POWER_FLOOR_MW) > fitness(personal_rows[particle], SCREEN_POWER_FLOOR_MW):
                personal_rows[particle], personal_positions[particle] = row, positions[particle].copy()
            if global_row is None or fitness(row, SCREEN_POWER_FLOOR_MW) > fitness(global_row, SCREEN_POWER_FLOOR_MW):
                global_row, global_position = row, positions[particle].copy()
            print(f"iter={iteration} p={particle} n={row['mirror_count']} power={row['field_power_mw']:.4f} unit={row['unit_area_power_kw_m2']:.6f}", flush=True)
        rp, rg = rng.random(positions.shape), rng.random(positions.shape)
        velocities = 0.68 * velocities + 1.45 * rp * (personal_positions - positions) + 1.45 * rg * (global_position - positions)
        velocities = np.clip(velocities, -0.08 * (UPPER - LOWER), 0.08 * (UPPER - LOWER))
        positions = np.clip(positions + velocities, LOWER, UPPER)

    selected, seen = [], set()
    for row in sorted(records, key=lambda item: fitness(item, SCREEN_POWER_FLOOR_MW), reverse=True):
        key = tuple(np.round(vectors[row["evaluation_id"]], 5))
        if key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) == args.full_count:
            break
    full_rows = []
    for rank, source in enumerate(selected, start=1):
        vector = vectors[source["evaluation_id"]]
        row, candidate = evaluate(vector, all_states, tuple(args.full_samples))
        row.update({"candidate_rank": rank, "source_evaluation_id": source["evaluation_id"]})
        full_rows.append(row)
        write_csv(TABLE_DIR / f"q3_global_candidate_{rank}_layout.csv", layout_rows(candidate))
        print(f"full={rank} n={row['mirror_count']} power={row['field_power_mw']:.4f} unit={row['unit_area_power_kw_m2']:.6f}", flush=True)
    feasible = [row for row in full_rows if row["field_power_mw"] >= 60.0]
    best = max(feasible, key=lambda row: row["unit_area_power_kw_m2"]) if feasible else None
    q2_formal = json.loads(Q2_METRIC.read_text(encoding="utf-8"))["formal"]
    summary = {"schema_version": 1, "question": "Q3", "round": ROUND_DIR.name, "status": "success", "approved_decision_id": "q3_global_restart_revision", "method_id": "M3G", "optimizer": {"name": "particle_swarm_optimization", "seed": PSO_SEED, "particles": args.particles, "iterations": args.iterations, "parameters": list(NAMES), "lower": LOWER.tolist(), "upper": UPPER.tolist()}, "initialization": "independent_empty_field_generation; Q2 coordinates not read", "screen_states": "all_60_states", "screen_samples": args.screen_samples, "full_samples": args.full_samples, "evaluations": len(records), "full_candidates": full_rows, "best_full_feasible": best, "q2_comparison_only": {"field_power_mw": q2_formal["field_power_mw"], "unit_area_power_kw_m2": q2_formal["unit_area_power_kw_m2"]}, "formal_recheck_required": best is not None, "runtime_seconds": time.perf_counter() - started, "environment": {"python": sys.version, "platform": platform.platform()}}
    (METRIC_DIR / "q3_global_pso_trace.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(TABLE_DIR / "q3_global_full_rechecks.csv", full_rows)
    (ROUND_DIR / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_full_feasible": best, "runtime_seconds": summary["runtime_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
