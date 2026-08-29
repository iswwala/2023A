#!/usr/bin/env python3
"""PSO refinement of symmetric zoned honeycomb layouts with transition bands."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/Q2"))
import q2_optical_model as q2  # noqa: E402
import q2_round13_symmetric_global_pso as symmetric  # noqa: E402
import q2_round15_zoned_probe as zoned  # noqa: E402
import q2_staged_search as staged  # noqa: E402

ROUND_DIR = ROOT / "results/Q2/experiments/round16_zoned_transition_pso"
TABLE_DIR = ROUND_DIR / "tables"
METRIC_DIR = ROUND_DIR / "metrics"
PSO_SEED = 2023
SCREEN_POWER_FLOOR_MW = 58.5
PARAMETER_NAMES = (
    "width_m", "height_ratio", "clearance_m", "tower_y_m", "inner_boundary_m",
    "zone_gap_m", "inner_phase", "middle_phase", "outer_phase",
    "transition_factor",
)
LOWER = np.asarray((5.80, 0.65, 0.50, -120.0, 135.0, 60.0, 0.0, 0.0, 0.0, 0.75))
UPPER = np.asarray((8.00, 1.00, 0.90, 40.0, 210.0, 125.0, 0.50, 0.50, 0.50, 2.50))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def values(vector: np.ndarray) -> dict[str, float]:
    result = {name: float(value) for name, value in zip(PARAMETER_NAMES, vector)}
    result["height_m"] = max(2.0, result["width_m"] * result["height_ratio"])
    result["middle_boundary_m"] = min(325.0, result["inner_boundary_m"] + result["zone_gap_m"])
    return result


def make_design(vector: np.ndarray, identity: str) -> q2.Design:
    value = values(vector)
    return q2.Design(
        identity, "M2ZT", "zoned_transition_main", 0.0, value["tower_y_m"],
        value["width_m"], value["height_m"],
        value["height_m"] / 2.0 + value["clearance_m"],
    )


def transition_layout(vector: np.ndarray, design: q2.Design) -> np.ndarray:
    value = values(vector)
    tower = np.asarray([0.0, design.tower_y_m])
    spacing = design.width_m + 5.0
    band = value["transition_factor"] * spacing
    r1 = value["inner_boundary_m"]
    r2 = value["middle_boundary_m"]
    phase_values = (value["inner_phase"], value["middle_phase"], value["outer_phase"])
    groups: list[tuple[int, np.ndarray]] = []

    base = symmetric.symmetric_triangular_layout(design, 0.0)
    base_radius = np.linalg.norm(base - tower, axis=1)
    transition = base[(np.abs(base_radius - r1) <= band) | (np.abs(base_radius - r2) <= band)]
    groups.extend(zoned.pair_groups(transition, 0))

    boundaries = (q2.EXCLUSION_RADIUS, r1, r2, np.inf)
    for zone_index, phase in enumerate(phase_values):
        pool = symmetric.symmetric_triangular_layout(design, phase)
        radius = np.linalg.norm(pool - tower, axis=1)
        interior = (
            (radius >= boundaries[zone_index] - 1e-9)
            & (radius < boundaries[zone_index + 1] - 1e-9)
            & (np.abs(radius - r1) > band)
            & (np.abs(radius - r2) > band)
        )
        groups.extend(zoned.pair_groups(pool[interior], zone_index + 1))

    # A final phase-zero pool fills legal holes left by cross-zone stitching.
    groups.extend(zoned.pair_groups(base, 10))
    points = zoned.accept_symmetric_groups(groups, spacing)
    return points[np.lexsort((points[:, 0], points[:, 1]))]


def evaluate(
    vector: np.ndarray,
    identity: str,
    states: list[q2.q1.SolarState],
    samples: tuple[int, int],
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray, q2.Design]:
    value = values(vector)
    design = make_design(vector, identity)
    points = transition_layout(vector, design)
    geometry = q2.geometry_checks(points, design)
    symmetry_check = symmetric.symmetry_check(points)
    if geometry["status"] != "PASS" or symmetry_check["status"] != "PASS":
        raise RuntimeError(f"Invalid transition layout: {geometry}, {symmetry_check}")
    components, _, runtime = q2.evaluate_resolution(
        points, design, states, samples[0], samples[1],
        (q2.SEEDS[0],), batch_size,
    )
    annual = staged.annual_from_components(
        components, states, design.width_m * design.height_m
    )
    row = {
        **value,
        "tower_x_m": 0.0,
        "installation_height_m": design.installation_height_m,
        "mirror_count": len(points),
        "total_area_m2": len(points) * design.width_m * design.height_m,
        "minimum_spacing_m": geometry["minimum_spacing_m"],
        "unpaired_mirror_count": symmetry_check["unpaired_mirror_count"],
        **annual,
        "power_margin_mw": annual["field_power_mw"] - q2.TARGET_POWER_MW,
        "runtime_seconds": runtime,
    }
    return row, points, design


def fitness(row: dict[str, Any]) -> tuple[int, float, float]:
    power = float(row["field_power_mw"])
    unit = float(row["unit_area_power_kw_m2"])
    return (1, unit, power) if power >= SCREEN_POWER_FLOOR_MW else (0, power, unit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--full-count", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    started = time.perf_counter()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(PSO_SEED)
    all_states = q2.q1.build_solar_states()
    screen_states = [
        replace(all_states[source], state_index=index)
        for index, source in enumerate(staged.SCREEN_STATE_INDICES)
    ]
    positions = rng.uniform(LOWER, UPPER, size=(args.particles, len(LOWER)))
    seeds = np.asarray([
        (6.70, 1.00, 0.55, -40.0, 160.0, 90.0, 0.0, 0.25, 0.50, 1.0),
        (6.70, 0.90, 0.55, -20.0, 170.0, 90.0, 0.0, 0.25, 0.50, 1.5),
        (6.70, 0.80, 0.55, -60.0, 180.0, 90.0, 0.50, 0.0, 0.50, 1.0),
        (7.20, 0.90, 0.55, -40.0, 160.0, 100.0, 0.0, 0.25, 0.50, 1.0),
        (6.70, 1.00, 0.55, -40.0, 170.0, 90.0, 0.0, 0.0, 0.0, 1.0),
    ])
    positions[: min(len(seeds), args.particles)] = seeds[: args.particles]
    velocities = rng.uniform(-0.08 * (UPPER - LOWER), 0.08 * (UPPER - LOWER), positions.shape)
    personal_positions = positions.copy()
    personal_rows: list[dict[str, Any]] = []
    global_position = positions[0].copy()
    global_row: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    vectors: dict[int, np.ndarray] = {}
    evaluation_id = 0
    for iteration in range(args.iterations):
        for particle in range(args.particles):
            evaluation_id += 1
            row, _, _ = evaluate(
                positions[particle], f"screen_{evaluation_id}", screen_states,
                (2, 2), args.batch_size,
            )
            row.update({"evaluation_id": evaluation_id, "iteration": iteration, "particle": particle})
            records.append(row)
            vectors[evaluation_id] = positions[particle].copy()
            if iteration == 0:
                personal_rows.append(row)
            elif fitness(row) > fitness(personal_rows[particle]):
                personal_rows[particle] = row
                personal_positions[particle] = positions[particle].copy()
            if global_row is None or fitness(row) > fitness(global_row):
                global_row = row
                global_position = positions[particle].copy()
            print(
                f"iter={iteration} p={particle} y={row['tower_y_m']:.1f} n={row['mirror_count']} "
                f"power={row['field_power_mw']:.6f} unit={row['unit_area_power_kw_m2']:.6f}", flush=True,
            )
        if iteration + 1 < args.iterations:
            rp = rng.random(positions.shape)
            rg = rng.random(positions.shape)
            velocities = 0.72 * velocities + 1.49 * rp * (personal_positions - positions) + 1.49 * rg * (global_position - positions)
            velocities = np.clip(velocities, -0.22 * (UPPER - LOWER), 0.22 * (UPPER - LOWER))
            positions = np.clip(positions + velocities, LOWER, UPPER)
    write_csv(TABLE_DIR / "q2_zoned_transition_pso_trace.csv", records)

    selected: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for row in sorted(records, key=fitness, reverse=True):
        vector = vectors[int(row["evaluation_id"])]
        key = tuple(np.round(vector, 5))
        if key in seen:
            continue
        selected.append(row)
        seen.add(key)
        if len(selected) == args.full_count:
            break
    full_rows: list[dict[str, Any]] = []
    for rank, source in enumerate(selected, start=1):
        vector = vectors[int(source["evaluation_id"])]
        row, points, design = evaluate(vector, f"full_{rank}", all_states, (8, 4), args.batch_size)
        row.update({"candidate_rank": rank, "source_evaluation_id": source["evaluation_id"]})
        full_rows.append(row)
        write_csv(
            TABLE_DIR / f"q2_zoned_transition_candidate_{rank}_layout.csv",
            [
                {"heliostat_id": i, "x_m": p[0], "y_m": p[1], "z_m": design.installation_height_m,
                 "width_m": design.width_m, "height_m": design.height_m}
                for i, p in enumerate(points, start=1)
            ],
        )
        print(
            f"full rank={rank} power={row['field_power_mw']:.6f} "
            f"unit={row['unit_area_power_kw_m2']:.6f}", flush=True,
        )
    write_csv(TABLE_DIR / "q2_zoned_transition_full_rechecks.csv", full_rows)
    feasible = [row for row in full_rows if row["field_power_mw"] >= 60.0]
    best = max(feasible, key=lambda row: row["unit_area_power_kw_m2"]) if feasible else None
    summary = {
        "schema_version": 1,
        "question": "Q2",
        "round": "round16_zoned_transition_pso",
        "status": "success",
        "approved_decision_id": "q2_zoned_layout_revision",
        "optimizer": {"name": "particle_swarm_optimization", "seed": PSO_SEED,
                      "particles": args.particles, "iterations": args.iterations,
                      "parameter_names": list(PARAMETER_NAMES),
                      "lower_bounds": LOWER.tolist(), "upper_bounds": UPPER.tolist()},
        "screen_samples": [2, 2],
        "full_samples": [8, 4],
        "full_candidates": full_rows,
        "best_screen_feasible": best,
        "formal_recheck_required": best is not None,
        "runtime_seconds": time.perf_counter() - started,
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    (METRIC_DIR / "q2_zoned_transition_pso_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROUND_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
