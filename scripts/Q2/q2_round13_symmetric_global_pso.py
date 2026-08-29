#!/usr/bin/env python3
"""Global-y PSO for an exactly east-west symmetric dense triangular field."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/Q2"))
import q2_staged_search as staged  # noqa: E402

ROUND_DIR = ROOT / "results/Q2/experiments/round13_symmetric_global_pso"
TABLE_DIR = ROUND_DIR / "tables"
METRIC_DIR = ROUND_DIR / "metrics"
PSO_SEED = 2023
PARAMETER_NAMES = ("size_m", "clearance_m", "tower_y_m", "phase_y_fraction")
LOWER = np.asarray((5.5, 0.5, -350.0, 0.0), dtype=np.float64)
UPPER = np.asarray((8.0, 1.0, 350.0, 1.0), dtype=np.float64)
SCREEN_POWER_FLOOR_MW = 58.5


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fields(vector: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(PARAMETER_NAMES, vector)}


def design_for(vector: np.ndarray, identity: str) -> staged.q2.Design:
    value = fields(vector)
    return staged.q2.Design(
        design_id=identity,
        method_id="M2S",
        role="symmetric_global_main_candidate",
        tower_x_m=0.0,
        tower_y_m=value["tower_y_m"],
        width_m=value["size_m"],
        height_m=value["size_m"],
        installation_height_m=value["size_m"] / 2.0 + value["clearance_m"],
    )


def symmetric_triangular_layout(
    design: staged.q2.Design, phase_y_fraction: float
) -> np.ndarray:
    spacing = design.width_m + 5.0
    row_spacing = math.sqrt(3.0) * spacing / 2.0
    tower = np.asarray([0.0, design.tower_y_m])
    maximum_relative = staged.q2.FIELD_RADIUS + abs(design.tower_y_m) + spacing
    row_limit = int(math.ceil(maximum_relative / row_spacing)) + 1
    column_limit = int(math.ceil(maximum_relative / spacing)) + 1
    rows: list[np.ndarray] = []
    for row_index in range(-row_limit, row_limit + 1):
        relative_y = (row_index + phase_y_fraction) * row_spacing
        offset = 0.5 if row_index % 2 else 0.0
        positive = (np.arange(0, column_limit + 1, dtype=np.float64) + offset) * spacing
        if offset == 0.0:
            x_values = np.concatenate((-positive[:0:-1], positive))
        else:
            x_values = np.concatenate((-positive[::-1], positive))
        relative = np.column_stack((x_values, np.full_like(x_values, relative_y)))
        absolute = relative + tower
        legal = (
            (np.linalg.norm(absolute, axis=1) <= staged.q2.FIELD_RADIUS + 1e-9)
            & (np.linalg.norm(relative, axis=1) >= staged.q2.EXCLUSION_RADIUS - 1e-9)
        )
        rows.append(absolute[legal])
    points = np.vstack(rows)
    order = np.lexsort((points[:, 0], points[:, 1]))
    return points[order]


def symmetry_check(points: np.ndarray) -> dict[str, Any]:
    rounded = {(round(float(x), 8), round(float(y), 8)) for x, y in points}
    missing = sum(
        (round(float(-x), 8), round(float(y), 8)) not in rounded for x, y in points
    )
    return {
        "mirror_count": len(points),
        "unpaired_mirror_count": int(missing),
        "status": "PASS" if missing == 0 else "FAIL",
    }


def evaluate(
    vector: np.ndarray,
    identity: str,
    states: list[staged.q2.q1.SolarState],
    samples: tuple[int, int],
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray, staged.q2.Design]:
    value = fields(vector)
    design = design_for(vector, identity)
    points = symmetric_triangular_layout(design, value["phase_y_fraction"])
    geometry = staged.q2.geometry_checks(points, design)
    symmetry = symmetry_check(points)
    if geometry["status"] != "PASS" or symmetry["status"] != "PASS":
        raise RuntimeError(f"Illegal symmetric layout: {geometry}, {symmetry}")
    components, _, runtime = staged.q2.evaluate_resolution(
        points, design, states, samples[0], samples[1],
        (staged.q2.SEEDS[0],), batch_size,
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
        "unpaired_mirror_count": symmetry["unpaired_mirror_count"],
        **annual,
        "power_margin_mw": annual["field_power_mw"] - staged.q2.TARGET_POWER_MW,
        "runtime_seconds": runtime,
    }
    return row, points, design


def fitness(row: dict[str, Any]) -> tuple[int, float, float]:
    power = float(row["field_power_mw"])
    unit = float(row["unit_area_power_kw_m2"])
    return (1, unit, power) if power >= SCREEN_POWER_FLOOR_MW else (0, power, unit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--full-count", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    started = time.perf_counter()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(PSO_SEED)
    all_states = staged.q2.q1.build_solar_states()
    screen_states = [
        replace(all_states[source], state_index=index)
        for index, source in enumerate(staged.SCREEN_STATE_INDICES)
    ]
    positions = rng.uniform(LOWER, UPPER, size=(args.particles, len(LOWER)))
    initial_y = (-300.0, -200.0, -100.0, -40.0, 0.0, 40.0, 100.0, 200.0, 300.0)
    for index, tower_y in enumerate(initial_y[: args.particles]):
        positions[index] = (6.7, 0.55, tower_y, 0.0)
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
                f"iter={iteration} particle={particle} y={row['tower_y_m']:.2f} "
                f"n={row['mirror_count']} power={row['field_power_mw']:.6f} "
                f"unit={row['unit_area_power_kw_m2']:.6f}", flush=True,
            )
        if iteration + 1 < args.iterations:
            rp = rng.random(positions.shape)
            rg = rng.random(positions.shape)
            velocities = 0.72 * velocities + 1.49 * rp * (personal_positions - positions) + 1.49 * rg * (global_position - positions)
            velocities = np.clip(velocities, -0.22 * (UPPER - LOWER), 0.22 * (UPPER - LOWER))
            positions = np.clip(positions + velocities, LOWER, UPPER)
    write_csv(TABLE_DIR / "q2_symmetric_global_pso_trace.csv", records)

    ranked = sorted(records, key=fitness, reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for row in ranked:
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
        row, points, design = evaluate(
            vector, f"full_{rank}", all_states, (8, 4), args.batch_size
        )
        row.update({"candidate_rank": rank, "source_evaluation_id": source["evaluation_id"]})
        full_rows.append(row)
        write_csv(
            TABLE_DIR / f"q2_symmetric_full_candidate_{rank}_layout.csv",
            staged.layout_rows(points, design),
        )
        print(
            f"full rank={rank} y={row['tower_y_m']:.2f} power={row['field_power_mw']:.6f} "
            f"unit={row['unit_area_power_kw_m2']:.6f}", flush=True,
        )
    write_csv(TABLE_DIR / "q2_symmetric_full_rechecks.csv", full_rows)
    feasible = [row for row in full_rows if row["field_power_mw"] >= 60.0]
    best = max(feasible, key=lambda row: row["unit_area_power_kw_m2"]) if feasible else None
    summary = {
        "schema_version": 1,
        "question": "Q2",
        "round": "round13_symmetric_global_pso",
        "status": "success",
        "approved_decision_id": "q2_tower_domain_symmetry_revision",
        "symmetry_contract": {"tower_x_m": 0.0, "unpaired_mirror_count": 0},
        "optimizer": {
            "name": "particle_swarm_optimization",
            "seed": PSO_SEED,
            "particles": args.particles,
            "iterations": args.iterations,
            "parameter_names": list(PARAMETER_NAMES),
            "lower_bounds": LOWER.tolist(),
            "upper_bounds": UPPER.tolist(),
        },
        "screen_states": list(staged.SCREEN_STATE_INDICES),
        "screen_samples": [2, 2],
        "full_samples": [8, 4],
        "full_candidates": full_rows,
        "best_screen_feasible": best,
        "formal_recheck_required": best is not None,
        "runtime_seconds": time.perf_counter() - started,
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    (METRIC_DIR / "q2_symmetric_global_pso_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROUND_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
