#!/usr/bin/env python3
"""GA over mirror count and two-zone regenerated Q3 positions."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import replace
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
import q3_optical_model as q3  # noqa: E402
import q3_round7_global_pso as common  # noqa: E402

ROUND_DIR = ROOT / os.environ.get(
    "Q3_LAYOUT_GA_ROUND_DIR", "results/Q3/experiments/round28_layout_count_ga",
)
TABLE_DIR = ROUND_DIR / "tables"
METRIC_DIR = ROUND_DIR / "metrics"
SEED = 20230901
NAMES = ("tower_y_m", "boundary_m", "inner_width_m", "outer_width_m",
         "inner_phase", "outer_phase", "inner_ratio", "outer_ratio", "clearance_m")
LOWER = np.asarray((-120.0, 120.0, 2.0, 2.0, 0.0, 0.0, 0.85, 0.85, 0.50))
UPPER = np.asarray((20.0, 320.0, 8.0, 8.0, 0.99, 0.99, 1.00, 1.00, 0.90))
SCREEN_INDICES = (0, 2, 5, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57)


def fields(vector: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(NAMES, np.clip(vector, LOWER, UPPER))}


def generate(vector: np.ndarray) -> tuple[common.Candidate, q3.HeterogeneousDesign, dict[str, float]]:
    value = fields(vector)
    tower = np.asarray((0.0, value["tower_y_m"]))
    pools = []
    labels = []
    for zone, (width, phase) in enumerate(((value["inner_width_m"], value["inner_phase"]),
                                            (value["outer_width_m"], value["outer_phase"]))):
        design = q2.Design(f"M3G_layout_pool_{zone}", "M3G", "proposal", 0.0,
                           value["tower_y_m"], width, width, width / 2.0 + value["clearance_m"])
        pool = symmetric.symmetric_triangular_layout(design, phase)
        radius = np.linalg.norm(pool - tower, axis=1)
        chosen = pool[radius < value["boundary_m"]] if zone == 0 else pool[radius >= value["boundary_m"]]
        pools.append(chosen)
        labels.append(np.full(len(chosen), zone, dtype=np.int64))
    points = np.vstack(pools)
    zone = np.concatenate(labels)
    widths = np.where(zone == 0, value["inner_width_m"], value["outer_width_m"])
    ratios = np.where(zone == 0, value["inner_ratio"], value["outer_ratio"])
    heights = widths * ratios
    installation = np.maximum(2.0, heights / 2.0 + value["clearance_m"])
    accepted = common.accept_variable_groups(points, zone.astype(float), widths)
    points, widths, heights, installation, zone = (
        array[accepted] for array in (points, widths, heights, installation, zone)
    )
    order = np.lexsort((points[:, 0], points[:, 1]))
    points, widths, heights, installation, zone = (
        array[order] for array in (points, widths, heights, installation, zone)
    )
    design = q3.HeterogeneousDesign("M3G_layout_count", "M3G", "main_candidate",
                                    0.0, value["tower_y_m"])
    geometry = q3.geometry_checks(points, widths, heights, installation, design)
    candidate = common.Candidate(points, widths, heights, installation, geometry)
    value["inner_count"] = int(np.sum(zone == 0))
    value["outer_count"] = int(np.sum(zone == 1))
    return candidate, design, value


def evaluate(vector: np.ndarray, states: list[q1.SolarState], samples: tuple[int, int]) -> tuple[dict[str, Any], common.Candidate]:
    candidate, design, value = generate(vector)
    if candidate.geometry["status"] != "PASS":
        return {**value, "geometry_status": "FAIL", "mirror_count": len(candidate.points),
                "field_power_mw": -1e9, "unit_area_power_kw_m2": -1e9}, candidate
    components, _, runtime = q3.evaluate_resolution(
        candidate.points, candidate.widths, candidate.heights,
        candidate.installation_heights, design, states, samples[0], samples[1], (SEED,), 64)
    _, annual = q3.aggregate(components, states, candidate.widths * candidate.heights)
    return {**value, "geometry_status": "PASS", "mirror_count": len(candidate.points), **annual,
            "power_margin_mw": annual["field_power_mw"] - 60.0,
            "minimum_spacing_slack_m": candidate.geometry["minimum_spacing_slack_m"],
            "runtime_seconds": runtime}, candidate


def rank_key(row: dict[str, Any], power_floor: float) -> tuple[int, float, float]:
    power, unit = float(row["field_power_mw"]), float(row["unit_area_power_kw_m2"])
    return (1, unit, power) if power >= power_floor else (0, power, unit)


def seed_vectors(rng: np.random.Generator) -> list[np.ndarray]:
    output = []
    for tower_y, boundary, inner, outer in (
        (-60, 180, 2.0, 6.8), (-60, 240, 4.0, 6.8),
        (-60, 180, 6.8, 2.0), (-60, 240, 6.8, 4.0),
        (-40, 220, 8.0, 6.8), (-60, 220, 6.8, 8.0),
        (-60, 220, 6.8, 6.8), (-20, 180, 6.6, 6.8),
    ):
        vector = rng.uniform(LOWER, UPPER)
        vector[:4] = (tower_y, boundary, inner, outer)
        vector[4:6] = 0.0
        vector[6:8] = 1.0
        vector[8] = 0.55
        output.append(vector)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--full-count", type=int, default=5)
    args = parser.parse_args()
    started = time.perf_counter()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    all_states = q1.build_solar_states()
    screen_states = [replace(all_states[source], state_index=index)
                     for index, source in enumerate(SCREEN_INDICES)]
    population = rng.uniform(LOWER, UPPER, size=(args.population, len(NAMES)))
    seeds = seed_vectors(rng)
    population[:len(seeds)] = seeds
    records = []
    vectors = {}
    screen_power_floor = 12.8

    for generation in range(args.generations):
        evaluated = []
        for member, vector in enumerate(population):
            row, _ = evaluate(vector, screen_states, (2, 1))
            evaluation_id = len(records) + 1
            row.update({"evaluation_id": evaluation_id, "generation": generation, "member": member})
            records.append(row)
            vectors[evaluation_id] = vector.copy()
            evaluated.append((row, vector.copy()))
            print(f"gen={generation} m={member} n={row['mirror_count']} y={row['tower_y_m']:.1f} "
                  f"power={row['field_power_mw']:.4f} unit={row['unit_area_power_kw_m2']:.6f}", flush=True)
        if generation == 0:
            screen_power_floor = max(row["field_power_mw"] for row, _ in evaluated) - 0.15
        evaluated.sort(key=lambda item: rank_key(item[0], screen_power_floor), reverse=True)
        next_population = [item[1].copy() for item in evaluated[:3]]
        while len(next_population) < args.population - 2:
            tournament = rng.choice(len(evaluated), 4, replace=False)
            parents = sorted((evaluated[i] for i in tournament),
                             key=lambda item: rank_key(item[0], screen_power_floor), reverse=True)[:2]
            alpha = rng.uniform(0.2, 0.8, len(NAMES))
            child = alpha * parents[0][1] + (1.0 - alpha) * parents[1][1]
            mutation = rng.random(len(NAMES)) < 0.30
            child += mutation * rng.normal(0.0, 0.07, len(NAMES)) * (UPPER - LOWER)
            next_population.append(np.clip(child, LOWER, UPPER))
        coverage = seed_vectors(rng)
        next_population.extend(coverage[rng.integers(0, len(coverage))] for _ in range(2))
        population = np.asarray(next_population[:args.population])

    selected = []
    seen = set()
    for row in sorted(records, key=lambda item: rank_key(item, screen_power_floor), reverse=True):
        vector = vectors[int(row["evaluation_id"])]
        key = tuple(np.round(vector, 5))
        if key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) >= args.full_count:
            break
    full_rows = []
    for rank, source in enumerate(selected, start=1):
        vector = vectors[int(source["evaluation_id"])]
        row, candidate = evaluate(vector, all_states, (8, 4))
        row.update({"candidate_rank": rank, "source_evaluation_id": source["evaluation_id"]})
        full_rows.append(row)
        common.write_csv(TABLE_DIR / f"q3_layout_candidate_{rank}.csv", common.layout_rows(candidate))
        print(f"full={rank} n={row['mirror_count']} power={row['field_power_mw']:.4f} "
              f"unit={row['unit_area_power_kw_m2']:.6f}", flush=True)
    feasible = [row for row in full_rows if row["field_power_mw"] >= 60.0]
    best = max(feasible, key=lambda row: row["unit_area_power_kw_m2"]) if feasible else None
    summary = {
        "schema_version": 1, "question": "Q3", "round": ROUND_DIR.name, "status": "success",
        "approved_decision_id": "q3_layout_count_joint_revision",
        "method_id": "M3G_two_zone_layout_count_GA", "role": "main_candidate_outer_search",
        "optimizer": {"name": "genetic_algorithm", "seed": SEED,
                      "population": args.population, "generations": args.generations,
                      "parameter_names": list(NAMES), "lower": LOWER.tolist(), "upper": UPPER.tolist()},
        "initialization": "every candidate regenerated from empty field; Q2 layout not read",
        "screen": {"state_indices": list(SCREEN_INDICES), "samples": [2, 1],
                   "power_floor_mw_on_representative_states": screen_power_floor},
        "evaluations": len(records), "full_samples": [8, 4], "full_candidates": full_rows,
        "best_full_feasible": best, "continuous_inner_refinement_required": best is not None,
        "output_degeneracy": {"mirror_count_min": min(row["mirror_count"] for row in records),
                              "mirror_count_max": max(row["mirror_count"] for row in records),
                              "unique_mirror_counts": len({row["mirror_count"] for row in records})},
        "runtime_seconds": time.perf_counter() - started,
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    (METRIC_DIR / "q3_layout_count_ga_trace.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    common.write_csv(TABLE_DIR / "q3_layout_full_rechecks.csv", full_rows)
    (ROUND_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_full_feasible": best,
                      "runtime_seconds": summary["runtime_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
