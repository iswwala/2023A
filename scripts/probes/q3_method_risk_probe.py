#!/usr/bin/env python3
"""Q3 method-screening probe for heterogeneous mirror specifications."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
Q2 = ROOT / "results/Q2/experiments/round17_zoned_formal"
OUT = ROOT / "workspace/methods/Q3/probes/risk_probe_raw_metrics.json"


def load_layout() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with (Q2 / "tables/q2_final_layout.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    xy = np.asarray([[float(r["x_m"]), float(r["y_m"])] for r in rows])
    width = np.asarray([float(r["width_m"]) for r in rows])
    height = np.asarray([float(r["height_m"]) for r in rows])
    z = np.asarray([float(r["z_m"]) for r in rows])
    return xy, width, height, z


def spacing_stats(xy: np.ndarray, width: np.ndarray) -> dict[str, float | int]:
    pairs = np.asarray(list(cKDTree(xy).query_pairs(14.0)), dtype=int)
    delta = xy[pairs[:, 0]] - xy[pairs[:, 1]]
    distance = np.linalg.norm(delta, axis=1)
    required = (width[pairs[:, 0]] + width[pairs[:, 1]]) / 2.0 + 5.0
    slack = distance - required
    return {
        "checked_pair_count": int(len(pairs)),
        "violation_count": int(np.sum(slack < -1e-7)),
        "minimum_slack_m": float(np.min(slack)),
        "p01_slack_m": float(np.quantile(slack, 0.01)),
    }


def main() -> None:
    started = time.perf_counter()
    xy, width, height, z = load_layout()
    evidence = json.loads((Q2 / "metrics/q2_zoned_formal.json").read_text(encoding="utf-8"))
    formal = evidence["formal"]
    n = len(xy)
    radius = np.linalg.norm(xy - np.asarray([formal["tower_x_m"], formal["tower_y_m"]]), axis=1)
    score = (radius - radius.min()) / (radius.max() - radius.min())

    # A tiny continuous heterogeneity profile tests only the geometry interface.
    hetero_width = width - (1e-4 + 9e-4 * score)
    hetero_height = height - (2e-4 + 8e-4 * (1.0 - score))
    hetero_z = 0.5 + 0.5 * np.sqrt(hetero_width**2 + hetero_height**2)

    direct_variables = 3 * n + 2
    formal_runtime = float(json.loads((Q2 / "run_summary.json").read_text())["runtime_seconds"])
    raw = {
        "schema_version": 1,
        "question_id": "Q3",
        "q2_feasible_embedding": {
            "mirror_count": n,
            "field_power_mw": formal["field_power_mw"],
            "unit_area_power_kw_m2": formal["unit_area_power_kw_m2"],
            "all_seeds_feasible": formal["all_seeds_feasible"],
            "geometry_status": evidence["geometry"]["status"],
        },
        "search_dimension": {
            "per_mirror_spec_variables": 3 * n,
            "tower_variables": 2,
            "total_without_position_refinement": direct_variables,
        },
        "baseline_spacing": spacing_stats(xy, width),
        "width_plus_0p1pct_spacing": spacing_stats(xy, width * 1.001),
        "width_minus_0p1pct_spacing": spacing_stats(xy, width * 0.999),
        "heterogeneous_geometry_interface": {
            "spacing": spacing_stats(xy, hetero_width),
            "unique_width_rounded_6dp": int(len(np.unique(np.round(hetero_width, 6)))),
            "unique_height_rounded_6dp": int(len(np.unique(np.round(hetero_height, 6)))),
            "unique_installation_height_rounded_6dp": int(len(np.unique(np.round(hetero_z, 6)))),
            "minimum_ground_clearance_m": float(np.min(hetero_z - 0.5 * np.sqrt(hetero_width**2 + hetero_height**2))),
            "width_range_m": [float(hetero_width.min()), float(hetero_width.max())],
            "height_range_m": [float(hetero_height.min()), float(hetero_height.max())],
            "note": "Geometry-only probe; no optical improvement claim.",
        },
        "scale": {
            "q2_formal_evaluation_runtime_seconds": formal_runtime,
            "formal_evaluations_in_3h": int(3 * 3600 // formal_runtime),
            "formal_evaluations_in_6h": int(6 * 3600 // formal_runtime),
            "direct_pso_particles_10_iterations_6_runtime_hours": formal_runtime * 60 / 3600,
        },
        "boundary_risk": {
            "pooled_power_margin_mw": formal["power_margin_mw"],
            "minimum_seed_margin_mw": formal["minimum_fine_seed_power_mw"] - 60.0,
            "seed_power_range_mw": evidence["convergence"]["fine_seed_field_power_mw_range"],
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(raw, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
