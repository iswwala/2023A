#!/usr/bin/env python3
"""Run the approved Q2 staged size search with zoned radial layout priors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "Q2"))
import q2_optical_model as q2  # noqa: E402


SCREEN_STATE_INDICES = (0, 2, 4, 15, 17, 19, 30, 32, 34, 45, 47, 49)
APPROVED_DECISION_ID = "q2_method_choice_revision_size_layout_search"


@dataclass(frozen=True)
class LayoutStrategy:
    strategy_id: str
    radial_boundaries_m: tuple[float, float]
    radial_keep_fractions: tuple[float, float, float]
    sector_keep_fractions: tuple[float, float, float]
    lattice_angle_deg: float = 0.0
    phase_fraction: float = 0.0


BALANCED = LayoutStrategy(
    "zoned_radial_balanced",
    (180.0, 260.0),
    (1.0, 1.0, 0.99),
    (1.0, 1.0, 0.98),
)
NORTH_WEIGHTED = LayoutStrategy(
    "zoned_radial_north_weighted",
    (180.0, 260.0),
    (1.0, 1.0, 0.985),
    (1.0, 0.995, 0.96),
)
UNIFORM_RADIAL = LayoutStrategy(
    "radial_uniform_reference",
    (180.0, 260.0),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
)


def output_dirs(profile: str) -> tuple[Path, Path, Path]:
    if profile == "search":
        root = ROOT / "results" / "Q2" / "experiments" / "round2"
    else:
        root = ROOT / "workspace" / "analysis" / "Q2" / "staged_search_smoke"
    tables = root / "tables"
    metrics = root / "metrics"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    return root, tables, metrics


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sector_index(relative_xy: np.ndarray) -> int:
    radius = float(np.linalg.norm(relative_xy))
    north_fraction = float(relative_xy[1] / radius)
    if north_fraction >= 0.5:
        return 0
    if north_fraction <= -0.5:
        return 2
    return 1


def deterministic_keep(ring_index: int, point_index: int, fraction: float) -> bool:
    if fraction >= 1.0:
        return True
    hashed = ((ring_index + 1) * 73856093) ^ ((point_index + 1) * 19349663)
    return (hashed % 1000003) / 1000003.0 < fraction


def zoned_staggered_layout(design: q2.Design, strategy: LayoutStrategy) -> np.ndarray:
    tower = np.array([design.tower_x_m, design.tower_y_m], dtype=np.float64)
    mother_design = replace(
        design,
        lattice_angle_deg=strategy.lattice_angle_deg,
        phase_x_fraction=strategy.phase_fraction,
        phase_y_fraction=strategy.phase_fraction,
    )
    mother_pool = q2.hexagonal_layout(mother_design)
    selected: list[np.ndarray] = []
    for point_index, absolute in enumerate(mother_pool):
        relative = absolute - tower
        radius = float(np.linalg.norm(relative))
        zone = 0 if radius < strategy.radial_boundaries_m[0] else (
            1 if radius < strategy.radial_boundaries_m[1] else 2
        )
        keep_fraction = (
            strategy.radial_keep_fractions[zone]
            * strategy.sector_keep_fractions[sector_index(relative)]
        )
        if deterministic_keep(zone, point_index, keep_fraction):
            selected.append(absolute)
    if not selected:
        raise ValueError("Zoned staggered strategy generated no legal heliostats")
    return np.asarray(selected, dtype=np.float64)


def installation_heights(height_m: float) -> tuple[float, ...]:
    minimum = max(2.0, height_m / 2.0 + 0.5)
    middle = (minimum + 6.0) / 2.0
    return tuple(sorted({round(minimum, 6), round(middle, 6), 6.0}))


def coarse_specifications(profile: str) -> list[tuple[float, float, float]]:
    widths = (2.0, 5.0, 8.0) if profile == "smoke" else tuple(np.arange(2.0, 8.01, 1.0))
    result: list[tuple[float, float, float]] = []
    for width in widths:
        if profile == "smoke":
            heights = sorted({2.0, width})
        else:
            heights = np.arange(2.0, width + 0.01, 1.0)
        for height in heights:
            for installation_height in installation_heights(float(height)):
                result.append((float(width), float(height), installation_height))
    return result


def make_design(
    stage: str,
    width: float,
    height: float,
    installation_height: float,
    strategy: LayoutStrategy,
) -> q2.Design:
    design_id = (
        f"M2R_{stage}_w{width:.2f}_h{height:.2f}_z{installation_height:.2f}_"
        f"{strategy.strategy_id}"
    )
    return q2.Design(
        design_id,
        "M2R",
        "main_candidate",
        0.0,
        -40.0,
        width,
        height,
        installation_height,
    )


def annual_from_components(
    components: dict[str, np.ndarray],
    states: list[q2.q1.SolarState],
    area_m2: float,
) -> dict[str, float]:
    dni = np.asarray([state.dni_kw_m2 for state in states], dtype=np.float64)
    state_mean = {
        name: values.mean(axis=1)
        for name, values in components.items()
        if name.startswith("eta_")
    }
    unit_power = dni * state_mean["eta_total"]
    return {
        "eta_total": float(np.mean(state_mean["eta_total"])),
        "eta_cos": float(np.mean(state_mean["eta_cos"])),
        "eta_sb": float(np.mean(state_mean["eta_sb"])),
        "eta_trunc": float(np.mean(state_mean["eta_trunc"])),
        "field_power_mw": float(np.mean(unit_power) * components["eta_total"].shape[1] * area_m2 / 1000.0),
        "unit_area_power_kw_m2": float(np.mean(unit_power)),
    }


def evaluate_candidate(
    design: q2.Design,
    strategy: LayoutStrategy,
    states: list[q2.q1.SolarState],
    samples: tuple[int, int],
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray]:
    points = zoned_staggered_layout(design, strategy)
    geometry = q2.geometry_checks(points, design)
    if geometry["status"] != "PASS":
        raise RuntimeError(f"Illegal generated layout: {design.design_id}")
    mean_dni = float(np.mean([state.dni_kw_m2 for state in states]))
    upper_power_mw = (
        len(points)
        * design.width_m
        * design.height_m
        * mean_dni
        * q2.q1.REFLECTIVITY
        / 1000.0
    )
    base = {
        "design_id": design.design_id,
        "strategy_id": strategy.strategy_id,
        "width_m": design.width_m,
        "height_m": design.height_m,
        "installation_height_m": design.installation_height_m,
        "mirror_count": len(points),
        "total_area_m2": len(points) * design.width_m * design.height_m,
        "minimum_spacing_m": geometry["minimum_spacing_m"],
        "ground_clearance_m": geometry["ground_clearance_m"],
        "capacity_upper_bound_mw": upper_power_mw,
    }
    if upper_power_mw < q2.TARGET_POWER_MW:
        return {
            **base,
            "status": "UPPER_BOUND_REJECT",
            "eta_total": "",
            "eta_cos": "",
            "eta_sb": "",
            "eta_trunc": "",
            "field_power_mw": "",
            "unit_area_power_kw_m2": "",
            "runtime_seconds": 0.0,
        }, points
    components, _, runtime = q2.evaluate_resolution(
        points,
        design,
        states,
        samples[0],
        samples[1],
        (q2.SEEDS[0],),
        batch_size,
    )
    annual = annual_from_components(
        components, states, design.width_m * design.height_m
    )
    return {
        **base,
        "status": "EVALUATED",
        **annual,
        "runtime_seconds": runtime,
    }, points


def evaluated(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["status"] == "EVALUATED"]


def normalized_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return math.sqrt(
        ((float(left["width_m"]) - float(right["width_m"])) / 6.0) ** 2
        + ((float(left["height_m"]) - float(right["height_m"])) / 6.0) ** 2
        + ((float(left["installation_height_m"]) - float(right["installation_height_m"])) / 4.0) ** 2
    )


def select_diverse(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    pool = evaluated(rows)
    if not pool:
        raise RuntimeError("No coarse specification survived the capacity upper bound")
    ordered_unit = sorted(
        pool,
        key=lambda row: (
            float(row["field_power_mw"]) >= 58.5,
            float(row["unit_area_power_kw_m2"]),
        ),
        reverse=True,
    )
    ordered_boundary = sorted(pool, key=lambda row: abs(float(row["field_power_mw"]) - 60.8))
    selected: list[dict[str, Any]] = []

    size_bands = (
        [row for row in pool if float(row["width_m"]) < 6.0],
        [row for row in pool if 6.0 <= float(row["width_m"]) < 7.0],
        [row for row in pool if float(row["width_m"]) >= 7.0],
    )
    band_anchors: list[dict[str, Any]] = []
    for band in size_bands:
        if not band:
            continue
        power_anchor = max(band, key=lambda row: float(row["field_power_mw"]))
        band_anchors.append(power_anchor)
        competitive = [
            row
            for row in band
            if float(row["field_power_mw"]) >= 0.95 * float(power_anchor["field_power_mw"])
        ]
        band_anchors.append(
            max(competitive, key=lambda row: float(row["unit_area_power_kw_m2"]))
        )

    for row in band_anchors + ordered_unit[:2] + ordered_boundary[:2]:
        if row["design_id"] not in {item["design_id"] for item in selected}:
            selected.append(row)
    while len(selected) < min(count, len(pool)):
        remaining = [row for row in pool if row["design_id"] not in {item["design_id"] for item in selected}]
        chosen = max(
            remaining,
            key=lambda row: min(normalized_distance(row, item) for item in selected),
        )
        selected.append(chosen)
    return selected[:count]


def fine_specifications(
    coarse_rows: list[dict[str, Any]], step: float
) -> list[tuple[float, float, float]]:
    candidates: set[tuple[float, float, float]] = set()
    for row in coarse_rows:
        center = (
            float(row["width_m"]),
            float(row["height_m"]),
            float(row["installation_height_m"]),
        )
        offsets = [
            (0.0, 0.0, 0.0),
            (-step, 0.0, 0.0),
            (step, 0.0, 0.0),
            (0.0, -step, 0.0),
            (0.0, step, 0.0),
            (0.0, 0.0, -step),
            (0.0, 0.0, step),
            (-step, -step, 0.0),
            (step, step, 0.0),
        ]
        for delta in offsets:
            width = round(center[0] + delta[0], 6)
            height = round(center[1] + delta[1], 6)
            installation_height = round(center[2] + delta[2], 6)
            if not (2.0 <= height <= width <= 8.0):
                continue
            if not (2.0 <= installation_height <= 6.0):
                continue
            if installation_height - height / 2.0 < 0.5 - 1e-9:
                continue
            candidates.add((width, height, installation_height))
    return sorted(candidates)


def shortlist_for_full_recheck(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    pool = evaluated(rows)
    return select_diverse(pool, min(count, len(pool)))


def layout_rows(points: np.ndarray, design: q2.Design) -> list[dict[str, Any]]:
    return [
        {
            "heliostat_id": index,
            "x_m": float(point[0]),
            "y_m": float(point[1]),
            "z_m": design.installation_height_m,
            "width_m": design.width_m,
            "height_m": design.height_m,
        }
        for index, point in enumerate(points, start=1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "search"), default="smoke")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    started = time.perf_counter()
    root, table_dir, metric_dir = output_dirs(args.profile)
    all_states = q2.q1.build_solar_states()
    screen_states = [
        replace(all_states[source_index], state_index=local_index)
        for local_index, source_index in enumerate(SCREEN_STATE_INDICES)
    ]
    if args.profile == "search":
        coarse_samples = (2, 2)
        fine_samples = (4, 2)
        full_samples = (8, 4)
        coarse_keep = 8
        full_keep = 8
        fine_strategies = (BALANCED, NORTH_WEIGHTED)
    else:
        coarse_samples = (1, 1)
        fine_samples = (2, 1)
        full_samples = (4, 2)
        coarse_keep = 2
        full_keep = 3
        fine_strategies = (BALANCED,)

    strategy_checks: list[dict[str, Any]] = []
    reference_design = make_design("strategy_check", 7.0, 7.0, 4.0, BALANCED)
    for strategy in (BALANCED, NORTH_WEIGHTED, UNIFORM_RADIAL):
        points = zoned_staggered_layout(reference_design, strategy)
        strategy_checks.append(
            {
                "strategy": asdict(strategy),
                "mirror_count": len(points),
                "geometry": q2.geometry_checks(points, reference_design),
            }
        )
    write_json(metric_dir / "q2_layout_strategy_checks.json", {
        "schema_version": 1,
        "status": "PASS" if all(row["geometry"]["status"] == "PASS" for row in strategy_checks) else "FAIL",
        "checks": strategy_checks,
    })

    coarse_rows: list[dict[str, Any]] = []
    for width, height, installation_height in coarse_specifications(args.profile):
        design = make_design("coarse", width, height, installation_height, BALANCED)
        row, _ = evaluate_candidate(
            design, BALANCED, screen_states, coarse_samples, args.batch_size
        )
        coarse_rows.append(row)
        print(
            f"coarse {design.design_id} status={row['status']} "
            f"power={row['field_power_mw']} unit={row['unit_area_power_kw_m2']}",
            flush=True,
        )
    write_csv(table_dir / "q2_coarse_size_candidates.csv", coarse_rows)

    coarse_shortlist = select_diverse(coarse_rows, coarse_keep)
    fine_rows: list[dict[str, Any]] = []
    for width, height, installation_height in fine_specifications(coarse_shortlist, 0.5):
        for strategy in fine_strategies:
            design = make_design("fine", width, height, installation_height, strategy)
            row, _ = evaluate_candidate(
                design, strategy, screen_states, fine_samples, args.batch_size
            )
            fine_rows.append(row)
            print(
                f"fine {design.design_id} status={row['status']} "
                f"power={row['field_power_mw']} unit={row['unit_area_power_kw_m2']}",
                flush=True,
            )
    write_csv(table_dir / "q2_fine_size_candidates.csv", fine_rows)

    full_candidates = shortlist_for_full_recheck(fine_rows, full_keep)
    full_rows: list[dict[str, Any]] = []
    layouts: dict[str, np.ndarray] = {}
    designs: dict[str, q2.Design] = {}
    strategies = {item.strategy_id: item for item in (BALANCED, NORTH_WEIGHTED, UNIFORM_RADIAL)}
    for candidate in full_candidates:
        strategy = strategies[str(candidate["strategy_id"])]
        design = make_design(
            "full60",
            float(candidate["width_m"]),
            float(candidate["height_m"]),
            float(candidate["installation_height_m"]),
            strategy,
        )
        row, points = evaluate_candidate(
            design, strategy, all_states, full_samples, args.batch_size
        )
        full_rows.append(row)
        layouts[design.design_id] = points
        designs[design.design_id] = design
        print(
            f"full60 {design.design_id} power={row['field_power_mw']} "
            f"unit={row['unit_area_power_kw_m2']}",
            flush=True,
        )
    write_csv(table_dir / "q2_full_recheck_candidates.csv", full_rows)

    ranked_full = sorted(
        evaluated(full_rows),
        key=lambda row: (
            float(row["field_power_mw"]) >= q2.TARGET_POWER_MW,
            float(row["unit_area_power_kw_m2"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranked_full[: min(5, len(ranked_full))], start=1):
        design_id = str(row["design_id"])
        write_csv(
            table_dir / f"q2_shortlisted_layout_{rank}.csv",
            layout_rows(layouts[design_id], designs[design_id]),
        )

    baseline_design = q2.Design(
        "B2_round2_search_reference", "B2", "usable_baseline", 0.0, 0.0, 7.0, 7.0, 4.0
    )
    baseline_points = q2.hexagonal_layout(baseline_design)
    baseline_components, _, baseline_runtime = q2.evaluate_resolution(
        baseline_points,
        baseline_design,
        all_states,
        full_samples[0],
        full_samples[1],
        (q2.SEEDS[0],),
        args.batch_size,
    )
    baseline_annual = annual_from_components(baseline_components, all_states, 49.0)

    best = ranked_full[0] if ranked_full else None
    summary = {
        "schema_version": 1,
        "question_id": "Q2",
        "profile": args.profile,
        "approved_decision_id": APPROVED_DECISION_ID,
        "search_contract": {
            "coarse_domain": "2 <= h <= w <= 8; 2 <= z_H <= 6; clearance >= 0.5 m",
            "coarse_samples": list(coarse_samples),
            "fine_step_m": 0.5,
            "fine_samples": list(fine_samples),
            "full_60_state_samples": list(full_samples),
            "screen_state_indices": list(SCREEN_STATE_INDICES),
            "coarse_shortlist_count": len(coarse_shortlist),
            "full_recheck_count": len(full_rows),
            "formal_high_precision_required": True,
        },
        "counts": {
            "coarse_total": len(coarse_rows),
            "coarse_evaluated": len(evaluated(coarse_rows)),
            "fine_total": len(fine_rows),
            "fine_evaluated": len(evaluated(fine_rows)),
            "full_rechecked": len(full_rows),
        },
        "best_search_candidate": best,
        "baseline_search_reference": {
            **baseline_annual,
            "mirror_count": len(baseline_points),
            "runtime_seconds": baseline_runtime,
        },
        "limitations": [
            "The search result is a multi-fidelity candidate, not a frozen formal optimum.",
            "The zoned radial strategy parameters are screened priors and do not prove global optimality.",
            "All shortlisted candidates require formal 16x8 and 32x16 four-seed re-evaluation.",
        ],
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(metric_dir / "q2_staged_search_summary.json", summary)
    run_summary = {
        "schema_version": 1,
        "question": "Q2",
        "round": "round2_search" if args.profile == "search" else "round2_smoke",
        "implementation_target": "python",
        "profile": args.profile,
        "approved_decision_id": APPROVED_DECISION_ID,
        "methods": [
            {
                "method_id": "M2R",
                "role": "main_candidate",
                "script": "scripts/Q2/q2_staged_search.py",
                "status": "success_search_only",
                "metrics_summary": best or {},
            },
            {
                "method_id": "B2",
                "role": "usable_baseline",
                "script": "scripts/Q2/q2_optical_model.py",
                "status": "success_search_reference",
                "metrics_summary": baseline_annual,
            },
        ],
        "output_files": [
            str((table_dir / "q2_coarse_size_candidates.csv").relative_to(ROOT)),
            str((table_dir / "q2_fine_size_candidates.csv").relative_to(ROOT)),
            str((table_dir / "q2_full_recheck_candidates.csv").relative_to(ROOT)),
            str((metric_dir / "q2_staged_search_summary.json").relative_to(ROOT)),
            str((metric_dir / "q2_layout_strategy_checks.json").relative_to(ROOT)),
        ],
        "fallback_trigger": {
            "fallback_id": None,
            "condition": "No M2R candidate survives full 60-state recheck with a credible route to 60 MW.",
            "observed": not bool(ranked_full),
            "evidence": str((metric_dir / "q2_staged_search_summary.json").relative_to(ROOT)),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "tensorflow": q2.tf.__version__,
            "physical_gpus": [device.name for device in q2.tf.config.list_physical_devices("GPU")],
        },
        "warnings": ["Search-only evidence; formal high-precision re-evaluation is pending."],
        "errors": [],
    }
    write_json(root / "run_summary.json", run_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
